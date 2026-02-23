from pathlib import Path
from typing import List, Optional
import pandas as pd
from llm_agent_tools import find_dataset_dir
from agent_config_loader import load_config
import json
import hashlib
import numpy as np

_INDEX_PATH = Path(__file__).resolve().parent.parent / "opendata_table_index.json"
_ACCESS_STATUS_PATH = Path(__file__).resolve().parent.parent / "data" / "table_access_status.json"

def _get_domain_from_index(table_id: str) -> Optional[str]:
    """Find domain from opendata_table_index.json by table_id."""
    if not _INDEX_PATH.exists():
        return None
    try:
        with open(_INDEX_PATH, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception:
        return None
    if not isinstance(index, list):
        return None
    tid = str(table_id).strip()
    for entry in index:
        if isinstance(entry, dict) and str(entry.get("id", "")).strip() == tid:
            return entry.get("domain")
    return None

def _update_table_access_status(table_id: str, fetched: bool, reason: Optional[str] = None) -> None:
    """Update data/table_access_status.json, record whether the table is successfully fetched."""
    _ACCESS_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if _ACCESS_STATUS_PATH.exists():
        try:
            with open(_ACCESS_STATUS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    data[str(table_id)] = {"fetched": fetched, "reason": reason}
    with open(_ACCESS_STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_candidate_table(
    table_id: str,
    opendata_domain: Optional[str] = None
) -> tuple[pd.DataFrame, dict]:
    """
    Fetch full candidate table from API.
    Returns: (df, status) where status = {"success": bool, "reason": str|None}
    """
    from datalake_client import SocrataDatalakeClient

    domain = opendata_domain
    if domain is None:
        domain = _get_domain_from_index(table_id)
        if not domain:
            return pd.DataFrame(), {"success": False, "reason": "domain not found in index"}

    cfg = load_config()
    client = SocrataDatalakeClient(cfg.get("data", {}).get("datalake", {}))
    rows = client.read_data(table_id, domain, max_rows=None)

    if not rows or not isinstance(rows, list):
        return pd.DataFrame(), {"success": False, "reason": "empty or invalid response"}

    if isinstance(rows, dict) and rows.get("error"):
        return pd.DataFrame(), {"success": False, "reason": rows.get("message", "API error")}

    df = pd.DataFrame(rows)

    # Rename columns from fieldName to columns_name so they match index (for description lookup)
    from tools.column_descriptions import get_fieldname_to_columns_name_mapping

    rename_map = get_fieldname_to_columns_name_mapping(table_id, domain)
    if rename_map:
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

    return df, {"success": True, "reason": None}


_HASH_MAX = 2**64 - 1  # Max value for normalizing to [0, 1]


def hash_value_to_int(value) -> int:
    """
    Step 1: Hash any value (str, int, float, None, etc.) to a non-negative integer.
    Uses SHA-256 for consistency across runs.
    """
    if pd.isna(value) or value is None:
        s = "__NA__"
    else:
        s = str(value).strip().lower()
    h = hashlib.sha256(s.encode("utf-8", errors="replace")).digest()
    return int.from_bytes(h[:8], byteorder="big")


def hash_int_to_unit(i: int) -> float:
    """
    Step 2: Map a non-negative integer to [0, 1].
    """
    return (i % (_HASH_MAX + 1)) / _HASH_MAX


def hash_value_to_unit(value) -> float:
    """Hash value to int, then to [0, 1]."""
    return hash_int_to_unit(hash_value_to_int(value))


def hash_column_to_int(series: pd.Series) -> pd.Series:
    """Hash all values in a column to integers."""
    return series.apply(hash_value_to_int)


def hash_column_to_unit(series: pd.Series) -> pd.Series:
    """Hash all values in a column to [0, 1]."""
    return series.apply(hash_value_to_unit)


def hash_df_columns_to_int(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Hash all specified columns to integers. If columns is None, hash all columns."""
    cols = columns if columns is not None else list(df.columns)
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = hash_column_to_int(out[c].astype(object))
    return out


def hash_df_columns_to_unit(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Hash all specified columns to [0, 1]. If columns is None, hash all columns."""
    cols = columns if columns is not None else list(df.columns)
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = hash_column_to_unit(out[c].astype(object))
    return out

# ---- Bottom-k sketch ----

DEFAULT_SKETCH_K = 512


def bottom_k_sketch_column(series: pd.Series, k: int = DEFAULT_SKETCH_K) -> np.ndarray:
    """
    Create bottom-k sketch for a column.
    Hashes each value to [0,1], keeps the k smallest distinct hashes, returns sorted array.
    If column has fewer than k distinct values, returns all distinct hashes (sorted).
    """
    hashed = hash_column_to_unit(series.astype(object))
    unique = np.unique(hashed.dropna().values)
    if len(unique) <= k:
        return np.sort(unique)
    # Partition to get k smallest (partial sort)
    k_smallest = np.partition(unique, k)[:k]
    return np.sort(k_smallest)


def bottom_k_sketch_df(
    df: pd.DataFrame,
    k: int = DEFAULT_SKETCH_K,
    columns: Optional[List[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Create bottom-k sketches for all specified columns.
    Returns: dict mapping column_name -> sorted array of k smallest hashes in [0,1].
    """
    cols = columns if columns is not None else list(df.columns)
    sketches = {}
    for c in cols:
        if c in df.columns:
            sketches[c] = bottom_k_sketch_column(df[c], k=k)
    return sketches

# ---- Jaccard similarity ----

def jaccard_similarity_sketches(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two bottom-k sketches.
    Jaccard = |A ∩ B| / |A ∪ B|.
    Returns 0.0 if both sketches are empty.
    """
    if len(sketch_a) == 0 and len(sketch_b) == 0:
        return 0.0
    inter = np.intersect1d(sketch_a, sketch_b)
    union_size = len(sketch_a) + len(sketch_b) - len(inter)
    if union_size == 0:
        return 0.0
    return len(inter) / union_size


def jaccard_sketch_with_columns(
    join_sketch: np.ndarray,
    candidate_sketches: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compare join column sketch with each candidate column sketch.
    Returns: dict mapping column_name -> Jaccard similarity with join column.
    """
    return {
        col: jaccard_similarity_sketches(join_sketch, sketch)
        for col, sketch in candidate_sketches.items()
    }

# ---- Top-k Jaccard selection ----

DEFAULT_TOPK_JOIN_COLUMNS = 5


def select_topk_jaccard_columns(
    jaccards: dict[str, float],
    k: int = DEFAULT_TOPK_JOIN_COLUMNS,
    min_jaccard: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Select top-k columns by Jaccard similarity.
    Only includes columns with jaccard >= min_jaccard.
    Returns list of (column_name, jaccard) sorted by jaccard desc.
    """
    eligible = [(col, j) for col, j in jaccards.items() if j >= min_jaccard]
    eligible.sort(key=lambda x: -x[1])
    return eligible[:k]


def select_join_columns_for_candidate(
    join_sketch: np.ndarray | dict[str, np.ndarray], 
    cand_df: pd.DataFrame,
    k_columns: int = DEFAULT_TOPK_JOIN_COLUMNS,
    min_jaccard: float = 0.5,
    sketch_k: int = DEFAULT_SKETCH_K,
) -> list[tuple[str, float]] | dict[str, list[tuple[str, float]]]:
    """
    Support single column or composite key.
    - join_sketch is np.ndarray: return [(col, jaccard), ...]
    - join_sketch is dict: return {join_col: [(col, jaccard), ...], ...}
    """
    cand_sketches = bottom_k_sketch_df(cand_df, k=sketch_k)
    
    if isinstance(join_sketch, dict):
        # Composite key
        result = {}
        for join_col, sketch in join_sketch.items():
            jaccards = jaccard_sketch_with_columns(sketch, cand_sketches)
            result[join_col] = select_topk_jaccard_columns(jaccards, k=k_columns, min_jaccard=min_jaccard)
        return result
    else:
        # Single key
        jaccards = jaccard_sketch_with_columns(join_sketch, cand_sketches)
        return select_topk_jaccard_columns(jaccards, k=k_columns, min_jaccard=min_jaccard)
