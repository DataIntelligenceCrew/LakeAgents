from pathlib import Path
from typing import List, Optional
import pandas as pd
from tools.llm_agent_tools import find_dataset_dir
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
    rows = client.read_data(table_id, domain, max_rows=200000)

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

def _normalize_for_hash(value) -> str:
    if pd.isna(value) or value is None:
        return "__NA__"
    s = str(value).strip().lower()
    if not s:
        return "__NA__"
    if "t" in s and ("-" in s or s.count("/") == 2):
        try:
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        except Exception:
            pass
    if s.count("/") == 2 or (s.count("-") >= 2 and len(s) <= 12):
        try:
            return pd.to_datetime(value).strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        num = float(value)
        if num == int(num):  # 14.0 -> 14
            return str(int(num))
        return str(num)
    except (ValueError, TypeError):
        pass  
    return s

def hash_value_to_int(value) -> int:
    s = _normalize_for_hash(value) 
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
SKETCH_RATIO = 0.1
SKETCH_K_MAX = 10000
SKETCH_K_MIN = 256


def _effective_k(
    n_unique: int,
    k: int,
    ratio: Optional[float] = None,
    k_max: Optional[int] = None,
    k_min: Optional[int] = None,
) -> int:
    k_eff = k
    if ratio is not None and n_unique > 0:
        k_from_ratio = int(np.ceil(ratio * n_unique))
        k_from_ratio = max(k_from_ratio, k_min if k_min is not None else SKETCH_K_MIN)
        k_from_ratio = max(k_from_ratio, k)
        k_eff = min(k_from_ratio, k_max if k_max is not None else SKETCH_K_MAX)
    return min(k_eff, n_unique)


def bottom_k_sketch_column(
    series: pd.Series,
    k: int = DEFAULT_SKETCH_K,
    ratio: Optional[float] = SKETCH_RATIO,
    k_max: Optional[int] = SKETCH_K_MAX,
) -> np.ndarray:
    hashed = hash_column_to_unit(series.astype(object))
    unique = np.unique(hashed.dropna().values)
    n_unique = len(unique)
    k_eff = _effective_k(n_unique, k, ratio, k_max)
    if n_unique <= k_eff:
        return np.sort(unique)
    k_smallest = np.partition(unique, k_eff)[:k_eff]
    return np.sort(k_smallest)

def bottom_k_sketch_column_with_samples(
    series: pd.Series,
    k: int = DEFAULT_SKETCH_K,
    ratio: Optional[float] = SKETCH_RATIO,
    k_max: Optional[int] = SKETCH_K_MAX,
) -> tuple[np.ndarray, list, str]:
    col_name = series.name if series.name is not None else ""
    series_clean = series.astype(object).dropna()
    if len(series_clean) == 0:
        return np.array([]), [], col_name

    distinct_vals = series_clean.unique()
    n_unique = len(distinct_vals)
    k_eff = _effective_k(n_unique, k, ratio, k_max)

    value_to_hash = [(v, hash_value_to_unit(v)) for v in distinct_vals]
    value_to_hash.sort(key=lambda x: x[1])
    if n_unique <= k_eff:
        hashes = np.sort(np.array([h for _, h in value_to_hash]))
        values = [v for v, _ in value_to_hash]
        return hashes, values, col_name
    bottom_k_pairs = value_to_hash[:k_eff]
    hashes = np.sort(np.array([h for _, h in bottom_k_pairs]))
    values = [v for v, _ in bottom_k_pairs]
    return hashes, values, col_name

def bottom_k_sketch_df(
    df: pd.DataFrame,
    k: int = DEFAULT_SKETCH_K,
    columns: Optional[List[str]] = None,
    ratio: Optional[float] = SKETCH_RATIO,
    k_max: Optional[int] = SKETCH_K_MAX,
) -> dict[str, np.ndarray]:
    """
    Create bottom-k sketches for all specified columns.
    k_eff = min(max(k, ceil(ratio * n_unique)), k_max) per column when ratio is set.
    """
    cols = columns if columns is not None else list(df.columns)
    sketches = {}
    for c in cols:
        if c in df.columns:
            sketches[c] = bottom_k_sketch_column(df[c], k=k, ratio=ratio, k_max=k_max)
    return sketches

def bottom_k_sketch_df_with_samples(
    df: pd.DataFrame,
    k: int = DEFAULT_SKETCH_K,
    columns: Optional[List[str]] = None,
    ratio: Optional[float] = SKETCH_RATIO,
    k_max: Optional[int] = SKETCH_K_MAX,
) -> dict[str, tuple[np.ndarray, list, str]]:
    """
    Create bottom-k sketches and preserve original values that form each sketch.
    k_eff per column: min(max(k, ceil(ratio * n_unique)), k_max).
    """
    cols = columns if columns is not None else list(df.columns)
    result = {}
    for c in cols:
        if c in df.columns:
            sketch, values, name = bottom_k_sketch_column_with_samples(
                df[c], k=k, ratio=ratio, k_max=k_max
            )
            result[c] = (sketch, values, name or c)
    return result
