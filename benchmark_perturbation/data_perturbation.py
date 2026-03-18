import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.aggregation import classify_column_type, convert_numeric_columns
from tools.column_descriptions import get_column_datatypes_from_local_metadata
from benchmark_perturbation.metadata_perturbation import _apply_replacements_to_text
import yaml

PERTURBATION_CONFIG_PATH = _PROJECT_ROOT / "configs" / "perturbation.yaml"

def load_perturbation_config() -> dict:
    if not PERTURBATION_CONFIG_PATH.exists():
        return {}
    with open(PERTURBATION_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("tables", {})

QUERY_BASE = _PROJECT_ROOT / "query_table"
OUTPUT_BASE = _PROJECT_ROOT / "query_table"
DATA_FILENAME = "rows.csv"
ORIGINAL_FILENAME = "rows_original.csv"  # save original x for binning
BETA = 0.1  # data perturbation level
RANDOM_SEED = 42

TABLE_CONFIG = load_perturbation_config()

def get_numerical_columns(df, base_dir, table_folder, exclude_columns):
    column_datatypes = get_column_datatypes_from_local_metadata(str(base_dir), table_folder)
    numerical_cols = []
    for col in df.columns:
        if col in exclude_columns:
            continue
        col_type = classify_column_type(df[col], col, column_datatypes=column_datatypes)
        if col_type == "numerical":
            numerical_cols.append(col)
    return numerical_cols


def apply_numerical_noise(
    df: pd.DataFrame,
    numerical_cols: list,
    beta: float = BETA,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    x' = x + ε, ε ~ N(0, σ²), σ = β * std(x).
    Only add noise to numerical columns, keep NaN.

    Returns:
        df_perturbed: perturbed DataFrame
        df_original: original DataFrame (for binning)
    """
    df_original = df.copy()
    df_perturbed = df.copy()
    rng = np.random.RandomState(random_state)

    for col in numerical_cols:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
        valid = s.notna()
        if valid.sum() < 2:
            continue
        x = s[valid].astype(float)
        std_x = x.std()
        if std_x == 0 or np.isnan(std_x):
            continue
        sigma = beta * std_x
        epsilon = rng.normal(0, sigma, size=len(x))
        x_perturbed = x.values + epsilon
        df_perturbed.loc[valid, col] = x_perturbed

    return df_perturbed, df_original

def add_freedman_diaconis_bins(
    df_perturbed: pd.DataFrame,
    df_original: pd.DataFrame,
    numerical_cols: list,
) -> pd.DataFrame:
    """
    Bin original data using Freedman-Diaconis rule, add bin columns to df_perturbed.
    Creates {col}_bin for each numerical column (bin index 0, 1, 2, ...).
    """
    df_out = df_perturbed.copy()
    for col in numerical_cols:
        s = df_original[col]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
        valid = s.notna()
        x = s[valid].astype(float)
        if len(x) < 2:
            continue
        try:
            bin_edges = np.histogram_bin_edges(x, bins="fd")
        except (ValueError, TypeError):
            continue
        if len(bin_edges) < 2:
            continue
        bin_col = col + "_bin"
        df_out[bin_col] = pd.cut(
            s, bins=bin_edges, labels=False, include_lowest=True, duplicates="drop"
        )
    return df_out

def process_numerical_perturb(
    table_folder: str,
    base_dir: Path,
    output_dir: Path,
    data_filename: str = DATA_FILENAME,
    original_filename: str = ORIGINAL_FILENAME,
    beta: float = BETA,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Add noise to numerical columns, save perturbed and original."""
    table_path = base_dir / table_folder
    rows_path = table_path / data_filename
    if not rows_path.exists():
        return {"table": table_folder, "status": "skip", "reason": "rows.csv not found"}
    df = pd.read_csv(rows_path, low_memory=False)
    table_cfg = TABLE_CONFIG.get(table_folder, {})
    orig_join_cols = table_cfg.get("join_columns", []) or []
    if isinstance(orig_join_cols, str):
        orig_join_cols = [orig_join_cols]
    jaccard_perturbed_dir = Path(__file__).resolve().parent / "jaccard_perturbed"
    perturbed_json_path = jaccard_perturbed_dir / f"{table_folder}_perturbed.json"
    replacements_for_this_table = {}
    if perturbed_json_path.exists():
        with open(perturbed_json_path, "r", encoding="utf-8") as f:
            perturbed_data = json.load(f)
        replacements_for_this_table = perturbed_data.get("replacements", {})
    exclude = [
        _get_perturbed_name(c, replacements_for_this_table) for c in orig_join_cols
    ]
    df = convert_numeric_columns(df, exclude_columns=exclude)
    if df.empty:
        return {"table": table_folder, "status": "skip", "reason": "empty"}
    num_cols = get_numerical_columns(df, base_dir, table_folder, exclude)
    if not num_cols:
        return {"table": table_folder, "status": "ok", "numerical_cols": 0}
    df_perturbed, df_original = apply_numerical_noise(df, num_cols, beta=beta, random_state=random_state)
    # df_perturbed = add_freedman_diaconis_bins(df_perturbed, df_original, num_cols)

    df_perturbed = shuffle_rows_jointly_on_selected_features(
        df_perturbed,
        table_name=table_folder,
        table_config=TABLE_CONFIG,
        replacements=replacements_for_this_table,
        beta=beta,
        random_state=random_state,
    )
    out_path = output_dir / table_folder
    out_path.mkdir(parents=True, exist_ok=True)
    df_original.to_csv(out_path / original_filename, index=False)
    df_perturbed.to_csv(out_path / data_filename, index=False)
    return {"table": table_folder, "status": "ok", "numerical_cols": len(num_cols)}


def _get_perturbed_name(original: str, replacements: dict) -> str:
    """Apply same synonym replacements as metadata during runtime."""
    if not original:
        return original
    return _apply_replacements_to_text(original, replacements or {})

def shuffle_rows_jointly_on_selected_features(
    df: pd.DataFrame,
    table_name: str,
    table_config: dict,
    replacements: dict,
    beta: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    1) Get original join/target names from config and apply replacements to get perturbed column names
    2) Sample beta% of row indices
    3) Sample beta% of feature columns (excluding join/target columns)
    4) For sampled rows_idx, use the same permutation to rearrange the selected feature submatrix
       join/target columns remain unchanged
    """
    if df.empty or beta <= 0:
        return df

    # 1. Get original join/target names from config and apply replacements to get perturbed column names
    cfg = table_config.get(table_name, {}) if table_config is not None else {}
    orig_join_cols = cfg.get("join_columns", []) or []
    orig_target_col = cfg.get("target_column")

    # 2. Apply replacements to get perturbed column names
    perturbed_join_cols = [
        _get_perturbed_name(c, replacements) for c in orig_join_cols
    ]
    perturbed_target_col = (
        _get_perturbed_name(orig_target_col, replacements) if orig_target_col else None
    )

    # 3. Only keep protected columns that actually exist in df
    existing_cols = set(df.columns)
    join_cols = [c for c in perturbed_join_cols if c in existing_cols]
    protected = set(join_cols)
    if perturbed_target_col and perturbed_target_col in existing_cols:
        protected.add(perturbed_target_col)

    # 4. Candidate feature columns
    feature_cols = [c for c in df.columns if c not in protected]
    if not feature_cols:
        return df

    # 5. Sample beta% of row indices
    n = len(df)
    n_rows = int(np.floor(beta * n))
    if n_rows < 2:
        return df

    # 6. Sample beta% of feature columns
    m = len(feature_cols)
    n_feat = int(np.floor(beta * m))
    if n_feat < 1:
        return df

    rng = np.random.RandomState(random_state)
    rows_idx = rng.choice(df.index.to_numpy(), size=n_rows, replace=False)

    selected_feats = rng.choice(
        np.array(feature_cols, dtype=object), size=n_feat, replace=False
    ).tolist()

    # 7. Use the same permutation to rearrange the selected feature submatrix
    permuted_rows_idx = rows_idx.copy()
    rng.shuffle(permuted_rows_idx)
    col_positions = [df.columns.get_loc(c) for c in selected_feats]
    out = df.copy()
    out.iloc[rows_idx, col_positions] = (
        df.iloc[permuted_rows_idx, col_positions].to_numpy()
    )
    return out