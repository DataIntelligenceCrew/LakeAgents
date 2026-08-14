"""
Correlation and distance computation between target and candidate features.

- Both numerical: Pearson correlation
- One or both categorical (vector): Distance correlation
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from tools.aggregation import (
    aggregate_target_by_join_key,
    aggregate_candidate_by_join_key,
)


def _fuzzy_map_values_to_query(
    cand_series: pd.Series,
    query_values: List,
    threshold: int = 80,
) -> pd.Series:
    """
    Map candidate join column values to query's values using rapidfuzz.
    Uses utils.default_process for case-insensitive matching (lowercase, strip, etc.).
    """
    from rapidfuzz import process, utils

    query_strs = [str(v) for v in query_values if pd.notna(v)]
    if not query_strs:
        return cand_series

    unique_cand = cand_series.dropna().unique().tolist()
    mapping = {}
    for v in unique_cand:
        s = str(v)
        match = process.extractOne(
            s, query_strs,
            processor=utils.default_process,
            score_cutoff=threshold,
        )
        if match:
            # match[0] is the matched query string, map to original query value
            idx = query_strs.index(match[0])
            mapping[v] = query_values[idx]

    return cand_series.map(mapping)


def merge_target_with_candidate(
    target_agg: pd.DataFrame,
    cand_agg: pd.DataFrame,
    join_columns: List[str],
    target_col: Optional[str] = None,
    target_type: Optional[str] = None,
    fuzzy_match_threshold: int = 80,
) -> pd.DataFrame:
    """
    Merge aggregated target with aggregated candidate on join key.

    Uses fuzzy matching (rapidfuzz) to align candidate join values with query
    values when strings differ (e.g., "Bronx" vs "BRONX", "Staten Is" vs "Staten Island").

    Args:
        target_agg: Aggregated target DataFrame
        cand_agg: Aggregated candidate DataFrame (join column names must match)
        join_columns: Join column(s)
        target_col: If provided, drop target columns from cand before merge (for same-table case)
        target_type: Used with target_col
        fuzzy_match_threshold: Minimum rapidfuzz score (0-100) to accept a match

    Returns:
        Merged DataFrame (inner join).
    """
    cand = cand_agg.copy()

    # Drop target columns from candidate if overlapping
    if target_col is not None and target_type is not None:
        cols_to_drop = [target_col]
        if target_type == "categorical":
            cols_to_drop.extend([f"{target_col}_vector", f"{target_col}_categories"])
        for c in cols_to_drop:
            if c in cand.columns:
                cand = cand.drop(columns=[c])


    # Ensure join columns exist in both
    for jc in join_columns:
        if jc not in target_agg.columns or jc not in cand.columns:
            return target_agg.merge(cand, on=join_columns, how="inner")

    # Fuzzy map candidate values to query values for each join column
    for jc in join_columns:
        query_vals = target_agg[jc].dropna().unique().tolist()
        cand[jc] = _fuzzy_map_values_to_query(
            cand[jc], query_vals, threshold=fuzzy_match_threshold
        )

    # Drop rows where mapping failed (NaN)
    cand = cand.dropna(subset=join_columns)

    return target_agg.merge(cand, on=join_columns, how="inner")


def _is_vector_column(col: str, df: pd.DataFrame) -> bool:
    return col.endswith("_vector") and col in df.columns

def _is_text_column(col: str, df: pd.DataFrame) -> bool:
    """*_text columns (concat), exclude *_summary."""
    return col.endswith("_text") and col in df.columns and not col.endswith("_summary")


def _get_feature_columns(
    merged_df: pd.DataFrame,
    join_columns: List[str],
    target_col: str,
    target_type: str,
) -> List[Tuple[str, str]]:
    target_cols = [target_col]
    if target_type == "categorical":
        target_cols.extend([f"{target_col}_vector", f"{target_col}_categories"])

    feature_specs = []
    for col in merged_df.columns:
        if col in join_columns or col in target_cols or col.endswith("_categories"):
            continue
        if _is_vector_column(col, merged_df):
            feature_specs.append((col, "categorical"))
        elif _is_text_column(col, merged_df):
            feature_specs.append((col, "text"))
        elif pd.api.types.is_numeric_dtype(merged_df[col]):
            feature_specs.append((col, "numerical"))
    
    # #region agent log
    import json as _json; _ts = __import__('time').time_ns() // 1000000
    _specs_data = [{"col":c,"type":t,"dtype":str(merged_df[c].dtype),"is_numeric":bool(pd.api.types.is_numeric_dtype(merged_df[c])),"is_vector":bool(_is_vector_column(c,merged_df)),"is_text":bool(_is_text_column(c,merged_df))} for c,t in feature_specs[:20]]
    with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_J2","timestamp":_ts,"location":"correlation.py:_get_feature_columns","message":"feature classification results","data":{"feature_specs_count":len(feature_specs),"feature_specs_sample":_specs_data},"hypothesisId":"J"}) + '\n')
    # #endregion
    
    return feature_specs


def _to_array(series: pd.Series, is_vector: bool = False) -> np.ndarray:
    if is_vector:
        arr = np.array([np.asarray(v) if hasattr(v, "__len__") and not isinstance(v, str) else [v] for v in series])
        if arr.size == 0:
            return np.empty((0, 1))
        return arr.reshape(-1, arr.shape[-1]) if arr.ndim == 1 else arr.astype(float)
    return series.values.astype(float).reshape(-1, 1)


def _drop_na_pairs(target_arr: np.ndarray, feature_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t_flat = np.nanmean(target_arr, axis=1) if target_arr.ndim > 1 else target_arr.ravel()
    f_flat = np.nanmean(feature_arr, axis=1) if feature_arr.ndim > 1 else feature_arr.ravel()
    mask = ~(np.isnan(t_flat) | np.isnan(f_flat))
    t_out = target_arr[mask] if target_arr.ndim > 1 else target_arr[mask].reshape(-1, 1)
    f_out = feature_arr[mask] if feature_arr.ndim > 1 else feature_arr[mask].reshape(-1, 1)
    return (t_out.reshape(-1, 1) if t_out.ndim == 1 else t_out), (f_out.reshape(-1, 1) if f_out.ndim == 1 else f_out)


def _compute_metric(target_arr: np.ndarray, feature_arr: np.ndarray, target_type: str, feature_type: str) -> float:
    target_arr, feature_arr = _drop_na_pairs(target_arr, feature_arr)
    if len(target_arr) < 2:
        return np.nan
    both_numerical = target_type == "numerical" and feature_type == "numerical"
    if both_numerical:
        return float(np.corrcoef(target_arr.ravel(), feature_arr.ravel())[0, 1])
    try:
        import dcor
    except ImportError:
        raise ImportError("pip install dcor")
    t = target_arr if target_arr.ndim == 2 and target_arr.shape[1] > 1 else target_arr.ravel()
    f = feature_arr if feature_arr.ndim == 2 and feature_arr.shape[1] > 1 else feature_arr.ravel()
    return float(dcor.distance_correlation(t, f))


def compute_feature_correlations(
    merged_df: pd.DataFrame,
    join_columns: List[str],
    target_col: str,
    target_type: str,
    fasttext_model_path: Optional[str] = "fasttext.bin",
) -> pd.DataFrame:
    target_data_col = target_col if target_type == "numerical" else f"{target_col}_vector"
    if target_data_col not in merged_df.columns:
        raise ValueError(f"Target column '{target_data_col}' not in merged DataFrame")
    target_arr = _to_array(merged_df[target_data_col], is_vector=(target_type == "categorical"))
    rows = []

    for feat_col, feat_type in _get_feature_columns(merged_df, join_columns, target_col, target_type):
        if feat_type == "text":
            if not fasttext_model_path:
                continue
            try:
                from tools.text_integration import embed_texts_with_fasttext
                texts = merged_df[feat_col].fillna("").astype(str).tolist()
                embs = embed_texts_with_fasttext(texts, model_path=fasttext_model_path)
                valid = merged_df[feat_col].notna() & (merged_df[feat_col].astype(str).str.strip() != "")
                embs[~valid.values] = np.nan
                feat_arr = embs.astype(float)
                # Text + numerical target: reduce embedding to 1D (row mean), use Pearson correlation
                if target_type == "numerical":
                    feat_arr_1d = np.nanmean(feat_arr, axis=1, keepdims=True)
                    try:
                        value = _compute_metric(target_arr, feat_arr_1d, target_type, "numerical")
                    except Exception:
                        value = np.nan
                    metric = "correlation"
                else:
                    try:
                        value = _compute_metric(target_arr, feat_arr, target_type, feat_type)
                    except Exception:
                        value = np.nan
                    metric = "distance_correlation"
                base_name = feat_col.replace("_vector", "").replace("_text", "") if feat_col.endswith(("_vector", "_text")) else feat_col
                rows.append({"feature": base_name, "metric": metric, "value": value, "feature_type": feat_type})
                continue
            except (FileNotFoundError, ImportError) as e:
                import warnings
                warnings.warn(f"Skipping text column {feat_col}: {e}", UserWarning)
                continue
        else:
            feat_arr = _to_array(merged_df[feat_col], is_vector=(feat_type == "categorical"))

        both_num = target_type == "numerical" and feat_type == "numerical"
        try:
            value = _compute_metric(target_arr, feat_arr, target_type, feat_type)
        except Exception:
            value = np.nan

        base_name = feat_col.replace("_vector", "").replace("_text", "") if feat_col.endswith(("_vector", "_text")) else feat_col
        metric = "correlation" if both_num else "distance_correlation"
        rows.append({"feature": base_name, "metric": metric, "value": value, "feature_type": feat_type})

    return pd.DataFrame(rows)


def compute_correlations_for_candidate(
    query_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    join_columns: List[str],
    target_column: str,
) -> pd.DataFrame:
    """Full pipeline: aggregate, merge, compute correlations."""
    target_agg, target_type = aggregate_target_by_join_key(query_df, join_columns, target_column)
    cand_agg = aggregate_candidate_by_join_key(cand_df, join_columns)
    merged = merge_target_with_candidate(target_agg, cand_agg, join_columns)
    return compute_feature_correlations(merged, join_columns, target_column, target_type)