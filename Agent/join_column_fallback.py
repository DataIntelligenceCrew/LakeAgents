from typing import Any, Dict, List

import pandas as pd

from tools.join_column_tool import fuzzy_string_match, semantic_column_similarity
from tools.sketch import _normalize_for_hash


def _normalize_key_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return normalized key frame for merge/lookup."""
    norm = pd.DataFrame(index=df.index)
    for i, col in enumerate(columns):
        key_col = f"__k{i}"
        if col in df.columns:
            norm[key_col] = df[col].apply(_normalize_for_hash)
        else:
            norm[key_col] = "__NA__"
    return norm


def evaluate_exact_join_health(
    join_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    join_columns: List[str],
    selected_columns: List[str],
    coverage_threshold: float = 0.5,
    explosion_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate exact-join health by coverage and explosion."""
    if not join_columns or not selected_columns:
        return {
            "coverage": 0.0,
            "explosion": float("inf"),
            "coverage_threshold": coverage_threshold,
            "explosion_threshold": explosion_threshold,
            "coverage_ok": False,
            "explosion_ok": False,
            "is_anomaly": True,
            "reason": "missing join columns",
            "matched_rows": 0,
            "merged_rows": 0,
            "left_rows": int(len(join_df)),
        }
    if len(join_columns) != len(selected_columns):
        return {
            "coverage": 0.0,
            "explosion": float("inf"),
            "coverage_threshold": coverage_threshold,
            "explosion_threshold": explosion_threshold,
            "coverage_ok": False,
            "explosion_ok": False,
            "is_anomaly": True,
            "reason": "join column length mismatch",
            "matched_rows": 0,
            "merged_rows": 0,
            "left_rows": int(len(join_df)),
        }

    left_keys = _normalize_key_frame(join_df, join_columns)
    right_keys = _normalize_key_frame(cand_df, selected_columns)

    left_rows = len(left_keys)
    if left_rows == 0:
        return {
            "coverage": 0.0,
            "explosion": 0.0,
            "coverage_threshold": coverage_threshold,
            "explosion_threshold": explosion_threshold,
            "coverage_ok": False,
            "explosion_ok": True,
            "is_anomaly": True,
            "reason": "empty left table",
            "matched_rows": 0,
            "merged_rows": 0,
            "left_rows": 0,
        }

    left_tuples = [tuple(x) for x in left_keys.to_numpy().tolist()]
    right_tuples_set = {tuple(x) for x in right_keys.to_numpy().tolist()}
    matched_rows = sum(1 for t in left_tuples if t in right_tuples_set)
    coverage = matched_rows / left_rows

    merged_rows = len(pd.merge(left_keys, right_keys, on=list(left_keys.columns), how="inner"))
    explosion = merged_rows / left_rows

    coverage_ok = coverage >= coverage_threshold
    explosion_ok = explosion <= explosion_threshold
    is_anomaly = (not coverage_ok) or (not explosion_ok)

    if is_anomaly:
        if not coverage_ok and not explosion_ok:
            reason = "coverage and explosion both abnormal"
        elif not coverage_ok:
            reason = "coverage below threshold"
        else:
            reason = "explosion above threshold"
    else:
        reason = "healthy exact join"

    return {
        "coverage": round(float(coverage), 4),
        "explosion": round(float(explosion), 4),
        "coverage_threshold": coverage_threshold,
        "explosion_threshold": explosion_threshold,
        "coverage_ok": coverage_ok,
        "explosion_ok": explosion_ok,
        "is_anomaly": is_anomaly,
        "reason": reason,
        "matched_rows": int(matched_rows),
        "merged_rows": int(merged_rows),
        "left_rows": int(left_rows),
    }


def build_fallback_attempts(
    selected_columns: List[str],
    top5_candidates_by_query_col: Dict[str, List[str]],
    query_join_columns: List[str],
    max_candidates: int = 5,
) -> List[List[str]]:
    """Build ordered attempt list: selected first, then alternative candidates from top5."""
    attempts: List[List[str]] = []
    seen = set()

    def _add(cols: List[str]) -> None:
        key = tuple(cols)
        if key in seen:
            return
        seen.add(key)
        attempts.append(cols)

    if selected_columns and len(selected_columns) == len(query_join_columns):
        _add(selected_columns)

    if len(query_join_columns) == 1:
        jc = query_join_columns[0]
        for c in top5_candidates_by_query_col.get(jc, [])[:max_candidates]:
            _add([c])
        return attempts[:max_candidates]

    ranked_lists = [top5_candidates_by_query_col.get(jc, [])[:max_candidates] for jc in query_join_columns]
    max_rank = max((len(lst) for lst in ranked_lists), default=0)
    for r in range(max_rank):
        combo: List[str] = []
        valid = True
        for lst in ranked_lists:
            if r >= len(lst):
                valid = False
                break
            combo.append(lst[r])
        if valid:
            _add(combo)
    return attempts[:max_candidates]


def _sample_non_empty_values(series: pd.Series, max_values: int = 1500) -> List[str]:
    """Sample non-empty string values (deduplicated) from a series."""
    if series is None or len(series) == 0:
        return []
    values: List[str] = []
    seen = set()
    for v in series:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(s)
        if len(values) >= max_values:
            break
    return values


def evaluate_similarity_gate(
    join_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    join_columns: List[str],
    selected_columns: List[str],
    query_col_descs: Dict[str, str],
    candidate_col_descs: Dict[str, str],
    fuzzy_threshold: int = 80,
    semantic_threshold: float = 0.8,
    max_values: int = 1500,
) -> Dict[str, Any]:
    """Compute fuzzy + semantic similarity and decide if fuzzy join is allowed."""
    if len(join_columns) != 1 or len(selected_columns) != 1:
        return {
            "supported": False,
            "gate_pass": False,
            "reason": "fuzzy gate currently supports single-column join only",
        }

    q_col = join_columns[0]
    c_col = selected_columns[0]
    if q_col not in join_df.columns or c_col not in cand_df.columns:
        return {
            "supported": False,
            "gate_pass": False,
            "reason": "join column not found in data",
        }

    query_values = _sample_non_empty_values(join_df[q_col], max_values=max_values)
    candidate_values = _sample_non_empty_values(cand_df[c_col], max_values=max_values)

    fuzzy_result = fuzzy_string_match(
        query_values=query_values,
        candidate_values=candidate_values,
        threshold=fuzzy_threshold,
    )
    semantic_result = semantic_column_similarity(
        query_col_name=q_col,
        query_col_description=query_col_descs.get(q_col, ""),
        query_sample_values=query_values[:5],
        candidate_col_name=c_col,
        candidate_col_description=candidate_col_descs.get(c_col, ""),
        candidate_sample_values=candidate_values[:5],
    )

    fuzzy_score = float(fuzzy_result.get("avg_score", 0.0) or 0.0)
    semantic_score = float(semantic_result.get("semantic_similarity", 0.0) or 0.0)
    fuzzy_pass = fuzzy_score >= float(fuzzy_threshold)
    semantic_pass = semantic_score >= float(semantic_threshold)
    gate_pass = fuzzy_pass or semantic_pass

    return {
        "supported": True,
        "gate_pass": gate_pass,
        "fuzzy_threshold": fuzzy_threshold,
        "semantic_threshold": semantic_threshold,
        "fuzzy_pass": fuzzy_pass,
        "semantic_pass": semantic_pass,
        "fuzzy_result": fuzzy_result,
        "semantic_result": semantic_result,
        "query_samples": query_values[:5],
        "candidate_samples": candidate_values[:5],
    }


def _build_fuzzy_key_mapping(
    query_keys: List[str],
    candidate_keys: List[str],
    fuzzy_threshold: int = 80,
) -> Dict[str, str]:
    """Map normalized candidate key -> normalized query key by fuzzy best match."""
    from rapidfuzz import fuzz, process

    mapping: Dict[str, str] = {}
    query_choices = list({k for k in query_keys if k and k != "__NA__"})
    if not query_choices:
        return mapping

    for cand in {k for k in candidate_keys if k and k != "__NA__"}:
        result = process.extractOne(
            cand,
            query_choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=fuzzy_threshold,
        )
        if result:
            mapping[cand] = result[0]
    return mapping


def evaluate_fuzzy_join_health(
    join_df: pd.DataFrame,
    cand_df: pd.DataFrame,
    join_columns: List[str],
    selected_columns: List[str],
    coverage_threshold: float = 0.5,
    explosion_threshold: float = 2.0,
    fuzzy_threshold: int = 80,
) -> Dict[str, Any]:
    """Apply fuzzy key mapping then re-evaluate coverage/explosion."""
    if len(join_columns) != 1 or len(selected_columns) != 1:
        return {
            "supported": False,
            "fuzzy_key_mapping": {},
            "health": {
                "coverage": 0.0,
                "explosion": float("inf"),
                "coverage_threshold": coverage_threshold,
                "explosion_threshold": explosion_threshold,
                "coverage_ok": False,
                "explosion_ok": False,
                "is_anomaly": True,
                "reason": "fuzzy join currently supports single-column join only",
                "matched_rows": 0,
                "merged_rows": 0,
                "left_rows": int(len(join_df)),
            },
        }

    q_col = join_columns[0]
    c_col = selected_columns[0]
    if q_col not in join_df.columns or c_col not in cand_df.columns:
        return {
            "supported": False,
            "fuzzy_key_mapping": {},
            "health": {
                "coverage": 0.0,
                "explosion": float("inf"),
                "coverage_threshold": coverage_threshold,
                "explosion_threshold": explosion_threshold,
                "coverage_ok": False,
                "explosion_ok": False,
                "is_anomaly": True,
                "reason": "join column not found in data",
                "matched_rows": 0,
                "merged_rows": 0,
                "left_rows": int(len(join_df)),
            },
        }

    query_keys = join_df[q_col].apply(_normalize_for_hash).tolist()
    candidate_keys = cand_df[c_col].apply(_normalize_for_hash).tolist()
    mapping = _build_fuzzy_key_mapping(
        query_keys=query_keys,
        candidate_keys=candidate_keys,
        fuzzy_threshold=fuzzy_threshold,
    )

    cand_fuzzy = cand_df.copy()
    normalized_cand = cand_fuzzy[c_col].apply(_normalize_for_hash)
    mapped = normalized_cand.map(mapping)
    cand_fuzzy[c_col] = mapped
    cand_fuzzy = cand_fuzzy[cand_fuzzy[c_col].notna()].copy()

    health = evaluate_exact_join_health(
        join_df=join_df,
        cand_df=cand_fuzzy,
        join_columns=join_columns,
        selected_columns=selected_columns,
        coverage_threshold=coverage_threshold,
        explosion_threshold=explosion_threshold,
    )
    return {
        "supported": True,
        "fuzzy_threshold": fuzzy_threshold,
        "fuzzy_key_mapping": mapping,
        "mapped_rows": int(len(cand_fuzzy)),
        "health": health,
    }
