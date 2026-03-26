"""Join column selection tools.

Provides 5 tool functions for join column matching:
1. jaccard_similarity          - set-level overlap
2. containment_score           - bidirectional containment (A in B / B in A)
3. normalized_overlap          - overlap after lowercase/trim/number normalization
4. date_normalized_overlap     - overlap after date format unification
5. fuzzy_string_match          - approximate string matching via rapidfuzz
6. semantic_column_similarity  - embedding-based similarity (sentence-transformers)

Also provides:
- select_join_columns_for_candidate: sketch-based prefilter (moved from sketch.py)
- column_profile: generate per-column profile for Stage A routing
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from tools.sketch import (
    bottom_k_sketch_df,
    DEFAULT_SKETCH_K,
    SKETCH_RATIO,
    SKETCH_K_MAX,
)


DEFAULT_TOPK_JOIN_COLUMNS = 5


def jaccard_similarity_sketches(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    """Jaccard similarity between two bottom-k sketches."""
    if len(sketch_a) == 0 and len(sketch_b) == 0:
        return 0.0
    inter = np.intersect1d(sketch_a, sketch_b)
    union_size = len(sketch_a) + len(sketch_b) - len(inter)
    if union_size == 0:
        return 0.0
    return len(inter) / union_size


def containment_similarity_sketches(
    join_sketch: np.ndarray, cand_sketch: np.ndarray
) -> float:
    """Containment of join_sketch in cand_sketch (single direction)."""
    if len(join_sketch) == 0:
        return 0.0
    inter = np.intersect1d(join_sketch, cand_sketch)
    return len(inter) / len(join_sketch)


def sketch_scores_with_columns(
    join_sketch: np.ndarray,
    candidate_sketches: dict[str, np.ndarray],
) -> dict[str, float]:
    """Score each candidate column against the join sketch (max of jaccard, containment)."""
    result = {}
    for col, cand_sketch in candidate_sketches.items():
        jaccard = jaccard_similarity_sketches(join_sketch, cand_sketch)
        containment = containment_similarity_sketches(join_sketch, cand_sketch)
        score = max(jaccard, containment)
        result[col] = score
    return result


def select_topk_jaccard_columns(
    jaccards: dict[str, float],
    k: int = DEFAULT_TOPK_JOIN_COLUMNS,
    min_jaccard: float = 0.5,
) -> list[tuple[str, float]]:
    """Select top-k columns by score, filtering by min_jaccard."""
    eligible = [(col, j) for col, j in jaccards.items() if j >= min_jaccard]
    eligible.sort(key=lambda x: -x[1])
    return eligible[:k]


def select_join_columns_for_candidate(
    join_sketch: np.ndarray | dict[str, np.ndarray],
    cand_df: pd.DataFrame,
    k_columns: int = DEFAULT_TOPK_JOIN_COLUMNS,
    min_jaccard: float = 0.5,
    sketch_k: int = DEFAULT_SKETCH_K,
    sketch_ratio: Optional[float] = SKETCH_RATIO,
    sketch_k_max: Optional[int] = SKETCH_K_MAX,
) -> list[tuple[str, float]] | dict[str, list[tuple[str, float]]]:
    """Sketch-based prefilter for join column candidates (single or composite key)."""
    cand_sketches = bottom_k_sketch_df(
        cand_df, k=sketch_k, ratio=sketch_ratio, k_max=sketch_k_max
    )
    if isinstance(join_sketch, dict):
        result = {}
        for join_col, sketch in join_sketch.items():
            scores = sketch_scores_with_columns(sketch, cand_sketches)
            result[join_col] = select_topk_jaccard_columns(
                scores, k=k_columns, min_jaccard=min_jaccard
            )
        return result
    else:
        scores = sketch_scores_with_columns(join_sketch, cand_sketches)
        return select_topk_jaccard_columns(scores, k=k_columns, min_jaccard=min_jaccard)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_value(value) -> str:
    """Lowercase, trim, normalize numbers."""
    if pd.isna(value) or value is None:
        return ""
    s = str(value).strip().lower()
    if not s:
        return ""
    try:
        num = float(s)
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, TypeError):
        pass
    return s


def _normalize_date(value) -> str:
    """Try to parse value as date and return YYYY-MM-DD, else return empty string."""
    if pd.isna(value) or value is None:
        return ""
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Column profile (Stage A)
# ---------------------------------------------------------------------------

def column_profile(
    series: pd.Series,
    n_samples: int = 5,
) -> Dict[str, Any]:
    """Generate a lightweight profile for a single column."""
    non_null = series.dropna()
    total = len(series)
    non_null_count = len(non_null)

    unique_vals = non_null.astype(str).unique()
    uniqueness_ratio = len(unique_vals) / non_null_count if non_null_count > 0 else 0.0

    samples = list(unique_vals[:n_samples])

    dtype_guess = "unknown"
    if pd.api.types.is_numeric_dtype(series):
        dtype_guess = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        dtype_guess = "datetime"
    else:
        date_like = sum(1 for v in samples if _normalize_date(v))
        if date_like >= len(samples) * 0.6 and samples:
            dtype_guess = "date_string"
        elif uniqueness_ratio > 0.9:
            dtype_guess = "id_like"
        else:
            dtype_guess = "categorical"

    pattern_tags: List[str] = []
    if dtype_guess in ("datetime", "date_string"):
        pattern_tags.append("date")
    if dtype_guess == "id_like":
        pattern_tags.append("id")
    if dtype_guess == "numeric":
        pattern_tags.append("numeric")
    if uniqueness_ratio < 0.05 and non_null_count > 0:
        pattern_tags.append("low_cardinality")

    return {
        "column_name": series.name,
        "dtype_guess": dtype_guess,
        "sample_values": samples,
        "non_null_ratio": round(non_null_count / total, 4) if total > 0 else 0.0,
        "uniqueness_ratio": round(uniqueness_ratio, 4),
        "pattern_tags": pattern_tags,
    }


# ---------------------------------------------------------------------------
# Tool 1: Jaccard similarity (value-set level)
# ---------------------------------------------------------------------------

def jaccard_similarity(
    query_values: List[str],
    candidate_values: List[str],
) -> Dict[str, Any]:
    """Compute Jaccard similarity between two value sets."""
    set_a = {_normalize_value(v) for v in query_values if _normalize_value(v)}
    set_b = {_normalize_value(v) for v in candidate_values if _normalize_value(v)}
    if not set_a and not set_b:
        return {"jaccard": 0.0}
    inter = set_a & set_b
    union = set_a | set_b
    return {"jaccard": round(len(inter) / len(union), 4) if union else 0.0}


# ---------------------------------------------------------------------------
# Tool 2: Containment (bidirectional)
# ---------------------------------------------------------------------------

def containment_score(
    query_values: List[str],
    candidate_values: List[str],
) -> Dict[str, Any]:
    """Bidirectional containment between two value sets."""
    set_q = {_normalize_value(v) for v in query_values if _normalize_value(v)}
    set_c = {_normalize_value(v) for v in candidate_values if _normalize_value(v)}
    inter = set_q & set_c
    c_q_in_c = round(len(inter) / len(set_q), 4) if set_q else 0.0
    c_c_in_q = round(len(inter) / len(set_c), 4) if set_c else 0.0
    return {
        "query_in_candidate": c_q_in_c,
        "candidate_in_query": c_c_in_q,
    }


# ---------------------------------------------------------------------------
# Tool 3a: Normalized overlap
# ---------------------------------------------------------------------------

def normalized_overlap(
    query_values: List[str],
    candidate_values: List[str],
) -> Dict[str, Any]:
    """Overlap after basic normalization (lower, trim, number normalization)."""
    set_q = {_normalize_value(v) for v in query_values if _normalize_value(v)}
    set_c = {_normalize_value(v) for v in candidate_values if _normalize_value(v)}
    inter = set_q & set_c
    union = set_q | set_c
    overlap = round(len(inter) / len(union), 4) if union else 0.0
    return {
        "normalized_overlap": overlap,
        "matched_count": len(inter),
        "query_count": len(set_q),
        "candidate_count": len(set_c),
    }


# ---------------------------------------------------------------------------
# Tool 3b: Date-normalized overlap
# ---------------------------------------------------------------------------

def date_normalized_overlap(
    query_values: List[str],
    candidate_values: List[str],
) -> Dict[str, Any]:
    """Overlap after date format unification to YYYY-MM-DD."""
    set_q = set()
    for v in query_values:
        d = _normalize_date(v)
        if d:
            set_q.add(d)
    set_c = set()
    for v in candidate_values:
        d = _normalize_date(v)
        if d:
            set_c.add(d)
    if not set_q and not set_c:
        return {"date_normalized_overlap": 0.0, "parseable_query": 0, "parseable_candidate": 0}
    inter = set_q & set_c
    union = set_q | set_c
    return {
        "date_normalized_overlap": round(len(inter) / len(union), 4) if union else 0.0,
        "matched_count": len(inter),
        "parseable_query": len(set_q),
        "parseable_candidate": len(set_c),
    }


# ---------------------------------------------------------------------------
# Tool 4: Fuzzy string match
# ---------------------------------------------------------------------------

def fuzzy_string_match(
    query_values: List[str],
    candidate_values: List[str],
    threshold: int = 80,
) -> Dict[str, Any]:
    """Match query values to candidate values using rapidfuzz token_sort_ratio."""
    from rapidfuzz import process, fuzz

    query_strs = [str(v).strip().lower() for v in query_values if pd.notna(v) and str(v).strip()]
    cand_strs = list({str(v).strip().lower() for v in candidate_values if pd.notna(v) and str(v).strip()})

    if not query_strs or not cand_strs:
        return {"matched_ratio": 0.0, "avg_score": 0.0, "matched_count": 0}

    matched = 0
    total_score = 0.0
    for q in query_strs:
        result = process.extractOne(q, cand_strs, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
        if result:
            matched += 1
            total_score += result[1]

    return {
        "matched_ratio": round(matched / len(query_strs), 4),
        "avg_score": round(total_score / matched, 2) if matched else 0.0,
        "matched_count": matched,
        "total_query": len(query_strs),
    }


# ---------------------------------------------------------------------------
# Tool 5: Semantic column similarity (sentence-transformers)
# ---------------------------------------------------------------------------

_st_model = None


def semantic_column_similarity(
    query_col_name: str,
    query_col_description: str,
    query_sample_values: List[str],
    candidate_col_name: str,
    candidate_col_description: str,
    candidate_sample_values: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> Dict[str, Any]:
    """Compute semantic similarity between two columns using sentence-transformers."""
    global _st_model
    try:
        if _st_model is None:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer(model_name)

        def _build_text(name: str, desc: str, samples: List[str]) -> str:
            parts = [name]
            if desc:
                parts.append(desc)
            if samples:
                parts.append("values: " + ", ".join(str(s) for s in samples[:5]))
            return " | ".join(parts)

        text_q = _build_text(query_col_name, query_col_description, query_sample_values)
        text_c = _build_text(candidate_col_name, candidate_col_description, candidate_sample_values)

        embeddings = _st_model.encode([text_q, text_c], normalize_embeddings=True)
        sim = float(np.dot(embeddings[0], embeddings[1]))
        return {"semantic_similarity": round(sim, 4)}
    except ImportError:
        return {"semantic_similarity": 0.0, "error": "sentence-transformers not installed"}
    except Exception as e:
        return {"semantic_similarity": 0.0, "error": str(e)}
