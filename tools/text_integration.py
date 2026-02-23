import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy.typing as npt

# Lazy-loaded FastText model cache: {resolved_path: model}
_fasttext_model_cache: Dict[str, object] = {}


def embed_texts_with_fasttext(
    texts: List[str],
    model_path: str = "fasttext.bin",
) -> npt.NDArray[np.float64]:
    """
    Embed each text using FastText sentence vector (average of word vectors).
    Uses get_sentence_vector for the full string.

    Args:
        texts: List of text strings to embed
        model_path: Path to FastText .bin model (default: fasttext.bin)

    Returns:
        np.ndarray of shape (len(texts), dim), dtype float64
    """
    global _fasttext_model_cache
    try:
        from fasttext import FastText
    except ImportError:
        raise ImportError("fasttext package required. Install: pip install fasttext-wheel")

    path = Path(model_path)
    if not path.is_absolute():
        path = Path.cwd() / model_path
    path_str = str(path.resolve())
    if path_str not in _fasttext_model_cache:
        if not path.exists():
            raise FileNotFoundError(
                f"FastText model not found: {path}. "
                "Run download_fasttext.py or download from "
                "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
            )
        _fasttext_model_cache[path_str] = FastText.load_model(path_str)

    model = _fasttext_model_cache[path_str]
    dim = model.get_dimension()
    embs = np.zeros((len(texts), dim), dtype=np.float64)
    for i, t in enumerate(texts):
        s = str(t).strip()
        if s:
            embs[i] = model.get_sentence_vector(s)
        # else: leave zeros
    return embs


def get_fasttext_embed_fn(
    model_path: str = "fasttext.bin",
) -> Callable[[List[str]], npt.NDArray[np.float64]]:
    """
    Return an embed_fn that uses FastText for text embedding.
    Can be used when a custom embed_fn is needed; otherwise FastText is used internally.

    Args:
        model_path: Path to FastText .bin model (default: fasttext.bin)

    Returns:
        embed_fn(texts) -> np.ndarray of shape (n, dim)
    """
    def _embed(texts: List[str]) -> npt.NDArray[np.float64]:
        return embed_texts_with_fasttext(texts, model_path=model_path)
    return _embed


def greedy_max_coverage(
    sim_matrix: npt.NDArray[np.float64],
    texts: List[str],
    k: int = 10,
    min_marginal_gain: float = 1e-6,
) -> List[str]:
    """
    Greedy maximize f(S) = Σ_x max_{s∈S} sim(x,s).
    Select up to k texts that best cover all texts in X.
    """
    n = len(texts)
    if n == 0:
        return []
    if k >= n:
        return list(texts)

    best_sim = np.zeros(n)
    S_indices: List[int] = []

    total_sim_per_s = sim_matrix.sum(axis=0)
    s0 = int(np.argmax(total_sim_per_s))
    S_indices.append(s0)
    best_sim = np.maximum(best_sim, sim_matrix[:, s0])

    for _ in range(k - 1):
        marginal_gains = np.zeros(n)
        for j in range(n):
            if j in S_indices:
                marginal_gains[j] = -np.inf
            else:
                marginal_gains[j] = np.sum(
                    np.maximum(0, sim_matrix[:, j] - best_sim)
                )
        j_star = int(np.argmax(marginal_gains))
        gain = marginal_gains[j_star]
        if gain < min_marginal_gain:
            break
        S_indices.append(j_star)
        best_sim = np.maximum(best_sim, sim_matrix[:, j_star])

    return [texts[i] for i in S_indices]


def _cosine_sim_matrix(embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_norm = embeddings / norms
    sim = emb_norm @ emb_norm.T
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float64)


def greedy_max_coverage_with_embeddings(
    texts: List[str],
    k: int = 10,
    min_marginal_gain: float = 1e-6,
    model_path: str = "fasttext.bin",
) -> List[str]:
    """
    Greedy select subset of texts that best cover all texts.
    Uses FastText for embedding, then cosine similarity for greedy coverage.
    """
    embed_fn = get_fasttext_embed_fn(model_path=model_path)
    embeddings = embed_fn(texts)
    sim_matrix = _cosine_sim_matrix(embeddings)
    return greedy_max_coverage(sim_matrix, texts, k, min_marginal_gain)


def _to_join_key(val: Union[object, np.ndarray]) -> Tuple:
    if isinstance(val, np.ndarray):
        return tuple(val.tolist())
    if isinstance(val, (list, tuple)):
        return tuple(val)
    return (val,)


def subset_map_to_dataframe(
    subset_map: Dict[Tuple, Dict[str, List[str]]],
    join_columns: List[str],
    text_cols: List[str],
    sep: str = " | ",
    suffix: str = "_text",
) -> "pd.DataFrame":
    """
    Convert subset_map from select_text_subset_per_join_key to DataFrame for merge.

    Returns:
        DataFrame with join_columns + {text_col}{suffix} columns (concat of selected texts).
    """
    rows = []
    for jk, col_dict in subset_map.items():
        row = {join_columns[i]: jk[i] for i in range(len(join_columns))}
        for col in text_cols:
            selected = col_dict.get(col, [])
            row[f"{col}{suffix}"] = sep.join(str(s) for s in selected) if selected else ""
        rows.append(row)
    return pd.DataFrame(rows)


def select_text_subset_per_join_key(
    cand_df: pd.DataFrame,
    join_columns: List[str],
    text_cols: List[str],
    k: int = 10,
    min_marginal_gain: float = 1e-6,
    model_path: str = "fasttext.bin",
    verbose: bool = True,
) -> Dict[Tuple, Dict[str, List[str]]]:
    """
    For each (join_key, text_col), greedily select a subset of texts.
    Each join key is computed separately.
    Returns: {join_key: {text_col: [selected_texts]}}
    """
    result: Dict[Tuple, Dict[str, List[str]]] = {}

    for join_key_vals, group in cand_df.groupby(join_columns):
        jk = _to_join_key(join_key_vals)
        result[jk] = {}
        n_rows = len(group)
        if verbose:
            print(f"Join key {jk}: n_rows={n_rows}")

        for text_col in text_cols:
            if text_col not in group.columns:
                continue
            vals = group[text_col].dropna().astype(str).str.strip()
            unique_texts = vals.unique().tolist()
            n_unique = len(unique_texts)
            if not unique_texts:
                continue
            if len(unique_texts) == 1:
                result[jk][text_col] = unique_texts
                if verbose:
                    print(f"  {text_col}: n_unique={n_unique}, n_selected=1")
                continue

            try:
                selected = greedy_max_coverage_with_embeddings(
                    unique_texts,
                    k=k,
                    min_marginal_gain=min_marginal_gain,
                    model_path=model_path,
                )
            except Exception:
                result[jk][text_col] = unique_texts[:k]
                if verbose:
                    print(f"  {text_col}: n_unique={n_unique}, n_selected={min(k, n_unique)} (embed failed)")
                continue

            result[jk][text_col] = selected
            if verbose:
                print(f"  {text_col}: n_unique={n_unique}, n_selected={len(selected)}")

    return result

def get_text_columns_from_candidate(
    cand_df: pd.DataFrame,
    join_columns: List[str],
    table_id: Optional[str] = None,
    column_datatypes: Optional[dict[str, str]] = None,
) -> List[str]:
    """
    Return column names classified as 'text' (high cardinality text, not categorical).
    Uses same classification logic as aggregate_candidate_by_join_key.

    Args:
        cand_df: Candidate table DataFrame
        join_columns: Join column(s) - these are excluded
        table_id: Opendata table ID for metadata lookup
        column_datatypes: Optional pre-fetched {column_name: dataTypeName}

    Returns:
        List of column names classified as 'text'
    """
    from tools.aggregation import classify_column_type, convert_numeric_columns

    cand_df = convert_numeric_columns(cand_df.copy())
    agg_cols = [c for c in cand_df.columns if c not in join_columns]

    if not column_datatypes and table_id:
        from tools.column_descriptions import get_column_datatypes_from_index
        column_datatypes = get_column_datatypes_from_index(table_id)

    text_cols = []
    for col in agg_cols:
        if col not in cand_df.columns:
            continue
        col_type = classify_column_type(
            cand_df[col], col, table_id=table_id, column_datatypes=column_datatypes
        )
        if col_type == "text":
            text_cols.append(col)
    return text_cols

def _summarize_single_text_column(
    text_agg_df: pd.DataFrame,
    join_columns: List[str],
    text_col: str,
    text_suffix: str,
    llm_client,
    provider: str,
) -> pd.DataFrame:
    """
    Summarize a single *_text column using LLM.
    Returns DataFrame with join_columns + {base_name}_summary.
    """
    import json
    import re

    base_name = text_col.replace(text_suffix, "")
    output_column_name = f"{base_name}_summary"

    parts = []
    for _, row in text_agg_df.iterrows():
        jk_vals = [row[c] for c in join_columns]
        jk_str = " | ".join(str(v) for v in jk_vals)
        val = row.get(text_col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(f"Join key: {jk_str}\n  {base_name}: {val}")
    if not parts:
        return text_agg_df[join_columns].copy()
    prompt_body = "\n\n---\n\n".join(parts)

    prompt = (
        "You are analyzing aggregated text data by join key.\n\n"
        f"{prompt_body}\n\n"
        "---\n\n"
        "For each join key, provide summary (brief) and distinct_from_others (what differs vs others).\n"
        "Return JSON only: {\"keys\": [{\"join_key\": \"...\", \"summary\": \"...\", \"distinct_from_others\": \"...\"}]}\n"
        "Use exact join_key values as shown above."
    )

    response = llm_client.ask(prompt)
    if not response:
        return text_agg_df[join_columns].copy()

    cleaned = response.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return text_agg_df[join_columns].copy()
        else:
            return text_agg_df[join_columns].copy()

    rows = []
    for item in data.get("keys", []):
        jk_str = str(item.get("join_key", ""))
        if len(join_columns) == 1:
            row = {join_columns[0]: jk_str}
        else:
            parts_jk = [p.strip() for p in jk_str.split(" | ")]
            row = {
                join_columns[i]: parts_jk[i] if i < len(parts_jk) else ""
                for i in range(len(join_columns))
            }
        row[output_column_name] = str(item.get("summary", ""))
        rows.append(row)

    if not rows:
        return text_agg_df[join_columns].copy()
    return pd.DataFrame(rows)


def summarize_text_per_join_key_with_llm(
    text_agg_df: pd.DataFrame,
    join_columns: List[str],
    text_suffix: str = "_text",
    llm_client=None,
    provider: str = "openai",
) -> pd.DataFrame:
    
    text_cols = [
        c for c in text_agg_df.columns
        if c.endswith(text_suffix) and c not in join_columns
    ]
    if not text_cols:
        return text_agg_df[join_columns].copy()

    if llm_client is None:
        from test_llm_prompt import create_llm_client
        llm_client = create_llm_client(provider=provider)

    llm_dfs = []
    for text_col in text_cols:
        single_df = _summarize_single_text_column(
            text_agg_df, join_columns, text_col, text_suffix, llm_client, provider
        )
        llm_dfs.append(single_df)

    result = llm_dfs[0]
    for df in llm_dfs[1:]:
        result = result.merge(df, on=join_columns, how="outer")
    return result