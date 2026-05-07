#!/usr/bin/env python3
"""
Non-agentic baseline — join-key bottom-k sketch on query tables.

Sketch params match Pipeline/phase_join_columns.py (single-column path):
  bottom_k_sketch_column_with_samples(..., k=1024, ratio=None, k_max=1024)

Modes (no extra YAML; τ/β grid hardcoded as [0.1,0.5,0.9]×[0.1,0.5,0.9]):
  grid (default) — step 1: 12 TASKS tables × 9 cells; each cell reads
      <repo>/perturbed_{tau}_{beta}/<table>/rows.csv
  single — step 1 only: query_table/<table>/rows.csv
  search — step 2: same search_domains / search_q as Pipeline/phase_table_selection
    (tools.llm_agent_tools.build_opendata_search_params + datalake read_metadata),
    using TASK_DIMENSIONS–style specs per table; excludes query base dataset id from results.
  pick-join — step 3–4: for each search candidate, fetch via API, pick join column by sketch-key
    coverage; then bottom-k sketch on that column (same params as step 1 / phase_join_columns).
    Single query join column per task only (matches current 12 TASKS yaml).
  corr — step 5: read pick-join JSON; per ok candidate merge query target with candidate aggregates;
    numerical features: convert_numeric_columns + pd.to_numeric on remaining non-numeric columns
    (API text-as-number); compute_feature_correlations; keep top 10 by |value|.

Examples:
  python scripts/non_agentic_baseline.py
  python scripts/non_agentic_baseline.py -o data/non_agentic_sketch_grid.json
  python scripts/non_agentic_baseline.py single -o data/non_agentic_query_join_sketches.json
  python scripts/non_agentic_baseline.py search -o data/non_agentic_datalake_search.json
  python scripts/non_agentic_baseline.py pick-join -o data/non_agentic_pick_join.json
  python scripts/non_agentic_baseline.py pick-join --search-json data/non_agentic_datalake_search.json -o out.json
  python scripts/non_agentic_baseline.py corr --pick-join-json data/non_agentic_pick_join.json -o out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from tools.sketch import (  # noqa: E402
    _normalize_for_hash,
    bottom_k_sketch_column_with_samples,
    get_candidate_table,
)
from tools.llm_agent_tools import build_opendata_search_params  # noqa: E402
from datalake_client import SocrataDatalakeClient  # noqa: E402
from tools.aggregation import (  # noqa: E402
    aggregate_candidate_by_join_key,
    aggregate_target_by_join_key,
    convert_numeric_columns,
)
from tools.correlation import compute_feature_correlations, merge_target_with_candidate  # noqa: E402

PERTURBATION_YAML = _PROJECT / "configs" / "perturbation.yaml"
DEFAULT_PIPELINE_CONFIG = _PROJECT / "configs" / "agent_pipeline_config.yaml"
DEFAULT_BASE_DIR = _PROJECT / "query_table"
DEFAULT_DATA_FILENAME = "rows.csv"

# Same 12 benchmark query tables as experiments/run_perturbation_experiments.TASKS
TAU_VALUES: tuple[float, ...] = (0.1, 0.5, 0.9)
BETA_VALUES: tuple[float, ...] = (0.1, 0.5, 0.9)

DEFAULT_TABLE_FOLDERS: tuple[str, ...] = (
    "COVID-Chicago",
    "Demo-Chicago",
    "Economic-Chicago",
    "Education-Chicago",
    "COVID-NYC",
    "Demo-NYC",
    "Economic-NYC",
    "Education-NYC",
    "Environment_NYC",
    "Food Inspections-NYC",
    "Food Inspections-Chicago",
    "Building Permits-Chicago",
)

# Same dimensions as experiments/run_perturbation_experiments.TASK_DIMENSIONS (+ Food Inspections-NYC).
TASK_DIMENSIONS: dict[str, dict[str, list]] = {
    "COVID-Chicago": {"Domain/Field": ["covid-19"], "Geographic": ["Chicago"], "Temporal": ["daily"], "Population Group": ["all"]},
    "Demo-Chicago": {"Domain/Field": ["public health"], "Geographic": ["Chicago"], "Temporal": ["all"], "Population Group": ["all"]},
    "Economic-Chicago": {"Domain/Field": ["community survey data"], "Geographic": ["Chicago"], "Temporal": ["all"], "Population Group": ["annual income lower than $25000"]},
    "Education-Chicago": {"Domain/Field": ["public school"], "Geographic": ["Chicago"], "Temporal": ["2011-2012"], "Population Group": ["all"]},
    "COVID-NYC": {"Domain/Field": ["covid-19"], "Geographic": ["New York City"], "Temporal": ["daily"], "Population Group": ["all"]},
    "Demo-NYC": {"Domain/Field": ["Poverty"], "Geographic": ["New York City"], "Temporal": ["2018"], "Population Group": ["all"]},
    "Economic-NYC": {"Domain/Field": ["Poverty"], "Geographic": ["New York City"], "Temporal": ["2018"], "Population Group": ["all"]},
    "Education-NYC": {"Domain/Field": ["education"], "Geographic": ["New York City"], "Temporal": ["2009-2010"], "Population Group": ["all"]},
    "Environment_NYC": {"Domain/Field": ["tree"], "Geographic": ["New York City"], "Temporal": ["2015"], "Population Group": ["all"]},
    "Food Inspections-NYC": {"Domain/Field": ["food inspections"], "Geographic": ["New York City"], "Temporal": ["all"], "Population Group": ["all"]},
    "Food Inspections-Chicago": {"Domain/Field": ["food inspections"], "Geographic": ["Chicago"], "Temporal": ["all"], "Population Group": ["all"]},
    "Building Permits-Chicago": {"Domain/Field": ["building permits"], "Geographic": ["Chicago"], "Temporal": ["all"], "Population Group": ["all"]},
}


def _query_resource_id(base_dir: Path, folder: str) -> str | None:
    """Socrata dataset id from query_table/<folder>/metadata.json (resource.id)."""
    meta = base_dir / folder / "metadata.json"
    if not meta.is_file():
        return None
    try:
        with open(meta, encoding="utf-8") as f:
            data = json.load(f)
        rid = (data.get("resource") or {}).get("id")
        return str(rid).strip() if rid else None
    except Exception:
        return None


def _load_table_join_config() -> dict[str, Any]:
    if not PERTURBATION_YAML.is_file():
        raise FileNotFoundError(f"Missing perturbation config: {PERTURBATION_YAML}")
    with open(PERTURBATION_YAML, encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    tables = root.get("tables")
    if not isinstance(tables, dict):
        raise ValueError(f"Expected 'tables' dict in {PERTURBATION_YAML}")
    return tables


def _normalize_join_columns(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x).strip()]
    return []


def sketch_join_column_series(
    series: pd.Series,
    *,
    k: int = 1024,
    k_max: int = 1024,
) -> dict[str, Any]:
    """Match phase_join_columns single-key sketch (ratio=None)."""
    col = series.name if series.name is not None else "join_key"
    series = series.rename(col)
    sketch_hashes, sketch_values, _name = bottom_k_sketch_column_with_samples(
        series, k=k, ratio=None, k_max=k_max
    )
    n_keep = min(k, len(sketch_values))
    capped_vals = sketch_values[:n_keep]
    capped_hashes: list[float] = []
    if hasattr(sketch_hashes, "tolist"):
        h_list = sketch_hashes.tolist()
        n_h = min(n_keep, len(h_list))
        capped_hashes = [float(x) for x in h_list[:n_h]]
    return {
        "join_column": str(col),
        "sketch_k": k,
        "sketch_k_max": k_max,
        "sketch_ratio": None,
        "n_distinct_non_null": int(series.dropna().astype(object).nunique()),
        "sketch_distinct_count": len(sketch_values),
        "sketch_values": capped_vals,
        "sketch_hashes": capped_hashes,
    }


def run_query_join_sketches(
    table_folders: list[str],
    *,
    base_dir: Path,
    data_filename: str,
) -> dict[str, Any]:
    tables_cfg = _load_table_join_config()
    out: dict[str, Any] = {
        "config_source": str(PERTURBATION_YAML),
        "base_dir": str(base_dir),
        "data_filename": data_filename,
        "tables": {},
    }

    for folder in table_folders:
        cfg = tables_cfg.get(folder) or {}
        join_cols = _normalize_join_columns(cfg.get("join_columns"))
        rows_path = base_dir / folder / data_filename
        entry: dict[str, Any] = {
            "join_columns_config": join_cols,
            "target_column": cfg.get("target_column"),
            "rows_path": str(rows_path),
            "status": "pending",
            "per_join_column": {},
        }
        if not join_cols:
            entry["status"] = "error"
            entry["error"] = "no join_columns in perturbation.yaml for this table"
            out["tables"][folder] = entry
            continue
        if not rows_path.is_file():
            entry["status"] = "error"
            entry["error"] = f"rows file not found: {rows_path}"
            out["tables"][folder] = entry
            continue

        df = pd.read_csv(rows_path, low_memory=False)
        entry["n_rows"] = int(len(df))
        ok = True
        for jc in join_cols:
            if jc not in df.columns:
                entry["per_join_column"][jc] = {
                    "error": f"join column not in CSV columns: {jc!r}",
                }
                ok = False
                continue
            try:
                entry["per_join_column"][jc] = sketch_join_column_series(df[jc])
            except Exception as e:  # noqa: BLE001
                entry["per_join_column"][jc] = {"error": str(e)}
                ok = False
        entry["status"] = "ok" if ok else "error"
        if not ok:
            entry["error"] = "one or more join columns failed sketch; see per_join_column"
        out["tables"][folder] = entry

    return out


def run_datalake_search_for_tables(
    table_folders: list[str],
    *,
    query_table_base: Path,
    pipeline_config_path: Path,
    metadata_limit: int = 10000,
    candidate_cap: int | None = None,
) -> dict[str, Any]:
    """
    Step 2: discovery API search (same entrypoint as phase_table_selection).

    Uses build_opendata_search_params(dimension_specifications, join_col_csv, target)
    and SocrataDatalakeClient.read_metadata with the same limit/params style as the pipeline.
    """
    from agent_config_loader import load_config

    cfg = load_config(str(pipeline_config_path))
    datalake_cfg = (cfg.get("data") or {}).get("datalake") or {}
    client = SocrataDatalakeClient(datalake_cfg)
    max_tables = int(datalake_cfg.get("max_tables") or 10)
    if candidate_cap is None:
        candidate_cap = max(max_tables * 5, 50)

    tables_yaml = _load_table_join_config()
    out: dict[str, Any] = {
        "mode": "datalake_search",
        "pipeline_config": str(pipeline_config_path.resolve()),
        "metadata_limit": metadata_limit,
        "candidate_cap": candidate_cap,
        "tables": {},
    }

    for folder in table_folders:
        dims = TASK_DIMENSIONS.get(folder) or {}
        tcfg = tables_yaml.get(folder) or {}
        join_cols = _normalize_join_columns(tcfg.get("join_columns"))
        target = tcfg.get("target_column")
        join_col_for_search = ", ".join(join_cols) if join_cols else ""
        search_domains, search_q = build_opendata_search_params(
            dims, join_col_for_search, str(target).strip() if target else None
        )
        domain_for_fetch = (search_domains[0] if search_domains else None) or (
            (datalake_cfg.get("domains") or [None])[0]
        )
        exclude_id = _query_resource_id(query_table_base, folder)
        exclude_list: list[str] = [x for x in [exclude_id] if x]

        meta_resp = client.read_metadata(
            search_domains=search_domains,
            search_q=search_q if search_q.strip() else None,
            exclude_tables=exclude_list,
            limit=metadata_limit,
            offset=0,
        )
        by_ds = meta_resp.get("metadata_by_dataset") or {}
        if not isinstance(by_ds, dict):
            by_ds = {}
        ordered_ids = [str(k).strip() for k in by_ds if str(k).strip()]
        capped_ids = ordered_ids[:candidate_cap]

        entry: dict[str, Any] = {
            "dimension_specifications": dims,
            "join_columns": join_cols,
            "target_column": target,
            "search_domains": search_domains,
            "search_q": search_q,
            "domain_for_fetch": domain_for_fetch,
            "exclude_query_dataset_id": exclude_id,
            "raw_results_count": meta_resp.get("raw_results_count"),
            "resultSetSize": meta_resp.get("resultSetSize"),
            "errors": meta_resp.get("errors") or {},
            "candidate_table_ids": capped_ids,
            "n_candidates_returned": len(capped_ids),
            "metadata_by_dataset": {i: by_ds[i] for i in capped_ids if i in by_ds},
        }
        out["tables"][folder] = entry

    return out


def _sketch_unique_normalized_keys(sketch_values: list[Any]) -> tuple[list[str], set[str]]:
    """Stable order of first-seen normalized keys; set for membership."""
    ordered: list[str] = []
    seen: set[str] = set()
    for v in sketch_values or []:
        n = _normalize_for_hash(v)
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered, seen


def _column_sketch_coverage(sketch_norm_unique: set[str], cand_series: pd.Series) -> tuple[int, int, float]:
    """matched count, denominator, rate in [0,1]."""
    denom = len(sketch_norm_unique)
    if denom == 0:
        return 0, 0, 0.0
    cand_set = {_normalize_for_hash(v) for v in cand_series.dropna().astype(object)}
    matched = sum(1 for u in sketch_norm_unique if u in cand_set)
    return matched, denom, matched / denom


def _pick_best_column_coverage(cand_df: pd.DataFrame, sketch_norm_unique: set[str]) -> dict[str, Any]:
    """Top-1 by coverage; tie-break lexicographically smallest column name."""
    if sketch_norm_unique is None or len(sketch_norm_unique) == 0:
        return {"error": "empty_sketch_key_set"}
    best_col: str | None = None
    best_cov = -1.0
    for col in cand_df.columns:
        _m, _d, cov = _column_sketch_coverage(sketch_norm_unique, cand_df[col])
        if cov > best_cov + 1e-15:
            best_cov, best_col = cov, col
        elif abs(cov - best_cov) <= 1e-15 and best_col is not None and col < best_col:
            best_col = col
        elif abs(cov - best_cov) <= 1e-15 and best_col is None:
            best_col = col
    return {
        "top1_column": best_col,
        "top1_coverage": best_cov if best_col is not None else None,
    }


def run_pick_join_by_sketch_coverage(
    table_folders: list[str],
    *,
    query_table_base: Path,
    pipeline_config_path: Path,
    data_filename: str,
    search_json_path: Path | None,
    metadata_limit: int,
    candidate_cap: int | None,
) -> dict[str, Any]:
    """
    Query sketch from query_table/<task>/rows.csv; candidates from step 2; per candidate:
    coverage pick for join column, then sketch on chosen column (sketch_join_column_series).
    """
    if search_json_path is not None:
        if not search_json_path.is_file():
            raise FileNotFoundError(f"--search-json not found: {search_json_path}")
        search_report = json.loads(search_json_path.read_text(encoding="utf-8"))
    else:
        search_report = run_datalake_search_for_tables(
            table_folders,
            query_table_base=query_table_base,
            pipeline_config_path=pipeline_config_path,
            metadata_limit=metadata_limit,
            candidate_cap=candidate_cap,
        )

    sketch_pack = run_query_join_sketches(
        table_folders,
        base_dir=query_table_base,
        data_filename=data_filename,
    )

    out: dict[str, Any] = {"mode": "pick_join_sketch_coverage", "tables": {}}

    for folder in table_folders:
        t_search = (search_report.get("tables") or {}).get(folder) or {}
        t_sketch = (sketch_pack.get("tables") or {}).get(folder) or {}
        join_cols = _normalize_join_columns(t_sketch.get("join_columns_config"))
        entry: dict[str, Any] = {"candidates": []}

        if len(join_cols) != 1:
            entry["status"] = "error"
            entry["error"] = (
                f"pick-join supports exactly one join column; got {len(join_cols)}: {join_cols}"
            )
            out["tables"][folder] = entry
            continue

        jc = join_cols[0]
        pj = (t_sketch.get("per_join_column") or {}).get(jc) or {}
        if t_sketch.get("status") != "ok" or "error" in pj:
            entry["status"] = "error"
            entry["error"] = t_sketch.get("error") or pj.get("error") or "sketch failed"
            out["tables"][folder] = entry
            continue

        sketch_vals = pj.get("sketch_values") or []
        _, sketch_norm_unique = _sketch_unique_normalized_keys(sketch_vals)
        entry["query_join_column"] = jc

        cand_ids = [str(x).strip() for x in (t_search.get("candidate_table_ids") or []) if str(x).strip()]
        domain = t_search.get("domain_for_fetch")
        if not cand_ids:
            entry["status"] = "error"
            entry["error"] = "no candidate_table_ids from search step"
            out["tables"][folder] = entry
            continue
        if not domain:
            entry["status"] = "error"
            entry["error"] = "missing domain_for_fetch from search step"
            out["tables"][folder] = entry
            continue

        for cid in cand_ids:
            one: dict[str, Any] = {"candidate_table_id": cid, "domain": domain}
            try:
                df, status = get_candidate_table(cid, domain)
            except Exception as e:  # noqa: BLE001
                one["status"] = "fetch_error"
                one["error"] = str(e)
                entry["candidates"].append(one)
                continue
            if not status.get("success") or df is None or df.empty:
                one["status"] = "fetch_failed"
                one["error"] = status.get("reason") or "empty dataframe"
                entry["candidates"].append(one)
                continue

            pick = _pick_best_column_coverage(df, sketch_norm_unique)
            if "error" in pick:
                one["status"] = "pick_error"
                one["error"] = pick["error"]
                entry["candidates"].append(one)
                continue

            jn = pick.get("top1_column")
            if not jn or jn not in df.columns:
                one["status"] = "pick_error"
                one["error"] = "top1 join column missing in candidate dataframe"
                entry["candidates"].append(one)
                continue

            try:
                one["candidate_join_sketch"] = sketch_join_column_series(df[jn])
            except Exception as e:  # noqa: BLE001
                one["status"] = "sketch_error"
                one["error"] = str(e)
                entry["candidates"].append(one)
                continue

            one["status"] = "ok"
            one["top1_join_column"] = jn
            one["top1_coverage"] = pick.get("top1_coverage")
            entry["candidates"].append(one)

        if any(c.get("status") == "ok" for c in entry["candidates"]):
            entry["status"] = "ok"
        else:
            entry["status"] = "error"
            entry["error"] = "no candidate produced a successful coverage pick"

        out["tables"][folder] = entry

    return out


def _prepare_df_and_list_numerical_features_for_corr(
    cand_df: pd.DataFrame,
    join_columns: list[str],
    *,
    min_convert_ratio: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a working copy with text-like columns coerced via pd.to_numeric (errors='coerce'),
    matching convert_numeric_columns rules, then extend to any remaining non-numeric dtypes
    (e.g. string) so API-fetched tables still expose numerical features for correlation.

    Returns (prepared_df, feature_column_names) excluding join columns; drops constant/all-NaN numerics.
    """
    jc_set = set(join_columns)
    df_work = convert_numeric_columns(cand_df.copy(), exclude_columns=list(join_columns))

    for col in df_work.columns:
        if col in jc_set:
            continue
        if pd.api.types.is_numeric_dtype(df_work[col]):
            continue
        converted = pd.to_numeric(df_work[col], errors="coerce")
        original_non_null = df_work[col].notna().sum()
        if int(original_non_null) == 0:
            continue
        if converted.notna().sum() / int(original_non_null) >= float(min_convert_ratio):
            df_work[col] = converted

    out: list[str] = []
    for c in df_work.columns:
        if c in jc_set:
            continue
        s = df_work[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if int(s.notna().sum()) < 2:
            continue
        if int(s.nunique(dropna=True)) <= 1:
            continue
        out.append(c)
    return df_work, out


def run_top10_numerical_correlations(
    table_folders: list[str],
    *,
    query_table_base: Path,
    data_filename: str,
    pick_join_json: Path,
    top_k: int,
) -> dict[str, Any]:
    """Step 5: merge + correlation; numerical features via to_numeric coercion; top-k by |value|."""
    payload = json.loads(pick_join_json.read_text(encoding="utf-8"))
    tables_yaml = _load_table_join_config()
    bd = str(query_table_base.resolve())
    out: dict[str, Any] = {"mode": "top10_numerical_correlation", "tables": {}}

    for folder in table_folders:
        entry: dict[str, Any] = {"candidates": []}
        pj = (payload.get("tables") or {}).get(folder) or {}
        tcfg = tables_yaml.get(folder) or {}
        join_cols = _normalize_join_columns(tcfg.get("join_columns"))
        tgt = tcfg.get("target_column")
        rows_path = query_table_base / folder / data_filename

        if len(join_cols) != 1 or not tgt:
            entry["status"] = "error"
            entry["error"] = "need single join column and target in perturbation.yaml"
            out["tables"][folder] = entry
            continue
        if not rows_path.is_file():
            entry["status"] = "error"
            entry["error"] = f"missing {rows_path}"
            out["tables"][folder] = entry
            continue

        jc = join_cols[0]
        query_df = pd.read_csv(rows_path, low_memory=False)
        try:
            target_agg, target_type = aggregate_target_by_join_key(
                query_df,
                [jc],
                str(tgt),
                base_dir=bd,
                join_table_folder=folder,
            )
        except Exception as e:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = str(e)
            out["tables"][folder] = entry
            continue

        entry["target_agg_rows"] = int(len(target_agg))
        entry["query_join_column"] = jc
        entry["target_column"] = str(tgt)
        entry["target_type"] = target_type

        for cand in pj.get("candidates") or []:
            rec: dict[str, Any] = {"candidate_table_id": cand.get("candidate_table_id")}
            if cand.get("status") != "ok":
                rec["status"] = "skipped"
                entry["candidates"].append(rec)
                continue

            cid = str(cand.get("candidate_table_id") or "").strip()
            domain = cand.get("domain")
            jn = cand.get("top1_join_column")
            if not cid or not domain or not jn:
                rec["status"] = "error"
                rec["error"] = "missing candidate id, domain, or top1_join_column"
                entry["candidates"].append(rec)
                continue

            try:
                df, st = get_candidate_table(cid, domain)
            except Exception as e:  # noqa: BLE001
                rec["status"] = "fetch_error"
                rec["error"] = str(e)
                entry["candidates"].append(rec)
                continue
            if not st.get("success") or df is None or df.empty or jn not in df.columns:
                rec["status"] = "fetch_failed"
                rec["error"] = (st.get("reason") if isinstance(st, dict) else None) or "empty or missing join col"
                entry["candidates"].append(rec)
                continue

            df_prep, num_cols = _prepare_df_and_list_numerical_features_for_corr(df, [str(jn)])
            rec["pick_join_column"] = str(jn)
            rec["n_num_cols"] = len(num_cols)
            if not num_cols:
                rec["status"] = "ok"
                rec["top10"] = []
                rec["cand_agg_rows"] = None
                rec["merged_rows"] = None
                rec["corr_skip_reason"] = "no_numerical_columns"
                entry["candidates"].append(rec)
                continue

            try:
                sub = df_prep[[str(jn)] + num_cols].copy()
                cand_agg = aggregate_candidate_by_join_key(
                    sub,
                    [str(jn)],
                    table_id=cid,
                    fasttext_model_path=None,
                    llm_summarize=False,
                )
                rec["cand_agg_rows"] = int(len(cand_agg))
                if str(jn) != jc:
                    cand_agg = cand_agg.rename(columns={str(jn): jc})
                merged = merge_target_with_candidate(
                    target_agg,
                    cand_agg,
                    [jc],
                    target_col=str(tgt),
                    target_type=target_type,
                )
                rec["merged_rows"] = int(len(merged))
                if merged.empty or len(merged) < 2:
                    rec["status"] = "ok"
                    rec["top10"] = []
                    rec["corr_skip_reason"] = "merged_too_small"
                    entry["candidates"].append(rec)
                    continue
                corr_df = compute_feature_correlations(
                    merged,
                    [jc],
                    str(tgt),
                    target_type,
                    fasttext_model_path=None,
                )
                rec["corr_df_rows"] = int(len(corr_df))
            except Exception as e:  # noqa: BLE001
                rec["status"] = "error"
                rec["error"] = str(e)
                entry["candidates"].append(rec)
                continue

            eligible = set(num_cols)
            flt = corr_df[
                corr_df["feature"].isin(eligible) & (corr_df["feature_type"] == "numerical")
            ].copy()
            flt["value"] = pd.to_numeric(flt["value"], errors="coerce")
            flt = flt.dropna(subset=["value"])
            rec["n_finite_corr_rows"] = int(len(flt))
            if flt.empty:
                rec["status"] = "ok"
                rec["top10"] = []
                rec["corr_skip_reason"] = "no_finite_correlations"
                entry["candidates"].append(rec)
                continue
            flt = flt.assign(_abs=flt["value"].abs()).sort_values(
                ["_abs", "feature"], ascending=[False, True]
            )
            top = flt.head(max(0, int(top_k))).drop(columns=["_abs"], errors="ignore")
            rec["status"] = "ok"
            rec["top10"] = top[["feature", "value", "metric"]].to_dict(orient="records")
            rec["corr_skip_reason"] = None
            entry["candidates"].append(rec)

        entry["status"] = "ok"
        out["tables"][folder] = entry

    return out


def _perturbed_base(project: Path, tau: float, beta: float) -> Path:
    """Same directory convention as experiments/run_perturbation_experiments._build_perturbed_pipeline_config."""
    return project / f"perturbed_{tau}_{beta}"


def _strip_sketch_payload_for_compact(cell: dict[str, Any], *, drop_values: bool) -> None:
    """Mutate one cell's per_join_column entries to shrink JSON."""
    pjc = cell.get("per_join_column")
    if not isinstance(pjc, dict):
        return
    for _jc, col in pjc.items():
        if not isinstance(col, dict) or "error" in col:
            continue
        col.pop("sketch_hashes", None)
        if drop_values:
            col.pop("sketch_values", None)


def run_perturbation_grid_sketches(
    table_folders: list[str],
    *,
    project_root: Path,
    data_filename: str,
    taus: tuple[float, ...] = TAU_VALUES,
    betas: tuple[float, ...] = BETA_VALUES,
    compact_hashes: bool = True,
    drop_sketch_values: bool = True,
) -> dict[str, Any]:
    """12 tables × |τ|×|β| sketch runs on perturbed_{τ}_{β}/<table>/rows.csv."""
    out: dict[str, Any] = {
        "mode": "perturbation_grid_sketch",
        "project_root": str(project_root.resolve()),
        "tau_values": list(taus),
        "beta_values": list(betas),
        "perturbed_dir_pattern": "perturbed_{tau}_{beta}/<join_table>/rows.csv",
        "config_source": str(PERTURBATION_YAML),
        "data_filename": data_filename,
        "tables": {},
    }
    for folder in table_folders:
        cells: list[dict[str, Any]] = []
        for tau in taus:
            for beta in betas:
                base = _perturbed_base(project_root, tau, beta)
                sub = run_query_join_sketches(
                    [folder],
                    base_dir=base,
                    data_filename=data_filename,
                )
                cell = sub["tables"].get(folder, {})
                if isinstance(cell, dict):
                    cell = {
                        "tau": tau,
                        "beta": beta,
                        "perturbed_base": str(base),
                        **cell,
                    }
                else:
                    cell = {
                        "tau": tau,
                        "beta": beta,
                        "perturbed_base": str(base),
                        "status": "error",
                        "error": "unexpected sketch payload",
                    }
                if compact_hashes or drop_sketch_values:
                    _strip_sketch_payload_for_compact(cell, drop_values=drop_sketch_values)
                cells.append(cell)
        out["tables"][folder] = {"cells": cells}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "mode",
        nargs="?",
        default="grid",
        choices=("grid", "single", "search", "pick-join", "corr"),
        help="grid | single | search | pick-join | corr",
    )
    ap.add_argument(
        "--tables",
        nargs="*",
        default=list(DEFAULT_TABLE_FOLDERS),
        metavar="FOLDER",
        help="Query table folder names (default: 12 TASKS tables)",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory with per-table folders: single/pick-join sketch CSV + metadata; search exclude id (default: query_table/)",
    )
    ap.add_argument(
        "--data-filename",
        default=DEFAULT_DATA_FILENAME,
        help="CSV inside each table folder (default: rows.csv)",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT,
        help="[grid mode] Repo root where perturbed_{tau}_{beta} live",
    )
    ap.add_argument(
        "--pipeline-config",
        type=Path,
        default=DEFAULT_PIPELINE_CONFIG,
        help="[search mode] agent_pipeline_config.yaml (datalake app token / domains / max_tables)",
    )
    ap.add_argument(
        "--search-metadata-limit",
        type=int,
        default=10000,
        help="[search mode] read_metadata limit (same order of magnitude as phase_table_selection)",
    )
    ap.add_argument(
        "--search-candidate-cap",
        type=int,
        default=None,
        help="[search / pick-join] max candidate ids (default: max(5*max_tables,50) from config)",
    )
    ap.add_argument(
        "--search-json",
        type=Path,
        default=None,
        help="[pick-join] reuse step-2 JSON instead of calling discovery API",
    )
    ap.add_argument(
        "--pick-join-json",
        type=Path,
        default=None,
        help="[corr] JSON produced by pick-join mode",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="[corr] max features per candidate (default 10)",
    )
    ap.add_argument(
        "--include-sketch-values",
        action="store_true",
        help="[grid mode] Keep sketch_values in each cell (large JSON)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path (UTF-8)",
    )
    ap.add_argument(
        "--compact",
        action="store_true",
        help="[single mode] Omit sketch_hashes. [grid mode] default already omits hashes & values unless --include-sketch-values",
    )
    args = ap.parse_args()

    if args.mode == "corr" and not args.pick_join_json:
        print("corr mode requires --pick-join-json", file=sys.stderr)
        return 1

    if args.mode == "grid":
        report = run_perturbation_grid_sketches(
            list(args.tables),
            project_root=args.project_root.resolve(),
            data_filename=args.data_filename,
            compact_hashes=True,
            drop_sketch_values=not args.include_sketch_values,
        )
    elif args.mode == "search":
        report = run_datalake_search_for_tables(
            list(args.tables),
            query_table_base=args.base_dir.resolve(),
            pipeline_config_path=args.pipeline_config.resolve(),
            metadata_limit=args.search_metadata_limit,
            candidate_cap=args.search_candidate_cap,
        )
    elif args.mode == "pick-join":
        report = run_pick_join_by_sketch_coverage(
            list(args.tables),
            query_table_base=args.base_dir.resolve(),
            pipeline_config_path=args.pipeline_config.resolve(),
            data_filename=args.data_filename,
            search_json_path=args.search_json.resolve() if args.search_json else None,
            metadata_limit=args.search_metadata_limit,
            candidate_cap=args.search_candidate_cap,
        )
    elif args.mode == "corr":
        report = run_top10_numerical_correlations(
            list(args.tables),
            query_table_base=args.base_dir.resolve(),
            data_filename=args.data_filename,
            pick_join_json=args.pick_join_json.resolve(),
            top_k=max(0, args.top_k),
        )
    else:
        report = run_query_join_sketches(
            list(args.tables),
            base_dir=args.base_dir.resolve(),
            data_filename=args.data_filename,
        )
        if args.compact:
            for _name, t in report.get("tables", {}).items():
                if not isinstance(t, dict):
                    continue
                for _jc, col in (t.get("per_join_column") or {}).items():
                    if isinstance(col, dict) and "sketch_hashes" in col:
                        col.pop("sketch_hashes", None)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
