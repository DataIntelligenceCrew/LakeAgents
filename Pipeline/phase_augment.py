import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from Classification_regression import (
    preprocess_data,
    run_classification_task,
    run_classification_task_xgboost,
    run_regression_task,
)
from tools.aggregation import aggregate_categorical_column, aggregate_selected_tables, aggregate_target_by_join_key
from tools.column_descriptions import get_column_descriptions_from_index
from tools.correlation import compute_feature_correlations, merge_target_with_candidate
from tools.data_quality import apply_quality_actions
from tools.sketch import _normalize_for_hash, get_candidate_table

from Pipeline.context import PipelineContext
from Pipeline.utils import (
    extract_json_by_key_from_full_text,
    first_non_empty_values,
    normalize_augment_column_list,
)


def _evaluate_metric(
    df: pd.DataFrame,
    target_column: str,
    join_columns: List[str],
    task_type: str,
    classification_model: str = "fast",
) -> Optional[float]:
    try:
        features = [c for c in df.columns if c != target_column and c not in join_columns]
        if not features:
            return None
        tmp = df[features + [target_column]].dropna(subset=[target_column])
        if len(tmp) < 10:
            return None
        X, y, target_encoder, _ = preprocess_data(tmp, target_column, task_type)
        if X is None or len(X) == 0:
            return None
        if task_type == "classification":
            if str(classification_model).lower() == "xgboost":
                return run_classification_task_xgboost(X, y, target_encoder).get("f1_score")
            return run_classification_task(X, y, target_encoder).get("f1_score")
        return run_regression_task(X, y).get("r2_score")
    except Exception:
        return None


def _build_feature_frame_by_dtype(
    cand_df: pd.DataFrame,
    selected_cols_local: List[str],
    feature_col: str,
    dtype_final: str,
    join_columns: List[str],
) -> Optional[pd.DataFrame]:
    if feature_col not in cand_df.columns:
        return None
    cols = selected_cols_local + [feature_col]
    work = cand_df[cols].copy()
    if selected_cols_local and len(selected_cols_local) == len(join_columns):
        work = work.rename(columns=dict(zip(selected_cols_local, join_columns)))
    for jc in join_columns:
        if jc not in work.columns:
            return None
        work[jc] = work[jc].apply(_normalize_for_hash)
    dtype_norm = (dtype_final or "").strip().lower()
    if dtype_norm == "numerical":
        work[feature_col] = pd.to_numeric(work[feature_col], errors="coerce")
        return work.groupby(join_columns)[feature_col].mean().reset_index()
    if dtype_norm == "categorical":
        work[feature_col] = work[feature_col].fillna("__NA__").astype(str)
        return aggregate_categorical_column(
            work, join_columns, feature_col, method="proportion", return_as_vector=False
        )
    work[feature_col] = work[feature_col].fillna("").astype(str)
    return work.groupby(join_columns)[feature_col].apply(
        lambda s: " | ".join([v for v in s.tolist() if str(v).strip()][:3])
    ).reset_index()


def _merge_with_suffix(base_df: pd.DataFrame, to_merge_df: pd.DataFrame, cand_name: str, join_columns: List[str]) -> pd.DataFrame:
    to_merge = to_merge_df.copy()
    new_cols = [c for c in to_merge.columns if c not in join_columns]
    suffix = f"_{(cand_name or 'unknown').replace('-', '_')}"
    rename_map = {c: f"{c}{suffix}" for c in new_cols if c in base_df.columns}
    if rename_map:
        to_merge = to_merge.rename(columns=rename_map)
    return base_df.merge(to_merge, on=join_columns, how="left")


async def run_augment(ctx: PipelineContext) -> None:
    config = ctx.config
    aug_cfg = ((config.config or {}).get("pipeline_thresholds", {}) or {}).get("augment", {})
    top_corr_k = int(aug_cfg.get("top_corr_k", 5))
    bottom_corr_k = int(aug_cfg.get("bottom_corr_k", 5))
    max_ranked_steps = int(aug_cfg.get("max_ranked_steps", 10))
    min_metric_gain_delta = float(aug_cfg.get("min_metric_gain_delta", 0.001))
    augment_runner = ctx.state["augment_runner"]
    final_selected_tables = ctx.state.get("final_selected_tables", []) or []
    quality_enhanced_tables = ctx.state.get("quality_enhanced_tables", {}) or {}
    domain_for_fetch = ctx.state.get("domain_for_fetch")
    join_columns = ctx.state.get("join_columns", [])
    join_df = ctx.state.get("join_df")
    join_df_full = ctx.state.get("join_df_full")
    join_col_descs = ctx.state.get("join_col_descs", {})
    query_table_description = ctx.state.get("query_table_description", "")
    inherited_augment_columns = ctx.state.get("inherited_augment_columns", {})
    if not isinstance(inherited_augment_columns, dict):
        inherited_augment_columns = {}
    decision_log = ctx.state.get("decision_log", {})
    phase_log: Dict[str, Any] = {
        "selected": [],
        "excluded": [],
    }

    t_baseline = time.perf_counter()
    join_df_ml = join_df.copy()
    baseline_metric = _evaluate_metric(join_df_ml, ctx.target_column, join_columns, ctx.task_type)
    ctx.pipeline_timings["09_baseline_sketch_ml"] = time.perf_counter() - t_baseline

    t_target_agg = time.perf_counter()
    target_agg, target_type = aggregate_target_by_join_key(
        join_df_ml, join_columns, ctx.target_column, base_dir=ctx.base_dir, join_table_folder=ctx.real_join_table_name
    )
    ctx.pipeline_timings["10_aggregate_target_by_join_key"] = time.perf_counter() - t_target_agg

    t_agg = time.perf_counter()
    aggregated_results = aggregate_selected_tables(
        final_selected_tables,
        base_dir=ctx.base_dir,
        opendata_domain=domain_for_fetch,
        join_key_filter=ctx.state.get("join_sketch_original_values"),
        query_join_columns=join_columns,
        llm_join_keys=ctx.state.get("join_sketch_original_values"),
        target_agg=target_agg,
        target_column=ctx.target_column,
        target_type=target_type,
        table_overrides=quality_enhanced_tables,
    )
    ctx.pipeline_timings["11_aggregate_candidate_tables"] = time.perf_counter() - t_agg

    t_corr = time.perf_counter()
    for result in aggregated_results:
        cand_agg = result.get("aggregated_df")
        if cand_agg is None or cand_agg.empty:
            result["correlation_table"] = pd.DataFrame(columns=["feature", "metric", "value"])
            continue
        selected_cols = result.get("selected_columns", [])
        if selected_cols and len(selected_cols) == len(join_columns):
            cand_agg = cand_agg.rename(columns=dict(zip(selected_cols, join_columns)))
        merged = merge_target_with_candidate(target_agg, cand_agg, join_columns, target_col=ctx.target_column, target_type=target_type)
        result["correlation_table"] = compute_feature_correlations(merged, join_columns, ctx.target_column, target_type)
    ctx.pipeline_timings["12_correlation_per_candidate"] = time.perf_counter() - t_corr

    t_rank = time.perf_counter()
    for result in aggregated_results:
        cand_name = result.get("candidate_table", "?")
        corr_df = result.get("correlation_table", pd.DataFrame())
        if corr_df.empty:
            result["selected_augment_columns"] = []
            result["augment_reasoning"] = "No correlation data."
            continue
        cand_descs = get_column_descriptions_from_index(cand_name)
        target_desc = join_col_descs.get(ctx.target_column, "")
        cand_agg = result.get("aggregated_df")
        candidate_meta: Dict[str, Dict[str, Any]] = {}
        for _, row in corr_df.iterrows():
            feature_name = row["feature"]
            value = float(row["value"]) if pd.notna(row["value"]) else None
            examples = first_non_empty_values(cand_agg[feature_name], n=5) if cand_agg is not None and feature_name in cand_agg.columns else []
            candidate_meta[feature_name] = {
                "feature": feature_name,
                "metric": row["metric"],
                "value": value,
                "description": cand_descs.get(feature_name, ""),
                "feature_type": row.get("feature_type", "unknown"),
                "example_values": examples,
                "source_tags": [],
            }
        values_for_rank = {name: meta["value"] for name, meta in candidate_meta.items() if meta.get("value") is not None}
        top_corr_names = sorted(values_for_rank, key=lambda n: values_for_rank[n], reverse=True)[:top_corr_k]
        bottom_corr_names = sorted(values_for_rank, key=lambda n: values_for_rank[n])[:bottom_corr_k]
        merged_names: List[str] = []
        for name in top_corr_names + bottom_corr_names:
            if name not in merged_names:
                merged_names.append(name)
        candidate_columns = [candidate_meta[n] for n in merged_names if n in candidate_meta]
        prompt = f"""
task_mode: "rank_select"
task_type: "{ctx.task_type}"
target_column: "{ctx.target_column}"
target_column_description: "{target_desc}"
user_intent: "{ctx.user_intent}"
candidate_table: "{cand_name}"
ç: "{query_table_description}"
candidate_pool_strategy: "top-k correlation + bottom-k correlation"
candidate_columns: {json.dumps(candidate_columns, ensure_ascii=False, indent=2)}

Rank candidate_columns first, then select augment columns. Return JSON with selected_augment_columns, ranked_candidates, and reasoning.
Also return column_decisions with dtype_rule, dtype_final, and override_reason for each ranked column.
"""
        events = await augment_runner.run_debug(prompt, quiet=True)
        full_text = ""
        for event in events:
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for part in event.content.parts:
                    t = getattr(part, "text", None)
                    if t:
                        full_text += t
        parsed = extract_json_by_key_from_full_text(full_text, "selected_augment_columns", prefer_non_empty_list=True)
        parsed_rank = extract_json_by_key_from_full_text(full_text, "ranked_candidates", prefer_non_empty_list=True)
        parsed_decisions = extract_json_by_key_from_full_text(full_text, "column_decisions", prefer_non_empty_list=False)
        result["selected_augment_columns"] = normalize_augment_column_list(parsed.get("selected_augment_columns", []))
        result["augment_ranked_candidates"] = normalize_augment_column_list(
            parsed_rank.get("ranked_candidates", parsed.get("ranked_candidates", []))
        )
        result["augment_column_decisions"] = parsed_decisions.get("column_decisions", [])
        result["augment_reasoning"] = parsed.get("reasoning", "") or parsed_rank.get("reasoning", "")
    ctx.pipeline_timings["13_augment_rank_select_llm"] = time.perf_counter() - t_rank

    t_greedy = time.perf_counter()
    augmented_df = join_df_ml.copy()
    for jc in join_columns:
        if jc in augmented_df.columns:
            augmented_df[jc] = augmented_df[jc].apply(_normalize_for_hash)
    phase_selected: List[Dict[str, Any]] = []
    phase_excluded: List[Dict[str, Any]] = []
    for result in aggregated_results:
        cand_name = result.get("candidate_table")
        selected_cols = result.get("selected_columns", [])
        if cand_name in quality_enhanced_tables:
            cand_df_full = quality_enhanced_tables[cand_name].copy()
        else:
            cand_df_full, status = get_candidate_table(cand_name, domain_for_fetch)
            if not status.get("success"):
                continue
        selected_raw = normalize_augment_column_list(result.get("selected_augment_columns", []) or [])
        decision_map = {
            str(d["column"]): d
            for d in (result.get("augment_column_decisions", []) or [])
            if isinstance(d, dict) and d.get("column")
        }
        accepted_cols: List[str] = []
        rejected_cols: List[Dict[str, Any]] = []
        inherited_items = inherited_augment_columns.get(cand_name, []) or []
        if isinstance(inherited_items, list):
            for item in inherited_items:
                if isinstance(item, dict):
                    col = str(item.get("column") or "").strip()
                    dtype_final_inherited = str(item.get("dtype_final") or "").strip().lower()
                else:
                    col = str(item).strip()
                    dtype_final_inherited = ""
                if not col or col in accepted_cols or col not in cand_df_full.columns:
                    continue
                dtype_final = (
                    dtype_final_inherited
                    or str((decision_map.get(col) or {}).get("dtype_final") or "numerical").lower()
                )
                feat_frame = _build_feature_frame_by_dtype(cand_df_full, selected_cols, col, dtype_final, join_columns)
                if feat_frame is None or feat_frame.empty:
                    rejected_cols.append(
                        {
                            "column": col,
                            "reason_code": "AUG_EMPTY_FEATURE_FRAME",
                            "reason": "inherited column produced empty feature frame",
                        }
                    )
                    continue
                augmented_df = _merge_with_suffix(augmented_df, feat_frame, cand_name, join_columns)
                accepted_cols.append(col)
        for col in selected_raw[:max_ranked_steps]:
            if col in accepted_cols:
                continue
            if col not in cand_df_full.columns:
                continue
            dtype_final = str((decision_map.get(col) or {}).get("dtype_final") or "numerical").lower()
            feat_frame = _build_feature_frame_by_dtype(cand_df_full, selected_cols, col, dtype_final, join_columns)
            if feat_frame is None or feat_frame.empty:
                rejected_cols.append(
                    {"column": col, "reason_code": "AUG_EMPTY_FEATURE_FRAME", "reason": "feature frame is empty after aggregation"}
                )
                continue
            augmented_df = _merge_with_suffix(augmented_df, feat_frame, cand_name, join_columns)
            accepted_cols.append(col)
        result["selected_augment_columns"] = accepted_cols
        for c in accepted_cols:
            phase_selected.append(
                {
                    "table_id": cand_name,
                    "column": c,
                    "decision": "selected",
                    "reason_code": "AUG_SELECTED_BY_LLM_BATCH",
                    "reason": "accepted by LLM selection and batch merge",
                }
            )
        for rc in rejected_cols:
            phase_excluded.append(
                {
                    "table_id": cand_name,
                    "column": rc.get("column"),
                    "decision": "excluded",
                    "reason_code": rc.get("reason_code"),
                    "reason": rc.get("reason"),
                }
            )
    augmented_metric = _evaluate_metric(augmented_df, ctx.target_column, join_columns, ctx.task_type)
    phase_log["selected"] = phase_selected
    phase_log["excluded"] = phase_excluded
    ctx.pipeline_timings["14_greedy_sequential_and_sketch_augmented_metric"] = time.perf_counter() - t_greedy

    t_full = time.perf_counter()
    baseline_metric_full = _evaluate_metric(
        join_df_full,
        ctx.target_column,
        join_columns,
        ctx.task_type,
        classification_model="xgboost",
    )
    augmented_metric_full = baseline_metric_full
    try:
        augmented_df_full = join_df_full.copy()
        for jc in join_columns:
            if jc in augmented_df_full.columns:
                augmented_df_full[jc] = augmented_df_full[jc].apply(_normalize_for_hash)
        for result in aggregated_results:
            cand_name = result.get("candidate_table")
            selected_cols = result.get("selected_columns", []) or []
            accepted_cols = result.get("selected_augment_columns", []) or []
            if not cand_name or not selected_cols or not accepted_cols:
                continue
            cand_df_eval, status = get_candidate_table(cand_name, domain_for_fetch)
            if not status.get("success"):
                continue
            dq_dropped_columns = result.get("dq_dropped_columns", []) or []
            cand_df_eval = cand_df_eval.drop(columns=dq_dropped_columns, errors="ignore")
            dq_actions = result.get("dq_agent_column_actions", []) or []
            if dq_actions:
                cand_df_eval, _ = apply_quality_actions(cand_df_eval, actions=dq_actions, protected_columns=selected_cols)
            decision_map_eval = {
                str(d["column"]): d
                for d in (result.get("augment_column_decisions", []) or [])
                if isinstance(d, dict) and d.get("column")
            }
            for col in accepted_cols:
                if col not in cand_df_eval.columns:
                    continue
                dtype_final = str((decision_map_eval.get(col) or {}).get("dtype_final") or "numerical").lower()
                feat_frame = _build_feature_frame_by_dtype(cand_df_eval, selected_cols, col, dtype_final, join_columns)
                if feat_frame is None or feat_frame.empty:
                    continue
                augmented_df_full = _merge_with_suffix(augmented_df_full, feat_frame, cand_name, join_columns)
        augmented_metric_full = _evaluate_metric(
            augmented_df_full,
            ctx.target_column,
            join_columns,
            ctx.task_type,
            classification_model="xgboost",
        )
    except Exception:
        pass
    ctx.pipeline_timings["15_final_full_table_eval"] = time.perf_counter() - t_full

    if baseline_metric_full is not None:
        baseline_metric = baseline_metric_full
    if augmented_metric_full is not None:
        augmented_metric = augmented_metric_full
    metric_name = "r2_score" if ctx.task_type == "regression" else "f1_score"
    augment_output = [
        {
            "candidate_table": r.get("candidate_table", "?"),
            "selected_augment_columns": r.get("selected_augment_columns", []),
            "ranked_candidates": r.get("augment_ranked_candidates", []),
            "column_decisions": r.get("augment_column_decisions", []),
            "early_stop_log": r.get("early_stop_log", []),
            "reasoning": r.get("augment_reasoning", ""),
        }
        for r in aggregated_results
    ]
    ctx.state["output"] = {
        "augment_results": augment_output,
        "baseline_metric": baseline_metric,
        "augmented_metric": augmented_metric,
        "metric_name": metric_name,
        "pipeline_timings_seconds": ctx.pipeline_timings,
    }
    if isinstance(decision_log, dict):
        decision_log.setdefault("phases", {})["augment"] = phase_log
        snap = decision_log.get("threshold_snapshot", {})
        if isinstance(snap, dict):
            snap["augment"] = {
                "top_corr_k": top_corr_k,
                "bottom_corr_k": bottom_corr_k,
                "max_ranked_steps": max_ranked_steps,
                "min_metric_gain_delta": min_metric_gain_delta,
            }
            decision_log["threshold_snapshot"] = snap

