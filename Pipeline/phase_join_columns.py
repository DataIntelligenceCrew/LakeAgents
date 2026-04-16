import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from Agent.join_column_fallback import (
    build_fallback_attempts,
    evaluate_exact_join_health,
    evaluate_fuzzy_join_health,
    evaluate_similarity_gate,
)
from tools.column_descriptions import (
    get_column_descriptions_from_index,
    get_column_descriptions_from_local_metadata,
)
from tools.data_quality import apply_hard_quality_rules
from tools.join_column_tool import column_profile
from tools.sketch import _normalize_for_hash, bottom_k_sketch_column_with_samples, get_candidate_table

from Pipeline.context import PipelineContext
from Pipeline.utils import extract_json_by_key_from_full_text, first_non_empty_rows


def _filter_candidate_df_by_sketch_keys(
    cand_df_local: pd.DataFrame,
    selected_cols_local: List[str],
    join_columns: List[str],
    join_sketch_original_values: Optional[List[Any]],
    fuzzy_key_mapping_local: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    if (
        cand_df_local is None
        or cand_df_local.empty
        or not join_sketch_original_values
        or len(join_columns) != 1
        or len(selected_cols_local) != 1
    ):
        return cand_df_local
    sc = selected_cols_local[0]
    if sc not in cand_df_local.columns:
        return cand_df_local
    work_df = cand_df_local.copy()
    if fuzzy_key_mapping_local:
        work_df[sc] = work_df[sc].apply(_normalize_for_hash).map(lambda x: fuzzy_key_mapping_local.get(x))
    sketch_key_set = {_normalize_for_hash(v) for v in join_sketch_original_values}
    return work_df[work_df[sc].apply(_normalize_for_hash).isin(sketch_key_set)].copy()


async def run_join_columns(ctx: PipelineContext) -> None:
    config = ctx.config
    phase_cfg = ((config.config or {}).get("pipeline_thresholds", {}) or {}).get("join", {})
    joincol_runner = ctx.state["joincol_runner"]
    relevant_list = ctx.state.get("relevant_list", []) or []
    domain_for_fetch = ctx.state.get("domain_for_fetch")
    query_table_description = ctx.state.get("query_table_description", "")

    run_start_time = datetime.now()
    run_record = {"table_id": [], "status": [], "reason": []}
    top5_profile: Dict[str, Any] = {}
    join_fallback_record: Dict[str, Any] = {}
    final_selected_tables: List[Dict[str, Any]] = []
    fallback_coverage_threshold = float(phase_cfg.get("fallback_coverage_threshold", 0.35))
    fuzzy_score_threshold = float(phase_cfg.get("fuzzy_score_threshold", 80))
    topk_candidates = int(phase_cfg.get("topk_candidates", 5))
    hard_dq_missing_threshold = float(phase_cfg.get("hard_dq_missing_threshold", 0.30))
    hard_dq_top1_ratio_threshold = float(phase_cfg.get("hard_dq_top1_ratio_threshold", 0.90))
    decision_log = ctx.state.get("decision_log", {})
    phase_log: Dict[str, Any] = {
        "selected": [],
        "excluded": [],
        "thresholds": {
            "fallback_coverage_threshold": fallback_coverage_threshold,
            "fuzzy_score_threshold": fuzzy_score_threshold,
        },
    }

    t_join_prep = time.perf_counter()
    join_df = pd.read_csv(ctx.base_path_obj / ctx.real_join_table_name / config.data_filename, low_memory=False)
    join_df_full = join_df.copy()
    join_columns = config.join_column if isinstance(config.join_column, list) else [config.join_column]
    join_sketch_original_values = None
    if len(join_columns) == 1 and join_columns[0] in join_df.columns:
        sketch_col = join_columns[0]
        try:
            _, sketch_values, _ = bottom_k_sketch_column_with_samples(
                join_df[sketch_col], k=1024, ratio=None, k_max=1024
            )
            join_sketch_original_values = sketch_values[:1024]
            if join_sketch_original_values:
                sketch_key_set = {_normalize_for_hash(v) for v in join_sketch_original_values}
                join_df = join_df[join_df[sketch_col].apply(_normalize_for_hash).isin(sketch_key_set)].copy()
        except Exception as e:
            print(f"   ⚠️  Join-key sketch sampling skipped: {e}")

    join_col_descs = get_column_descriptions_from_local_metadata(ctx.base_dir, ctx.real_join_table_name)
    query_profiles = {jc: column_profile(join_df[jc], n_samples=5) for jc in join_columns if jc in join_df.columns}
    query_rows_5 = first_non_empty_rows(join_df, join_columns, n=5)
    ctx.pipeline_timings["06a_join_table_load_sketch_profiles"] = time.perf_counter() - t_join_prep

    t_join_phase = time.perf_counter()
    for tbl in relevant_list:
        cand_name = tbl.get("table_id")
        df, status = get_candidate_table(table_id=cand_name, opendata_domain=domain_for_fetch)
        if not status.get("success"):
            run_record["status"].append("failed")
            run_record["reason"].append(status.get("reason"))
            top5_profile[cand_name] = []
            continue

        run_record["status"].append("success")
        run_record["reason"].append(None)
        cand_col_descs = get_column_descriptions_from_index(cand_name)
        candidate_profiles = []
        for col in df.columns:
            p = column_profile(df[col], n_samples=5)
            p["description"] = cand_col_descs.get(col, "")
            candidate_profiles.append(p)

        top5_by_join_col: Dict[str, List[str]] = {}
        for jc in join_columns:
            step1_payload = {
                "task_mode": "profile_top5",
                "candidate_table": cand_name,
                "query_join_column": jc,
                "query_join_column_description": join_col_descs.get(jc, ""),
                "query_table_description": query_table_description,
                "query_join_column_profile": query_profiles.get(jc, {}),
                "candidate_column_profiles": candidate_profiles,
                "topk": topk_candidates,
            }
            step1_prompt = (
                "Select top5 possible join columns by profile only. "
                "Return JSON with key top5_candidates.\n"
                f"input: {json.dumps(step1_payload, ensure_ascii=False)}"
            )
            try:
                events = await joincol_runner.run_debug(step1_prompt, quiet=True)
                full_text = ""
                for event in events:
                    if getattr(event, "content", None) and getattr(event.content, "parts", None):
                        for part in event.content.parts:
                            t = getattr(part, "text", None)
                            if t:
                                full_text += t
                step1_json = extract_json_by_key_from_full_text(full_text, "top5_candidates", prefer_non_empty_list=True)
                raw_top5 = step1_json.get("top5_candidates", []) or []
                normalized_top5: List[str] = []
                for item in raw_top5:
                    if isinstance(item, str):
                        normalized_top5.append(item)
                    elif isinstance(item, dict):
                        name = item.get("name") or item.get("column_name")
                        if name:
                            normalized_top5.append(str(name))
                top5_by_join_col[jc] = normalized_top5[:5]
            except Exception:
                top5_by_join_col[jc] = []

        top5_profile[cand_name] = top5_by_join_col if len(join_columns) > 1 else top5_by_join_col.get(join_columns[0], [])
        candidate_cols_union: List[str] = []
        for jc in join_columns:
            for c in top5_by_join_col.get(jc, []):
                if c not in candidate_cols_union:
                    candidate_cols_union.append(c)
        has_top5_candidates = any(len(top5_by_join_col.get(jc, [])) > 0 for jc in join_columns)
        fallback_max_candidates = topk_candidates

        selected_cols: List[str] = []
        if candidate_cols_union:
            step2_payload = {
                "task_mode": "final_select",
                "candidate_table": cand_name,
                "query_join_columns": join_columns,
                "query_join_column_descriptions": {jc: join_col_descs.get(jc, "") for jc in join_columns},
                "query_table_description": query_table_description,
                "query_rows_non_empty_5": query_rows_5,
                "candidate_rows_non_empty_5": first_non_empty_rows(df, candidate_cols_union, n=5),
                "query_profiles": query_profiles,
                "candidate_profiles_top5": [p for p in candidate_profiles if p.get("column_name") in candidate_cols_union],
                "top5_candidates_by_query_col": top5_by_join_col,
                "instruction": "Use tools when needed and return selected_columns.",
            }
            step2_prompt = (
                "Choose final join columns from provided top5 candidates. "
                "Use tools if needed. Return JSON with key selected_columns.\n"
                f"input: {json.dumps(step2_payload, ensure_ascii=False)}"
            )
            try:
                events = await joincol_runner.run_debug(step2_prompt, quiet=True)
                full_text = ""
                for event in events:
                    if getattr(event, "content", None) and getattr(event.content, "parts", None):
                        for part in event.content.parts:
                            t = getattr(part, "text", None)
                            if t:
                                full_text += t
                parsed = extract_json_by_key_from_full_text(full_text, "selected_columns", prefer_non_empty_list=True)
                val = parsed.get("selected_columns", [])
                if isinstance(val, dict):
                    selected_cols = [v for v in val.values() if v]
                elif isinstance(val, list):
                    selected_cols = [v for v in val if v]
                elif val:
                    selected_cols = [val]
            except Exception:
                selected_cols = [top5_by_join_col[jc][0] for jc in join_columns if top5_by_join_col.get(jc)]

        selected_cols = [c for c in selected_cols if c in candidate_cols_union]
        if len(join_columns) == 1:
            if selected_cols:
                selected_cols = [selected_cols[0]]

        if not selected_cols:
            # If top5 exists but final_select is empty, try one fallback from top1 only once.
            if has_top5_candidates:
                selected_cols = [
                    top5_by_join_col[jc][0]
                    for jc in join_columns
                    if isinstance(top5_by_join_col.get(jc), list) and top5_by_join_col.get(jc)
                ]
                selected_cols = [c for c in selected_cols if c in candidate_cols_union]
                if len(join_columns) == 1 and selected_cols:
                    selected_cols = [selected_cols[0]]
                fallback_max_candidates = 1

        if not selected_cols:
            phase_log["excluded"].append(
                {
                    "table_id": cand_name,
                    "decision": "excluded",
                    "reason_code": "JOIN_TOPK_EMPTY",
                    "reason": "top5 candidates empty or invalid for fallback",
                    "top5_by_join_col": top5_by_join_col,
                }
            )
            continue

        attempts = build_fallback_attempts(
            selected_columns=selected_cols,
            top5_candidates_by_query_col=top5_by_join_col,
            query_join_columns=join_columns,
            max_candidates=fallback_max_candidates,
        )
        if not attempts:
            phase_log["excluded"].append(
                {
                    "table_id": cand_name,
                    "decision": "excluded",
                    "reason_code": "JOIN_TOPK_EMPTY",
                    "reason": "no valid fallback attempts could be built",
                    "top5_by_join_col": top5_by_join_col,
                }
            )
            continue
        chosen_cols = None
        chosen_join_mode = "exact"
        chosen_fuzzy_mapping: Dict[str, str] = {}
        fallback_logs: List[Dict[str, Any]] = []
        for idx, attempt_cols in enumerate(attempts, start=1):
            health = evaluate_exact_join_health(
                join_df=join_df, cand_df=df, join_columns=join_columns, selected_columns=attempt_cols, coverage_threshold=fallback_coverage_threshold
            )
            attempt_log: Dict[str, Any] = {"attempt_index": idx, "selected_columns": attempt_cols, "exact_join_health": health}
            if not health.get("is_anomaly", True):
                chosen_cols = attempt_cols
                fallback_logs.append(attempt_log)
                break
            similarity_gate = evaluate_similarity_gate(
                join_df=join_df, cand_df=df, join_columns=join_columns, selected_columns=attempt_cols, fuzzy_threshold=fuzzy_score_threshold
            )
            attempt_log["similarity_gate"] = similarity_gate
            if similarity_gate.get("gate_pass", False):
                fuzzy_eval = evaluate_fuzzy_join_health(
                    join_df=join_df, cand_df=df, join_columns=join_columns, selected_columns=attempt_cols,
                    coverage_threshold=fallback_coverage_threshold, fuzzy_threshold=fuzzy_score_threshold
                )
                attempt_log["fuzzy_join"] = fuzzy_eval
                if not (fuzzy_eval.get("health", {}) or {}).get("is_anomaly", True):
                    chosen_cols = attempt_cols
                    chosen_join_mode = "fuzzy"
                    chosen_fuzzy_mapping = fuzzy_eval.get("fuzzy_key_mapping", {}) or {}
                    fallback_logs.append(attempt_log)
                    break
            fallback_logs.append(attempt_log)

        join_fallback_record[cand_name] = {
            "coverage_threshold": fallback_coverage_threshold,
            "fuzzy_score_threshold": fuzzy_score_threshold,
            "attempts": fallback_logs,
        }
        if not chosen_cols:
            had_similarity_pass = any((a.get("similarity_gate") or {}).get("gate_pass", False) for a in fallback_logs)
            phase_log["excluded"].append(
                {
                    "table_id": cand_name,
                    "decision": "excluded",
                    "reason_code": "JOIN_COVERAGE_LOW" if had_similarity_pass else "JOIN_SIMILARITY_FAIL",
                    "reason": "all join fallback attempts failed health checks",
                    "attempts": fallback_logs,
                }
            )
            continue
        tbl_with_join = dict(tbl)
        df_for_hard_dq = _filter_candidate_df_by_sketch_keys(
            df, chosen_cols, join_columns, join_sketch_original_values,
            chosen_fuzzy_mapping if chosen_join_mode == "fuzzy" else None
        )
        _, dq_report = apply_hard_quality_rules(
            df=df_for_hard_dq,
            protected_columns=chosen_cols,
            missing_threshold=hard_dq_missing_threshold,
            top1_ratio_threshold=hard_dq_top1_ratio_threshold,
        )
        dq_dropped_columns = dq_report.get("dropped_columns", [])
        tbl_with_join["candidate_table"] = cand_name
        tbl_with_join["selected_columns"] = chosen_cols
        tbl_with_join["join_col_descs"] = join_col_descs
        tbl_with_join["cand_col_descs"] = cand_col_descs
        tbl_with_join["use_fuzzy_join"] = chosen_join_mode == "fuzzy"
        tbl_with_join["fuzzy_key_mapping"] = chosen_fuzzy_mapping if chosen_join_mode == "fuzzy" else {}
        tbl_with_join["dq_dropped_columns"] = dq_dropped_columns
        tbl_with_join["dq_report"] = dq_report
        final_selected_tables.append(tbl_with_join)
        candidate_cols_union = list(dict.fromkeys([c for vals in top5_by_join_col.values() for c in vals]))
        rejected_cols = [c for c in candidate_cols_union if c not in chosen_cols]
        phase_log["selected"].append(
            {
                "table_id": cand_name,
                "decision": "selected",
                "reason_code": "JOIN_COLUMNS_SELECTED",
                "selected_columns": chosen_cols,
                "rejected_columns": rejected_cols,
                "join_mode": chosen_join_mode,
                "attempts": fallback_logs,
                "hard_dq_dropped_columns": dq_dropped_columns,
            }
        )

    ctx.pipeline_timings["06b_join_column_per_candidate_loop"] = time.perf_counter() - t_join_phase
    run_record["top5_profile"] = top5_profile
    run_record["join_fallback"] = join_fallback_record
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_sid = re.sub(r"[^\w\-]", "_", str(ctx.session_id or "default").strip()) or "default"
    filename = run_start_time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{safe_sid}.json"
    ctx.run_record_path = data_dir / filename
    with open(ctx.run_record_path, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)
    print(f"[Run Record] Saved to {ctx.run_record_path}")

    ctx.state["run_record"] = run_record
    ctx.state["relevant_list"] = final_selected_tables
    ctx.state["final_selected_tables"] = final_selected_tables
    ctx.state["join_df"] = join_df
    ctx.state["join_df_full"] = join_df_full
    ctx.state["join_columns"] = join_columns
    ctx.state["join_sketch_original_values"] = join_sketch_original_values
    ctx.state["join_col_descs"] = join_col_descs
    if isinstance(decision_log, dict):
        decision_log.setdefault("phases", {})["join_column_selection"] = phase_log
        snap = decision_log.get("threshold_snapshot", {})
        if isinstance(snap, dict):
            snap["join_column"] = phase_log["thresholds"]
            snap["join_hard_dq"] = {
                "missing_threshold": hard_dq_missing_threshold,
                "top1_ratio_threshold": hard_dq_top1_ratio_threshold,
                "topk_candidates": topk_candidates,
            }
            decision_log["threshold_snapshot"] = snap

