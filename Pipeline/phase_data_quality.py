import json
import os
import time
import uuid
from typing import Any, Dict, List, Sequence

import pandas as pd
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner

from Agent.augment_column_selection_agent import build_augment_column_selection_agent
from Agent.data_quality_agent import build_data_quality_agent
from tools.column_descriptions import get_column_datatypes_from_index
from tools.data_quality import apply_quality_actions, compute_table_quality_summary
from tools.sketch import get_candidate_table

from Pipeline.context import PipelineContext
from Pipeline.phase_join_columns import _filter_candidate_df_by_sketch_keys
from Pipeline.utils import (
    close_runner_safely,
    extract_json_by_key_from_full_text,
    first_non_empty_rows,
    first_non_empty_values,
    normalize_augment_column_list,
)


def _dq_max_llm_calls(dq_cfg: Dict[str, Any]) -> int:
    """Cap ADK LLM calls per DQ/coarse runner (ADK default is 500)."""
    raw = os.environ.get("DQ_MAX_LLM_CALLS")
    if raw is None or str(raw).strip() == "":
        raw = dq_cfg.get("max_llm_calls", 40)
    try:
        return max(1, int(raw))
    except Exception:
        return 40


def _infer_feature_type_for_coarse(series: pd.Series, metadata_datatype: str = "") -> str:
    md = (metadata_datatype or "").strip().lower()
    if md in ("number", "checkbox"):
        return "numerical"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    non_null = series.dropna()
    if non_null.empty:
        return "text"
    # Count-like object columns (e.g. "0","1","2") are numerical for regression/classification.
    sample = non_null.astype(str).head(30)
    numeric_like = 0
    for v in sample.tolist():
        try:
            float(str(v).replace(",", "").strip())
            numeric_like += 1
        except Exception:
            pass
    if len(sample) > 0 and (numeric_like / len(sample)) >= 0.8:
        return "numerical"
    unique_non_null = non_null.astype(str).nunique()
    unique_ratio = unique_non_null / max(len(non_null), 1)
    if unique_non_null <= 20 or unique_ratio <= 0.05:
        return "categorical"
    return "text"


def _coarse_is_protected_numerical(col_meta: Dict[str, Any]) -> bool:
    """Numerical / number-typed columns must not be dropped at coarse stage."""
    hint = str(col_meta.get("feature_type_hint") or "").strip().lower()
    md = str(col_meta.get("metadata_datatype") or "").strip().lower()
    return hint == "numerical" or md in ("number", "checkbox")





def _chunked(items: Sequence[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        size = 15
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _coarse_batch_size() -> int:
    try:
        return max(1, int(os.environ.get("COARSE_FEATURE_BATCH_SIZE", "15")))
    except Exception:
        return 15


async def run_data_quality(ctx: PipelineContext) -> None:
    config = ctx.config
    dq_cfg = ((config.config or {}).get("pipeline_thresholds", {}) or {}).get("data_quality", {})
    coarse_preselect_topk = int(dq_cfg.get("coarse_preselect_topk", 20))
    coarse_input_max_cols = int(dq_cfg.get("coarse_input_max_cols", 80))
    dq_max_llm_calls = _dq_max_llm_calls(dq_cfg if isinstance(dq_cfg, dict) else {})
    dq_run_config = RunConfig(max_llm_calls=dq_max_llm_calls)
    final_selected_tables = ctx.state.get("final_selected_tables", []) or []
    domain_for_fetch = ctx.state.get("domain_for_fetch")
    join_columns = ctx.state.get("join_columns", [])
    join_sketch_original_values = ctx.state.get("join_sketch_original_values")
    query_table_description = ctx.state.get("query_table_description", "")

    print(
        f"\n🧹 Running Data Quality Agent (compact tool-round context, "
        f"max_llm_calls={dq_max_llm_calls}/table)..."
    )
    decision_log = ctx.state.get("decision_log", {})
    phase_log: Dict[str, Any] = {
        "selected": [],
        "excluded": [],
        "coarse_selected": [],
        "coarse_excluded": [],
    }
    quality_enhanced_tables: Dict[str, pd.DataFrame] = {}
    dq_phase_record: Dict[str, Any] = {}
    t_dq = time.perf_counter()
    for tbl in final_selected_tables:
        cand_name = tbl.get("candidate_table")
        selected_cols = tbl.get("selected_columns", []) or []
        if not cand_name:
            continue
        cand_df, status = get_candidate_table(cand_name, domain_for_fetch)
        if not status.get("success"):
            continue
        dq_dropped_columns = tbl.get("dq_dropped_columns", []) or []
        cand_df_input = cand_df.drop(columns=dq_dropped_columns, errors="ignore")
        cand_df_input = _filter_candidate_df_by_sketch_keys(
            cand_df_input,
            selected_cols,
            join_columns,
            join_sketch_original_values,
            tbl.get("fuzzy_key_mapping", {}) if tbl.get("use_fuzzy_join") else None,
        )
        candidate_descs = tbl.get("cand_col_descs", {}) or {}
        non_join_columns = [c for c in cand_df_input.columns if c not in selected_cols]
        quality_summary = compute_table_quality_summary(
            cand_df_input, column_descriptions=candidate_descs, exclude_columns=selected_cols
        )
        row_examples = first_non_empty_rows(cand_df_input, non_join_columns, n=5)
        dq_payload = {
            "candidate_table": cand_name,
            "query_table_name": ctx.query_table_display_name,
            "query_join_columns": join_columns,
            "selected_join_columns": selected_cols,
            "row_examples": row_examples,
            "metadata": [
                {
                    "column_name": item.get("column_name"),
                    "column_description": item.get("column_description", ""),
                    "dtype": item.get("dtype", ""),
                }
                for item in quality_summary
            ],
            "quality_stats": quality_summary,
            "available_tools": ["winsorize(lower_q=0.05, upper_q=0.95)", "median", "mode", "bayesian_ridge", "random_forest"],
        }
        dq_prompt = (
            "Assess column-level data quality and decide whether each column should be dropped or improved. "
            "Use tools when needed and return JSON with key column_actions.\n"
            f"input: {json.dumps(dq_payload, ensure_ascii=False)}"
        )
        column_actions: List[Dict[str, Any]] = []
        dq_reasoning = ""
        data_quality_runner = None
        try:
            # Use one fresh runner per candidate table to avoid cross-table context carryover.
            data_quality_runner = InMemoryRunner(agent=build_data_quality_agent(config=config))
            events = await data_quality_runner.run_debug(
                dq_prompt,
                quiet=True,
                session_id=f"dq_{uuid.uuid4().hex[:12]}",
                run_config=dq_run_config,
            )
            full_text = ""
            for event in events:
                if getattr(event, "content", None) and getattr(event.content, "parts", None):
                    for part in event.content.parts:
                        t = getattr(part, "text", None)
                        if t:
                            full_text += t
            parsed = extract_json_by_key_from_full_text(full_text, "column_actions", prefer_non_empty_list=True)
            raw_actions = parsed.get("column_actions", []) or []
            if isinstance(raw_actions, list):
                column_actions = [a for a in raw_actions if isinstance(a, dict) and a.get("column")]
            dq_reasoning = str(parsed.get("reasoning", "") or "")
        except Exception as e:
            err_name = type(e).__name__
            if "LlmCallsLimit" in err_name or "LlmCallsLimit" in str(e):
                print(
                    f"[DataQuality] {cand_name}: hit max_llm_calls={dq_max_llm_calls}; "
                    "continuing with empty column_actions / hard rules only"
                )
            dq_reasoning = f"dq agent failed: {e}"
        finally:
            await close_runner_safely(data_quality_runner)
        quality_df, quality_report = apply_quality_actions(cand_df_input, actions=column_actions, protected_columns=selected_cols)
        quality_enhanced_tables[cand_name] = quality_df
        chosen_quality_cols: List[str] = []
        for action in column_actions:
            col = action.get("column")
            if not col:
                continue
            a = str(action.get("action", "keep")).lower()
            m = str(action.get("method", "none")).lower()
            if a == "keep" and m != "none":
                chosen_quality_cols.append(col)
        tbl["dq_agent_column_actions"] = column_actions
        tbl["dq_agent_reasoning"] = dq_reasoning
        tbl["dq_selected_quality_columns"] = sorted(set(chosen_quality_cols))
        tbl["dq_apply_report"] = quality_report
        tbl["quality_enhanced_columns"] = [c for c in quality_df.columns if c not in selected_cols]
        dq_phase_record[cand_name] = {
            "selected_join_columns": selected_cols,
            "column_actions": column_actions,
            "reasoning": dq_reasoning,
            "apply_report": quality_report,
        }
        action_by_col = {str(a.get("column")): a for a in column_actions if isinstance(a, dict) and a.get("column")}
        for col in chosen_quality_cols:
            phase_log["selected"].append(
                {
                    "table_id": cand_name,
                    "column": col,
                    "decision": "selected",
                    "reason_code": "DQ_KEEP_WITH_TRANSFORM",
                    "reason": f"action={action_by_col.get(col, {}).get('action', 'keep')}, method={action_by_col.get(col, {}).get('method', 'none')}",
                }
            )
        for col in tbl.get("dq_dropped_columns", []) or []:
            phase_log["excluded"].append(
                {
                    "table_id": cand_name,
                    "column": col,
                    "decision": "excluded",
                    "reason_code": "DQ_HIGH_DOMINANCE_OR_MISSING",
                    "reason": "dropped by hard quality rules before DQ agent",
                }
            )
        for action in column_actions:
            col = action.get("column")
            if not col:
                continue
            a = str(action.get("action", "keep")).lower()
            m = str(action.get("method", "none")).lower()
            if a == "drop":
                phase_log["excluded"].append(
                    {
                        "table_id": cand_name,
                        "column": col,
                        "decision": "excluded",
                        "reason_code": "DQ_AGENT_DROP",
                        "reason": f"action=drop, method={m}",
                    }
                )
    ctx.pipeline_timings["07_data_quality_agent"] = time.perf_counter() - t_dq
    ctx.state["quality_enhanced_tables"] = quality_enhanced_tables
    run_record = ctx.state.get("run_record", {})
    run_record["data_quality"] = dq_phase_record

    print("\n🧭 Running Coarse Feature Screening (metadata/examples only)...")
    coarse_phase_record: Dict[str, Any] = {}
    t_coarse = time.perf_counter()
    for tbl in final_selected_tables:
        cand_name = tbl.get("candidate_table")
        selected_cols = tbl.get("selected_columns", []) or []
        if not cand_name:
            continue
        cand_df_coarse = quality_enhanced_tables.get(cand_name)
        if cand_df_coarse is None or cand_df_coarse.empty:
            cand_df_tmp, status = get_candidate_table(cand_name, domain_for_fetch)
            if not status.get("success"):
                continue
            cand_df_coarse = cand_df_tmp
        candidate_descs = tbl.get("cand_col_descs", {}) or {}
        candidate_datatypes = get_column_datatypes_from_index(cand_name)
        candidate_cols = [c for c in cand_df_coarse.columns if c not in selected_cols]
        if not candidate_cols:
            tbl["coarse_selected_columns"] = []
            coarse_phase_record[cand_name] = {"coarse_selected_columns": [], "reasoning": "No candidate columns."}
            continue
        candidate_columns = []
        for col in candidate_cols[:coarse_input_max_cols]:
            s = cand_df_coarse[col]
            non_null = int(s.notna().sum())
            unique_non_null = int(s.dropna().astype(str).nunique()) if non_null > 0 else 0
            desc = str(candidate_descs.get(col, "") or "")
            if len(desc) > 80:
                desc = desc[:80] + "..."
            examples = [str(v)[:40] for v in first_non_empty_values(s, n=3)]
            candidate_columns.append(
                {
                    "feature": col,
                    "description": desc,
                    "metadata_datatype": candidate_datatypes.get(col, "unknown"),
                    "feature_type_hint": _infer_feature_type_for_coarse(s, candidate_datatypes.get(col, "unknown")),
                    "non_null_ratio": round(float(non_null / max(len(s), 1)), 4),
                    "unique_ratio": round(float(unique_non_null / max(non_null, 1)) if non_null > 0 else 0.0, 4),
                    "example_values": examples,
                }
            )
        q_desc = str(query_table_description or "")
        if len(q_desc) > 600:
            q_desc = q_desc[:600] + "...[truncated]"
        coarse_selected: List[str] = []
        coarse_reason_parts: List[str] = []
        seen_sel: set = set()
        for bi, batch in enumerate(_chunked(candidate_columns, _coarse_batch_size())):
            coarse_prompt = (
                f'task_mode: "coarse_screen"\n'
                f'task_type: "{ctx.task_type}"\n'
                f'target_column: "{ctx.target_column}"\n'
                f'user_intent: "{ctx.user_intent}"\n'
                f'candidate_table: "{cand_name}"\n'
                f'query_table_description: "{q_desc}"\n'
                f"max_candidates: {coarse_preselect_topk}\n"
                f"batch_index: {bi}\n"
                f"batch_total: {len(_chunked(candidate_columns, _coarse_batch_size()))}\n"
                f"candidate_columns: {json.dumps(batch, ensure_ascii=False, separators=(',', ':'))}\n\n"
                "Prune THIS BATCH by deleting unnecessary columns (you decide which). "
                "Return JSON with dropped_columns, coarse_selected_columns (kept remainder = batch minus dropped), and reasoning.\n"
                "Execution keeps (batch - dropped_columns); do not omit kept columns from the remainder.\n"
                "HARD RULE for coarse_screen: NEVER drop a column whose feature_type_hint is 'numerical' "
                "or metadata_datatype is 'number'/'checkbox'. Demographic count breakdowns "
                "(age/gender/race totals) that are numerical MUST be kept; do not call them categorical.\n"
            )
            try:
                print(
                    f"[CoarseScreen] {cand_name} batch {bi + 1} "
                    f"n={len(batch)} prompt_chars={len(coarse_prompt)} (fresh_runner)"
                )
                coarse_runner = InMemoryRunner(agent=build_augment_column_selection_agent(config))
                try:
                    events = await coarse_runner.run_debug(
                        coarse_prompt,
                        quiet=True,
                        session_id=f"coarse_{uuid.uuid4().hex[:12]}",
                        run_config=dq_run_config,
                    )
                finally:
                    await close_runner_safely(coarse_runner)
                full_text = ""
                for event in events:
                    if getattr(event, "content", None) and getattr(event.content, "parts", None):
                        for part in event.content.parts:
                            t = getattr(part, "text", None)
                            if t:
                                full_text += t
                batch_names = [str(x.get("feature")) for x in batch if x.get("feature")]
                batch_name_set = set(batch_names)
                batch_meta = {
                    str(x.get("feature")): x
                    for x in batch
                    if isinstance(x, dict) and x.get("feature")
                }
                protected_num = {
                    name
                    for name, meta in batch_meta.items()
                    if _coarse_is_protected_numerical(meta)
                }
                # Deletion-first: keep = batch - dropped (empty dropped => keep all in batch).
                parsed_drop = extract_json_by_key_from_full_text(
                    full_text, "dropped_columns", prefer_non_empty_list=False
                )
                if "dropped_columns" in parsed_drop:
                    dropped = set(normalize_augment_column_list(parsed_drop.get("dropped_columns", [])))
                    dropped &= batch_name_set
                    # Guardrail: never drop numerical/number columns at coarse stage.
                    blocked = dropped & protected_num
                    if blocked:
                        print(
                            f"[CoarseScreen] {cand_name} batch {bi + 1}: "
                            f"restore {len(blocked)} numerical drops (LLM override blocked)"
                        )
                        dropped -= blocked
                    batch_sel = [c for c in batch_names if c not in dropped]
                    parsed = parsed_drop
                else:
                    # Fallback if model omitted dropped_columns.
                    parsed = extract_json_by_key_from_full_text(
                        full_text, "coarse_selected_columns", prefer_non_empty_list=True
                    )
                    batch_sel = [
                        c
                        for c in normalize_augment_column_list(parsed.get("coarse_selected_columns", []))
                        if c in batch_name_set
                    ]
                    # Ensure protected numerical columns are present even if whitelist omitted them.
                    for name in batch_names:
                        if name in protected_num and name not in batch_sel:
                            batch_sel.append(name)
                    if not batch_sel:
                        batch_sel = list(batch_names)
                for c in batch_sel:
                    if c not in seen_sel:
                        seen_sel.add(c)
                        coarse_selected.append(c)
                reason = str(parsed.get("reasoning", "") or "")
                if reason:
                    coarse_reason_parts.append(reason)
            except Exception as e:
                coarse_reason_parts.append(f"batch {bi} failed: {e}")
        coarse_reasoning = " | ".join(coarse_reason_parts)
        if not coarse_selected:
            # Only when every batch failed/returned nothing: soft quality fallback + soft topk.
            fallback_sorted = sorted(candidate_columns, key=lambda x: (x.get("non_null_ratio", 0.0), x.get("unique_ratio", 0.0)), reverse=True)
            coarse_selected = [x["feature"] for x in fallback_sorted[:coarse_preselect_topk]]
        coarse_selected = [
            c for c in coarse_selected if c in cand_df_coarse.columns and c not in selected_cols
        ]
        # Soft topk: only truncate when far above the soft bound (deletion-first default keeps remainder).
        # Prefer keeping numerical columns if truncation is required.
        soft_cap = max(int(coarse_preselect_topk) * 3, int(coarse_preselect_topk))
        if len(coarse_selected) > soft_cap:
            meta_by_name = {str(x.get("feature")): x for x in candidate_columns if x.get("feature")}
            num_first = [c for c in coarse_selected if _coarse_is_protected_numerical(meta_by_name.get(c, {}))]
            other = [c for c in coarse_selected if c not in set(num_first)]
            ordered = num_first + other
            print(
                f"[CoarseScreen] {cand_name} kept={len(coarse_selected)} > soft_cap={soft_cap}; "
                f"truncating with numerical-first (n_num={len(num_first)})"
            )
            coarse_selected = ordered[:soft_cap]
        tbl["coarse_selected_columns"] = coarse_selected
        tbl["coarse_screen_reasoning"] = coarse_reasoning
        keep_cols = list(dict.fromkeys(selected_cols + coarse_selected))
        keep_cols = [c for c in keep_cols if c in cand_df_coarse.columns]
        quality_enhanced_tables[cand_name] = cand_df_coarse[keep_cols].copy()
        coarse_phase_record[cand_name] = {
            "coarse_selected_columns": coarse_selected,
            "reasoning": coarse_reasoning,
            "input_columns": len(candidate_columns),
            "mode": "deletion_first_keep_remainder",
        }
        excluded_coarse = [x["feature"] for x in candidate_columns if x.get("feature") not in coarse_selected]
        for col in coarse_selected:
            phase_log["coarse_selected"].append(
                {
                    "table_id": cand_name,
                    "column": col,
                    "decision": "selected",
                    "reason_code": "COARSE_SELECTED",
                    "reason": "selected in coarse screening",
                }
            )
        for col in excluded_coarse:
            phase_log["coarse_excluded"].append(
                {
                    "table_id": cand_name,
                    "column": col,
                    "decision": "excluded",
                    "reason_code": "COARSE_NOT_SELECTED",
                    "reason": "not selected in coarse screening",
                }
            )
    ctx.pipeline_timings["08_coarse_feature_screening"] = time.perf_counter() - t_coarse
    run_record["coarse_screen"] = coarse_phase_record
    ctx.state["run_record"] = run_record
    if ctx.run_record_path is not None:
        try:
            with open(ctx.run_record_path, "w", encoding="utf-8") as f:
                json.dump(run_record, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    if isinstance(decision_log, dict):
        decision_log.setdefault("phases", {})["data_quality"] = phase_log
        snap = decision_log.get("threshold_snapshot", {})
        if isinstance(snap, dict):
            snap["data_quality"] = {
                "coarse_preselect_topk": coarse_preselect_topk,
                "coarse_input_max_cols": coarse_input_max_cols,
            }
            decision_log["threshold_snapshot"] = snap

