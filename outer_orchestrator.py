import asyncio
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google.adk.runners import InMemoryRunner

from Agent.log_analysis_agent import build_log_analysis_agent
from Agent.modification_agent import build_modification_agent
from Pipeline.logging_utils import classify_outcome
from Pipeline.utils import close_runner_safely, extract_json, extract_json_by_key_from_full_text
from agent_config_loader import AgentPipelineConfig
from orchestrator import run_orchestrator


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _count_reason_codes(decision_log: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    phases = (decision_log or {}).get("phases", {}) if isinstance(decision_log, dict) else {}
    for _, phase_payload in phases.items():
        if not isinstance(phase_payload, dict):
            continue
        excluded = phase_payload.get("excluded", []) or phase_payload.get("excluded_tables", []) or []
        if not isinstance(excluded, list):
            continue
        for item in excluded:
            if not isinstance(item, dict):
                continue
            code = str(item.get("reason_code") or "").strip()
            if not code:
                continue
            counts[code] = counts.get(code, 0) + 1
    return counts


def _first_empty_stage(decision_log: Dict[str, Any]) -> str:
    phases = (decision_log or {}).get("phases", {}) if isinstance(decision_log, dict) else {}
    ts = phases.get("table_selection", {}) if isinstance(phases, dict) else {}
    if not (ts.get("selected_tables") or []):
        return "table_selection"
    js = phases.get("join_column_selection", {}) if isinstance(phases, dict) else {}
    if not (js.get("selected") or []):
        return "join_column_selection"
    dq = phases.get("data_quality", {}) if isinstance(phases, dict) else {}
    if len(dq.get("selected", []) or []) == 0:
        return "data_quality"
    aug = phases.get("augment", {}) if isinstance(phases, dict) else {}
    if not (aug.get("selected") or []):
        return "augment"
    return "none"


def _get_current_values(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    join_cfg = (((config_dict or {}).get("pipeline_thresholds", {}) or {}).get("join", {}) or {})
    dq_cfg = (((config_dict or {}).get("pipeline_thresholds", {}) or {}).get("data_quality", {}) or {})
    aug_cfg = (((config_dict or {}).get("pipeline_thresholds", {}) or {}).get("augment", {}) or {})
    data_cfg = (((config_dict or {}).get("data", {}) or {}).get("datalake", {}) or {})
    return {
        "data_max_tables": data_cfg.get("max_tables"),
        "join_fallback_coverage_threshold": join_cfg.get("fallback_coverage_threshold"),
        "join_fuzzy_score_threshold": join_cfg.get("fuzzy_score_threshold"),
        "join_hard_dq_missing_threshold": join_cfg.get("hard_dq_missing_threshold"),
        "join_hard_dq_top1_ratio_threshold": join_cfg.get("hard_dq_top1_ratio_threshold"),
        "dq_coarse_preselect_topk": dq_cfg.get("coarse_preselect_topk"),
        "augment_min_metric_gain_delta": aug_cfg.get("min_metric_gain_delta"),
    }


def _apply_selected_actions(config_dict: Dict[str, Any], selected_actions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    new_cfg = copy.deepcopy(config_dict)
    applied: List[Dict[str, Any]] = []
    for action in selected_actions:
        if not isinstance(action, dict):
            continue
        param = str(action.get("parameter") or "").strip()
        if not param:
            continue
        target = action.get("target_value")
        if param == "data_max_tables":
            new_cfg.setdefault("data", {}).setdefault("datalake", {})["max_tables"] = int(target)
        elif param == "join_fallback_coverage_threshold":
            new_cfg.setdefault("pipeline_thresholds", {}).setdefault("join", {})["fallback_coverage_threshold"] = float(target)
        elif param == "join_fuzzy_score_threshold":
            new_cfg.setdefault("pipeline_thresholds", {}).setdefault("join", {})["fuzzy_score_threshold"] = float(target)
        elif param == "join_hard_dq_missing_threshold":
            new_cfg.setdefault("pipeline_thresholds", {}).setdefault("join", {})["hard_dq_missing_threshold"] = float(target)
        elif param == "join_hard_dq_top1_ratio_threshold":
            new_cfg.setdefault("pipeline_thresholds", {}).setdefault("join", {})["hard_dq_top1_ratio_threshold"] = float(target)
        elif param == "dq_coarse_preselect_topk":
            new_cfg.setdefault("pipeline_thresholds", {}).setdefault("data_quality", {})["coarse_preselect_topk"] = int(target)
        elif param == "augment_min_metric_gain_delta":
            new_cfg.setdefault("pipeline_thresholds", {}).setdefault("augment", {})["min_metric_gain_delta"] = float(target)
        else:
            continue
        applied.append({"parameter": param, "target_value": target, "reason": action.get("reason", "")})
    return new_cfg, applied


def _task_key(join_table_name: Optional[str], task_type: Optional[str]) -> str:
    return f"{join_table_name or 'unknown'}|{task_type or 'unknown'}"


def _is_perturbation_task(config_dict: Dict[str, Any]) -> bool:
    base_dir = str((((config_dict or {}).get("data") or {}).get("base_dir") or ""))
    return base_dir.startswith("perturbed_")


def _history_fingerprint(current_values: Dict[str, Any]) -> str:
    payload = json.dumps(current_values or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _load_history_records(history_path: Path, task_key: str, session_id: Optional[str]) -> List[Dict[str, Any]]:
    if not history_path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("task_key") == task_key and str(obj.get("session_id")) == str(session_id):
                    out.append(obj)
    except Exception:
        return []
    return out


def _compress_history_for_llm(records: List[Dict[str, Any]], recent_k: int = 5, top_k: int = 2, worst_k: int = 2) -> List[Dict[str, Any]]:
    if not records:
        return []
    recent = records[-recent_k:]
    sortable = [r for r in records if isinstance(r, dict) and r.get("improvement") is not None]
    best = sorted(sortable, key=lambda r: float(r.get("improvement", 0.0)), reverse=True)[:top_k]
    worst = sorted(sortable, key=lambda r: float(r.get("improvement", 0.0)))[:worst_k]
    merged: List[Dict[str, Any]] = []
    seen = set()
    for r in recent + best + worst:
        fp = str(r.get("config_fingerprint") or "")
        if fp and fp in seen:
            continue
        if fp:
            seen.add(fp)
        merged.append(
            {
                "round": r.get("round"),
                "outcome": r.get("outcome"),
                "improvement": r.get("improvement"),
                "improvement_pct": r.get("improvement_pct"),
                "config_delta": r.get("config_delta", {}),
                "selected_actions": r.get("selected_actions", []),
                "effective_augment_columns": r.get("effective_augment_columns", {}),
                "selected_inherited_augment_columns": r.get("selected_inherited_augment_columns", {}),
                "first_empty_stage": r.get("first_empty_stage"),
                "config_fingerprint": fp,
            }
        )
    return merged


def _append_history_entry(history_path: Path, entry: Dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_inherited_augment_columns(output: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    inherited: Dict[str, List[Dict[str, str]]] = {}
    augment_results = (output or {}).get("augment_results", [])
    if not isinstance(augment_results, list):
        return inherited
    for item in augment_results:
        if not isinstance(item, dict):
            continue
        cand = str(item.get("candidate_table") or "").strip()
        if not cand:
            continue
        selected_cols = item.get("selected_augment_columns", []) or []
        if not isinstance(selected_cols, list) or not selected_cols:
            continue
        decisions = item.get("column_decisions", []) or []
        dtype_map: Dict[str, str] = {}
        if isinstance(decisions, list):
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                col = str(d.get("column") or "").strip()
                if not col:
                    continue
                dtype_map[col] = str(d.get("dtype_final") or "numerical").strip().lower()
        entries: List[Dict[str, str]] = []
        for col in selected_cols:
            col_name = str(col).strip()
            if not col_name:
                continue
            entries.append(
                {
                    "column": col_name,
                    "dtype_final": dtype_map.get(col_name, "numerical"),
                }
            )
        if entries:
            inherited[cand] = entries
    return inherited


def _normalize_inherited_augment_columns(raw: Any) -> Dict[str, List[Dict[str, str]]]:
    normalized: Dict[str, List[Dict[str, str]]] = {}
    if not isinstance(raw, dict):
        return normalized
    for cand, items in raw.items():
        cand_name = str(cand or "").strip()
        if not cand_name or not isinstance(items, list):
            continue
        seen_cols = set()
        entries: List[Dict[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                col = str(item.get("column") or "").strip()
                dtype_final = str(item.get("dtype_final") or "numerical").strip().lower()
            else:
                col = str(item or "").strip()
                dtype_final = "numerical"
            if not col or col in seen_cols:
                continue
            seen_cols.add(col)
            entries.append({"column": col, "dtype_final": dtype_final or "numerical"})
        if entries:
            normalized[cand_name] = entries
    return normalized


async def _run_log_analysis_once(
    *,
    decision_log: Dict[str, Any],
    output: Dict[str, Any],
    config: AgentPipelineConfig,
) -> Dict[str, Any]:
    baseline = output.get("baseline_metric")
    augmented = output.get("augmented_metric")
    outcome = classify_outcome(baseline, augmented)
    payload = {
        "task_id": decision_log.get("task_id"),
        "session_id": decision_log.get("session_id"),
        "join_table_name": decision_log.get("join_table_name"),
        "task_type": decision_log.get("task_type"),
        "target_column": decision_log.get("target_column"),
        "outcome": outcome,
        "baseline_metric": baseline,
        "augmented_metric": augmented,
        "phases": decision_log.get("phases", {}),
    }
    prompt = (
        "Analyze pipeline decision log and suggest minimal next-step actions. "
        "Return strict JSON as requested.\n"
        f"input: {json.dumps(payload, ensure_ascii=False)}"
    )
    runner = InMemoryRunner(agent=build_log_analysis_agent(config=config))
    try:
        events = await runner.run_debug(prompt, quiet=True)
        full_text = ""
        for event in events:
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for part in event.content.parts:
                    t = getattr(part, "text", None)
                    if t:
                        full_text += t
        parsed = extract_json_by_key_from_full_text(full_text, "suggestions", prefer_non_empty_list=True)
        if not parsed:
            parsed = extract_json(full_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await close_runner_safely(runner)


async def _run_modification_once(
    *,
    config: AgentPipelineConfig,
    first_empty_stage: str,
    reason_code_counts: Dict[str, int],
    log_analysis_report: Dict[str, Any],
    config_history_for_llm: List[Dict[str, Any]],
    previous_round_effective_augment_columns: Dict[str, List[Dict[str, str]]],
    previous_round_outcome: str,
    previous_round_improvement: Optional[float],
    max_actions_per_round: int,
) -> Dict[str, Any]:
    cfg_dict = config.config if isinstance(config.config, dict) else {}
    adaptation_cfg = (cfg_dict.get("adaptation") or {})
    discrete_options = adaptation_cfg.get("discrete_options", {}) if isinstance(adaptation_cfg, dict) else {}
    payload = {
        "first_empty_stage": first_empty_stage,
        "reason_code_counts": reason_code_counts,
        "log_analysis": log_analysis_report,
        "config_history": config_history_for_llm,
        "previous_round_effective_augment_columns": previous_round_effective_augment_columns,
        "previous_round_outcome": previous_round_outcome,
        "previous_round_improvement": previous_round_improvement,
        "current_values": _get_current_values(cfg_dict),
        "discrete_options": discrete_options,
        "max_actions_per_round": max_actions_per_round,
    }
    prompt = (
        "Given hard signals and history, propose config updates for next round. "
        "Return strict JSON.\n"
        f"input: {json.dumps(payload, ensure_ascii=False)}"
    )
    runner = InMemoryRunner(agent=build_modification_agent(config=config))
    try:
        events = await runner.run_debug(prompt, quiet=True)
        full_text = ""
        for event in events:
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for part in event.content.parts:
                    t = getattr(part, "text", None)
                    if t:
                        full_text += t
        parsed = extract_json_by_key_from_full_text(full_text, "selected_actions", prefer_non_empty_list=False)
        if not parsed:
            parsed = extract_json(full_text)
        if not isinstance(parsed, dict):
            return {"selected_actions": [], "overall_reasoning": "parse_failed"}
        selected = parsed.get("selected_actions", [])
        if not isinstance(selected, list):
            selected = []
        parsed["selected_actions"] = selected[:max_actions_per_round]
        parsed["selected_inherited_augment_columns"] = _normalize_inherited_augment_columns(
            parsed.get("selected_inherited_augment_columns", {})
        )
        return parsed
    except Exception as e:
        return {"selected_actions": [], "overall_reasoning": f"modification_agent_failed: {e}"}
    finally:
        await close_runner_safely(runner)


async def run_outer_orchestrator(
    *,
    config: AgentPipelineConfig,
    user_intent: Optional[str] = None,
    session_id: Optional[str] = None,
    join_table_name: Optional[str] = None,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    max_rounds: Optional[int] = None,
    improvement_threshold_pct: Optional[float] = None,
    config_history_file: Optional[str] = None,
) -> Dict[str, Any]:
    base_cfg = config.config if isinstance(config.config, dict) else {}
    adaptation_cfg = base_cfg.get("adaptation", {}) if isinstance(base_cfg.get("adaptation"), dict) else {}
    max_rounds = int(max_rounds if max_rounds is not None else adaptation_cfg.get("max_rounds", 3))
    improvement_threshold_pct = float(
        improvement_threshold_pct if improvement_threshold_pct is not None else adaptation_cfg.get("improvement_threshold_pct", 50.0)
    )
    no_gain_abort_streak = int(adaptation_cfg.get("no_gain_abort_streak", 2))
    max_actions_per_round = int(adaptation_cfg.get("max_actions_per_round", 2))

    history_file = config_history_file
    if history_file is None:
        history_file = ((base_cfg.get("output") or {}).get("config_history_file")) if isinstance(base_cfg.get("output"), dict) else None
    history_path = None
    if history_file:
        history_path = Path(history_file)
        if not history_path.is_absolute():
            history_path = (Path(__file__).resolve().parent / history_path).resolve()

    current_config = AgentPipelineConfig(config_dict=copy.deepcopy(base_cfg))
    rounds: List[Dict[str, Any]] = []
    no_gain_streak = 0
    abort_reason: Optional[str] = None
    round1_reuse_context: Optional[Dict[str, Any]] = None
    inherited_augment_columns: Dict[str, List[Dict[str, str]]] = {}
    previous_round_effective_columns: Dict[str, List[Dict[str, str]]] = {}
    anchor_baseline: Optional[float] = None
    prev_augmented: Optional[float] = None

    for round_idx in range(1, max_rounds + 1):
        reuse_context = None
        if round_idx > 1 and isinstance(round1_reuse_context, dict):
            reuse_context = copy.deepcopy(round1_reuse_context)
            if inherited_augment_columns:
                reuse_context["inherited_augment_columns"] = inherited_augment_columns
        output = await run_orchestrator(
            config=current_config,
            user_intent=user_intent,
            session_id=session_id,
            join_table_name=join_table_name,
            target_column=target_column,
            task_type=task_type,
            reuse_context=reuse_context,
        )
        if round1_reuse_context is None and isinstance(output.get("_reuse_context"), dict):
            round1_reuse_context = output.get("_reuse_context")
        baseline = output.get("baseline_metric")
        augmented = output.get("augmented_metric")
        improvement = None
        improvement_pct = None
        if baseline is not None and augmented is not None:
            improvement = augmented - baseline
            improvement_pct = (improvement / abs(baseline) * 100) if baseline != 0 else 0.0
        if anchor_baseline is None and baseline is not None:
            anchor_baseline = baseline
        cumulative_improvement = None
        cumulative_improvement_pct = None
        if anchor_baseline is not None and augmented is not None:
            cumulative_improvement = augmented - anchor_baseline
            cumulative_improvement_pct = (cumulative_improvement / abs(anchor_baseline) * 100) if anchor_baseline != 0 else 0.0
        marginal_from_prev_augmented = None
        if prev_augmented is not None and augmented is not None:
            marginal_from_prev_augmented = augmented - prev_augmented
        if augmented is not None:
            prev_augmented = augmented
        outcome = classify_outcome(baseline, augmented)

        decision_log = _load_json(output.get("decision_log_path"))
        first_empty_stage = _first_empty_stage(decision_log)
        reason_code_counts = _count_reason_codes(decision_log)
        round_record: Dict[str, Any] = {
            "round": round_idx,
            "baseline_metric": baseline,
            "augmented_metric": augmented,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "anchor_baseline_metric": anchor_baseline,
            "cumulative_improvement": cumulative_improvement,
            "cumulative_improvement_pct": cumulative_improvement_pct,
            "marginal_from_prev_augmented": marginal_from_prev_augmented,
            "outcome": outcome,
            "decision_log_path": output.get("decision_log_path"),
            "first_empty_stage": first_empty_stage,
            "reason_code_counts": reason_code_counts,
        }
        inherited_count = sum(len(v) for v in inherited_augment_columns.values())
        round_record["inherited_augment_columns_count"] = inherited_count
        previous_round_effective_columns = _extract_inherited_augment_columns(output)
        round_record["previous_round_effective_augment_columns_count"] = sum(
            len(v) for v in previous_round_effective_columns.values()
        )
        task_key = _task_key(
            decision_log.get("join_table_name") or join_table_name,
            decision_log.get("task_type") or task_type,
        )
        perturbation_task = _is_perturbation_task(current_config.config)

        selected_inherited_for_history: Dict[str, List[Dict[str, str]]] = {}
        applied_actions: List[Dict[str, Any]] = []
        config_delta: Dict[str, Any] = {}
        current_values_after = _get_current_values(current_config.config)

        def _try_append_round_history() -> None:
            if history_path is None:
                return
            entry = {
                "session_id": session_id,
                "task_key": task_key,
                "round": round_idx,
                "outcome": outcome,
                "improvement": improvement,
                "improvement_pct": improvement_pct,
                "cumulative_improvement": cumulative_improvement,
                "cumulative_improvement_pct": cumulative_improvement_pct,
                "first_empty_stage": first_empty_stage,
                "selected_actions": applied_actions,
                "config_delta": config_delta,
                "effective_augment_columns": previous_round_effective_columns,
                "selected_inherited_augment_columns": selected_inherited_for_history,
                "config_fingerprint": _history_fingerprint(current_values_after),
                "is_perturbation_task": perturbation_task,
            }
            try:
                _append_history_entry(history_path, entry)
            except Exception:
                pass

        eval_improvement_pct = (
            cumulative_improvement_pct
            if cumulative_improvement_pct is not None
            else improvement_pct
        )
        reached_target = eval_improvement_pct is not None and eval_improvement_pct >= improvement_threshold_pct
        if reached_target:
            round_record["stop_reason"] = f"cumulative_improvement_pct>={improvement_threshold_pct}"
            rounds.append(round_record)
            _try_append_round_history()
            break

        if outcome in ("no_gain", "regression"):
            no_gain_streak += 1
        else:
            no_gain_streak = 0

        if no_gain_streak >= no_gain_abort_streak:
            abort_reason = "two_consecutive_no_gain_or_regression"
            round_record["stop_reason"] = abort_reason
            rounds.append(round_record)
            _try_append_round_history()
            break

        if round_idx >= max_rounds:
            round_record["stop_reason"] = "max_rounds_reached"
            rounds.append(round_record)
            _try_append_round_history()
            break

        analysis_report = await _run_log_analysis_once(
            decision_log=decision_log,
            output=output,
            config=current_config,
        )
        round_record["log_analysis"] = analysis_report
        history_for_llm: List[Dict[str, Any]] = []
        if not perturbation_task and history_path is not None:
            history_records = _load_history_records(history_path, task_key=task_key, session_id=session_id)
            history_for_llm = _compress_history_for_llm(history_records)
        round_record["history_used_count"] = len(history_for_llm)

        modification_plan = await _run_modification_once(
            config=current_config,
            first_empty_stage=first_empty_stage,
            reason_code_counts=reason_code_counts,
            log_analysis_report=analysis_report,
            config_history_for_llm=history_for_llm,
            previous_round_effective_augment_columns=previous_round_effective_columns,
            previous_round_outcome=outcome,
            previous_round_improvement=improvement,
            max_actions_per_round=max_actions_per_round,
        )
        round_record["modification"] = modification_plan

        inherited_augment_columns = _normalize_inherited_augment_columns(
            (modification_plan or {}).get("selected_inherited_augment_columns", {})
        )
        selected_inherited_for_history = inherited_augment_columns
        round_record["next_round_selected_inherited_augment_columns_count"] = sum(
            len(v) for v in inherited_augment_columns.values()
        )

        selected_actions = modification_plan.get("selected_actions", []) if isinstance(modification_plan, dict) else []
        current_values_before = _get_current_values(current_config.config)
        updated_cfg_dict, applied_actions = _apply_selected_actions(current_config.config, selected_actions)
        current_values_after = _get_current_values(updated_cfg_dict)
        config_delta = {}
        for k, old_v in current_values_before.items():
            new_v = current_values_after.get(k)
            if new_v != old_v:
                config_delta[k] = [old_v, new_v]
        round_record["applied_actions"] = applied_actions
        round_record["config_delta"] = config_delta
        current_config = AgentPipelineConfig(config_dict=updated_cfg_dict)
        rounds.append(round_record)
        _try_append_round_history()

    return {
        "rounds": rounds,
        "abort_reason": abort_reason,
        "max_rounds": max_rounds,
        "improvement_threshold_pct": improvement_threshold_pct,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run outer adaptation loop orchestrator")
    parser.add_argument("--config", type=str, default="configs/agent_pipeline_config.yaml")
    parser.add_argument("--user-intent", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--join-table", type=str, default=None)
    parser.add_argument("--target-column", type=str, default=None)
    parser.add_argument("--task-type", type=str, default=None, choices=["regression", "classification"])
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--improvement-threshold-pct", type=float, default=None)
    parser.add_argument("--config-history-file", type=str, default=None)
    args = parser.parse_args()

    cfg = AgentPipelineConfig(args.config)
    result = asyncio.run(
        run_outer_orchestrator(
            config=cfg,
            user_intent=args.user_intent,
            session_id=args.session_id,
            join_table_name=args.join_table,
            target_column=args.target_column,
            task_type=args.task_type,
            max_rounds=args.max_rounds,
            improvement_threshold_pct=args.improvement_threshold_pct,
            config_history_file=args.config_history_file,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

