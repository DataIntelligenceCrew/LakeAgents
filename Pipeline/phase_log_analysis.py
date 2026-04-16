import json
import time
from typing import Any, Dict, List

from Pipeline.context import PipelineContext
from Pipeline.logging_utils import classify_outcome
from Pipeline.utils import extract_json, extract_json_by_key_from_full_text


def _build_fallback_report(decision_log: Dict[str, Any], outcome: str) -> Dict[str, Any]:
    reason_counts: Dict[str, int] = {}
    phase_counts: Dict[str, int] = {}
    phases = (decision_log or {}).get("phases", {}) if isinstance(decision_log, dict) else {}
    for phase_name, phase_payload in phases.items():
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
            reason_counts[code] = reason_counts.get(code, 0) + 1
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
    top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:3]
    primary_phase = "none"
    if phase_counts:
        primary_phase = sorted(phase_counts.items(), key=lambda x: -x[1])[0][0]
    return {
        "primary_failure_phase": primary_phase,
        "top_reasons": [
            {"reason_code": code, "count": cnt, "evidence": "counted from excluded decisions"}
            for code, cnt in top_reasons
        ],
        "suggestions": [
            {
                "action": "review_top_failure_phase",
                "target": primary_phase if primary_phase != "none" else "pipeline",
                "rationale": f"outcome={outcome}, inspect dominant reason codes first",
                "priority": "medium",
            }
        ],
        "confidence": "low",
    }


async def run_log_analysis(ctx: PipelineContext) -> None:
    log_analysis_runner = ctx.state["log_analysis_runner"]
    decision_log = ctx.state.get("decision_log", {}) or {}
    output = ctx.state.get("output", {}) or {}
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

    t0 = time.perf_counter()
    report: Dict[str, Any] = {}
    try:
        events = await log_analysis_runner.run_debug(prompt, quiet=True)
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
        report = parsed if isinstance(parsed, dict) else {}
    except Exception:
        report = {}

    if not report or "suggestions" not in report:
        report = _build_fallback_report(decision_log, outcome)
    if "primary_failure_phase" not in report:
        report["primary_failure_phase"] = "none"
    if "top_reasons" not in report or not isinstance(report.get("top_reasons"), list):
        report["top_reasons"] = []
    if "suggestions" not in report or not isinstance(report.get("suggestions"), list):
        report["suggestions"] = []
    if "confidence" not in report:
        report["confidence"] = "low"

    ctx.pipeline_timings["16_log_analysis_agent"] = time.perf_counter() - t0
    ctx.state["log_analysis_report"] = report
    if isinstance(decision_log, dict):
        decision_log["log_analysis"] = report
    if isinstance(output, dict):
        output["log_analysis"] = report
        ctx.state["output"] = output

