"""Minimal collaborative orchestrator for table selection."""
import json
from typing import Any, Dict, List

from google.adk.runners import InMemoryRunner

from Agent.table_selection_recall_agent import build_table_selection_recall_agent
from Agent.table_selection_precision_agent import build_table_selection_precision_agent
from Agent.table_selection_decider_agent import build_table_selection_decider_agent


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from model text."""
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _collect_text(events: List[Any]) -> str:
    """Collect plain text from ADK events."""
    buf: List[str] = []
    for event in events:
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                t = getattr(part, "text", None)
                if t:
                    buf.append(t)
    return "".join(buf)


def _normalize_table_ids(items: Any) -> List[str]:
    """Normalize table identifiers from mixed representations."""
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
            continue
        if isinstance(item, dict):
            table_id = item.get("table_id") or item.get("id") or item.get("table_name")
            if table_id:
                out.append(str(table_id).strip())
    return out


async def run_table_selection_collab_orchestrator(
    config: object,
    candidate_ids: List[str],
    user_intent: str,
    task_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a minimal 3-role collaboration:
    - Recall agent
    - Precision agent 
    - Decider agent
    """
    recall_runner = InMemoryRunner(agent=build_table_selection_recall_agent(config=config))
    precision_runner = InMemoryRunner(agent=build_table_selection_precision_agent(config=config))
    decider_runner = InMemoryRunner(agent=build_table_selection_decider_agent(config=config))

    base_payload = {
        "candidate_ids": candidate_ids,
        "user_intent": user_intent,
        "task_info": task_info,
    }
    base_prompt = (
        "Select tables from candidate_ids and return strict JSON.\n"
        f"Input:\n{json.dumps(base_payload, ensure_ascii=False)}"
    )

    recall_events = await recall_runner.run_debug(base_prompt, quiet=True)
    precision_events = await precision_runner.run_debug(base_prompt, quiet=True)

    recall_json = _extract_json(_collect_text(recall_events))
    precision_json = _extract_json(_collect_text(precision_events))

    recall_ids = set(_normalize_table_ids(recall_json.get("relevant_tables", [])))
    precision_ids = set(_normalize_table_ids(precision_json.get("relevant_tables", [])))
    risk_ids = set(
        _normalize_table_ids(precision_json.get("high_risk_tables", []))
    )
    # Backward compatibility: treat old veto output as high risk.
    risk_ids |= set(_normalize_table_ids(precision_json.get("veto_tables", [])))

    intersection_ids = sorted(list(recall_ids & precision_ids))
    recall_only_ids = sorted(list(recall_ids - precision_ids))
    precision_only_ids = sorted(list(precision_ids - recall_ids))

    decider_input = {
        "intersection_ids": intersection_ids,
        "recall_only_ids": recall_only_ids,
        "precision_only_ids": precision_only_ids,
        "precision_risk_ids": sorted(list(risk_ids)),
        "rule": {
            "intersection_priority": "high",
            "recall_only_risk": "high",
            "precision_risk": "tag_only_not_default_drop",
        },
    }
    decider_prompt = (
        "Decide final table list based on the collaboration rule and return strict JSON.\n"
        f"Input:\n{json.dumps(decider_input, ensure_ascii=False)}"
    )
    decider_events = await decider_runner.run_debug(decider_prompt, quiet=True)
    decider_json = _extract_json(_collect_text(decider_events))

    # Minimal deterministic fallback if decider fails.
    final_tables = decider_json.get("final_tables", [])
    if not isinstance(final_tables, list) or len(final_tables) == 0:
        final_tables = (
            [{"table_id": tid, "risk": "low", "reason": "Selected by both recall and precision."} for tid in intersection_ids]
            + [{"table_id": tid, "risk": "high", "reason": "Selected by recall only."} for tid in recall_only_ids]
            + [{"table_id": tid, "risk": "high", "reason": "Precision-only candidate."} for tid in precision_only_ids]
        )

    return {
        "final_tables": final_tables,
        "debug": {
            "recall_ids": sorted(list(recall_ids)),
            "precision_ids": sorted(list(precision_ids)),
            "precision_risk_ids": sorted(list(risk_ids)),
            "intersection_ids": intersection_ids,
            "recall_only_ids": recall_only_ids,
            "precision_only_ids": precision_only_ids,
        },
    }
