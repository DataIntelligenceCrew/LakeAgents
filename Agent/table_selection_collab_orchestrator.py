"""Minimal collaborative orchestrator for table selection."""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Iterable, List, Sequence

from google.adk.runners import InMemoryRunner

from Agent.table_selection_recall_agent import build_table_selection_recall_agent
from Agent.table_selection_precision_agent import build_table_selection_precision_agent
from Agent.table_selection_decider_agent import build_table_selection_decider_agent
from Pipeline.utils import close_runner_safely


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


def _chunked(items: Sequence[str], size: int) -> List[List[str]]:
    if size <= 0:
        size = 3
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _slim_examples_for_ids(
    examples_by_id: Any,
    batch_ids: Iterable[str],
    *,
    max_rows: int = 3,
    max_cols: int = 12,
    max_cell_chars: int = 80,
) -> Dict[str, List[Dict[str, Any]]]:
    """Keep only this batch's examples and aggressively truncate wide tables."""
    if not isinstance(examples_by_id, dict):
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for tid in batch_ids:
        rows = examples_by_id.get(tid) or examples_by_id.get(str(tid))
        if not isinstance(rows, list):
            continue
        slim_rows: List[Dict[str, Any]] = []
        for row in rows[:max_rows]:
            if not isinstance(row, dict):
                continue
            keys = list(row.keys())[:max_cols]
            slim: Dict[str, Any] = {}
            for k in keys:
                v = row.get(k)
                if v is None:
                    slim[k] = None
                else:
                    s = str(v)
                    slim[k] = s if len(s) <= max_cell_chars else (s[:max_cell_chars] + "...")
            slim_rows.append(slim)
        out[str(tid)] = slim_rows
    return out


def _task_info_for_batch(task_info: Dict[str, Any], batch_ids: Sequence[str]) -> Dict[str, Any]:
    """Clone task_info with examples restricted to the current candidate batch."""
    ti = dict(task_info or {})
    desc = str(ti.get("query_table_description") or "")
    if len(desc) > 800:
        ti["query_table_description"] = desc[:800] + "...[truncated]"
    ti["table_examples_5rows_all_columns"] = _slim_examples_for_ids(
        ti.get("table_examples_5rows_all_columns"),
        batch_ids,
    )
    ti["candidate_batch_ids"] = list(batch_ids)
    return ti


def _batch_size() -> int:
    try:
        return max(1, int(os.environ.get("TABLE_SELECTION_BATCH_SIZE", "3")))
    except Exception:
        return 3


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

    Recall/precision run in candidate batches so prompts fit local vLLM context
    (e.g. max_model_len=16384). Decider still runs once on the merged shortlists.
    """
    recall_runner = None
    precision_runner = None
    decider_runner = None

    clean_ids = [str(x).strip() for x in (candidate_ids or []) if str(x).strip()]
    batches = _chunked(clean_ids, _batch_size()) or [[]]
    print(
        f"[TableSelectionCollab] candidates={len(clean_ids)} "
        f"batches={len(batches)} batch_size={_batch_size()}"
    )

    recall_ids: set = set()
    precision_ids: set = set()
    risk_ids: set = set()
    batch_debug: List[Dict[str, Any]] = []

    try:
        for bi, batch in enumerate(batches):
            if not batch:
                continue
            payload = {
                "candidate_ids": batch,
                "user_intent": user_intent,
                "task_info": _task_info_for_batch(task_info, batch),
                "batch_index": bi,
                "batch_total": len(batches),
            }
            prompt = (
                "Select tables ONLY from candidate_ids in this batch and return strict JSON.\n"
                "Do not invent ids outside candidate_ids.\n"
                f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            # Rough size guardrail for logs
            print(
                f"[TableSelectionCollab] batch {bi + 1}/{len(batches)} "
                f"n={len(batch)} prompt_chars={len(prompt)} (fresh_runner)"
            )

            # Fresh runners per batch: ADK run_debug reuses session_id by default
            # and otherwise accumulates history across batches until context overflows.
            recall_runner = InMemoryRunner(agent=build_table_selection_recall_agent(config=config))
            precision_runner = InMemoryRunner(agent=build_table_selection_precision_agent(config=config))
            try:
                sid = f"tsel_{bi}_{uuid.uuid4().hex[:8]}"
                recall_events = await recall_runner.run_debug(
                    prompt, quiet=True, session_id=f"recall_{sid}"
                )
                precision_events = await precision_runner.run_debug(
                    prompt, quiet=True, session_id=f"precision_{sid}"
                )
            finally:
                await close_runner_safely(recall_runner)
                await close_runner_safely(precision_runner)
                recall_runner = None
                precision_runner = None

            recall_json = _extract_json(_collect_text(recall_events))
            precision_json = _extract_json(_collect_text(precision_events))

            b_recall = set(_normalize_table_ids(recall_json.get("relevant_tables", [])))
            b_precision = set(_normalize_table_ids(precision_json.get("relevant_tables", [])))
            b_risk = set(_normalize_table_ids(precision_json.get("high_risk_tables", [])))
            b_risk |= set(_normalize_table_ids(precision_json.get("veto_tables", [])))

            # Keep only ids that were in this batch (reject hallucinations)
            batch_set = set(batch)
            b_recall &= batch_set
            b_precision &= batch_set
            b_risk &= batch_set

            recall_ids |= b_recall
            precision_ids |= b_precision
            risk_ids |= b_risk
            batch_debug.append(
                {
                    "batch_index": bi,
                    "candidate_ids": batch,
                    "recall_ids": sorted(b_recall),
                    "precision_ids": sorted(b_precision),
                    "risk_ids": sorted(b_risk),
                    "prompt_chars": len(prompt),
                }
            )

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
        decider_runner = InMemoryRunner(agent=build_table_selection_decider_agent(config=config))
        decider_events = await decider_runner.run_debug(
            decider_prompt,
            quiet=True,
            session_id=f"decider_{uuid.uuid4().hex[:12]}",
        )
        decider_json = _extract_json(_collect_text(decider_events))

        # Minimal deterministic fallback if decider fails.
        final_tables = decider_json.get("final_tables", [])
        if not isinstance(final_tables, list) or len(final_tables) == 0:
            final_tables = (
                [
                    {
                        "table_id": tid,
                        "risk": "low",
                        "reason": "Selected by both recall and precision.",
                    }
                    for tid in intersection_ids
                ]
                + [
                    {
                        "table_id": tid,
                        "risk": "high",
                        "reason": "Selected by recall only.",
                    }
                    for tid in recall_only_ids
                ]
                + [
                    {
                        "table_id": tid,
                        "risk": "high",
                        "reason": "Precision-only candidate.",
                    }
                    for tid in precision_only_ids
                ]
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
                "batched": True,
                "batch_size": _batch_size(),
                "batches": batch_debug,
            },
        }
    finally:
        await close_runner_safely(recall_runner)
        await close_runner_safely(precision_runner)
        await close_runner_safely(decider_runner)
