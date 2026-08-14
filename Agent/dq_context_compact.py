"""Compact DataQuality multi-turn context after tool rounds (option-2).

ADK run_debug keeps full chat history across tool calls. Wide tables already
consume most of a 16k window on turn-1, so turn-2+ overflows. Before each
subsequent LLM call we rewrite llm_request.contents to:
  task summary (constraints + slim quality stats) + tool traces so far.
Session storage still grows, but the model only sees the compact view.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from google.genai import types

_TOOL_RESULT_MAX_CHARS = 2500
_COMPACT_PROMPT_MAX_CHARS = 12000
_QUALITY_STAT_KEYS = (
    "column_name",
    "dtype",
    "missing_rate",
    "outlier_rate",
    "variance",
    "relative_cardinality",
    "top1_ratio",
    "column_description",
)


def _safe_json(obj: Any, max_chars: int) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def _slim_quality_stats(stats: Any) -> List[Dict[str, Any]]:
    if not isinstance(stats, list):
        return []
    slim: List[Dict[str, Any]] = []
    for item in stats:
        if not isinstance(item, dict):
            continue
        row = {k: item.get(k) for k in _QUALITY_STAT_KEYS if k in item}
        desc = str(row.get("column_description") or "")
        if len(desc) > 80:
            row["column_description"] = desc[:80] + "..."
        slim.append(row)
    return slim


def _task_summary_from_user_text(user_text: str) -> Dict[str, Any]:
    """Pull structural constraints from the original DQ user prompt."""
    payload: Dict[str, Any] = {}
    m = re.search(r"input:\s*(\{.*\})\s*$", user_text, flags=re.S)
    raw = m.group(1) if m else None
    if raw:
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
    if not payload:
        # Fallback: keep a truncated raw prompt without row dumps.
        return {
            "raw_task_preview": user_text[:2000] + ("..." if len(user_text) > 2000 else ""),
        }
    return {
        "candidate_table": payload.get("candidate_table"),
        "query_table_name": payload.get("query_table_name"),
        "query_join_columns": payload.get("query_join_columns"),
        "selected_join_columns": payload.get("selected_join_columns"),
        "available_tools": payload.get("available_tools"),
        "metadata": payload.get("metadata") or [],
        "quality_stats": _slim_quality_stats(payload.get("quality_stats")),
        # Intentionally omit row_examples — largest token sink and already
        # summarized via quality_stats for subsequent tool rounds.
    }


def _first_user_text(contents: List[Any]) -> str:
    for content in contents:
        role = getattr(content, "role", None)
        if role not in (None, "user"):
            continue
        parts = getattr(content, "parts", None) or []
        texts = []
        for part in parts:
            t = getattr(part, "text", None)
            if t:
                texts.append(t)
        if texts:
            return "\n".join(texts)
    return ""


def _has_function_response(contents: List[Any]) -> bool:
    for content in contents:
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "function_response", None) is not None:
                return True
    return False


def _collect_tool_traces(contents: List[Any]) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for content in contents:
        for part in getattr(content, "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                args = getattr(fc, "args", None) or {}
                if hasattr(args, "items"):
                    args = dict(args)
                # Drop huge value arrays from tool args in the compact view.
                if isinstance(args, dict) and "values" in args:
                    vals = args.get("values")
                    n = len(vals) if isinstance(vals, list) else "?"
                    args = {**args, "values": f"[omitted n={n}]"}
                traces.append(
                    {
                        "type": "call",
                        "name": getattr(fc, "name", None),
                        "args": args,
                    }
                )
            fr = getattr(part, "function_response", None)
            if fr is not None:
                resp = getattr(fr, "response", None)
                traces.append(
                    {
                        "type": "response",
                        "name": getattr(fr, "name", None),
                        "response": _safe_json(resp, _TOOL_RESULT_MAX_CHARS),
                    }
                )
    return traces


def build_compact_continuation_prompt(contents: List[Any]) -> str:
    task = _task_summary_from_user_text(_first_user_text(contents))
    traces = _collect_tool_traces(contents)
    prompt = (
        "Continue data-quality decisions with a FRESH context "
        "(prior chat turns cleared to fit the context window).\n\n"
        "TASK SUMMARY (constraints + slim quality stats; row_examples omitted):\n"
        f"{_safe_json(task, 8000)}\n\n"
        "TOOL TRACES SO FAR (calls + truncated results):\n"
        f"{_safe_json(traces, 6000)}\n\n"
        "Rules:\n"
        "- Never drop query_join_columns or selected_join_columns.\n"
        "- Prefer keep/repair over drop unless the column is clearly unusable.\n"
        "- Do not repeat tools that already returned conclusive results.\n"
        "- Either call another tool if still needed, or return ONLY the final JSON "
        "with key column_actions.\n"
    )
    if len(prompt) > _COMPACT_PROMPT_MAX_CHARS:
        prompt = prompt[:_COMPACT_PROMPT_MAX_CHARS] + "\n...[compact_prompt_truncated]"
    return prompt


def dq_compact_before_model(callback_context: Any, llm_request: Any) -> None:
    """Rewrite model contents after the first tool round (option-2)."""
    contents = list(getattr(llm_request, "contents", None) or [])
    if not contents or not _has_function_response(contents):
        return None
    compact = build_compact_continuation_prompt(contents)
    llm_request.contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=compact)])
    ]
    return None


def dq_truncate_after_tool(
    tool: Any,
    args: Dict[str, Any],
    tool_context: Any,
    tool_response: Any,
) -> Optional[Dict[str, Any]]:
    """Keep tool responses small so compact traces stay within budget."""
    if tool_response is None:
        return None
    try:
        text = json.dumps(tool_response, ensure_ascii=False, default=str)
    except Exception:
        text = str(tool_response)
    if len(text) <= _TOOL_RESULT_MAX_CHARS:
        return None
    # Prefer returning a truncated dict when possible.
    if isinstance(tool_response, dict):
        out = dict(tool_response)
        out["_truncated"] = True
        out["_original_chars"] = len(text)
        # Keep a short preview of the full dump.
        out["_preview"] = text[:_TOOL_RESULT_MAX_CHARS]
        # Drop bulky keys commonly returned by preview tools.
        for k in ("values", "imputed_values", "preview_rows", "samples"):
            if k in out and isinstance(out[k], list) and len(out[k]) > 8:
                out[k] = out[k][:8] + [f"...(+{len(tool_response[k]) - 8} more)"]
        return out
    return {"_truncated": True, "_preview": text[:_TOOL_RESULT_MAX_CHARS]}
