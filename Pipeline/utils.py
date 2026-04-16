import json
import inspect
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pandas as pd


@contextmanager
def timed_section(accum: Dict[str, float], name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        accum[name] = accum.get(name, 0.0) + (time.perf_counter() - t0)


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def extract_json_by_key_from_full_text(
    text: str, key: str, prefer_non_empty_list: bool = True
) -> Dict[str, Any]:
    if not text or not text.strip():
        return {}
    candidates: List[Dict[str, Any]] = []
    i = 0
    while i < len(text):
        pos = text.find("{", i)
        if pos < 0:
            break
        depth = 0
        start = pos
        for j in range(pos, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : j + 1])
                        if key in obj:
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                    break
        i = pos + 1
    if not candidates:
        return {}
    if prefer_non_empty_list:
        for c in candidates:
            v = c.get(key)
            if isinstance(v, list) and len(v) > 0:
                return c
    return candidates[-1]


def coerce_augment_column_name(item: Any) -> Optional[str]:
    if item is None:
        return None
    try:
        if item is pd.NA:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(item, str):
        s = item.strip()
        return s if s else None
    if isinstance(item, dict):
        for key in ("column", "name", "column_name", "feature"):
            v = item.get(key)
            if v is not None:
                s = str(v).strip()
                if s:
                    return s
        return None
    if isinstance(item, (int, float, bool)):
        s = str(item).strip()
        return s if s else None
    return None


def normalize_augment_column_list(items: Any) -> List[str]:
    if not items:
        return []
    seq = items if isinstance(items, list) else [items]
    out: List[str] = []
    for it in seq:
        name = coerce_augment_column_name(it)
        if name:
            out.append(name)
    return out


def first_non_empty_rows(df: pd.DataFrame, columns: List[str], n: int = 5) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return []
    rows = []
    for _, row in df[cols].iterrows():
        rec = {}
        has_non_empty = False
        for c in cols:
            v = row[c]
            if pd.notna(v) and str(v).strip():
                rec[c] = str(v)
                has_non_empty = True
            else:
                rec[c] = None
        if has_non_empty:
            rows.append(rec)
        if len(rows) >= n:
            break
    return rows


def first_non_empty_values(series: pd.Series, n: int = 5) -> List[str]:
    values: List[str] = []
    if series is None:
        return values
    for v in series:
        if pd.notna(v) and str(v).strip():
            values.append(str(v))
        if len(values) >= n:
            break
    return values


async def close_runner_safely(runner: Any) -> None:
    """Best-effort close for ADK runners to avoid unclosed client sessions."""
    if runner is None:
        return
    for method_name in ("aclose", "close"):
        method = getattr(runner, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass
        return

