import hashlib
import json
from datetime import datetime
from typing import Any, Dict


def build_task_id(session_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = str(session_id or "default").strip() or "default"
    return f"{ts}_{sid}"


def compute_config_version(config_dict: Dict[str, Any]) -> str:
    payload = json.dumps(config_dict or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def classify_outcome(baseline: Any, augmented: Any) -> str:
    if baseline is None or augmented is None:
        return "unknown"
    try:
        improvement = float(augmented) - float(baseline)
    except Exception:
        return "unknown"
    if improvement < 0:
        return "regression"
    if improvement == 0:
        return "no_gain"
    if improvement >= 0.05:
        return "large_gain"
    return "small_gain"


def init_decision_log(
    *,
    task_id: str,
    session_id: str,
    join_table_name: str,
    task_type: str,
    target_column: str,
    config_version: str,
    threshold_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "session_id": session_id,
        "join_table_name": join_table_name,
        "task_type": task_type,
        "target_column": target_column,
        "config_version": config_version,
        "threshold_snapshot": threshold_snapshot,
        "created_at": datetime.now().isoformat(),
        "phases": {
            "table_selection": {},
            "join_column_selection": {},
            "data_quality": {},
            "augment": {},
        },
    }

