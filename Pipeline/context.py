from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PipelineContext:
    config: Any
    session_id: str
    join_table_name: str
    target_column: str
    task_type: str
    user_intent: str
    base_dir: str
    real_join_table_name: str
    query_table_display_name: str
    base_path_obj: Path
    join_meta_path: Path
    pipeline_timings: Dict[str, float] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    run_record_path: Optional[Path] = None

