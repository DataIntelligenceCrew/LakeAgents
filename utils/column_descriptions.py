import json
from pathlib import Path

def get_column_descriptions_from_index(table_id: str) -> dict[str, str]:
    """Get column_name -> description from opendata_table_index.json."""
    index_path = Path(__file__).resolve().parent.parent / "opendata_table_index.json"
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    tid = str(table_id).strip()
    for entry in index:
        if isinstance(entry, dict) and str(entry.get("id", "")).strip() == tid:
            names = entry.get("columns_name") or []
            descs = entry.get("columns_description") or []
            return {str(n): str(descs[i]) if i < len(descs) else "" 
                    for i, n in enumerate(names)}
    return {}

def get_column_descriptions_from_local_metadata(
    base_dir: str, dataset_folder_name: str
) -> dict[str, str]:
    """Get column_name -> description from local metadata.json in dataset folder."""
    meta_path = Path(base_dir) / dataset_folder_name / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = data.get("resource") or {}
    names = res.get("columns_name") or []
    descs = res.get("columns_description") or []
    return {str(n): str(descs[i]) if i < len(descs) else "" for i, n in enumerate(names)}