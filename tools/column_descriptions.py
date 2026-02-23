import json
from pathlib import Path
from typing import Optional


def _get_domain_from_index(table_id: str) -> Optional[str]:
    """Get domain for table_id from opendata_table_index.json (avoids circular import)."""
    index_path = Path(__file__).resolve().parent.parent / "opendata_table_index.json"
    if not index_path.exists():
        return None
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception:
        return None
    if not isinstance(index, list):
        return None
    tid = str(table_id).strip()
    for entry in index:
        if isinstance(entry, dict) and str(entry.get("id", "")).strip() == tid:
            return entry.get("domain")
    return None


def get_fieldname_to_columns_name_mapping(
    table_id: str, domain: Optional[str] = None
) -> dict[str, str]:
    """
    Get fieldName -> columns_name mapping for renaming API column names to match index.
    Tries opendata_metadata_cache first, then Socrata API as fallback.
    """
    # 1. Try metadata cache
    if domain:
        safe_domain = domain.replace(".", "_")
        cache_path = (
            Path(__file__).resolve().parent.parent
            / "opendata_metadata_cache"
            / safe_domain
            / f"{table_id}.json"
        )
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cols = data.get("columns", [])
                return {
                    str(c["fieldName"]): str(c["name"])
                    for c in cols
                    if c.get("fieldName") and c.get("name")
                }
            except Exception:
                pass

    # 2. Fallback: fetch metadata from Socrata API
    try:
        if not domain:
            domain = _get_domain_from_index(table_id)
        if domain:
            from agent_config_loader import load_config
            from datalake_client import SocrataDatalakeClient

            cfg = load_config()
            client = SocrataDatalakeClient(cfg.get("data", {}).get("datalake", {}))
            meta = client.get_dataset_metadata(table_id, domain)
            if isinstance(meta, dict) and "error" not in meta:
                cols = meta.get("columns", [])
                return {
                    str(c["fieldName"]): str(c["name"])
                    for c in cols
                    if c.get("fieldName") and c.get("name")
                }
    except Exception:
        pass

    return {}


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

def get_column_datatypes_from_index(table_id: str) -> dict[str, str]:
    """Get column_name -> dataTypeName from opendata_table_index.json."""
    index_path = Path(__file__).resolve().parent.parent / "opendata_table_index.json"
    if not index_path.exists():
        return {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception:
        return {}
    if not isinstance(index, list):
        return {}
    tid = str(table_id).strip()
    for entry in index:
        if isinstance(entry, dict) and str(entry.get("id", "")).strip() == tid:
            names = entry.get("columns_name") or []
            dtypes = entry.get("columns_datatype") or []
            return {str(n): str(dtypes[i]) if i < len(dtypes) else "unknown" for i, n in enumerate(names)}
    return {}

def get_column_datatypes_from_local_metadata(
    base_dir: str, dataset_folder_name: str
) -> dict[str, str]:
    """Get column_name -> dataTypeName from local metadata.json in dataset folder."""
    meta_path = Path(base_dir) / dataset_folder_name / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    res = data.get("resource") or {}
    names = res.get("columns_name") or []
    dtypes = res.get("columns_datatype") or []
    return {str(n): str(dtypes[i]) if i < len(dtypes) else "unknown" for i, n in enumerate(names)}


def get_column_descriptions_from_local_metadata(
    base_dir: str, dataset_folder_name: str
) -> dict[str, str]:
    """Get column_name -> description from local metadata.json in dataset folder."""
    meta_path = Path(base_dir) / dataset_folder_name / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    res = data.get("resource") or {}
    names = res.get("columns_name") or []
    descs = res.get("columns_description") or []
    return {str(n): str(descs[i]) if i < len(descs) else "" for i, n in enumerate(names)}