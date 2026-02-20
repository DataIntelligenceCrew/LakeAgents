
from pathlib import Path
from typing import List, Optional

import pandas as pd

from llm_agent_tools import find_dataset_dir
from agent_config_loader import load_config
import json

_INDEX_PATH = Path(__file__).resolve().parent.parent / "opendata_table_index.json"


def _get_domain_from_index(table_id: str) -> Optional[str]:
    """Find domain from opendata_table_index.json by table_id."""
    if not _INDEX_PATH.exists():
        return None
    try:
        with open(_INDEX_PATH, "r", encoding="utf-8") as f:
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

def get_query_column_values(
    table_id: str,
    join_column: str,
    base_dir: str,
    data_filename: str = "rows.csv"

) -> List[str]:
    """
    Get values from a local query table.
    
    Args:
        table_id: Table name/directory name
        join_column: Column name
        base_dir: Local data root directory
        data_filename: CSV file name    
        max_rows: Maximum number of rows
    
    Returns:
        List of values (stripped and lowercased)
    """
    real_name = find_dataset_dir(table_id, base_dir)
    path = Path(base_dir) / real_name / data_filename
    df = pd.read_csv(path, low_memory=False)
    if join_column not in df.columns:
        return []
    s = df[join_column].dropna().astype(str).str.strip().str.lower()
    vals = [v for v in s if v]
    return vals


def get_candidate_table(
    table_id: str,
    opendata_domain: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch full candidate table from API. Only needs table_id.
    When opendata_domain is None, looks up domain from opendata_table_index.json.
    """
    from datalake_client import SocrataDatalakeClient

    domain = opendata_domain
    if domain is None:
        domain = _get_domain_from_index(table_id)
        if not domain:
            return pd.DataFrame()

    cfg = load_config()
    client = SocrataDatalakeClient(cfg.get("data", {}).get("datalake", {}))
    rows = client.read_data(table_id, domain, max_rows=None)
    # DEBUG
    print("[DEBUG] domain:", domain, "len(rows):", len(rows) if isinstance(rows, list) else type(rows))
    if rows and isinstance(rows, list) and len(rows) > 0:
        print("[DEBUG] first row keys:", list(rows[0].keys()) if isinstance(rows[0], dict) else type(rows[0]))
    return pd.DataFrame(rows) if rows else pd.DataFrame()

   
