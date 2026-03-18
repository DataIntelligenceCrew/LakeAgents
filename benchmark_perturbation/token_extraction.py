
import json
import sys
from pathlib import Path
from typing import Union, List

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
# Avoid loading benchmark_perturbation.py (script) as the package when run directly
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

from benchmark_perturbation.metadata_perturbation import (
    extract_query_table_jaccard_fields,
    tokenize,
)
import re
import json
from pathlib import Path
from typing import Union, List

QUERY_BASE = Path("/localdisk3/ytang49/opendata/query_table")
OUTPUT_DIR = Path("/localdisk3/ytang49/opendata/benchmark_perturbation/jaccard_tokenized")

TABLES = [
    ("COVID-NYC", "Number_deaths", "extract_date"),
    ("COVID-Chicago", "Deaths - Total", "Date"),
    ("Demo-Chicago", "No High School Diploma", "Community Area"),
    ("Demo-NYC", "EducAttain", "SERIALNO"),
    ("Economic-Chicago", "Under $25,000", "Community Area"),
    ("Economic-NYC", "HHT", "SERIALNO"),
    ("Education-Chicago", "CPS Performance Policy Status", "School ID"),
    ("Education-NYC", "2009-2010 OVERALL GRADE", "DBN"),
    ("Taxi-Chicago", "Trip Total", "Trip ID"),
    ("Traffic_Chicago", "INJURIES_TOTAL", "CRASH_RECORD_ID"),
    ("Taxi-NYC", "total_amount", ["PULocationID", "DOLocationID"]),
    ("Environment_NYC", "health", "tree_id"),
    ("Food Inspections-NYC", "SCORE", "CAMIS"),
    ("Food Inspections-Chicago", "Risk", "Inspection ID"),
    ("Building Permits-Chicago", "PERMIT_STATUS", "ID"),
]

def fields_to_tokenized_dict(fields: dict) -> dict:
    """Convert 6 fields to raw + tokenized structure, tokens use list for JSON serialization."""
    result = {}
    for key in ["query_table_name", "query_table_description", "target_column", "target_column_description"]:
        raw = fields.get(key, "")
        result[key] = {"raw": raw, "tokens": sorted(tokenize(raw))}
    # join_column and join_column_description are lists
    join_cols = fields.get("join_column", [])
    join_descs = fields.get("join_column_description", [])
    result["join_column"] = [
        {"raw": c, "tokens": sorted(tokenize(c))} for c in join_cols
    ]
    result["join_column_description"] = [
        {"raw": d, "tokens": sorted(tokenize(d))} for d in join_descs
    ]
    return result

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder, target_col, join_col in TABLES:
        meta_path = QUERY_BASE / folder / "metadata.json"
        if not meta_path.exists():
            print(f"Skip {folder}: metadata.json not found")
            continue
        fields = extract_query_table_jaccard_fields(meta_path, join_col, target_col)
        tokenized = fields_to_tokenized_dict(fields)
        out_path = OUTPUT_DIR / f"{folder}_tokenized.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tokenized, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()