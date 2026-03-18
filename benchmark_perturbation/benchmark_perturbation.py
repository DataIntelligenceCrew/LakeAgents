import sys
from pathlib import Path
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))
_script_dir = str(Path(__file__).resolve().parent)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

from benchmark_perturbation.metadata_perturbation import (
    run_all_tokenized_files,
    map_perturbed_to_query_table,
)
from benchmark_perturbation.data_perturbation import process_numerical_perturb
from agent_config_loader import load_config, apply_replacements_to_task_config, load_replacements_for_table

import yaml

def _load_perturbation_params() -> dict:
    path = _PROJECT / "configs" / "perturbation.yaml"
    if not path.exists():
        return {"threshold": 0.85, "beta": 0.1}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        p = cfg.get("perturbation", {})
        return {
            "threshold": float(p.get("threshold", 0.85)),
            "beta": float(p.get("beta", 0.1)),
        }
    except Exception:
        return {"threshold": 0.85, "beta": 0.1}


def _load_run_tables():
    """If config has run_tables, return that list; else return None (run all)."""
    path = _PROJECT / "configs" / "perturbation.yaml"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        run_tables = cfg.get("run_tables")
        if run_tables is None or (isinstance(run_tables, list) and len(run_tables) == 0):
            return None
        return list(run_tables)
    except Exception:
        return None

_perturb_params = _load_perturbation_params()
THRESHOLD = _perturb_params["threshold"]
BETA = _perturb_params["beta"]

QUERY_TABLE = _PROJECT / "query_table"
JACCARD_TOKENIZED = _PROJECT / "benchmark_perturbation/jaccard_tokenized"
JACCARD_PERTURBED = _PROJECT / "benchmark_perturbation/jaccard_perturbed"
SYNONYM_DICT = _PROJECT / "benchmark_perturbation/Synonym Replacement Dict.json"

def run_full_pipeline(threshold=THRESHOLD, beta=BETA, tables=None):
    """
    tables: If None, process all tables. If list e.g. ["Taxi-Chicago", "Traffic_Chicago", "Taxi-NYC", "Environment_NYC"], only process those.
    """
    output_dir = _PROJECT / f"perturbed_{threshold}_{beta}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata perturbation
    run_all_tokenized_files(
        JACCARD_TOKENIZED, SYNONYM_DICT, threshold=threshold, tables=tables
    )
    map_perturbed_to_query_table(
        JACCARD_PERTURBED, QUERY_TABLE, _PROJECT,
        beta=beta,
        tables=tables,
    )

    # 2. Data perturbation (do on output_dir)
    folders = [d.name for d in output_dir.iterdir() if d.is_dir()]
    if tables is not None:
        folders = [f for f in folders if f in set(tables)]
    for folder in folders:
        process_numerical_perturb(
            folder,
            base_dir=output_dir,   # read from perturbed directory
            output_dir=output_dir,
            beta=beta,
        )
    print(f"Done: {output_dir}")


def get_perturbed_pipeline_config(
    table_folder: str,
    threshold: float = THRESHOLD,
    beta: float = BETA,
) -> dict:
    """
    Build config for running pipeline on perturbed data.
    Uses perturbation.yaml for join/target, applies replacements, sets base_dir.
    Use: config = AgentPipelineConfig(config_dict=get_perturbed_pipeline_config("COVID-Chicago"))
    """
    from benchmark_perturbation.data_perturbation import load_perturbation_config

    cfg = load_config()
    table_cfg = load_perturbation_config().get(table_folder, {})
    cfg["task"] = {**cfg.get("task", {})}
    cfg["task"]["join_table_name"] = table_folder
    cfg["task"]["join_column"] = table_cfg.get("join_columns", cfg["task"].get("join_column"))
    cfg["task"]["target_column"] = table_cfg.get("target_column", cfg["task"].get("target_column"))

    perturbed_dir = _PROJECT / f"perturbed_{threshold}_{beta}"
    replacements = load_replacements_for_table(
        table_folder, _PROJECT, perturbed_base_dir=perturbed_dir
    )
    cfg = apply_replacements_to_task_config(cfg, replacements)

    cfg["data"] = {**cfg.get("data", {})}
    cfg["data"]["base_dir"] = str(perturbed_dir)
    return cfg


if __name__ == "__main__":
    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tables", "-t", nargs="+",
        help="Only run these tables (e.g. -t Taxi-Chicago Traffic_Chicago Taxi-NYC Environment_NYC)",
    )
    parser.add_argument(
        "--one-by-one", action="store_true",
        help="Process each table in a separate subprocess to free memory between runs (reduces segfault risk for large tables)",
    )
    args = parser.parse_args()

    tables = args.tables if args.tables else _load_run_tables()
    if args.one_by_one and tables and len(tables) > 1:
        # Run each table in separate subprocess to avoid memory buildup
        script = Path(__file__).resolve()
        for table in tables:
            print(f"\n=== Processing table: {table} ===")
            ret = subprocess.run(
                [sys.executable, str(script), "-t", table],
                cwd=str(_PROJECT),
            )
            if ret.returncode != 0:
                print(f"Failed: {table} (exit code {ret.returncode})")
                sys.exit(ret.returncode)
        print(f"\nDone: all {len(tables)} tables processed")
    else:
        run_full_pipeline(tables=tables)