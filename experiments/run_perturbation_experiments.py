#!/usr/bin/env python3
"""
"""
import litellm
import os
import sys
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime
from agent_config_loader import load_config

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

TAU_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

TASKS = [
    {"join_table": "COVID-Chicago", "task_type": "regression", "user_intent": "Predict the daily mortality rate of covid in Chicago"},
    {"join_table": "Demo-Chicago", "task_type": "regression", "user_intent": "Predict the ratio of people without high school diploma using Public Health data in Chicago"},
    {"join_table": "Economic-Chicago", "task_type": "regression", "user_intent": "Predict the number of people having annual income lower than $25000 using community survey data in Chicago"},
    {"join_table": "Education-Chicago", "task_type": "classification", "user_intent": "predict the public school performance from 2011-2012 in Chicago"},
    {"join_table": "COVID-NYC", "task_type": "regression", "user_intent": "Predict the daily mortality rate of covid in NYC"},
    {"join_table": "Demo-NYC", "task_type": "classification", "user_intent": "predict the education level of people in NYC based on the data about poverty in 2018"},
    {"join_table": "Economic-NYC", "task_type": "classification", "user_intent": "predict the household/family type in NYC using the poverty data in 2018"},
    {"join_table": "Education-NYC", "task_type": "classification", "user_intent": "predict the grade of school in nyc using the education record in 2009-2010"},
    {"join_table": "Environment_NYC", "task_type": "classification", "user_intent": "predict the health condition of street trees in NYC from the 2015 street tree census"},
    # Food Inspections-NYC: disabled (very slow in grid); uncomment line + TASK_DIMENSIONS below to restore.
    # {"join_table": "Food Inspections-NYC", "task_type": "classification", "user_intent": "Predict the score of different restaurants in food inspection of NYC"},
    {"join_table": "Food Inspections-Chicago", "task_type": "classification", "user_intent": "Predict the risk level of food inspection in Chicago"},
    {"join_table": "Building Permits-Chicago", "task_type": "classification", "user_intent": "Predict the building permit type of buildings in Chicago"},
]

TASK_DIMENSIONS = {
    "COVID-Chicago": {"Domain/Field": ["covid-19"], "Geographic": ["Chicago"], "Temporal": ["daily"], "Population Group": ["all"]},
    "Demo-Chicago": {"Domain/Field": ["public health"], "Geographic": ["Chicago"], "Temporal": ["all"], "Population Group": ["all"]},
    "Economic-Chicago": {"Domain/Field": ["community survey data"], "Geographic": ["Chicago"], "Temporal": ["all"], "Population Group": ["annual income lower than $25000"]},
    "Education-Chicago": {"Domain/Field": ["public school"], "Geographic": ["Chicago"], "Temporal": ["2011-2012"], "Population Group": ["all"]},
    "COVID-NYC": {"Domain/Field": ["covid-19"], "Geographic": ["New York City"], "Temporal": ["daily"], "Population Group": ["all"]},
    "Demo-NYC": {"Domain/Field": ["Poverty"], "Geographic": ["New York City"], "Temporal": ["2018"], "Population Group": ["all"]},
    "Economic-NYC": {"Domain/Field": ["Poverty"], "Geographic": ["New York City"], "Temporal": ["2018"], "Population Group": ["all"]},
    "Education-NYC": {"Domain/Field": ["education"], "Geographic": ["New York City"], "Temporal": ["2009-2010"], "Population Group": ["all"]},
    "Environment_NYC": {"Domain/Field": ["tree"], "Geographic": ["New York City"], "Temporal": ["2015"], "Population Group": ["all"]},
    # "Food Inspections-NYC": {"Domain/Field": ["food inspections"], "Geographic": ["New York City"], "Temporal": ["all"], "Population Group": ["all"]},
    "Food Inspections-Chicago": {"Domain/Field": ["food inspections"], "Geographic": ["Chicago"], "Temporal": ["all"], "Population Group": ["all"]},
    "Building Permits-Chicago": {"Domain/Field": ["building permits"], "Geographic": ["Chicago"], "Temporal": ["all"], "Population Group": ["all"]},
}


def update_perturbation_yaml(threshold: float, beta: float) -> None:
    import yaml
    path = _PROJECT / "configs" / "perturbation.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("perturbation", {})
    cfg["perturbation"]["threshold"] = threshold
    cfg["perturbation"]["beta"] = beta
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def run_grid(verbose: bool = True, tables=None):
    """"""
    from benchmark_perturbation.benchmark_perturbation import run_full_pipeline

    results = []
    total = len(TAU_VALUES) * len(BETA_VALUES)
    idx = 0
    for tau in TAU_VALUES:
        for beta in BETA_VALUES:
            idx += 1
            if verbose:
                print(f"\n{'='*60}")
                print(f"[Grid {idx}/{total}] τ={tau}, β={beta}")
                print("="*60)
            try:
                run_full_pipeline(threshold=tau, beta=beta, tables=tables)
                results.append({"tau": tau, "beta": beta, "status": "ok"})
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"tau": tau, "beta": beta, "status": "error", "error": str(e)})
    return results


def run_orchestrator_inprocess(
    join_table: str, user_intent: str, task_type: str, session_id: str,
    base_dir: str, data_filename: str = "rows.csv", tau: float = None, beta: float = None,
    provider: str = None, session_checked_dir: str = None, config_history_file: str = None,
):
    """Run OUTER orchestrator in-process. Returns (returncode, output_dict)."""
    from agent_config_loader import AgentPipelineConfig
    from benchmark_perturbation.benchmark_perturbation import get_perturbed_pipeline_config

    cfg_dict = get_perturbed_pipeline_config(
        table_folder=join_table, threshold=tau or 0.85, beta=beta or 0.1,
    )
    cfg_dict["data"] = cfg_dict.get("data", {})
    cfg_dict["data"]["base_dir"] = base_dir
    cfg_dict["data"]["data_filename"] = data_filename
    cfg_dict.setdefault("task", {}).setdefault("session", {})
    cfg_dict["task"]["session"]["session_id"] = session_id

    if provider is not None:  
        cfg_dict.setdefault("agents", {})["default_provider"] = provider
    if session_checked_dir is not None:
        cfg_dict.setdefault("output", {})["session_checked_dir"] = session_checked_dir
    if config_history_file is not None:
        cfg_dict.setdefault("output", {})["config_history_file"] = config_history_file

    config = AgentPipelineConfig(config_dict=cfg_dict)

    dims = TASK_DIMENSIONS.get(join_table, {})
    responses = []
    for dim_name in ["Domain/Field", "Geographic", "Temporal", "Population Group"]:
        vals = dims.get(dim_name, ["all"])
        responses.append(", ".join(str(v) for v in vals) if isinstance(vals, list) else str(vals))

    import builtins
    _orig_input = builtins.input
    def _mock_input(_prompt=""):
        return responses.pop(0) if responses else "done"
    builtins.input = _mock_input

    from outer_orchestrator import run_outer_orchestrator
    try:
        output = asyncio.run(run_outer_orchestrator(
            config=config,
            user_intent=user_intent,
            session_id=session_id,
            join_table_name=join_table,
            task_type=task_type,
            config_history_file=config_history_file,
        ))
        return 0, output
    except Exception as e:
        return 1, {"error": str(e)}
    finally:
        builtins.input = _orig_input


def _load_json(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _summarize_outer_output(output: dict) -> dict:
    rounds = output.get("rounds", []) if isinstance(output, dict) else []
    best_round = {}
    if isinstance(rounds, list) and rounds:
        ranked_rounds = []
        for idx, r in enumerate(rounds):
            if not isinstance(r, dict):
                continue
            b = r.get("baseline_metric")
            a = r.get("augmented_metric")
            imp = (a - b) if (b is not None and a is not None) else None
            ranked_rounds.append((imp if imp is not None else float("-inf"), a if a is not None else float("-inf"), idx, r))
        if ranked_rounds:
            ranked_rounds.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
            best_round = ranked_rounds[0][3]
        else:
            best_round = rounds[-1] if isinstance(rounds[-1], dict) else {}
    baseline = best_round.get("baseline_metric") if isinstance(best_round, dict) else None
    augmented = best_round.get("augmented_metric") if isinstance(best_round, dict) else None
    improvement = (augmented - baseline) if (baseline is not None and augmented is not None) else None
    decision_log_path = best_round.get("decision_log_path") if isinstance(best_round, dict) else None
    decision_log = _load_json(decision_log_path)
    metric_name = decision_log.get("metric_name", "r2_score")
    selected_candidates = []
    augment_cols = {}
    augment_phase = ((decision_log.get("phases", {}) or {}).get("augment", {}) if isinstance(decision_log, dict) else {})
    selected_entries = augment_phase.get("selected", []) if isinstance(augment_phase, dict) else []
    if isinstance(selected_entries, list):
        for item in selected_entries:
            if not isinstance(item, dict):
                continue
            table_id = str(item.get("table_id") or "").strip()
            col = str(item.get("column") or "").strip()
            if not table_id or not col:
                continue
            if table_id not in augment_cols:
                augment_cols[table_id] = []
            if col not in augment_cols[table_id]:
                augment_cols[table_id].append(col)
        selected_candidates = list(augment_cols.keys())
    return {
        "baseline_metric": baseline,
        "augmented_metric": augmented,
        "improvement": improvement,
        "metric_name": metric_name,
        "selected_candidate_tables": selected_candidates,
        "augment_columns_per_candidate": augment_cols,
        "best_round": best_round,
    }


def append_experiment_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_experiments(tau_list, beta_list, data_filename, session_start, log_path, verbose=True):
    """"""
    total = len(tau_list) * len(beta_list) * len(TASKS)
    session_id = session_start
    results = []

    for tau in tau_list:
        for beta in beta_list:
            base_dir = f"perturbed_{tau}_{beta}"
            if not (_PROJECT / base_dir).exists():
                print(f"Skip {base_dir} (not found)")
                continue

            if verbose:
                print(f"\n{'='*60}\nτ={tau}, β={beta} -> {base_dir}")
            update_perturbation_yaml(tau, beta)

            for task in TASKS:
                sid = f"{session_id:03d}"
                if verbose:
                    print(f"\n--- [{session_id}/{total}] {task['join_table']} ({task['task_type']}) sess={sid} ---")
                ret, output = run_orchestrator_inprocess(
                    join_table=task["join_table"], user_intent=task["user_intent"],
                    task_type=task["task_type"], session_id=sid, base_dir=base_dir,
                    data_filename=data_filename, tau=tau, beta=beta,
                    provider=args.provider,
                )
                results.append({"tau": tau, "beta": beta, "join_table": task["join_table"], "session_id": sid, "returncode": ret})

                baseline = output.get("baseline_metric") if output else None
                augmented = output.get("augmented_metric") if output else None
                augment_results = output.get("augment_results", []) if output else []
                selected_candidates = [r.get("candidate_table", "?") for r in augment_results]
                augment_cols = {r.get("candidate_table", "?"): r.get("selected_augment_columns", []) for r in augment_results}
                improvement = (augmented - baseline) if (baseline is not None and augmented is not None) else None

                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "tau": tau, "beta": beta, "join_table": task["join_table"], "task_type": task["task_type"],
                    "session_id": sid, "returncode": ret,
                    "selected_candidate_tables": selected_candidates,
                    "augment_columns_per_candidate": augment_cols,
                    "baseline_metric": float(baseline) if baseline is not None else None,
                    "augmented_metric": float(augmented) if augmented is not None else None,
                    "improvement": float(improvement) if improvement is not None else None,
                    "metric_name": output.get("metric_name", "r2_score") if output else None,
                }
                if ret != 0 and output:
                    log_entry["error"] = output.get("error")
                append_experiment_log(log_path, log_entry)

                session_id += 1

    return results

def load_log_entries(log_path: Path) -> list[dict]:
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def replace_entry_in_log(log_path: Path, tau: float, beta: float, join_table: str, new_entry: dict) -> None:
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                if e.get("tau") == tau and e.get("beta") == beta and e.get("join_table") == join_table:
                    entries.append(new_entry)
                else:
                    entries.append(e)
            except json.JSONDecodeError:
                pass
    with open(log_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def rerun_failed_tasks(log_path, args):
    """"""
    entries = load_log_entries(log_path)
    failed = [e for e in entries if e.get("returncode", 0) != 0 or "error" in e]
    print(f"Found {len(failed)} failed tasks\n")

    for i, e in enumerate(failed):
        tau, beta, join_table = e["tau"], e["beta"], e["join_table"]
        session_id = e.get("session_id", "???")
        task = next((t for t in TASKS if t["join_table"] == join_table), None)
        if not task:
            continue
        base_dir = f"perturbed_{tau}_{beta}"
        if not (_PROJECT / base_dir).exists():
            print(f"Skip {base_dir} (not found)")
            continue

        print(f"[{i+1}/{len(failed)}] τ={tau} β={beta} {join_table} sess={session_id}")
        update_perturbation_yaml(tau, beta)

        ret, output = run_orchestrator_inprocess(
            join_table=task["join_table"],
            user_intent=task["user_intent"],
            task_type=task["task_type"],
            session_id=session_id,
            base_dir=str(_PROJECT / base_dir),
            data_filename=args.data_filename,
            tau=tau,
            beta=beta,
            provider=args.provider,
            session_checked_dir=args.session_checked_dir,
            config_history_file=args.config_history_file,
        )
        summary = _summarize_outer_output(output if isinstance(output, dict) else {})
        baseline = summary["baseline_metric"]
        augmented = summary["augmented_metric"]
        selected = summary["selected_candidate_tables"]
        augment_cols = summary["augment_columns_per_candidate"]
        improvement = summary["improvement"]

        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "tau": tau, "beta": beta, "join_table": join_table, "task_type": task["task_type"],
            "session_id": session_id, "returncode": ret,
            "selected_candidate_tables": selected,
            "augment_columns_per_candidate": augment_cols,
            "baseline_metric": float(baseline) if baseline is not None else None,
            "augmented_metric": float(augmented) if augmented is not None else None,
            "improvement": float(improvement) if improvement is not None else None,
            "metric_name": summary["metric_name"],
        }
        if ret != 0 and output:
            new_entry["error"] = output.get("error")

        replace_entry_in_log(log_path, tau, beta, join_table, new_entry)
        print(f"  -> {'ok' if ret == 0 else 'FAILED'} (replaced in log)")

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--skip-grid", action="store_true", help="")
    parser.add_argument("--grid-only", action="store_true", help="")
    parser.add_argument("--session-start", type=int, default=1)
    parser.add_argument("--data-filename", type=str, default="rows.csv")
    parser.add_argument("--tau-only", type=float, nargs="+")
    parser.add_argument("--beta-only", type=float, nargs="+")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start-from-session", type=int, default=None,
                    help="")
    parser.add_argument("--provider", type=str, default=None, choices=["local", "openai"],
                    help="")
    parser.add_argument("--session-checked-dir", type=str, default=None,
                        help="")
    parser.add_argument("--config-history-file", type=str, default=None,
                        help="Override output.config_history_file for outer orchestrator")
    parser.add_argument("--rerun-failed", action="store_true", help="")
    parser.add_argument("--tables", "-t", nargs="+", help="Only run these join_table names (e.g. -t Taxi-Chicago Traffic_Chicago Taxi-NYC Environment_NYC)")
    parser.add_argument("--debug-table-selection", action="store_true",
                        help="Enable Table Selection debug: print full_text preview and parsed relevant_tables (and set DEBUG_TABLE_SELECTION=1)")
    parser.add_argument("--debug-llm", action="store_true",
                        help="Enable LiteLLM debug (litellm._turn_on_debug()) to see LLM request/response")

    args = parser.parse_args()

    if args.debug_table_selection:
        os.environ["DEBUG_TABLE_SELECTION"] = "1"
    if args.debug_llm:
        litellm._turn_on_debug()

    tasks_to_run = [t for t in TASKS if t["join_table"] in set(args.tables)] if args.tables else TASKS

    if args.rerun_failed:
        rerun_failed_tasks(log_path, args)
        return 0

    tau_list = args.tau_only if args.tau_only else TAU_VALUES
    beta_list = args.beta_only if args.beta_only else BETA_VALUES
    cfg = load_config()
    default_log = cfg.get("output", {}).get("experiment_log_file", "experiments/experiment_log_llm.json")
    log_path = Path(args.log_file) if args.log_file else _PROJECT / default_log
    verbose = not args.quiet

    session_id = args.session_start
    exp_results = []

    for tau in tau_list:
        for beta in beta_list:
            base_dir = f"perturbed_{tau}_{beta}"

            if not args.skip_grid:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Generating perturbed_{tau}_{beta}")
                    print("="*60)
                try:
                    from benchmark_perturbation.benchmark_perturbation import run_full_pipeline
                    run_full_pipeline(threshold=tau, beta=beta, tables=args.tables if args.tables else None)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue
            else:
                if not (_PROJECT / base_dir).exists():
                    print(f"Skip {base_dir} (not found)")
                    continue

            if not args.grid_only:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Running experiments τ={tau}, β={beta} -> {base_dir}")
                    print("="*60)
                update_perturbation_yaml(tau, beta)

                total = len(tau_list) * len(beta_list) * len(tasks_to_run)
                for task in tasks_to_run:
                    if not hasattr(main, '_current_session'):
                        main._current_session = 0
                    main._current_session += 1
                    if args.start_from_session is not None and main._current_session < args.start_from_session:
                        session_id += 1
                        continue
                    sid = f"{session_id:03d}"
                    if verbose:
                        print(f"\n--- [{session_id}/{total}] {task['join_table']} ({task['task_type']}) sess={sid} ---")
                    ret, output = run_orchestrator_inprocess(
                        join_table=task["join_table"], user_intent=task["user_intent"],
                        task_type=task["task_type"], session_id=sid, base_dir=base_dir,
                        data_filename=args.data_filename, tau=tau, beta=beta,
                        provider=args.provider, session_checked_dir=args.session_checked_dir,
                        config_history_file=args.config_history_file,
                    )
                    exp_results.append({"tau": tau, "beta": beta, "join_table": task["join_table"], "session_id": sid, "returncode": ret})
                    summary = _summarize_outer_output(output if isinstance(output, dict) else {})
                    baseline = summary["baseline_metric"]
                    augmented = summary["augmented_metric"]
                    selected_candidates = summary["selected_candidate_tables"]
                    augment_cols = summary["augment_columns_per_candidate"]
                    improvement = summary["improvement"]

                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "tau": tau, "beta": beta, "join_table": task["join_table"], "task_type": task["task_type"],
                        "session_id": sid, "returncode": ret,
                        "selected_candidate_tables": selected_candidates,
                        "augment_columns_per_candidate": augment_cols,
                        "baseline_metric": float(baseline) if baseline is not None else None,
                        "augmented_metric": float(augmented) if augmented is not None else None,
                        "improvement": float(improvement) if improvement is not None else None,
                        "metric_name": summary["metric_name"],
                    }
                    if ret != 0 and output:
                        log_entry["error"] = output.get("error")
                    append_experiment_log(log_path, log_entry)

                    session_id += 1

    if not args.grid_only and exp_results:
        ok = sum(1 for r in exp_results if r["returncode"] == 0)
        print(f"\nDone: {ok}/{len(exp_results)} succeeded, Log: {log_path}")
        for r in exp_results:
            if r["returncode"] != 0:
                print(f"  Failed: τ={r['tau']} β={r['beta']} {r['join_table']} sess={r['session_id']}")


if __name__ == "__main__":
    main()
