#!/usr/bin/env python3
"""
完整 benchmark：先生成 perturbed 文件，再跑 pipeline 实验。
1. Phase 1（可选）：对每个 (τ, β) 运行 run_full_pipeline 生成 perturbed_{τ}_{β}
2. Phase 2：对每个 (τ, β) 跑 8 个任务，记录到 experiment_log.json

用法:
  python experiments/run_perturbation_experiments.py                    # 生成 + 实验
  python experiments/run_perturbation_experiments.py --skip-grid       # 仅实验（跳过生成，用已有 perturbed）
  python experiments/run_perturbation_experiments.py --grid-only       # 仅生成，不跑实验
"""
import litellm
# litellm._turn_on_debug()
import sys
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

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


def run_grid(verbose: bool = True):
    """Phase 1: 生成 perturbed 文件。"""
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
                run_full_pipeline(threshold=tau, beta=beta)
                results.append({"tau": tau, "beta": beta, "status": "ok"})
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"tau": tau, "beta": beta, "status": "error", "error": str(e)})
    return results


def run_orchestrator_inprocess(
    join_table: str, user_intent: str, task_type: str, session_id: str,
    base_dir: str, data_filename: str = "rows.csv", tau: float = None, beta: float = None,
):
    """Run orchestrator in-process. Returns (returncode, output_dict)."""
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

    config = AgentPipelineConfig(config_dict=cfg_dict)

    # 为 dimension 确认自动填充预设值
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

    from orchestrator import run_orchestrator
    try:
        output = asyncio.run(run_orchestrator(
            config=config, user_intent=user_intent, session_id=session_id, task_type=task_type,
        ))
        return 0, output
    except Exception as e:
        return 1, {"error": str(e)}
    finally:
        builtins.input = _orig_input


def append_experiment_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_experiments(tau_list, beta_list, data_filename, session_start, log_path, verbose=True):
    """Phase 2: 对每个 (τ, β) 跑 8 个任务。"""
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


def main():
    parser = argparse.ArgumentParser(description="完整 benchmark：每个 (τ,β) 先生成再跑实验")
    parser.add_argument("--skip-grid", action="store_true", help="跳过生成，直接用已有 perturbed（需每个 pair 都存在）")
    parser.add_argument("--grid-only", action="store_true", help="仅生成，不跑实验")
    parser.add_argument("--session-start", type=int, default=1)
    parser.add_argument("--data-filename", type=str, default="rows.csv")
    parser.add_argument("--tau-only", type=float, nargs="+")
    parser.add_argument("--beta-only", type=float, nargs="+")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tau_list = args.tau_only if args.tau_only else TAU_VALUES
    beta_list = args.beta_only if args.beta_only else BETA_VALUES
    log_path = Path(args.log_file) if args.log_file else _PROJECT / "experiments" / "experiment_log_llm.json"
    verbose = not args.quiet

    session_id = args.session_start
    exp_results = []

    for tau in tau_list:
        for beta in beta_list:
            base_dir = f"perturbed_{tau}_{beta}"

            # 本 pair：先生成 perturbed
            if not args.skip_grid:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"生成 perturbed_{tau}_{beta}")
                    print("="*60)
                try:
                    from benchmark_perturbation.benchmark_perturbation import run_full_pipeline
                    run_full_pipeline(threshold=tau, beta=beta)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue
            else:
                if not (_PROJECT / base_dir).exists():
                    print(f"Skip {base_dir} (not found)")
                    continue

            # 本 pair：再跑 8 个任务
            if not args.grid_only:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"跑实验 τ={tau}, β={beta} -> {base_dir}")
                    print("="*60)
                update_perturbation_yaml(tau, beta)

                total = len(tau_list) * len(beta_list) * len(TASKS)
                for task in TASKS:
                    sid = f"{session_id:03d}"
                    if verbose:
                        print(f"\n--- [{session_id}/{total}] {task['join_table']} ({task['task_type']}) sess={sid} ---")
                    ret, output = run_orchestrator_inprocess(
                        join_table=task["join_table"], user_intent=task["user_intent"],
                        task_type=task["task_type"], session_id=sid, base_dir=base_dir,
                        data_filename=args.data_filename, tau=tau, beta=beta,
                    )
                    exp_results.append({"tau": tau, "beta": beta, "join_table": task["join_table"], "session_id": sid, "returncode": ret})

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

    if not args.grid_only and exp_results:
        ok = sum(1 for r in exp_results if r["returncode"] == 0)
        print(f"\n实验完成: {ok}/{len(exp_results)} 成功, Log: {log_path}")
        for r in exp_results:
            if r["returncode"] != 0:
                print(f"  Failed: τ={r['tau']} β={r['beta']} {r['join_table']} sess={r['session_id']}")


if __name__ == "__main__":
    main()
