#!/usr/bin/env python3
"""
只重跑 experiment log 中失败的任务，并把结果写回 log（替换对应行，不追加）。
"""
import json
import argparse
from pathlib import Path
from datetime import datetime

_PROJECT = Path(__file__).resolve().parent.parent


def load_entries(log_path: Path) -> list[dict]:
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


def has_error(entry: dict) -> bool:
    return entry.get("returncode", 0) != 0 or "error" in entry


def replace_entry_in_log(log_path: Path, tau: float, beta: float, join_table: str, new_entry: dict) -> None:
    """用 new_entry 替换 log 中 (tau, beta, join_table) 匹配的那一行。"""
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


def main():
    parser = argparse.ArgumentParser(description="Rerun failed tasks and replace entries in experiment log")
    parser.add_argument("--log-file", type=str, required=True, help="Path to experiment log (JSONL)")
    parser.add_argument("--provider", type=str, default=None, choices=["local", "openai"])
    parser.add_argument("--session-checked-dir", type=str, default=None)
    parser.add_argument("--data-filename", type=str, default="rows.csv")
    parser.add_argument("--dry-run", action="store_true", help="Only list failed tasks, do not run")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = _PROJECT / log_path
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return 1

    sys.path.insert(0, str(_PROJECT))
    from experiments.run_perturbation_experiments import (
        TASKS,
        run_orchestrator_inprocess,
        update_perturbation_yaml,
    )

    entries = load_entries(log_path)
    failed = [e for e in entries if has_error(e)]
    print(f"Found {len(failed)} failed tasks in {len(entries)} total entries\n")

    if args.dry_run:
        for e in failed:
            print(f"  τ={e['tau']} β={e['beta']} {e['join_table']} sess={e.get('session_id')}")
        return 0

    for i, e in enumerate(failed):
        tau = e["tau"]
        beta = e["beta"]
        join_table = e["join_table"]
        session_id = e.get("session_id", "???")
        task = next((t for t in TASKS if t["join_table"] == join_table), None)
        if not task:
            print(f"Skip {join_table}: no task config")
            continue

        base_dir = f"perturbed_{tau}_{beta}"
        if not (_PROJECT / base_dir).exists():
            print(f"Skip τ={tau} β={beta} {join_table}: {base_dir} not found")
            continue

        print(f"\n[{i+1}/{len(failed)}] Rerunning τ={tau} β={beta} {join_table} sess={session_id} ...")
        update_perturbation_yaml(tau, beta)

        ret, output = run_orchestrator_inprocess(
            join_table=join_table,
            user_intent=task["user_intent"],
            task_type=task["task_type"],
            session_id=session_id,
            base_dir=str(_PROJECT / base_dir),
            data_filename=args.data_filename,
            tau=tau,
            beta=beta,
            provider=args.provider,
            session_checked_dir=args.session_checked_dir,
        )

        baseline = output.get("baseline_metric") if output else None
        augmented = output.get("augmented_metric") if output else None
        augment_results = output.get("augment_results", []) if output else []
        selected = [r.get("candidate_table", "?") for r in augment_results]
        augment_cols = {r.get("candidate_table", "?"): r.get("selected_augment_columns", []) for r in augment_results}
        improvement = (augmented - baseline) if (baseline is not None and augmented is not None) else None

        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "tau": tau,
            "beta": beta,
            "join_table": join_table,
            "task_type": task["task_type"],
            "session_id": session_id,
            "returncode": ret,
            "selected_candidate_tables": selected,
            "augment_columns_per_candidate": augment_cols,
            "baseline_metric": float(baseline) if baseline is not None else None,
            "augmented_metric": float(augmented) if augmented is not None else None,
            "improvement": float(improvement) if improvement is not None else None,
            "metric_name": output.get("metric_name", "r2_score") if output else None,
        }
        if ret != 0 and output:
            new_entry["error"] = output.get("error")

        replace_entry_in_log(log_path, tau, beta, join_table, new_entry)
        status = "ok" if ret == 0 else f"FAILED: {new_entry.get('error', '')[:50]}"
        print(f"  -> {status} (replaced in log)")

    print(f"\nDone. Log updated: {log_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())