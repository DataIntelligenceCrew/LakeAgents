#!/usr/bin/env python3
"""
对指定的实验 NDJSON 文件（每行一条记录）按 join_table 分组汇总：
  - 平均 improvement 及样本标准差（stdev，至少 2 个有效点才计算）
  - total_token_usage 的 input_tokens / output_tokens 分别平均及 stdev
  - 平均 total_duration_seconds 及 stdev

仅当 error 字段为 null 或缺失时计入成功样本；带 error 的记录不参与计数与平均。
若某 join_table 下成功样本数为 0，则标记为 failure。

默认只统计扰动网格 tau×beta ∈ {0.1,0.5,0.9}² 的行（字段 tau、beta）。
若缺少 tau/beta 或取值不在网格内则整行跳过。可用 --no-tau-beta-filter 关闭。

用法（直接传实验 NDJSON 路径，可多个）:
  python scripts/summarize_experiment_metrics.py experiments/00_mini_full_experiment.json
  python scripts/summarize_experiment_metrics.py experiments/00_mini_full_experiment.json -o experiments/summary.csv
  python scripts/summarize_experiment_metrics.py experiments/00_qwen_full_experiment.json experiments/02_qwen_augment_experiment.json

未传任何 JSON 路径时，退化为在 --experiments-dir 下用 --glob 批量处理（与旧行为一致）。

指定 **-o/--output 路径** 即按 CSV 写入该文件（每行一个 join_table），并同时将同一份 CSV 打印到终端。
默认（不写 -o、不加 --json）仍为易读文本摘要。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _is_success(row: dict[str, Any]) -> bool:
    err = row.get("error")
    return err is None or err == ""


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _token_pair(row: dict[str, Any]) -> tuple[int | None, int | None]:
    tu = row.get("total_token_usage")
    if not isinstance(tu, dict):
        return None, None
    inp = tu.get("input_tokens")
    out = tu.get("output_tokens")
    try:
        inp_i = int(inp) if inp is not None else None
    except (TypeError, ValueError):
        inp_i = None
    try:
        out_i = int(out) if out is not None else None
    except (TypeError, ValueError):
        out_i = None
    return inp_i, out_i


def _duration(row: dict[str, Any]) -> float | None:
    v = row.get("total_duration_seconds")
    f = _safe_float(v)
    return f


def _close_to_any_level(x: float, levels: tuple[float, ...], *, eps: float = 1e-6) -> bool:
    return any(abs(x - lv) <= eps for lv in levels)


def _row_matches_tau_beta_grid(
    row: dict[str, Any],
    levels: tuple[float, ...],
) -> bool:
    """若 tau、beta 均落在 levels（各与某一档近似相等），则纳入汇总。"""
    tau = _safe_float(row.get("tau"))
    beta = _safe_float(row.get("beta"))
    if tau is None or beta is None:
        return False
    return _close_to_any_level(tau, levels) and _close_to_any_level(beta, levels)


def _sample_stdev(values: list[float] | list[int]) -> float | None:
    """样本标准差；少于 2 个点时无定义，返回 None。"""
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _bootstrap_mean_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Bootstrap percentile CI for the mean. Returns (low, high)."""
    if not values or len(values) < 2:
        return None
    if resamples <= 0:
        return None
    conf = float(confidence)
    if conf <= 0.0 or conf >= 1.0:
        return None

    rng = random.Random(int(seed))
    n = len(values)
    means: list[float] = []
    for _ in range(int(resamples)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(float(statistics.mean(sample)))
    means.sort()
    alpha = 1.0 - conf
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    lo_i = int((len(means) - 1) * lo_q)
    hi_i = int((len(means) - 1) * hi_q)
    lo_i = max(0, min(lo_i, len(means) - 1))
    hi_i = max(0, min(hi_i, len(means) - 1))
    return means[lo_i], means[hi_i]


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return rows


def summarize_file(
    path: Path,
    task_key: str = "join_table",
    *,
    tau_beta_levels: tuple[float, ...] | None = (0.1, 0.5, 0.9),
    improvement_ci_confidence: float = 0.95,
    improvement_ci_resamples: int = 2000,
    improvement_ci_seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """返回 {task_id: 汇总字典}。"""
    rows = load_ndjson(path)
    if tau_beta_levels:
        rows = [r for r in rows if _row_matches_tau_beta_grid(r, tau_beta_levels)]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tid = row.get(task_key)
        if tid is None:
            tid = "<missing_task_key>"
        by_task[str(tid)].append(row)

    out: dict[str, dict[str, Any]] = {}
    for task_id, task_rows in sorted(by_task.items()):
        total = len(task_rows)
        ok_rows = [r for r in task_rows if _is_success(r)]
        n_ok = len(ok_rows)
        n_err = total - n_ok

        if n_ok == 0:
            out[task_id] = {
                "status": "failure",
                "n_total": total,
                "n_error_skipped": n_err,
                "n_counted": 0,
                "average_improvement": None,
                "stdev_improvement": None,
                "average_total_token_input": None,
                "stdev_total_token_input": None,
                "average_total_token_output": None,
                "stdev_total_token_output": None,
                "average_total_duration_seconds": None,
                "stdev_total_duration_seconds": None,
            }
            continue

        improvements = [
            float(v)
            for r in ok_rows
            if (v := _safe_float(r.get("improvement"))) is not None
        ]
        improvement_ci = _bootstrap_mean_ci(
            improvements,
            confidence=improvement_ci_confidence,
            resamples=improvement_ci_resamples,
            seed=improvement_ci_seed,
        )
        inputs: list[int] = []
        outputs: list[int] = []
        for r in ok_rows:
            inp, outp = _token_pair(r)
            if inp is not None:
                inputs.append(inp)
            if outp is not None:
                outputs.append(outp)
        durations = [d for r in ok_rows if (d := _duration(r)) is not None]

        out[task_id] = {
            "status": "ok",
            "n_total": total,
            "n_error_skipped": n_err,
            "n_counted": n_ok,
            "average_improvement": statistics.mean(improvements)
            if improvements
            else None,
            "stdev_improvement": _sample_stdev(improvements) if improvements else None,
            "improvement_ci_low": improvement_ci[0] if improvement_ci else None,
            "improvement_ci_high": improvement_ci[1] if improvement_ci else None,
            "n_used_for_improvement": len(improvements),
            "average_total_token_input": statistics.mean(inputs) if inputs else None,
            "stdev_total_token_input": _sample_stdev(inputs) if inputs else None,
            "n_used_for_input_tokens": len(inputs),
            "average_total_token_output": statistics.mean(outputs) if outputs else None,
            "stdev_total_token_output": _sample_stdev(outputs) if outputs else None,
            "n_used_for_output_tokens": len(outputs),
            "average_total_duration_seconds": statistics.mean(durations)
            if durations
            else None,
            "stdev_total_duration_seconds": _sample_stdev(durations)
            if durations
            else None,
            "n_used_for_duration": len(durations),
        }
    return out


CSV_COLUMNS = [
    "join_table",
    "status",
    "n_total",
    "n_error_skipped",
    "n_counted",
    "average_improvement",
    "stdev_improvement",
    "improvement_ci_low",
    "improvement_ci_high",
    "average_total_token_input",
    "stdev_total_token_input",
    "average_total_token_output",
    "stdev_total_token_output",
    "average_total_duration_seconds",
    "stdev_total_duration_seconds",
    "n_used_for_improvement",
    "n_used_for_input_tokens",
    "n_used_for_output_tokens",
    "n_used_for_duration",
]


def emit_csv(
    report: dict[str, dict[str, dict[str, Any]]],
    out_stream,
) -> None:
    w = csv.DictWriter(out_stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    for _key, bundle in report.items():
        for join_table, s in sorted(bundle["by_join_table"].items()):
            row: dict[str, Any] = {
                "join_table": join_table,
                "status": s.get("status", ""),
                "n_total": s.get("n_total", ""),
                "n_error_skipped": s.get("n_error_skipped", ""),
                "n_counted": s.get("n_counted", ""),
                "average_improvement": s.get("average_improvement", ""),
                "stdev_improvement": s.get("stdev_improvement", ""),
                "improvement_ci_low": s.get("improvement_ci_low", ""),
                "improvement_ci_high": s.get("improvement_ci_high", ""),
                "average_total_token_input": s.get("average_total_token_input", ""),
                "stdev_total_token_input": s.get("stdev_total_token_input", ""),
                "average_total_token_output": s.get("average_total_token_output", ""),
                "stdev_total_token_output": s.get("stdev_total_token_output", ""),
                "average_total_duration_seconds": s.get(
                    "average_total_duration_seconds", ""
                ),
                "stdev_total_duration_seconds": s.get(
                    "stdev_total_duration_seconds", ""
                ),
                "n_used_for_improvement": s.get("n_used_for_improvement", ""),
                "n_used_for_input_tokens": s.get("n_used_for_input_tokens", ""),
                "n_used_for_output_tokens": s.get("n_used_for_output_tokens", ""),
                "n_used_for_duration": s.get("n_used_for_duration", ""),
            }
            for k, v in list(row.items()):
                if v is None:
                    row[k] = ""
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "experiment_json",
        nargs="*",
        type=str,
        metavar="EXPERIMENT.json",
        help="实验 NDJSON 路径（可多个），例如 experiments/00_mini_full_experiment.json",
    )
    ap.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "experiments",
        help="未传实验 JSON 路径时：在此目录下按 --glob 查找文件",
    )
    ap.add_argument(
        "--glob",
        default="0[0-2]_*.json",
        help="未传实验 JSON 路径时：匹配文件名（相对 experiments-dir）",
    )
    ap.add_argument(
        "--task-key",
        default="join_table",
        help="作为「不同 task」分组用的 JSON 字段名，默认 join_table",
    )
    ap.add_argument(
        "--no-tau-beta-filter",
        action="store_true",
        help="不过滤 tau/beta；默认只保留 tau、beta ∈ 指定离散档（见 --tau-beta-levels）",
    )
    ap.add_argument(
        "--tau-beta-levels",
        type=str,
        default="0.1,0.5,0.9",
        help="逗号分隔的 perturb 档位（默认 0.1,0.5,0.9）；仅保留 tau 与 beta 均落在这些值之一",
    )
    ap.add_argument(
        "--improvement-ci-confidence",
        type=float,
        default=0.95,
        help="improvement 均值的 bootstrap 置信水平（默认 0.95）",
    )
    ap.add_argument(
        "--improvement-ci-resamples",
        type=int,
        default=2000,
        help="improvement 均值 bootstrap 重采样次数（默认 2000）",
    )
    ap.add_argument(
        "--improvement-ci-seed",
        type=int,
        default=0,
        help="improvement 均值 bootstrap 随机种子（默认 0）",
    )
    out_group = ap.add_mutually_exclusive_group()
    out_group.add_argument(
        "--json",
        action="store_true",
        help="将汇总结果以 JSON 打印到 stdout",
    )
    out_group.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="OUT.csv",
        help="将汇总结果以 CSV 写入该文件，并同时将同一份 CSV 打印到终端",
    )
    args = ap.parse_args()

    tau_beta_levels: tuple[float, ...] | None
    if args.no_tau_beta_filter:
        tau_beta_levels = None
    else:
        try:
            tau_beta_levels = tuple(
                float(x.strip())
                for x in args.tau_beta_levels.split(",")
                if x.strip()
            )
        except ValueError as e:
            print(f"无效 --tau-beta-levels: {args.tau_beta_levels!r} ({e})", file=sys.stderr)
            return 1
        if not tau_beta_levels:
            print("--tau-beta-levels 解析后为空", file=sys.stderr)
            return 1

    if args.experiment_json:
        paths: list[Path] = []
        for raw in args.experiment_json:
            p = Path(raw).expanduser()
            if not p.is_file():
                print(f"文件不存在或不是普通文件: {p.resolve()}", file=sys.stderr)
                return 1
            paths.append(p.resolve())
    else:
        exp_dir: Path = args.experiments_dir
        if not exp_dir.is_dir():
            print(f"目录不存在: {exp_dir}", file=sys.stderr)
            return 1
        paths = sorted(exp_dir.glob(args.glob))
        if not paths:
            print(
                f"未找到匹配文件: {exp_dir}/{args.glob}\n"
                "请直接传入实验 JSON 路径，例如:\n"
                "  python scripts/summarize_experiment_metrics.py experiments/00_mini_full_experiment.json",
                file=sys.stderr,
            )
            return 1

    report: dict[str, dict[str, dict[str, Any]]] = {}
    for p in paths:
        report[str(p)] = {
            "experiment_file": str(p),
            "by_join_table": summarize_file(
                p,
                task_key=args.task_key,
                tau_beta_levels=tau_beta_levels,
                improvement_ci_confidence=args.improvement_ci_confidence,
                improvement_ci_resamples=args.improvement_ci_resamples,
                improvement_ci_seed=args.improvement_ci_seed,
            ),
        }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.output is not None:
        buf = io.StringIO()
        emit_csv(report, buf)
        text = buf.getvalue()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        sys.stdout.write(text)
        return 0

    for fname, bundle in report.items():
        tasks = bundle["by_join_table"]
        print(f"=== {bundle['experiment_file']} ===")
        for task_id, s in tasks.items():
            if s["status"] == "failure":
                print(f"  [{task_id}] failure (n_total={s['n_total']}, 全部带 error 或无可计样本)")
                continue
            print(f"  [{task_id}]")
            print(f"    n_total={s['n_total']}, n_error_skipped={s['n_error_skipped']}, n_counted={s['n_counted']}")
            ai = s["average_improvement"]
            si = s.get("stdev_improvement")
            ci_lo = s.get("improvement_ci_low")
            ci_hi = s.get("improvement_ci_high")
            print(
                f"    average_improvement={ai}, stdev_improvement={si}, improvement_mean_CI=({ci_lo}, {ci_hi})"
                + (f" (over {s['n_used_for_improvement']} rows with numeric improvement)" if s.get("n_used_for_improvement") != s["n_counted"] else "")
            )
            ti = s["average_total_token_input"]
            to = s["average_total_token_output"]
            sti = s.get("stdev_total_token_input")
            sto = s.get("stdev_total_token_output")
            print(
                f"    average_total_token_input={ti}, stdev_total_token_input={sti}, "
                f"average_total_token_output={to}, stdev_total_token_output={sto}"
            )
            if (
                s.get("n_used_for_input_tokens") != s["n_counted"]
                or s.get("n_used_for_output_tokens") != s["n_counted"]
            ):
                print(
                    f"      (input averaged over {s.get('n_used_for_input_tokens')}, "
                    f"output over {s.get('n_used_for_output_tokens')} rows with valid total_token_usage)"
                )
            td = s["average_total_duration_seconds"]
            std = s.get("stdev_total_duration_seconds")
            print(
                f"    average_total_duration_seconds={td}, stdev_total_duration_seconds={std}"
                + (
                    f" (over {s['n_used_for_duration']} rows)"
                    if s.get("n_used_for_duration") != s["n_counted"]
                    else ""
                )
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
