#!/usr/bin/env python3
"""
比较 baseline 预测效果：
对 rows_4omini.csv、rows.csv、rows_llm.csv 分别训练模型预测 target_column，
输出 R²（回归）或 F1（分类）并保存到 CSV。
"""
import sys
from pathlib import Path
import yaml
import json
import contextlib
import io

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from Classification_regression import preprocess_data, run_classification_task, run_regression_task


def load_table_config() -> dict:
    path = _PROJECT / "configs" / "perturbation.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("tables", {})


TASK_TYPES = {
    "COVID-Chicago": "regression",
    "Demo-Chicago": "regression",
    "Economic-Chicago": "regression",
    "Education-Chicago": "classification",
    "COVID-NYC": "regression",
    "Demo-NYC": "classification",
    "Economic-NYC": "classification",
    "Education-NYC": "classification",
}


def run_prediction(table_name: str, base_dir: str, rows_file: str, n_rows: int = 500) -> tuple:
    table_cfg = load_table_config().get(table_name, {})
    target_column = table_cfg.get("target_column", "")
    join_columns = table_cfg.get("join_columns", [])
    if isinstance(join_columns, str):
        join_columns = [join_columns]

    if not target_column:
        return None, None, f"No target_column for {table_name}"

    task_type = TASK_TYPES.get(table_name, "regression")
    metric_name = "r2_score" if task_type == "regression" else "f1_score"

    path = Path(base_dir) / table_name / rows_file
    if not path.exists():
        return None, metric_name, f"File not found: {path}"

    import pandas as pd
    df = pd.read_csv(path, low_memory=False, nrows=n_rows)

    if target_column not in df.columns:
        return None, metric_name, f"Target column '{target_column}' not in {rows_file}"

  
    to_drop = [c for c in join_columns if c in df.columns and c != target_column]
    if to_drop:
        df = df.drop(columns=to_drop)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = preprocess_data(df, target_column, task_type)
        if result[0] is None:
            return None, metric_name, "Preprocess failed (insufficient data or no valid samples)"
        X, y_encoded, target_encoder, scaler = result
        if task_type == "classification":
            metrics = run_classification_task(X, y_encoded, target_encoder)
            metric = metrics["f1_score"]
        else:
            metrics = run_regression_task(X, y_encoded)
            metric = metrics["r2_score"]
        return float(metric), metric_name, None
    except Exception as e:
        return None, metric_name, str(e)


def main():
    import argparse
    import csv
    parser = argparse.ArgumentParser(description="Compare prediction: rows_4omini, rows.csv, rows_qwen.csv (first n rows)")
    parser.add_argument("--base-dir", type=str, default="query_table")
    parser.add_argument("--n-rows", type=int, default=500)
    parser.add_argument("--tables", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    files_to_compare = [
        ("rows_4omini", "rows_4omini.csv"),
        ("rows", "rows.csv"),
        ("rows_qwen", "rows_qwen.csv"),
    ]

    base_dir = _PROJECT / args.base_dir
    tables_cfg = load_table_config()
    tables = args.tables or list(tables_cfg.keys())

    results = []
    print(f"{'Table':<20} {'Task Type':<14} {'rows_4omini':<12} {'rows.csv':<12} {'rows_qwen':<12} {'Status'}")
    print("-" * 82)

    for table_name in tables:
        if table_name not in tables_cfg:
            print(f"{table_name:<20} (skip - not in config)")
            continue

        task_type = TASK_TYPES.get(table_name, "regression")
        metric_name = "r2_score" if task_type == "regression" else "f1_score"

        metrics = {}
        for key, fname in files_to_compare:
            m, _, err = run_prediction(table_name, str(base_dir), fname, args.n_rows)
            metrics[key] = (m, err)

        m_4omini, err_4omini = metrics["rows_4omini"]
        m_rows, err_rows = metrics["rows"]
        m_qwen, err_qwen = metrics["rows_qwen"]

        m_4omini_str = f"{m_4omini:.4f}" if m_4omini is not None else "N/A"
        m_rows_str = f"{m_rows:.4f}" if m_rows is not None else "N/A"
        m_qwen_str = f"{m_qwen:.4f}" if m_qwen is not None else "N/A"

        has_err = any(err for _, err in metrics.values())
        status = "error" if has_err else "ok"
        if err_4omini:
            status = f"4omini: {str(err_4omini)[:40]}"
        elif err_rows:
            status = f"rows: {str(err_rows)[:40]}"
        elif err_qwen:
            status = f"qwen: {str(err_qwen)[:40]}"

        entry = {
            "table": table_name,
            "task_type": task_type,
            "metric_name": metric_name,
            "n_rows": args.n_rows,
            "metric_rows_4omini": m_4omini,
            "metric_rows": m_rows,
            "metric_rows_qwen": m_qwen,
            "error_rows_4omini": err_4omini,
            "error_rows": err_rows,
            "error_rows_qwen": err_qwen,
        }
        results.append(entry)
        print(f"{table_name:<20} {task_type:<14} {m_4omini_str:<12} {m_rows_str:<12} {m_qwen_str:<12} {status}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")

        csv_path = out_path.with_suffix(".csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Table", "Task Type", "rows_4omini", "rows.csv", "rows_qwen", "Status"])
            for r in results:
                m_4omini = r.get("metric_rows_4omini")
                m_rows = r.get("metric_rows")
                m_qwen = r.get("metric_rows_qwen")
                m_4omini_str = f"{m_4omini:.4f}" if m_4omini is not None else "N/A"
                m_rows_str = f"{m_rows:.4f}" if m_rows is not None else "N/A"
                m_qwen_str = f"{m_qwen:.4f}" if m_qwen is not None else "N/A"
                has_err = any(r.get(f"error_{k}") for k in ["rows_4omini", "rows", "rows_qwen"])
                status = "error" if has_err else "ok"
                writer.writerow([r["table"], r["task_type"], m_4omini_str, m_rows_str, m_qwen_str, status])
        print(f"CSV saved to {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())