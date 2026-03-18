#!/usr/bin/env python3
"""
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TASKS_ORDER = [
    "COVID-Chicago", "COVID-NYC", "Demo-Chicago", "Demo-NYC",
    "Economic-Chicago", "Economic-NYC", "Education-Chicago", "Education-NYC",
]


def load_entries(log_path: str) -> list[dict]:
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


def build_improvement_matrix(entries: list[dict], join_table: str) -> tuple[np.ndarray, list, list]:
    """"""
    tau_set = sorted(set(e["tau"] for e in entries))
    beta_set = sorted(set(e["beta"] for e in entries))
    tau_to_idx = {t: i for i, t in enumerate(tau_set)}
    beta_to_idx = {b: j for j, b in enumerate(beta_set)}

    mat = np.full((len(tau_set), len(beta_set)), np.nan)
    for e in entries:
        if e.get("join_table") != join_table:
            continue
        i = tau_to_idx.get(e["tau"])
        j = beta_to_idx.get(e["beta"])
        imp = e.get("improvement")
        if i is not None and j is not None and imp is not None:
            mat[i, j] = float(imp)

    return mat, tau_set, beta_set


def main():
    parser = argparse.ArgumentParser(description="Plot 2×4 improvement heatmaps from experiment log")
    parser.add_argument("log_file", type=str, default="experiments/experiment_log_mini.json", nargs="?",
                        help="Path to experiment log (JSONL)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output figure path")
    parser.add_argument("--title", type=str, default="Improvement by (τ, β)", help="Overall figure title")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"File not found: {log_path}")
        return 1

    entries = load_entries(str(log_path))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS_ORDER):
        ax = axes[idx]
        mat, tau_list, beta_list = build_improvement_matrix(entries, task)

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.0)
        ax.set_xticks(range(len(beta_list)))
        ax.set_xticklabels([f"{b}" for b in beta_list])
        ax.set_yticks(range(len(tau_list)))
        ax.set_yticklabels([f"{t}" for t in tau_list])
        ax.set_xlabel("β")
        ax.set_ylabel("τ")
        ax.set_title(task, fontsize=10)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    text = "err"
                else:
                    text = f"{v:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    plt.suptitle(args.title, fontsize=12)
    plt.tight_layout()



    out_path = args.output or log_path.with_suffix(".improvement_heatmaps.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()
    return 0


if __name__ == "__main__":
    exit(main())