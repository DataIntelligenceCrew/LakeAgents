#!/usr/bin/env python3
"""
Two-panel scatter: x = average performance improvement.
  Left:  y = average duration (seconds)
  Right: y = average tokens (input + output)

Each join_table has a distinct color + marker in both panels. Legend outside
the right subplot (same anchor style as before).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "experiments" / "qwen_full.csv",
        help="Path to summary CSV (default: experiments/qwen_full.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Figure output path (default: <csv_stem>_improvement_vs_cost.png next to CSV)",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()

    df = pd.read_csv(csv_path)
    imp = pd.to_numeric(df.get("average_improvement"), errors="coerce")
    tsec = pd.to_numeric(df.get("average_total_duration_seconds"), errors="coerce")
    tin = pd.to_numeric(df.get("average_total_token_input"), errors="coerce")
    tout = pd.to_numeric(df.get("average_total_token_output"), errors="coerce")
    tokens = tin + tout
    labels = df.get("join_table", pd.Series(range(len(df)))).astype(str)

    mask = imp.notna() & tsec.notna() & tin.notna() & tout.notna()
    if "status" in df.columns:
        mask &= df["status"].astype(str).str.lower().eq("ok")
    imp = imp[mask].reset_index(drop=True)
    tsec = tsec[mask].reset_index(drop=True)
    tokens = tokens[mask].reset_index(drop=True)
    labels = labels[mask].reset_index(drop=True)

    unique_labels = sorted(labels.unique())
    marker_cycle = ("o", "s", "^", "v", "D", "p", "P", "*", "X", "<", ">", "h")
    task_markers = {lab: marker_cycle[i % len(marker_cycle)] for i, lab in enumerate(unique_labels)}
    n_lab = len(unique_labels)
    cm = plt.get_cmap("tab20" if n_lab > 10 else "tab10")
    task_colors = {lab: cm(i / max(n_lab - 1, 1)) for i, lab in enumerate(unique_labels)}

    fig, (ax_dur, ax_tok) = plt.subplots(
        1,
        2,
        figsize=(11, 5.2),
        sharex=True,
        gridspec_kw={"wspace": 0.52},
    )

    for x, y_t, y_k, lab in zip(imp, tsec, tokens, labels):
        m = task_markers[lab]
        c = task_colors[lab]
        kw = dict(s=55, c=[c], marker=m, alpha=0.9, edgecolors="white", linewidths=0.65, zorder=3)
        ax_dur.scatter(x, y_t, **kw)
        ax_tok.scatter(x, y_k, **kw)

    ax_dur.set_ylabel("Average duration (seconds)", color="black")
    ax_tok.set_ylabel("Average tokens ", color="black")
    for ax in (ax_dur, ax_tok):
        ax.tick_params(colors="black")
        ax.grid(True, linestyle="--", alpha=0.35)

    # ax_dur.set_title("Average duration")
    # ax_tok.set_title("Average tokens")

    x_label = "Average performance improvement"
    ax_dur.set_xlabel(x_label, color="black", fontsize=11)
    ax_tok.set_xlabel(x_label, color="black", fontsize=11)

    task_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker=task_markers[lab],
            color=task_colors[lab],
            markerfacecolor=task_colors[lab],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7.5,
            label=lab,
        )
        for lab in unique_labels
    ]

    # Tight to right panel (smaller x = legend closer to ax_tok)
    legend_x = 1.03
    ax_tok.legend(
        handles=task_handles,
        labels=unique_labels,
        title="Task",
        loc="upper left",
        bbox_to_anchor=(legend_x, 1.0),
        bbox_transform=ax_tok.transAxes,
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
        framealpha=0.95,
    )

    fig.tight_layout(rect=[0.02, 0.06, 0.94, 0.98])

    out = args.output
    if out is None:
        out = csv_path.with_name(f"{csv_path.stem}_improvement_vs_cost.png")
    else:
        if not out.is_absolute():
            out = (root / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
