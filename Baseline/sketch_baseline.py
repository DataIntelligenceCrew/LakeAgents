#!/usr/bin/env python3
"""
Multi-table sketch baseline using step5 (corr) JSON + pipeline aggregation.

Workflow (aligned with repo intent):
  1. For each task, load step5_* JSON. For every candidate with status ok, take
     top10 rows with value > 0 (strictly positive Pearson r in the stored top10).
  2. Match each such candidate to step34 pick-join (same candidate_table_id, status ok).
  3. Fetch the candidate table, prepare numeric columns (same helpers as corr step),
     aggregate by the pick's join column using aggregate_candidate_by_join_key —
     numerical features use mean (pipeline default numerical_agg=\"mean\").
  4. Left-merge all aggregated aug columns onto the query rows CSV (single join column
     from perturbation.yaml), prefixing aug columns per candidate to avoid clashes.
  5. Fit the same shallow models as Classification_regression sketch path and report
     improvement (classification: Δf1_weighted; regression: Δr²).

Tau×β grid follows experiments/run_perturbation_experiments and non_agentic_baseline.

Perturbed rows.csv column names may differ from configs/perturbation.yaml because
benchmark_perturbation rewrites CSV headers via the same replacement map as metadata
(agent_config_loader.load_replacements_for_table + metadata_perturbation._apply_replacements_to_text).
We resolve join/target names per perturbed_{τ}_{β}/<task>/ before merge and ML.

Examples:
  python Baseline/sketch_baseline.py -o logs/sketch_baseline_jobs/out.json
  python Baseline/sketch_baseline.py --tables COVID-Chicago Demo-Chicago -o /tmp/sk.json
  python Baseline/sketch_baseline.py --print-launcher
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_BASELINE_DIR = Path(__file__).resolve().parent
_PROJECT = _BASELINE_DIR.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))
_scripts = _PROJECT / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import non_agentic_baseline as _nab  # noqa: E402

from Classification_regression import (  # noqa: E402
    preprocess_data,
    run_classification_task,
    run_regression_task,
)
from agent_config_loader import load_replacements_for_table  # noqa: E402
from benchmark_perturbation.metadata_perturbation import (  # noqa: E402
    _apply_replacements_to_text,
)
from tools.aggregation import aggregate_candidate_by_join_key  # noqa: E402
from tools.sketch import get_candidate_table  # noqa: E402

PERTURBATION_YAML = _PROJECT / "configs" / "perturbation.yaml"

TAU_VALUES: tuple[float, ...] = (0.1, 0.5, 0.9)
BETA_VALUES: tuple[float, ...] = (0.1, 0.5, 0.9)

DEFAULT_TWELVE_TASKS: tuple[str, ...] = (
    "COVID-Chicago",
    "Demo-Chicago",
    "Economic-Chicago",
    "Education-Chicago",
    "COVID-NYC",
    "Demo-NYC",
    "Economic-NYC",
    "Education-NYC",
    "Environment_NYC",
    "Food Inspections-NYC",
    "Food Inspections-Chicago",
    "Building Permits-Chicago",
)

TASK_TYPES: dict[str, str] = {
    "COVID-Chicago": "regression",
    "Demo-Chicago": "regression",
    "Economic-Chicago": "regression",
    "Education-Chicago": "classification",
    "COVID-NYC": "regression",
    "Demo-NYC": "classification",
    "Economic-NYC": "classification",
    "Education-NYC": "classification",
    "Environment_NYC": "classification",
    "Food Inspections-NYC": "classification",
    "Food Inspections-Chicago": "classification",
    "Building Permits-Chicago": "classification",
}


def _fname_safe(task_folder: str) -> str:
    return task_folder.replace(" ", "_")


def _resolve_df_join_target(
    columns: pd.Index,
    *,
    join_col_yaml: str,
    target_col_yaml: str,
    replacements: dict[str, Any],
) -> tuple[str | None, str | None, dict[str, Any]]:
    """
    Map yaml join/target to names actually present on perturbed rows.csv.

    Applies the same word-boundary replacement rules as CSV header rewriting; if the
    perturbed header is still the original spelling, falls back to yaml literals.
    """
    info: dict[str, Any] = {
        "n_replacement_rules": len(replacements or {}),
    }
    repl = replacements or {}
    join_eff = _apply_replacements_to_text(str(join_col_yaml), repl)
    tgt_eff = _apply_replacements_to_text(str(target_col_yaml), repl)
    info["join_after_synonym_rules"] = join_eff
    info["target_after_synonym_rules"] = tgt_eff
    colset = set(columns)

    def pick(yaml_nm: str, eff_nm: str) -> tuple[str | None, str]:
        if eff_nm in colset:
            return eff_nm, "perturbed_header"
        if yaml_nm in colset:
            return yaml_nm, "yaml_literal_fallback"
        return None, "missing"

    j_use, j_how = pick(str(join_col_yaml), join_eff)
    t_use, t_how = pick(str(target_col_yaml), tgt_eff)
    info["join_resolution"] = j_how
    info["target_resolution"] = t_how
    return j_use, t_use, info


def _normalize_join_columns(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x).strip()]
    return []


def _load_tables_yaml() -> dict[str, Any]:
    with open(PERTURBATION_YAML, encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    tbl = root.get("tables") or {}
    if not isinstance(tbl, dict):
        raise ValueError("perturbation.yaml missing tables dict")
    return tbl


def _positive_top10_features_sorted(corr_candidate: dict[str, Any]) -> list[tuple[str, float]]:
    """From stored top10, keep strictly positive correlations; sort by value desc."""
    rows: list[tuple[str, float]] = []
    for row in corr_candidate.get("top10") or []:
        if not isinstance(row, dict):
            continue
        fv = pd.to_numeric(row.get("value"), errors="coerce")
        name = row.get("feature")
        if pd.isna(fv) or name is None:
            continue
        v = float(fv)
        if v > 0:
            rows.append((str(name), v))
    rows.sort(key=lambda t: (-t[1], t[0]))
    return rows


def _index_pick_candidates(payload: dict[str, Any], task_folder: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in payload.get("tables", {}).get(task_folder, {}).get("candidates") or []:
        cid = str(c.get("candidate_table_id") or "").strip()
        if cid:
            out[cid] = c
    return out


def _candidate_metric(task_folder: str) -> str:
    return "r2_score" if TASK_TYPES.get(task_folder) == "regression" else "f1_score"


def _evaluate_metrics(X, y_enc, encoder, task_type: str) -> dict[str, float] | None:
    if task_type == "classification":
        if encoder is None:
            return None
        try:
            r = run_classification_task(X, y_enc, encoder)
        except ValueError:
            return None
        return {"f1_score": float(r["f1_score"])}
    try:
        r = run_regression_task(X, y_enc)
    except ValueError:
        return None
    return {"r2_score": float(r["r2_score"])}


def _merge_one_augment_table(
    query_df: pd.DataFrame,
    join_col: str,
    pick_cand: dict[str, Any],
    corr_candidate: dict[str, Any],
    positive_features_sorted: list[tuple[str, float]],
) -> tuple[pd.DataFrame, str | None, list[str]]:
    """
    Fetch candidate, aggregate numerics by mean on pick join column, rename join to query key,
    left-merge prefixed aug columns onto query_df.
    """
    cid = str(pick_cand.get("candidate_table_id") or "").strip()
    domain = pick_cand.get("domain")
    jn = str(pick_cand.get("top1_join_column") or "").strip()
    if not cid or not domain or not jn:
        return query_df, "missing candidate id, domain, or top1_join_column in pick JSON", []

    feats = [f for f, _ in positive_features_sorted]
    df, st = get_candidate_table(cid, domain)
    if not isinstance(st, dict) or not st.get("success") or df is None or df.empty:
        return query_df, (st.get("reason") if isinstance(st, dict) else None) or "candidate fetch empty", []

    if jn not in df.columns:
        return query_df, f"candidate missing join column {jn!r}", []

    df_prep, num_cols = _nab._prepare_df_and_list_numerical_features_for_corr(df, [jn])
    need = [c for c in feats if c in num_cols]
    if not need:
        return query_df, "no_requested_positive_top10_cols_present_after_numeric_prep", []

    cols = list(dict.fromkeys([jn] + need))
    sub = df_prep[cols].copy()
    try:
        agg = aggregate_candidate_by_join_key(
            sub,
            [jn],
            numerical_agg="mean",
            table_id=cid,
            fasttext_model_path=None,
            llm_summarize=False,
        )
    except Exception as e:  # noqa: BLE001
        return query_df, str(e), []

    jc = join_col
    agg = agg.rename(columns={jn: jc}) if jn != jc else agg

    prefix = f"aug_{_fname_safe(cid)}__"
    renames = {c: prefix + str(c).replace(" ", "_") for c in agg.columns if c != jc}
    agg_re = agg.rename(columns=renames)
    overlap = set(agg_re.columns) & set(query_df.columns)
    overlap.discard(jc)
    if overlap:
        agg_re = agg_re.drop(columns=list(overlap), errors="ignore")

    merged = query_df.merge(agg_re, on=jc, how="left")
    added = list(renames.values())
    return merged, None, added


def _run_one_cell(
    rows_path: Path,
    *,
    project: Path,
    join_col: str,
    target_col: str,
    task_folder: str,
    pick_payload: dict[str, Any],
    corr_payload: dict[str, Any],
    task_type: str,
) -> dict[str, Any]:
    pj_path = corr_payload.get("__path_pick__")  # diagnostic if set
    cj_path = corr_payload.get("__path_corr__")
    metric_name = _candidate_metric(task_folder)
    row: dict[str, Any] = {
        "rows_path": str(rows_path),
        "join_column_yaml": join_col,
        "target_column_yaml": target_col,
        "task_type": task_type,
        "primary_metric": metric_name,
    }
    try:
        query_df = pd.read_csv(rows_path)
    except Exception as e:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"read_csv failed: {e}"
        return row

    perturbed_base = rows_path.parent.parent
    replacements = load_replacements_for_table(
        task_folder,
        project_root=project.resolve(),
        perturbed_base_dir=perturbed_base,
    )
    join_df, tgt_df, resinfo = _resolve_df_join_target(
        query_df.columns,
        join_col_yaml=join_col,
        target_col_yaml=target_col,
        replacements=replacements,
    )
    row["column_name_resolution"] = resinfo
    row["join_column"] = join_df
    row["target_column"] = tgt_df

    if tgt_df is None:
        row["status"] = "error"
        row["error"] = (
            f"missing target column (yaml={target_col!r}, after_rules={resinfo['target_after_synonym_rules']!r})"
        )
        return row
    if join_df is None:
        row["status"] = "error"
        row["error"] = (
            f"missing join column (yaml={join_col!r}, after_rules={resinfo['join_after_synonym_rules']!r})"
        )
        return row

    join_col_use = join_df
    target_col_use = tgt_df

    corrs_section = corr_payload.get("tables", {}).get(task_folder, {})
    corr_cands = list(corrs_section.get("candidates") or [])
    picks_by_id = _index_pick_candidates(pick_payload, task_folder)

    aug_df = query_df.copy()
    used: list[dict[str, Any]] = []
    skip_notes: list[str] = []

    for cc in corr_cands:
        if cc.get("status") != "ok":
            continue
        positives = _positive_top10_features_sorted(cc)
        if not positives:
            continue
        cid = str(cc.get("candidate_table_id") or "").strip()
        pc = picks_by_id.get(cid)
        if not pc or pc.get("status") != "ok":
            skip_notes.append(f"{cid}: no ok pick pairing")
            continue

        merged, err, cols_added = _merge_one_augment_table(
            aug_df,
            join_col_use,
            pc,
            cc,
            positives,
        )
        if err:
            skip_notes.append(f"{cid}: {err}")
            continue
        aug_df = merged
        used.append(
            {
                "candidate_table_id": cid,
                "domain": pc.get("domain"),
                "join_on_candidate": pc.get("top1_join_column"),
                "positive_top10_columns": [{"feature": n, "value": v} for n, v in positives],
                "columns_merged": cols_added,
            }
        )

    Xb, yb, enc_b, _ = preprocess_data(query_df.copy(), target_col_use, task_type)
    if Xb is None:
        row["status"] = "error"
        row["error"] = "preprocess baseline failed"
        row["skipped_candidates"] = skip_notes
        return row

    Xa, ya, enc_a, _ = preprocess_data(aug_df.copy(), target_col_use, task_type)
    if Xa is None:
        row["status"] = "error"
        row["error"] = "preprocess augmented failed"
        row["skipped_candidates"] = skip_notes
        return row

    m_base = _evaluate_metrics(Xb, yb, enc_b, task_type)
    m_aug = _evaluate_metrics(Xa, ya, enc_a, task_type)
    if m_base is None or m_aug is None:
        row["status"] = "error"
        row["error"] = "model evaluation failed (insufficient data or stratify issue)"
        row["skipped_candidates"] = skip_notes
        return row

    bv = float(m_base[metric_name])
    av = float(m_aug[metric_name])
    row.update(
        {
            "baseline_metrics": m_base,
            "augmented_metrics": m_aug,
            "improvement": av - bv,
            "augment_candidates_used": used,
            "n_augment_tables": len(used),
            "skipped_candidates": skip_notes,
            "status": ("ok_with_augment" if used else "ok_baseline_only"),
        },
    )
    if pj_path:
        row["pick_join_path"] = pj_path
    if cj_path:
        row["corr_path"] = cj_path
    return row


def _task_block(
    task_folder: str,
    *,
    project: Path,
    pick_dir: Path,
    corr_dir: Path,
) -> dict[str, Any]:
    safe = _fname_safe(task_folder)
    pj_file = pick_dir / f"step34_{safe}.json"
    cj_file = corr_dir / f"step5_{safe}.json"

    tbl_yml = _load_tables_yaml()
    tcfg = tbl_yml.get(task_folder) or {}
    jc_list = _normalize_join_columns(tcfg.get("join_columns"))
    tgt = tcfg.get("target_column")

    section: dict[str, Any] = {
        "task_type": TASK_TYPES.get(task_folder, "classification"),
        "pick_join_path": str(pj_file),
        "corr_path": str(cj_file),
        "cells": [],
    }

    if len(jc_list) != 1 or not tgt:
        section["status"] = "error"
        section["error"] = "need single join column and target in perturbation.yaml"
        return section

    join_col = jc_list[0]
    if not pj_file.is_file() or not cj_file.is_file():
        section["status"] = "error"
        section["error"] = (
            f"missing pick or corr json (pj.exists()={pj_file.is_file()} "
            f"cj.exists()={cj_file.is_file()})"
        )
        return section

    pick_payload = json.loads(pj_file.read_text(encoding="utf-8"))
    corr_payload = json.loads(cj_file.read_text(encoding="utf-8"))
    corr_payload["__path_pick__"] = str(pj_file)
    corr_payload["__path_corr__"] = str(cj_file)

    task_type = TASK_TYPES.get(task_folder, "classification")
    for tau in TAU_VALUES:
        for beta in BETA_VALUES:
            base = project / f"perturbed_{tau}_{beta}"
            rows_path = base / task_folder / "rows.csv"
            cell = _run_one_cell(
                rows_path,
                project=project,
                join_col=join_col,
                target_col=str(tgt),
                task_folder=task_folder,
                pick_payload=pick_payload,
                corr_payload=corr_payload,
                task_type=task_type,
            )
            cell["tau"] = tau
            cell["beta"] = beta
            section["cells"].append(cell)

    ok_cells = sum(1 for c in section["cells"] if str(c.get("status", "")).startswith("ok"))
    section["status"] = "ok" if ok_cells == len(section["cells"]) else "partial"
    section["cells_ok_count"] = ok_cells
    return section


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-table corr>0 augment baseline vs query rows CSV.")
    p.add_argument(
        "--tables",
        nargs="*",
        default=list(DEFAULT_TWELVE_TASKS),
        help="Task folder names under perturbation.yaml (default: 12 TASKS)",
    )
    p.add_argument(
        "--pick-dir",
        type=Path,
        default=_PROJECT / "logs" / "non_agentic_pick_join_parallel",
        help="Directory with step34_*.json",
    )
    p.add_argument(
        "--corr-dir",
        type=Path,
        default=_PROJECT / "logs" / "non_agentic_corr_parallel",
        help="Directory with step5_*.json",
    )
    p.add_argument("--project-root", type=Path, default=_PROJECT, help="Repo root (perturbed_* dirs live here)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_PROJECT / "logs" / "sketch_baseline_jobs" / "sketch_baseline_summary.json",
        help="Write combined JSON summary for all processed tables",
    )
    p.add_argument(
        "--print-launcher",
        action="store_true",
        help="Print a bash snippet to run one subprocess per task",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    proj: Path = args.project_root.resolve()

    if args.print_launcher:
        exe = sys.executable
        script = (_BASELINE_DIR / "sketch_baseline.py").resolve()
        out_dir = proj / "logs" / "sketch_baseline_jobs"
        tables = args.tables if args.tables else list(DEFAULT_TWELVE_TASKS)
        print("#!/usr/bin/env bash")
        print(f'mkdir -p "{out_dir}"')
        print("set -euo pipefail")
        for tf in tables:
            safe = _fname_safe(tf)
            of = out_dir / f"sketch_baseline_{safe}.json"
            print(
                f'nohup {exe} "{script}" --project-root "{proj}" --tables "{tf}" '
                f'--pick-dir "{args.pick_dir}" --corr-dir "{args.corr_dir}" '
                f'-o "{of}" >/dev/null 2>&1 &'
            )
        print("wait")
        print('echo "all sketch_baseline jobs finished"')
        return

    out: dict[str, Any] = {
        "mode": "sketch_baseline_multi_table_positive_top10_ml",
        "project_root": str(proj),
        "pick_dir": str(args.pick_dir.resolve()),
        "corr_dir": str(args.corr_dir.resolve()),
        "tables": {},
    }

    for tf in args.tables:
        out["tables"][tf] = _task_block(tf, project=proj, pick_dir=args.pick_dir, corr_dir=args.corr_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
