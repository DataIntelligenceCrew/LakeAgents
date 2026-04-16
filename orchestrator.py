import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from google.adk.runners import InMemoryRunner

from Agent.analyze_user_intent_agent import build_analyze_user_intent_agent
from Agent.augment_column_selection_agent import build_augment_column_selection_agent
from Agent.data_quality_agent import build_data_quality_agent
from Agent.join_column_selection_agent import build_join_column_choose_agent
from Pipeline.context import PipelineContext
from Pipeline.phase_augment import _evaluate_metric, run_augment
from Pipeline.phase_data_quality import run_data_quality
from Pipeline.phase_intent_and_dimensions import run_intent_and_dimensions
from Pipeline.phase_join_columns import run_join_columns
from Pipeline.phase_table_selection import run_table_selection
from Pipeline.logging_utils import (
    build_task_id,
    classify_outcome,
    compute_config_version,
    init_decision_log,
)
from Pipeline.utils import close_runner_safely
from agent_config_loader import AgentPipelineConfig, load_config
from benchmark_perturbation.benchmark_perturbation import get_perturbed_pipeline_config
from tools.llm_agent_tools import find_dataset_dir


for key in ["GOOGLE_API_KEY", "OPENAI_API_KEY"]:
    val = os.getenv(key)
    if val:
        os.environ[key] = val


def _finalize_pipeline_timeout(ctx: PipelineContext, *, skipped_phases: List[str]) -> None:
    """Skip remaining phases; join keys stay in decision_log; augment columns left empty."""
    decision_log = ctx.state.get("decision_log")
    if isinstance(decision_log, dict):
        decision_log["timeout"] = {
            "skipped_phases": skipped_phases,
            "result_quality": "degraded_due_to_timeout",
            "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    metric_name = "r2_score" if ctx.task_type == "regression" else "f1_score"
    join_columns = ctx.state.get("join_columns") or []
    if isinstance(join_columns, str):
        join_columns = [join_columns]
    join_df = ctx.state.get("join_df")
    baseline_metric = None
    augmented_metric = None
    if join_df is not None and join_columns:
        baseline_metric = _evaluate_metric(join_df, ctx.target_column, list(join_columns), ctx.task_type)
    augment_results: List[Dict[str, Any]] = []
    for tbl in ctx.state.get("final_selected_tables") or []:
        cand = tbl.get("candidate_table") or "?"
        selected_join = tbl.get("selected_columns") or []
        reasoning = (
            "pipeline_timeout; candidate join columns from log: "
            + ", ".join(str(c) for c in selected_join)
            if selected_join
            else "pipeline_timeout_skipped_remaining_phases"
        )
        augment_results.append(
            {
                "candidate_table": cand,
                "selected_augment_columns": [],
                "ranked_candidates": [],
                "column_decisions": [],
                "early_stop_log": [],
                "reasoning": reasoning,
            }
        )
    ctx.state["output"] = {
        "augment_results": augment_results,
        "baseline_metric": baseline_metric,
        "augmented_metric": augmented_metric,
        "metric_name": metric_name,
        "pipeline_timings_seconds": ctx.pipeline_timings,
        "stop_reason": "pipeline_timeout_fallback",
    }


async def run_orchestrator(
    join_table_name: Optional[str] = None,
    join_column: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    user_intent: Optional[str] = None,
    session_id: Optional[str] = None,
    config_path: Optional[str] = None,
    config: Optional[AgentPipelineConfig] = None,
    reuse_context: Optional[Dict[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    if config is None:
        config = AgentPipelineConfig(config_path)

    if join_table_name is None:
        join_table_name = config.join_table_name
    if join_column is None:
        join_column = config.join_column
    if target_column is None:
        target_column = config.target_column
    if task_type is None:
        task_type = config.task_type
    if user_intent is None:
        user_intent = getattr(config, "user_intent", None) or f"predict the {target_column}"
    session_id = session_id if session_id is not None else config.session_id
    task_id = build_task_id(session_id)

    base_dir = config.base_dir
    real_join_table_name = find_dataset_dir(join_table_name, base_dir)
    base_path_obj = Path(__file__).resolve().parent / base_dir
    join_meta_path = base_path_obj / real_join_table_name / "metadata.json"
    query_table_display_name = join_table_name
    if join_meta_path.exists():
        try:
            with open(join_meta_path, "r", encoding="utf-8") as f:
                join_meta = json.load(f)
            display = (join_meta.get("resource") or {}).get("name", "").strip()
            if display:
                query_table_display_name = display
        except Exception:
            pass

    ctx = PipelineContext(
        config=config,
        session_id=session_id,
        join_table_name=join_table_name,
        target_column=target_column,
        task_type=task_type,
        user_intent=user_intent,
        base_dir=base_dir,
        real_join_table_name=real_join_table_name,
        query_table_display_name=query_table_display_name,
        base_path_obj=base_path_obj,
        join_meta_path=join_meta_path,
    )
    config_dict = config.config if isinstance(config.config, dict) else {}
    config_version = compute_config_version(config_dict)
    threshold_snapshot = {
        "perturbation": (config_dict.get("perturbation") or {}),
        "datalake": (config_dict.get("data", {}).get("datalake", {}) if isinstance(config_dict.get("data"), dict) else {}),
    }
    ctx.state["task_id"] = task_id
    ctx.state["decision_log"] = init_decision_log(
        task_id=task_id,
        session_id=session_id,
        join_table_name=join_table_name,
        task_type=task_type,
        target_column=target_column,
        config_version=config_version,
        threshold_snapshot=threshold_snapshot,
    )
    ctx.state["join_column"] = join_column
    if isinstance(reuse_context, dict):
        reused_dims = reuse_context.get("dimension_specifications")
        reused_table_selection = reuse_context.get("table_selection")
        reused_augment = reuse_context.get("inherited_augment_columns")
        if isinstance(reused_dims, dict) and reused_dims:
            ctx.state["reuse_dimension_specifications"] = reused_dims
        if isinstance(reused_table_selection, dict) and reused_table_selection:
            ctx.state["reuse_table_selection"] = reused_table_selection
        if isinstance(reused_augment, dict) and reused_augment:
            ctx.state["inherited_augment_columns"] = reused_augment
    ctx.state["analyze_intent_runner"] = InMemoryRunner(agent=build_analyze_user_intent_agent(config=config))
    ctx.state["joincol_runner"] = InMemoryRunner(agent=build_join_column_choose_agent(config=config))
    ctx.state["augment_runner"] = InMemoryRunner(agent=build_augment_column_selection_agent(config))
    ctx.state["data_quality_runner"] = InMemoryRunner(agent=build_data_quality_agent(config=config))

    deadline: Optional[float] = None
    if timeout_seconds is not None:
        deadline = time.perf_counter() + float(timeout_seconds)

    def _past_deadline() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    pipeline_steps: List[Tuple[str, Any]] = [
        ("intent_and_dimensions", run_intent_and_dimensions),
        ("table_selection", run_table_selection),
        ("join_column_selection", run_join_columns),
        ("data_quality", run_data_quality),
        ("augment", run_augment),
    ]

    t0 = time.perf_counter()
    try:
        for idx, (_, phase_fn) in enumerate(pipeline_steps):
            await phase_fn(ctx)
            if _past_deadline():
                skipped = [n for n, _ in pipeline_steps[idx + 1 :]]
                if skipped:
                    _finalize_pipeline_timeout(ctx, skipped_phases=skipped)
                break
    finally:
        ctx.pipeline_timings["00_total_wall_seconds"] = time.perf_counter() - t0
        await close_runner_safely(ctx.state.get("analyze_intent_runner"))
        await close_runner_safely(ctx.state.get("joincol_runner"))
        await close_runner_safely(ctx.state.get("augment_runner"))
        await close_runner_safely(ctx.state.get("data_quality_runner"))

    if ctx.run_record_path is not None:
        run_record = ctx.state.get("run_record", {})
        run_record["pipeline_timings_seconds"] = dict(sorted(ctx.pipeline_timings.items()))
        try:
            with open(ctx.run_record_path, "w", encoding="utf-8") as f:
                json.dump(run_record, f, indent=2, ensure_ascii=False)
            print(f"[Run Record] Updated with pipeline timings: {ctx.run_record_path}")
        except Exception:
            pass

    output = ctx.state.get(
        "output",
        {
            "augment_results": [],
            "baseline_metric": None,
            "augmented_metric": None,
            "metric_name": "r2_score" if task_type == "regression" else "f1_score",
            "pipeline_timings_seconds": ctx.pipeline_timings,
        },
    )
    decision_log = ctx.state.get("decision_log", {})
    if isinstance(decision_log, dict):
        decision_log["pipeline_timings_seconds"] = dict(sorted(ctx.pipeline_timings.items()))
        decision_log["outcome"] = classify_outcome(
            output.get("baseline_metric"),
            output.get("augmented_metric"),
        )
        decision_log["metric_name"] = output.get("metric_name")
        decision_log["baseline_metric"] = output.get("baseline_metric")
        decision_log["augmented_metric"] = output.get("augmented_metric")
        decision_log["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        decision_log_path = data_dir / f"{task_id}_decision_log.json"
        try:
            with open(decision_log_path, "w", encoding="utf-8") as f:
                json.dump(decision_log, f, indent=2, ensure_ascii=False)
            output["decision_log_path"] = str(decision_log_path)
        except Exception:
            pass

    # Keep a compact reuse payload for outer-loop rounds.
    decision_log_dict = decision_log if isinstance(decision_log, dict) else {}
    output["_reuse_context"] = {
        "dimension_specifications": ctx.state.get("dimension_specifications", {}),
        "table_selection": {
            "selected_tables": ((decision_log_dict.get("phases", {}) or {}).get("table_selection", {}) or {}).get("selected_tables", []),
            "excluded_tables": ((decision_log_dict.get("phases", {}) or {}).get("table_selection", {}) or {}).get("excluded_tables", []),
            "candidate_ids_for_run": ctx.state.get("table_selection_candidate_ids", []),
            "domain_for_fetch": ctx.state.get("domain_for_fetch"),
            "query_table_description": ctx.state.get("query_table_description", ""),
        },
    }

    print("\n⏱ Pipeline phase timings (seconds, descending):")
    for name, sec in sorted(ctx.pipeline_timings.items(), key=lambda x: -x[1]):
        print(f"   {name}: {sec:.3f}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-agent data augmentation pipeline")
    parser.add_argument(
        "--user-intent",
        type=str,
        help='User intent/prediction goal (e.g., "I would like to predict the crime rate in New York City")',
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--join-table", type=str, default=None, help="Join table name")
    parser.add_argument("--target-column", type=str, default=None, help="Target column to predict")
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        choices=["regression", "classification"],
        help="Task type",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for this query task (for per-session checked dataset)",
    )
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use original (unperturbed) data instead of perturbed data",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Override data base directory (e.g. perturbed_0.85_0.1)",
    )
    parser.add_argument(
        "--data-filename",
        type=str,
        default=None,
        help="Override data filename (e.g. rows_original.csv)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Soft wall-clock limit; remaining phases are skipped and output is built from current logs (e.g. 1800 for 30 minutes).",
    )
    args = parser.parse_args()

    try:
        if args.use_original:
            config = AgentPipelineConfig(args.config)
            config.config["data"] = {**config.config.get("data", {}), "base_dir": "query_table"}
        else:
            perturb_path = Path(__file__).resolve().parent / "configs" / "perturbation.yaml"
            perturb_cfg = {}
            if perturb_path.exists():
                with open(perturb_path, "r", encoding="utf-8") as f:
                    p = (yaml.safe_load(f) or {}).get("perturbation", {})
                    perturb_cfg = {"threshold": p.get("threshold", 0.85), "beta": p.get("beta", 0.1)}
            table_folder = args.join_table
            if table_folder is None:
                base_cfg = load_config(args.config)
                table_folder = (base_cfg.get("task") or {}).get("join_table_name")
            cfg_dict = get_perturbed_pipeline_config(
                table_folder=table_folder,
                threshold=perturb_cfg.get("threshold", 0.85),
                beta=perturb_cfg.get("beta", 0.1),
            )
            config = AgentPipelineConfig(config_dict=cfg_dict)

        if args.base_dir is not None:
            config.config.setdefault("data", {})
            config.config["data"]["base_dir"] = args.base_dir
        if args.data_filename is not None:
            config.config.setdefault("data", {})
            config.config["data"]["data_filename"] = args.data_filename

        output = asyncio.run(
            run_orchestrator(
                config=config,
                user_intent=args.user_intent,
                session_id=args.session_id,
                join_table_name=None,
                target_column=None,
                task_type=args.task_type,
                timeout_seconds=args.timeout_seconds,
            )
        )

        print("\n--- Final Results ---")
        if output:
            metric_name = output.get("metric_name", "r2_score")
            baseline = output.get("baseline_metric")
            augmented = output.get("augmented_metric")
            baseline_str = f"{baseline:.4f}" if baseline is not None else "N/A"
            augmented_str = f"{augmented:.4f}" if augmented is not None else "N/A"
            print(f"📊 Baseline ({metric_name}): {baseline_str}")
            print(f"📊 Augmented ({metric_name}): {augmented_str}")
            if baseline is not None and augmented is not None:
                improvement = augmented - baseline
                pct = (improvement / abs(baseline) * 100) if baseline != 0 else 0
                print(f"   Improvement: {improvement:+.4f} ({pct:+.1f}%)")

        if config.save_results:
            output_file = Path(config.results_file)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")

        if config.print_results and output:
            for item in output.get("augment_results", []):
                payload = {
                    "selected_augment_columns": item.get("selected_augment_columns", []),
                    "reasoning": item.get("reasoning", ""),
                }
                print("\nAugmentColumnSelectionAgent >", json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Workflow failed: {e}")
        import traceback

        traceback.print_exc()

