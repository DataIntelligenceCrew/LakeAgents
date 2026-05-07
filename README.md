# OpenData-Clean — Agent Pipeline

There are two layers:

- **Inner pipeline** (`orchestrator.py` — `run_orchestrator`): one full pass over intent → tables → joins → (optional) DQ → augment, with **baseline vs. augmented** metrics (R² / F1) from a lightweight ML sketch.  
- **Outer layer** (`outer_orchestrator.py`): **multi-round** adaptation. After each inner run it can call a **log-analysis** agent on the decision log, then a **modification** agent to propose discrete config updates and inherited augment columns; the next round re-runs the inner pipeline with the updated config.

## Inner pipeline stages (in order)

These map to `pipeline_steps` in `orchestrator.py`.

1. **Intent and dimensions** — Parse `user_intent` and derive search dimensions.  
2. **Table selection** — Search the datalake and collaboratively filter candidates.  
3. **Join-column selection** — Pick join columns on each candidate aligned to the query table.  
4. **Data quality (optional)** — Column-level DQ unless disabled via ablation.  
5. **Augment** — Coarse feature screen, correlations, augment-column choice, and metric evaluation.

## Outer layer (per round, when enabled)

Configured under `adaptation` in `configs/agent_pipeline_config.yaml` (e.g. `max_rounds`, `run_log_analysis_agent`, `run_modification_agent`).

After an inner run completes, the outer loop may:

1. **Log analysis** — LLM reads the structured **decision log** (phase outcomes, reason codes, thresholds) and surfaces issues or suggestions. Implemented via `Agent/log_analysis_agent.py` (prompt: `prompt/log_analysis_agent_prompt.txt`).  
2. **Modification** — LLM chooses a small set of **discrete** parameter changes (from `adaptation.discrete_options`) and optional **inherited augment columns** for the next round. Implemented via `Agent/modification_agent.py` (prompt: `prompt/modification_agent_prompt.txt`).

Then the inner pipeline runs again with the merged config until `max_rounds` or early-stop rules trigger.

## Configuration

- **Main config:** `configs/agent_pipeline_config.yaml`  
  - `task`: `join_table_name`, `join_column`, `target_column`, `task_type`, `session.session_id`  
  - `data.base_dir`: root folder for query-table data (e.g. `query_table`)  
  - `data.datalake`: domains, `max_tables`, credentials  
  - `agents`: models and provider (OpenAI / Gemini / local, etc.)

- **Perturbation / per-task join & target:** `configs/perturbation.yaml` (aligned with benchmark folder names and column names)

## How to run

From the repository root:

```bash
# Use the task block in the config; pass user_intent explicitly (recommended)
python orchestrator.py --config configs/agent_pipeline_config.yaml \
  --user-intent "Your prediction goal in natural language"

# Use original query_table data (skip perturbed-path merged config)
python orchestrator.py --config configs/agent_pipeline_config.yaml --use-original \
  --user-intent "Your prediction goal in natural language"

# Optional: session id, task type, soft wall-clock timeout (seconds)
python orchestrator.py --session-id "001" --task-type regression --timeout-seconds 3600
```

**Outer loop (inner + log + modification, multiple rounds):**

```bash
python outer_orchestrator.py --config configs/agent_pipeline_config.yaml \
  --user-intent "Your prediction goal in natural language"
```

**Default (without `--use-original`):** the inner CLI builds config for a **perturbed data directory** using `threshold` / `beta` from `configs/perturbation.yaml` (`get_perturbed_pipeline_config`). For a clean baseline on `query_table`, use **`--use-original`** or **`--base-dir`** to point at your data folder.

## Environment variables

Set API keys for the LLM backend you use, for example:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` (if using Gemini)

Details follow your `agents` section and LiteLLM routing.

## Outputs

- Terminal: per-phase timings, baseline / augmented metrics, and improvement.  
- If enabled in config, results JSON path from the config.  
- Decision trace: `data/<task_id>_decision_log.json` (`task_id` is derived from `session_id`, etc.).

## Experiments and plots (optional)

- Grid runs / logs: `experiments/run_perturbation_experiments.py`  
- Metric summaries: `scripts/summarize_experiment_metrics.py`  
- Heatmaps: `experiments/plot_improvement_heatmaps.py`
