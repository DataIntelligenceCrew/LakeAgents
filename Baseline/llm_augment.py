"""
LLM-based augment: suggest columns, then generate values per join_key.
Uses Google ADK Agent (no search tool). Outputs rows_llm.csv and metadata_llm.json.
Supports config provider: gemini, openai, local.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import asyncio
import json
import re
import yaml
from pathlib import Path
import pandas as pd
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from agent_config_loader import AgentPipelineConfig, load_config


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM output, handling markdown code blocks."""
    if not text or not isinstance(text, str):
        return {}
    text = text.strip()
    if not text:
        return {}
    if "```json" in text:
        parts = text.split("```json", 1)[1].split("```", 1)
        json_str = parts[0].strip() if parts else None
    elif "```" in text:
        parts = text.split("```", 1)[1].split("```", 1)
        json_str = parts[0].strip() if parts else None
    else:
        json_str = text
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _load_prompt_section(section: str) -> str:
    """Load section from prompt/llm_augment_suggest_prompt.txt. Sections: SUGGEST, GENERATE_VALUES."""
    root = Path(__file__).resolve().parent.parent
    path = root / "prompt" / "llm_augment_suggest_prompt.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^=== (\w+) ===\s*$", text, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts) and parts[i].strip().upper() == section.upper():
            return parts[i + 1].strip()
    return text.strip()


def load_query_table_metadata(base_dir: str, table_folder: str) -> Dict[str, Any]:
    """Load metadata.json for a query table."""
    path = Path(base_dir) / table_folder / "metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1: Suggest augment columns (ADK Agent, no tools)
# ---------------------------------------------------------------------------

def _build_llm_for_agent(config: AgentPipelineConfig, agent_name: str = "utility_gain"):
    """Build LLM by config provider: gemini, openai, or local."""
    provider = config.get_provider(agent_name)
    model_name = config.get_model_name(agent_name, provider)
    try:
        retry_cfg = config.get_retry_config()
    except Exception:
        retry_cfg = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504])

    if provider == "gemini":
        from google.adk.models.google_llm import Gemini
        return Gemini(model=model_name, retry_options=retry_cfg)
    from google.adk.models.lite_llm import LiteLlm
    if provider == "openai":
        return LiteLlm(model=model_name)
    kw = {"model": model_name}
    if provider == "local":
        kw["api_base"] = "http://localhost:8080/v1"
        kw["api_key"] = "not-needed"
    return LiteLlm(**kw)


def _build_suggest_prompt(
    table_name: str,
    table_description: str,
    column_names: List[str],
    column_descriptions: List[str],
    join_column: str,
    target_column: str,
    target_column_description: str,
    task_type: str,
    user_intent: str,
) -> str:
    col_info = []
    for name, desc in zip(column_names, column_descriptions):
        col_info.append(f"  - {name}: {desc or '(no description)'}")
    return f"""table_name: {table_name}
table_description: {table_description}
column_names: {column_names}
column_descriptions:
{chr(10).join(col_info)}
join_column: {join_column}
target_column: {target_column}
target_column_description: {target_column_description}
task_type: {task_type}
user_intent: {user_intent}

Follow the SUGGEST section instructions. Return ONLY valid JSON with key "suggested_augment_columns"."""


def _extract_last_text_from_events(events) -> str:
    last_text = ""
    for event in events:
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                t = getattr(part, "text", None)
                if t:
                    last_text = t
    return last_text


async def suggest_llm_augment_columns(
    base_dir: str,
    table_folder: str,
    join_column: str,
    target_column: str,
    task_type: str,
    user_intent: str,
    config: Optional[AgentPipelineConfig] = None,
) -> List[Dict[str, Any]]:
    """Suggest augment columns using ADK Agent. Uses config provider (gemini/openai/local)."""
    if config is None:
        config = AgentPipelineConfig(config_dict=load_config())
    meta = load_query_table_metadata(base_dir, table_folder)
    if not meta:
        return []
    res = meta.get("resource") or {}
    table_name = res.get("name") or table_folder
    table_description = res.get("description") or ""
    column_names = res.get("columns_name") or []
    column_descriptions = res.get("columns_description") or [""] * len(column_names)
    if len(column_descriptions) < len(column_names):
        column_descriptions = column_descriptions + [""] * (len(column_names) - len(column_descriptions))
    target_idx = next((i for i, c in enumerate(column_names) if c == target_column), -1)
    target_desc = column_descriptions[target_idx] if target_idx >= 0 else ""

    instruction = _load_prompt_section("SUGGEST")
    if not instruction:
        instruction = "Suggest augment columns that help the prediction task. Return JSON with suggested_augment_columns."

    llm = _build_llm_for_agent(config)
    agent = Agent(
        name="AugmentSuggestAgent",
        model=llm,
        instruction=instruction,
        generate_content_config=types.GenerateContentConfig(temperature=0.3),
    )
    runner = InMemoryRunner(agent=agent)
    prompt = _build_suggest_prompt(
        table_name=table_name,
        table_description=table_description,
        column_names=column_names,
        column_descriptions=column_descriptions[:len(column_names)],
        join_column=join_column,
        target_column=target_column,
        target_column_description=target_desc,
        task_type=task_type,
        user_intent=user_intent,
    )
    events = await runner.run_debug(prompt)
    last_text = _extract_last_text_from_events(events)
    parsed = _extract_json(last_text)
    suggested = parsed.get("suggested_augment_columns", [])
    return suggested[:15] if isinstance(suggested, list) else []


# ---------------------------------------------------------------------------
# Step 2: Generate values per join_key (ADK Agent, no tools)
# ---------------------------------------------------------------------------

def _normalize_suggested_columns(suggested: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for item in suggested:
        if isinstance(item, dict):
            name = item.get("column_name") or item.get("name") or ""
            out.append({
                "column_name": str(name).strip() if name else "",
                "description": item.get("description", ""),
                "column_type": item.get("column_type", "numerical"),
            })
        elif isinstance(item, str) and item.strip():
            out.append({"column_name": item.strip(), "description": "", "column_type": "numerical"})
    return [x for x in out if x.get("column_name")]


def _build_generate_prompt(
    join_key: str,
    augment_columns: List[Dict[str, Any]],
    table_context: str,
) -> str:
    instruction = _load_prompt_section("GENERATE_VALUES")
    if not instruction:
        instruction = "Provide values from your knowledge. Return JSON with column names as keys. Use null if you do not know."
    col_lines = []
    for c in augment_columns:
        name = c.get("column_name", "")
        desc = c.get("description", "") or "(no description)"
        ctype = c.get("column_type", "numerical")
        col_lines.append(f"  - {name} ({ctype}): {desc}")
    col_names = [c.get("column_name", "") for c in augment_columns]
    return f"""join_key: {join_key}
table_context: {table_context[:1500]}

augment_columns:
{chr(10).join(col_lines)}

Follow GENERATE_VALUES rules. Return ONLY valid JSON with keys: {', '.join(col_names)}. Use null for any column you do not know."""


async def generate_llm_augment_values(
    base_dir: str,
    table_folder: str,
    join_column: str,
    suggested_columns: List[Any],
    table_context: str = "",
    config: Optional[AgentPipelineConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """For each unique join_key, call ADK Agent. Returns {join_key: {col: value_or_none}}."""
    if config is None:
        config = AgentPipelineConfig(config_dict=load_config())
    meta = load_query_table_metadata(base_dir, table_folder)
    if not meta:
        return {}
    res = meta.get("resource") or {}
    table_context = table_context or res.get("description", "") or res.get("name", table_folder)
    data_filename = config.config.get("data", {}).get("data_filename", "rows.csv")
    path = Path(base_dir) / table_folder / data_filename
    if not path.exists():
        path = Path(base_dir) / table_folder / "rows.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    jc = join_column if isinstance(join_column, str) else (join_column[0] if join_column else None)
    if jc not in df.columns:
        return {}
    aug_cols = _normalize_suggested_columns(suggested_columns)
    if not aug_cols:
        return {}

    instruction = _load_prompt_section("GENERATE_VALUES")
    if not instruction:
        instruction = "Provide values from your knowledge. Return JSON. Use null if you do not know."
    llm = _build_llm_for_agent(config)
    agent = Agent(
        name="AugmentValueAgent",
        model=llm,
        instruction=instruction,
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )
    join_keys = df[jc].dropna().astype(str).str.strip().unique().tolist()
    if len(join_keys) > 500:
        join_keys = join_keys[:500]
    results = {}
    for jk in join_keys:
        runner = InMemoryRunner(agent=agent)  # Fresh session per join_key to avoid history accumulation
        prompt = _build_generate_prompt(jk, aug_cols, table_context)
        try:
            events = await runner.run_debug(prompt)
            last_text = _extract_last_text_from_events(events)
            parsed = _extract_json(last_text)
            row_vals = {}
            for c in aug_cols:
                name = c.get("column_name", "")
                v = parsed.get(name)
                if v is None or (isinstance(v, str) and str(v).lower() in ("null", "nan", "none", "")):
                    row_vals[name] = None
                else:
                    row_vals[name] = v
            results[jk] = row_vals
        except Exception:
            results[jk] = {c.get("column_name", ""): None for c in aug_cols}
    return results


# ---------------------------------------------------------------------------
# Full pipeline: suggest -> generate -> save
# ---------------------------------------------------------------------------

async def run_llm_augment_and_save(
    base_dir: str,
    table_folder: str,
    join_column: str,
    target_column: str,
    task_type: str,
    user_intent: str,
    config: Optional[AgentPipelineConfig] = None,
    llm_rows_file: Optional[str] = None,
    llm_metadata_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Full pipeline: suggest columns -> generate values -> save rows_llm.csv and metadata_llm.json."""
    if config is None:
        config = AgentPipelineConfig(config_dict=load_config())
    base_path = Path(base_dir) / table_folder
    meta = load_query_table_metadata(base_dir, table_folder)
    if not meta:
        return {"error": "metadata not found", "rows_path": None, "metadata_path": None}
    res = meta.get("resource") or {}

    suggested = await suggest_llm_augment_columns(
        base_dir=base_dir,
        table_folder=table_folder,
        join_column=join_column,
        target_column=target_column,
        task_type=task_type,
        user_intent=user_intent,
        config=config,
    )
    aug_cols = _normalize_suggested_columns(suggested)
    if not aug_cols:
        return {"error": "no suggested columns", "rows_path": None, "metadata_path": None}

    values_by_key = await generate_llm_augment_values(
        base_dir=base_dir,
        table_folder=table_folder,
        join_column=join_column,
        suggested_columns=aug_cols,
        table_context=res.get("description", ""),
        config=config,
    )

    data_filename = config.config.get("data", {}).get("data_filename", "rows.csv")
    csv_path = base_path / data_filename
    if not csv_path.exists():
        csv_path = base_path / "rows.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    jc = join_column if isinstance(join_column, str) else (join_column[0] if join_column else None)
    if jc not in df.columns:
        return {"error": f"join_column {jc} not in table", "rows_path": None, "metadata_path": None}

    for c in aug_cols:
        name = c.get("column_name", "")
        df[name] = df[jc].astype(str).str.strip().map(
            lambda k: values_by_key.get(k, {}).get(name)
        )

    output_cfg = config.config.get("output", {})
    rows_filename = llm_rows_file or output_cfg.get("llm_rows_file", "rows_llm.csv")
    meta_filename = llm_metadata_file or output_cfg.get("llm_metadata_file", "metadata_llm.json")
    out_csv = base_path / rows_filename
    df.to_csv(out_csv, index=False, encoding="utf-8")

    meta_out = dict(meta)
    r = meta_out.get("resource") or {}
    r = dict(r)
    r.setdefault("columns_name", [])
    r.setdefault("columns_description", [])
    r.setdefault("columns_datatype", [])
    r.setdefault("columns_field_name", [])
    for c in aug_cols:
        name = c.get("column_name", "")
        if name and name not in r["columns_name"]:
            r["columns_name"].append(name)
            r["columns_description"].append(c.get("description", ""))
            r["columns_datatype"].append("Number" if c.get("column_type") == "numerical" else "Text")
            r["columns_field_name"].append(name.lower().replace(" ", "_").replace("-", "_"))
    meta_out["resource"] = r

    out_meta = base_path / meta_filename
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    return {
        "rows_path": str(out_csv),
        "metadata_path": str(out_meta),
        "suggested_columns": [c.get("column_name", "") for c in aug_cols],
        "augment_cols_added": len(aug_cols),
    }

# Task config for each table (join_table -> task_type, user_intent)
TASKS = {
    "COVID-Chicago": {"task_type": "regression", "user_intent": "Predict the daily mortality rate of covid in Chicago"},
    "Demo-Chicago": {"task_type": "regression", "user_intent": "Predict the ratio of people without high school diploma using Public Health data in Chicago"},
    "Economic-Chicago": {"task_type": "regression", "user_intent": "Predict the number of people having annual income lower than $25000 using community survey data in Chicago"},
    "Education-Chicago": {"task_type": "classification", "user_intent": "predict the public school performance from 2011-2012 in Chicago"},
    "COVID-NYC": {"task_type": "regression", "user_intent": "Predict the daily mortality rate of covid in NYC"},
    "Demo-NYC": {"task_type": "classification", "user_intent": "predict the education level of people in NYC based on the data about poverty in 2018"},
    "Economic-NYC": {"task_type": "classification", "user_intent": "predict the household/family type in NYC using the poverty data in 2018"},
    "Education-NYC": {"task_type": "classification", "user_intent": "predict the grade of school in nyc using the education record in 2009-2010"},
}


def main():
    parser = argparse.ArgumentParser(description="LLM augment: suggest & generate augment columns per join_key")
    parser.add_argument("table_name", type=str, help="Query table folder name, e.g. Education-NYC")
    parser.add_argument("--base-dir", type=str, default="query_table", help="Base dir for query tables")
    parser.add_argument("--rows-file", type=str, default=None, help="Output rows CSV filename (e.g. rows_gpt4o.csv)")
    parser.add_argument("--metadata-file", type=str, default=None, help="Output metadata JSON filename (e.g. metadata_gpt4o.json)")
    args = parser.parse_args()

    _project = Path(__file__).resolve().parent.parent
    perturb_path = _project / "configs" / "perturbation.yaml"
    if not perturb_path.exists():
        print(f"Error: {perturb_path} not found")
        return 1
    with open(perturb_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    table_cfg = cfg.get("tables", {}).get(args.table_name)
    if not table_cfg:
        print(f"Error: table '{args.table_name}' not in configs/perturbation.yaml tables")
        return 1
    join_cols = table_cfg.get("join_columns", [])
    join_column = join_cols[0] if join_cols else None
    target_column = table_cfg.get("target_column", "")
    if not join_column:
        print(f"Error: no join_columns for {args.table_name}")
        return 1
    task_info = TASKS.get(args.table_name, {})
    task_type = task_info.get("task_type", "regression")
    user_intent = task_info.get("user_intent", f"Predict {target_column}")

    print(f"Running llm_augment for {args.table_name} (join={join_column}, target={target_column})")
    result = asyncio.run(run_llm_augment_and_save(
        base_dir=args.base_dir,
        table_folder=args.table_name,
        join_column=join_column,
        target_column=target_column,
        task_type=task_type,
        user_intent=user_intent,
        llm_rows_file=args.rows_file,
        llm_metadata_file=args.metadata_file,
    ))
    print(result)
    return 0 if result.get("rows_path") else 1


if __name__ == "__main__":
    exit(main())