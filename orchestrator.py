import json
import re
import os
import pandas as pd
import asyncio # Required for running the async entry point
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from google.adk.runners import InMemoryRunner
from google.genai import types
from table_selection_agent import build_table_selection_agent
from join_column_selection_agent import build_join_column_choose_agent
from callback import JoinValidatorCallback, AugmentValidatorCallback
import fasttext
from functools import partial
from llm_agent_tools import find_dataset_dir, build_opendata_search_params, get_fasttext_sim
from augment_column_selection_agent import build_utility_gain_agent
from agent_config_loader import AgentPipelineConfig
from analyze_user_intent_agent import build_analyze_user_intent_agent
from datalake_client import SocrataDatalakeClient
from datetime import datetime
from utils.sketch import (
    get_candidate_table,
    _update_table_access_status,
    bottom_k_sketch_column,
    select_join_columns_for_candidate,
)
from utils.column_descriptions import (
    get_column_descriptions_from_index,
    get_column_descriptions_from_local_metadata,
)

# Local cache for opendata dataset metadata (skip re-fetch if already read)
OPENDATA_METADATA_CACHE_DIR = Path(__file__).resolve().parent / "opendata_metadata_cache"
SESSION_CHECKED_DIR = Path(__file__).resolve().parent / "session_checked"

def _session_checked_path(session_id: str) -> Path:
    """Sanitize session_id for use as filename."""
    safe = re.sub(r"[^\w\-]", "_", str(session_id).strip()) or "default"
    return SESSION_CHECKED_DIR / f"{safe}.json"

def _load_session_checked(session_id: Optional[str]) -> tuple:
    """Load checked table for this session. Returns (checked_set, checked_table).
    checked_set: set of IDs for exclude_tables. checked_table: list of full entries (dict with id, description, possible_join_column, etc.).
    Old format (list of IDs only) -> checked_table=[], checked_set=those IDs."""
    if not session_id:
        return set(), []
    path = _session_checked_path(session_id)
    if not path.exists():
        return set(), []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set(), []
    if isinstance(data, list) and data:
        if isinstance(data[0], dict) and data[0].get("id") is not None:
            checked_table = [e for e in data if e and isinstance(e, dict) and e.get("id")]
            checked_set = {str(e.get("id")).strip() for e in checked_table}
            return checked_set, checked_table
        ids = [i for i in data if i is not None]
        checked_set = {str(i).strip() for i in ids}
        return checked_set, []
    if isinstance(data, dict) and data.get("checked_tables"):
        tbl = data["checked_tables"]
        if tbl and isinstance(tbl[0], dict):
            checked_table = [e for e in tbl if e and isinstance(e, dict) and e.get("id")]
            checked_set = {str(e.get("id")).strip() for e in checked_table}
            return checked_set, checked_table
        checked_set = {str(i).strip() for i in tbl if i}
        return checked_set, []
    return set(), []

def _save_session_checked(session_id: Optional[str], checked_table: list) -> None:
    """Save checked table (list of full entries) for this session."""
    if not session_id:
        return
    path = _session_checked_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checked_table, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _metadata_cache_path(domain: str, dataset_id: str) -> Path:
    safe_id = re.sub(r"[^\w\-]", "_", dataset_id)
    return OPENDATA_METADATA_CACHE_DIR / domain.replace(".", "_") / f"{safe_id}.json"

def _load_metadata_from_cache(domain: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    path = _metadata_cache_path(domain, dataset_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_metadata_to_cache(domain: str, dataset_id: str, data: Dict[str, Any]) -> None:
    path = _metadata_cache_path(domain, dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _debug_log(payload: Dict[str, Any]) -> None:
    try:
        payload.setdefault("timestamp", int(time.time() * 1000))
        with open("/localdisk3/ytang49/opendata/.cursor/debug.log", "a") as debug_file:
            debug_file.write(json.dumps(payload) + "\n")
    except Exception:
        pass

# Environment setup
for key in ["GOOGLE_API_KEY", "OPENAI_API_KEY"]:
    val = os.getenv(key)
    if val: os.environ[key] = val

def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from model output."""
    # Check if text looks like markdown (agent might be waiting for approval)
    if text.strip().startswith("###") or ("**" in text and "{" not in text):
        # Agent is likely showing chain-of-thoughts, not final JSON
        # Return empty structure
        return {"relevant_tables": []}
    
    try:
        # First try standard JSON parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON-like structure (handle Python dict with single quotes)
        try:
            # Use ast.literal_eval for Python dict syntax
            import ast
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Try regex to find JSON object
            match = re.search(r'\{.*\}', text, flags=re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    # Try standard JSON first
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try replacing single quotes with double quotes
                    json_str_fixed = json_str.replace("'", '"')
                    try:
                        return json.loads(json_str_fixed)
                    except json.JSONDecodeError:
                        # Last resort: use ast.literal_eval
                        try:
                            return ast.literal_eval(json_str)
                        except (ValueError, SyntaxError):
                            pass
            # If no JSON found, return empty structure
            return {"relevant_tables": []}

async def run_orchestrator(
    join_table_name: Optional[str] = None,
    join_column: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    user_intent: Optional[str] = None,  
    session_id: Optional[str] = None,
    config_path: Optional[str] = None,
    config: Optional[AgentPipelineConfig] = None
) -> Dict[str, Any]:
    """
    Run the multi-agent data augmentation pipeline.
    
    Args:
        join_table_name: Name of base/join table. If None, uses config.
        join_column: Join columns. If None, uses config.
        target_column: Target column for prediction. If None, uses config.
        task_type: Task type ("regression" or "classification"). If None, uses config.
        user_intent: User's intent/prediction goal (e.g., "predict the shooting count in each borough"). 
                     If None, will be constructed from target_column.
        session_id: Unique session identifier. If None, will be generated.
        config_path: Path to config file. If None, uses default.
        config: AgentPipelineConfig object. If provided, uses this instead of loading from file.
    
    Returns:
        Dictionary with pipeline results.
    """
    # Load config if not provided
    if config is None:
        config = AgentPipelineConfig(config_path)
    
    # Use config values if parameters not provided
    if join_table_name is None:
        join_table_name = config.join_table_name
    if join_column is None:
        join_column = config.join_column
    if target_column is None:
        target_column = config.target_column
    if task_type is None:
        task_type = config.task_type
    
    # Handle user_intent: prioritize parameter, then config, then construct default
    if user_intent is None:
        user_intent = getattr(config, 'user_intent', None)
    if user_intent is None:
        user_intent = f"predict the {target_column}"  # Default fallback
    
    session_id = config.session_id
    BASE_DIR = config.base_dir
    
    # Build agents with config
    analyze_intent_runner = InMemoryRunner(agent=build_analyze_user_intent_agent(config=config))
    table_runner = InMemoryRunner(agent=build_table_selection_agent(config=config))
    joincol_runner = InMemoryRunner(agent=build_join_column_choose_agent(config=config))

    base_path = Path(BASE_DIR)
    candidate_names = [
        item.name for item in base_path.iterdir()
        if item.is_dir() and (item / "metadata.json").exists() and item.name != join_table_name
    ]

#---- Phase 1: Table Selection ----

    print("🚀 Running Table Selection Agent...")
    print(f"📝 User Intent: {user_intent}")

    analyze_intent_prompt = f"""
User Intent: {user_intent}

Task Information:
- Target Column: {target_column}
- Task Type: {task_type}
- Join Table: {join_table_name}
- Join Columns: {join_column}

Please analyze the user intent and return the result in JOSN format according to the prompt.
"""

    # Run agent, call analyze_user_intent
    # Run analyze_user_intent agent and extract result
    analyze_intent_events = await analyze_intent_runner.run_debug(analyze_intent_prompt)
    last_text = ""
    for event in analyze_intent_events:
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                t = getattr(part, "text", None)
                if t:
                    last_text = t
    analyzed_intent = extract_json(last_text) if "domain_field" in last_text else None

    dimension_specifications = {}

    for dim_key, dim_display_name in [
        ("domain_field", "Domain/Field"),
        ("geographic", "Geographic"),
        ("temporal", "Temporal"),
        ("population_group", "Population Group"),
    ]:
        dim_info = analyzed_intent.get(dim_key) if analyzed_intent else None
        if not isinstance(dim_info, dict):
            continue
        is_explicitly_mentioned = dim_info.get("is_explicitly_mentioned") is True
        explicitly_mentioned_value = dim_info.get("explicitly_mentioned_value")

        if is_explicitly_mentioned:
            raw = explicitly_mentioned_value
            if isinstance(raw, list):
                value_str = ", ".join(str(x) for x in raw) if raw else ""
            else:
                value_str = str(raw) if raw else ""
            print(f"\nDimension '{dim_display_name}' is set to: {value_str}")
            print("Reply 'done' to confirm, or type the correct value.")
            user_input = input("Your reply: ").strip()
            if user_input.lower() == "done":
                confirmed_value = explicitly_mentioned_value
            else:
                confirmed_value = user_input
            dimension_specifications[dim_display_name] = [confirmed_value] if not isinstance(confirmed_value, list) else (confirmed_value if isinstance(confirmed_value, list) else [confirmed_value])
        else:
            suggested = dim_info.get("suggested_values") or []
            sug_str = f" Suggested: {', '.join(str(x) for x in suggested)}." if suggested else ""
            print(f"\nDimension '{dim_display_name}' was not specified in your intent.")
            print(f"Enter a value{sug_str} or type 'skip' to skip.")
            user_input = input("Your reply: ").strip()
            if user_input.lower() != "skip" and user_input:
                dimension_specifications[dim_display_name] = [user_input]
            else:
                dimension_specifications[dim_display_name] = ['all'] 
    print("="*40, "Dimension Specifications", "="*40)
    print(dimension_specifications)

    # Call Opendata API when data_source is datalake
    opendata_metadata = None
    data_source = config.config.get("data", {}).get("data_source", "local")
    if data_source == "datalake":
        join_column = ", ".join(join_column) if isinstance(join_column, list) else join_column
        target_column = str(target_column).strip() if target_column else None
        search_domains, search_q = build_opendata_search_params(dimension_specifications, join_column, target_column)
        print("="*80)        
        print(f"Search Domains: {search_domains}")
        print(f"Search Query: {search_q}")
        print("="*80)        
        datalake_config = config.config.get("data", {}).get("datalake", {})
        client = SocrataDatalakeClient(datalake_config)
        max_tables = datalake_config.get("max_tables") or 10
        domain_for_fetch = (search_domains[0] if search_domains else None) or (datalake_config.get("domains") or [None])[0]
        join_columns_list = config.join_column if isinstance(config.join_column, list) else [config.join_column]
        join_columns_set = {str(j).strip().lower() for j in join_columns_list if j}
        index_path = Path(__file__).resolve().parent / "opendata_table_index.json"
        exclude_base = [join_table_name] if join_table_name else []
        checked_set, checked_table = _load_session_checked(session_id)

        # Load existing index (cumulative)
        existing_index = []
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    existing_index = json.load(f)
                if not isinstance(existing_index, list):
                    existing_index = []
            except Exception:
                existing_index = []
        existing_id_set = {str(e.get("id")).strip() for e in existing_index if e and e.get("id")}

        new_entries = []
        candidate_ids_for_run = []
        api_fetch_limit = 10000

        if domain_for_fetch:
            # exclude_tables = exclude_base + list(checked_set)
            exclude_tables = exclude_base
            opendata_metadata = client.read_metadata(
                search_domains=search_domains,
                search_q=search_q if search_q.strip() else None,
                exclude_tables=exclude_tables,
                limit=api_fetch_limit,
                offset=0,
            )
            batch = list((opendata_metadata.get("metadata_by_dataset") or {}).keys())
            print(f"[Opendata] One-time fetch: {len(batch)} tables (API limit={api_fetch_limit}, user max_tables={max_tables})")

            for ds_id in batch:
                if len(candidate_ids_for_run) >= max_tables:
                    break
                ds_id_str = str(ds_id).strip()
                if ds_id_str in checked_set:
                    continue
                if ds_id_str in existing_id_set:
                    candidate_ids_for_run.append(ds_id_str)
                    if session_id:
                        for entry in existing_index:
                            if entry and str(entry.get("id")).strip() == ds_id_str:
                                cols_name = entry.get("columns_name") or []
                                possible = [c for c in cols_name if c and str(c).strip().lower() in join_columns_set]
                                session_entry = dict(entry)
                                session_entry["possible_join_column"] = possible
                                checked_table.append(session_entry)
                                break
                        checked_set.add(ds_id_str)
                        _save_session_checked(session_id, checked_table)
                    continue
                full_meta = _load_metadata_from_cache(domain_for_fetch, ds_id)
                if full_meta is None:
                    full_meta = client.get_dataset_metadata(ds_id, domain_for_fetch)
                    if "error" not in full_meta:
                        _save_metadata_to_cache(domain_for_fetch, ds_id, full_meta)
                if full_meta and "error" not in full_meta:
                    res = full_meta.get("resource") or {}
                    meta_id = full_meta.get("id") or res.get("id")
                    meta_desc = full_meta.get("description") or res.get("description") or ""
                    meta_attr = full_meta.get("attribution") or res.get("attribution")
                    cols_name = list(res.get("columns_name") or [])
                    cols_desc = [str((c.get("description") or "")).strip() for c in full_meta.get("columns") if isinstance(c, dict)]
                    if not cols_name:
                        for col in full_meta.get("columns") or []:
                            if isinstance(col, dict) and col.get("name"):
                                cols_name.append(col["name"])
                    possible = [c for c in cols_name if c and str(c).strip().lower() in join_columns_set]
                    entry = {"id": meta_id, "description": meta_desc, "attribution": meta_attr}
                    entry["columns_name"] = res.get("columns_name") or cols_name
                    entry["columns_description"] = cols_desc 
                    raw_class = full_meta.get("classification") or {}
                    if not raw_class and (full_meta.get("category") is not None or full_meta.get("tags")):
                        raw_class = {
                            "categories": [],
                            "tags": list(full_meta.get("tags") or []),
                            "domain_category": str(full_meta.get("category") or "").strip(),
                            "domain_tags": [],
                        }

                    entry["classification"] = {
                        "categories": raw_class.get("categories") if isinstance(raw_class, dict) else [],
                        "tags": raw_class.get("tags") if isinstance(raw_class, dict) else [],
                        "domain_category": raw_class.get("domain_category") or "",
                        "domain_tags": raw_class.get("domain_tags") or [],
                    }   

                    entry["domain"] = (full_meta.get("metadata") or {}).get("domain") or domain_for_fetch
                    new_entries.append(entry)
                    existing_id_set.add(str(meta_id).strip())
                    candidate_ids_for_run.append(ds_id_str)
                    if session_id:
                        session_entry = dict(entry)
                        session_entry["possible_join_column"] = possible
                        session_entry["possible_join_column_is_true"] = bool(possible)
                        checked_table.append(session_entry)
                        checked_set.add(ds_id_str)
                        _save_session_checked(session_id, checked_table)

        merged_index = existing_index + new_entries
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(merged_index, f, indent=2, ensure_ascii=False)
        print(f"[Index] cumulative: {len(existing_index)} existing + {len(new_entries)} new = {len(merged_index)} total, {len(candidate_ids_for_run)} for this run (limit={max_tables})")

        table_selection_prompt = f"""Candidate table IDs from Opendata search (call read_table_index with these IDs to load their index):
candidate_ids = {candidate_ids_for_run}

User Intent: {user_intent}

Task Information:
- Target Column: {target_column}
- Task Type: {task_type}
- Join Table: {join_table_name}
- Join Column(s): {join_column}
- Confirmed dimension specifications: {json.dumps(dimension_specifications, ensure_ascii=False)}

Call read_table_index(candidate_ids={candidate_ids_for_run}) to get index entries, then select relevant tables according to your prompt and return JSON with key "relevant_tables"."""

        print("\n📋 Running Table Selection Agent (datalake)...")
        table_selection_events = await table_runner.run_debug(table_selection_prompt)
        last_table_text = ""
        for event in table_selection_events:
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for part in event.content.parts:
                    t = getattr(part, "text", None)
                    if t:
                        last_table_text = t
        table_data = extract_json(last_table_text) if last_table_text.strip() else {}
        relevant_list = table_data.get("relevant_tables", [])
        print(f"[Table Selection] {len(relevant_list)} tables selected")
        for i, tbl in enumerate(relevant_list[:10], 1):
            name = tbl.get("table_name", "?")
            print(f"   {i}. {name}")
        if len(relevant_list) > 10:
            print(f"   ... and {len(relevant_list) - 10} more")

        #---- Phase 2: Join Column Selection ----

        run_start_time = datetime.now()
        run_record = {
            "table_id": [],
            "status": [],
            "reason": [],
        }
        topk_jaccard = {}  # table_id -> list of selected column names
        final_selected_tables = []

        # Load join table and create join column sketch (support composite key)
        real_join_table_name = find_dataset_dir(join_table_name, BASE_DIR)
        join_df = pd.read_csv(Path(BASE_DIR) / real_join_table_name / config.data_filename, low_memory=False)
        join_columns = config.join_column if isinstance(config.join_column, list) else [config.join_column]
        
        # Create sketch for each join column
        if len(join_columns) == 1:
            join_sketch = bottom_k_sketch_column(join_df[join_columns[0]])
        else:
            join_sketch = {jc: bottom_k_sketch_column(join_df[jc]) for jc in join_columns if jc in join_df.columns}

        # Join column descriptions
        join_col_descs = get_column_descriptions_from_local_metadata(BASE_DIR, real_join_table_name)

        topk_join = config.config.get("task", {}).get("topk_join_columns", 5) or 5

        for tbl in relevant_list:
            cand_name = tbl.get("table_name")
            if not cand_name:
                continue
            run_record["table_id"].append(cand_name)

            df, status = get_candidate_table(table_id=cand_name, opendata_domain=domain_for_fetch)
            if status["success"]:
                run_record["status"].append("success")
                run_record["reason"].append(None)
                cols_with_jaccard = select_join_columns_for_candidate(
                    join_sketch, df, k_columns=topk_join, min_jaccard=0.5
                )
                cand_col_descs = get_column_descriptions_from_index(cand_name)

                # Handle composite key vs single key
                if isinstance(cols_with_jaccard, dict):
                    # Composite key: {join_col: [(cand_col, jaccard), ...], ...}
                    join_info = [
                        {
                            "join_column": jc,
                            "description": join_col_descs.get(jc, ""),
                            "top_candidates": [
                                {"name": col, "jaccard": round(j, 4), "description": cand_col_descs.get(col, "")}
                                for col, j in cols_with_jaccard.get(jc, [])
                            ]
                        }
                        for jc in join_columns
                    ]
                    jc_prompt = f"""Select the best matching columns for composite join key from candidate table {cand_name}.

Composite join key from query table:
{json.dumps(join_info, ensure_ascii=False, indent=2)}

For each join_column, select the single best candidate column from top_candidates.

Return JSON:
{{
  "selected_columns": {{"join_col1": "cand_col1", "join_col2": "cand_col2"}},
  "reasoning": "..."
}}"""
                    has_candidates = any(len(info["top_candidates"]) > 0 for info in join_info)
                else:
                    # Single key: [(col, jaccard), ...]
                    candidates_for_llm = [
                        {"name": col, "jaccard": round(j, 4), "description": cand_col_descs.get(col, "")}
                        for col, j in cols_with_jaccard
                    ]
                    jc_prompt = f"""Select the join column for candidate table {cand_name}.

query_join_column: {join_columns[0]}
query_join_column_description: {join_col_descs.get(join_columns[0], "")}
candidate_columns: {json.dumps(candidates_for_llm, ensure_ascii=False)}

Return JSON with selected_columns and reasoning."""
                    has_candidates = len(cols_with_jaccard) > 0

                if has_candidates:
                    try:
                        jc_events = await joincol_runner.run_debug(jc_prompt)
                        last_text = ""
                        for event in reversed(jc_events):
                            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                                for part in event.content.parts:
                                    t = getattr(part, "text", None)
                                    if t:
                                        last_text = t
                                        break
                            if last_text:
                                break
                        jc_json = extract_json(last_text.strip()) if last_text else {}
                        
                        # Parse selected_columns based on composite vs single
                        if isinstance(cols_with_jaccard, dict):
                            # Composite: {"BORO": "Borough", "YEAR": "report_year"}
                            selected_dict = jc_json.get("selected_columns", {})
                            selected_cols = list(selected_dict.values()) if isinstance(selected_dict, dict) else []
                        else:
                            # Single: ["Borough"]
                            selected_cols = jc_json.get("selected_columns", []) or []
                            if not isinstance(selected_cols, list):
                                selected_cols = [selected_cols] if selected_cols else []
                    except Exception as e:
                        print(f"   ⚠️  Join Agent error: {e}")
                        # Fallback to top jaccard
                        if isinstance(cols_with_jaccard, dict):
                            selected_cols = [cols_with_jaccard[jc][0][0] for jc in join_columns if cols_with_jaccard.get(jc)]
                        else:
                            selected_cols = [cols_with_jaccard[0][0]] if cols_with_jaccard else []
                else:
                    selected_cols = []

                # Store topk_jaccard for record
                if isinstance(cols_with_jaccard, dict):
                    topk_jaccard[cand_name] = {jc: [c for c, _ in cols_with_jaccard[jc]] for jc in cols_with_jaccard}
                else:
                    topk_jaccard[cand_name] = [c for c, _ in cols_with_jaccard]
                if selected_cols:
                    tbl_with_join = dict(tbl)
                    tbl_with_join["candidate_table"] = cand_name
                    tbl_with_join["selected_columns"] = selected_cols
                    tbl_with_join["join_col_descs"] = join_col_descs
                    tbl_with_join["cand_col_descs"] = cand_col_descs
                    final_selected_tables.append(tbl_with_join)
                    print(f"   ✅ {cand_name}: LLM selected {selected_cols}")
            else:
                run_record["status"].append("failed")
                run_record["reason"].append(status["reason"])
                topk_jaccard[cand_name] = []

        run_record["topk_jaccard"] = topk_jaccard
        relevant_list = final_selected_tables
        join_columns_list = config.join_column if isinstance(config.join_column, list) else [config.join_column]

        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_sid = re.sub(r"[^\w\-]", "_", str(session_id or "default").strip()) or "default"
        filename = run_start_time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{safe_sid}.json"
        filepath = data_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2, ensure_ascii=False)
        print(f"[Run Record] Saved to {filepath}")


# ---- Phase 3: Augment Column Selection ----

        # Aggregation phase
        print(f"\n📊 Aggregating candidate tables by join key...")
        aggregated_results = aggregate_selected_tables(
            final_selected_tables,
            base_dir=BASE_DIR,
            opendata_domain=domain_for_fetch
        )




    # print(f"\n📊 Starting Augment Column Selection...")
    # print(f"   Target: {target_column} ({task_type})")
    
    # utility_runner = InMemoryRunner(agent=build_utility_gain_agent(config=config))
    # augment_results = []
    
    # # Process each table that passed Phase 2
    # for result in relevant_list:
    #     cand_name = result.get("candidate_table")
    #     if not cand_name:
    #         continue
    #     selected_join_cols = result.get("selected_columns", [])  # Join columns from Phase 2
        
    #     print(f"\n🔍 Evaluating columns in '{cand_name}' for augmenting '{target_column}'...")
        
    #     # Load candidate table to get all available columns
    #     cand_df = None
    #     if data_source == "datalake" and domain_for_fetch:
    #         try:
    #             rows = client.read_data(cand_name, domain_for_fetch, max_rows=config.sample_size * 2)
    #             cand_df = pd.DataFrame(rows) if rows else None
    #         except Exception as e:
    #             print(f"   ⚠️ API fetch failed for {cand_name}: {e}")
    #     if cand_df is None or cand_df.empty:
    #         try:
    #             real_cand_name = find_dataset_dir(cand_name, BASE_DIR)
    #             cand_df = pd.read_csv(Path(BASE_DIR) / real_cand_name / config.data_filename, low_memory=False)
    #         except Exception as e:
    #             print(f"   ⚠️ Local load failed for {cand_name}: {e}")
    #             continue
    #     if cand_df.empty or len(cand_df.columns) == 0:
    #         continue
        
    #     # Get candidate columns to evaluate (exclude join columns)
    #     candidate_columns = [
    #         col for col in cand_df.columns 
    #         if col not in selected_join_cols
    #     ]
        
    #     if len(candidate_columns) == 0:
    #         print(f"   ⚠️  No columns available for augmentation (all are join columns)")
    #         continue
        
    #     print(f"   Checking {len(candidate_columns)} candidate columns...")
        
    #     column_results = []
        
    #     for col in candidate_columns:
    #         try:
    #             print(f"      Checking: {col}")
                
    #             ug_prompt = f"""
    #             Compute utility gain and evaluate suitability with these parameters:
    #             - base_table_name: "{join_table_name}"
    #             - candidate_table_name: "{cand_name}"
    #             - base_join_columns: {config.join_column}
    #             - candidate_join_columns: {selected_join_cols}
    #             - candidate_column: "{col}"
    #             - target_column: "{target_column}"
    #             - task_type: "{task_type}"
    #             - base_dir: "{BASE_DIR}"
    #             - sample_size: {config.sample_size}
    #             - opendata_domain: "{domain_for_fetch or ''}"
                
    #             Call compute_integration_quality, compute_feature_importance, and compute_utility_gain_from_params.
    #             Based on IQ, FI, and Utility Gain values, determine if this column is suitable for augmentation.
    #             Return the JSON result with iq, fi, utility_gain, is_suitable, and reason.
    #             """
                
    #             ug_events = await utility_runner.run_debug(ug_prompt)
                
    #             ug_json_str = "{}"
    #             for event in reversed(ug_events):
    #                 if hasattr(event, 'actions') and getattr(event.actions, "state_delta", None):
    #                     if "utility_gain_result" in getattr(event.actions, "state_delta", None):
    #                         ug_json_str = getattr(event.actions, "state_delta", None)["utility_gain_result"]
    #                         break
                
    #             ug_result = extract_json(ug_json_str)
                
    #             # Handle string JSON
    #             if isinstance(ug_result, str):
    #                 ug_result = extract_json(ug_result)
                
    #             # Check if result is a dictionary before using 'in' operator
    #             if not isinstance(ug_result, dict):
    #                 print(f"         ❌ Error: Unexpected result type {type(ug_result)}: {ug_result}")
    #                 continue
                
    #             if "error" in ug_result:
    #                 print(f"         ❌ Error: {ug_result.get('error', 'Unknown error')}")
    #                 continue
                
    #             column_results.append({
    #                 "column": col,
    #                 "iq": ug_result.get("iq", 0.0),
    #                 "fi": ug_result.get("fi", 0.0),
    #                 "utility_gain": ug_result.get("utility_gain", 0.0),
    #                 "is_suitable": ug_result.get("is_suitable", False),
    #                 "reason": ug_result.get("reason", "")
    #             })
                
    #             status = "✓" if ug_result.get("is_suitable", False) else "✗"
    #             print(f"         {status} UG: {ug_result.get('utility_gain', 0.0):.4f} - {ug_result.get('reason', '')}")
                
    #             await asyncio.sleep(config.delay_between_columns)
                
    #         except Exception as e:
    #             print(f"         ❌ Error evaluating {col}: {e}")
    #             continue
        
    #     # Filter to only suitable columns and sort by utility_gain
    #     suitable_columns = [r for r in column_results if r.get("is_suitable", False)]
    #     suitable_columns.sort(key=lambda x: x["utility_gain"], reverse=True)
        
    #     augment_results.append({
    #         "candidate_table": cand_name,
    #         "join_columns": selected_join_cols,
    #         "all_evaluated_columns": column_results,
    #         "suitable_columns": suitable_columns,
    #         "total_evaluated": len(column_results),
    #         "total_suitable": len(suitable_columns)
    #     })
        
    #     print(f"   ✅ Found {len(suitable_columns)} suitable columns out of {len(column_results)} evaluated")
    
    # augment_callback = AugmentValidatorCallback(
    #     base_table_df=join_df,
    #     target_column=target_column,
    #     task_type=task_type,
    #     join_columns=join_columns_list,
    #     base_dir=BASE_DIR,
    #     config=config
    # )

    # # Validate each candidate table's suitable columns
    # for result in augment_results:
    #     cand_name = result["candidate_table"]
    #     suitable_cols = result["suitable_columns"]
    #     selected_join_cols = result["join_columns"]
        
    #     if len(suitable_cols) == 0:
    #         print(f"   ⚠️  No suitable columns found in '{cand_name}'")
    #         continue
        
    #     # Extract suitable columns names
    #     selected_column_names = [col["column"] for col in suitable_cols]
        
    #     print(f"\n🔬 Validating augmentation for '{cand_name}' with {len(selected_column_names)} columns...")
        
    #     # Validate: merge selected columns and run task
    #     validation_result = augment_callback.verify(
    #         candidate_table_name=cand_name,
    #         selected_columns=selected_column_names,
    #         candidate_join_columns=selected_join_cols,
    #         opendata_domain=domain_for_fetch if data_source == "datalake" else None,
    #     )
        
    #     # Add validation result to result
    #     result["validation"] = validation_result
        
    #     if "error" in validation_result:
    #         print(f"   ❌ Validation failed: {validation_result['error']}")
    #     else:
    #         baseline = validation_result.get("baseline_metric")
    #         augmented = validation_result.get("augmented_metric", validation_result.get("metric"))
    #         improvement = validation_result.get("improvement")
    #         improvement_pct = validation_result.get("improvement_percent")
    #         base_count = validation_result.get("base_features_count", 0)
    #         augment_count = validation_result.get("augment_features_count", 0)
    #         total_count = validation_result.get("total_features_count", 0)
            
    #         print(f"   ✅ Validation passed")
    #         if baseline is not None:
    #             print(f"      Baseline: {baseline:.4f} → Augmented: {augmented:.4f}")
    #             if improvement is not None:
    #                 sign = "+" if improvement >= 0 else ""
    #                 print(f"      Improvement: {sign}{improvement:.4f} ({sign}{improvement_pct:.2f}%)")
    #         else:
    #             print(f"      Augmented metric: {augmented:.4f}")
    #         print(f"      Features: {base_count} base + {augment_count} augment = {total_count} total")


    # return {
    #     "join_table": join_table_name,
    #     "join_column": join_columns_list,
    #     "target_column": target_column,
    #     "task_type": task_type,
    #     "joinable_tables": relevant_list,
    #     "augment_results": augment_results
    # }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multi-agent data augmentation pipeline')
    parser.add_argument('--user-intent', type=str, 
                       help='User intent/prediction goal (e.g., "I would like to predict the crime rate in New York City")')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--join-table', type=str, default=None,
                       help='Join table name')
    parser.add_argument('--target-column', type=str, default=None,
                       help='Target column to predict')
    parser.add_argument('--task-type', type=str, default=None,
                       choices=['regression', 'classification'],
                       help='Task type')
    parser.add_argument('--session-id', type=str, default=None,
                       help='Session ID for this query task (for per-session checked dataset)')    
    args = parser.parse_args()
    
    try:
        # Load config (will use default path if None)
        config = AgentPipelineConfig(args.config)
        
        # Run orchestrator with config and user_intent
        output = asyncio.run(run_orchestrator(
            config=config,
            user_intent=args.user_intent,  # Pass user_intent from command line
            session_id=args.session_id,
            join_table_name=args.join_table,
            target_column=args.target_column,
            task_type=args.task_type
        ))
        
        print("\n--- Final Results ---")
        if config.save_results:
            # Save results to file
            output_file = Path(config.results_file)
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {output_file}")
        
        if config.print_results:
            print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()