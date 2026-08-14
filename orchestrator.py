import inspect
import json
import re
import os
import pandas as pd
import asyncio # Required for running the async entry point
import time
import uuid
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from google.adk.runners import InMemoryRunner
from google.genai import types
from table_selection_agent import build_table_selection_agent
from join_column_selection_agent import build_join_column_choose_agent
from callback import JoinValidatorCallback, AugmentValidatorCallback
import fasttext
from functools import partial
from llm_agent_tools import find_dataset_dir, build_opendata_search_params, get_fasttext_sim, _train_and_evaluate
from augment_column_selection_agent import build_augment_column_selection_agent
from agent_config_loader import AgentPipelineConfig, load_config
from analyze_user_intent_agent import build_analyze_user_intent_agent
from datalake_client import SocrataDatalakeClient
from datetime import datetime
from tools.sketch import (
    get_candidate_table,
    _update_table_access_status,
    bottom_k_sketch_column,
    select_join_columns_for_candidate,
)
from tools.column_descriptions import (
    get_column_descriptions_from_index,
    get_column_descriptions_from_local_metadata,
)
from tools.aggregation import aggregate_selected_tables, aggregate_target_by_join_key
from tools.correlation import merge_target_with_candidate, compute_feature_correlations
from benchmark_perturbation.benchmark_perturbation import get_perturbed_pipeline_config
from agent_config_loader import AgentPipelineConfig


async def _close_runner_safely(runner: Any) -> None:
    """Best-effort close for ADK runners to avoid unclosed client sessions."""
    if runner is None:
        return
    for method_name in ("aclose", "close"):
        method = getattr(runner, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass
        return


async def _run_debug_fresh(build_agent_fn, prompt: str, *, quiet: bool = False, tag: str = "lin") -> List[Any]:
    """One-shot agent call with a fresh runner/session (no history carryover)."""
    runner = InMemoryRunner(agent=build_agent_fn())
    try:
        return await runner.run_debug(
            prompt,
            quiet=quiet,
            session_id=f"{tag}_{uuid.uuid4().hex[:12]}",
        )
    finally:
        await _close_runner_safely(runner)

# Local cache for opendata dataset metadata (skip re-fetch if already read)
OPENDATA_METADATA_CACHE_DIR = Path(__file__).resolve().parent / "opendata_metadata_cache"
SESSION_CHECKED_DIR = Path(__file__).resolve().parent / "session_checked"

SKIP_TABLES = {}


    
def _session_checked_path(session_id: str, base_dir: Optional[Path] = None) -> Path:
    """Sanitize session_id for use as filename."""
    safe = re.sub(r"[^\w\-]", "_", str(session_id).strip()) or "default"
    return (base_dir or SESSION_CHECKED_DIR) / f"{safe}.json"

def _load_session_checked(session_id: Optional[str], base_dir: Optional[Path] = None) -> tuple:
    """Load checked table for this session. Returns (checked_set, checked_table).
    checked_set: set of IDs for exclude_tables. checked_table: list of full entries (dict with id, description, possible_join_column, etc.).
    Old format (list of IDs only) -> checked_table=[], checked_set=those IDs."""
    if not session_id:
        return set(), []
    path = _session_checked_path(session_id, base_dir)
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

def _save_session_checked(session_id: Optional[str], checked_table: list, base_dir: Optional[Path] = None) -> None:
    """Save checked table (list of full entries) for this session."""
    if not session_id:
        return
    path = _session_checked_path(session_id, base_dir)
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
        with open("/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log", "a") as debug_file:
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


def extract_relevant_tables_from_full_text(text: str) -> Dict[str, Any]:
    """Search full model output for JSON containing relevant_tables (handles multi-part
    output where final JSON may appear before hallucinated error messages)."""
    if not text or not text.strip():
        return {"relevant_tables": []}

    candidates = []
    i = 0
    while i < len(text):
        pos = text.find("{", i)
        if pos < 0:
            break
        depth = 0
        start = pos
        for j in range(pos, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : j + 1])
                        if "relevant_tables" in obj:
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                    break
        i = pos + 1

    if not candidates:
        return {"relevant_tables": []}
    # Prefer non-empty relevant_tables; otherwise use last (most likely final answer)
    for c in candidates:
        rt = c.get("relevant_tables", [])
        if isinstance(rt, list) and len(rt) > 0:
            return c
    return candidates[-1]


def extract_json_by_key_from_full_text(
    text: str, key: str, prefer_non_empty_list: bool = True
) -> Dict[str, Any]:
    """Search full model output for JSON containing the given key (handles multi-part
    output where valid JSON may appear before hallucinated error messages)."""
    if not text or not text.strip():
        return {}

    candidates = []
    i = 0
    while i < len(text):
        pos = text.find("{", i)
        if pos < 0:
            break
        depth = 0
        start = pos
        for j in range(pos, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : j + 1])
                        if key in obj:
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                    break
        i = pos + 1

    if not candidates:
        return {}
    if prefer_non_empty_list:
        val = candidates[0].get(key)
        if isinstance(val, list) and len(val) > 0:
            return candidates[0]
        for c in candidates:
            v = c.get(key)
            if isinstance(v, list) and len(v) > 0:
                return c
    return candidates[-1]


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
    
    session_id = session_id if session_id is not None else config.session_id
    BASE_DIR = config.base_dir

    _session_dir: Optional[Path] = None
    if getattr(config, "session_checked_dir", None):
        _p = config.session_checked_dir
        _session_dir = Path(_p)
        if not _session_dir.is_absolute():
            _session_dir = (Path(__file__).resolve().parent / _p).resolve()

    # Resolve display name for LLM: use metadata resource.name (perturbed) if available, else folder name
    real_join_table_name = find_dataset_dir(join_table_name, BASE_DIR)
    base_path_obj = Path(__file__).resolve().parent / BASE_DIR
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
    
    # Build agents with config. Join/augment use fresh runners per candidate
    # (ADK run_debug reuses session_id by default and otherwise accumulates history).
    analyze_intent_runner = InMemoryRunner(agent=build_analyze_user_intent_agent(config=config))
    table_runner = InMemoryRunner(agent=build_table_selection_agent(config=config))

    base_path = Path(BASE_DIR)
    candidate_names = [
        item.name for item in base_path.iterdir()
        if item.is_dir() and (item / "metadata.json").exists()
    ]

#---- Phase 1: Table Selection ----

    print("🚀 Running Table Selection Agent...")
    print(f"📝 User Intent: {user_intent}")

    analyze_intent_prompt = f"""
User Intent: {user_intent}

Task Information:
- Target Column: {target_column}
- Task Type: {task_type}
- Join Table: {query_table_display_name}
- Join Columns: {join_column}

Please analyze the user intent and return the result in JSON format according to the prompt.
"""

    # Run agent, call analyze_user_intent
    # Run analyze_user_intent agent and extract result
    analyze_intent_events = await analyze_intent_runner.run_debug(
        analyze_intent_prompt,
        session_id=f"intent_{uuid.uuid4().hex[:12]}",
    )
    last_text = ""
    for event in analyze_intent_events:
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                t = getattr(part, "text", None)
                if t:
                    last_text = t
    # Prefer key-aware extraction: greedy extract_json() often matches CoT fragments
    # and returns a useless sentinel like {"relevant_tables": []}, which would skip
    # all interactive confirmations and leave search_query empty.
    analyzed_intent = None
    if "domain_field" in (last_text or ""):
        analyzed_intent = extract_json_by_key_from_full_text(
            last_text, "domain_field", prefer_non_empty_list=False
        )
        if not isinstance(analyzed_intent, dict) or not isinstance(
            analyzed_intent.get("domain_field"), dict
        ):
            analyzed_intent = None
    if analyzed_intent is None:
        print(
            "[warn] analyze_user_intent JSON parse failed; "
            "falling back to interactive prompts for all dimensions"
        )
        analyzed_intent = {
            "domain_field": {"is_explicitly_mentioned": False},
            "geographic": {"is_explicitly_mentioned": False},
            "temporal": {"is_explicitly_mentioned": False},
            "population_group": {"is_explicitly_mentioned": False},
        }

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

    # R1.Q1 intermediate stats (retrieval → join quality → augment → eval)
    intermediate_stats: Dict[str, Any] = {
        "schema_version": "r1q1_intermediate_v1",
        "retrieval": {},
        "table_selection": {},
        "join_column_selection": {"per_table": [], "summary": {}},
        "join_quality": {"per_table": [], "summary": {}},
        "augment": {"per_table": [], "summary": {}},
        "evaluation": {},
        "phase_timings_seconds": {},
        "dimension_specifications": dimension_specifications,
    }
    _t_pipe0 = time.perf_counter()
    _t_phase = _t_pipe0

    def _mark_phase(name: str) -> None:
        nonlocal _t_phase
        now = time.perf_counter()
        intermediate_stats["phase_timings_seconds"][name] = now - _t_phase
        _t_phase = now

    def _best_jaccard(cols_with_jaccard: Any, selected_cols: List[str], jcols: List[str]) -> Optional[float]:
        if not selected_cols:
            return None
        scores: List[float] = []
        if isinstance(cols_with_jaccard, dict):
            for jc, sel in zip(jcols, selected_cols):
                for col, j in cols_with_jaccard.get(jc, []) or []:
                    if col == sel:
                        try:
                            scores.append(float(j))
                        except Exception:
                            pass
                        break
            return max(scores) if scores else None
        for col, j in cols_with_jaccard or []:
            if col == selected_cols[0]:
                try:
                    return float(j)
                except Exception:
                    return None
        try:
            return float(cols_with_jaccard[0][1]) if cols_with_jaccard else None
        except Exception:
            return None

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
        checked_set, checked_table = _load_session_checked(session_id, _session_dir)

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
                        _save_session_checked(session_id, checked_table, _session_dir)
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
                    entry = {"id": meta_id, "name": res.get("name") or "", "description": meta_desc, "attribution": meta_attr}
                    # entry["columns_name"] = res.get("columns_name") or cols_name
                    # entry["columns_description"] = cols_desc 
                    # raw_class = full_meta.get("classification") or {}

                    entry["columns_name"] = res.get("columns_name") or cols_name
                    entry["columns_description"] = cols_desc
                    # columns_datatype: name -> dataTypeName from Socrata metadata
                    col_list = full_meta.get("columns") or []
                    name_to_dtype = {
                        str(c.get("name", "")).strip(): str(c.get("dataTypeName") or "").strip() or "unknown"
                        for c in col_list if isinstance(c, dict) and c.get("name")
                    }
                    entry["columns_datatype"] = [
                        name_to_dtype.get(str(n).strip(), "unknown") for n in entry["columns_name"]
                    ]
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
                        _save_session_checked(session_id, checked_table, _session_dir)

        merged_index = existing_index + new_entries
        # Ensure every entry has "name" (table_name) before writing
        for e in merged_index:
            if not e or not e.get("id"):
                continue
            if (e.get("name") or "").strip():
                continue
            domain = e.get("domain")
            if domain:
                meta = _load_metadata_from_cache(domain, str(e.get("id", "")).strip())
                if meta:
                    name = ((meta.get("resource") or {}).get("name") or meta.get("name") or "").strip()
                    if name:
                        e["name"] = name
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(merged_index, f, indent=2, ensure_ascii=False)
        print(f"[Index] cumulative: {len(existing_index)} existing + {len(new_entries)} new = {len(merged_index)} total, {len(candidate_ids_for_run)} for this run (limit={max_tables})")
        intermediate_stats["retrieval"] = {
            "search_domains": search_domains,
            "search_query": search_q,
            "n_api_fetched": len(locals().get("batch") or []),
            "n_candidates_for_agent": len(candidate_ids_for_run),
            "max_tables_limit": max_tables,
            "domain_for_fetch": domain_for_fetch,
        }
        _mark_phase("retrieval")
        
        real_join_table_name = find_dataset_dir(join_table_name, BASE_DIR)
        base_path = Path(__file__).resolve().parent / BASE_DIR 
        join_meta_path = base_path / real_join_table_name / "metadata.json"
        query_table_description = ""
        if join_meta_path.exists():
            try:
                with open(join_meta_path, "r", encoding="utf-8") as f:
                    join_meta = json.load(f)
                query_table_description = (join_meta.get("resource") or {}).get("description", "") or ""
            except Exception:
                pass

        # Apply prompt truncation when enabled (unified switch)
        trunc_cfg = config.config.get("agents", {}).get("prompt_truncation", {}) if config else {}
        if trunc_cfg.get("enabled"):
            max_desc = trunc_cfg.get("max_query_table_description_chars", 400)
            if query_table_description and len(query_table_description) > max_desc:
                query_table_description = query_table_description[:max_desc] + "...[truncated]"
            max_dims = trunc_cfg.get("max_dimension_specs_chars", 300)
            dims_str = json.dumps(dimension_specifications, ensure_ascii=False)
            if len(dims_str) > max_dims:
                dims_str = dims_str[:max_dims] + "...[truncated]"
        else:
            dims_str = json.dumps(dimension_specifications, ensure_ascii=False)

        table_selection_prompt = f"""Candidate table IDs from Opendata search (call read_table_index with these IDs to load their index):
candidate_ids = {candidate_ids_for_run}

User Intent: {user_intent}

Task Information:
- Target Column: {target_column}
- Task Type: {task_type}
- Query Table: {query_table_display_name}
- Join Columns(s): {join_column}
- Query Table Description: {query_table_description}
- Confirmed dimension specifications: {dims_str}

Call read_table_index(candidate_ids={candidate_ids_for_run}) to get index entries, then select relevant tables according to your prompt and return JSON with key "relevant_tables"."""

        print("\n📋 Running Table Selection Agent (datalake)...")
        table_selection_events = await table_runner.run_debug(
            table_selection_prompt,
            session_id=f"tsel_{uuid.uuid4().hex[:12]}",
        )
        full_table_text = ""
        table_data_from_state = None
        for event in table_selection_events:
            # Prefer state_delta (ADK output_key saves agent result here)
            if getattr(event, "actions", None) and getattr(event.actions, "state_delta", None):
                delta = event.actions.state_delta
                if isinstance(delta, dict) and "relevant_tables" in delta:
                    val = delta["relevant_tables"]
                    if isinstance(val, dict) and "relevant_tables" in val:
                        rt = val.get("relevant_tables", [])
                        if isinstance(rt, list) and len(rt) > 0:
                            table_data_from_state = val
                    elif isinstance(val, list) and len(val) > 0:
                        table_data_from_state = {"relevant_tables": val}
                    elif isinstance(val, str) and val.strip():
                        parsed = extract_relevant_tables_from_full_text(val)
                        if parsed.get("relevant_tables") and len(parsed["relevant_tables"]) > 0:
                            table_data_from_state = parsed
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for part in event.content.parts:
                    t = getattr(part, "text", None)
                    if t:
                        full_table_text += t
        if table_data_from_state is not None:
            table_data = table_data_from_state
        else:
            table_data = extract_relevant_tables_from_full_text(full_table_text)
        relevant_list = table_data.get("relevant_tables", [])

        # Debug: when DEBUG_TABLE_SELECTION=1 or when no tables selected (to diagnose "选不出来")
        _debug_ts = os.environ.get("DEBUG_TABLE_SELECTION", "").lower() in ("1", "true", "yes") or len(relevant_list) == 0
        if _debug_ts:
            _src = "state_delta" if table_data_from_state is not None else "full_text"
            _preview_len = 2000
            _preview = (full_table_text[: _preview_len] + "...") if len(full_table_text) > _preview_len else full_table_text
            _debug_payload = {
                "location": "orchestrator.table_selection",
                "source": _src,
                "full_text_len": len(full_table_text),
                "full_text_preview": _preview,
                "table_data_keys": list(table_data.keys()) if isinstance(table_data, dict) else None,
                "relevant_tables_count": len(relevant_list),
                "relevant_tables_sample": relevant_list[:5] if relevant_list else [],
            }
            print("\n[DEBUG] Table Selection:")
            print(f"  source={_src}, full_text_len={len(full_table_text)}, relevant_tables_count={len(relevant_list)}")
            print(f"  full_text_preview (first {min(_preview_len, len(full_table_text))} chars):\n  {repr(_preview[:500])}...")
            if relevant_list:
                print(f"  relevant_tables_sample: {_debug_payload['relevant_tables_sample']}")
            else:
                print("  relevant_tables is EMPTY — check above: did model call read_table_index? return valid JSON with key 'relevant_tables'?")
            _debug_log(_debug_payload)

        # Normalize: table_id (id), table_name (human-readable name from metadata)
        id_to_name = {}
        for e in merged_index:
            if not e or not e.get("id"):
                continue
            eid = str(e.get("id", "")).strip()
            name = (e.get("name") or "").strip()
            if not name and e.get("domain"):
                meta = _load_metadata_from_cache(e.get("domain"), eid)
                if meta:
                    name = ((meta.get("resource") or {}).get("name") or meta.get("name") or "").strip()
            id_to_name[eid] = name
        for tbl in relevant_list:
            table_id = tbl.get("table_id") or tbl.get("table_name") or tbl.get("id")
            if table_id:
                table_id = str(table_id).strip()
                tbl["table_id"] = table_id
                tbl["table_name"] = id_to_name.get(table_id, "")

        # Exclude join table from relevant_list (table_id must not equal join table id)

        join_table_id = None
        if join_meta_path.exists():
            try:
                with open(join_meta_path, "r", encoding="utf-8") as f:
                    join_meta = json.load(f)
                join_table_id = (join_meta.get("resource") or {}).get("id") or join_meta.get("id")
            except Exception:
                pass

        print(f"[Table Selection] {len(relevant_list)} tables selected")
        for i, tbl in enumerate(relevant_list[:10], 1):
            name = tbl.get("table_name", "") or tbl.get("table_id", "?")
            print(f"   {i}. {name} (id={tbl.get('table_id', '?')})")
        if len(relevant_list) > 10:
            print(f"   ... and {len(relevant_list) - 10} more")

        intermediate_stats["table_selection"] = {
            "n_relevant_tables": len(relevant_list),
            "relevant_tables": [
                {
                    "table_id": t.get("table_id"),
                    "table_name": t.get("table_name"),
                    "confidence": t.get("confidence"),
                    "reason": (str(t.get("reason") or t.get("reasoning") or "")[:240] or None),
                }
                for t in relevant_list
            ],
        }
        _mark_phase("table_selection")

        #---- Phase 2: Join Column Selection ----

        run_start_time = datetime.now()
        run_record = {
            "table_id": [],
            "status": [],
            "reason": [],
        }
        topk_jaccard = {}  # table_id -> list of selected column names
        final_selected_tables = []

        # Load join table and create join column sketch
        base_path = Path(__file__).resolve().parent / BASE_DIR
        join_df = pd.read_csv(base_path / real_join_table_name / config.data_filename, low_memory=False)
        join_columns = config.join_column if isinstance(config.join_column, list) else [config.join_column]
        
        from tools.sketch import bottom_k_sketch_column_with_samples

        # Create sketch for each join column
        if len(join_columns) == 1:
            join_sketch, join_sketch_original_values, _ = bottom_k_sketch_column_with_samples(
                join_df[join_columns[0]], k=1024)  

        else:
            join_sketch = {jc: bottom_k_sketch_column(join_df[jc]) for jc in join_columns if jc in join_df.columns}
            join_sketch_original_values = None


        # Join column descriptions
        join_col_descs = get_column_descriptions_from_local_metadata(BASE_DIR, real_join_table_name)

        topk_join = config.config.get("task", {}).get("topk_join_columns", 5) or 5

        for tbl in relevant_list:
            cand_name = tbl.get("table_id")
            # if not cand_name:
            #     continue
            # if cand_name in SKIP_TABLES:
            #     print(f"   ⏭️  Skipping {cand_name} (in skip list)")
            #     continue
            # run_record["table_id"].append(cand_name)

            df, status = get_candidate_table(table_id=cand_name, opendata_domain=domain_for_fetch)
            if status["success"]:
                run_record["status"].append("success")
                run_record["reason"].append(None)
                cols_with_jaccard = select_join_columns_for_candidate(
                    join_sketch, df, k_columns=topk_join, min_jaccard=0.1, sketch_k=1024
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
query_table_description: {query_table_description}

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
query_table_description: {query_table_description}
candidate_columns: {json.dumps(candidates_for_llm, ensure_ascii=False)}

Return JSON with selected_columns and reasoning."""
                    has_candidates = len(cols_with_jaccard) > 0

                if has_candidates:
                    try:
                        jc_events = await _run_debug_fresh(
                            lambda: build_join_column_choose_agent(config=config),
                            jc_prompt,
                            tag="joincol",
                        )
                        full_jc_text = ""
                        for event in jc_events:
                            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                                for part in event.content.parts:
                                    t = getattr(part, "text", None)
                                    if t:
                                        full_jc_text += t
                        jc_json = extract_json_by_key_from_full_text(
                            full_jc_text, "selected_columns", prefer_non_empty_list=True
                        ) if full_jc_text.strip() else {}
                        
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
                best_j = _best_jaccard(cols_with_jaccard, selected_cols, join_columns)
                intermediate_stats["join_column_selection"]["per_table"].append(
                    {
                        "table_id": cand_name,
                        "fetch_ok": True,
                        "selected_columns": list(selected_cols) if selected_cols else [],
                        "best_jaccard": best_j,
                        "n_topk_candidates": (
                            sum(len(v) for v in cols_with_jaccard.values())
                            if isinstance(cols_with_jaccard, dict)
                            else len(cols_with_jaccard or [])
                        ),
                    }
                )
                if selected_cols:
                    tbl_with_join = dict(tbl)
                    tbl_with_join["candidate_table"] = cand_name
                    tbl_with_join["selected_columns"] = selected_cols
                    tbl_with_join["join_col_descs"] = join_col_descs
                    tbl_with_join["cand_col_descs"] = cand_col_descs
                    tbl_with_join["best_jaccard"] = best_j
                    final_selected_tables.append(tbl_with_join)
                    print(f"   ✅ {cand_name}: LLM selected {selected_cols}")
            else:
                run_record["status"].append("failed")
                run_record["reason"].append(status["reason"])
                topk_jaccard[cand_name] = []
                intermediate_stats["join_column_selection"]["per_table"].append(
                    {
                        "table_id": cand_name,
                        "fetch_ok": False,
                        "selected_columns": [],
                        "best_jaccard": None,
                        "reason": status.get("reason"),
                    }
                )

        run_record["topk_jaccard"] = topk_jaccard
        relevant_list = final_selected_tables
        join_columns_list = config.join_column if isinstance(config.join_column, list) else [config.join_column]

        _jc_rows = intermediate_stats["join_column_selection"]["per_table"]
        _ok = [r for r in _jc_rows if r.get("fetch_ok") and r.get("selected_columns")]
        _jacs = [float(r["best_jaccard"]) for r in _ok if r.get("best_jaccard") is not None]
        intermediate_stats["join_column_selection"]["summary"] = {
            "n_tables_attempted": len(_jc_rows),
            "n_tables_with_join": len(_ok),
            "mean_best_jaccard": (sum(_jacs) / len(_jacs)) if _jacs else None,
            "max_best_jaccard": max(_jacs) if _jacs else None,
            "min_best_jaccard": min(_jacs) if _jacs else None,
        }
        _mark_phase("join_selection")

        data_dir = Path(__file__).resolve().parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        safe_sid = re.sub(r"[^\w\-]", "_", str(session_id or "default").strip()) or "default"
        filename = run_start_time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{safe_sid}.json"
        filepath = data_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2, ensure_ascii=False)
        print(f"[Run Record] Saved to {filepath}")
        intermediate_stats["run_record_path"] = str(filepath)


# ---- Phase 3: Augment Column Selection ----

        from Classification_regression import preprocess_data, run_classification_task, run_regression_task

        JOIN_KEY_LIMIT = 500
        jc0 = join_columns[0] if join_columns else None
        join_df_ml = join_df
        join_key_filter_500 = None
        if jc0 and jc0 in join_df.columns:
            unique_keys = join_df[jc0].dropna().astype(str).str.strip().unique().tolist()
            if len(unique_keys) > JOIN_KEY_LIMIT:
                keys_to_keep = set(unique_keys[:JOIN_KEY_LIMIT])
                join_df_ml = join_df[join_df[jc0].astype(str).str.strip().isin(keys_to_keep)].copy()
                join_key_filter_500 = unique_keys[:JOIN_KEY_LIMIT]

        baseline_metric = None
        baseline_features = [c for c in join_df_ml.columns 
                            if c != target_column and c not in join_columns]
        if len(baseline_features) == 0:
            baseline_features = [c for c in join_df_ml.columns if c != target_column]
        metric_name = "r2_score" if task_type == "regression" else "f1_score"  
        if len(baseline_features) > 0:
            baseline_df = join_df_ml[baseline_features + [target_column]].dropna(subset=[target_column])
            
            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_J1","timestamp":_ts,"location":"orchestrator.py:683","message":"baseline_df before preprocess","data":{"shape":[int(x) for x in baseline_df.shape],"columns":list(baseline_df.columns),"dtypes":{col:str(dtype) for col,dtype in baseline_df.dtypes.items()},"target_column":target_column,"baseline_features":baseline_features},"hypothesisId":"J,M"}) + '\n')
            # #endregion
            metric_name = "r2_score" if task_type == "regression" else "f1_score"  
            try:
                X, y, target_encoder, scaler = preprocess_data(baseline_df, target_column, task_type)
                
                # #region agent log
                import json as _json; _ts = __import__('time').time_ns() // 1000000
                if X is not None and y is not None:
                    _y_sample = [float(v) if hasattr(v,'item') else v for v in (y[:5] if len(y)>0 else [])]
                    with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_K1","timestamp":_ts,"location":"orchestrator.py:693","message":"baseline X,y after preprocess","data":{"X_shape":[int(x) for x in X.shape],"X_columns":list(X.columns),"X_dtypes":{col:str(dtype) for col,dtype in X.dtypes.items()},"y_shape":[int(x) for x in y.shape] if hasattr(y,'shape') else [len(y)],"y_dtype":str(y.dtype) if hasattr(y,'dtype') else str(type(y)),"y_sample":_y_sample},"hypothesisId":"K,L"}) + '\n')
                # #endregion
                if X is not None and len(X) > 0:
                    # #region agent log
                    import json as _json; _ts = __import__('time').time_ns() // 1000000
                    with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_P1","timestamp":_ts,"location":"orchestrator.py:699","message":"before running task","data":{"task_type":task_type,"X_shape":[int(x) for x in X.shape]},"hypothesisId":"P,Q"}) + '\n')
                    # #endregion
                    
                    if task_type == 'classification':
                        metrics = run_classification_task(X, y, target_encoder)
                        
                        # #region agent log
                        import json as _json; _ts = __import__('time').time_ns() // 1000000
                        with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_Q1","timestamp":_ts,"location":"orchestrator.py:709","message":"metrics returned","data":{"metrics_keys":list(metrics.keys()),"metrics_types":{k:str(type(v).__name__) for k,v in metrics.items()}},"hypothesisId":"Q"}) + '\n')
                        # #endregion
                        
                        baseline_metric = metrics['f1_score']
                        metric_name = "f1_score"
                        
                        # #region agent log
                        import json as _json; _ts = __import__('time').time_ns() // 1000000
                        _bm_type = type(baseline_metric).__name__
                        _bm_val = float(baseline_metric) if hasattr(baseline_metric,'item') else baseline_metric
                        with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_Q2","timestamp":_ts,"location":"orchestrator.py:720","message":"baseline_metric extracted","data":{"baseline_metric_type":_bm_type,"baseline_metric_value":_bm_val},"hypothesisId":"Q"}) + '\n')
                        # #endregion
                    elif task_type == 'regression':
                        metrics = run_regression_task(X, y)
                        baseline_metric = metrics['r2_score']
                        metric_name = "r2_score"
                    if baseline_metric is not None:
                        print(f"\n📊 Baseline ({metric_name}): {baseline_metric:.4f}")
                    else:
                        print(f"\n📊 Baseline ({metric_name}): N/A (no valid data)")
            except Exception as e:
                # #region agent log
                import json as _json; _ts = __import__('time').time_ns() // 1000000
                import traceback as _tb
                with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_P2","timestamp":_ts,"location":"orchestrator.py:732","message":"baseline computation exception","data":{"error":str(e),"error_type":type(e).__name__,"traceback":_tb.format_exc()},"hypothesisId":"P,Q"}) + '\n')
                # #endregion
                print(f"\n📊 Baseline failed: {e}")
                

        # Correlation phase: compute target aggregation and feature correlations per candidate
        print(f"\n📈 Computing correlations with target '{target_column}'...")
        target_agg, target_type = aggregate_target_by_join_key(
            join_df_ml, join_columns, target_column,
            base_dir=BASE_DIR, join_table_folder=real_join_table_name,
        )

        # Aggregation phase
        join_key_filter_ml = join_key_filter_500 if join_key_filter_500 is not None else join_sketch_original_values
        print(f"\n📊 Aggregating candidate tables by join key...")
        aggregated_results = aggregate_selected_tables(
            final_selected_tables,
            base_dir=BASE_DIR,
            opendata_domain=domain_for_fetch,
            join_key_filter=join_key_filter_ml,
            query_join_columns=join_columns,
            llm_join_keys=join_key_filter_ml, 
            target_agg=target_agg,
            target_column=target_column,
            target_type=target_type,
        )

        for result in aggregated_results:
            cand_agg = result.get("aggregated_df")
            if cand_agg is None or cand_agg.empty:
                result["correlation_table"] = pd.DataFrame(columns=["feature", "metric", "value"])
                continue
            # Align join column names: candidate uses selected_columns, query uses join_columns
            selected_cols = result.get("selected_columns", [])
            if selected_cols and len(selected_cols) == len(join_columns):
                rename_map = dict(zip(selected_cols, join_columns))
                cand_agg = cand_agg.rename(columns=rename_map)
            
            for jc in join_columns:
                if jc in target_agg.columns:
                    target_agg[jc] = target_agg[jc].astype(str)
                if jc in cand_agg.columns:
                    cand_agg[jc] = cand_agg[jc].astype(str)

            merged = merge_target_with_candidate(target_agg, cand_agg, join_columns, target_col=target_column, target_type=target_type)
            n_query_keys = int(len(target_agg)) if target_agg is not None else 0
            n_merged = int(len(merged)) if merged is not None else 0
            coverage = (n_merged / n_query_keys) if n_query_keys > 0 else None
            result["join_coverage"] = coverage
            result["n_merged_rows"] = n_merged
            result["n_query_keys"] = n_query_keys
            intermediate_stats["join_quality"]["per_table"].append(
                {
                    "table_id": result.get("candidate_table"),
                    "n_query_keys": n_query_keys,
                    "n_merged_rows": n_merged,
                    "coverage": coverage,
                    "n_corr_features": None,  # filled after correlation
                }
            )

            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            _merged_dtypes = {col: str(dtype) for col, dtype in merged.dtypes.items()}
            _merged_sample = {col: str(merged[col].iloc[0]) if len(merged) > 0 else None for col in list(merged.columns)[:15]}
            with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_J1","timestamp":_ts,"location":"orchestrator.py:745","message":"merged_df before correlation","data":{"shape":[int(x) for x in merged.shape],"dtypes":_merged_dtypes,"sample":_merged_sample,"target_column":target_column,"target_type":target_type,"join_columns":join_columns},"hypothesisId":"J,M"}) + '\n')
            # #endregion
            
            corr_df = compute_feature_correlations(
                merged, join_columns, target_column, target_type
            )
            result["correlation_table"] = corr_df
            if intermediate_stats["join_quality"]["per_table"]:
                intermediate_stats["join_quality"]["per_table"][-1]["n_corr_features"] = int(len(corr_df)) if corr_df is not None else 0
            cand_name = result.get("candidate_table", "?")
            print(f"   {cand_name}: {len(corr_df)} features with correlation/distance")
            print(corr_df.to_string(index=False))
            print()

        for result in aggregated_results:
            cand_name = result.get("candidate_table", "?")
            corr_df = result.get("correlation_table", pd.DataFrame())
            if corr_df.empty:
                result["selected_augment_columns"] = []
                result["augment_reasoning"] = "No correlation data."
                continue
            
            cand_descs = get_column_descriptions_from_index(cand_name)
            target_desc = join_col_descs.get(target_column, "")  # get target column description from join table metadata
            
            cand_agg = result.get("aggregated_df")

            # construct candidate_columns
            existing_columns_lower = {str(c).strip().lower() for c in join_df.columns} 
            candidate_columns = []
            for _, row in corr_df.iterrows():
                feature_name = row["feature"]
                if str(feature_name).strip().lower() in existing_columns_lower:
                    continue

                entry = {
                    "feature": feature_name,
                    "metric": row["metric"],
                    "value": float(row["value"]) if pd.notna(row["value"]) else None,
                    "description": cand_descs.get(row["feature"], ""),
                    "feature_type": row.get("feature_type", "unknown"),
                }
                feat_type = row.get("feature_type", "unknown")
                if feat_type == "text" and cand_agg is not None:
                    summary_col = f"{row['feature']}_summary"
                    if summary_col in cand_agg.columns:
                        parts = []
                        for _, r in cand_agg.iterrows():
                            jk = " | ".join(str(r[c]) for c in join_columns if c in cand_agg.columns)
                            s = r.get(summary_col, "")
                            if pd.notna(s) and str(s).strip():
                                parts.append(f"{jk}: {s}")
                        entry["summary"] = "\n".join(parts) if parts else ""
                    else:
                        entry["summary"] = ""
                else:
                    entry["summary"] = ""
                candidate_columns.append(entry)

            prompt = f"""
task_type: "{task_type}"
target_column: "{target_column}"
target_column_description: "{target_desc}"
user_intent: "{user_intent}"
candidate_table: "{cand_name}"
query_table_description: "{query_table_description}"
candidate_columns: {json.dumps(candidate_columns, ensure_ascii=False, indent=2)}

Delete unnecessary columns from candidate_columns (you decide which), keep the remainder. Return JSON with dropped_columns, selected_augment_columns, and reasoning.
"""
            events = await _run_debug_fresh(
                lambda: build_augment_column_selection_agent(config),
                prompt,
                quiet=True,
                tag="augment",
            )
            full_aug_text = ""
            for event in events:
                if getattr(event, "content", None) and getattr(event.content, "parts", None):
                    for part in event.content.parts:
                        t = getattr(part, "text", None)
                        if t:
                            full_aug_text += t
            parsed = extract_json_by_key_from_full_text(
                full_aug_text, "selected_augment_columns", prefer_non_empty_list=True
            ) if full_aug_text.strip() else {}
            result["selected_augment_columns"] = parsed.get("selected_augment_columns", [])
            result["augment_reasoning"] = parsed.get("reasoning", "")

        # ---- Build augmented table and evaluate metrics ----
        augmented_df = join_df_ml.copy()
        from tools.sketch import _normalize_for_hash
        for jc in join_columns:
            if jc in augmented_df.columns:
                augmented_df[jc] = augmented_df[jc].apply(_normalize_for_hash)
        for result in aggregated_results:

            aug_cols = result.get("selected_augment_columns", [])
            if not aug_cols:
                continue
            cand_name = result.get("candidate_table")
            selected_cols = result.get("selected_columns", [])

         
            from tools.aggregation import aggregate_candidate_by_join_key

            cand_df_full, status = get_candidate_table(cand_name, domain_for_fetch)
            if not status["success"]:
                continue

            # join columns + augment columns
            cols_needed = selected_cols + [c for c in aug_cols if c in cand_df_full.columns]
            cand_df_subset = cand_df_full[cols_needed]
            cand_agg = cand_df_subset
            # cand_agg = aggregate_candidate_by_join_key(cand_df_subset, selected_cols, table_id=cand_name, llm_summarize=False)
            if cand_agg is None or cand_agg.empty:
                continue
            selected_cols = result.get("selected_columns", [])
            if selected_cols and len(selected_cols) == len(join_columns):
                rename_map = dict(zip(selected_cols, join_columns))
                cand_agg = cand_agg.rename(columns=rename_map)
            cols_to_add = [c for c in aug_cols if c in cand_agg.columns]
            if not cols_to_add:
                continue
            to_merge = cand_agg[join_columns + cols_to_add].drop_duplicates(subset=join_columns)
            
            # Normalize join columns in to_merge to match augmented_df
            for jc in join_columns:
                if jc in to_merge.columns:
                    to_merge[jc] = to_merge[jc].apply(_normalize_for_hash)
            
            # Rename duplicate columns in to_merge to avoid merge conflicts
            cand_name = result.get("candidate_table", "unknown")
            cand_suffix = f"_{cand_name.replace('-', '_')}"
            rename_map = {}
            for col in cols_to_add:
                if col in augmented_df.columns:
                    new_name = f"{col}{cand_suffix}"
                    rename_map[col] = new_name
            if rename_map:
                to_merge = to_merge.rename(columns=rename_map)
            
            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_F2","timestamp":_ts,"location":"orchestrator.py:835","message":"before merge","data":{"candidate_table":result.get("candidate_table","?"),"augmented_df_columns":list(augmented_df.columns),"to_merge_columns":list(to_merge.columns),"cols_to_add":cols_to_add,"rename_map":rename_map},"hypothesisId":"F,G,H"}) + '\n')
            # #endregion
            
            augmented_df = augmented_df.merge(to_merge, on=join_columns, how="left")

 

            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_FIX2","timestamp":_ts,"location":"orchestrator.py:862","message":"after merge","data":{"candidate_table":cand_name,"augmented_df_columns":list(augmented_df.columns)},"hypothesisId":"FIX"}) + '\n')
            # #endregion

        # Evaluate augmented metric
        augmented_metric = None
        aug_features = [c for c in augmented_df.columns if c != target_column and c not in join_columns]
        if len(aug_features) > 0:
            aug_df = augmented_df[aug_features + [target_column]].dropna(subset=[target_column])


            # #region agent log
            import json as _json; _ts = __import__('time').time_ns() // 1000000
            with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_N1","timestamp":_ts,"location":"orchestrator.py:866","message":"aug_df before preprocess","data":{"shape":[int(x) for x in aug_df.shape],"columns":list(aug_df.columns),"dtypes":{col:str(dtype) for col,dtype in aug_df.dtypes.items()},"target_column":target_column,"aug_features":aug_features},"hypothesisId":"N,M"}) + '\n')
            # #endregion
            
            if len(aug_df) >= 10:
                try:
                    X, y, target_encoder, scaler = preprocess_data(aug_df, target_column, task_type)

                    # #region agent log
                    import json as _json; _ts = __import__('time').time_ns() // 1000000
                    if X is not None and y is not None:
                        _y_sample = [float(v) if hasattr(v,'item') else v for v in (y[:5] if len(y)>0 else [])]
                        with open('/fs/ess/PDS0349/fangzy96/bear/.cache/cursor_debug.log', 'a') as _f: _f.write(_json.dumps({"id":f"log_{_ts}_K2","timestamp":_ts,"location":"orchestrator.py:878","message":"augmented X,y after preprocess","data":{"X_shape":[int(x) for x in X.shape],"X_columns":list(X.columns),"X_dtypes":{col:str(dtype) for col,dtype in X.dtypes.items()},"y_shape":[int(x) for x in y.shape] if hasattr(y,'shape') else [len(y)],"y_dtype":str(y.dtype) if hasattr(y,'dtype') else str(type(y)),"y_sample":_y_sample},"hypothesisId":"K,L"}) + '\n')
                    # #endregion
                    if X is not None and len(X) > 0:
                        
                        if task_type == 'classification':
                            metrics = run_classification_task(X, y, target_encoder)
                            
                            augmented_metric = metrics['f1_score']
                        elif task_type == 'regression':
                            metrics = run_regression_task(X, y)
                            augmented_metric = metrics['r2_score']

                    metric_name = "r2_score" if task_type == "regression" else "f1_score"
                    print(f"\n📊 Augmented ({metric_name}): {augmented_metric:.4f}")
                    if baseline_metric is not None:
                        print(f"\n📊 Augmented ({metric_name}): {augmented_metric:.4f}")
                        improvement = augmented_metric - baseline_metric
                        pct = (improvement / abs(baseline_metric) * 100) if baseline_metric != 0 else 0
                        print(f"   Improvement: {improvement:+.4f} ({pct:+.1f}%)")
                except Exception as e:
                    print(f"\n📊 Augmented ({metric_name}): N/A (no valid data)")

        augment_output = [
            {
                "candidate_table": r.get("candidate_table", "?"),
                "selected_augment_columns": r.get("selected_augment_columns", []),
                "reasoning": r.get("augment_reasoning", ""),
            }
            for r in aggregated_results
        ]
        metric_name = "r2_score" if task_type == "regression" else "f1_score"

        # Finalize intermediate stats
        _jq = intermediate_stats["join_quality"]["per_table"]
        _covs = [float(r["coverage"]) for r in _jq if r.get("coverage") is not None]
        intermediate_stats["join_quality"]["summary"] = {
            "n_tables": len(_jq),
            "mean_coverage": (sum(_covs) / len(_covs)) if _covs else None,
            "max_coverage": max(_covs) if _covs else None,
            "min_coverage": min(_covs) if _covs else None,
        }
        _aug_rows = []
        for r in aggregated_results:
            cols = r.get("selected_augment_columns") or []
            _aug_rows.append(
                {
                    "table_id": r.get("candidate_table"),
                    "n_augment_columns": len(cols),
                    "selected_augment_columns": cols,
                    "join_coverage": r.get("join_coverage"),
                    "best_jaccard": r.get("best_jaccard"),
                }
            )
        # attach best_jaccard from join phase onto aug rows
        _jac_by_id = {
            r.get("table_id"): r.get("best_jaccard")
            for r in intermediate_stats["join_column_selection"]["per_table"]
        }
        for row in _aug_rows:
            if row.get("best_jaccard") is None:
                row["best_jaccard"] = _jac_by_id.get(row.get("table_id"))
        intermediate_stats["augment"]["per_table"] = _aug_rows
        _ns = [int(r.get("n_augment_columns") or 0) for r in _aug_rows]
        intermediate_stats["augment"]["summary"] = {
            "n_tables": len(_aug_rows),
            "n_tables_with_aug_cols": sum(1 for n in _ns if n > 0),
            "n_augment_columns_total": sum(_ns),
            "mean_augment_columns": (sum(_ns) / len(_ns)) if _ns else None,
        }
        improvement = None
        if baseline_metric is not None and augmented_metric is not None:
            try:
                improvement = float(augmented_metric) - float(baseline_metric)
            except Exception:
                improvement = None
        intermediate_stats["evaluation"] = {
            "baseline_metric": baseline_metric,
            "augmented_metric": augmented_metric,
            "improvement": improvement,
            "metric_name": metric_name,
        }
        _mark_phase("augment_eval")
        intermediate_stats["phase_timings_seconds"]["total"] = time.perf_counter() - _t_pipe0

        return {
            "augment_results": augment_output,
            "baseline_metric": baseline_metric,
            "augmented_metric": augmented_metric,
            "metric_name": metric_name,
            "intermediate_stats": intermediate_stats,
            "final_selected_tables": [
                {
                    "table_id": t.get("candidate_table") or t.get("table_id"),
                    "selected_columns": t.get("selected_columns"),
                    "best_jaccard": t.get("best_jaccard"),
                }
                for t in final_selected_tables
            ],
        }


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
    parser.add_argument('--use-original', action='store_true',
                    help='Use original (unperturbed) data instead of perturbed data')
    parser.add_argument('--base-dir', type=str, default=None,
                    help='Override data base directory (e.g. perturbed_0.85_0.1)')
    parser.add_argument('--data-filename', type=str, default=None,
                        help='Override data filename (e.g. rows_original.csv)')

    args = parser.parse_args()
    try:
        if args.use_original:
            config = AgentPipelineConfig(args.config)
            config.config["data"] = {**config.config.get("data", {}), "base_dir": "query_table"}
        else:
    
            from benchmark_perturbation.data_perturbation import load_perturbation_config
            full_perturb = load_perturbation_config  
            perturb_path = Path(__file__).resolve().parent / "configs" / "perturbation.yaml"
            perturb_cfg = {}
            if perturb_path.exists():
                with open(perturb_path) as f:
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

            
        output = asyncio.run(run_orchestrator(
            config=config,
            user_intent=args.user_intent,
            session_id=args.session_id,
            join_table_name=None,     
            target_column=None,
            task_type=args.task_type,
        ))
 
            
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
            # Save results to file
            output_file = Path(config.results_file)
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {output_file}")
        
        if config.print_results and output:
            for item in output.get("augment_results", []):
                payload = {
                    "selected_augment_columns": item["selected_augment_columns"],
                    "reasoning": item["reasoning"]
                }
                print("\nAugmentColumnSelectionAgent >", json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()