import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Agent.table_selection_collab_orchestrator import run_table_selection_collab_orchestrator
from datalake_client import SocrataDatalakeClient
from tools.llm_agent_tools import build_opendata_search_params

from Pipeline.context import PipelineContext
from Pipeline.utils import timed_section


def _session_checked_path(base_dir: Path, session_id: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", str(session_id).strip()) or "default"
    return base_dir / f"{safe}.json"


def _load_session_checked(base_dir: Path, session_id: str) -> Tuple[set, list]:
    path = _session_checked_path(base_dir, session_id)
    if not path.exists():
        return set(), []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set(), []
    if isinstance(data, list):
        table = [e for e in data if isinstance(e, dict) and e.get("id")]
        return {str(e.get("id")).strip() for e in table}, table
    return set(), []


def _save_session_checked(base_dir: Path, session_id: str, checked_table: list) -> None:
    path = _session_checked_path(base_dir, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checked_table, f, indent=2, ensure_ascii=False)


async def run_table_selection(ctx: PipelineContext) -> None:
    config = ctx.config
    data_source = config.config.get("data", {}).get("data_source", "local")
    join_column = ctx.state["join_column"]
    dimension_specifications = ctx.state["dimension_specifications"]

    decision_log = ctx.state.get("decision_log", {})
    phase_log = {
        "selected_tables": [],
        "excluded_tables": [],
        "stats": {},
    }

    reuse_table_selection = ctx.state.get("reuse_table_selection")
    if isinstance(reuse_table_selection, dict):
        selected_tables = reuse_table_selection.get("selected_tables", []) or []
        excluded_tables = reuse_table_selection.get("excluded_tables", []) or []
        candidate_ids_for_run = reuse_table_selection.get("candidate_ids_for_run", []) or []
        domain_for_fetch = reuse_table_selection.get("domain_for_fetch")
        query_table_description = reuse_table_selection.get("query_table_description", "")

        relevant_list: List[Dict[str, Any]] = []
        for item in selected_tables:
            if not isinstance(item, dict):
                continue
            tid = str(item.get("table_id") or "").strip()
            if not tid:
                continue
            relevant_list.append(
                {
                    "table_id": tid,
                    "table_name": item.get("table_name", "") or tid,
                    "risk": "low",
                    "reason": "reused from round 1 table selection",
                }
            )

        if not candidate_ids_for_run:
            candidate_ids_for_run = [str(t.get("table_id") or "").strip() for t in selected_tables if isinstance(t, dict)]
            candidate_ids_for_run += [str(t.get("table_id") or "").strip() for t in excluded_tables if isinstance(t, dict)]
            candidate_ids_for_run = [x for x in candidate_ids_for_run if x]

        phase_log["selected_tables"] = selected_tables if isinstance(selected_tables, list) else []
        phase_log["excluded_tables"] = excluded_tables if isinstance(excluded_tables, list) else []
        phase_log["stats"] = {
            "data_source": "datalake",
            "candidate_count": len(candidate_ids_for_run),
            "selected_count": len(relevant_list),
            "excluded_count": len(phase_log["excluded_tables"]),
            "reused_from_round1": True,
        }
        if isinstance(decision_log, dict):
            decision_log.setdefault("phases", {})["table_selection"] = phase_log

        print("\n📋 Reusing round 1 table selection (skip metadata fetch/API).")
        print(f"[Table Selection] {len(relevant_list)} tables selected (reused)")
        for i, tbl in enumerate(relevant_list[:10], 1):
            name = tbl.get("table_name", "") or tbl.get("table_id", "?")
            print(f"   {i}. {name} (id={tbl.get('table_id', '?')})")
        if len(relevant_list) > 10:
            print(f"   ... and {len(relevant_list) - 10} more")

        ctx.state["domain_for_fetch"] = domain_for_fetch
        ctx.state["query_table_description"] = query_table_description
        ctx.state["relevant_list"] = relevant_list
        ctx.state["table_selection_candidate_ids"] = candidate_ids_for_run
        return

    if data_source != "datalake":
        ctx.state["relevant_list"] = []
        ctx.state["domain_for_fetch"] = None
        ctx.state["query_table_description"] = ""
        ctx.state["table_selection_candidate_ids"] = []
        phase_log["stats"] = {"data_source": data_source, "candidate_count": 0, "selected_count": 0}
        if isinstance(decision_log, dict):
            decision_log.setdefault("phases", {})["table_selection"] = phase_log
        return

    join_col_for_search = ", ".join(join_column) if isinstance(join_column, list) else str(join_column)
    target_column = str(ctx.target_column).strip() if ctx.target_column else None
    search_domains, search_q = build_opendata_search_params(
        dimension_specifications, join_col_for_search, target_column
    )
    print("=" * 80)
    print(f"Search Domains: {search_domains}")
    print(f"Search Query: {search_q}")
    print("=" * 80)

    datalake_config = config.config.get("data", {}).get("datalake", {})
    client = SocrataDatalakeClient(datalake_config)
    max_tables = datalake_config.get("max_tables") or 10
    domain_for_fetch = (search_domains[0] if search_domains else None) or (datalake_config.get("domains") or [None])[0]
    index_path = Path(__file__).resolve().parent.parent / "opendata_table_index.json"
    session_base = Path(config.session_checked_dir) if getattr(config, "session_checked_dir", None) else None
    if session_base and not session_base.is_absolute():
        session_base = (Path(__file__).resolve().parent.parent / session_base).resolve()

    existing_index = []
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                existing_index = json.load(f)
            if not isinstance(existing_index, list):
                existing_index = []
        except Exception:
            existing_index = []
    existing_id_set = {str(e.get("id")).strip() for e in existing_index if isinstance(e, dict) and e.get("id")}

    checked_set = set()
    checked_table = []
    if session_base and ctx.session_id:
        checked_set, checked_table = _load_session_checked(session_base, ctx.session_id)

    new_entries: List[Dict[str, Any]] = []
    candidate_ids_for_run: List[str] = []
    with timed_section(ctx.pipeline_timings, "03_datalake_metadata_and_index"):
        if domain_for_fetch:
            opendata_metadata = client.read_metadata(
                search_domains=search_domains,
                search_q=search_q if search_q.strip() else None,
                exclude_tables=[ctx.join_table_name] if ctx.join_table_name else [],
                limit=10000,
                offset=0,
            )
            batch = list((opendata_metadata.get("metadata_by_dataset") or {}).keys())
            print(f"[Opendata] One-time fetch: {len(batch)} tables")
            for ds_id in batch:
                if len(candidate_ids_for_run) >= max_tables:
                    break
                ds_id_str = str(ds_id).strip()
                if not ds_id_str:
                    continue
                if ds_id_str in checked_set:
                    continue
                if ds_id_str in existing_id_set:
                    candidate_ids_for_run.append(ds_id_str)
                    continue
                full_meta = client.get_dataset_metadata(ds_id_str, domain_for_fetch)
                if not full_meta or "error" in full_meta:
                    continue
                res = full_meta.get("resource") or {}
                columns_name = list(res.get("columns_name") or [])
                columns_desc = [str((c.get("description") or "")).strip() for c in full_meta.get("columns") if isinstance(c, dict)]
                col_list = full_meta.get("columns") or []
                name_to_dtype = {
                    str(c.get("name", "")).strip(): str(c.get("dataTypeName") or "").strip() or "unknown"
                    for c in col_list if isinstance(c, dict) and c.get("name")
                }
                entry = {
                    "id": full_meta.get("id") or res.get("id"),
                    "name": res.get("name") or "",
                    "description": full_meta.get("description") or res.get("description") or "",
                    "attribution": full_meta.get("attribution") or res.get("attribution"),
                    "columns_name": columns_name,
                    "columns_description": columns_desc,
                    "columns_datatype": [name_to_dtype.get(str(n).strip(), "unknown") for n in columns_name],
                    "classification": full_meta.get("classification") or {},
                    "domain": (full_meta.get("metadata") or {}).get("domain") or domain_for_fetch,
                }
                new_entries.append(entry)
                existing_id_set.add(str(entry["id"]).strip())
                candidate_ids_for_run.append(ds_id_str)
                if session_base and ctx.session_id:
                    checked_table.append(entry)
                    checked_set.add(ds_id_str)
                    _save_session_checked(session_base, ctx.session_id, checked_table)
        merged_index = existing_index + new_entries
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(merged_index, f, indent=2, ensure_ascii=False)

    query_table_description = ""
    if ctx.join_meta_path.exists():
        try:
            with open(ctx.join_meta_path, "r", encoding="utf-8") as f:
                join_meta = json.load(f)
            query_table_description = (join_meta.get("resource") or {}).get("description", "") or ""
        except Exception:
            pass

    table_examples_by_id: Dict[str, List[Dict[str, Any]]] = {}
    with timed_section(ctx.pipeline_timings, "04_datalake_table_examples_5rows"):
        id_to_domain = {
            str(e.get("id", "")).strip(): e.get("domain")
            for e in (existing_index + new_entries) if isinstance(e, dict) and e.get("id")
        }
        for ds_id in candidate_ids_for_run:
            ds_id_str = str(ds_id).strip()
            if not ds_id_str:
                continue
            try:
                rows = client.read_data(ds_id_str, id_to_domain.get(ds_id_str) or domain_for_fetch, max_rows=5)
            except Exception:
                rows = []
            clean_rows = []
            for row in (rows if isinstance(rows, list) else [])[:5]:
                if not isinstance(row, dict):
                    continue
                out_row = {}
                for k, v in row.items():
                    if v is None:
                        out_row[k] = None
                    else:
                        s = str(v)
                        out_row[k] = s[:120] + "...[truncated]" if len(s) > 120 else s
                clean_rows.append(out_row)
            table_examples_by_id[ds_id_str] = clean_rows

    task_info = {
        "target_column": ctx.target_column,
        "task_type": ctx.task_type,
        "query_table_name": ctx.query_table_display_name,
        "join_columns": join_col_for_search,
        "query_table_description": query_table_description,
        "dimension_specifications": json.dumps(dimension_specifications, ensure_ascii=False),
        "table_examples_5rows_all_columns": table_examples_by_id,
    }
    print("\n📋 Running Collaborative Table Selection (datalake)...")
    with timed_section(ctx.pipeline_timings, "05_table_selection_collab"):
        collab_result = await run_table_selection_collab_orchestrator(
            config=config,
            candidate_ids=candidate_ids_for_run,
            user_intent=ctx.user_intent,
            task_info=task_info,
        )
    relevant_list = collab_result.get("final_tables", [])
    id_to_name = {str(e.get("id", "")).strip(): (e.get("name") or "") for e in (existing_index + new_entries) if isinstance(e, dict)}
    for tbl in relevant_list:
        table_id = tbl.get("table_id") or tbl.get("table_name") or tbl.get("id")
        if table_id:
            table_id = str(table_id).strip()
            tbl["table_id"] = table_id
            tbl["table_name"] = id_to_name.get(table_id, "")
    selected_ids = {str(t.get("table_id", "")).strip() for t in relevant_list if t.get("table_id")}
    for tid in candidate_ids_for_run:
        if tid in selected_ids:
            phase_log["selected_tables"].append(
                {
                    "table_id": tid,
                    "table_name": id_to_name.get(tid, ""),
                    "decision": "selected",
                    "reason_code": "TABLE_SELECTED_BY_AGENT",
                    "reason": "selected by table selection orchestrator",
                }
            )
        else:
            phase_log["excluded_tables"].append(
                {
                    "table_id": tid,
                    "table_name": id_to_name.get(tid, ""),
                    "decision": "excluded",
                    "reason_code": "TABLE_NOT_SELECTED_BY_AGENT",
                    "reason": "candidate provided but not selected",
                }
            )
    phase_log["stats"] = {
        "data_source": data_source,
        "candidate_count": len(candidate_ids_for_run),
        "selected_count": len(selected_ids),
        "excluded_count": len(phase_log["excluded_tables"]),
    }
    if isinstance(decision_log, dict):
        decision_log.setdefault("phases", {})["table_selection"] = phase_log

    print(f"[Table Selection] {len(relevant_list)} tables selected")
    for i, tbl in enumerate(relevant_list[:10], 1):
        name = tbl.get("table_name", "") or tbl.get("table_id", "?")
        print(f"   {i}. {name} (id={tbl.get('table_id', '?')})")
    if len(relevant_list) > 10:
        print(f"   ... and {len(relevant_list) - 10} more")

    ctx.state["domain_for_fetch"] = domain_for_fetch
    ctx.state["query_table_description"] = query_table_description
    ctx.state["relevant_list"] = relevant_list
    ctx.state["table_selection_candidate_ids"] = candidate_ids_for_run

