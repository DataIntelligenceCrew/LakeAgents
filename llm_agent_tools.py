import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from scipy.spatial.distance import cosine
from fasttext import FastText
import re
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import warnings
from agent_config_loader import load_config
warnings.filterwarnings('ignore')
from typing import Dict, Any, List
from google.adk.tools.tool_context import ToolContext

# Opendata search: where -> domains, domain/field + when + who -> q
_opendata_search_domains: Optional[List[str]] = None
_opendata_search_q: Optional[str] = None

GEOGRAPHIC_TO_DOMAIN = {
    "nyc": "data.cityofnewyork.us",
    "new york": "data.cityofnewyork.us",
    "new york city": "data.cityofnewyork.us",
    "chicago": "data.cityofchicago.org",
}


def build_opendata_search_params(dimension_specifications: Dict[str, list], join_column: Optional[List[str]] = None, target_column: Optional[str] = None) -> tuple:
    """Build (search_domains, search_q) from dimension_specifications. where -> domains; domain/field, when, who -> q."""
    domains = []
    where_specs = dimension_specifications.get("Geographic", []) or []
    for s in where_specs:
        key = (s or "").lower().strip()
        if key and key in GEOGRAPHIC_TO_DOMAIN and GEOGRAPHIC_TO_DOMAIN[key] not in domains:
            domains.append(GEOGRAPHIC_TO_DOMAIN[key])
    if not domains:
        config = load_config()
        domains = config.get('data', {}).get('datalake', {}).get('domains', []) or []

    SKIP_WORDS = {"done", "skip", "all"}
    dim_order = ["Domain/Field", "Temporal", "Population Group"]
    keywords = []
    for dim_name in dim_order:
        vals = dimension_specifications.get(dim_name) or []
        for v in vals:
            s = (v if isinstance(v, str) else str(v)).strip()
            if s and s.lower() not in SKIP_WORDS:
                keywords.append(s)

    search_q = " ".join(keywords)

    return (domains, search_q)


def set_opendata_search(domains: Optional[List[str]] = None, q: Optional[str] = None) -> None:
    global _opendata_search_domains, _opendata_search_q
    _opendata_search_domains = domains
    _opendata_search_q = q



def analyze_user_intent(
    user_intent: str,
    target_column: str,
    join_columns: List[str] = None,
    task_type: str = None
) -> Dict[str, Any]:
    """
    Analyze user intent and determine which dimensions are explicitly mentioned.
    This function returns instructions for the LLM to perform the analysis.
    The LLM should analyze the user_intent and return a structured result.
    
    Args:
        user_intent: User's intent/prediction goal
        target_column: Target column name to predict
        join_columns: List of join column names
        task_type: Task type ("regression" or "classification")
        
    Returns:
        Dictionary with analysis instructions and structure.
        The LLM should fill in the actual values based on the user_intent.
    """
    # This is a guidance structure - LLM should analyze and fill in the values
    return {
        "user_intent": user_intent,
        "target_column": target_column,
        "join_columns": join_columns if join_columns else [],
        "task_type": task_type,
        "analysis_instructions": {
            "domain_field": {
                "check": "Analyze if user_intent explicitly mentions any domain/field (e.g., 'demographics', 'education', 'economy', 'crime', 'health').",
                "if_mentioned": "Extract and list the domains mentioned.",
                "if_not_mentioned": "Return empty list and suggest relevant domains based on target_column."
            },
            "geographic": {
                "check": "Analyze if user_intent or join_columns explicitly specify a geographic level (e.g., 'NYC', 'Borough', 'City', 'State', 'County').",
                "if_mentioned": "Extract the geographic level mentioned.",
                "if_not_mentioned": "Return None and suggest appropriate geographic level based on join_columns."
            },
            "temporal": {
                "check": "Analyze if user_intent explicitly mentions time periods (e.g., '2020', 'historical', 'seasonal', 'quarterly').",
                "if_mentioned": "Extract the time period mentioned.",
                "if_not_mentioned": "Return None and suggest appropriate temporal dimensions."
            },
            "population_group": {
                "check": "Analyze if user_intent explicitly mentions population groups (e.g., 'by age', 'by income', 'by education', 'by race').",
                "if_mentioned": "Extract and list the population groups mentioned.",
                "if_not_mentioned": "Return empty list and suggest relevant population groups based on target_column."
            }
        },
        "expected_output_structure": {
            "domain_field": {
                "is_explicitly_mentioned": "Boolean: True if domain/field is mentioned in user_intent, False otherwise",
                "explicitly_mentioned_value": "List[str] or None: Domains mentioned if any, otherwise None"
            },
            "geographic": {
                "is_explicitly_mentioned": "Boolean: True if geographic level is mentioned, False otherwise",
                "explicitly_mentioned_value": "str or None: Geographic level mentioned if any, otherwise None"
            },
            "temporal": {
                "is_explicitly_mentioned": "Boolean: True if time period is mentioned, False otherwise",
                "explicitly_mentioned_value": "str or None: Time period mentioned if any, otherwise None"
            },
            "population_group": {
                "is_explicitly_mentioned": "Boolean: True if population groups are mentioned, False otherwise",
                "explicitly_mentioned_value": "List[str] or None: Population groups mentioned if any, otherwise None"
            }
        }
    }

def _analyze_user_response_with_llm(
    user_response: str,
    dimension_name: str,
    previous_specifications: List[str] = None
) -> Dict[str, Any]:
    """
    Use LLM to analyze user response and determine if dimension should be marked as complete.
    
    Args:
        user_response: User's text response
        dimension_name: Name of the dimension being asked about
        previous_specifications: Previous specifications for this dimension (if any)
        
    Returns:
        Dictionary with:
        - dimension_should_be_complete: bool
        - reason: str (explanation)
        - interpreted_value: str (what the user actually wants, if any)
    """
    try:
        import os
        from openai import OpenAI
        
        # Use OpenAI API directly for quick analysis
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback to keyword-based detection if API key not available
            raise ValueError("OPENAI_API_KEY not set")
        
        client = OpenAI(api_key=api_key)
        
        previous_specs_text = ""
        if previous_specifications:
            previous_specs_text = f"\nPrevious specifications for this dimension: {', '.join(previous_specifications)}"
        
        analysis_prompt = f"""Analyze the user's response to a question about the "{dimension_name}" dimension.

User's response: "{user_response}"
{previous_specs_text}

Determine if the user's response indicates:
1. They want to mark this dimension as complete (e.g., "no preference", "all", "any", "doesn't matter", "skip", "done", just saying thanks/acknowledgment)
2. They provided a specific value/requirement
3. They want to continue specifying (need more information)

Respond in JSON format:
{{
    "dimension_should_be_complete": true/false,
    "reason": "brief explanation",
    "interpreted_value": "the actual value/requirement if provided, or null if just acknowledgment/completion"
}}

Examples:
- "no preference" → {{"dimension_should_be_complete": true, "reason": "User indicated no preference", "interpreted_value": null}}
- "all" or "include all" → {{"dimension_should_be_complete": true, "reason": "User wants all options included", "interpreted_value": "all"}}
- "thanks" or "thank you" (short response) → {{"dimension_should_be_complete": true, "reason": "User acknowledged without providing new specification", "interpreted_value": null}}
- "California" → {{"dimension_should_be_complete": false, "reason": "User provided specific value", "interpreted_value": "California"}}
- "by age group" → {{"dimension_should_be_complete": false, "reason": "User provided specific requirement", "interpreted_value": "by age group"}}
"""
        
        # Call LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes user responses. Always respond with valid JSON."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        response_text = response.choices[0].message.content
        
        # Extract JSON from response
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*"dimension_should_be_complete"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result
        
        # Fallback: try to parse the entire response as JSON
        try:
            result = json.loads(response_text)
            return result
        except:
            # If parsing fails, return default (conservative: don't auto-complete)
            return {
                "dimension_should_be_complete": False,
                "reason": "Failed to parse LLM response",
                "interpreted_value": user_response
            }
            
    except Exception as e:
        # If LLM call fails, return default (conservative: don't auto-complete)
        return {
            "dimension_should_be_complete": False,
            "reason": f"LLM analysis failed: {str(e)}",
            "interpreted_value": user_response
        }

def confirm_dimension_requirement(
    dimension_name: str,
    dimension_type: str,  # "Domain/Field", "Geographic", "Temporal", "Population Group"
    is_explicitly_mentioned: bool,
    tool_context: ToolContext,
    explicitly_mentioned_value: str = None,
    suggested_values: List[str] = None,
    reasoning: str = None,
    question: str = None
) -> Dict[str, Any]:
    """
    Ask user to confirm or specify requirements for a dimension.
    This is used when a dimension is not explicitly mentioned in user_intent.
    
    Args:
        dimension_name: Name of the dimension (e.g., "Geographic", "Domain/Field")
        dimension_type: Type of dimension
        is_explicitly_mentioned: Whether this dimension was explicitly mentioned
        explicitly_mentioned_value: The value if explicitly mentioned
        suggested_values: List of suggested values if not explicitly mentioned
        reasoning: Why this dimension is relevant
        question: Question to ask the user
        
    Returns:
        Dictionary with confirmation result
    """

    if is_explicitly_mentioned:
        if not tool_context.tool_confirmation:
            v = explicitly_mentioned_value
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v) if v else ""
            else:
                v = str(v) if v else ""
            tool_context.request_confirmation(
                hint=f"Dimension '{dimension_name}' is set to: {v}. Reply 'done' to confirm or type the correct value.",
                payload={"dimension_name": dimension_name, "explicitly_mentioned_value": explicitly_mentioned_value}
            )
            return {"dimension_name": dimension_name, "pending_confirmation": True, "message": "Waiting for user to confirm or correct."}
        # user responded
        resp = None
        if hasattr(tool_context.tool_confirmation, "confirmed"):
            resp = getattr(tool_context.tool_confirmation, "user_response", None) or getattr(tool_context.tool_confirmation, "payload", {}).get("user_response")
        if isinstance(tool_context.tool_confirmation, dict):
            resp = tool_context.tool_confirmation.get("user_response")
        resp = (resp or "").strip().lower()
        val = explicitly_mentioned_value if resp == "done" else (resp if resp else explicitly_mentioned_value)
        return {"dimension_name": dimension_name, "dimension_type": dimension_type, "is_explicitly_mentioned": True, "confirmed_value": val, "dimension_should_be_complete": True}

    # If not explicitly mentioned, ask user
    if not tool_context.tool_confirmation:
        # First call - request user confirmation
        if question is None:
            question = f"Do you want to specify a {dimension_name} dimension for table selection?"
            if suggested_values:
                question += f" Suggested options: {', '.join(suggested_values)}"
        
        # Build hint message
        hint_text = f"""
📋 Dimension Requirement Specification

Dimension: {dimension_name}
Type: {dimension_type}

Reasoning: {reasoning or f"This dimension might be relevant for the prediction task"}

Question: {question}
"""
        if suggested_values:
            hint_text += f"\nSuggested options: {', '.join(suggested_values)}"
        
        hint_text += """

Please specify your requirement for this dimension.
Examples:
- For Geographic: "Borough", "Zip Code", "Neighborhood", "California", "Los Angeles County", etc.
- For Domain/Field: "Demographics", "Education", "Economy", etc.
- For Temporal: "Historical trends", "2020-2023", "Seasonal patterns", etc.
- For Population Group: "by Age Group", "by Income Level", "by Education", "18-25, 26-35", etc.

You can:
- Provide a specific value (e.g., "California", "by Age Group")
- Type "done" to finish specifying this dimension
- Type "skip" to skip this dimension entirely
- Say "no preference" or "all" to include all options
"""
        
        tool_context.request_confirmation(
            hint=hint_text,
            payload={
                "dimension_name": dimension_name,
                "dimension_type": dimension_type,
                "reasoning": reasoning or f"This dimension might be relevant for the prediction task",
                "suggested_values": suggested_values or [],
                "question": question
            }
        )
        
        return {
            "dimension_name": dimension_name,
            "dimension_type": dimension_type,
            "is_explicitly_mentioned": False,
            "pending_confirmation": True,
            "question": question,
            "suggested_values": suggested_values or [],
            "dimension_should_be_complete": False,  # Not complete yet
            "message": f"Waiting for user confirmation on {dimension_name} dimension"
        }
    
    # Second call - user has responded
    is_confirmed = tool_context.tool_confirmation

    # Get user's text response
    user_response = None
    if hasattr(tool_context, 'get_user_response'):
        user_response = tool_context.get_user_response()
    elif hasattr(tool_context, 'user_response'):
        user_response = tool_context.user_response
    elif isinstance(tool_context.tool_confirmation, dict):
        user_response = tool_context.tool_confirmation.get('user_response')
    
    # Use LLM to analyze user response and determine if dimension should be marked as complete
    dimension_should_be_complete = False
    auto_complete_reason = None
    interpreted_value = None
    
    if is_confirmed and user_response:
        # Get previous specifications for this dimension (if available)
        # Note: We can't access dimension_specifications from here, so we'll pass None
        # The orchestrator can pass this if needed in the future
        previous_specs = None  # Could be passed as parameter if needed
        
        # Use LLM to analyze the user response
        llm_analysis = _analyze_user_response_with_llm(
            user_response=str(user_response),
            dimension_name=dimension_name,
            previous_specifications=previous_specs
        )
        
        dimension_should_be_complete = llm_analysis.get("dimension_should_be_complete", False)
        auto_complete_reason = llm_analysis.get("reason", None)
        interpreted_value = llm_analysis.get("interpreted_value", None)
        
        # If LLM interpreted a value, use it; otherwise use the original response
        if interpreted_value:
            user_response = interpreted_value

    if is_confirmed:
        # User wants to specify this dimension
        return {
            "dimension_name": dimension_name,
            "dimension_type": dimension_type,
            "is_explicitly_mentioned": False,
            "user_wants_to_specify": True,
            "user_specified_value": user_response if user_response else None,
            "suggested_values": suggested_values or [],
            "dimension_should_be_complete": dimension_should_be_complete,
            "auto_complete_reason": auto_complete_reason,
            "message": f"User wants to specify {dimension_name} dimension" + 
                      (f" ({auto_complete_reason})" if auto_complete_reason else "")
        }
    else:
        # User doesn't want to specify this dimension
        return {
            "dimension_name": dimension_name,
            "dimension_type": dimension_type,
            "is_explicitly_mentioned": False,
            "user_wants_to_specify": False,
            "dimension_should_be_complete": True,  # User rejected, mark as complete (skipped)
            "auto_complete_reason": "User chose not to specify",
            "message": f"User does not want to specify {dimension_name} dimension - will not apply any restrictions to this dimension"
        }
        
def generate_table_selection_plan(
    analyzed_intent: Dict[str, Any],
    candidate_tables: List[str],
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Generate a chain-of-thoughts plan based on analyzed intent.
    First call: LLM generates plan structure and requests user approval.
    Second call: Returns approved/rejected status.
    
    Args:
        analyzed_intent: Output from analyze_user_intent
        candidate_tables: List of candidate table names
        tool_context: Tool context for user interaction
        
    Returns:
        Dictionary with plan and status
    """
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 1: First call - LLM should have generated a plan, request user confirmation
    if not tool_context.tool_confirmation:
        # The plan content comes from LLM's reasoning when calling this tool
        # We format it for user confirmation
        
        user_intent = analyzed_intent.get("user_intent", "")
        target = analyzed_intent.get("target_variable", "")
        
        tool_context.request_confirmation(
            hint=f"""
📋 Table Selection Plan Generated

🎯 User Intent: {user_intent}
🎯 Target Variable: {target}
📊 Available Candidate Tables: {len(candidate_tables)} tables

The agent has generated a plan for selecting relevant tables. 
Please review the plan details above and confirm if you want to proceed.
""",
            payload={
                "analyzed_intent": analyzed_intent,
                "candidate_tables": candidate_tables,
                "candidate_table_count": len(candidate_tables)
            }
        )
        
        return {
            "status": "pending_approval",
            "message": "Plan generated. Waiting for user approval.",
            "analyzed_intent": analyzed_intent
        }
    
    # -----------------------------------------------------------------------------------------------
    # SCENARIO 2: User confirmed or rejected
    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "message": "Plan approved. Proceeding with table search.",
            "plan": tool_context.tool_confirmation.payload
        }
    else:
        return {
            "status": "rejected",
            "message": "Plan rejected by user."
        }

def confirm_table_category(
    category_name: str,
    category_type: str,
    reasoning: str,
    question: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Confirm a table category requirement with the user.
    
    Args:
        category_name: Name of the category (e.g., "Demographics")
        category_type: Type of dimension (e.g., "Domain/Field", "Population Group")
        reasoning: Why this category is relevant
        question: Question to ask the user
        tool_context: Tool context for user interaction
        
    Returns:
        Dictionary with confirmation status
    """
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"""
📋 Table Category Confirmation

Category: {category_name}
Type: {category_type}

Reasoning: {reasoning}

Question: {question}

Please confirm if you need tables in this category (yes/no).
""",
            payload={
                "category_name": category_name,
                "category_type": category_type,
                "reasoning": reasoning,
                "question": question
            }
        )
        
        return {
            "status": "pending_approval",
            "message": f"Waiting for confirmation on category: {category_name}"
        }
    
    # User has responded
    is_confirmed = tool_context.tool_confirmation.get("confirmed", False)
    
    return {
        "status": "confirmed" if is_confirmed else "rejected",
        "category_name": category_name,
        "category_type": category_type,
        "confirmed": is_confirmed,
        "message": f"Category '{category_name}' {'confirmed' if is_confirmed else 'rejected'}"
    } 

def _get_default_base_dir() -> str:
    """Get default base_dir from config file."""
    
    config = load_config()
    return config['data']['base_dir']

def find_dataset_dir(dataset_name: str, base_dir: str = None) -> str:
    """
    Find the real directory name based on the cleaned dataset name (ignoring trailing spaces, case, etc.).
    
    Args:
        dataset_name: The dataset name returned by LLM (possibly without trailing spaces)
        base_dir: The base directory path (if None, reads from config)
        
    Returns:
        The real directory name (with spaces, etc.)
        
    Raises:
        FileNotFoundError: If the base directory does not exist
    """
    if base_dir is None:
        base_dir = _get_default_base_dir()
    
    base_path = Path(base_dir).resolve()
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_path}")
    
    # Clean the input name: convert to lowercase, remove trailing spaces, remove trailing numbers

    clean_input = dataset_name.lower().strip()
    clean_input = re.sub(r'\s+\d+$', '', clean_input).strip()
    
    # Create a mapping: cleaned name -> real directory name
    name_map = {}
    for d in base_path.iterdir():
        if not d.is_dir():
            continue
        clean_dir_name = d.name.lower().strip()
        clean_dir_name = re.sub(r'\s+\d+$', '', clean_dir_name).strip()
        name_map[clean_dir_name] = d.name
    
    # Find a match
    if clean_input in name_map:
        return name_map[clean_input]
    
    # If no match, try to match directly (possibly the user has given an exact name)
    direct_path = base_path / dataset_name
    if direct_path.exists() and direct_path.is_dir():
        return dataset_name
    
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found in {base_path}. "
        f"Available datasets: {sorted(name_map.values())}"
    )

def read_table_index(
    candidate_ids: List[str],
    index_path: Optional[str] = None,
    max_entries: Optional[int] = None,
    max_chars_per_field: Optional[int] = None,
    max_list_items: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load the Opendata table index and return only entries whose id is in candidate_ids.
    Use this when you have a list of candidate table IDs (e.g. from Opendata search) and need
    to read their index entries (id, description, attribution, possible_join_column, classification, domain)
    for ranking or selection.

    Args:
        candidate_ids: List of dataset/table IDs to look up (e.g. from metadata_by_dataset.keys()).
        index_path: Optional path to opendata_table_index.json. If None, uses default next to this module.
        max_entries: Optional. If set, limit the number of index entries returned (for prompt truncation).
        max_chars_per_field: Optional. If set, truncate string/list fields longer than this (for prompt truncation).
        max_list_items: Optional. If set, limit list fields (columns_name, columns_description, etc.) to first N items.

    Returns:
        Dict with keys: index_entries (list of matching index entries), count (number of matches),
        requested_count (len(candidate_ids)). Missing IDs in the index are simply omitted.
    """
    if index_path is None:
        index_path = str(Path(__file__).resolve().parent / "opendata_table_index.json")
    path = Path(index_path)
    if not path.exists():
        return {
            "index_entries": [],
            "count": 0,
            "requested_count": len(candidate_ids),
            "error": f"Index file not found: {index_path}",
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            full_index = json.load(f)
    except Exception as e:
        return {
            "index_entries": [],
            "count": 0,
            "requested_count": len(candidate_ids),
            "error": str(e),
        }
    if not isinstance(full_index, list):
        return {
            "index_entries": [],
            "count": 0,
            "requested_count": len(candidate_ids),
            "error": "Index is not a list",
        }
    id_set = {str(i).strip() for i in candidate_ids if i is not None}
    entries = [e for e in full_index if isinstance(e, dict) and e.get("id") and str(e.get("id")).strip() in id_set]

    # Prompt truncation: limit entries, field lengths, and list sizes when requested
    if max_entries is not None:
        entries = entries[:max_entries]
    if max_chars_per_field is not None or max_list_items is not None:
        truncated = []
        for e in entries:
            ne = {}
            for k, v in e.items():
                if isinstance(v, str):
                    if max_chars_per_field is not None and len(v) > max_chars_per_field:
                        ne[k] = v[:max_chars_per_field] + "...[truncated]"
                    else:
                        ne[k] = v
                elif isinstance(v, list):
                    lst = v
                    if max_list_items is not None and k in ("columns_name", "columns_description", "columns_datatype"):
                        lst = lst[:max_list_items]
                    if max_chars_per_field is not None:
                        nv = []
                        for x in lst:
                            sx = str(x) if not isinstance(x, str) else x
                            nv.append(sx[:max_chars_per_field] + "..." if len(sx) > max_chars_per_field else x)
                        ne[k] = nv
                    else:
                        ne[k] = lst
                else:
                    ne[k] = v
            truncated.append(ne)
        entries = truncated

    return {
        "index_entries": entries,
        "count": len(entries),
        "requested_count": len(candidate_ids),
    }


def build_topk_join_column_request(
    query_join_column_name: str,
    candidate_id: Optional[str] = None,
    query_join_column_description: Optional[str] = None,
    task: Optional[str] = None,
    candidate_columns: Optional[List[Dict[str, Any]]] = None,
    k: int = 5,
    index_path: Optional[str] = None,
) -> str:
    """
    Build the user message for the top-k join column selection agent.
    Reads the candidate table's index to get column name and description,
    then formats the request for the agent.

    Use with instruction = contents of prompt/topk_join_column_prompt.txt.
    Agent output should be JSON: {"top_k_columns": ["col1", ...], "reasoning": "..."}.

    Args:
        query_join_column_name: Join column name in the query/base table.
        candidate_id: Dataset/table ID from relevant_list (e.g. tbl.get("table_id")).
            If provided, reads index and ignores candidate_columns.
        query_join_column_description: Optional description of the query join column.
        task: Optional task/user intent for additional context.
        candidate_columns: List of dicts with "name" and "description" (per prompt INPUT).
            Used only when candidate_id is not provided (backward compatibility).
        k: Number of top candidate columns to return (default 5).
        index_path: Optional path to opendata_table_index.json. If None, uses default.

    Returns:
        User message string to send to the agent (formatted per prompt INPUT).
        Returns empty string if candidate_id is given but not found in index.
    """
    if candidate_id:
        result = read_table_index([candidate_id], index_path)
        entries = result.get("index_entries") or []
        if not entries:
            return ""
        entry = entries[0]
        columns_name = entry.get("columns_name") or []
        columns_description = entry.get("columns_description") or []
        while len(columns_description) < len(columns_name):
            columns_description.append("")
        candidate_columns = [
            {
                "name": str(n).strip(),
                "description": (columns_description[i] if i < len(columns_description) else "").strip(),
            }
            for i, n in enumerate(columns_name)
        ]

    cand_list = [
        {"name": c.get("name", c.get("column_name", "")), "description": c.get("description", "") or ""}
        for c in candidate_columns
    ]
    query_desc = query_join_column_description or ""
    task_block = ""
    if task:
        task_block = f"\ntask: {task}\n"
    return f"""query_join_column:
- name: {query_join_column_name}
- description: {query_desc}
{task_block}candidate_columns (name, description):
{json.dumps(cand_list, ensure_ascii=False, indent=2)}

k: {k}

Return JSON only."""

def read_metadata(dataset_name: str = None, base_dir: str = None, exclude_tables: List[str] = None) -> Dict[str, Any]:
    config = load_config()
    data_source = config.get('data', {}).get('data_source', 'local')
    print(f"[read_metadata] data_source={data_source}") 
    if data_source == 'datalake':
        print("[read_metadata] Using opendata API")  
        from datalake_client import SocrataDatalakeClient
        datalake_config = config.get('data', {}).get('datalake', {})
        client = SocrataDatalakeClient(datalake_config)
        search_domains = _opendata_search_domains if _opendata_search_domains is not None else datalake_config.get('domains', [])
        search_q = _opendata_search_q or ""
        return client.read_metadata(
            dataset_name,
            exclude_tables,
            search_domains=search_domains,
            search_q=search_q,
        )
    else:
        if base_dir is None:
            base_dir = _get_default_base_dir()

        base_path = Path(base_dir).resolve()
        
        if not base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {base_path}")
        
        # Normalize exclude_tables for case-insensitive matching
        exclude_set = set()
        if exclude_tables:
            exclude_set = {name.lower().strip() for name in exclude_tables}
        
        valid_dirs = {d.name.lower().strip(): d.name for d in base_path.iterdir() if d.is_dir()}

        if dataset_name:
            import re
            clean_input = dataset_name.lower().strip()
            clean_input = re.sub(r'\s+\d+$', '', clean_input).strip()
            
        
            if clean_input in valid_dirs:
                dataset_name = valid_dirs[clean_input]
            elif dataset_name.lower().strip() in valid_dirs:
                dataset_name = valid_dirs[dataset_name.lower().strip()]
        
        out: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        
        for ds in base_path.iterdir():
            if not ds.is_dir(): continue
        
            if dataset_name and ds.name != dataset_name: continue
            
            # HARDCODE: Exclude join table(s)
            if exclude_set and ds.name.lower().strip() in exclude_set:
                continue
            
            mf = ds / "metadata.json"
            if not mf.exists(): continue
            try:
                with mf.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                resource = meta.get("resource", {})
                out[ds.name] = {
                    "table_description": resource.get("description", "")
                }
            except Exception as e:
                errors[ds.name] = str(e)
                
        return {"metadata_by_dataset": out, "errors": errors}

def get_fasttext_sim(model, text1, text2):
    
    v1 = model.get_sentence_vector(str(text1))
    v2 = model.get_sentence_vector(str(text2))
    return 1 - cosine(v1, v2)

def compute_statistics(
    dataset_name: str,
    join_table_name: str,
    join_column: List[str],
    base_dir: str = None,
    data_filename: str = "rows.csv",
    max_rows: int = 1000,
    opendata_domain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compute statistics for each column in candidate_df compared with a known join column.

    Args:
        candidate_df: Candidate table DataFrame
        join_column: Name of the known join column
        join_column_table: DataFrame containing the join column (e.g., non-candidate table)
        dataset_name: Dataset folder name to read metadata from
        base_dir: Base directory containing datasets
        max_rows: Maximum number of rows to use (default 100)

    Returns:
        List of dicts, each containing statistics for a candidate column compared with join_column:
        - candidate_column: Column name from candidate_df
        - join_column: The known join column name
        - jaccard_similarity: Jaccard similarity between value sets
        - containment1: How much candidate column is contained in join column
        - containment2: How much join column is contained in candidate column
        - embedding_similarity: Embedding similarity (placeholder)
        - uniqueness_ratio_candidate: Uniqueness ratio of candidate column
        - uniqueness_ratio_join_column: Uniqueness ratio of join column
        - missing_rate_candidate: Missing rate of candidate column
        - missing_rate_join_column: Missing rate of join column
        - candidate_column_name: Column name from metadata
        - candidate_column_description: Column description from metadata
        - join_column_name: Join column name from metadata
        - join_column_description: Join column description from metadata
    """
    if base_dir is None:
        base_dir = _get_default_base_dir()

    try:
        # Load FastText model (lazy loading - only when needed)
        global _fasttext_model
        if '_fasttext_model' not in globals():
            try:
                _fasttext_model = FastText.load_model("fasttext.bin")
            except Exception as e:
                # If model file doesn't exist, create a dummy model or skip embedding similarity
                print(f"Warning: Could not load FastText model: {e}")
                _fasttext_model = None

        real_join_table_name = find_dataset_dir(join_table_name, base_dir)
        join_path = Path(base_dir) / real_join_table_name / data_filename
        join_column_table = pd.read_csv(join_path, low_memory=False)

        if opendata_domain:
            from datalake_client import SocrataDatalakeClient
            cfg = load_config()
            client = SocrataDatalakeClient(cfg.get("data", {}).get("datalake", {}))
            rows = client.read_data(dataset_name, opendata_domain, max_rows=max_rows)
            candidate_df = pd.DataFrame(rows) if rows else pd.DataFrame()
            real_candidate_name = dataset_name
        else:
            real_candidate_name = find_dataset_dir(dataset_name, base_dir)
            candidate_path = Path(base_dir) / real_candidate_name / data_filename
            candidate_df = pd.read_csv(candidate_path, low_memory=False)

        if candidate_df.empty or len(candidate_df.columns) == 0:
            return []
     
        try:
            candidate_df_sample = candidate_df.sort_values(by=list(candidate_df.columns)).head(max_rows).copy()
        except (TypeError, ValueError):
            candidate_df_sample = candidate_df.head(max_rows).copy()

        try:
            join_column_table_sample = join_column_table.sort_values(by=list(join_column_table.columns)).head(max_rows).copy()
        except (TypeError, ValueError):
            join_column_table_sample = join_column_table.head(max_rows).copy()
        
        # Verify join column(s) exist
        if isinstance(join_column, list):
            missing = [c for c in join_column if c not in join_column_table_sample.columns]
            if missing:
                raise ValueError(f"Join column(s) {missing} not found in join table")
        else:
            if join_column not in join_column_table_sample.columns:
                raise ValueError(f"Join column '{join_column}' not found in join table")

        # Read metadata to get column descriptions
        dataset_path = Path(base_dir) / real_candidate_name
        metadata_file = dataset_path / "metadata.json"
        
        column_metadata = {}
        
        if metadata_file.exists():
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            if "resource" in metadata:
                resource = metadata["resource"]
                column_names = resource.get("columns_name", [])
                descriptions = resource.get("columns_description", [])
                
                for i, col_name in enumerate(column_names):
                    column_metadata[col_name] = {
                        "name": col_name,
                        "description": descriptions[i] if i < len(descriptions) else ""
                    }
        
        # Get join column metadata
        if isinstance(join_column, list):
            join_col_info = [
                column_metadata.get(c, {"name": c, "description": ""})
                for c in join_column
            ]
        else:
            join_col_info = column_metadata.get(join_column, {"name": join_column, "description": ""})

        # Extract value set for join column
        if isinstance(join_column, list):
            join_names = []
            join_descs = []
            for c in join_column:
                info = column_metadata.get(c, {"name": c, "description": ""})
                join_names.append(info.get("name", c))
                d = info.get("description", "")
                if d:
                    join_descs.append(f"{c}: {d}")
            join_col_info = {
                "name": "||".join(join_names),
                "description": " | ".join(join_descs)
            }
        else:
            join_col_info = column_metadata.get(join_column, {"name": join_column, "description": ""})
        # Extract value set for join column
        if isinstance(join_column, list):
            join_col_values = (
                join_column_table_sample[join_column]
                .dropna()
                .astype(str)
                .agg("||".join, axis=1)
                .str.lower()
                .str.strip()
            )
        else:
            join_col_values = (
                join_column_table_sample[join_column]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
            )

        join_col_set = set(join_col_values)

        main_keys = join_column if isinstance(join_column, list) else [join_column]
        candidate_all_cols = list(candidate_df.columns)
        
        mapping_results = {}
        total_sim = 0
        for m_key in main_keys:
            best_match = None
            max_sim = -1.0
            for c_col in candidate_all_cols:
                if _fasttext_model is not None:
                    sim = get_fasttext_sim(_fasttext_model, m_key, c_col)
                else:
                    # Fallback: use simple string similarity if model not available
                    sim = 0.0  # or implement a simple string similarity
                if sim > max_sim:
                    max_sim = sim
                    best_match = c_col
            mapping_results[m_key] = {"match": best_match, "score": max_sim}
            total_sim += max_sim
        
        avg_emb_score = total_sim / len(main_keys)
        embedding_similarity = float(round(avg_emb_score, 4))
        
        # Compute statistics for each column in candidate_df
        final_results = [] 
        total_rows_candidate = len(candidate_df_sample)
        
        for candidate_col in candidate_df_sample.columns:
            # Get candidate column metadata
            candidate_col_info = column_metadata.get(candidate_col, {"name": candidate_col, "description": ""})
            
            # Extract value set for candidate column
            candidate_col_values = candidate_df_sample[candidate_col].dropna().astype(str).str.lower().str.strip()
            raw = candidate_df_sample[candidate_col].dropna()
            candidate_col_set = set(str(v).lower().strip() for v in raw)
            
            # Jaccard similarity
            intersection = candidate_col_set & join_col_set
            union = candidate_col_set | join_col_set
            jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
            
            # Set containment
            containment1 = len(intersection) / len(join_col_set) if len(join_col_set) > 0 else 0.0
            containment2 = len(intersection) / len(candidate_col_set) if len(candidate_col_set) > 0 else 0.0
            
            # Uniqueness and missing
            unique_count = candidate_df_sample[candidate_col].astype(str).nunique()
            uniqueness_ratio = unique_count / total_rows_candidate if total_rows_candidate > 0 else 0.0
            missing_rate = candidate_df_sample[candidate_col].isna().sum() / total_rows_candidate if total_rows_candidate > 0 else 0.0
            
            
            col_stats = {
                "candidate_column": candidate_col,
                "join_column": join_column,
                "jaccard_similarity": jaccard_similarity,
                "containment1": containment1,
                "containment2": containment2,
                "uniqueness_ratio_candidate": uniqueness_ratio,
                "missing_rate_candidate": missing_rate,
                "candidate_column_name": candidate_col_info["name"],
                "candidate_column_description": candidate_col_info["description"],
                "join_column_name": join_col_info["name"],
                "join_column_description": join_col_info["description"],
                "embedding_similarity": embedding_similarity
            }
            
            final_results.append(col_stats)
        
        
        sorted_results = sorted(
            final_results,
            key=lambda r: max(r["containment1"], r["containment2"], r["jaccard_similarity"]),
            reverse=True
        )[:5]
        for r in sorted_results:
            r.pop("jaccard_similarity", None)
            r.pop("containment1", None)
            r.pop("containment2", None)
        
        return sorted_results

    except Exception as e:
        print(f"[compute_statistics] Error: {e}")
        return []    
def compute_integration_quality(
    base_table_name: str,
    candidate_table_name: str,
    base_join_columns: List[str],
    candidate_join_columns: List[str] = None,
    base_dir: str = None,
    data_filename: str = "rows.csv",
    opendata_domain: Optional[str] = None,   
    max_rows: int = 1000,
) -> float:
    """
    Compute Integration Quality (IQ): proportion of instances in the base table 
    that can be successfully augmented by the candidate table.
    
    Args:
        base_table_name: Name of the base/join table
        candidate_table_name: Name of the candidate table to join
        base_join_columns: Join columns in the base table
        candidate_join_columns: Join columns in the candidate table (if None, same as base_join_columns)
        base_dir: Base directory containing datasets
        data_filename: CSV filename (default: "rows.csv")
    
    Returns:
        IQ value (float between 0.0 and 1.0): proportion of base table rows successfully joined
    """
    if base_dir is None:
        base_dir = _get_default_base_dir()

    # Load base table (always local - join table)
    real_base_name = find_dataset_dir(base_table_name, base_dir)
    base_path = Path(base_dir) / real_base_name / data_filename
    base_df = pd.read_csv(base_path, low_memory=False)

    # Load candidate table (API if opendata_domain, else local)
    if opendata_domain:
        from datalake_client import SocrataDatalakeClient
        cfg = load_config()
        api_client = SocrataDatalakeClient(cfg.get("data", {}).get("datalake", {}))
        rows = api_client.read_data(candidate_table_name, opendata_domain, max_rows=max_rows)
        candidate_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    else:
        real_candidate_name = find_dataset_dir(candidate_table_name, base_dir)
        candidate_path = Path(base_dir) / real_candidate_name / data_filename
        candidate_df = pd.read_csv(candidate_path, low_memory=False)

    if candidate_df.empty or len(candidate_df.columns) == 0:
        return 0.0
    
    # Use same join columns if candidate_join_columns not specified
    if candidate_join_columns is None:
        candidate_join_columns = base_join_columns
    
    # Verify columns exist
    missing_base = [col for col in base_join_columns if col not in base_df.columns]
    if missing_base:
        raise ValueError(f"Join columns {missing_base} not found in base table")
    
    missing_candidate = [col for col in candidate_join_columns if col not in candidate_df.columns]
    if missing_candidate:
        raise ValueError(f"Join columns {missing_candidate} not found in candidate table")

    # Before merge - normalize for case-insensitive matching (same as JoinValidatorCallback)
    base_df_copy = base_df.copy()
    cand_df_copy = candidate_df.copy()
    for col in base_join_columns:
        if col in base_df_copy.columns:
            base_df_copy[col] = base_df_copy[col].astype(str).str.upper().str.strip()
    for col in candidate_join_columns:
        if col in cand_df_copy.columns:
            cand_df_copy[col] = cand_df_copy[col].astype(str).str.upper().str.strip()
            
    # Perform join
    merged = pd.merge(
        base_df_copy,
        cand_df_copy,
        left_on=base_join_columns,
        right_on=candidate_join_columns,
        how='inner'
    )
        
    # Calculate IQ: proportion of base table rows successfully augmented
    total_base_rows = len(base_df)
    if total_base_rows == 0:
        return 0.0
    
    successfully_augmented_rows = len(merged)
    iq = successfully_augmented_rows / total_base_rows
    
    return float(iq)

def compute_feature_importance(
    base_table_name: str,
    candidate_table_name: str,
    base_join_columns: List[str],
    candidate_column: str,
    target_column: str,
    task_type: str,
    candidate_join_columns: Optional[List[str]] = None,
    base_dir: str = None,
    data_filename: str = "rows.csv",
    opendata_domain: Optional[str] = None,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Compute Feature Importance (FI): the improvement in prediction metrics 
    when adding a candidate column to the base table.
    
    FI = metrics_with_candidate_column - metrics_without_candidate_column
    
    Args:
        base_table_name: Name of the base/join table
        candidate_table_name: Name of the candidate table
        base_join_columns: Join columns in the base table
        candidate_column: The column from candidate table to evaluate
        target_column: Target column to predict (in base table)
        task_type: "regression" or "classification"
        candidate_join_columns: Join columns in candidate table (if None, same as base_join_columns)
        base_dir: Base directory containing datasets
        data_filename: CSV filename
        sample_size: Number of rows to sample for training (default: 1000)
    
    Returns:
        Dictionary with FI value and metadata
    """
    if base_dir is None:
        base_dir = _get_default_base_dir()
    
    try:
        # Load base table (always local - join table)
        real_base_name = find_dataset_dir(base_table_name, base_dir)
        base_path = Path(base_dir) / real_base_name / data_filename
        base_df = pd.read_csv(base_path, low_memory=False)

        # Load candidate table (API if opendata_domain, else local)
        if opendata_domain:
            from datalake_client import SocrataDatalakeClient
            cfg = load_config()
            api_client = SocrataDatalakeClient(cfg.get("data", {}).get("datalake", {}))
            rows = api_client.read_data(candidate_table_name, opendata_domain, max_rows=sample_size * 2)
            candidate_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        else:
            real_candidate_name = find_dataset_dir(candidate_table_name, base_dir)
            candidate_path = Path(base_dir) / real_candidate_name / data_filename
            candidate_df = pd.read_csv(candidate_path, low_memory=False)

        if candidate_df.empty or len(candidate_df.columns) == 0:
            return {"error": "Candidate table empty", "feature_importance": 0.0}
        
        # Verify target column exists
        if target_column not in base_df.columns:
            return {
                "error": f"Target column '{target_column}' not found in base table",
                "feature_importance": 0.0
            }
        
        # Verify candidate column exists
        if candidate_column not in candidate_df.columns:
            return {
                "error": f"Candidate column '{candidate_column}' not found in candidate table",
                "feature_importance": 0.0
            }
        
        # Use same join columns if candidate_join_columns not specified
        if candidate_join_columns is None:
            candidate_join_columns = base_join_columns
        
        # Verify join columns exist
        missing_base = [col for col in base_join_columns if col not in base_df.columns]
        if missing_base:
            return {
                "error": f"Join columns {missing_base} not found in base table",
                "feature_importance": 0.0
            }
        
        missing_candidate = [col for col in candidate_join_columns if col not in candidate_df.columns]
        if missing_candidate:
            return {
                "error": f"Join columns {missing_candidate} not found in candidate table",
                "feature_importance": 0.0
            }
        
        # Before merge - normalize for case-insensitive matching (same as JoinValidatorCallback)
        base_df_copy = base_df.copy()
        cand_df_copy = candidate_df.copy()
        for col in base_join_columns:
            if col in base_df_copy.columns:
                base_df_copy[col] = base_df_copy[col].astype(str).str.upper().str.strip()
        for col in candidate_join_columns:
            if col in cand_df_copy.columns:
                cand_df_copy[col] = cand_df_copy[col].astype(str).str.upper().str.strip()

        merged = pd.merge(
            base_df_copy,
            cand_df_copy,
            left_on=base_join_columns,
            right_on=candidate_join_columns,
            how='inner'
        )
        
        if len(merged) == 0:
            return {
                "error": "No rows matched after join",
                "feature_importance": 0.0
            }
        
        # Sample rows (take first sample_size rows after join)
        if len(merged) > sample_size:
            merged_sampled = merged.head(sample_size).copy()
        else:
            merged_sampled = merged.copy()
        
        # Prepare baseline features (base table columns, excluding target and join columns)
        baseline_features = [col for col in base_df.columns 
                           if col != target_column and col not in base_join_columns]
        
        if len(baseline_features) == 0:
            return {
                "error": "No baseline features available (base table only has target and join columns)",
                "feature_importance": 0.0
            }
        
        # Prepare datasets
        # Baseline: base table features only
        baseline_df = merged_sampled[baseline_features + [target_column]].copy()
        
        # Augmented: base table features + candidate column
        augmented_df = merged_sampled[baseline_features + [candidate_column] + [target_column]].copy()
        
        # Remove rows with missing target
        baseline_df = baseline_df.dropna(subset=[target_column])
        augmented_df = augmented_df.dropna(subset=[target_column])
        
        if len(baseline_df) < 3 or len(augmented_df) < 3:
            return {
                "error": f"Insufficient data after removing missing values (baseline: {len(baseline_df)}, augmented: {len(augmented_df)})",
                "feature_importance": 0.0
            }
        
        # Evaluate baseline and augmented models
        baseline_metric = _train_and_evaluate(
            baseline_df, target_column, task_type
        )
        augmented_metric = _train_and_evaluate(
            augmented_df, target_column, task_type
        )
        
        # Feature Importance = improvement in metric
        fi = augmented_metric - baseline_metric
        
        return float(fi)    
        
    except Exception as e:
        return {
            "error": str(e),
            "feature_importance": 0.0
        }


def _train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    task_type: str
) -> float:
    """
    Train model and return evaluation metric.
    - Regression: Linear Regression, metric = R2 score
    - Classification: XGBoost, metric = F1 score (weighted)
    """
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle _vector columns: fill NaN with mean vector
    vector_cols = [c for c in X.columns if c.endswith('_vector')]
    for vec_col in vector_cols:
        # Collect all non-NaN vectors
        valid_vecs = []
        for v in X[vec_col]:
            if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                valid_vecs.append(np.array(v))
        
        if len(valid_vecs) > 0:
            # Calculate mean vector
            mean_vec = np.nanmean(valid_vecs, axis=0).tolist()
        else:
            mean_vec = [0.0]    
        
        # Fill NaN or invalid values with mean vector
        X[vec_col] = X[vec_col].apply(
            lambda v: v if isinstance(v, (list, np.ndarray)) and len(v) == len(mean_vec) else mean_vec
        )

    # Expand _vector columns to scalar columns
    for vec_col in vector_cols:
        vectors = X[vec_col].tolist()
        vec_dim = len(vectors[0]) if vectors else 0
        base_name = vec_col.replace('_vector', '')
        
        for i in range(vec_dim):
            X[f'{base_name}_dim{i}'] = [v[i] for v in vectors]
        
        X = X.drop(columns=[vec_col])

    # Delete _categories columns
    X = X.drop(columns=[c for c in X.columns if c.endswith('_categories')], errors='ignore')

    # Handle missing values in remaining scalar features
    for col in X.columns:
        if col.endswith('_vector'):
            continue  # vector 已处理
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna('Unknown')
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Force fill any remaining NaN
    X = X.fillna(0)
    y = y.fillna(y.mean() if pd.api.types.is_numeric_dtype(y) else 0)
    
    # Convert to numpy array
    X_array = X.values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_array, y.values, test_size=0.3, random_state=42
    )
    
    # Train and evaluate
    if task_type == "regression":
        # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric = r2_score(y_test, y_pred)
    elif task_type == "classification":
        # XGBoost
        n_classes = len(np.unique(y_train))
        xgb_params = {
            'objective': 'multi:softprob' if n_classes > 2 else 'binary:logistic',
            'random_state': 42,
            'eval_metric': 'mlogloss' if n_classes > 2 else 'logloss',
            'verbosity': 0
        }
        if n_classes > 2:
            xgb_params['num_class'] = n_classes
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'regression' or 'classification'")
    
    return float(metric)


def compute_utility_gain_from_params(
    base_table_name: str,
    candidate_table_name: str,
    base_join_columns: List[str],
    candidate_column: str,
    target_column: str,
    task_type: str,
    candidate_join_columns: Optional[List[str]] = None,
    base_dir: str = None,
    data_filename: str = "rows.csv",
    sample_size: int = 1000,
    opendata_domain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute Utility Gain by first calculating IQ and FI, then multiplying them.
    
    Returns a dictionary with utility_gain, iq, fi, and other metadata.
    """
    if base_dir is None:
        base_dir = _get_default_base_dir()
    
    # Compute IQ
    try:
        iq_result = compute_integration_quality(
            base_table_name=base_table_name,
            candidate_table_name=candidate_table_name,
            base_join_columns=base_join_columns,
            candidate_join_columns=candidate_join_columns,
            base_dir=base_dir,
            data_filename=data_filename,
            opendata_domain=opendata_domain,
            max_rows=5000,
        )
        
        # compute_integration_quality returns float, not dict
        if isinstance(iq_result, dict) and "error" in iq_result:
            return {
                "error": f"IQ computation failed: {iq_result['error']}",
                "utility_gain": 0.0,
                "iq": 0.0,
                "fi": 0.0
            }
        
        # iq_result is a float
        iq = float(iq_result) if not isinstance(iq_result, dict) else iq_result.get("iq", 0.0)
        
    except Exception as e:
        return {
            "error": f"IQ computation failed: {str(e)}",
            "utility_gain": 0.0,
            "iq": 0.0,
            "fi": 0.0
        }
    
    # Compute FI
    fi_result = compute_feature_importance(
        base_table_name=base_table_name,
        candidate_table_name=candidate_table_name,
        base_join_columns=base_join_columns,
        candidate_column=candidate_column,
        target_column=target_column,
        task_type=task_type,
        candidate_join_columns=candidate_join_columns,
        base_dir=base_dir,
        data_filename=data_filename,
        sample_size=sample_size,
        opendata_domain=opendata_domain,
    )
    
    # Check if fi_result is dict and has error
    if isinstance(fi_result, dict) and "error" in fi_result:
        return {
            "error": f"FI computation failed: {fi_result['error']}",
            "utility_gain": 0.0,
            "iq": iq,
            "fi": 0.0
        }
    
    # fi_result should be dict with "feature_importance" key, but handle float case too
    if isinstance(fi_result, dict):
        fi = fi_result.get("feature_importance", 0.0)
    else:
        fi = float(fi_result) if isinstance(fi_result, (int, float)) else 0.0
    
    # Compute Utility Gain
    utility_gain = iq * fi
    
    return {
        "utility_gain": float(utility_gain),
        "iq": float(iq),
        "fi": float(fi),
        "candidate_column": candidate_column
    }