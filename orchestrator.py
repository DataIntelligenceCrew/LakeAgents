import json
import re
import os
import pandas as pd
import asyncio # Required for running the async entry point
from pathlib import Path
from typing import Any, Dict, List, Optional
from google.adk.runners import InMemoryRunner
from google.genai import types
from table_selection_agent import build_table_selection_agent
from join_column_selection_agent import build_join_column_choose_agent
from callback import JoinValidatorCallback, AugmentValidatorCallback
import fasttext
from functools import partial
from llm_agent_tools import find_dataset_dir
from augment_column_selection_agent import build_utility_gain_agent
from agent_config_loader import AgentPipelineConfig

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
    user_intent: Optional[str] = None,  # NEW: User intent parameter
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
    
    BASE_DIR = config.base_dir
    
    # Build agents with config
    table_runner = InMemoryRunner(agent=build_table_selection_agent(config=config))
    joincol_runner = InMemoryRunner(agent=build_join_column_choose_agent(config=config))

    base_path = Path(BASE_DIR)
    candidate_names = [
        item.name for item in base_path.iterdir()
        if item.is_dir() and (item / "metadata.json").exists() and item.name != join_table_name
    ]

    # ---- Phase 1: Table Selection ----
    print("🚀 Running Table Selection Agent...")
    print(f"📝 User Intent: {user_intent}")
    
    table_prompt = f"""
User Intent: {user_intent}

Task Information:
- Target Column: {target_column}
- Task Type: {task_type}
- Join Table: {join_table_name}
- Join Columns: {join_column}
- Candidate Tables ({len(candidate_names)}): {', '.join(candidate_names[:10])}{'...' if len(candidate_names) > 10 else ''}

Please follow the workflow defined in your prompt to complete this task.
"""
   
    analyzed_intent = None  # Store the result from analyze_user_intent
    current_prompt = table_prompt
    max_iterations = 10  
    iteration = 0
    
    current_dimension = None  # dimension name
    dimension_specifications = {}  # store each dimension specification
    dimension_complete = set()  # completed dimensions

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*80}")
        print(f"--- Iteration {iteration} ---")
        print(f"{'='*80}\n")
        
        table_events = await table_runner.run_debug(current_prompt)
        
        # First, check for analyze_user_intent result
        for event in table_events:
            if hasattr(event, 'actions') and event.actions:
                if hasattr(event.actions, 'tool_calls'):
                    for tool_call in event.actions.tool_calls:
                        tool_name = None
                        if hasattr(tool_call, 'function_name'):
                            tool_name = tool_call.function_name
                        elif hasattr(tool_call, 'name'):
                            tool_name = tool_call.name
                        elif hasattr(tool_call, 'function'):
                            tool_name = tool_call.function
                        
                        if tool_name == "analyze_user_intent":
                            if hasattr(tool_call, 'result'):
                                analyzed_intent = tool_call.result
                                print(f"\n[DEBUG] Got analyzed_intent result")
                                break
        
        # Extract confirmation request from events
        pending_confirmation = False
        confirmation_hint = None
        confirmation_payload = None
        tool_call_name = None
        
        # Debug: Print all tool calls to understand what's happening
        print(f"\n[DEBUG] Checking {len(table_events)} events for tool calls...")
        for event_idx, event in enumerate(table_events):
            if hasattr(event, 'actions') and event.actions:
                if hasattr(event.actions, 'tool_calls'):
                    for tool_idx, tool_call in enumerate(event.actions.tool_calls):
                        tool_name = None
                        if hasattr(tool_call, 'function_name'):
                            tool_name = tool_call.function_name
                        elif hasattr(tool_call, 'name'):
                            tool_name = tool_call.name
                        elif hasattr(tool_call, 'function'):
                            tool_name = tool_call.function
                        
                        if tool_name:
                            print(f"[DEBUG] Found tool call: {tool_name}")
                            if hasattr(tool_call, 'result'):
                                result = tool_call.result
                                if isinstance(result, dict):
                                    print(f"[DEBUG] Result keys: {list(result.keys())}")
                                    print(f"[DEBUG] pending_confirmation: {result.get('pending_confirmation')}")
                                    print(f"[DEBUG] status: {result.get('status')}")
        
        # Check events for tool calls that returned pending_approval status or pending_confirmation
        # IMPORTANT: Only process the FIRST pending confirmation to ensure one-by-one interaction
        for event in table_events:
            if hasattr(event, 'actions') and event.actions:
                # Check tool calls
                if hasattr(event.actions, 'tool_calls'):
                    for tool_call in event.actions.tool_calls:
                        # Get tool call name - try different attribute names
                        if hasattr(tool_call, 'function_name'):
                            tool_call_name = tool_call.function_name
                        elif hasattr(tool_call, 'name'):
                            tool_call_name = tool_call.name
                        elif hasattr(tool_call, 'function'):
                            tool_call_name = tool_call.function
                        
                        # Check if this is confirm_dimension_requirement call
                        is_dimension_confirm = (tool_call_name == "confirm_dimension_requirement" or 
                                              ("dimension" in str(tool_call_name).lower() and "confirm" in str(tool_call_name).lower()))
                        
                        # Check tool call result
                        if hasattr(tool_call, 'result'):
                            result = tool_call.result
                            
                            # Check for pending status: either "pending_approval" or "pending_confirmation"
                            is_pending = False
                            if isinstance(result, dict):
                                is_pending = (result.get("status") == "pending_approval" or 
                                            result.get("pending_confirmation") == True)
                            
                            # If it's a dimension confirm call, we need to check it even if not explicitly pending
                            if is_dimension_confirm or is_pending:
                                # Extract information from tool call arguments - try different attribute names
                                args = {}
                                if hasattr(tool_call, 'args'):
                                    args = tool_call.args if isinstance(tool_call.args, dict) else {}
                                elif hasattr(tool_call, 'arguments'):
                                    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                                
                                # For confirm_dimension_requirement, check if it needs user input
                                if is_dimension_confirm:
                                    # Get dimension_name from args or result
                                    dimension_name = None
                                    if args and args.get("dimension_name"):
                                        dimension_name = args.get("dimension_name", "")
                                    elif isinstance(result, dict) and result.get("dimension_name"):
                                        dimension_name = result.get("dimension_name", "")
                                    
                                    if dimension_name:
                                        # Check if this dimension is already complete
                                        if dimension_name in dimension_complete:
                                            # Skip - this dimension is already done, continue to next event
                                            continue
                                        
                                        # Check if this dimension needs user input
                                        # It needs user input if:
                                        # 1. Result has pending_confirmation=True or status="pending_approval"
                                        # 2. is_explicitly_mentioned is False (in args or result)
                                        needs_user_input = False
                                        
                                        # Check result first
                                        if isinstance(result, dict):
                                            if result.get("pending_confirmation") == True or result.get("status") == "pending_approval":
                                                needs_user_input = True
                                            elif result.get("is_explicitly_mentioned") == False:
                                                needs_user_input = True
                                        
                                        # Check args if not determined yet
                                        if not needs_user_input:
                                            if args.get("is_explicitly_mentioned") == False:
                                                needs_user_input = True
                                        
                                        # If it's a confirm_dimension_requirement call and we don't know yet, assume it needs input
                                        # (since the tool is specifically for asking user)
                                        if not needs_user_input:
                                            needs_user_input = True
                                        
                                        # Only process if it needs user input and we haven't found a pending confirmation yet
                                        if needs_user_input and not pending_confirmation:
                                            pending_confirmation = True
                                            current_dimension = dimension_name  # Track current dimension
                                            
                                            # Get suggested_values from args or result
                                            suggested_values = args.get("suggested_values", [])
                                            if not suggested_values and isinstance(result, dict):
                                                suggested_values = result.get("suggested_values", [])
                                            
                                            suggested_text = ""
                                            if suggested_values:
                                                suggested_text = f"\nSuggested options: {', '.join(suggested_values)}"
                                            
                                            # Check if we have previous specifications for this dimension
                                            previous_specs = dimension_specifications.get(dimension_name, [])
                                            previous_specs_text = ""
                                            if previous_specs:
                                                previous_specs_text = f"\n\nPrevious specifications for this dimension: {', '.join(previous_specs)}"
                                            
                                            # Get question from result or args
                                            question = args.get("question", "")
                                            if not question and isinstance(result, dict):
                                                question = result.get("question", "")
                                            if not question:
                                                question = f"Do you want to specify a {dimension_name} dimension for table selection?"
                                            
                                            # Get reasoning from result or args
                                            reasoning = args.get("reasoning", "")
                                            if not reasoning and isinstance(result, dict):
                                                reasoning = result.get("message", "")
                                            
                                            # Get dimension_type from args or result
                                            dimension_type = args.get("dimension_type", "")
                                            if not dimension_type and isinstance(result, dict):
                                                dimension_type = result.get("dimension_type", "")
                                            
                                            confirmation_payload = {
                                                "dimension_name": dimension_name,
                                                "dimension_type": dimension_type,
                                                "reasoning": reasoning,
                                                "question": question,
                                                "suggested_values": suggested_values,
                                                "is_explicitly_mentioned": args.get("is_explicitly_mentioned", False) if args else (result.get("is_explicitly_mentioned", False) if isinstance(result, dict) else False),
                                                "confirmation_type": "dimension"
                                            }
                                            confirmation_hint = f"""
                            📋 Dimension Requirement Specification

                            Dimension: {dimension_name}
                            Type: {dimension_type}

                            Reasoning: {reasoning}

                            Question: {question}
                            {suggested_text}{previous_specs_text}

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
                            """
                                            tool_call_name = "confirm_dimension_requirement"
                                            print(f"[DEBUG] ✅ Detected pending confirmation for dimension: {dimension_name}")
                                            # Break after finding the first one
                                            break
                                
                                # Break outer loop if we found a pending confirmation
                                if pending_confirmation:
                                    break
                # Break event loop if we found a pending confirmation
                if pending_confirmation:
                    break
                
                # Also check for confirmation hints in the agent's output/content
                # Only check if we haven't found a confirmation from tool calls yet
                if not pending_confirmation and hasattr(event, 'content'):
                    content = str(event.content)
                    # Look for dimension requirement hints
                    if ("Dimension Requirement" in content or "Please specify" in content or ("Dimension:" in content and "Category:" not in content)):
                        # Try to extract dimension info from content using regex
                        import re
                        dimension_match = re.search(r'Dimension:\s*([^\n]+)', content, re.IGNORECASE)
                        type_match = re.search(r'Type:\s*([^\n]+)', content, re.IGNORECASE)
                        reasoning_match = re.search(r'Reasoning:\s*([^\n]+)', content, re.IGNORECASE)
                        question_match = re.search(r'Question:\s*([^\n]+)', content, re.IGNORECASE)
                        suggested_match = re.search(r'Suggested options:\s*([^\n]+)', content, re.IGNORECASE)
                        
                        if dimension_match:
                            pending_confirmation = True
                            tool_call_name = "confirm_dimension_requirement"
                            suggested_values = []
                            if suggested_match:
                                suggested_text = suggested_match.group(1).strip()
                                suggested_values = [v.strip() for v in suggested_text.split(',')]
                            
                            confirmation_payload = {
                                "dimension_name": dimension_match.group(1).strip(),
                                "dimension_type": type_match.group(1).strip() if type_match else "",
                                "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
                                "question": question_match.group(1).strip() if question_match else "",
                                "suggested_values": suggested_values,
                                "confirmation_type": "dimension"
                            }
                            confirmation_hint = content
                            break
        
        # Also check events for adk_request_confirmation calls
        # These are special function calls that contain toolConfirmation data
        # IMPORTANT: Only extract the FIRST confirmation to ensure one-by-one interaction
        if not pending_confirmation and table_events:
            import re
            # Check all events for adk_request_confirmation
            for event in table_events:
                if hasattr(event, 'content'):
                    content = str(event.content)
                    # Check if this is an adk_request_confirmation event
                    if "adk_request_confirmation" in content or "toolConfirmation" in content:
                        # Extract the FIRST confirmation from the content
                        # Pattern: look for the first occurrence of toolConfirmation
                        # The structure is: 'toolConfirmation': { 'confirmed': False, 'hint': """...""", 'payload': {...} }
                        # We need to find the first complete toolConfirmation block
                        
                        # First, find all toolConfirmation blocks
                        tool_confirmation_pattern = r"'toolConfirmation':\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}"
                        all_confirmations = re.finditer(tool_confirmation_pattern, content, re.DOTALL | re.IGNORECASE)
                        
                        # Get the first one
                        first_confirmation = next(all_confirmations, None)
                        
                        if first_confirmation and not pending_confirmation:
                            confirmation_block = first_confirmation.group(1)
                            
                            # Extract payload section
                            payload_match = re.search(r"'payload':\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}", confirmation_block, re.DOTALL | re.IGNORECASE)
                            payload_text = payload_match.group(1) if payload_match else ""
                            
                            # Extract hint
                            hint_match = re.search(r"'hint':\s*\"\"\"(.*?)\"\"\"", confirmation_block, re.DOTALL | re.IGNORECASE)
                            hint_text = hint_match.group(1).strip() if hint_match else ""
                            
                            # Check if this is a dimension requirement (has dimension_name)
                            dimension_match = re.search(r"'dimension_name':\s*'([^']+)'", payload_text, re.IGNORECASE)
                            
                            if dimension_match:
                                # This is a dimension requirement
                                first_dimension = dimension_match.group(1)
                                
                                pending_confirmation = True
                                tool_call_name = "confirm_dimension_requirement"
                                confirmation_hint = hint_text if hint_text else f"Please specify your requirement for the '{first_dimension}' dimension."
                                
                                # Extract payload fields
                                type_match = re.search(r"'dimension_type':\s*'([^']+)'", payload_text, re.IGNORECASE)
                                reasoning_match = re.search(r"'reasoning':\s*'([^']+)'", payload_text, re.IGNORECASE)
                                question_match = re.search(r"'question':\s*'([^']+)'", payload_text, re.IGNORECASE)
                                suggested_match = re.search(r"'suggested_values':\s*\[(.*?)\]", payload_text, re.DOTALL | re.IGNORECASE)
                                
                                suggested_values = []
                                if suggested_match:
                                    suggested_text = suggested_match.group(1)
                                    # Extract individual values from the list
                                    value_matches = re.findall(r"'([^']+)'", suggested_text)
                                    suggested_values = value_matches
                                
                                confirmation_payload = {
                                    "dimension_name": first_dimension,
                                    "dimension_type": type_match.group(1).strip() if type_match else "",
                                    "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
                                    "question": question_match.group(1).strip() if question_match else "",
                                    "suggested_values": suggested_values,
                                    "confirmation_type": "dimension"
                                }
                                
                                # Only process the first one - break immediately
                                break
        
        # If we found a pending confirmation, wait for user input BEFORE continuing
        if pending_confirmation:
            confirmation_type = confirmation_payload.get("confirmation_type", "dimension") if confirmation_payload else "dimension"
            
            if confirmation_hint:
                print("\n" + "="*80)
                print(confirmation_hint)
                print("="*80)
            else:
                # If no hint extracted, show a generic message
                print("\n" + "="*80)
                print("📋 Dimension Requirement Specification")
                if confirmation_payload and confirmation_payload.get("dimension_name"):
                    print(f"Dimension: {confirmation_payload.get('dimension_name')}")
                print("="*80)
            
            # Handle dimension requirement (needs specific value input)
            if confirmation_type == "dimension":
                # Wait for user input - user needs to provide specific value
                while True:
                    user_input = input("\nYour specification (or 'skip' to skip this dimension, 'done' to finish this dimension): ").strip()
                    if user_input.lower() == "skip":
                        # User doesn't want to specify this dimension
                        user_specified_value = None
                        user_wants_to_specify = False
                        user_response_text = "User chose to skip this dimension. Use suggested values or proceed without this dimension."
                        dimension_name = confirmation_payload.get("dimension_name", "")
                        dimension_complete.add(dimension_name)  # 标记为完成
                        break
                    elif user_input.lower() == "done":
                        # User wants to finish this dimension
                        dimension_name = confirmation_payload.get("dimension_name", "")
                        dimension_complete.add(dimension_name)  # 标记为完成
                        user_response_text = f"User has finished specifying the {dimension_name} dimension. Proceed to next dimension."
                        break
                    elif user_input:
                        # User provided a specific value
                        user_specified_value = user_input
                        user_wants_to_specify = True
                        dimension_name = confirmation_payload.get("dimension_name", "")
                        
                        # 记录到维度规格中（支持多值）
                        if dimension_name not in dimension_specifications:
                            dimension_specifications[dimension_name] = []
                        dimension_specifications[dimension_name].append(user_input)
                        
                        user_response_text = f"User specified: {user_input} for {dimension_name} dimension."
                        break
                    else:
                        print("⚠️  Please enter a specification, 'skip', or 'done'")
                
                # Record the dimension specification
                dimension_name = confirmation_payload.get("dimension_name", "")
                
                if user_input.lower() == "skip":
                    print(f"\n⏭️  Dimension '{dimension_name}': skipped (using suggested values)")
                elif user_input.lower() == "done":
                   
                    specs = dimension_specifications.get(dimension_name, [])
                    if specs:
                        print(f"\n✅ Dimension '{dimension_name}' completed with specifications: {', '.join(specs)}")
                    else:
                        print(f"\n✅ Dimension '{dimension_name}' completed")
                else:
                    specs = dimension_specifications.get(dimension_name, [])
                    print(f"\n📝 Dimension '{dimension_name}' specification: {user_input}")
                    if len(specs) > 1:
                        print(f"   (All specifications so far: {', '.join(specs)})")
                
              
                is_dimension_complete = dimension_name in dimension_complete
                
                # Build prompt to continue
                if is_dimension_complete:
               
                    completed_dimensions = ', '.join(dimension_complete) if dimension_complete else "None"
                    all_dimension_specs = "\n".join([
                        f"- {dim}: {', '.join(specs)}" 
                        for dim, specs in dimension_specifications.items() 
                        if specs
                    ])
                    
                    # Check if all dimensions are complete
                    # Normalize dimension names for comparison (case-insensitive)
                    normalized_complete = {dim.lower().replace("/", "").replace(" ", "") for dim in dimension_complete}
                    expected_dimensions = {"geographic", "domainfield", "temporal", "populationgroup"}
                    if normalized_complete.issuperset(expected_dimensions) or len(dimension_complete) >= 4:
                        # All dimensions complete, proceed to table selection
                        current_prompt = f"""
All dimensions have been confirmed:

All dimension specifications:
{all_dimension_specs if all_dimension_specs else "None"}

IMPORTANT: 
- All dimensions are now complete.
- Proceed to STEP 3: Select Tables Based on Confirmed Dimensions
- Call read_metadata(dataset_name=None) to get ALL candidate tables
- For each table, check if it matches the confirmed dimensions based on table description
- Match tables to the confirmed dimensions: {all_dimension_specs}
- Return the most relevant tables with reasoning explaining which dimensions each table matches
"""
                    else:
                        # Not all dimensions complete, continue with next dimension
                        current_prompt = f"""
The user has completed specifying the {dimension_name} dimension.

All completed dimensions: {completed_dimensions}
All dimension specifications:
{all_dimension_specs if all_dimension_specs else "None yet"}

IMPORTANT: 
- The {dimension_name} dimension is now complete. 
- Please proceed to the NEXT dimension that hasn't been completed yet.
- Process dimensions in this order: 1. Geographic, 2. Domain/Field, 3. Temporal, 4. Population Group
- Only ask about dimensions that haven't been completed yet.
"""
                else:
                   
                    current_specs = dimension_specifications.get(dimension_name, [])
                    specs_text = f" (All specifications so far: {', '.join(current_specs)})" if current_specs else ""
                    
                    current_prompt = f"""
The user has provided a specification for the {dimension_name} dimension:
- User Input: {user_input}
- Interpretation: {user_response_text}
{specs_text}

IMPORTANT: 
- You are still working on the {dimension_name} dimension.
- Based on the user's answer, you can ask a MORE SPECIFIC follow-up question if needed.
- For example:
  * If user said "California", you can ask: "Do you have a preference for specific counties or cities in California?"
  * If user said "by Age Group", you can ask: "Which age groups are you most interested in?"
  * If user said "Los Angeles County", you can ask: "Do you need data for specific cities or neighborhoods within Los Angeles County?"
- Call confirm_dimension_requirement again with a more specific question for the SAME dimension ({dimension_name}).
- Only move to the next dimension when you have gathered sufficient information, or when user says "done".
- If you think you have enough information, you can ask: "Is there anything else you'd like to specify for {dimension_name}? (or type 'done' to finish)"
"""
            
            continue
        
        # Check if there is a final result (relevant_tables)
        table_json_str = "{}"
        for event in reversed(table_events):
            if hasattr(event, 'actions') and event.actions.state_delta:
                if "relevant_tables" in event.actions.state_delta:
                    table_json_str = event.actions.state_delta["relevant_tables"]
                    break
        
        # If found the result, exit the loop
        if table_json_str != "{}":
            table_data = extract_json(table_json_str)
            if table_data.get("relevant_tables"):
                print("\n✅ Agent has completed table selection")
                break
        
        # If there is no pending confirmation, and no result, the agent may be waiting
        # Or need to continue running
        # May need to check the last output of the agent, to see if it needs to continue
        # Here simplified: if no progress for a certain number of iterations, exit the loop
        if iteration >= 10:
            print("⚠️  Reached maximum iterations, exit the interactive loop")
            break

    # Extract the final result
    table_json_str = "{}"
    for event in reversed(table_events):
        if hasattr(event, 'actions') and event.actions.state_delta:
            if "relevant_tables" in event.actions.state_delta:
                table_json_str = event.actions.state_delta["relevant_tables"]
                break
    
    table_data = extract_json(table_json_str)
    relevant_list = table_data.get("relevant_tables", [])

    # Print dimension specifications summary
    print(f"\n📊 Confirmed Dimensions:")
    for dim, specs in dimension_specifications.items():
        if specs:
            print(f"  - {dim}: {', '.join(specs)}")
    
    print(f"\n📋 Table Selection Results: {len(relevant_list)} tables found")
    if len(relevant_list) == 0:
        print("   ⚠️  No relevant tables found. Check table selection agent output.")
        print(f"   Debug: table_data = {table_data}")
    else:
        for i, table_info in enumerate(relevant_list, 1):
            print(f"   {i}. {table_info.get('table_name', 'Unknown')}")
            if 'reasoning' in table_info:
                print(f"      Reasoning: {table_info['reasoning']}")

    real_join_table_name = find_dataset_dir(join_table_name, BASE_DIR)
    join_df = pd.read_csv(Path(BASE_DIR) / real_join_table_name / config.data_filename, low_memory=False)
    callback = JoinValidatorCallback(join_table_df=join_df, base_dir=BASE_DIR, config=config)

    # ---- Phase 2: Join Column selection ----
    final_results = []
    for table_info in relevant_list:
        cand_name = table_info["table_name"]
        print(f"🔍 Verifying Candidate: {cand_name}")
        
        jc_prompt = f"""
        TASK: Verify if '{cand_name}' can join with '{join_table_name}'.
        
        REQUIRED PARAMETERS:
        - dataset_name: "{cand_name}"
        - join_table_name: "{join_table_name}"
        - join_column: {join_column}
        - base_dir: "{BASE_DIR}"

        INSTRUCTION:
        1. You MUST call 'compute_statistics' with the parameters above.
        2. Based on the tool output, determine the join_type and confidence.
        3. Return ONLY the JSON schema requested. Do not ask for more information.
        """
        table_events = await table_runner.run_debug(table_prompt)
        await asyncio.sleep(config.delay_between_tables)

        jc_events = await joincol_runner.run_debug(jc_prompt)
        
        jc_json_str = "{}"
        for event in reversed(jc_events):
            if hasattr(event, 'actions') and event.actions.state_delta:
                if "join_column_choice" in event.actions.state_delta:
                    jc_json_str = event.actions.state_delta["join_column_choice"]
                    break
                    
        jc_json = extract_json(jc_json_str)


        if jc_json.get("join_type") != "no_join_found":

            # Call back to verify the join
            callback.verify(jc_json, global_join_col=join_column)
            if callback.is_valid:
                print(f"✅ Physical Verification Passed ({callback.match_rate:.2%} match)")
                final_results.append({
                    "candidate_table": cand_name,
                    "selected_columns": jc_json.get("selected_columns", []),
                    "confidence": jc_json.get("confidence", 0.0),
                    "reason": jc_json.get("reason", table_info.get("reasoning", ""))
                })

            else:
                print(f"❌ Physical Verification Failed ({callback.reason})")

# ---- Phase 3: Augment Column Selection ----
    print(f"\n📊 Starting Augment Column Selection...")
    print(f"   Target: {target_column} ({task_type})")
    
    utility_runner = InMemoryRunner(agent=build_utility_gain_agent(config=config))
    augment_results = []
    
    # Process each table that passed Phase 2
    for result in final_results:
        cand_name = result["candidate_table"]
        selected_join_cols = result["selected_columns"]  # Join columns from Phase 2
        
        print(f"\n🔍 Evaluating columns in '{cand_name}' for augmenting '{target_column}'...")
        
        # Load candidate table to get all available columns
        real_cand_name = find_dataset_dir(cand_name, BASE_DIR)
        cand_df = pd.read_csv(Path(BASE_DIR) / real_cand_name / config.data_filename, low_memory=False)
        
        # Get candidate columns to evaluate (exclude join columns)
        candidate_columns = [
            col for col in cand_df.columns 
            if col not in selected_join_cols
        ]
        
        if len(candidate_columns) == 0:
            print(f"   ⚠️  No columns available for augmentation (all are join columns)")
            continue
        
        print(f"   Checking {len(candidate_columns)} candidate columns...")
        
        column_results = []
        
        for col in candidate_columns:
            try:
                print(f"      Checking: {col}")
                
                ug_prompt = f"""
                Compute utility gain and evaluate suitability with these parameters:
                - base_table_name: "{join_table_name}"
                - candidate_table_name: "{cand_name}"
                - base_join_columns: {join_column}
                - candidate_join_columns: {selected_join_cols}
                - candidate_column: "{col}"
                - target_column: "{target_column}"
                - task_type: "{task_type}"
                - base_dir: "{BASE_DIR}"
                - sample_size: {config.sample_size}
                
                Call compute_integration_quality, compute_feature_importance, and compute_utility_gain_from_params.
                Based on IQ, FI, and Utility Gain values, determine if this column is suitable for augmentation.
                Return the JSON result with iq, fi, utility_gain, is_suitable, and reason.
                """
                
                ug_events = await utility_runner.run_debug(ug_prompt)
                
                ug_json_str = "{}"
                for event in reversed(ug_events):
                    if hasattr(event, 'actions') and event.actions.state_delta:
                        if "utility_gain_result" in event.actions.state_delta:
                            ug_json_str = event.actions.state_delta["utility_gain_result"]
                            break
                
                ug_result = extract_json(ug_json_str)
                
                # Handle string JSON
                if isinstance(ug_result, str):
                    ug_result = extract_json(ug_result)
                
                # Check if result is a dictionary before using 'in' operator
                if not isinstance(ug_result, dict):
                    print(f"         ❌ Error: Unexpected result type {type(ug_result)}: {ug_result}")
                    continue
                
                if "error" in ug_result:
                    print(f"         ❌ Error: {ug_result.get('error', 'Unknown error')}")
                    continue
                
                column_results.append({
                    "column": col,
                    "iq": ug_result.get("iq", 0.0),
                    "fi": ug_result.get("fi", 0.0),
                    "utility_gain": ug_result.get("utility_gain", 0.0),
                    "is_suitable": ug_result.get("is_suitable", False),
                    "reason": ug_result.get("reason", "")
                })
                
                status = "✓" if ug_result.get("is_suitable", False) else "✗"
                print(f"         {status} UG: {ug_result.get('utility_gain', 0.0):.4f} - {ug_result.get('reason', '')}")
                
                await asyncio.sleep(config.delay_between_columns)
                
            except Exception as e:
                print(f"         ❌ Error evaluating {col}: {e}")
                continue
        
        # Filter to only suitable columns and sort by utility_gain
        suitable_columns = [r for r in column_results if r.get("is_suitable", False)]
        suitable_columns.sort(key=lambda x: x["utility_gain"], reverse=True)
        
        augment_results.append({
            "candidate_table": cand_name,
            "join_columns": selected_join_cols,
            "all_evaluated_columns": column_results,
            "suitable_columns": suitable_columns,
            "total_evaluated": len(column_results),
            "total_suitable": len(suitable_columns)
        })
        
        print(f"   ✅ Found {len(suitable_columns)} suitable columns out of {len(column_results)} evaluated")
    
    augment_callback = AugmentValidatorCallback(
        base_table_df=join_df,
        target_column=target_column,
        task_type=task_type,
        join_columns=join_column,
        base_dir=BASE_DIR,
        config=config
    )

    # Validate each candidate table's suitable columns
    for result in augment_results:
        cand_name = result["candidate_table"]
        suitable_cols = result["suitable_columns"]
        selected_join_cols = result["join_columns"]
        
        if len(suitable_cols) == 0:
            print(f"   ⚠️  No suitable columns found in '{cand_name}'")
            continue
        
        # Extract suitable columns names
        selected_column_names = [col["column"] for col in suitable_cols]
        
        print(f"\n🔬 Validating augmentation for '{cand_name}' with {len(selected_column_names)} columns...")
        
        # Validate: merge selected columns and run task
        validation_result = augment_callback.verify(
            candidate_table_name=cand_name,
            selected_columns=selected_column_names,
            candidate_join_columns=selected_join_cols
        )
        
        # Add validation result to result
        result["validation"] = validation_result
        
        if "error" in validation_result:
            print(f"   ❌ Validation failed: {validation_result['error']}")
        else:
            baseline = validation_result.get("baseline_metric")
            augmented = validation_result.get("augmented_metric", validation_result.get("metric"))
            improvement = validation_result.get("improvement")
            improvement_pct = validation_result.get("improvement_percent")
            base_count = validation_result.get("base_features_count", 0)
            augment_count = validation_result.get("augment_features_count", 0)
            total_count = validation_result.get("total_features_count", 0)
            
            print(f"   ✅ Validation passed")
            if baseline is not None:
                print(f"      Baseline: {baseline:.4f} → Augmented: {augmented:.4f}")
                if improvement is not None:
                    sign = "+" if improvement >= 0 else ""
                    print(f"      Improvement: {sign}{improvement:.4f} ({sign}{improvement_pct:.2f}%)")
            else:
                print(f"      Augmented metric: {augmented:.4f}")
            print(f"      Features: {base_count} base + {augment_count} augment = {total_count} total")


    return {
        "join_table": join_table_name,
        "join_column": join_column,
        "target_column": target_column,
        "task_type": task_type,
        "joinable_tables": final_results,
        "augment_results": augment_results
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
    
    args = parser.parse_args()
    
    try:
        # Load config (will use default path if None)
        config = AgentPipelineConfig(args.config)
        
        # Run orchestrator with config and user_intent
        output = asyncio.run(run_orchestrator(
            config=config,
            user_intent=args.user_intent,  # Pass user_intent from command line
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