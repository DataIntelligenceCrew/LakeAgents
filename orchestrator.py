import json
import re
import os
import pandas as pd
import asyncio # Required for running the async entry point
from pathlib import Path
from typing import Any, Dict, List
from google.adk.runners import InMemoryRunner
from google.genai import types
from table_selection_agent import build_table_selection_agent
from join_column_selection_agent import build_join_column_choose_agent
from callback import JoinValidatorCallback, AugmentValidatorCallback
import fasttext
from functools import partial
from llm_agent_tools import find_dataset_dir
from augment_column_selection_agent import build_utility_gain_agent

# Environment setup
for key in ["GOOGLE_API_KEY", "OPENAI_API_KEY"]:
    val = os.getenv(key)
    if val: os.environ[key] = val

BASE_DIR = "datasets_omnimatch2"

def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from model output."""
    try:
        # First try standard JSON parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON-like structure (handle Python dict with single quotes)
        # Replace single quotes with double quotes for JSON parsing
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
                        return ast.literal_eval(json_str)
            raise ValueError(f"Failed to extract JSON from: {text[:200]}")

# Change 1: Added 'async' keyword
async def run_orchestrator(
    join_table_name: str, 
    join_column: List[str],
    target_column: str,
    task_type: str
) -> Dict[str, Any]:
    table_runner = InMemoryRunner(agent=build_table_selection_agent())
    joincol_runner = InMemoryRunner(agent=build_join_column_choose_agent())

    base_path = Path(BASE_DIR)
    candidate_names = [
        item.name for item in base_path.iterdir()
        if item.is_dir() and (item / "metadata.json").exists() and item.name != join_table_name
    ]

    # ---- Phase 1: Table Selection ----
    print("🚀 Running Table Selection Agent...")
    table_prompt = f"Join Table: '{join_table_name}', Join Col: '{join_column}'. Candidates: {', '.join(candidate_names)}"
    
    # Change 2: Added 'await' before run_debug
    table_events = await table_runner.run_debug(table_prompt)
    
    table_json_str = "{}"
    for event in reversed(table_events):
        if hasattr(event, 'actions') and event.actions.state_delta:
            if "relevant_tables" in event.actions.state_delta:
                table_json_str = event.actions.state_delta["relevant_tables"]
                break
    
    table_data = extract_json(table_json_str)
    relevant_list = table_data.get("relevant_tables", [])

    real_join_table_name = find_dataset_dir(join_table_name, BASE_DIR)
    join_df = pd.read_csv(Path(BASE_DIR) / real_join_table_name / "rows.csv", low_memory=False)
    callback = JoinValidatorCallback(join_table_df=join_df, base_dir=BASE_DIR)

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
        await asyncio.sleep(5)

        jc_events = await joincol_runner.run_debug(jc_prompt)
        
        jc_json_str = "{}"
        for event in reversed(jc_events):
            if hasattr(event, 'actions') and event.actions.state_delta:
                if "join_column_choice" in event.actions.state_delta:
                    jc_json_str = event.actions.state_delta["join_column_choice"]
                    break
                    
        jc_json = extract_json(jc_json_str)


        if jc_json.get("join_type") != "no_join_found":

            #call back to verify the join
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
    
    utility_runner = InMemoryRunner(agent=build_utility_gain_agent())
    augment_results = []
    
    # Process each table that passed Phase 2
    for result in final_results:
        cand_name = result["candidate_table"]
        selected_join_cols = result["selected_columns"]  # Join columns from Phase 2
        
        print(f"\n🔍 Evaluating columns in '{cand_name}' for augmenting '{target_column}'...")
        
        # Load candidate table to get all available columns
        real_cand_name = find_dataset_dir(cand_name, BASE_DIR)
        cand_df = pd.read_csv(Path(BASE_DIR) / real_cand_name / "rows.csv", low_memory=False)
        
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
                - sample_size: 1000
                
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
                
                await asyncio.sleep(2)  # Small delay between evaluations
                
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
    base_dir=BASE_DIR
    )

    # Validate each candidate table's suitable columns
    for result in augment_results:
        cand_name = result["candidate_table"]
        suitable_cols = result["suitable_columns"]
        selected_join_cols = result["join_columns"]
        
        if len(suitable_cols) == 0:
            print(f"   ⚠️  No suitable columns found in '{cand_name}'")
            continue
        
        # extract suitable columns names
        selected_column_names = [col["column"] for col in suitable_cols]
        
        print(f"\n🔬 Validating augmentation for '{cand_name}' with {len(selected_column_names)} columns...")
        
        # validate: merge selected columns and run task
        validation_result = augment_callback.verify(
            candidate_table_name=cand_name,
            selected_columns=selected_column_names,
            candidate_join_columns=selected_join_cols
        )
        
        # add validation result to result
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
    try:
        output = asyncio.run(run_orchestrator(
            join_table_name="join table",
            join_column=["PID", "Date Reported As Of"],
            target_column="Budget Forecast",
            task_type="regression"
        ))
        print("\n--- Final Results ---")
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(f"Workflow failed: {e}")