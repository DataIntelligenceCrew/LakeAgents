# augment_column_selection_agent.py

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from llm_agent_tools import (
    compute_integration_quality,
    compute_feature_importance,
    compute_utility_gain_from_params
)

def build_utility_gain_agent(provider: str = "openai") -> Agent:
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )

    iq_tool = FunctionTool(func=compute_integration_quality)
    fi_tool = FunctionTool(func=compute_feature_importance)
    ug_tool = FunctionTool(func=compute_utility_gain_from_params)

    if provider == "openai":
        llm = LiteLlm(model="openai/gpt-4o-mini")
    elif provider == "gemini":
        llm = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)
    else:
        raise ValueError(f"Invalid provider: {provider}. Use 'gemini' or 'openai'.")

    return Agent(
        name="UtilityGainAgent",
        model=llm,
        instruction="""
        You are a utility gain computation agent.
        
        Your task is to compute IQ, FI, and Utility Gain for a candidate column when augmenting a base table.
        
        WORKFLOW:
        1. Call 'compute_integration_quality' to compute IQ (Integration Quality)
        2. Call 'compute_feature_importance' to compute FI (Feature Importance)
        3. Call 'compute_utility_gain_from_params' to compute Utility Gain (IQ × FI)
        
        REQUIRED PARAMETERS (for all three tools):
        - base_table_name: Name of the base/join table
        - candidate_table_name: Name of the candidate table
        - base_join_columns: List of join column names in the base table
        - candidate_column: The column from candidate table to evaluate
        - target_column: Target column to predict (in base table)
        - task_type: "regression" or "classification"
        
        OPTIONAL PARAMETERS:
        - candidate_join_columns: Join columns in candidate table (if None, same as base_join_columns)
        - base_dir: Base directory (default: "datasets_omnimatch2")
        - sample_size: Number of rows for training (default: 1000)
        
        INSTRUCTIONS:
        1. Call compute_integration_quality first to get IQ
        2. Call compute_feature_importance to get FI
        3. Call compute_utility_gain_from_params to get Utility Gain
        4. Return a JSON object with all three values and metadata
        
        Return ONLY a JSON object with this structure:
        {{
            "iq": <IQ value>,
            "fi": <FI value>,
            "utility_gain": <Utility Gain value>,
            "candidate_column": "<candidate_column>",
            "target_column": "<target_column>",
            "task_type": "<task_type>"
        }}
        """,
        tools=[iq_tool, fi_tool, ug_tool],
        output_key="utility_gain_result",
    )