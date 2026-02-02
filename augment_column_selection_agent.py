# augment_column_selection_agent.py

from typing import Optional
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

def build_utility_gain_agent(provider: Optional[str] = None, config: Optional[object] = None) -> Agent:
    """
    Build utility gain computation agent.
    
    Args:
        provider: LLM provider ("gemini" or "openai"). If None, uses config or default.
        config: AgentPipelineConfig object. If provided, overrides provider parameter.
    
    Returns:
        Configured Agent instance.
    """
    # Use config if provided
    if config is not None:
        provider = config.get_provider("utility_gain")
        retry_config = config.get_retry_config()
        model_name = config.get_model_name("utility_gain", provider)
    else:
        # Fallback to defaults
        if provider is None:
            provider = "openai"
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )
    
    model_name = "openai/gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash-lite"

    iq_tool = FunctionTool(func=compute_integration_quality)
    fi_tool = FunctionTool(func=compute_feature_importance)
    ug_tool = FunctionTool(func=compute_utility_gain_from_params)

    if provider == "openai":
        llm = LiteLlm(model=model_name)
    elif provider == "gemini":
        llm = Gemini(model=model_name, retry_options=retry_config)
    else:
        raise ValueError(f"Invalid provider: {provider}. Use 'gemini' or 'openai'.")

    generate_content_config = None
    if config is not None and hasattr(config, 'get_temperature'):
        temperature = config.get_temperature("utility_gain")
        generate_content_config = types.GenerateContentConfig(temperature=temperature)

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
        - base_dir: Base directory (will use config default if not specified)
        - sample_size: Number of rows for training (default: 1000)
        - opendata_domain: If provided (e.g. "data.cityofnewyork.us"), candidate table data will be fetched from API instead of local files. Pass this to compute_integration_quality and compute_feature_importance when given.
        
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
        generate_content_config=generate_content_config,
    )