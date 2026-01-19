# agents/join_column_choose_agent.py
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm 
from google.genai import types
from llm_agent_tools import compute_statistics

def build_join_column_choose_agent(provider: str = "openai") -> Agent:
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )

    compute_statistics_tool = FunctionTool(func=compute_statistics)

    if provider == "openai":
        llm = LiteLlm(model="openai/gpt-4o-mini")
    elif provider == "gemini":
        llm = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)
    else:
        raise ValueError(f"Invalid provider: {provider}. Use 'gemini' or 'openai'.")

    return Agent(
        name="JoinColumnChooseAgent",
        model=llm,
        instruction="""
        You are given ONE candidate table and ONE join table.
        Based on the statistics returned by compute_statistics,
        decide whether this candidate table is joinable.
        If yes, choose the best join column(s). Always keep 'Capital Projects' table currently.

        CRITICAL RULE:
        - The join_table specifies join_column(s) - this tells you how many columns to use for joining.
        - If join_column is a list with N columns, you MUST select EXACTLY N columns from the candidate table.
        - The number of selected columns MUST match the number of join columns from the join table.

        You MUST call compute_statistics first.

        Return ONLY JSON with this schema:
        {
        "candidate_table_name": "name",
        "join_type": "single" | "composite" | "no_join_found",
        "selected_columns": ["colA", ...],  // Must match the number of join columns from join table
        "confidence": 0.0,
        "reason": "short numeric reasoning",
        }
        """,
        tools=[compute_statistics_tool],
        output_key="join_column_choice",
    )
