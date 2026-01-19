# agents/table_selection_agent.py
import os
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from llm_agent_tools import read_metadata


def build_table_selection_agent(provider: str = "openai") -> Agent:
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504],
    )

    read_metadata_tool = FunctionTool(func=read_metadata)

    if provider == "openai":
        llm = LiteLlm(model="openai/gpt-4o-mini")
    elif provider == "gemini":
        llm = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)
    else:
        raise ValueError(f"Invalid provider: {provider}. Use 'gemini' or 'openai'.")
        
    return Agent(
        name="TableSelectionAgent",
        model=llm,
        instruction="""You are a data engineer agent. 
        Your task is to identify relevant tables from a provided candidate list and return ONLY a valid JSON object.

        CRITICAL RULES FOR TABLE NAMES:
        - I will provide candidate names wrapped in double quotes, e.g., "Civil list", "Agency spending".
        - You MUST use the EXACT string inside the quotes.
        - NEVER append numbers like "1", "2" or indices to the name. "Civil list 1" is WRONG. "Civil list" is CORRECT.
        - Treat the name as a literal folder path.

        Workflow:
        1. Use 'read_metadata' to inspect candidates.
        2. Analyze semantic relevance to the join table and column. Currently, always keep 'Capital Projects' table.

        OUTPUT FORMAT - CRITICAL:
        You MUST return ONLY a valid JSON object. Do NOT include any markdown, explanations, bullet points, or additional text.
        The JSON must start with { and end with }.

        Your response must be EXACTLY in this format (copy this structure):
        {
        "relevant_tables": [
            {"table_name": "Capital Projects", "reasoning": "Contains project information", "confidence": 0.8},
            {"table_name": "Civil_list", "reasoning": "Employee information", "confidence": 0.7}
        ]
        }

        DO NOT use markdown code blocks. DO NOT write explanatory text. DO NOT use bullet points.
        Return ONLY the raw JSON object.
        """,
        tools=[read_metadata_tool],
        output_key="relevant_tables",
    )
