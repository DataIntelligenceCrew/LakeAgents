# augment_column_selection_agent.py 

from typing import Optional
from pathlib import Path
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


def load_instruction(prompt_file: str) -> str:
    root = Path(__file__).resolve().parent
    path = root / prompt_file
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def build_augment_column_selection_agent(config=None) -> Agent:
    """Build agent that selects augment columns using correlation + descriptions."""
    provider = config.get_provider("utility_gain") if config else "openai"
    model_name = config.get_model_name("utility_gain", provider) if config else "openai/gpt-4o-mini"
    
    prompt_file = config.config.get("agents", {}).get("prompts", {}).get("augment_column_selection", "prompt/augment_column_selection_agent_prompt.txt") if config else "prompt/augment_column_selection_agent_prompt.txt"
    instruction = load_instruction(prompt_file)
    
    if provider == "openai":
        llm = LiteLlm(model=model_name)
    else:
        llm = Gemini(model=model_name, retry_options=config.get_retry_config())
    
    return Agent(
        name="AugmentColumnSelectionAgent",
        model=llm,
        instruction=instruction,
        output_key="augment_selection_result",
        generate_content_config=types.GenerateContentConfig(temperature=0.3),
    )