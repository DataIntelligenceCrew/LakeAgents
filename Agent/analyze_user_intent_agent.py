# analyze_user_intent_agent.py
"""Agent that only calls analyze_user_intent to extract dimensions from user intent."""
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.models.lite_llm import LiteLlm

from tools.llm_agent_tools import confirm_dimension_requirement


def _load_instruction(prompt_file: str = "prompt/analyze_user_intent_agent_prompt.txt") -> str:
    project_root = Path(__file__).parent
    prompt_path = project_root / prompt_file
    if not prompt_path.exists():
        prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_analyze_user_intent_agent(
    provider: Optional[str] = None,
    config: Optional[object] = None,
) -> Agent:
    """
    Build an agent that only calls analyze_user_intent.
    Uses table_selection provider/model from config if provided.
    """

    provider = config.get_provider("table_selection")
    retry_config = config.get_retry_config()
    model_name = config.get_model_name("table_selection", provider)

    # tool = FunctionTool(func=confirm_dimension_requirement)
    instruction = _load_instruction()

    if provider == "openai":
        llm = LiteLlm(model=model_name)
    elif provider == "gemini":
        llm = Gemini(model=model_name, retry_options=retry_config)
    else:

        kw = {"model": model_name}
        if provider == "local":
            kw["api_base"] = "http://localhost:8080/v1"
            kw["api_key"] = "not-needed"
        llm = LiteLlm(**kw)

    generate_content_config = None
    if config is not None and hasattr(config, "get_temperature"):
        temperature = config.get_temperature("table_selection")
        generate_content_config = types.GenerateContentConfig(temperature=temperature)

    return Agent(
        name="AnalyzeUserIntentAgent",
        model=llm,
        instruction=instruction,
        # tools=[tool],
        generate_content_config=generate_content_config,
    )