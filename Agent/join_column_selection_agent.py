from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.genai import types

from tools.join_column_tool import (
    containment_score,
    date_normalized_overlap,
    fuzzy_string_match,
    jaccard_similarity,
    normalized_overlap,
    semantic_column_similarity,
)


def load_agent_instruction(prompt_file: str = "prompt/join_column_selection_agent_prompt.txt") -> str:
    """Load instruction text for join-column agent."""
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / prompt_file
    if not prompt_path.exists():
        prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_path.read_text(encoding="utf-8")


def build_join_column_choose_agent(provider: Optional[str] = None, config: Optional[object] = None) -> Agent:
    """Build a minimal join-column agent with 6 tools."""
    if config is not None:
        provider = config.get_provider("join_column_selection")
        retry_config = config.get_retry_config()
        model_name = config.get_model_name("join_column_selection", provider)
    else:
        provider = provider or "openai"
        retry_config = types.HttpRetryOptions(
            attempts=5,
            exp_base=7,
            initial_delay=1,
            http_status_codes=[429, 500, 503, 504],
        )
        model_name = "openai/gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash-lite"

    if provider == "openai":
        llm = LiteLlm(model=model_name)
    elif provider == "gemini":
        llm = Gemini(model=model_name, retry_options=retry_config)
    else:
        kwargs = {"model": model_name}
        if provider == "local":
            kwargs["api_base"] = "http://localhost:8080/v1"
            kwargs["api_key"] = "not-needed"
        llm = LiteLlm(**kwargs)

    prompt_file = "prompt/join_column_selection_agent_prompt.txt"
    if config is not None and hasattr(config, "config"):
        prompt_file = config.config.get("agents", {}).get("prompts", {}).get("join_column_selection", prompt_file)
    instruction = load_agent_instruction(prompt_file)

    generate_content_config = None
    if config is not None and hasattr(config, "get_temperature"):
        generate_content_config = types.GenerateContentConfig(
            temperature=config.get_temperature("join_column_selection")
        )

    join_tools = [
        FunctionTool(func=jaccard_similarity),
        FunctionTool(func=containment_score),
        FunctionTool(func=normalized_overlap),
        FunctionTool(func=date_normalized_overlap),
        FunctionTool(func=fuzzy_string_match),
        FunctionTool(func=semantic_column_similarity),
    ]

    return Agent(
        name="JoinColumnChooseAgent",
        model=llm,
        instruction=instruction,
        tools=join_tools,
        output_key="join_column_choice",
        generate_content_config=generate_content_config,
    )
