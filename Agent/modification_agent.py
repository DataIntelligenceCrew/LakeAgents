from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


def load_agent_instruction(prompt_file: str = "prompt/modification_agent_prompt.txt") -> str:
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / prompt_file
    if not prompt_path.exists():
        prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_path.read_text(encoding="utf-8")


def build_modification_agent(provider: Optional[str] = None, config: Optional[object] = None) -> Agent:
    if config is not None:
        provider = config.get_provider("utility_gain")
        retry_config = config.get_retry_config()
        model_name = config.get_model_name("utility_gain", provider)
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

    prompt_file = "prompt/modification_agent_prompt.txt"
    if config is not None and hasattr(config, "config"):
        prompt_file = config.config.get("agents", {}).get("prompts", {}).get("modification", prompt_file)
    instruction = load_agent_instruction(prompt_file)

    generate_content_config = None
    if config is not None and hasattr(config, "get_temperature"):
        generate_content_config = types.GenerateContentConfig(
            temperature=config.get_temperature("utility_gain")
        )

    return Agent(
        name="ModificationAgent",
        model=llm,
        instruction=instruction,
        output_key="modification_result",
        generate_content_config=generate_content_config,
    )

