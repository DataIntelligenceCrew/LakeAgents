"""Decision agent for collaborative table selection."""
import os
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


def _load_instruction(
    prompt_file: str = "prompt/table_selection_decider_agent_prompt.txt",
) -> str:
    """Load instruction text from file, with a simple fallback."""
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / prompt_file
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are the decision agent for collaborative table selection.\n"
        "Apply these rules in order:\n"
        "1) Intersection-first: tables selected by recall and precision are highest priority.\n"
        "2) Union supplement: recall-only tables are optional and should be marked high-risk.\n"
        "3) Precision risk: risk marks are signals only and should not be default exclusion.\n"
        "Return strict JSON only:\n"
        '{ "final_tables": [{"table_id": "...", "risk": "low|high", "reason": "..."}] }'
    )


def build_table_selection_decider_agent(
    provider: Optional[str] = None,
    config: Optional[object] = None,
) -> Agent:
    """Build collaborative decision agent."""
    if config is not None:
        provider = config.get_provider("table_selection")
        retry_config = config.get_retry_config()
        model_name = config.get_model_name("table_selection", provider)
    else:
        provider = provider or "openai"
        retry_config = types.HttpRetryOptions(
            attempts=3,
            exp_base=2,
            initial_delay=1,
            http_status_codes=[429, 500, 503, 504],
        )
        model_name = "openai/gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash"

    if provider == "openai":
        llm = LiteLlm(model=model_name)
    elif provider == "gemini":
        llm = Gemini(model=model_name, retry_options=retry_config)
    else:
        kw = {"model": model_name}
        if provider == "local":
            kw["api_base"] = (
                os.environ.get("OPENAI_API_BASE")
                or f"http://localhost:{os.environ.get('VLLM_PORT', '8080')}/v1"
            )
            kw["api_key"] = "not-needed"
        llm = LiteLlm(**kw)

    generate_content_config = None
    if config is not None and hasattr(config, "get_temperature"):
        temperature = config.get_temperature("table_selection")
        generate_content_config = types.GenerateContentConfig(temperature=temperature)

    return Agent(
        name="TableSelectionDeciderAgent",
        model=llm,
        instruction=_load_instruction(),
        output_key="final_tables",
        generate_content_config=generate_content_config,
    )
