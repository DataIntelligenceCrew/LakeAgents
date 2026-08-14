"""Precision-focused table selection agent."""
import os
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


def _load_instruction(
    prompt_file: str = "prompt/table_selection_precision_agent_prompt.txt",
) -> str:
    """Load instruction text from file, with a simple fallback."""
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / prompt_file
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")

    return (
        "You are a precision-focused table selector.\n"
        "Select only high-confidence tables with likely joinability.\n"
        "If a table is uncertain, add it to high_risk_tables.\n"
        "Return strict JSON only:\n"
        '{ "relevant_tables": [{"table_id": "...", "reason": "..."}], '
        '"high_risk_tables": [{"table_id": "...", "reason": "..."}] }'
    )


def build_table_selection_precision_agent(
    provider: Optional[str] = None,
    config: Optional[object] = None,
) -> Agent:
    """Build precision-focused table selection agent."""
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
        name="TableSelectionPrecisionAgent",
        model=llm,
        instruction=_load_instruction(),
        output_key="relevant_tables_precision",
        generate_content_config=generate_content_config,
    )
