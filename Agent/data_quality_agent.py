import os
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.genai import types

from Agent.dq_context_compact import dq_compact_before_model, dq_truncate_after_tool
from tools.data_quality import (
    bayesian_ridge_impute_preview,
    column_quality_metrics,
    impute_median,
    impute_mode,
    random_forest_impute_preview,
    winsorize_values,
)


def load_agent_instruction(prompt_file: str = "prompt/data_quality_agent_prompt.txt") -> str:
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / prompt_file
    if not prompt_path.exists():
        prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_path.read_text(encoding="utf-8")


def build_data_quality_agent(provider: Optional[str] = None, config: Optional[object] = None) -> Agent:
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
            kwargs["api_base"] = (
                os.environ.get("OPENAI_API_BASE")
                or f"http://localhost:{os.environ.get('VLLM_PORT', '8080')}/v1"
            )
            kwargs["api_key"] = "not-needed"
        llm = LiteLlm(**kwargs)

    prompt_file = "prompt/data_quality_agent_prompt.txt"
    if config is not None and hasattr(config, "config"):
        prompt_file = config.config.get("agents", {}).get("prompts", {}).get("data_quality", prompt_file)
    instruction = load_agent_instruction(prompt_file)

    generate_content_config = None
    if config is not None and hasattr(config, "get_temperature"):
        generate_content_config = types.GenerateContentConfig(
            temperature=config.get_temperature("utility_gain")
        )

    dq_tools = [
        FunctionTool(func=column_quality_metrics),
        FunctionTool(func=winsorize_values),
        FunctionTool(func=impute_median),
        FunctionTool(func=impute_mode),
        FunctionTool(func=bayesian_ridge_impute_preview),
        FunctionTool(func=random_forest_impute_preview),
    ]

    return Agent(
        name="DataQualityAgent",
        model=llm,
        instruction=instruction,
        tools=dq_tools,
        output_key="data_quality_decision",
        generate_content_config=generate_content_config,
        # Option-2: after tool rounds, feed the model a compact fresh prompt
        # (task summary + tool traces) instead of full multi-turn history.
        before_model_callback=dq_compact_before_model,
        after_tool_callback=dq_truncate_after_tool,
    )
