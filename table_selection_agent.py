# agents/table_selection_agent.py
import os
from typing import Optional
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from llm_agent_tools import read_metadata, analyze_user_intent, confirm_table_category,confirm_dimension_requirement, read_table_index
from pathlib import Path

def load_agent_instruction(prompt_file: str = "prompt/table_selection_agent_prompt.txt") -> str:
    """
    Load agent instruction from a prompt file.
    
    Args:
        prompt_file: Path to the prompt file (relative to project root or absolute)
        
    Returns:
        The instruction text
    """
    # Try relative path first (from project root)
    project_root = Path(__file__).parent.parent
    prompt_path = project_root / prompt_file
    
    # If not found, try absolute path
    if not prompt_path.exists():
        prompt_path = Path(prompt_file)
    
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}. "
            f"Tried: {project_root / prompt_file} and {prompt_path}"
        )
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def build_table_selection_agent(provider: Optional[str] = None, config: Optional[object] = None) -> Agent:
    """
    Build table selection agent.
    
    Args:
        provider: LLM provider ("gemini" or "openai"). If None, uses config or default.
        config: AgentPipelineConfig object. If provided, overrides provider parameter.
    
    Returns:
        Configured Agent instance.
    """
    # Use config if provided
    if config is not None:
        provider = config.get_provider("table_selection")
        retry_config = config.get_retry_config()
        model_name = config.get_model_name("table_selection", provider)
        join_table_name = config.join_table_name  # Get join_table_name from config
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
        model_name = "openai/gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash"
        join_table_name = None  # Will be set later if needed

    # Create a wrapper function that hardcodes the exclusion of join_table
    def read_metadata_with_exclusion(dataset_name: str = None, base_dir: str = None) -> dict[str, any]:
        exclude_tables = [join_table_name] if join_table_name else []
        return read_metadata(dataset_name=dataset_name, base_dir=base_dir, exclude_tables=exclude_tables)
    
    # read_metadata_tool = FunctionTool(func=read_metadata_with_exclusion)

    def _read_table_index_with_truncation(candidate_ids, index_path=None):
        """Wrapper that applies config-based truncation when enabled."""
        trunc_cfg = {}
        if config is not None and hasattr(config, "config"):
            trunc_cfg = config.config.get("agents", {}).get("prompt_truncation", {})
        if not trunc_cfg.get("enabled", False):
            return read_table_index(candidate_ids, index_path)
        max_entries = trunc_cfg.get("max_index_entries", 25)
        max_chars = trunc_cfg.get("max_chars_per_index_field", 500)
        max_cols = trunc_cfg.get("max_columns_per_entry")
        return read_table_index(
            candidate_ids, index_path,
            max_entries=max_entries, max_chars_per_field=max_chars,
            max_list_items=max_cols,
        )

    read_table_index_tool = FunctionTool(func=_read_table_index_with_truncation)
    # analyze_intent_tool = FunctionTool(func=analyze_user_intent)
    # confirm_dimension_tool = FunctionTool(func=confirm_dimension_requirement)
    # confirm_category_tool = FunctionTool(func=confirm_table_category)

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
        
    prompt_file = "prompt/table_selection_agent_prompt.txt"

    # Load instruction from file
    if config is not None and hasattr(config, 'config'):
        prompt_file = config.config.get('agents', {}).get('prompt', {}).get('table_selection', prompt_file)
    
    instruction = load_agent_instruction(prompt_file)
    
    generate_content_config = None
    if config is not None and hasattr(config, 'get_temperature'):
        temperature = config.get_temperature("table_selection")
        generate_content_config = types.GenerateContentConfig(temperature=temperature)
    
    return Agent(
        name="TableSelectionAgent",
        model=llm,
        instruction=instruction,
        tools=[read_table_index_tool],
        output_key="relevant_tables",
        generate_content_config=generate_content_config,
    )
