# agents/join_column_choose_agent.py
from typing import Optional
from pathlib import Path
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm 
from google.genai import types
from llm_agent_tools import compute_statistics

def load_agent_instruction(prompt_file: str = "prompt/join_column_selection_agent_prompt.txt") -> str:
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

def build_join_column_choose_agent(provider: Optional[str] = None, config: Optional[object] = None) -> Agent:
    """
    Build join column selection agent.
    
    Args:
        provider: LLM provider ("gemini" or "openai"). If None, uses config or default.
        config: AgentPipelineConfig object. If provided, overrides provider parameter.
    
    Returns:
        Configured Agent instance.
    """
    # Use config if provided
    if config is not None:
        provider = config.get_provider("join_column_selection")
        retry_config = config.get_retry_config()
        model_name = config.get_model_name("join_column_selection", provider)
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
    model_name = "openai/gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash-lite"

    compute_statistics_tool = FunctionTool(func=compute_statistics)

    if provider == "openai":
        llm = LiteLlm(model=model_name)
    elif provider == "gemini":
        llm = Gemini(model=model_name, retry_options=retry_config)
    else:
        raise ValueError(f"Invalid provider: {provider}. Use 'gemini' or 'openai'.")

    prompt_file = "prompt/join_column_selection_agent_prompt.txt"
    
    # Load instruction from file
    if config is not None and hasattr(config, 'config'):
        prompt_file = config.config.get('agents', {}).get('prompts', {}).get('join_column_selection', prompt_file)
    
    instruction = load_agent_instruction(prompt_file)

    return Agent(
        name="JoinColumnChooseAgent",
        model=llm,
        instruction=instruction,
        tools=[compute_statistics_tool],
        output_key="join_column_choice",
    )
