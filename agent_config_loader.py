"""
Configuration loader for the multi-agent data augmentation pipeline.
"""
import json
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from google.genai import types

_PROJECT_ROOT = Path(__file__).resolve().parent


def _apply_replacements_to_text(text: str, replacements: dict) -> str:
    """Apply synonym replacements (word boundary, case-insensitive)."""
    if not text or not replacements:
        return str(text) if text is not None else ""
    s = str(text)
    for old, new in replacements.items():
        s = re.sub(r"\b" + re.escape(old) + r"\b", new, s, flags=re.IGNORECASE)
    return s


def apply_replacements_to_task_config(config: dict, replacements: dict) -> dict:
    """Return a new config with task.join_column and task.target_column replaced."""
    if not replacements or "task" not in config:
        return {**config, "task": {**config.get("task", {})}}
    out = {k: v for k, v in config.items()}
    out["task"] = {**config["task"]}
    t = out["task"]
    if t.get("join_column"):
        jc = t["join_column"]
        lst = jc if isinstance(jc, list) else [jc]
        new_lst = [_apply_replacements_to_text(c, replacements) for c in lst]
        t["join_column"] = new_lst if isinstance(jc, list) else new_lst[0]
    if t.get("target_column"):
        t["target_column"] = _apply_replacements_to_text(t["target_column"], replacements)
    return out


def load_replacements_for_table(table_folder: str, project_root: Optional[Path] = None) -> dict:
    """Load replacements from jaccard_perturbed/{table_folder}_perturbed.json."""
    root = project_root or _PROJECT_ROOT
    path = root / "benchmark_perturbation" / "jaccard_perturbed" / f"{table_folder}_perturbed.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("replacements", {})
    except Exception:
        return {}

def get_temperature(config: Dict[str, Any], agent_name: str) -> float:
    t = config.get('agents', {}).get('temperature', {})
    default = t.get('default', 0.7)
    overrides = t.get('overrides') or {}
    val = overrides.get(agent_name)
    return (float(val) if val is not None else default)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default path.
    
    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "agent_pipeline_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_retry_config(config: Dict[str, Any]) -> types.HttpRetryOptions:
    """
    Create HttpRetryOptions from config.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        HttpRetryOptions object.
    """
    retry_cfg = config['agents']['retry']
    return types.HttpRetryOptions(
        attempts=retry_cfg['attempts'],
        exp_base=retry_cfg['exp_base'],
        initial_delay=retry_cfg['initial_delay'],
        http_status_codes=retry_cfg['http_status_codes']
    )


def get_model_name(config: Dict[str, Any], agent_name: str, provider: Optional[str] = None) -> str:
    """
    Get model name for a specific agent.
    
    Args:
        config: Configuration dictionary.
        agent_name: Name of the agent ('table_selection', 'join_column_selection', 'utility_gain').
        provider: Override provider. If None, uses config default or per-agent override.
    
    Returns:
        Model name string.
    """
    if provider is None:
        # Check for per-agent override
        override = config['agents']['provider_overrides'].get(agent_name)
        if override is not None:
            provider = override
        else:
            provider = config['agents']['default_provider']
    
    return config['agents']['models'][provider][agent_name]


def get_provider(config: Dict[str, Any], agent_name: str) -> str:
    """
    Get provider for a specific agent.
    
    Args:
        config: Configuration dictionary.
        agent_name: Name of the agent.
    
    Returns:
        Provider name ('gemini' or 'openai').
    """
    override = config['agents']['provider_overrides'].get(agent_name)
    if override is not None:
        return override
    return config['agents']['default_provider']


class AgentPipelineConfig:
    """
    Configuration wrapper class for easy access to config values.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is not None:
            self.config = config_dict
        else:
            self.config = load_config(config_path)
    
    @property
    def base_dir(self) -> str:
        """Base directory for datasets."""
        return self.config['data']['base_dir']
    
    @property
    def data_filename(self) -> str:
        """Standard filename for data files."""
        return self.config['data']['data_filename']
    
    @property
    def join_table_name(self) -> str:
        """Name of the base/join table."""
        return self.config['task']['join_table_name']
    
    @property
    def join_column(self) -> list:
        """Join columns (list)."""
        join_col = self.config['task']['join_column']
        return join_col if isinstance(join_col, list) else [join_col]
    
    @property
    def target_column(self) -> str:
        """Target column for prediction."""
        return self.config['task']['target_column']
    
    @property
    def task_type(self) -> str:
        """Task type: 'regression' or 'classification'."""
        return self.config['task']['task_type']
    
    @property
    def session_id(self) -> Optional[str]:
        """Session ID for per-session checked dataset."""
        return self.config.get('task', {}).get('session', {}).get('session_id')
    
    @property
    def match_rate_threshold(self) -> float:
        """Minimum match rate for join validation."""
        return self.config['validation']['match_rate_threshold']
    
    @property
    def max_data_explosion_factor(self) -> float:
        """Maximum allowed rows after join."""
        return self.config['validation']['max_data_explosion_factor']
    
    @property
    def sample_size(self) -> int:
        """Number of rows to sample for ML training."""
        return self.config['sampling']['sample_size']
    
    @property
    def default_provider(self) -> str:
        """Default LLM provider."""
        return self.config['agents']['default_provider']
    
    def get_model_name(self, agent_name: str, provider: Optional[str] = None) -> str:
        """Get model name for an agent."""
        return get_model_name(self.config, agent_name, provider)
    
    def get_provider(self, agent_name: str) -> str:
        """Get provider for an agent."""
        return get_provider(self.config, agent_name)
    
    def get_retry_config(self) -> types.HttpRetryOptions:
        """Get HTTP retry configuration."""
        return get_retry_config(self.config)
    
    def get_temperature(self, agent_name: str) -> float:
        """Get temperature for an agent."""
        return get_temperature(self.config, agent_name)

    @property
    def delay_between_tables(self) -> int:
        """Delay between table evaluations (seconds)."""
        return self.config['delays']['between_table_evaluations']
    
    @property
    def delay_between_columns(self) -> int:
        """Delay between column evaluations (seconds)."""
        return self.config['delays']['between_column_evaluations']
    
    @property
    def ml_model_config(self) -> Dict[str, Any]:
        """ML model configuration for current task type."""
        return self.config['ml_models'][self.task_type]
    
    @property
    def verbose(self) -> bool:
        """Whether to print detailed progress."""
        return self.config['output']['verbose']
    
    @property
    def save_results(self) -> bool:
        """Whether to save results to file."""
        return self.config['output']['save_results']
    
    @property
    def results_file(self) -> str:
        """Results file name."""
        return self.config['output']['results_file']
    
    @property
    def print_results(self) -> bool:
        """Whether to print results to console."""
        return self.config['output']['print_results']

    