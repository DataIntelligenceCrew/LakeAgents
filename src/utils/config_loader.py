#!/usr/bin/env python3
"""
Configuration Loader
Loads and validates YAML configuration files
"""

import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigLoader:
    """Load and merge multiple YAML configuration files"""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize ConfigLoader
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a single config file
        
        Args:
            config_name: Name of config file (without .yaml extension)
            
        Returns:
            Dictionary containing configuration
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config if config is not None else {}
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files
        
        Returns:
            Dictionary with keys: 'data', 'model', 'training'
        """
        config_files = ['data_config', 'model_config', 'training_config']
        configs = {}
        
        for config_file in config_files:
            try:
                # Remove '_config' suffix for key name
                key = config_file.replace('_config', '')
                configs[key] = self.load_config(config_file)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                configs[key] = {}
        
        return configs
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple config dictionaries
        Later configs override earlier ones
        
        Args:
            *configs: Variable number of config dictionaries
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Recursively merge two dictionaries
        
        Args:
            base: Base dictionary
            update: Dictionary to merge into base
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, config: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that config contains required keys
        
        Args:
            config: Configuration dictionary
            required_keys: List of required key paths (e.g., 'data.layer2.positive_threshold')
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        for key_path in required_keys:
            keys = key_path.split('.')
            value = config
            
            for key in keys:
                if key not in value:
                    raise ValueError(f"Missing required config key: {key_path}")
                value = value[key]
        
        return True
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def get_nested_value(config: Dict, key_path: str, default: Any = None) -> Any:
        """
        Get value from nested dictionary using dot notation
        
        Args:
            config: Configuration dictionary
            key_path: Path to value (e.g., 'layer1.embedding_dim')
            default: Default value if key not found
            
        Returns:
            Value at key_path or default
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


def load_configs(config_dir: str = "configs") -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to load all configs
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Dictionary with keys: 'data', 'model', 'training'
    """
    loader = ConfigLoader(config_dir)
    return loader.load_all_configs()


if __name__ == "__main__":
    # Test configuration loading
    print("Testing ConfigLoader...")
    
    try:
        loader = ConfigLoader()
        configs = loader.load_all_configs()
        
        print("\nLoaded configs:")
        for name, config in configs.items():
            print(f"  - {name}: {len(config)} top-level keys")
        
        # Test nested value access
        print("\nTesting nested value access:")
        data_config = configs['data']
        threshold = ConfigLoader.get_nested_value(data_config, 'layer3.positive_threshold', default=0.01)
        print(f"  layer3.positive_threshold = {threshold}")
        
        embedding_dim = ConfigLoader.get_nested_value(configs['model'], 'layer1.embedding_dim', default=128)
        print(f"  layer1.embedding_dim = {embedding_dim}")
        
        print("\n✓ ConfigLoader test passed!")
        
    except Exception as e:
        print(f"\n✗ ConfigLoader test failed: {e}")

