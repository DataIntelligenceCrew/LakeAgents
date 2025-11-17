#!/usr/bin/env python3
"""
Logger Utilities
Setup logging for experiments with Aim integration
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("Warning: Aim not installed. Install with: pip install aim")


class ExperimentLogger:
    """Logger for experiments with Aim integration"""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "experiments",
        config: Optional[Dict[str, Any]] = None,
        use_aim: bool = True
    ):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
            config: Configuration dictionary (will be logged)
            use_aim: Whether to use Aim for tracking
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(log_dir) / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.setup_file_logging()
        
        # Setup Aim if available and enabled
        self.aim_run = None
        if use_aim and AIM_AVAILABLE:
            self.setup_aim(config)
        
        # Save config
        if config:
            self.save_config(config)
        
        self.logger.info(f"Experiment initialized: {experiment_name}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def setup_file_logging(self):
        """Setup logging to file and console"""
        # Create logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler
        log_file = self.experiment_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_aim(self, config: Optional[Dict[str, Any]] = None):
        """
        Setup Aim tracking
        
        Args:
            config: Configuration to log
        """
        try:
            # Get Aim config from training config
            aim_config = {}
            if config and 'training' in config:
                aim_config = config['training'].get('logging', {}).get('aim', {})
            
            if not aim_config.get('enabled', True):
                self.logger.info("Aim tracking disabled in config")
                return
            
            repo_path = aim_config.get('repo_path', './aim_logs')
            
            # Initialize Aim Run
            self.aim_run = Run(
                repo=repo_path,
                experiment=self.experiment_name,
                log_system_params=aim_config.get('log_system_params', True)
            )
            
            # Log configuration as hyperparameters
            if config:
                self.log_hparams(config)
            
            self.logger.info(f"Aim tracking initialized at: {repo_path}")
            self.logger.info(f"Run hash: {self.aim_run.hash}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Aim: {e}")
            self.aim_run = None
    
    def log_hparams(self, config: Dict[str, Any]):
        """
        Log hyperparameters to Aim
        
        Args:
            config: Configuration dictionary
        """
        if self.aim_run is None:
            return
        
        try:
            # Flatten config for logging
            hparams = self._flatten_dict(config)
            self.aim_run['hparams'] = hparams
            self.logger.info(f"Logged {len(hparams)} hyperparameters to Aim")
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log a metric value
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Training epoch
            context: Additional context (e.g., {'subset': 'train', 'layer': 2})
        """
        # Log to file
        context_str = f" {context}" if context else ""
        self.logger.info(f"Metric: {name}={value:.6f} (step={step}, epoch={epoch}){context_str}")
        
        # Log to Aim
        if self.aim_run is not None:
            try:
                self.aim_run.track(
                    value,
                    name=name,
                    step=step,
                    epoch=epoch,
                    context=context or {}
                )
            except Exception as e:
                self.logger.warning(f"Failed to log metric to Aim: {e}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log multiple metrics at once
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step
            epoch: Training epoch
            context: Additional context
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step, epoch, context)
    
    def save_config(self, config: Dict[str, Any]):
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Config saved to: {config_file}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def close(self):
        """Close logger and Aim run"""
        if self.aim_run is not None:
            try:
                self.aim_run.close()
                self.logger.info("Aim run closed")
            except Exception as e:
                self.logger.warning(f"Failed to close Aim run: {e}")
        
        # Close file handlers
        for handler in self.logger.handlers:
            handler.close()
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten nested dictionary
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def get_logger(name: str = "experiment", log_level: int = logging.INFO) -> logging.Logger:
    """
    Get a simple logger
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Only add handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


if __name__ == "__main__":
    # Test logger
    print("Testing ExperimentLogger...")
    
    # Test config
    test_config = {
        'data': {'layer3': {'positive_threshold': 0.01}},
        'model': {'layer1': {'embedding_dim': 128}},
        'training': {
            'batch_size': 16,
            'logging': {
                'aim': {
                    'enabled': True,
                    'repo_path': './test_aim_logs'
                }
            }
        }
    }
    
    # Create logger
    with ExperimentLogger(
        experiment_name="test_experiment",
        log_dir="test_logs",
        config=test_config,
        use_aim=AIM_AVAILABLE
    ) as logger:
        logger.info("Test info message")
        logger.warning("Test warning message")
        
        # Log some metrics
        logger.log_metric('loss', 0.5, step=1, epoch=1, context={'subset': 'train'})
        logger.log_metrics(
            {'accuracy': 0.85, 'f1': 0.82},
            step=1,
            epoch=1,
            context={'subset': 'val'}
        )
    
    print("\n✓ ExperimentLogger test completed!")
    print(f"  Check test_logs/ for log files")
    if AIM_AVAILABLE:
        print(f"  Check test_aim_logs/ for Aim data")
        print(f"  Run 'aim up' to view in browser")

