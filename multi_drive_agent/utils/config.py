import yaml
import os
from typing import Dict, Any, Optional
from copy import deepcopy


class Config:
    """
    Configuration management for multi-drive agent experiments.

    Supports:
    - YAML configuration files
    - Dictionary-based configuration
    - Hierarchical config access with dot notation
    - Config merging and overrides
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary (optional)
        """
        self._config = config_dict or {}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(deepcopy(config_dict))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation, e.g., 'agent.learning_rate')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value with dot notation support.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates (supports nested dicts)
        """
        self._deep_update(self._config, updates)

    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary (deep copy)
        """
        return deepcopy(self._config)

    def save(self, yaml_path: str):
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment."""
        self.set(key, value)

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration with optional overrides.

    Args:
        config_path: Path to YAML config file (uses default if None)
        overrides: Dictionary of configuration overrides

    Returns:
        Config instance
    """
    if config_path is None:
        # Use default config
        default_config_path = os.path.join(
            os.path.dirname(__file__),
            '../../configs/default_config.yaml'
        )
        config = Config.from_yaml(default_config_path)
    else:
        config = Config.from_yaml(config_path)

    # Apply overrides
    if overrides:
        config.update(overrides)

    return config
