"""
Parser compatibility wrapper for legacy code.

Provides a compatibility layer for code that expects a Parser class
with methods like get_safety_rule() and decision_rules property.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.config.manager import ConfigManager


class Parser:
    """
    Compatibility wrapper for legacy Parser interface.
    
    Wraps ConfigManager to provide the old Parser API.
    """

    def __init__(
        self,
        config_path: str,
        customer_name: str = "moprobo",
        platform: str = "meta",
    ):
        """Initialize parser with config file path.

        Args:
            config_path: Path to rules.yaml config file
            customer_name: Customer name
            platform: Platform name
        """
        self.config_path = Path(config_path)
        self.customer_name = customer_name
        self.platform = platform

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML directly for backward compatibility
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config_dict = yaml.safe_load(f) or {}

        # Also initialize ConfigManager for future use
        self._config_manager = ConfigManager(
            customer=customer_name,
            platform=platform,
            config_path_override=self.config_path,
        )

    def get_safety_rule(self, key: str, default: Any = None) -> Any:
        """Get a safety rule value.

        Args:
            key: Rule key name
            default: Default value if not found

        Returns:
            Rule value or default
        """
        safety_rules = self._config_dict.get("safety_rules", {})
        return safety_rules.get(key, default)

    @property
    def decision_rules(self) -> Dict[str, Any]:
        """Get decision rules dictionary.

        Returns:
            Dictionary of decision rules
        """
        return self._config_dict.get("decision_rules", {})

    @property
    def safety_rules(self) -> Dict[str, Any]:
        """Get safety rules dictionary.

        Returns:
            Dictionary of safety rules
        """
        return self._config_dict.get("safety_rules", {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config_dict

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_advanced_concept(self, key: str, default: Any = None) -> Any:
        """Get an advanced concept value.

        Args:
            key: Advanced concept key name
            default: Default value if not found

        Returns:
            Advanced concept value or default
        """
        advanced_concepts = self._config_dict.get("advanced_concepts", {})
        return advanced_concepts.get(key, default)
