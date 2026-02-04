"""
Configuration Loader for Diagnoser Agents.

Loads YAML configuration and injects values into prompt templates.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


class AgentConfig:
    """
    Agent configuration loader and injector.

    Loads configuration from YAML and provides methods for injection into prompts.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to agent_config.yaml. If None, uses default path.
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required. Install with: pip install pyyaml"
            )

        if config_path is None:
            config_path = Path(__file__).parent / "agent_config.yaml"
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        self.config = self._load_config()

        logger.info(f"Loaded agent configuration from {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.

        Args:
            key: Dot-separated key path (e.g., "business.avg_monthly_spend_per_ad")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get("business.avg_monthly_spend_per_ad")
            500
            >>> config.get("agents.pm.temperature")
            0.7
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def inject_into_prompt(self, prompt: str) -> str:
        """
        Inject configuration values into prompt template.

        Replaces {CONFIG:key.path} placeholders with actual configuration values.

        Args:
            prompt: Prompt template string

        Returns:
            Prompt with injected configuration values

        Examples:
            >>> prompt = "Monthly spend: {CONFIG:business.avg_monthly_spend_per_ad}"
            >>> config.inject_into_prompt(prompt)
            "Monthly spend: 500"
        """
        pattern = r'\{CONFIG:([^}]+)\}'
        matches = re.findall(pattern, prompt)

        injected_prompt = prompt
        for match in matches:
            value = self.get(match)
            if value is not None:
                injected_prompt = injected_prompt.replace(f'{{CONFIG:{match}}}', str(value))
            else:
                logger.warning(f"Configuration key not found: {match}")

        return injected_prompt

    def get_agent_settings(self, agent_name: str) -> Dict[str, Any]:
        """
        Get settings for a specific agent.

        Args:
            agent_name: Agent name (pm, coder, reviewer, judge, memory)

        Returns:
            Agent settings dictionary
        """
        return self.get(f"agents.{agent_name}", {})

    def get_temperature(self, agent_name: str) -> float:
        """Get temperature setting for an agent."""
        return self.get(f"agents.{agent_name}.temperature", 0.7)

    def get_max_tokens(self, agent_name: str) -> int:
        """Get max_tokens setting for an agent."""
        return self.get(f"agents.{agent_name}.max_tokens", 4096)

    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if an advanced feature is enabled.

        Args:
            feature_name: Feature name (semantic_search_enabled, telemetry_enabled, etc.)

        Returns:
            True if feature is enabled, False otherwise
        """
        return self.get(f"advanced_features.{feature_name}", False)

    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")


# Singleton instance for convenience
_global_config: Optional[AgentConfig] = None


def get_global_config() -> AgentConfig:
    """
    Get global configuration instance.

    Returns:
        AgentConfig singleton instance
    """
    global _global_config
    if _global_config is None:
        _global_config = AgentConfig()
    return _global_config
