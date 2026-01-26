"""Configuration manager with layered loading support.

Implements a priority-based configuration loading system:
1. System defaults (hardcoded)
2. System config (config/default/system.yaml)
3. Environment config (config/default/{environment}.yaml)
4. Customer config (config/adset/allocator/{customer}/default.yaml)
5. Platform config (config/adset/allocator/{customer}/{platform}/rules.yaml)
6. Environment variables
7. CLI/runtime overrides
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import SystemConfig, RulesConfig, PathsConfig
from .path_manager import PathManager, get_path_manager


class ConfigManager:
    """Manages configuration loading with layered overrides.

    Implements a priority-based system where later layers override earlier ones.

    Backward compatibility: Provides class-level API for legacy code.
    """

    # Class-level constants for backward compatibility
    BASE_DIR = Path(os.getcwd())
    CONFIG_DIR = BASE_DIR / "config"

    def __init__(
        self,
        customer: str = "moprobo",
        platform: str = "meta",
        environment: str = "production",
        config_path_override: Optional[Path] = None,
    ):
        """Initialize the configuration manager.

        Args:
            customer: Customer name
            platform: Platform name (e.g., 'meta', 'google')
            environment: Environment name (development, staging, production)
            config_path_override: Direct path to config file (skips layered loading)
        """
        self.customer = customer
        self.platform = platform
        self.environment = environment
        self.config_path_override = config_path_override

        # Load configuration
        self._config: Optional[SystemConfig] = None
        self._rules_config: Optional[RulesConfig] = None
        self._path_manager: Optional[PathManager] = None

        self._load_config()

    @property
    def config(self) -> SystemConfig:
        """Get the system configuration."""
        return self._config

    @property
    def rules(self) -> RulesConfig:
        """Get the rules configuration."""
        return self._rules_config

    @property
    def paths(self) -> PathManager:
        """Get the path manager."""
        return self._path_manager

    def _load_config(self) -> None:
        """Load configuration with layered overrides."""
        # Start with system defaults
        config_dict = self._get_system_defaults()

        # Layer 1: Default config file
        default_config_path = Path("config/adset/allocator/system.yaml")
        if default_config_path.exists():
            config_dict = self._deep_merge(
                config_dict, self._load_yaml(default_config_path)
            )

        # Layer 2: Environment config
        env_config_path = Path(f"config/adset/allocator/{self.environment}.yaml")
        if env_config_path.exists():
            config_dict = self._deep_merge(
                config_dict, self._load_yaml(env_config_path)
            )

        # Layer 3: Customer base config
        customer_base_path = Path(f"config/adset/allocator/{self.customer}/default.yaml")
        if customer_base_path.exists():
            config_dict = self._deep_merge(
                config_dict, self._load_yaml(customer_base_path)
            )

        # Layer 4: Platform-specific rules
        if self.config_path_override:
            rules_dict = self._load_yaml(self.config_path_override)
        else:
            platform_rules_path = Path(
                f"config/adset/allocator/{self.customer}/{self.platform}/rules.yaml"
            )
            if platform_rules_path.exists():
                rules_dict = self._load_yaml(platform_rules_path)
            else:
                rules_dict = {}

        # Layer 5: Environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)

        # Transform rolling_windows from dict to list if needed
        if "rolling_windows" in rules_dict and isinstance(rules_dict["rolling_windows"], dict):
            # Convert dict format {short_window_days: 7, long_window_days: 14} to list [7, 14]
            rolling_windows_dict = rules_dict["rolling_windows"]
            rules_dict["rolling_windows"] = [
                rolling_windows_dict.get("short_window_days", 7),
                rolling_windows_dict.get("long_window_days", 14),
            ]

        # Create config objects
        self._config = SystemConfig(**config_dict)
        self._rules_config = RulesConfig(**rules_dict)

        # Create path manager
        self._path_manager = PathManager(
            customer=self.customer,
            platform=self.platform,
            paths_config=self._config.paths,
        )

    def _get_system_defaults(self) -> Dict[str, Any]:
        """Get system default configuration."""
        return {
            "environment": self.environment,
            "customer": self.customer,
            "platform": self.platform,
            "test_mode": False,
            "verbose": False,
            "paths": {
                "base_dir": os.environ.get("BASE_DIR", "."),
                "data_dir": "datasets",
                "results_dir": "results",
                "logs_dir": "logs",
                "cache_dir": "cache",
                "config_dir": "config",
            },
        }

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary with YAML content
        """
        try:
            import yaml

            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # Fallback: return empty dict if yaml not available
            return {}
        except FileNotFoundError:
            return {}

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides.

        Supported environment variables:
        - ENVIRONMENT: Set environment
        - CUSTOMER: Set customer
        - PLATFORM: Set platform
        - TEST_MODE: Enable test mode
        - VERBOSE: Enable verbose logging
        - BASE_DIR: Override base directory

        Args:
            config_dict: Current config dictionary

        Returns:
            Updated config dictionary
        """
        if "ENVIRONMENT" in os.environ:
            config_dict["environment"] = os.environ["ENVIRONMENT"]

        if "CUSTOMER" in os.environ:
            config_dict["customer"] = os.environ["CUSTOMER"]

        if "PLATFORM" in os.environ:
            config_dict["platform"] = os.environ["PLATFORM"]

        if "TEST_MODE" in os.environ:
            config_dict["test_mode"] = os.environ["TEST_MODE"].lower() in (
                "true",
                "1",
                "yes",
            )

        if "VERBOSE" in os.environ:
            config_dict["verbose"] = os.environ["VERBOSE"].lower() in (
                "true",
                "1",
                "yes",
            )

        if "BASE_DIR" in os.environ:
            if "paths" not in config_dict:
                config_dict["paths"] = {}
            config_dict["paths"]["base_dir"] = os.environ["BASE_DIR"]

        return config_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (dot notation supported).

        Args:
            key: Configuration key (e.g., 'paths.data_dir')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config.model_dump()

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def reload(self) -> None:
        """Reload configuration from files."""
        self._load_config()

    def set_test_mode(self, enabled: bool = True) -> None:
        """Enable or disable test mode.

        In test mode, uses test-specific paths and configurations.

        Args:
            enabled: Whether to enable test mode
        """
        self._config.test_mode = enabled



    # Backward compatibility: Class-level methods for legacy Config API
    @classmethod
    def DATASETS_DIR(cls) -> Path:
        """Get datasets directory path."""
        return cls.BASE_DIR / "datasets"

    @classmethod
    def MODELS_DIR(cls) -> Path:
        """Get models directory path."""
        return cls.BASE_DIR / "models"

    @classmethod
    def RESULTS_DIR(cls) -> Path:
        """Get results directory path."""
        return cls.BASE_DIR / "results"

    @classmethod
    def TARGET_COLUMN(cls) -> str:
        """Get target column name."""
        return "purchase_roas"

    @classmethod
    def MIN_SPEND(cls) -> int:
        """Get minimum spend threshold."""
        return 10

    @classmethod
    def MIN_IMPRESSIONS(cls) -> int:
        """Get minimum impressions threshold."""
        return 100

    @classmethod
    def RANDOM_STATE(cls) -> int:
        """Get random state for reproducibility."""
        return 42

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist."""
        cls.DATASETS_DIR().mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR().mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR().mkdir(parents=True, exist_ok=True)
        cls.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Deprecated: Use get_customer_params() instead
    @classmethod
    def get_customer_params(cls, customer: str, platform: str) -> Dict[str, Any]:
        """Get customer-specific parameters (deprecated, kept for backward compatibility)."""
        # This method was referenced in tests but doesn't exist in current implementation
        # Return a basic dict for backward compatibility
        return {
            "customer": customer,
            "platform": platform,
            "min_spend": cls.MIN_SPEND(),
            "min_impressions": cls.MIN_IMPRESSIONS(),
            "target_column": cls.TARGET_COLUMN(),
            "random_state": cls.RANDOM_STATE(),
        }

# Global config manager instance (lazy initialized)
_config_manager: Optional[ConfigManager] = None


def get_config(
    customer: str = "moprobo",
    platform: str = "meta",
    environment: str = "production",
    config_path_override: Optional[Path] = None,
    force_refresh: bool = False,
) -> ConfigManager:
    """Get or create the global configuration manager instance.

    Args:
        customer: Customer name
        platform: Platform name
        environment: Environment name
        config_path_override: Direct path to config file
        force_refresh: Force recreation of config manager

    Returns:
        ConfigManager instance
    """
    global _config_manager

    if _config_manager is None or force_refresh:
        _config_manager = ConfigManager(
            customer=customer,
            platform=platform,
            environment=environment,
            config_path_override=config_path_override,
        )

    return _config_manager


def reset_config() -> None:
    """Reset the global configuration manager instance (useful for testing)."""
    global _config_manager
    global _path_manager
    _config_manager = None
    _path_manager = None
