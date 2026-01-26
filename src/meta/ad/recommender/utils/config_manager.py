"""Unified configuration manager for the project.

This module provides a centralized configuration loading system that eliminates
hard-coded paths and provides a consistent API for accessing configuration files
across all modules.

Usage:
    from src.utils import ConfigManager
    # or
    from src.utils.config_manager import ConfigManager

    # Load a configuration file (flat structure)
    prompts_config = ConfigManager.get_config(None, "ad/recommender/gpt4/prompts.yaml")
    features_config = ConfigManager.get_config(None, "ad/recommender/gpt4/features.yaml")

    # Access nested values
    prompt_template = prompts_config.get("prompt_template", "")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager for loading YAML/JSON configuration files.

    This class provides a centralized way to load configuration files from the
    config/ directory structure, eliminating the need for hard-coded paths
    in individual modules.

    Attributes:
        _config_cache: Cache for loaded configuration files to avoid redundant
            file I/O operations.
        _base_config_path: Base path to the configuration directory.
    """

    _config_cache: Dict[str, Dict[str, Any]] = {}
    _base_config_path: Optional[Path] = None

    @classmethod
    def _get_base_config_path(cls) -> Path:
        """Get the base path to the configuration directory.

        Returns:
            Path object pointing to config/ directory.

        Raises:
            RuntimeError: If the config directory cannot be found.
        """
        if cls._base_config_path is not None:
            return cls._base_config_path

        # Try to find the config directory relative to this file
        current_file = Path(__file__).resolve()

        # This file is in src/utils/config_manager.py
        # So config/ should be at the root of the repository
        repo_root = current_file.parent.parent.parent
        config_dir = repo_root / "config"

        if not config_dir.exists():
            raise RuntimeError(
                f"Configuration directory not found: {config_dir}"
            )

        cls._base_config_path = config_dir
        return config_dir

    @classmethod
    def get_config_path(cls, module: Optional[str], filename: str) -> Path:
        """Get the full path to a configuration file.

        Args:
            module: Optional module name (e.g., "features").
                If provided, the file will be looked up in
                config/{module}/{filename}. If None, the file will be
                looked up directly in config/{filename}.
            filename: Configuration file name (e.g., "ad/recommender/gpt4/prompts.yaml").

        Returns:
            Path object to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        base_path = cls._get_base_config_path()

        if module:
            # Support legacy module-based path structure
            config_path = base_path / module / filename
        else:
            # Flat structure: file directly in config directory
            config_path = base_path / filename

        if not config_path.exists():
            if module:
                expected_location = f"{base_path}/{module}/{filename}"
            else:
                expected_location = f"{base_path}/{filename}"
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Expected location: {expected_location}"
            )

        return config_path

    @classmethod
    def get_config(
        cls, module: Optional[str], filename: str, use_cache: bool = True
    ) -> Dict[str, Any]:
        """Load a configuration file.

        Args:
            module: Optional module name (e.g., "features").
                If provided, the file will be looked up in
                config/{module}/{filename}. If None, the file will be
                looked up directly in config/{filename}.
            filename: Configuration file name (e.g., "ad/recommender/gpt4/prompts.yaml").
            use_cache: If True, use cached configuration if available.
                Defaults to True.

        Returns:
            Dictionary containing the configuration data.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file cannot be parsed
                (invalid YAML/JSON).
        """
        cache_key = f"{module}/{filename}" if module else filename

        # Return cached config if available and caching is enabled
        if use_cache and cache_key in cls._config_cache:
            logger.debug("Returning cached config: %s", cache_key)
            return cls._config_cache[cache_key]

        # Get the file path
        config_path = cls.get_config_path(module, filename)

        try:
            # Load configuration file
            with open(config_path, "r", encoding="utf-8") as config_file:
                if filename.endswith((".yaml", ".yml")):
                    config = yaml.safe_load(config_file)
                elif filename.endswith(".json"):
                    config = json.load(config_file)
                else:
                    # Try YAML first, then JSON
                    try:
                        config_file.seek(0)
                        config = yaml.safe_load(config_file)
                    except yaml.YAMLError:
                        config_file.seek(0)
                        config = json.load(config_file)

            if config is None:
                config = {}

            # Cache the configuration
            if use_cache:
                cls._config_cache[cache_key] = config

            logger.debug("Loaded config: %s", cache_key)
            return config

        except yaml.YAMLError as yaml_error:
            raise ValueError(
                f"Error parsing YAML configuration file {config_path}: "
                f"{yaml_error}"
            ) from yaml_error
        except Exception as load_error:
            raise ValueError(
                f"Error loading configuration file {config_path}: "
                f"{load_error}"
            ) from load_error

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the configuration cache.

        This is useful for testing or when configuration files are updated
        at runtime and need to be reloaded.
        """
        cls._config_cache.clear()
        logger.debug("Configuration cache cleared")

    @classmethod
    def reload_config(
        cls, module: Optional[str], filename: str
    ) -> Dict[str, Any]:
        """Reload a configuration file, bypassing the cache.

        Args:
            module: Optional module name. If None, reloads flat structure file.
            filename: Configuration file name.

        Returns:
            Dictionary containing the configuration data.
        """
        cache_key = f"{module}/{filename}" if module else filename
        if cache_key in cls._config_cache:
            del cls._config_cache[cache_key]
        return cls.get_config(module, filename, use_cache=True)
