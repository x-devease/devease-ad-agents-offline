"""
Configuration Loader Module

Load and validate configuration files (YAML/JSON).
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """
    Configuration Loader: Load and validate configuration files

    This class loads configuration files (YAML or JSON), validates them,
    and merges with default configuration.
    """

    @staticmethod
    def _parse_customer_platform(
        config_path_obj: Path,
    ) -> tuple[str, str]:
        """
        Parse customer and platform from config file path.

        Expected structure: config/{customer}/{platform}/*.yaml

        Args:
            config_path_obj: Path object of config file

        Returns:
            Tuple of (customer, platform)

        Raises:
            ValueError: If path structure is invalid
        """
        # Expected: config/customer/platform/config.yaml
        # We need to find the 'config' directory in the path
        parts = config_path_obj.parts

        try:
            config_idx = parts.index("config")
        except ValueError as exc:
            raise ValueError(
                "Config file must be in a 'config' directory "
                "(e.g., config/customer/platform/config.yaml)"
            ) from exc

        # Check if we have at least: config/customer/platform/file.yaml
        if len(parts) < config_idx + 4:
            raise ValueError(
                "Config file path must follow structure: "
                "config/{customer}/{platform}/*.yaml"
            ) from None

        customer = parts[config_idx + 1]
        platform = parts[config_idx + 2]

        return customer, platform

    @staticmethod
    def _construct_paths(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct input_path, output_path, and log_file from customer/platform.

        Args:
            config: Configuration dict with customer, platform, and dataset

        Returns:
            Updated config with constructed paths
        """
        customer = config.get("customer", "default")
        platform = config.get("platform", "default")
        dataset = config.get("dataset", {})

        # Construct input_path
        input_file = dataset.get("input_file", "creative_image_features.csv")
        input_path = f"data/{customer}/{platform}/{input_file}"

        # Construct output_path
        output_path = f"results/{customer}/{platform}/"

        # Construct log file path (in reports subfolder)
        log_file = f"results/{customer}/{platform}/reports/analysis.log"

        # Update config
        config = copy.deepcopy(config)
        if "dataset" not in config:
            config["dataset"] = {}
        config["dataset"]["input_path"] = input_path
        config["dataset"]["output_path"] = output_path

        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["file"] = log_file

        return config

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file (YAML or JSON).

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary (merged with defaults)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path_obj = Path(config_path)

        # Check if file exists
        if not config_path_obj.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        # Detect file format and load
        if config_path_obj.suffix.lower() in [".yaml", ".yml"]:
            config = ConfigLoader._load_yaml(config_path_obj)
        elif config_path_obj.suffix.lower() == ".json":
            config = ConfigLoader._load_json(config_path_obj)
        else:
            # Try to detect by content
            try:
                config = ConfigLoader._load_yaml(config_path_obj)
            except (yaml.YAMLError, ValueError):
                config = ConfigLoader._load_json(config_path_obj)

        # Parse customer and platform from config file path
        customer, platform = ConfigLoader._parse_customer_platform(
            config_path_obj
        )
        config["customer"] = customer
        config["platform"] = platform

        # Construct paths dynamically
        config = ConfigLoader._construct_paths(config)

        # Validate configuration
        ConfigLoader.validate(config)

        # Merge with default configuration
        default_config = ConfigLoader.load_default()
        merged_config = ConfigLoader.merge(default_config, config)

        return merged_config

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
            if config is None:
                return {}
            return config
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML in configuration file: {error}"
            ) from error

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            return config
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in configuration file: {error}"
            ) from error

    @staticmethod
    def _validate_dataset_section(
        dataset: Dict[str, Any], errors: list
    ) -> None:
        """Validate dataset section of config."""
        # Accept either input_file or input_path (input_file preferred)
        if "input_file" not in dataset and "input_path" not in dataset:
            errors.append("Missing required key: dataset.input_file")
        if "roas_columns" not in dataset:
            errors.append("Missing required key: dataset.roas_columns")
        elif (
            not isinstance(dataset["roas_columns"], list)
            or len(dataset["roas_columns"]) == 0
        ):
            errors.append("dataset.roas_columns must be a non-empty list")

    @staticmethod
    def _validate_model_section(model: Dict[str, Any], errors: list) -> None:
        """Validate model section of config."""
        if not isinstance(model, dict):
            return
        model_dict: Dict[str, Any] = model
        if model_dict.get("random_state") is None:
            errors.append("Missing required key: model.random_state")
        elif not isinstance(model_dict.get("random_state"), int):
            errors.append("model.random_state must be an integer")

    @staticmethod
    def _validate_analysis_section(
        analysis: Dict[str, Any], errors: list
    ) -> None:
        """Validate analysis section of config."""
        if not isinstance(analysis, dict):
            return
        analysis_dict: Dict[str, Any] = analysis
        if analysis_dict.get("feature_importance") is None:
            errors.append("Missing required key: analysis.feature_importance")
        if analysis_dict.get("feature_value_analysis") is None:
            errors.append(
                "Missing required key: analysis.feature_value_analysis"
            )

    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        errors = []

        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            raise ValueError("Configuration must be a dictionary")

        required_keys = ["dataset", "model", "analysis"]
        config_dict: Dict[str, Any] = config
        for key in required_keys:
            if config_dict.get(key) is None:
                errors.append(f"Missing required key: {key}")

        dataset = config_dict.get("dataset")
        if dataset and isinstance(dataset, dict):
            ConfigLoader._validate_dataset_section(dataset, errors)
        model = config_dict.get("model")
        if model and isinstance(model, dict):
            ConfigLoader._validate_model_section(model, errors)
        analysis = config_dict.get("analysis")
        if analysis and isinstance(analysis, dict):
            ConfigLoader._validate_analysis_section(analysis, errors)

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            raise ValueError(error_msg)

        return True

    @staticmethod
    def load_default() -> Dict[str, Any]:
        """
        Load default configuration.

        Returns:
            Default configuration dictionary
        """
        # Try to load from default_config.yaml
        # config_loader is in src/utils/, default_config.yaml is in config/
        config_dir = Path(__file__).parent.parent.parent / "config"
        default_config_path = config_dir / "default_config.yaml"

        if default_config_path.exists():
            try:
                return ConfigLoader._load_yaml(default_config_path)
            except (yaml.YAMLError, ValueError):
                # Fall back to hardcoded defaults
                pass

        # Hardcoded defaults (fallback)
        return {
            "version": "1.0",
            "customer": "default",
            "dataset": {
                "input_path": "data/creative_image_features.csv",
                "output_path": "data/results/",
                "roas_columns": ["mean_roas", "total_roas", "weighted_roas"],
                "id_columns": ["filename", "creative_id"],
                "exclude_patterns": {
                    "roas": ["roas", "roi", "revenue", "purchase_value"],
                    "id": ["filename", "creative_id", "image_path", "ad_id"],
                    "metrics": ["spend", "impressions", "clicks", "ad_count"],
                },
                "data_quality": {
                    "min_samples": 50,
                    "max_missing_rate": 0.3,
                    "roas_range": [0, 100],
                },
            },
            "model": {
                "random_state": 42,
                "test_size": 0.2,
                "cv_folds": 5,
                "regression": {
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.05,
                },
                "classification": {
                    "iterations": 300,
                    "depth": 4,
                    "learning_rate": 0.05,
                },
            },
            "analysis": {
                "feature_importance": {"top_n_features": 10},
                "feature_value_analysis": {
                    "methods": {
                        "method_1a": True,
                        "method_1b": True,
                        "method_2": True,
                        "method_3": False,
                    },
                    "significance_level": 0.05,
                    "min_effect_size": 0.3,
                    "use_median": True,
                },
            },
            "output": {
                "formats": ["json", "html", "markdown"],
                "files": {
                    "recommendations": "recommendations.json",
                    "report": "report.html",
                    "metadata": "metadata.json",
                },
            },
            "logging": {"level": "INFO", "console": True},
        }

    @staticmethod
    def merge(
        default: Dict[str, Any], custom: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge default and custom configurations (deep merge).

        Args:
            default: Default configuration
            custom: Custom configuration

        Returns:
            Merged configuration (custom values override defaults)
        """
        merged = copy.deepcopy(default)

        for key, value in custom.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                merged[key] = ConfigLoader.merge(merged[key], value)
            else:
                # Override with custom value
                merged[key] = copy.deepcopy(value)

        return merged
