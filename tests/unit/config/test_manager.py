"""Unit tests for config manager module."""

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest

from src.config.manager import (
    ConfigManager,
    get_config,
    reset_config,
)
from src.config.schemas import SystemConfig, RulesConfig


class TestConfigManager(TestCase):
    """Test ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_config()
        self.temp_dir = tempfile.mkdtemp()
        self.customer = "test_customer"
        self.platform = "meta"
        self.environment = "development"

    def tearDown(self):
        """Clean up test fixtures."""
        reset_config()
        if os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_initialization_with_defaults(self):
        """Test ConfigManager initialization with default parameters."""
        manager = ConfigManager(
            customer=self.customer, platform=self.platform, environment=self.environment
        )

        assert manager.customer == self.customer
        assert manager.platform == self.platform
        assert manager.environment == self.environment
        assert manager.config_path_override is None

    def test_initialization_with_config_path_override(self):
        """Test ConfigManager with direct config path override."""
        config_path = Path(self.temp_dir) / "custom_config.yaml"
        manager = ConfigManager(
            customer=self.customer,
            platform=self.platform,
            config_path_override=config_path,
        )

        assert manager.config_path_override == config_path

    def test_config_property_returns_system_config(self):
        """Test config property returns SystemConfig instance."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        assert isinstance(manager.config, SystemConfig)
        assert manager.config.customer == self.customer
        assert manager.config.platform == self.platform

    def test_rules_property_returns_rules_config(self):
        """Test rules property returns RulesConfig instance."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        assert isinstance(manager.rules, RulesConfig)

    def test_paths_property_returns_path_manager(self):
        """Test paths property returns PathManager instance."""
        from src.config.path_manager import PathManager

        manager = ConfigManager(customer=self.customer, platform=self.platform)

        assert isinstance(manager.paths, PathManager)
        assert manager.paths.customer == self.customer
        assert manager.paths.platform == self.platform

    def test_get_system_defaults(self):
        """Test _get_system_defaults returns correct structure."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        defaults = manager._get_system_defaults()

        assert "environment" in defaults
        assert "customer" in defaults
        assert "platform" in defaults
        assert "test_mode" in defaults
        assert "verbose" in defaults
        assert "paths" in defaults

        # Check values
        assert (
            defaults["environment"] == "production"
        )  # Hardcoded in _get_system_defaults
        assert defaults["customer"] == self.customer
        assert defaults["platform"] == self.platform
        assert defaults["test_mode"] is False
        assert defaults["verbose"] is False

    def test_load_yaml_with_valid_file(self):
        """Test _load_yaml successfully loads valid YAML file."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        # Create a temporary YAML file
        yaml_path = Path(self.temp_dir) / "test.yaml"
        yaml_path.write_text("key1: value1\nkey2:\n  nested: value2\n")

        result = manager._load_yaml(yaml_path)
        assert result == {"key1": "value1", "key2": {"nested": "value2"}}

    def test_load_yaml_with_nonexistent_file(self):
        """Test _load_yaml returns empty dict for missing file."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        result = manager._load_yaml(Path("nonexistent.yaml"))
        assert result == {}

    def test_deep_merge_with_simple_dicts(self):
        """Test _deep_merge with simple dictionaries."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        base = {"key1": "value1", "key2": "value2"}
        override = {"key2": "new_value2", "key3": "value3"}

        result = manager._deep_merge(base, override)

        assert result["key1"] == "value1"  # Unchanged
        assert result["key2"] == "new_value2"  # Overridden
        assert result["key3"] == "value3"  # Added

    def test_deep_merge_with_nested_dicts(self):
        """Test _deep_merge with nested dictionaries."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        base = {"key1": "value1", "nested": {"inner1": "value1", "inner2": "value2"}}
        override = {"nested": {"inner2": "new_value2", "inner3": "value3"}}

        result = manager._deep_merge(base, override)

        assert result["key1"] == "value1"
        assert result["nested"]["inner1"] == "value1"  # Unchanged
        assert result["nested"]["inner2"] == "new_value2"  # Overridden
        assert result["nested"]["inner3"] == "value3"  # Added

    def test_apply_env_overrides_no_env_vars(self):
        """Test _apply_env_overrides with no environment variables set."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        config_dict = manager._get_system_defaults()
        result = manager._apply_env_overrides(config_dict)

        # Should be unchanged
        assert result["environment"] == config_dict["environment"]
        assert result["customer"] == config_dict["customer"]

    def test_apply_env_overrides_with_env_vars(self):
        """Test _apply_env_overrides with environment variables."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        config_dict = manager._get_system_defaults()

        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "staging",
                "CUSTOMER": "env_customer",
                "PLATFORM": "google",
                "TEST_MODE": "true",
                "VERBOSE": "yes",
                "BASE_DIR": "/custom/base",
            },
        ):
            result = manager._apply_env_overrides(config_dict)

            assert result["environment"] == "staging"
            assert result["customer"] == "env_customer"
            assert result["platform"] == "google"
            assert result["test_mode"] is True
            assert result["verbose"] is True
            assert result["paths"]["base_dir"] == "/custom/base"

    def test_apply_env_overrides_test_mode_variations(self):
        """Test _apply_env_overrides with various TEST_MODE values."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        config_dict = manager._get_system_defaults()

        # Test "true"
        with patch.dict(os.environ, {"TEST_MODE": "true"}):
            result = manager._apply_env_overrides(config_dict.copy())
            assert result["test_mode"] is True

        # Test "1"
        with patch.dict(os.environ, {"TEST_MODE": "1"}):
            result = manager._apply_env_overrides(config_dict.copy())
            assert result["test_mode"] is True

        # Test "yes"
        with patch.dict(os.environ, {"TEST_MODE": "yes"}):
            result = manager._apply_env_overrides(config_dict.copy())
            assert result["test_mode"] is True

        # Test "false"
        with patch.dict(os.environ, {"TEST_MODE": "false"}):
            result = manager._apply_env_overrides(config_dict.copy())
            assert result["test_mode"] is False

    def test_get_with_valid_key(self):
        """Test get method with valid key."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        # Test top-level key
        assert (
            manager.get("environment") == "production"
        )  # Uses default from _get_system_defaults
        assert manager.get("customer") == self.customer

        # Test nested key with dot notation
        assert manager.get("paths.data_dir") == "datasets"

    def test_get_with_invalid_key_returns_default(self):
        """Test get method with invalid key returns default."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        assert manager.get("nonexistent_key") is None
        assert manager.get("nonexistent_key", "default_value") == "default_value"
        assert manager.get("paths.nonexistent", "default") == "default"

    def test_set_test_mode(self):
        """Test set_test_mode method."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        assert manager.config.test_mode is False

        manager.set_test_mode(enabled=True)
        assert manager.config.test_mode is True

        manager.set_test_mode(enabled=False)
        assert manager.config.test_mode is False

    def test_reload_config(self):
        """Test reload method reloads configuration."""
        manager = ConfigManager(customer=self.customer, platform=self.platform)

        # Modify config
        manager.set_test_mode(enabled=True)
        assert manager.config.test_mode is True

        # Reload should reset to defaults
        manager.reload()
        assert manager.config.test_mode is False

    # === Global Instance Tests ===

    def test_get_config_returns_singleton(self):
        """Test get_config returns singleton instance."""
        config1 = get_config(customer=self.customer, platform=self.platform)
        config2 = get_config(customer=self.customer, platform=self.platform)
        assert config1 is config2

    def test_get_config_with_force_refresh(self):
        """Test get_config with force_refresh creates new instance."""
        config1 = get_config(customer=self.customer, platform=self.platform)
        config2 = get_config(
            customer=self.customer, platform=self.platform, force_refresh=True
        )
        assert config1 is not config2

    def test_get_config_with_different_params(self):
        """Test get_config with different parameters."""
        config1 = get_config(customer="customer1", platform="meta")
        config2 = get_config(customer="customer2", platform="google")

        # Should return same singleton (last call wins)
        config3 = get_config()
        assert config3 is config2

    def test_reset_config(self):
        """Test reset_config clears global instance."""
        config1 = get_config(customer=self.customer, platform=self.platform)
        reset_config()
        config2 = get_config(customer=self.customer, platform=self.platform)
        assert config1 is not config2
