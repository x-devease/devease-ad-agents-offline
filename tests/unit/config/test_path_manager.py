"""Unit tests for path_manager module."""

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pytest

from src.config.path_manager import (
    PathManager,
    get_path_manager,
    reset_path_manager,
)
from src.config.schemas import PathsConfig


class TestPathManager(TestCase):
    """Test PathManager class."""

    def setUp(self):
        """Set up test fixtures."""
        reset_path_manager()
        self.temp_dir = tempfile.mkdtemp()
        self.customer = "test_customer"
        self.platform = "meta"

    def tearDown(self):
        """Clean up test fixtures."""
        reset_path_manager()
        if os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_initialization_with_defaults(self):
        """Test PathManager initialization with default values."""
        pm = PathManager(customer=self.customer, platform=self.platform)

        assert pm.customer == self.customer
        assert pm.platform == self.platform
        assert pm.base_dir == Path(".")
        assert isinstance(pm.config, PathsConfig)

    def test_initialization_with_base_dir_override(self):
        """Test PathManager with base directory override."""
        pm = PathManager(
            customer=self.customer,
            platform=self.platform,
            base_dir_override=self.temp_dir,
        )

        assert pm.base_dir == Path(self.temp_dir)

    def test_initialization_with_env_base_dir(self):
        """Test PathManager with BASE_DIR environment variable."""
        with patch.dict(os.environ, {"BASE_DIR": self.temp_dir}):
            pm = PathManager(customer=self.customer, platform=self.platform)
            assert pm.base_dir == Path(self.temp_dir)

    def test_initialization_with_custom_config(self):
        """Test PathManager with custom paths config."""
        custom_config = PathsConfig(
            data_dir="custom_data", results_dir="custom_results"
        )
        pm = PathManager(
            customer=self.customer, platform=self.platform, paths_config=custom_config
        )

        assert "custom_data" in str(pm.data_base)
        assert "custom_results" in str(pm.results_base)

    # === Input Data Paths ===

    def test_data_base(self):
        """Test data_base property."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        expected = Path(".") / "datasets"
        assert pm.data_base == expected

    def test_raw_data_dir(self):
        """Test raw_data_dir method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        expected = Path(".") / "datasets" / self.customer / self.platform / "raw"
        assert pm.raw_data_dir() == expected

    def test_raw_data_dir_with_overrides(self):
        """Test raw_data_dir with customer and platform overrides."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        custom_customer = "custom_customer"
        custom_platform = "google"

        result = pm.raw_data_dir(customer=custom_customer, platform=custom_platform)
        expected = Path(".") / "datasets" / custom_customer / custom_platform / "raw"
        assert result == expected

    def test_features_dir(self):
        """Test features_dir method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        expected = Path(".") / "datasets" / self.customer / self.platform / "features"
        assert pm.features_dir() == expected

    def test_ad_features_path(self):
        """Test ad_features_path method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        result = pm.ad_features_path()
        assert "ad_features.csv" in str(result)
        assert self.customer in str(result)
        assert self.platform in str(result)

    def test_adset_features_path(self):
        """Test adset_features_path method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        result = pm.adset_features_path()
        assert "adset_features.csv" in str(result)
        assert self.customer in str(result)
        assert self.platform in str(result)

    # === Output Paths ===

    def test_results_base(self):
        """Test results_base property."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        expected = Path(".") / "results"
        assert pm.results_base == expected

    def test_results_dir_default_method(self):
        """Test results_dir with default method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        expected = Path(".") / "results" / self.customer / self.platform / "rules"
        assert pm.results_dir() == expected

    def test_results_dir_custom_method(self):
        """Test results_dir with custom method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        method = "rules"
        expected = Path(".") / "results" / self.customer / self.platform / method
        assert pm.results_dir(method=method) == expected

    def test_allocations_path(self):
        """Test allocations_path method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        result = pm.allocations_path()
        assert "budget_allocations.csv" in str(result)
        assert self.customer in str(result)

    def test_allocations_path_custom_filename(self):
        """Test allocations_path with custom filename."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        custom_filename = "custom_allocations.csv"
        result = pm.allocations_path(filename=custom_filename)
        assert custom_filename in str(result)

    # === Log Paths ===

    def test_logs_base(self):
        """Test logs_base property."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        expected = Path(".") / "logs"
        assert pm.logs_base == expected

    def test_log_file(self):
        """Test log_file method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        log_name = "test_log"
        result = pm.log_file(log_name)
        assert log_name in str(result)
        assert str(result).endswith(".log")

    # === Cache Paths ===

    def test_cache_base(self):
        """Test cache_base property."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        expected = Path(".") / "cache"
        assert pm.cache_base == expected

    def test_cache_path(self):
        """Test cache_path method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        cache_key = "test_cache_key"
        result = pm.cache_path(cache_key)
        assert cache_key in str(result)
        assert str(result).endswith(".pkl")

    # === Config Paths ===

    def test_config_base(self):
        """Test config_base property."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        expected = Path(".") / "config"
        assert pm.config_base == expected

    def test_rules_config_path(self):
        """Test rules_config_path method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        result = pm.rules_config_path()
        assert "rules.yaml" in str(result)
        assert self.customer in str(result)
        assert self.platform in str(result)

    def test_rules_config_path_custom_filename(self):
        """Test rules_config_path with custom filename."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        custom_filename = "custom_rules.yaml"
        result = pm.rules_config_path(filename=custom_filename)
        assert custom_filename in str(result)

    def test_system_config_path(self):
        """Test system_config_path method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        result = pm.system_config_path()
        # Default environment is "default", so returns "default.yaml"
        assert ".yaml" in str(result)

    def test_system_config_path_environment(self):
        """Test system_config_path with environment."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        environment = "development"
        result = pm.system_config_path(environment=environment)
        assert environment in str(result)

    # === Utility Methods ===

    def test_ensure_directories_creates_missing(self):
        """Test ensure_directories creates missing directories."""
        pm = PathManager(
            customer=self.customer,
            platform=self.platform,
            base_dir_override=self.temp_dir,
        )
        pm.ensure_directories(create=True)

        # Check that directories were created
        assert pm.raw_data_dir().exists()
        assert pm.features_dir().exists()
        assert pm.results_dir().exists()
        assert pm.logs_base.exists()
        assert pm.cache_base.exists()

    def test_ensure_directories_raises_when_not_exists(self):
        """Test ensure_directories raises FileNotFoundError when not exists."""
        pm = PathManager(
            customer=self.customer,
            platform=self.platform,
            base_dir_override=self.temp_dir,
        )

        # Don't create directories, should raise
        with pytest.raises(FileNotFoundError):
            pm.ensure_directories(create=False)

    def test_validate_paths_without_require_exists(self):
        """Test validate_paths returns True when not requiring existence."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        assert pm.validate_paths(require_exists=False) is True

    def test_validate_paths_with_require_exists(self):
        """Test validate_paths with require_exists."""
        pm = PathManager(
            customer=self.customer,
            platform=self.platform,
            base_dir_override=self.temp_dir,
        )

        # Should pass when directories exist
        pm.ensure_directories(create=True)
        assert pm.validate_paths(require_exists=True) is True

    def test_override_base_dir(self):
        """Test override_base_dir method."""
        pm = PathManager(customer=self.customer, platform=self.platform)
        new_base = Path(self.temp_dir)
        pm.override_base_dir(new_base)
        assert pm.base_dir == new_base

    # === Global Instance ===

    def test_get_path_manager_returns_singleton(self):
        """Test get_path_manager returns singleton instance."""
        pm1 = get_path_manager(customer=self.customer, platform=self.platform)
        pm2 = get_path_manager(customer=self.customer, platform=self.platform)
        assert pm1 is pm2

    def test_get_path_manager_force_refresh(self):
        """Test get_path_manager with force_refresh."""
        pm1 = get_path_manager(customer=self.customer, platform=self.platform)
        pm2 = get_path_manager(
            customer=self.customer, platform=self.platform, force_refresh=True
        )
        assert pm1 is not pm2

    def test_reset_path_manager(self):
        """Test reset_path_manager clears global instance."""
        pm1 = get_path_manager(customer=self.customer, platform=self.platform)
        reset_path_manager()
        pm2 = get_path_manager(customer=self.customer, platform=self.platform)
        assert pm1 is not pm2
