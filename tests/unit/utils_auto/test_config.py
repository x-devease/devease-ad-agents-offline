"""Test configuration module."""

import pytest
from pathlib import Path
from src.utils import Config


class TestConfig:
    """Test Config class."""

    def test_base_dir_exists(self):
        """Test that BASE_DIR exists."""
        assert Config.BASE_DIR.exists()

    def test_paths_exist(self):
        """Test that path properties are defined correctly."""
        assert Config.DATASETS_DIR() == Config.BASE_DIR / "datasets"
        assert Config.MODELS_DIR() == Config.BASE_DIR / "models"
        assert Config.RESULTS_DIR() == Config.BASE_DIR / "results"

    def test_config_dir_exists(self):
        """Test that CONFIG_DIR is defined correctly."""
        assert Config.CONFIG_DIR == Config.BASE_DIR / "config"

    def test_ensure_directories(self):
        """Test that ensure_directories creates required directories."""
        Config.ensure_directories()
        assert Config.DATASETS_DIR().exists()
        assert Config.MODELS_DIR().exists()
        assert Config.RESULTS_DIR().exists()
        assert Config.CONFIG_DIR.exists()

    def test_target_column(self):
        """Test target column configuration."""
        assert Config.TARGET_COLUMN() == "purchase_roas"

    def test_min_spend(self):
        """Test minimum spend configuration."""
        assert Config.MIN_SPEND() == 10

    def test_min_impressions(self):
        """Test minimum impressions configuration."""
        assert Config.MIN_IMPRESSIONS() == 100

    def test_random_state(self):
        """Test random state configuration."""
        assert Config.RANDOM_STATE() == 42

    def test_feature_lists_from_config(self):
        """Test feature lists from YAML config."""
        numerical = Config.get("features.numerical", [])
        categorical = Config.get("features.categorical", [])
        assert isinstance(numerical, list)
        assert isinstance(categorical, list)
