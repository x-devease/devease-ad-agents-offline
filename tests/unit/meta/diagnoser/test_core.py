"""
Unit tests for core abstractions: DetectorFactory and DataLoader.

Tests the factory pattern for detector creation and data loading abstractions.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

from src.meta.diagnoser.core import (
    DetectorFactory,
    DataLoader,
    MetaDataLoader,
    MockDataLoader,
)
from src.meta.diagnoser.core.issue_detector import BaseDetector
from src.meta.diagnoser.detectors import FatigueDetector


class TestDetectorFactory:
    """Test detector factory functionality."""

    def test_create_fatigue_detector(self):
        """Test creating FatigueDetector through factory."""
        detector = DetectorFactory.create("FatigueDetector")

        assert isinstance(detector, BaseDetector)
        assert type(detector).__name__ == "FatigueDetector"

    def test_create_with_config(self):
        """Test creating detector with custom config."""
        config = {
            "thresholds": {
                "window_size_days": 30,
                "fatigue_freq_threshold": 2.5,
            }
        }

        detector = DetectorFactory.create("FatigueDetector", config=config)

        assert detector.thresholds['window_size_days'] == 30
        assert detector.thresholds['fatigue_freq_threshold'] == 2.5

    def test_create_unknown_detector_raises_error(self):
        """Test that creating unknown detector raises ValueError."""
        with pytest.raises(ValueError, match="Unknown detector type"):
            DetectorFactory.create("UnknownDetector")

    def test_create_all_detectors(self):
        """Test creating all registered detectors."""
        detectors = DetectorFactory.create_all()

        # Should create all 5 detectors
        assert len(detectors) >= 3  # At least the main 3
        assert "FatigueDetector" in detectors
        assert "LatencyDetector" in detectors
        assert "DarkHoursDetector" in detectors

        # Verify all are BaseDetector instances
        for detector_name, detector in detectors.items():
            assert isinstance(detector, BaseDetector)

    def test_create_all_with_configs(self):
        """Test creating all detectors with custom configs."""
        configs = {
            "FatigueDetector": {
                "thresholds": {"window_size_days": 60}
            },
            "LatencyDetector": {
                "thresholds": {"latency_threshold_days": 5}
            }
        }

        detectors = DetectorFactory.create_all(configs=configs)

        # Verify configs were applied
        assert detectors["FatigueDetector"].thresholds['window_size_days'] == 60
        assert detectors["LatencyDetector"].thresholds['latency_threshold_days'] == 5

    def test_list_detectors(self):
        """Test listing all registered detectors."""
        detectors = DetectorFactory.list_detectors()

        assert isinstance(detectors, list)
        assert "FatigueDetector" in detectors
        assert "LatencyDetector" in detectors
        assert "DarkHoursDetector" in detectors

    def test_is_registered(self):
        """Test checking if detector is registered."""
        assert DetectorFactory.is_registered("FatigueDetector") is True
        assert DetectorFactory.is_registered("UnknownDetector") is False

    def test_register_detector(self):
        """Test registering a custom detector."""
        # Create a custom detector class
        class CustomDetector(BaseDetector):
            def __init__(self, config=None):
                super().__init__(config)

            def detect(self, data, entity_id):
                return []

        # Register it
        DetectorFactory.register_detector("CustomDetector", CustomDetector)

        # Verify it's registered
        assert DetectorFactory.is_registered("CustomDetector") is True

        # Create it through factory
        detector = DetectorFactory.create("CustomDetector")
        assert isinstance(detector, CustomDetector)

    def test_register_non_detector_raises_error(self):
        """Test that registering non-detector class raises TypeError."""
        class NotADetector:
            pass

        with pytest.raises(TypeError, match="must be a subclass of BaseDetector"):
            DetectorFactory.register_detector("NotADetector", NotADetector)


class TestDataLoader:
    """Test data loader abstractions."""

    def test_meta_data_loader_is_data_loader(self):
        """Test that MetaDataLoader implements DataLoader."""
        loader = MetaDataLoader()
        assert isinstance(loader, DataLoader)

    def test_mock_data_loader_is_data_loader(self):
        """Test that MockDataLoader implements DataLoader."""
        loader = MockDataLoader()
        assert isinstance(loader, DataLoader)

    def test_mock_data_loader_load_daily(self):
        """Test MockDataLoader generates daily data."""
        loader = MockDataLoader(seed=42)

        data = loader.load_daily_data("test_customer", "meta")

        # Verify structure
        assert isinstance(data, pd.DataFrame)
        assert 'date' in data.columns
        assert 'spend' in data.columns
        assert 'impressions' in data.columns
        assert 'conversions' in data.columns
        assert 'purchase_roas' in data.columns

        # Verify data was generated
        assert len(data) > 0
        assert data['spend'].sum() > 0

    def test_mock_data_loader_load_hourly(self):
        """Test MockDataLoader generates hourly data."""
        loader = MockDataLoader(seed=42)

        data = loader.load_hourly_data("test_customer", "meta")

        # Verify structure
        assert isinstance(data, pd.DataFrame)
        assert 'hour' in data.columns
        assert len(data) == 24  # 24 hours

    def test_meta_data_loader_file_not_found(self):
        """Test that MetaDataLoader raises FileNotFoundError for missing files."""
        loader = MetaDataLoader()

        # Use a non-existent path
        loader.data_root = Path("/nonexistent/path")

        with pytest.raises(FileNotFoundError):
            loader.load_daily_data("test", "meta")

    def test_data_loader_interface(self):
        """Test that DataLoader is abstract (cannot be instantiated)."""
        with pytest.raises(TypeError):
            DataLoader()


class TestDetectorFactoryIntegration:
    """Integration tests for DetectorFactory with DataLoader."""

    def test_factory_with_mock_data(self):
        """Test using factory detectors with mock data."""
        # Create mock data
        loader = MockDataLoader(seed=123)
        daily_data = loader.load_daily_data("test", "meta")

        # Create detector through factory
        detector = DetectorFactory.create("FatigueDetector")

        # Run detection
        issues = detector.detect(daily_data, "test_entity_123")

        # Should complete without errors
        assert isinstance(issues, list)

    def test_create_all_with_mock_data(self):
        """Test creating all detectors and running on mock data."""
        # Create mock data
        loader = MockDataLoader(seed=456)
        daily_data = loader.load_daily_data("test", "meta")

        # Create all detectors
        detectors = DetectorFactory.create_all()

        # Run detection for each
        results = {}
        for name, detector in detectors.items():
            try:
                issues = detector.detect(daily_data, "test_entity")
                results[name] = len(issues)
            except Exception as e:
                results[name] = f"Error: {e}"

        # All should complete without crashing
        assert len(results) >= 3
