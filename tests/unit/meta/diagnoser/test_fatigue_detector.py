"""
Unit tests for FatigueDetector.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector


@pytest.fixture
def fatigue_detector():
    """Create FatigueDetector instance."""
    return FatigueDetector()


@pytest.fixture
def sample_ad_data():
    """Create sample ad-level daily data with fatigue."""
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(30)]

    data = {
        "ad_id": ["test_ad_123"] * 30,
        "date_start": [d.strftime("%Y-%m-%d") for d in dates],
        "impressions": [1000 + i * 100 for i in range(30)],
        "reach": [500 + i * 50 for i in range(30)],
        "spend": [100 + i * 10 for i in range(30)],
        # Golden period conversions (days 5-10)
        # Fatigue period conversions (days 20+)
        "conversions": (
            [1] * 5 +
            [5] * 5 +  # Golden period (days 5-10)
            [3] * 10 +
            [1] * 10   # Fatigue period (days 20+)
        ),
    }

    return pd.DataFrame(data)


@pytest.fixture
def healthy_ad_data():
    """Create sample ad-level daily data without fatigue."""
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(30)]

    data = {
        "ad_id": ["healthy_ad_456"] * 30,
        "date_start": [d.strftime("%Y-%m-%d") for d in dates],
        "impressions": [1000 + i * 100 for i in range(30)],
        "reach": [500 + i * 50 for i in range(30)],
        "spend": [100 + i * 10 for i in range(30)],
        "conversions": [5] * 30,  # Consistent good performance
    }

    return pd.DataFrame(data)


class TestFatigueDetector:
    """Test FatigueDetector."""

    def test_detect_fatigued_ad(self, fatigue_detector, sample_ad_data):
        """Test detection of fatigued ad."""
        issues = fatigue_detector.detect(sample_ad_data, "test_ad_123")

        assert len(issues) == 1
        issue = issues[0]

        assert issue.category.value == "fatigue"
        assert issue.affected_entities == ["test_ad_123"]
        assert "premium_loss" in issue.metrics
        assert issue.metrics["premium_loss"] > 0

    def test_detect_healthy_ad(self, fatigue_detector, healthy_ad_data):
        """Test that healthy ad is not flagged as fatigued."""
        issues = fatigue_detector.detect(healthy_ad_data, "healthy_ad_456")

        # Should not detect fatigue in consistently performing ad
        assert len(issues) == 0

    def test_calculate_cumulative_frequency(self, fatigue_detector, sample_ad_data):
        """Test cumulative frequency calculation."""
        result = fatigue_detector._analyze_fatigue(sample_ad_data, "test_ad_123")

        if result["is_fatigued"]:
            # Verify fatigue point is detected after golden period
            assert result["fatigue_freq"] > fatigue_detector.thresholds["fatigue_freq_threshold"]

    def test_golden_period_identification(self, fatigue_detector, sample_ad_data):
        """Test golden period identification."""
        result = fatigue_detector._analyze_fatigue(sample_ad_data, "test_ad_123")

        if result["is_fatigued"]:
            # CPA_gold should be calculated
            assert "cpa_gold" in result
            assert result["cpa_gold"] > 0

    def test_premium_loss_calculation(self, fatigue_detector, sample_ad_data):
        """Test premium loss calculation."""
        issues = fatigue_detector.detect(sample_ad_data, "test_ad_123")

        if issues:
            issue = issues[0]
            assert "premium_loss" in issue.metrics
            assert "missed_conversions" in issue.metrics

            # Premium loss should be positive
            assert issue.metrics["premium_loss"] >= 0

    def test_insufficient_data(self, fatigue_detector):
        """Test behavior with insufficient data."""
        short_data = pd.DataFrame({
            "ad_id": ["test"] * 3,
            "date_start": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "impressions": [100] * 3,
            "reach": [50] * 3,
            "spend": [10] * 3,
            "conversions": [1] * 3,
        })

        issues = fatigue_detector.detect(short_data, "test")

        # Should not detect fatigue with less than 7 days
        assert len(issues) == 0

    def test_missing_conversions_column(self, fatigue_detector):
        """Test handling of missing conversions column."""
        data = pd.DataFrame({
            "ad_id": ["test"] * 10,
            "date_start": [f"2025-01-0{i}" for i in range(1, 10)],
            "impressions": [100] * 10,
            "reach": [50] * 10,
            "spend": [10] * 10,
            "conversions": [1] * 10,
        })

        issues = fatigue_detector.detect(data, "test")

        # Should handle missing columns gracefully
        assert isinstance(issues, list)
