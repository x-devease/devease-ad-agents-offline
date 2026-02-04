"""
Unit tests for LatencyDetector.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.meta.diagnoser.detectors.latency_detector import LatencyDetector, infer_status_changes


@pytest.fixture
def latency_detector():
    """Create LatencyDetector instance."""
    return LatencyDetector()


@pytest.fixture
def sample_adset_data_with_latency():
    """Create sample adset-level daily data with response latency."""
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(15)]

    # Simulate performance drop on day 7 with delayed response
    # Days 0-6: Good performance (ROAS > 1.0)
    # Days 7-10: Bad performance (ROAS < 1.0) - bleeding period
    # Days 11+: Response (PAUSED or recovery)
    roas_values = (
        [2.5, 2.8, 2.3, 2.6, 2.4, 2.7, 2.5] +  # Days 0-6: Good performance
        [0.3, 0.4, 0.5, 0.2] +                  # Days 7-10: Performance drop
        [2.2, 2.4, 2.6, 2.3]                    # Days 11-14: Recovery
    )

    spend_values = (
        [100] * 7 +      # Normal spend
        [150] * 4 +      # Still spending during bleed
        [0] * 4          # Paused (no spend)
    )

    data = {
        "adset_id": ["test_adset_123"] * 15,
        "date_start": [d.strftime("%Y-%m-%d") for d in dates],
        "spend": spend_values,
        "purchase_roas": roas_values,
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_adset_data_no_latency():
    """Create sample adset-level daily data without latency issues."""
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(15)]

    # Consistently good performance, no significant drops
    data = {
        "adset_id": ["healthy_adset_456"] * 15,
        "date_start": [d.strftime("%Y-%m-%d") for d in dates],
        "spend": [100] * 15,
        "purchase_roas": [2.5, 2.8, 2.3, 2.6, 2.4, 2.7, 2.5, 2.8, 2.3, 2.6, 2.4, 2.7, 2.5, 2.8, 2.3],
    }

    return pd.DataFrame(data)


@pytest.fixture
def status_changes_df():
    """Create sample status changes DataFrame."""
    return pd.DataFrame({
        "adset_id": ["test_adset_123"],
        "change_date": [datetime(2025, 1, 11)],
        "old_status": ["ACTIVE"],
        "new_status": ["PAUSED"],
    })


class TestLatencyDetector:
    """Test LatencyDetector."""

    def test_detect_latency_with_pause(self, latency_detector, sample_adset_data_with_latency, status_changes_df):
        """Test detection of latency with manual pause intervention."""
        issues = latency_detector.detect(sample_adset_data_with_latency, "test_adset_123", status_changes_df)

        assert len(issues) == 1
        issue = issues[0]

        assert issue.category.value == "performance"
        assert issue.affected_entities == ["test_adset_123"]
        assert "avg_delay_days" in issue.metrics
        assert "avg_responsiveness_score" in issue.metrics

        # Should detect some delay
        assert issue.metrics["avg_delay_days"] > 0

    def test_detect_latency_with_natural_recovery(self, latency_detector, sample_adset_data_with_latency):
        """Test detection of latency with natural recovery (no pause)."""
        issues = latency_detector.detect(sample_adset_data_with_latency, "test_adset_123", status_changes=None)

        assert len(issues) == 1
        issue = issues[0]

        # Should detect latency even without manual pause
        assert issue.metrics["avg_delay_days"] > 0
        assert issue.metrics["natural_recovery_interventions"] > 0

    def test_no_latency_healthy_adset(self, latency_detector, sample_adset_data_no_latency):
        """Test that healthy adset is not flagged with latency."""
        issues = latency_detector.detect(sample_adset_data_no_latency, "healthy_adset_456")

        # Should not detect latency in consistently performing adset
        assert len(issues) == 0

    def test_responsiveness_score_calculation(self, latency_detector, sample_adset_data_with_latency, status_changes_df):
        """Test responsiveness score calculation."""
        issues = latency_detector.detect(sample_adset_data_with_latency, "test_adset_123", status_changes_df)

        if issues:
            issue = issues[0]
            score = issue.metrics["avg_responsiveness_score"]

            # Score should be between 0 and 100
            assert 0 <= score <= 100

            # With ~4 days delay, score should be low (poor responsiveness)
            assert score <= 60

    def test_bleeding_spend_calculation(self, latency_detector, sample_adset_data_with_latency, status_changes_df):
        """Test bleeding spend calculation."""
        issues = latency_detector.detect(sample_adset_data_with_latency, "test_adset_123", status_changes_df)

        if issues:
            issue = issues[0]
            assert "total_bleeding_spend" in issue.metrics

            # Should have spent money during bleeding period
            assert issue.metrics["total_bleeding_spend"] > 0

    def test_insufficient_data(self, latency_detector):
        """Test behavior with insufficient data."""
        short_data = pd.DataFrame({
            "adset_id": ["test"] * 2,
            "date_start": ["2025-01-01", "2025-01-02"],
            "spend": [100] * 2,
            "purchase_roas": [2.5] * 2,
        })

        issues = latency_detector.detect(short_data, "test")

        # Should not detect latency with insufficient data
        assert len(issues) == 0

    def test_missing_required_columns(self, latency_detector):
        """Test handling of missing required columns."""
        data = pd.DataFrame({
            "adset_id": ["test"] * 10,
            "date_start": [f"2025-01-{i:02d}" for i in range(1, 11)],
            "spend": [100] * 10,
            # Missing purchase_roas column
        })

        issues = latency_detector.detect(data, "test")

        # Should handle missing columns gracefully
        assert isinstance(issues, list)
        assert len(issues) == 0

    def test_calculate_responsiveness_score(self, latency_detector):
        """Test responsiveness score calculation for different delay periods."""
        # 0-1 days = excellent (80-100 points)
        assert latency_detector._calculate_responsiveness_score(0) >= 80
        assert latency_detector._calculate_responsiveness_score(1) >= 80

        # 2 days = good (60-80 points)
        score_2_days = latency_detector._calculate_responsiveness_score(2)
        assert 60 <= score_2_days <= 80

        # 3-4 days = moderate (40-60 points)
        score_3_days = latency_detector._calculate_responsiveness_score(3)
        assert 40 <= score_3_days < 60

        # 5-7 days = poor (20-40 points)
        score_6_days = latency_detector._calculate_responsiveness_score(6)
        assert 20 <= score_6_days < 40

        # >7 days = critical (0-20 points)
        score_10_days = latency_detector._calculate_responsiveness_score(10)
        assert 0 <= score_10_days < 20

    def test_rolling_roas_calculation(self, latency_detector, sample_adset_data_with_latency):
        """Test that rolling ROAS is calculated correctly."""
        data = sample_adset_data_with_latency.copy()

        # The detector should calculate rolling ROAS internally
        # We're checking that it doesn't crash and handles the data
        issues = latency_detector.detect(data, "test_adset_123")
        assert isinstance(issues, list)


class TestInferStatusChanges:
    """Test status change inference function."""

    def test_infer_status_changes_basic(self):
        """Test basic status change inference."""
        data = pd.DataFrame({
            "adset_id": ["test"] * 5,
            "date_start": [f"2025-01-{i:02d}" for i in range(1, 6)],
            "adset_status": ["ACTIVE", "ACTIVE", "PAUSED", "PAUSED", "ACTIVE"],
        })

        changes = infer_status_changes(data)

        # Should detect 2 status changes
        assert len(changes) == 2

        # First change: ACTIVE -> PAUSED
        assert changes.iloc[0]["old_status"] == "ACTIVE"
        assert changes.iloc[0]["new_status"] == "PAUSED"

        # Second change: PAUSED -> ACTIVE
        assert changes.iloc[1]["old_status"] == "PAUSED"
        assert changes.iloc[1]["new_status"] == "ACTIVE"

    def test_infer_status_changes_multiple_adsets(self):
        """Test status change inference with multiple adsets."""
        data = pd.DataFrame({
            "adset_id": ["adset1", "adset1", "adset2", "adset2"],
            "date_start": ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"],
            "adset_status": ["ACTIVE", "PAUSED", "ACTIVE", "ACTIVE"],
        })

        changes = infer_status_changes(data)

        # Should detect 1 change for adset1, 0 for adset2
        assert len(changes) == 1
        assert changes.iloc[0]["adset_id"] == "adset1"

    def test_infer_status_changes_empty_data(self):
        """Test status change inference with empty data."""
        data = pd.DataFrame({
            "adset_id": [],
            "date_start": [],
            "adset_status": [],
        })

        changes = infer_status_changes(data)
        assert len(changes) == 0
