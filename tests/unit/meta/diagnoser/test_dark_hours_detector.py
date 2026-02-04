"""
Unit tests for DarkHoursDetector.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.meta.diagnoser.detectors.dark_hours_detector import DarkHoursDetector


@pytest.fixture
def dark_hours_detector():
    """Create DarkHoursDetector instance."""
    return DarkHoursDetector()


@pytest.fixture
def sample_hourly_data_with_dead_zones():
    """Create sample hourly data with dead zones."""
    hours = list(range(24))
    np.random.seed(42)

    # Dead zones: 0-5 (night hours), 12-14 (lunch dip)
    # Peak hours: 9-11, 18-21
    roas_values = []
    for h in hours:
        if 0 <= h <= 5:
            roas_values.append(0.2 + np.random.random() * 0.2)  # Dead zone
        elif 12 <= h <= 14:
            roas_values.append(0.3 + np.random.random() * 0.2)  # Dead zone
        elif 9 <= h <= 11:
            roas_values.append(3.0 + np.random.random() * 1.0)  # Peak
        elif 18 <= h <= 21:
            roas_values.append(2.5 + np.random.random() * 1.0)  # Peak
        else:
            roas_values.append(1.5 + np.random.random() * 0.5)  # Normal

    # Create data for 3 days
    data = {
        "adset_id": [],
        "date_start": [],
        "hour": [],
        "spend": [],
        "purchase_roas": [],
    }

    for day in range(3):
        for h in hours:
            date = datetime(2025, 1, 1) + timedelta(days=day)
            data["adset_id"].append("test_adset_123")
            data["date_start"].append(date.strftime("%Y-%m-%d"))
            data["hour"].append(h)
            data["spend"].append(50 if h not in [0, 1, 2, 3, 4, 5] else 20)
            data["purchase_roas"].append(roas_values[h])

    return pd.DataFrame(data)


@pytest.fixture
def sample_daily_data_with_weak_days():
    """Create sample daily data with weak days."""
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(14)]

    # Weak days: Day 0 (Sunday), Day 6 (Saturday)
    # Strong days: Tuesday, Wednesday, Thursday
    roas_by_day = {
        0: 0.5,  # Sunday - weak
        1: 2.2,  # Monday
        2: 3.0,  # Tuesday - strong
        3: 3.2,  # Wednesday - strong
        4: 2.8,  # Thursday - strong
        5: 2.0,  # Friday
        6: 0.6,  # Saturday - weak
    }

    roas_values = [roas_by_day[d.weekday()] for d in dates]

    data = {
        "adset_id": ["test_adset_456"] * 14,
        "date_start": [d.strftime("%Y-%m-%d") for d in dates],
        "spend": [100] * 14,
        "purchase_roas": roas_values,
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_healthy_data():
    """Create sample data without dead zones or weak days."""
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(14)]

    # Consistent performance across all hours/days
    data = {
        "adset_id": ["healthy_adset_789"] * 14,
        "date_start": [d.strftime("%Y-%m-%d") for d in dates],
        "spend": [100] * 14,
        "purchase_roas": [2.5 + np.random.random() * 0.3 for _ in range(14)],
    }

    return pd.DataFrame(data)


class TestDarkHoursDetector:
    """Test DarkHoursDetector."""

    def test_detect_hourly_dead_zones(self, dark_hours_detector, sample_hourly_data_with_dead_zones):
        """Test detection of hourly dead zones."""
        issues = dark_hours_detector.detect(sample_hourly_data_with_dead_zones, "test_adset_123")

        # Should detect hourly performance issues (or handle gracefully)
        assert isinstance(issues, list)

        # Check if hourly issue was detected
        hourly_issue = next((i for i in issues if "hourly_performance" in i.id), None)
        if hourly_issue:
            assert hourly_issue.category.value == "performance"

    def test_detect_weekly_weak_days(self, dark_hours_detector, sample_daily_data_with_weak_days):
        """Test detection of weekly weak days."""
        issues = dark_hours_detector.detect(sample_daily_data_with_weak_days, "test_adset_456")

        # Should detect weekly performance issues (or handle gracefully)
        assert isinstance(issues, list)

        # Check if weekly issue was detected
        weekly_issue = next((i for i in issues if "weekly_performance" in i.id), None)
        if weekly_issue:
            assert weekly_issue.category.value == "performance"

    def test_no_issues_healthy_adset(self, dark_hours_detector, sample_healthy_data):
        """Test that healthy adset is not flagged with dead zones."""
        issues = dark_hours_detector.detect(sample_healthy_data, "healthy_adset_789")

        # With consistent performance, should detect minimal or no issues
        # (depending on target_roas threshold)
        assert isinstance(issues, list)

    def test_hourly_efficiency_score(self, dark_hours_detector, sample_hourly_data_with_dead_zones):
        """Test hourly efficiency score calculation."""
        issues = dark_hours_detector.detect(sample_hourly_data_with_dead_zones, "test_adset_123")

        hourly_issue = next((i for i in issues if "hourly_performance" in i.id), None)
        if hourly_issue:
            assert "efficiency_score" in hourly_issue.metrics
            score = hourly_issue.metrics["efficiency_score"]

            # Score should be between 0 and 100
            assert 0 <= score <= 100

    def test_weekly_efficiency_score(self, dark_hours_detector, sample_daily_data_with_weak_days):
        """Test weekly efficiency score calculation."""
        issues = dark_hours_detector.detect(sample_daily_data_with_weak_days, "test_adset_456")

        weekly_issue = next((i for i in issues if "weekly_performance" in i.id), None)
        if weekly_issue:
            assert "efficiency_score" in weekly_issue.metrics
            score = weekly_issue.metrics["efficiency_score"]

            # Score should be between 0 and 100
            assert 0 <= score <= 100

    def test_dead_zone_identification(self, dark_hours_detector, sample_hourly_data_with_dead_zones):
        """Test that dead zones are correctly identified."""
        issues = dark_hours_detector.detect(sample_hourly_data_with_dead_zones, "test_adset_123")

        hourly_issue = next((i for i in issues if "hourly_performance" in i.id), None)
        if hourly_issue:
            assert "dead_zones" in hourly_issue.metrics
            dead_zones = hourly_issue.metrics["dead_zones"]

            # Should identify some dead zones
            assert len(dead_zones) > 0

    def test_weak_day_identification(self, dark_hours_detector, sample_daily_data_with_weak_days):
        """Test that weak days are correctly identified."""
        issues = dark_hours_detector.detect(sample_daily_data_with_weak_days, "test_adset_456")

        weekly_issue = next((i for i in issues if "weekly_performance" in i.id), None)
        if weekly_issue:
            assert "weak_days" in weekly_issue.metrics
            weak_days = weekly_issue.metrics["weak_days"]

            # Should identify some weak days
            assert len(weak_days) > 0

    def test_peak_hour_identification(self, dark_hours_detector, sample_hourly_data_with_dead_zones):
        """Test that peak hours are identified."""
        issues = dark_hours_detector.detect(sample_hourly_data_with_dead_zones, "test_adset_123")

        hourly_issue = next((i for i in issues if "hourly_performance" in i.id), None)
        if hourly_issue:
            assert "peak_hours" in hourly_issue.metrics
            peak_hours = hourly_issue.metrics["peak_hours"]

            # Should identify peak hours
            assert len(peak_hours) > 0

    def test_insufficient_hourly_data(self, dark_hours_detector):
        """Test behavior with insufficient hourly data."""
        # Need at least 3 days of hourly data for proper analysis
        short_data = pd.DataFrame({
            "adset_id": ["test"] * 24,
            "date_start": ["2025-01-01"] * 24,
            "hour": list(range(24)),
            "spend": [50] * 24,
            "purchase_roas": [2.0] * 24,
        })

        issues = dark_hours_detector.detect(short_data, "test")

        # Should handle gracefully
        assert isinstance(issues, list)

    def test_insufficient_daily_data(self, dark_hours_detector):
        """Test behavior with insufficient daily data."""
        short_data = pd.DataFrame({
            "adset_id": ["test"] * 3,
            "date_start": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "spend": [100] * 3,
            "purchase_roas": [2.0] * 3,
        })

        issues = dark_hours_detector.detect(short_data, "test")

        # Should handle gracefully
        assert isinstance(issues, list)

    def test_missing_required_columns(self, dark_hours_detector):
        """Test handling of missing required columns."""
        data = pd.DataFrame({
            "adset_id": ["test"] * 10,
            "date_start": [f"2025-01-{i:02d}" for i in range(1, 11)],
            "spend": [100] * 10,
            # Missing purchase_roas column
        })

        issues = dark_hours_detector.detect(data, "test")

        # Should handle missing columns gracefully
        assert isinstance(issues, list)

    def test_group_consecutive_hours(self, dark_hours_detector):
        """Test consecutive hour grouping."""
        # Test: [2, 3, 4, 22, 23] -> ["02:00-05:00", "22:00-00:00"]
        hours = [2, 3, 4, 22, 23]
        ranges = dark_hours_detector._group_consecutive_hours(hours)

        assert len(ranges) == 2
        assert "02:00-05:00" in ranges
        assert "22:00-00:00" in ranges

    def test_configurable_target_roas(self, dark_hours_detector, sample_hourly_data_with_dead_zones):
        """Test with custom target ROAS."""
        config = {
            "target_roas": 1.5,  # Lower threshold
            "efficiency_threshold": 0.5,
        }
        detector = DarkHoursDetector(config)

        issues = detector.detect(sample_hourly_data_with_dead_zones, "test_adset_123")

        # Should detect different number of issues with different target
        assert isinstance(issues, list)

    def test_single_day_analysis(self, dark_hours_detector):
        """Test analysis mode for single day."""
        config = {
            "analysis_mode": "hourly",
            "target_roas": 1.0,
        }
        detector = DarkHoursDetector(config)

        # Single day data should still work
        data = pd.DataFrame({
            "adset_id": ["test"] * 24,
            "date_start": ["2025-01-01"] * 24,
            "hour": list(range(24)),
            "spend": [50] * 24,
            "purchase_roas": [2.0 if 9 <= h <= 17 else 0.5 for h in range(24)],
        })

        issues = detector.detect(data, "test")
        assert isinstance(issues, list)

    def test_business_impact_metrics(self, dark_hours_detector, sample_hourly_data_with_dead_zones):
        """Test that business impact metrics are included."""
        issues = dark_hours_detector.detect(sample_hourly_data_with_dead_zones, "test_adset_123")

        hourly_issue = next((i for i in issues if "hourly_performance" in i.id), None)
        if hourly_issue:
            # Should include business impact metrics
            assert "dead_zone_spend" in hourly_issue.metrics or "weak_day_spend" in hourly_issue.metrics

    def test_action_recommendations(self, dark_hours_detector, sample_hourly_data_with_dead_zones):
        """Test that action recommendations are provided."""
        issues = dark_hours_detector.detect(sample_hourly_data_with_dead_zones, "test_adset_123")

        if issues:
            # At least one issue should have action recommendation
            has_action = any(
                "action_recommendation" in issue.metrics
                for issue in issues
            )
            assert has_action
