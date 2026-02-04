"""
Unit tests for diagnoser utility modules.

Tests the shared utility functions used across evaluation and optimization scripts.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from src.meta.diagnoser.scripts.utils import (
    data_loader,
    sliding_windows,
    results_aggregator,
    metrics_utils,
)


class TestDataLoader:
    """Test data loading utilities."""

    def test_preprocess_daily_data(self):
        """Test daily data preprocessing."""
        # Create sample data
        data = pd.DataFrame({
            'date_start': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'spend': ['100.5', '200.3', 'invalid'],
            'impressions': [1000, 2000, 3000],
            'purchase_roas': ['[{"value": "2.5"}]', 'invalid', '[]'],
        })

        result = data_loader.preprocess_daily_data(data)

        # Verify conversions
        assert pd.api.types.is_numeric_dtype(result['spend'])
        assert result['spend'].iloc[2] == 0  # 'invalid' becomes 0
        assert result['purchase_roas'].iloc[0] == 2.5
        assert result['purchase_roas'].iloc[1] == 0.0  # 'invalid' becomes 0

    def test_preprocess_hourly_data(self):
        """Test hourly data preprocessing."""
        data = pd.DataFrame({
            'date_start': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
            'spend': ['50.0', '60.0'],
            'purchase_roas': ['[{"value": "1.5"}]', '[{"value": "2.0"}]'],
        })

        result = data_loader.preprocess_hourly_data(data)

        assert result['purchase_roas'].iloc[0] == 1.5
        assert result['purchase_roas'].iloc[1] == 2.0


class TestSlidingWindows:
    """Test sliding window generation utilities."""

    def setup_method(self):
        """Create sample data for testing."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)]
        self.data = pd.DataFrame({
            'date': dates,
            'date_start': [d.strftime('%Y-%m-%d') for d in dates],
            'spend': np.random.uniform(100, 500, 40),
        })

    def test_generate_sliding_windows_daily(self):
        """Test daily window generation."""
        windows = sliding_windows.generate_sliding_windows_daily(
            self.data,
            window_size_days=10,
            step_days=5,
            max_windows=5
        )

        # Should generate 5 windows
        assert len(windows) == 5

        # Verify window structure
        window = windows[0]
        assert 'window_num' in window
        assert 'start_date' in window
        assert 'end_date' in window
        assert 'data' in window
        assert len(window['data']) == 10

    def test_generate_sliding_windows_insufficient_data(self):
        """Test with insufficient data."""
        small_data = self.data.head(5)  # Only 5 days

        windows = sliding_windows.generate_sliding_windows_daily(
            small_data,
            window_size_days=10,
            step_days=5,
            max_windows=5
        )

        # Should return empty list
        assert len(windows) == 0

    def test_generate_sliding_windows_max_limit(self):
        """Test max_windows limit."""
        windows = sliding_windows.generate_sliding_windows_daily(
            self.data,
            window_size_days=5,
            step_days=3,
            max_windows=3  # Limit to 3
        )

        # Should respect max_windows limit
        assert len(windows) <= 3


class TestResultsAggregator:
    """Test results aggregation utilities."""

    def setup_method(self):
        """Create mock evaluation results."""
        # Create mock result objects
        class MockAccuracy:
            def __init__(self, tp, fp, fn):
                self.true_positives = tp
                self.false_positives = fp
                self.false_negatives = fn
                self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0

        class MockResult:
            def __init__(self, tp, fp, fn, score):
                self.accuracy = MockAccuracy(tp, fp, fn)
                self.overall_score = score
                self.details = {}

        self.results = [
            MockResult(tp=80, fp=10, fn=20, score=85.0),
            MockResult(tp=90, fp=5, fn=15, score=88.0),
            MockResult(tp=70, fp=15, fn=25, score=82.0),
        ]

    def test_aggregate_results(self):
        """Test results aggregation."""
        aggregation = results_aggregator.aggregate_results(
            self.results,
            "FatigueDetector"
        )

        # Verify aggregated values
        assert aggregation['detector_name'] == "FatigueDetector"
        assert aggregation['total_windows'] == 3

        accuracy = aggregation['accuracy']
        assert accuracy['total_tp'] == 240  # 80 + 90 + 70
        assert accuracy['total_fp'] == 30   # 10 + 5 + 15
        assert accuracy['total_fn'] == 60   # 20 + 15 + 25

        # Verify metrics are calculated correctly
        expected_precision = 240 / (240 + 30)
        assert abs(accuracy['precision'] - expected_precision) < 0.001

        expected_recall = 240 / (240 + 60)
        assert abs(accuracy['recall'] - expected_recall) < 0.001

    def test_calculate_metrics_summary(self):
        """Test metrics summary calculation."""
        aggregation = results_aggregator.aggregate_results(
            self.results,
            "FatigueDetector"
        )

        summary = results_aggregator.calculate_metrics_summary(aggregation)

        # Verify summary contains all expected keys
        assert 'detector_name' in summary
        assert 'f1_score' in summary
        assert 'precision' in summary
        assert 'recall' in summary
        assert 'avg_score' in summary
        assert 'min_score' in summary
        assert 'max_score' in summary

        # Verify score statistics
        assert summary['min_score'] == 82.0
        assert summary['max_score'] == 88.0
        assert summary['avg_score'] == (85.0 + 88.0 + 82.0) / 3

    def test_aggregate_results_empty(self):
        """Test aggregation with empty results."""
        result = results_aggregator.aggregate_results([], "FatigueDetector")

        assert result is None


class TestMetricsUtils:
    """Test metrics utilities."""

    def test_check_satisfaction(self):
        """Test satisfaction checking."""
        metrics = {
            "precision": 0.75,
            "recall": 0.85,
            "f1_score": 0.80,
        }

        targets = {
            "precision": 0.70,
            "recall": 0.80,
            "f1_score": 0.75,
        }

        all_satisfied, satisfaction = metrics_utils.check_satisfaction(metrics, targets)

        # All metrics should be satisfied
        assert all_satisfied is True

        # Check satisfaction details
        assert satisfaction['precision']['satisfied'] is True
        assert satisfaction['recall']['satisfied'] is True
        assert satisfaction['f1_score']['satisfied'] is True

    def test_check_satisfaction_not_met(self):
        """Test satisfaction when targets not met."""
        metrics = {
            "precision": 0.65,  # Below target
            "recall": 0.85,
        }

        targets = {
            "precision": 0.70,
            "recall": 0.80,
        }

        all_satisfied, satisfaction = metrics_utils.check_satisfaction(metrics, targets)

        # Should not be all satisfied
        assert all_satisfied is False
        assert satisfaction['precision']['satisfied'] is False
        assert satisfaction['recall']['satisfied'] is True

    def test_format_metrics_report(self):
        """Test metrics report formatting."""
        metrics = {
            "precision": 0.75,
            "recall": 0.85,
            "f1_score": 0.80,
            "tp": 100,
            "fp": 20,
            "fn": 30,
        }

        targets = {
            "f1_score": 0.75,
        }

        report = metrics_utils.format_metrics_report("TestDetector", metrics, targets)

        # Verify report contains key information
        assert "TestDetector" in report
        assert "0.7500" in report  # precision
        assert "0.8500" in report  # recall
        assert "0.8000" in report  # f1_score

    def test_load_metrics_from_file(self):
        """Test loading metrics from JSON file."""
        # Create a temporary report file
        report_data = {
            "aggregated_metrics": {
                "precision": 0.85,
                "recall": 0.90,
                "f1_score": 0.87,
                "total_tp": 150,
                "total_fp": 25,
                "total_fn": 15,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "test_report.json"

            with open(report_path, 'w') as f:
                json.dump(report_data, f)

            # Load metrics with custom detector map
            detector_map = {"DarkHoursDetector": "test_report.json"}
            metrics = metrics_utils.load_metrics(
                "DarkHoursDetector",
                report_path=Path(tmpdir),
                detector_map=detector_map
            )

            # Verify loaded metrics
            assert metrics['precision'] == 0.85
            assert metrics['recall'] == 0.90
            assert metrics['f1_score'] == 0.87

    def test_load_metrics_old_format(self):
        """Test loading metrics from old format (with 'accuracy' key)."""
        report_data = {
            "accuracy": {
                "precision": 0.80,
                "recall": 0.75,
                "f1_score": 0.77,
                "total_tp": 120,
                "total_fp": 30,
                "total_fn": 40,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "test_report_old.json"

            with open(report_path, 'w') as f:
                json.dump(report_data, f)

            # Load metrics with custom detector map
            detector_map = {"FatigueDetector": "test_report_old.json"}
            metrics = metrics_utils.load_metrics(
                "FatigueDetector",
                report_path=Path(tmpdir),
                detector_map=detector_map
            )

            # Verify loaded metrics
            assert metrics['precision'] == 0.80
            assert metrics['recall'] == 0.75
            assert metrics['f1_score'] == 0.77
