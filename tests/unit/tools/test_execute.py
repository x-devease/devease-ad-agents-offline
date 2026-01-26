"""
Unit tests for execute.py tool.

Tests for the rule-based budget allocation CLI tool.
Most of execute.py consists of thin wrappers around the core rules allocator,
so these tests focus on the utility functions that have extractable logic.

Key testable functions:
- _extract_metrics_from_row: Extract metrics from DataFrame row
- _process_results: Process and scale allocation results
- _print_summary: Summary statistics calculations
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
from argparse import Namespace


class TestExtractMetricsFromRow:
    """Test _extract_metrics_from_row function"""

    def test_extract_basic_metrics(self):
        """Test extraction of basic metrics"""
        from src.meta.adset.allocator.cli.commands.execute import _extract_metrics_from_row

        row = pd.Series(
            {
                "purchase_roas_rolling_7d": 2.5,
                "roas_trend": 0.1,
                "spend": 100.0,
                "impressions": 1000,
                "clicks": 50,
                "health_score": 0.8,
                "days_since_start": 10,
                "day_of_week": 3,
                "is_weekend": False,
                "week_of_year": 45,
            }
        )

        metrics = _extract_metrics_from_row(
            row=row,
            adset_id="test_adset",
            current_budget=50.0,
            total_budget_today=1000.0,
        )

        assert metrics["adset_id"] == "test_adset"
        assert metrics["current_budget"] == 50.0
        assert metrics["roas_7d"] == pytest.approx(2.5)
        assert metrics["roas_trend"] == pytest.approx(0.1)
        assert metrics["spend"] == pytest.approx(100.0)
        assert metrics["impressions"] == 1000
        assert metrics["clicks"] == 50
        assert metrics["health_score"] == pytest.approx(0.8)
        assert metrics["days_active"] == 10
        assert metrics["day_of_week"] == 3
        assert metrics["is_weekend"] is False
        assert metrics["week_of_year"] == 45
        assert metrics["total_budget_today"] == 1000.0

    def test_extract_metrics_with_nan_values(self):
        """Test extraction with NaN values"""
        from src.meta.adset.allocator.cli.commands.execute import _extract_metrics_from_row

        row = pd.Series(
            {
                "purchase_roas_rolling_7d": None,
                "roas_trend": None,
                "spend": 100.0,
                "impressions": 1000,
                "clicks": 50,
                "health_score": None,
                "days_since_start": 0,
            }
        )

        metrics = _extract_metrics_from_row(
            row=row,
            adset_id="test_adset",
            current_budget=50.0,
            total_budget_today=1000.0,
        )

        # NaN values should be converted to defaults
        assert metrics["roas_7d"] == 0.0
        assert metrics["roas_trend"] == 0.0
        assert metrics["health_score"] == 0.5  # Default health score
        assert metrics["days_active"] == 0

    def test_extract_metrics_with_roas_comparisons(self):
        """Test extraction of ROAS comparison metrics"""
        from src.meta.adset.allocator.cli.commands.execute import _extract_metrics_from_row

        row = pd.Series(
            {
                "spend": 100.0,
                "adset_roas": 2.5,
                "campaign_roas": 2.8,
                "account_roas": 3.0,
            }
        )

        metrics = _extract_metrics_from_row(
            row=row,
            adset_id="test_adset",
            current_budget=50.0,
            total_budget_today=1000.0,
        )

        assert metrics["adset_roas"] == pytest.approx(2.5)
        assert metrics["campaign_roas"] == pytest.approx(2.8)
        assert metrics["account_roas"] == pytest.approx(3.0)

    def test_extract_metrics_with_efficiency(self):
        """Test extraction of efficiency metrics"""
        from src.meta.adset.allocator.cli.commands.execute import _extract_metrics_from_row

        row = pd.Series(
            {
                "spend": 100.0,
                "efficiency": 0.85,
                "revenue_per_impression": 0.15,
                "revenue_per_click": 3.0,
            }
        )

        metrics = _extract_metrics_from_row(
            row=row,
            adset_id="test_adset",
            current_budget=50.0,
            total_budget_today=1000.0,
        )

        assert metrics["efficiency"] == pytest.approx(0.85)
        assert metrics["revenue_per_impression"] == pytest.approx(0.15)
        assert metrics["revenue_per_click"] == pytest.approx(3.0)

    def test_extract_metrics_with_budget_utilization(self):
        """Test extraction of budget utilization metrics"""
        from src.meta.adset.allocator.cli.commands.execute import _extract_metrics_from_row

        row = pd.Series(
            {
                "spend": 100.0,
                "budget_utilization": 0.92,
                "adaptive_target_roas": 2.2,
                "static_target_roas": 2.0,
                "marginal_roas": 3.0,
            }
        )

        metrics = _extract_metrics_from_row(
            row=row,
            adset_id="test_adset",
            current_budget=50.0,
            total_budget_today=1000.0,
        )

        assert metrics["budget_utilization"] == pytest.approx(0.92)
        assert metrics["adaptive_target_roas"] == pytest.approx(2.2)
        assert metrics["static_target_roas"] == pytest.approx(2.0)
        assert metrics["marginal_roas"] == pytest.approx(3.0)

    def test_extract_metrics_with_ad_level_features(self):
        """Test extraction of ad-level statistical features"""
        from src.meta.adset.allocator.cli.commands.execute import _extract_metrics_from_row

        row = pd.Series(
            {
                "spend": 100.0,
                "num_ads": 5,
                "num_active_ads": 3,
                "ad_diversity": 4,
                "ad_roas_mean": 2.5,
                "ad_roas_std": 0.5,
                "ad_roas_range": 2.0,
                "ad_spend_gini": 0.3,
                "top_ad_spend_pct": 0.6,
                "video_ads_ratio": 0.4,
                "format_diversity_score": 3,
            }
        )

        metrics = _extract_metrics_from_row(
            row=row,
            adset_id="test_adset",
            current_budget=50.0,
            total_budget_today=1000.0,
        )

        assert metrics["num_ads"] == 5
        assert metrics["num_active_ads"] == 3
        assert metrics["ad_diversity"] == 4
        assert metrics["ad_roas_mean"] == pytest.approx(2.5)
        assert metrics["ad_roas_std"] == pytest.approx(0.5)
        assert metrics["ad_roas_range"] == pytest.approx(2.0)
        assert metrics["ad_spend_gini"] == pytest.approx(0.3)
        assert metrics["top_ad_spend_pct"] == pytest.approx(0.6)
        assert metrics["video_ads_ratio"] == pytest.approx(0.4)
        assert metrics["format_diversity_score"] == 3

    def test_extract_metrics_defaults_for_missing_ad_features(self):
        """Test default values when ad-level features are missing"""
        from src.meta.adset.allocator.cli.commands.execute import _extract_metrics_from_row

        row = pd.Series({"spend": 100.0})

        metrics = _extract_metrics_from_row(
            row=row,
            adset_id="test_adset",
            current_budget=50.0,
            total_budget_today=1000.0,
        )

        # Should use sensible defaults
        assert metrics["num_ads"] == 1  # Default
        assert metrics["num_active_ads"] == 0
        assert metrics["ad_diversity"] == 1
        assert metrics["ad_roas_mean"] == pytest.approx(0.0)
        assert metrics["video_ads_ratio"] == pytest.approx(0.0)
        assert metrics["format_diversity_score"] == 1


class TestProcessResults:
    """Test _process_results function"""

    def test_process_results_basic_scaling(self):
        """Test basic budget scaling"""
        from src.meta.adset.allocator.cli.commands.execute import _process_results

        results = [
            {
                "adset_id": "adset1",
                "current_budget": 50.0,
                "new_budget": 100.0,
                "change_pct": 100.0,
                "roas_7d": 2.5,
                "health_score": 0.8,
                "days_active": 10,
                "decision_path": "increase -> high_roas",
            },
            {
                "adset_id": "adset2",
                "current_budget": 50.0,
                "new_budget": 100.0,
                "change_pct": 100.0,
                "roas_7d": 3.0,
                "health_score": 0.9,
                "days_active": 15,
                "decision_path": "increase -> very_high_roas",
            },
        ]

        total_budget_today = 300.0  # Sum of new_budget is 200, so scale by 1.5x

        result_df = _process_results(results, total_budget_today)

        assert len(result_df) == 2
        # Budgets should be scaled to match total_budget_today
        assert result_df["new_budget"].sum() == pytest.approx(total_budget_today)
        # Each budget should be scaled by 1.5x
        assert result_df["new_budget"].iloc[0] == pytest.approx(150.0)
        assert result_df["new_budget"].iloc[1] == pytest.approx(150.0)
        # Change percentage should be recalculated
        assert result_df["change_pct"].iloc[0] == pytest.approx(
            200.0
        )  # (150-50)/50 * 100

    def test_process_results_zero_total_budget(self):
        """Test processing when total new budget is zero"""
        from src.meta.adset.allocator.cli.commands.execute import _process_results

        results = [
            {
                "adset_id": "adset1",
                "current_budget": 50.0,
                "new_budget": 0.0,
                "change_pct": 0.0,
                "roas_7d": 0.5,
                "health_score": 0.2,
                "days_active": 5,
                "decision_path": "decrease -> low_roas",
            }
        ]

        total_budget_today = 100.0

        result_df = _process_results(results, total_budget_today)

        # Should not scale when total is zero
        assert result_df["new_budget"].iloc[0] == pytest.approx(0.0)

    def test_process_results_preserves_other_fields(self):
        """Test that other fields are preserved during processing"""
        from src.meta.adset.allocator.cli.commands.execute import _process_results

        results = [
            {
                "adset_id": "adset1",
                "current_budget": 50.0,
                "new_budget": 100.0,
                "change_pct": 100.0,
                "roas_7d": 2.5,
                "health_score": 0.8,
                "days_active": 10,
                "decision_path": "test_decision",
            }
        ]

        result_df = _process_results(results, 100.0)

        assert result_df["adset_id"].iloc[0] == "adset1"
        assert result_df["current_budget"].iloc[0] == pytest.approx(50.0)
        assert result_df["roas_7d"].iloc[0] == pytest.approx(2.5)
        assert result_df["health_score"].iloc[0] == pytest.approx(0.8)
        assert result_df["days_active"].iloc[0] == 10
        assert result_df["decision_path"].iloc[0] == "test_decision"


class TestLoadData:
    """Test _load_data function"""

    @patch("pandas.read_csv")
    def test_load_data_success(self, mock_read_csv):
        """Test successful data loading"""
        from src.meta.adset.allocator.cli.commands.execute import _load_data

        mock_df = pd.DataFrame(
            {"adset_id": ["adset1", "adset2"], "spend": [100.0, 200.0]}
        )
        mock_read_csv.return_value = mock_df

        result = _load_data("test.csv")

        assert result is not None
        assert len(result) == 2
        mock_read_csv.assert_called_once_with("test.csv")

    @patch("pandas.read_csv")
    @patch("src.meta.adset.allocator.cli.commands.execute.logger")
    def test_load_data_file_not_found(self, mock_logger, mock_read_csv):
        """Test loading with FileNotFoundError"""
        from src.meta.adset.allocator.cli.commands.execute import _load_data

        mock_read_csv.side_effect = FileNotFoundError("File not found")

        result = _load_data("missing.csv")

        assert result is None
        mock_logger.error.assert_called()

    @patch("pandas.read_csv")
    @patch("src.meta.adset.allocator.cli.commands.execute.logger")
    def test_load_data_empty_file(self, mock_logger, mock_read_csv):
        """Test loading with empty data error"""
        from src.meta.adset.allocator.cli.commands.execute import _load_data

        mock_read_csv.side_effect = pd.errors.EmptyDataError("Empty file")

        result = _load_data("empty.csv")

        assert result is None
        mock_logger.error.assert_called()


class TestProcessAdsets:
    """Test _process_adsets function"""

    @patch("src.meta.adset.allocator.cli.commands.execute._extract_metrics_from_row")
    def test_process_adsets_basic(self, mock_extract):
        """Test basic adset processing"""
        from src.meta.adset.allocator.cli.commands.execute import _process_adsets

        # Mock the extract function
        mock_extract.return_value = {
            "adset_id": "adset1",
            "current_budget": 50.0,
            "roas_7d": 2.5,
            "health_score": 0.8,
            "days_active": 10,
        }

        # Create mock allocator
        mock_allocator = Mock()
        mock_allocator.allocate_budget.return_value = (100.0, ["increase"])

        # Create test data
        adset_groups = pd.DataFrame(
            {"adset_id": ["adset1"], "spend": [100.0]}
        ).set_index("adset_id")

        total_budget_today = 1000.0

        results = _process_adsets(mock_allocator, adset_groups, total_budget_today)

        assert len(results) == 1
        assert results[0]["adset_id"] == "adset1"
        assert results[0]["new_budget"] == pytest.approx(100.0)
        assert results[0]["decision_path"] == "increase"

    @patch("src.meta.adset.allocator.cli.commands.execute._extract_metrics_from_row")
    def test_process_adsets_calculates_change_pct(self, mock_extract):
        """Test that change percentage is calculated correctly"""
        from src.meta.adset.allocator.cli.commands.execute import _process_adsets

        mock_extract.return_value = {
            "adset_id": "adset1",
            "current_budget": 100.0,  # Changed from 50.0
            "roas_7d": 2.5,
            "health_score": 0.8,
            "days_active": 10,
        }

        mock_allocator = Mock()
        mock_allocator.allocate_budget.return_value = (
            75.0,
            ["decrease"],
        )  # Changed from increase

        adset_groups = pd.DataFrame(
            {"adset_id": ["adset1"], "spend": [100.0]}
        ).set_index("adset_id")

        results = _process_adsets(mock_allocator, adset_groups, 1000.0)

        # Change should be (75 - 100) / 100 * 100 = -25%
        assert results[0]["change_pct"] == pytest.approx(-25.0)


class TestIntegration:
    """Integration tests for execute.py utilities"""

    def test_extract_and_process_pipeline(self):
        """Test extracting metrics and processing results"""
        from src.meta.adset.allocator.cli.commands.execute import _extract_metrics_from_row, _process_results

        # Create test data
        row = pd.Series(
            {
                "purchase_roas_rolling_7d": 2.5,
                "spend": 100.0,
                "impressions": 1000,
                "clicks": 50,
                "health_score": 0.8,
                "days_since_start": 10,
            }
        )

        # Extract metrics
        metrics = _extract_metrics_from_row(
            row=row,
            adset_id="test_adset",
            current_budget=50.0,
            total_budget_today=1000.0,
        )

        # Simulate allocation result
        results = [
            {
                "adset_id": metrics["adset_id"],
                "current_budget": metrics["current_budget"],
                "new_budget": 75.0,  # Simulated allocation
                "change_pct": 50.0,
                "roas_7d": metrics["roas_7d"],
                "health_score": metrics["health_score"],
                "days_active": metrics["days_active"],
                "decision_path": "increase",
            }
        ]

        # Process results
        result_df = _process_results(results, 1000.0)

        assert len(result_df) == 1
        assert result_df["adset_id"].iloc[0] == "test_adset"
        assert result_df["current_budget"].iloc[0] == pytest.approx(50.0)
