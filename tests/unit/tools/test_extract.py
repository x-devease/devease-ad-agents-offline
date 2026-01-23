"""
Unit tests for extract.py tool.

Tests utility functions for data aggregation, metrics calculation,
and feature engineering from ad-level to adset-level data.

Key testable functions:
- _build_agg_dict: Build aggregation dictionary from column lists
- _recalculate_roas_metrics: Recalculate ROAS-related metrics
- _calculate_rolling_metrics: Calculate rolling and EMA metrics
- _create_lagged_features: Create lagged features to prevent lookahead bias
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch


# Note: These tests focus on the utility functions in extract.py
# The main orchestration functions are tested via integration tests


class TestBuildAggDict:
    """Test _build_agg_dict function"""

    def test_basic_aggregation(self):
        """Test building basic aggregation dictionary"""
        from src.cli.commands.extract import _build_agg_dict

        sum_cols = ["spend", "impressions", "clicks"]
        mean_cols = ["ctr", "cpc"]
        first_cols = ["adset_id", "date_start"]
        other_cols = ["campaign_id"]
        groupby_cols = ["adset_id", "date_start"]

        result = _build_agg_dict(
            sum_cols, mean_cols, first_cols, other_cols, groupby_cols
        )

        assert result["spend"] == "sum"
        assert result["impressions"] == "sum"
        assert result["clicks"] == "sum"
        assert result["ctr"] == "mean"
        assert result["cpc"] == "mean"
        assert result["campaign_id"] == "first"
        # Note: groupby_cols ARE removed from the dict in the implementation
        assert "adset_id" not in result
        assert "date_start" not in result

    def test_empty_column_lists(self):
        """Test with empty column lists"""
        from src.cli.commands.extract import _build_agg_dict

        result = _build_agg_dict([], [], [], ["adset_id"], ["adset_id"])

        assert len(result) == 0

    def test_other_cols_default_to_first(self):
        """Test that other columns default to 'first' aggregation"""
        from src.cli.commands.extract import _build_agg_dict

        result = _build_agg_dict(
            sum_cols=["spend"],
            mean_cols=[],
            first_cols=["adset_id"],
            other_cols=["custom_col1", "custom_col2"],
            groupby_cols=["adset_id"],
        )

        assert result["custom_col1"] == "first"
        assert result["custom_col2"] == "first"


class TestRecalculateROASMetrics:
    """Test ROAS metric recalculation functions"""

    def test_recalculate_revenue_metrics(self):
        """Test revenue metric calculations"""
        from src.cli.commands.extract import _calculate_revenue_metrics

        df = pd.DataFrame(
            {
                "revenue": [100.0, 200.0, 300.0],
                "impressions": [1000, 2000, 3000],
                "clicks": [50, 100, 150],
            }
        )

        # Function modifies df in-place
        _calculate_revenue_metrics(df)

        assert "revenue_per_impression" in df.columns
        assert "revenue_per_click" in df.columns
        assert df["revenue_per_impression"].iloc[0] == pytest.approx(0.1)
        assert df["revenue_per_click"].iloc[0] == pytest.approx(2.0)

    def test_recalculate_revenue_metrics_handles_zero(self):
        """Test revenue metrics with zero impressions/clicks"""
        from src.cli.commands.extract import _calculate_revenue_metrics

        df = pd.DataFrame(
            {"revenue": [100.0, 200.0], "impressions": [0, 1000], "clicks": [100, 0]}
        )

        # Function modifies df in-place
        _calculate_revenue_metrics(df)

        # Should handle division by zero with NaN
        assert pd.isna(df["revenue_per_impression"].iloc[0])
        assert df["revenue_per_impression"].iloc[1] == pytest.approx(0.2)
        assert df["revenue_per_click"].iloc[0] == pytest.approx(1.0)
        assert pd.isna(df["revenue_per_click"].iloc[1])

    def test_calculate_roas_comparisons(self):
        """Test ROAS comparison metrics"""
        from src.cli.commands.extract import _calculate_roas_comparisons

        df = pd.DataFrame(
            {
                "purchase_roas": [3.0, 2.5, 4.0],
                "adset_roas": [2.5, 2.0, 3.5],
                "campaign_roas": [2.8, 2.2, 3.8],
                "account_roas": [3.0, 2.5, 4.0],
            }
        )

        # Function modifies df in-place
        _calculate_roas_comparisons(df)

        assert "roas_vs_adset" in df.columns
        assert "roas_vs_campaign" in df.columns
        assert "roas_vs_account" in df.columns

        # Check calculations
        assert df["roas_vs_adset"].iloc[0] == pytest.approx(3.0 / 2.5)
        assert df["roas_vs_campaign"].iloc[0] == pytest.approx(3.0 / 2.8)
        assert df["roas_vs_account"].iloc[0] == pytest.approx(3.0 / 3.0)

    def test_calculate_roas_comparisons_missing_columns(self):
        """Test ROAS comparisons with missing columns"""
        from src.cli.commands.extract import _calculate_roas_comparisons

        df = pd.DataFrame(
            {
                "purchase_roas": [3.0, 2.5],
                "adset_roas": [2.5, 2.0],
                # Missing campaign_roas and account_roas
            }
        )

        # Function modifies df in-place
        _calculate_roas_comparisons(df)

        # Should only create available comparisons
        assert "roas_vs_adset" in df.columns
        assert "roas_vs_campaign" not in df.columns
        assert "roas_vs_account" not in df.columns


class TestRollingMetrics:
    """Test rolling and EMA metric calculations"""

    def test_calc_roas_rolling_metrics(self):
        """Test ROAS rolling window calculations"""
        from src.cli.commands.extract import _calc_roas_rolling_metrics

        df = pd.DataFrame(
            {
                "adset_id": [
                    "adset1",
                    "adset1",
                    "adset1",
                    "adset1",
                    "adset2",
                    "adset2",
                ],
                "date_start": pd.date_range("2024-01-01", periods=6),
                "purchase_roas": [2.0, 2.5, 3.0, 3.5, 1.5, 2.0],
            }
        )

        result = _calc_roas_rolling_metrics(df.copy())

        assert "purchase_roas_rolling_7d" in result.columns
        assert "purchase_roas_rolling_14d" in result.columns
        assert "purchase_roas_ema_7d" in result.columns
        assert "purchase_roas_ema_14d" in result.columns

        # Check that rolling means are calculated
        # First row should have value (min_periods=1)
        assert result["purchase_roas_rolling_7d"].iloc[0] == pytest.approx(2.0)
        # Second row should average first 2 values
        assert result["purchase_roas_rolling_7d"].iloc[1] == pytest.approx(2.25)

    def test_calc_spend_rolling_metrics(self):
        """Test spend rolling window calculations"""
        from src.cli.commands.extract import _calc_spend_rolling_metrics

        df = pd.DataFrame(
            {
                "adset_id": ["adset1", "adset1", "adset1"],
                "date_start": pd.date_range("2024-01-01", periods=3),
                "spend": [100.0, 150.0, 200.0],
            }
        )

        result = _calc_spend_rolling_metrics(df.copy())

        assert "spend_rolling_7d" in result.columns
        assert "spend_rolling_14d" in result.columns
        assert "spend_ema_7d" in result.columns
        assert "spend_rolling_7d_std" in result.columns

        # Check 7-day rolling sum
        assert result["spend_rolling_7d"].iloc[0] == pytest.approx(100.0)
        assert result["spend_rolling_7d"].iloc[1] == pytest.approx(250.0)
        assert result["spend_rolling_7d"].iloc[2] == pytest.approx(450.0)

    def test_calc_roas_std_metrics(self):
        """Test ROAS rolling standard deviation calculations"""
        from src.cli.commands.extract import _calc_roas_std_metrics

        df = pd.DataFrame(
            {
                "adset_id": ["adset1", "adset1", "adset1"],
                "date_start": pd.date_range("2024-01-01", periods=3),
                "purchase_roas": [2.0, 2.5, 3.0],
            }
        )

        result = _calc_roas_std_metrics(df.copy())

        assert "purchase_roas_rolling_7d_std" in result.columns
        assert "purchase_roas_rolling_14d_std" in result.columns

        # First row should have NaN std (single value)
        assert pd.isna(result["purchase_roas_rolling_7d_std"].iloc[0])
        # Second row should have valid std
        assert result["purchase_roas_rolling_7d_std"].iloc[1] > 0

    def test_calc_roas_trend(self):
        """Test ROAS trend calculation"""
        from src.cli.commands.extract import _calc_roas_trend

        df = pd.DataFrame(
            {
                "adset_id": ["adset1"] * 10,
                "date_start": pd.date_range("2024-01-01", periods=10),
                "purchase_roas_rolling_7d": [
                    2.0,
                    2.2,
                    2.4,
                    2.6,
                    2.8,
                    3.0,
                    3.2,
                    3.4,
                    3.6,
                    3.8,
                ],
            }
        )

        result = _calc_roas_trend(df.copy())

        assert "roas_trend" in result.columns

        # Trend should be positive (increasing ROAS)
        # Day 8 vs Day 1: (3.4 - 2.0) / 2.0 = 0.7
        assert result["roas_trend"].iloc[7] > 0

    def test_calc_rolling_window_coverage(self):
        """Test rolling window coverage calculation"""
        from src.cli.commands.extract import _calc_rolling_window_coverage

        df = pd.DataFrame(
            {
                "adset_id": ["adset1"] * 10,
                "date_start": pd.date_range("2024-01-01", periods=10),
            }
        )

        result = _calc_rolling_window_coverage(df.copy())

        assert "days_of_data" in result.columns
        assert "rolling_7d_coverage" in result.columns
        assert "rolling_14d_coverage" in result.columns
        assert "rolling_low_quality" in result.columns

        # Check coverage calculations
        assert result["days_of_data"].iloc[0] == 1
        assert result["days_of_data"].iloc[6] == 7

        # Day 3: 3/7 = 0.43
        assert result["rolling_7d_coverage"].iloc[2] == pytest.approx(3 / 7, rel=1e-2)
        # Day 10: 7/7 = 1.0 (clipped)
        assert result["rolling_7d_coverage"].iloc[9] == pytest.approx(1.0)

        # Day 2 should be flagged as low quality (< 50% coverage)
        assert result["rolling_low_quality"].iloc[1] == 1
        # Day 7 should not be low quality (100% coverage)
        assert result["rolling_low_quality"].iloc[6] == 0


class TestLaggedFeatures:
    """Test lagged feature creation to prevent lookahead bias"""

    def test_create_lagged_roas_features(self):
        """Test creation of lagged ROAS features"""
        from src.cli.commands.extract import _create_lagged_features

        df = pd.DataFrame(
            {
                "adset_id": ["adset1", "adset1", "adset1", "adset2"],
                "date_start": pd.date_range("2024-01-01", periods=4),
                "purchase_roas_rolling_7d": [2.0, 2.5, 3.0, 2.2],
                "purchase_roas_rolling_14d": [2.2, 2.6, 3.1, 2.4],
            }
        )

        result = _create_lagged_features(df.copy())

        assert "purchase_roas_rolling_7d_lagged" in result.columns
        assert "purchase_roas_rolling_14d_lagged" in result.columns

        # First row should be NaN (no previous value)
        assert pd.isna(result["purchase_roas_rolling_7d_lagged"].iloc[0])
        # Second row should have first row's value
        assert result["purchase_roas_rolling_7d_lagged"].iloc[1] == pytest.approx(2.0)

    def test_create_lagged_spend_features(self):
        """Test creation of lagged spend features"""
        from src.cli.commands.extract import _create_lagged_features

        df = pd.DataFrame(
            {
                "adset_id": ["adset1", "adset1", "adset1"],
                "date_start": pd.date_range("2024-01-01", periods=3),
                "spend": [
                    100.0,
                    150.0,
                    200.0,
                ],  # Must include "spend" for spend rolling cols to be lagged
                "spend_rolling_7d": [100.0, 150.0, 200.0],
                "spend_rolling_14d": [110.0, 160.0, 210.0],
                "spend_ema_7d": [105.0, 140.0, 190.0],
            }
        )

        result = _create_lagged_features(df.copy())

        # Check which columns get lagged (from the function's list)
        assert "spend_rolling_7d_lagged" in result.columns
        assert "spend_rolling_14d_lagged" in result.columns
        assert "spend_ema_7d_lagged" in result.columns

        # Check lagging
        assert pd.isna(result["spend_rolling_7d_lagged"].iloc[0])
        assert result["spend_rolling_7d_lagged"].iloc[1] == pytest.approx(100.0)
        assert result["spend_rolling_7d_lagged"].iloc[2] == pytest.approx(150.0)

    def test_lagged_features_per_adset(self):
        """Test that lagging is done per adset (not globally)"""
        from src.cli.commands.extract import _create_lagged_features

        df = pd.DataFrame(
            {
                "adset_id": ["adset1", "adset1", "adset2", "adset2"],
                "date_start": pd.date_range("2024-01-01", periods=4),
                "purchase_roas_rolling_7d": [2.0, 2.5, 3.0, 3.5],
            }
        )

        result = _create_lagged_features(df.copy())

        # First row of each adset should be NaN
        assert pd.isna(result["purchase_roas_rolling_7d_lagged"].iloc[0])
        assert pd.isna(result["purchase_roas_rolling_7d_lagged"].iloc[2])

        # Second row of each adset should have first row's value
        assert result["purchase_roas_rolling_7d_lagged"].iloc[1] == pytest.approx(2.0)
        assert result["purchase_roas_rolling_7d_lagged"].iloc[3] == pytest.approx(3.0)


class TestBudgetMetrics:
    """Test budget-related metric calculations"""

    def test_recalculate_budget_metrics(self):
        """Test budget utilization and headroom calculations"""
        from src.cli.commands.extract import _recalculate_budget_metrics

        df = pd.DataFrame(
            {
                "adset_daily_budget": [100.0, 150.0, 200.0],
                "adset_spend": [80.0, 150.0, 180.0],
                "purchase_roas": [2.5, 3.0, 2.8],
            }
        )

        result = _recalculate_budget_metrics(df.copy())

        assert "budget_utilization_rate" in result.columns
        assert "budget_headroom" in result.columns
        assert "budget_roas_efficiency" in result.columns

        # Check calculations
        assert result["budget_utilization_rate"].iloc[0] == pytest.approx(80.0)
        assert result["budget_headroom"].iloc[0] == pytest.approx(20.0)
        assert result["budget_headroom"].iloc[1] == pytest.approx(
            0.0
        )  # spend == budget
        assert result["budget_headroom"].iloc[2] == pytest.approx(
            20.0
        )  # overspend clipped to 0

    def test_budget_metrics_handles_zero_budget(self):
        """Test budget metrics with zero budget"""
        from src.cli.commands.extract import _recalculate_budget_metrics

        df = pd.DataFrame(
            {
                "adset_daily_budget": [0.0, 100.0],
                "adset_spend": [50.0, 80.0],
                "purchase_roas": [2.5, 3.0],
            }
        )

        result = _recalculate_budget_metrics(df.copy())

        # Zero budget should result in NaN for utilization
        assert pd.isna(result["budget_utilization_rate"].iloc[0])
        assert result["budget_utilization_rate"].iloc[1] == pytest.approx(80.0)


class TestCostMetrics:
    """Test cost metric calculations"""

    def test_recalculate_cost_metrics(self):
        """Test CPC, CPM, CTR calculations"""
        from src.cli.commands.extract import _recalculate_cost_metrics

        df = pd.DataFrame(
            {
                "spend": [100.0, 200.0, 300.0],
                "clicks": [50, 100, 150],
                "impressions": [1000, 2000, 3000],
            }
        )

        result = _recalculate_cost_metrics(df.copy())

        assert "cpc" in result.columns
        assert "cpm" in result.columns
        assert "ctr" in result.columns

        # Check calculations
        assert result["cpc"].iloc[0] == pytest.approx(2.0)
        assert result["cpm"].iloc[0] == pytest.approx(100.0)  # (100/1000)*1000
        assert result["ctr"].iloc[0] == pytest.approx(5.0)  # (50/1000)*100

    def test_cost_metrics_handles_zero_clicks_impressions(self):
        """Test cost metrics with zero clicks/impressions"""
        from src.cli.commands.extract import _recalculate_cost_metrics

        df = pd.DataFrame(
            {"spend": [100.0, 200.0], "clicks": [0, 100], "impressions": [1000, 0]}
        )

        result = _recalculate_cost_metrics(df.copy())

        # Zero clicks should give NaN CPC
        assert pd.isna(result["cpc"].iloc[0])
        assert result["cpc"].iloc[1] == pytest.approx(2.0)

        # Zero impressions should give NaN CPM
        assert result["cpm"].iloc[0] == pytest.approx(100.0)
        assert pd.isna(result["cpm"].iloc[1])


class TestEngagementMetrics:
    """Test engagement and reach metrics"""

    def test_recalculate_engagement_metrics(self):
        """Test engagement rate and reach efficiency"""
        from src.cli.commands.extract import _recalculate_engagement_metrics

        df = pd.DataFrame(
            {
                "clicks": [50, 100, 150],
                "impressions": [1000, 2000, 3000],
                "reach": [800, 1600, 2400],
            }
        )

        result = _recalculate_engagement_metrics(df.copy())

        assert "engagement_rate" in result.columns
        assert "reach_efficiency" in result.columns

        # Check calculations
        assert result["engagement_rate"].iloc[0] == pytest.approx(5.0)  # (50/1000)*100
        assert result["reach_efficiency"].iloc[0] == pytest.approx(1.25)  # 1000/800


class TestInteractionMetrics:
    """Test interaction feature calculations"""

    def test_recalc_interaction_metrics(self):
        """Test ROAS-spend and CTR-CPC interactions"""
        from src.cli.commands.extract import _recalc_interaction_metrics

        df = pd.DataFrame(
            {
                "revenue": [200.0, 400.0, 600.0],
                "impressions": [1000, 2000, 3000],
                "ctr": [5.0, 5.5, 6.0],
                "cpc": [2.0, 2.1, 2.2],
            }
        )

        result = _recalc_interaction_metrics(df.copy())

        assert "roas_spend_interaction" in result.columns
        assert "expected_revenue" in result.columns
        assert "expected_clicks" in result.columns
        assert "ctr_cpc_interaction" in result.columns

        # Check calculations
        assert result["roas_spend_interaction"].iloc[0] == pytest.approx(200.0)
        assert result["expected_clicks"].iloc[0] == pytest.approx(
            50.0
        )  # 1000 * 5.0 / 100
        assert result["ctr_cpc_interaction"].iloc[0] == pytest.approx(10.0)  # 5.0 * 2.0


class TestRollingMetricsIntegration:
    """Integration tests for rolling metrics calculation"""

    def test_calculate_rolling_metrics_missing_columns(self):
        """Test rolling metrics with missing required columns"""
        from src.cli.commands.extract import _calculate_rolling_metrics

        # Missing purchase_roas column
        df = pd.DataFrame(
            {"adset_id": ["adset1"], "date_start": [pd.Timestamp("2024-01-01")]}
        )

        result = _calculate_rolling_metrics(df.copy())

        # Should return unchanged if required columns missing
        assert "purchase_roas_rolling_7d" not in result.columns

    @patch("src.cli.commands.extract.logger")
    def test_calculate_rolling_metrics_full_pipeline(self, mock_logger):
        """Test full rolling metrics calculation pipeline"""
        from src.cli.commands.extract import _calculate_rolling_metrics

        df = pd.DataFrame(
            {
                "adset_id": ["adset1"] * 10,
                "date_start": pd.date_range("2024-01-01", periods=10),
                "purchase_roas": [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8],
                "spend": [
                    100.0,
                    110.0,
                    120.0,
                    130.0,
                    140.0,
                    150.0,
                    160.0,
                    170.0,
                    180.0,
                    190.0,
                ],
            }
        )

        result = _calculate_rolling_metrics(df.copy())

        # Check all rolling metrics are created
        assert "purchase_roas_rolling_7d" in result.columns
        assert "purchase_roas_rolling_14d" in result.columns
        assert "purchase_roas_ema_7d" in result.columns
        assert "spend_rolling_7d" in result.columns
        assert "spend_ema_7d" in result.columns
        assert "roas_trend" in result.columns
        assert "days_of_data" in result.columns
        assert "rolling_7d_coverage" in result.columns

        # Verify logger was called
        assert mock_logger.info.called
