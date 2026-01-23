"""
Unit tests for DecisionRulesHelpers class.
Tests helper functions for decision rules calculations.
"""

from src.adset.lib.decision_rules_helpers import DecisionRulesHelpers


class TestDecisionRulesHelpers:
    """Test suite for DecisionRulesHelpers"""

    def test_grad_adj_disabled(self):
        """Test gradient adjustment when disabled"""
        result = DecisionRulesHelpers.gradient_adjustment(
            value=2.5,
            threshold=2.0,
            base_pct=0.15,
            gradient_enabled=False,
        )

        # Should return base_pct when disabled
        assert result == 0.15

    def test_grad_adj_range_none(self):
        """Test gradient adjustment with None range_size"""
        result = DecisionRulesHelpers.gradient_adjustment(
            value=2.5,
            threshold=2.0,
            base_pct=0.15,
            gradient_enabled=True,
            range_size=None,  # Should calculate from threshold
        )

        # Should calculate range_size from threshold
        assert isinstance(result, float)
        assert result >= 0

    def test_grad_adj_decreasing(self):
        """Test gradient adjustment with decreasing=True"""
        # Value below threshold - range_size should get full adjustment
        result = DecisionRulesHelpers.gradient_adjustment(
            value=1.0,  # Below threshold (2.0) - range_size
            threshold=2.0,
            base_pct=0.15,
            gradient_enabled=True,
            increasing=False,
            range_size=0.5,
        )

        assert result == 0.15  # Full adjustment

    def test_grad_adj_dec_in_range(self):
        """Test gradient adjustment decreasing in transition range"""
        result = DecisionRulesHelpers.gradient_adjustment(
            value=1.8,  # Between threshold - range_size and threshold
            threshold=2.0,
            base_pct=0.15,
            gradient_enabled=True,
            increasing=False,
            range_size=0.5,
        )

        # Should be scaled between gradient_start and gradient_end
        assert 0 < result < 0.15

    def test_trend_scaling_disabled(self):
        """Test trend scaling when disabled"""
        result = DecisionRulesHelpers.trend_scaling(
            trend=0.15,
            base_pct=0.10,
            trend_scaling_enabled=False,
        )

        # Should return base_pct when disabled
        assert result == 0.10

    def test_trend_scaling_moderate(self):
        """Test trend scaling for moderate trend"""
        result = DecisionRulesHelpers.trend_scaling(
            trend=0.12,  # Moderate trend (between 0.10 and 0.20)
            base_pct=0.10,
            trend_scaling_enabled=True,
        )

        # Should scale based on moderate trend
        assert isinstance(result, float)
        assert result > 0

    def test_trend_scaling_weak(self):
        """Test trend scaling for weak trend"""
        result = DecisionRulesHelpers.trend_scaling(
            trend=0.06,  # Weak trend (between 0.05 and 0.10)
            base_pct=0.10,
            trend_scaling_enabled=True,
        )

        # Should scale based on weak trend
        assert isinstance(result, float)
        assert result > 0

    def test_rel_perf_grad_below(self):
        """Test relative performance gradient when ratio < threshold"""
        result = DecisionRulesHelpers.relative_performance_gradient(
            ratio=1.0,  # Below threshold 1.2
            threshold=1.2,
            base_pct=0.10,
            is_above=True,
        )

        # Should return 0.0 when ratio < threshold and is_above=True
        assert result == 0.0

    def test_rel_perf_grad_not_above(self):
        """Test relative performance gradient with is_above=False"""
        result = DecisionRulesHelpers.relative_performance_gradient(
            ratio=0.8,  # Below threshold 1.2
            threshold=1.2,
            base_pct=0.10,
            is_above=False,  # Below threshold means increase
        )

        # Should scale based on deficit
        assert isinstance(result, float)
        assert result > 0

    def test_rel_perf_grad_above(self):
        """Test relative performance when ratio > threshold, is_above=False."""
        result = DecisionRulesHelpers.relative_performance_gradient(
            ratio=1.3,  # Above threshold 1.2
            threshold=1.2,
            base_pct=0.10,
            is_above=False,
        )

        # Should return 0.0 when ratio > threshold and is_above=False
        assert result == 0.0

    def test_health_score_mult_disabled(self):
        """Test health score multiplier when disabled"""
        result = DecisionRulesHelpers.health_score_multiplier(
            health_score=0.8,
            health_score_multiplier_enabled=False,
        )

        # Should return 1.0 when disabled
        assert result == 1.0

    def test_budget_rel_scale_disabled(self):
        """Test budget relative scaling when disabled"""
        result = DecisionRulesHelpers.budget_relative_scaling(
            current_budget=100.0,
            base_pct=0.15,
            budget_relative_scaling_enabled=False,
        )

        # Should return base_pct when disabled
        assert result == 0.15

    def test_sample_conf_none(self):
        """Test sample size confidence with None clicks"""
        result = DecisionRulesHelpers.sample_size_confidence(
            clicks=None,
            base_pct=0.10,
        )

        # Should apply low multiplier when clicks is None
        assert result == 0.10 * 0.5  # low_mult = 0.5

    def test_sample_size_confidence_low(self):
        """Test sample size confidence with low clicks"""
        result = DecisionRulesHelpers.sample_size_confidence(
            clicks=30,  # Below low threshold (50)
            base_pct=0.10,
        )

        # Should apply low multiplier
        assert result == 0.10 * 0.5

    def test_sample_conf_medium(self):
        """Test sample size confidence with medium clicks"""
        result = DecisionRulesHelpers.sample_size_confidence(
            clicks=150,  # Between low (50) and medium (200)
            base_pct=0.10,
        )

        # Should apply medium multiplier
        assert result == 0.10 * 0.75

    def test_q4_dynamic_boost_none(self):
        """Test Q4 dynamic boost with None week"""
        result = DecisionRulesHelpers.q4_dynamic_boost(week_of_year=None)

        # Should return 0.0 when week is None
        assert result == 0.0

    def test_q4_dynamic_boost_week_48(self):
        """Test Q4 dynamic boost for week 48"""
        result = DecisionRulesHelpers.q4_dynamic_boost(week_of_year=48)

        assert result == 0.01

    def test_q4_dynamic_boost_week_49(self):
        """Test Q4 dynamic boost for week 49"""
        result = DecisionRulesHelpers.q4_dynamic_boost(week_of_year=49)

        assert result == 0.02

    def test_q4_dynamic_boost_week_50(self):
        """Test Q4 dynamic boost for week 50"""
        result = DecisionRulesHelpers.q4_dynamic_boost(week_of_year=50)

        assert result == 0.03

    def test_q4_dynamic_boost_week_51(self):
        """Test Q4 dynamic boost for week 51"""
        result = DecisionRulesHelpers.q4_dynamic_boost(week_of_year=51)

        assert result == 0.03

    def test_q4_dynamic_boost_week_52(self):
        """Test Q4 dynamic boost for week 52"""
        result = DecisionRulesHelpers.q4_dynamic_boost(week_of_year=52)

        assert result == 0.025

    def test_q4_dynamic_boost_non_q4(self):
        """Test Q4 dynamic boost for non-Q4 week"""
        result = DecisionRulesHelpers.q4_dynamic_boost(week_of_year=20)

        # Should return 0.0 for non-Q4 weeks
        assert result == 0.0

    def test_budget_rel_scaling_large(self):
        """Test budget relative scaling for large budgets"""
        result = DecisionRulesHelpers.budget_relative_scaling(
            current_budget=600.0,  # Large budget
            base_pct=0.20,
            budget_relative_scaling_enabled=True,
        )

        # Should cap at large_max (0.15)
        assert result == 0.15

    def test_budget_rel_scale_medium(self):
        """Test budget relative scaling for medium budgets"""
        result = DecisionRulesHelpers.budget_relative_scaling(
            current_budget=150.0,  # Medium budget
            base_pct=0.25,
            budget_relative_scaling_enabled=True,
        )

        # Should cap at medium_max (0.20)
        assert result == 0.20

    def test_budget_rel_scaling_small(self):
        """Test budget relative scaling for small budgets"""
        result = DecisionRulesHelpers.budget_relative_scaling(
            current_budget=50.0,  # Small budget
            base_pct=0.30,
            budget_relative_scaling_enabled=True,
        )

        # Should cap at small_max (0.25)
        assert result == 0.25

    def test_budget_rel_scale_negative(self):
        """Test budget relative scaling with negative base_pct"""
        result = DecisionRulesHelpers.budget_relative_scaling(
            current_budget=50.0,
            base_pct=-0.20,  # Negative (decrease)
            budget_relative_scaling_enabled=True,
        )

        # Should maintain sign but cap absolute value
        assert result < 0
        assert abs(result) <= 0.25

    def test_sample_conf_high(self):
        """Test sample size confidence with high clicks"""
        result = DecisionRulesHelpers.sample_size_confidence(
            clicks=250,  # Above medium threshold (200)
            base_pct=0.10,
        )

        # Should return full base_pct (line 312)
        assert result == 0.10
