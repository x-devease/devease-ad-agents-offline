"""Unit tests for core Allocator class."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest

from src.adset.core.allocator import Allocator
from src.adset.lib.safety_rules import SafetyRules
from src.adset.lib.decision_rules import DecisionRules
from src.adset.lib.models import BudgetAllocationMetrics


class TestAllocator(TestCase):
    """Test Allocator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.safety_rules = MagicMock(spec=SafetyRules)
        self.decision_rules = MagicMock(spec=DecisionRules)
        self.config = MagicMock()
        self.allocator = Allocator(
            safety_rules=self.safety_rules,
            decision_rules=self.decision_rules,
            config=self.config,
        )

        # Default metrics
        self.default_metrics = {
            "adset_id": "test_adset",
            "current_budget": 100.0,
            "roas_7d": 2.0,
            "roas_trend": "stable",
            "health_score": 0.8,
            "adset_roas": 2.0,
            "campaign_roas": 2.0,
            "account_roas": 2.0,
            "roas_vs_adset": 1.0,
            "roas_vs_campaign": 1.0,
            "roas_vs_account": 1.0,
            "efficiency": 0.5,
            "revenue_per_impression": 0.1,
            "revenue_per_click": 1.0,
            "spend": 200.0,
            "spend_rolling_7d": 1400.0,
            "impressions": 2000,
            "clicks": 200,
            "reach": 1500,
            "adset_spend": 600.0,
            "campaign_spend": 1800.0,
            "expected_clicks": 200,
            "days_active": 30,
            "day_of_week": 3,
            "is_weekend": False,
            "week_of_year": 12,
        }

    def test_initialization(self):
        """Test Allocator initialization."""
        assert self.allocator.safety_rules == self.safety_rules
        assert self.allocator.decision_rules == self.decision_rules
        assert self.allocator.config == self.config

    def test_initialization_without_config(self):
        """Test Allocator initialization without config."""
        allocator = Allocator(
            safety_rules=self.safety_rules, decision_rules=self.decision_rules
        )
        assert allocator.config is None

    # === allocate_budget main method ===

    def test_allocate_budget_returns_tuple(self):
        """Test allocate_budget returns (new_budget, decision_path) tuple."""
        self.safety_rules.should_freeze.return_value = False
        self.decision_rules.calculate_budget_adjustment.return_value = (
            1.0,
            "test_reason",
        )
        self.decision_rules.budget_relative_scaling.return_value = 0.0
        self.safety_rules.apply_learning_shock_protection.return_value = 100.0
        self.safety_rules.apply_cold_start_protection.return_value = 100.0
        self.safety_rules.apply_budget_caps.return_value = 100.0

        result = self.allocator.allocate_budget(**self.default_metrics)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (int, float))  # new_budget
        assert isinstance(result[1], list)  # decision_path

    def test_allocate_budget_with_frozen_by_safety_rules(self):
        """Test allocate_budget when frozen by safety rules."""
        self.safety_rules.should_freeze.return_value = True
        self.safety_rules.freeze_roas_threshold = 0.5

        new_budget, decision_path = self.allocator.allocate_budget(
            **self.default_metrics
        )

        assert new_budget == 0.0
        assert "frozen_by_safety_rules" in decision_path

    def test_allocate_budget_keep_frozen_still_underperforming(self):
        """Test allocate_budget keeps frozen when still underperforming."""
        self.safety_rules.should_freeze.return_value = False
        self.safety_rules.freeze_roas_threshold = 0.5

        metrics = self.default_metrics.copy()
        metrics["current_budget"] = 0.0
        metrics["roas_7d"] = 0.5  # Below threshold * 1.2 = 0.6

        new_budget, decision_path = self.allocator.allocate_budget(**metrics)

        assert new_budget == 0.0
        assert "keep_frozen_still_underperforming" in decision_path

    def test_allocate_budget_normal_flow(self):
        """Test allocate_budget normal flow without freezing."""
        self.safety_rules.should_freeze.return_value = False
        self.decision_rules.calculate_budget_adjustment.return_value = (
            1.2,
            "good_performance",
        )
        self.decision_rules.budget_relative_scaling.return_value = 0.2
        self.safety_rules.apply_learning_shock_protection.return_value = 120.0
        self.safety_rules.apply_cold_start_protection.return_value = 120.0
        self.safety_rules.apply_budget_caps.return_value = 120.0

        # Disable advanced concepts
        self.config.get_advanced_concept.return_value = False

        new_budget, decision_path = self.allocator.allocate_budget(
            **self.default_metrics
        )

        assert new_budget == 120.0
        assert "decision_rule: good_performance" in decision_path

    # === _check_safety_freeze ===

    def test_check_safety_freeze_returns_none_when_not_frozen(self):
        """Test _check_safety_freeze returns None when not frozen."""
        self.safety_rules.should_freeze.return_value = False
        metrics = BudgetAllocationMetrics.from_dict(self.default_metrics)

        result = self.allocator._check_safety_freeze(metrics, [])

        assert result is None

    def test_check_safety_freeze_returns_zero_budget_when_frozen(self):
        """Test _check_safety_freeze returns (0.0, decision_path) when frozen."""
        self.safety_rules.should_freeze.return_value = True
        metrics = BudgetAllocationMetrics.from_dict(self.default_metrics)

        result = self.allocator._check_safety_freeze(metrics, [])

        assert result == (0.0, ["frozen_by_safety_rules"])

    # === _calculate_decision_adjustment ===

    def test_calculate_decision_adjustment(self):
        """Test _calculate_decision_adjustment calculates adjustment."""
        self.decision_rules.calculate_budget_adjustment.return_value = (
            1.5,
            "excellent_performance",
        )
        self.decision_rules.budget_relative_scaling.return_value = 0.5

        metrics = BudgetAllocationMetrics.from_dict(self.default_metrics)
        adjustment_factor, decision_path = (
            self.allocator._calculate_decision_adjustment(metrics, [])
        )

        assert adjustment_factor == 1.5  # 1.0 + 0.5
        assert "decision_rule: excellent_performance" in decision_path
        self.decision_rules.calculate_budget_adjustment.assert_called_once()
        self.decision_rules.budget_relative_scaling.assert_called_once_with(100.0, 0.5)

    # === _apply_safety_constraints ===

    def test_apply_safety_constraints_applies_all_constraints(self):
        """Test _apply_safety_constraints applies all safety constraints."""
        self.safety_rules.apply_learning_shock_protection.return_value = 110.0
        self.safety_rules.apply_cold_start_protection.return_value = 105.0
        self.safety_rules.apply_budget_caps.return_value = 100.0

        metrics = BudgetAllocationMetrics.from_dict(self.default_metrics)

        result = self.allocator._apply_safety_constraints(120.0, metrics, [])

        # Result is 105.0 because apply_budget_caps is only called if total_budget_today is set
        # Since default_metrics doesn't have total_budget_today, caps are not applied
        assert result == 105.0
        self.safety_rules.apply_learning_shock_protection.assert_called_once()
        self.safety_rules.apply_cold_start_protection.assert_called_once()
        # apply_budget_caps should NOT be called since total_budget_today is None
        self.safety_rules.apply_budget_caps.assert_not_called()

    def test_apply_safety_constraints_with_smoothing(self):
        """Test _apply_safety_constraints applies smoothing when enabled."""
        self.safety_rules.apply_learning_shock_protection.return_value = 120.0
        self.safety_rules.apply_cold_start_protection.return_value = 120.0
        self.safety_rules.apply_budget_caps.return_value = 114.0  # After smoothing

        self.config.get_advanced_concept.side_effect = lambda key, default: {
            "smoothing_enabled": True,
            "smoothing_alpha": 0.7,
        }.get(key, default)

        metrics = BudgetAllocationMetrics.from_dict(
            {**self.default_metrics, "previous_budget": 100.0}
        )

        result = self.allocator._apply_safety_constraints(120.0, metrics, [])

        # smoothing: 0.7 * 120.0 + 0.3 * 100.0 = 84.0 + 30.0 = 114.0
        assert result == 114.0

    def test_apply_safety_constraints_without_previous_budget_no_smoothing(self):
        """Test smoothing is skipped when previous_budget is None."""
        self.safety_rules.apply_learning_shock_protection.return_value = 120.0
        self.safety_rules.apply_cold_start_protection.return_value = 120.0
        self.safety_rules.apply_budget_caps.return_value = 120.0

        self.config.get_advanced_concept.return_value = True

        metrics = BudgetAllocationMetrics.from_dict(
            {**self.default_metrics, "previous_budget": None}
        )

        result = self.allocator._apply_safety_constraints(120.0, metrics, [])

        assert result == 120.0

    def test_apply_safety_constraints_with_total_budget_cap(self):
        """Test safety constraints apply total budget cap."""
        self.safety_rules.apply_learning_shock_protection.return_value = 120.0
        self.safety_rules.apply_cold_start_protection.return_value = 120.0
        self.safety_rules.apply_budget_caps.return_value = 100.0

        self.config.get_advanced_concept.return_value = False

        metrics = BudgetAllocationMetrics.from_dict(
            {**self.default_metrics, "total_budget_today": 1000.0}
        )

        result = self.allocator._apply_safety_constraints(120.0, metrics, [])

        assert result == 100.0
        self.safety_rules.apply_budget_caps.assert_called_once_with(120.0, 1000.0)

    # === _apply_post_mods ===

    def test_apply_post_mods_calls_all_modifications(self):
        """Test _apply_post_mods calls all post-modification methods."""
        self.config.get_advanced_concept.return_value = False

        metrics = BudgetAllocationMetrics.from_dict(self.default_metrics)

        with patch.object(
            self.allocator, "_apply_adaptive_target", return_value=(120.0, [])
        ):
            with patch.object(
                self.allocator,
                "_apply_marginal_roas_adjustment",
                return_value=(120.0, []),
            ):
                with patch.object(
                    self.allocator,
                    "_apply_budget_utilization",
                    return_value=(120.0, 1.2, []),
                ):
                    new_budget, decision_path = self.allocator._apply_post_mods(
                        100.0, metrics, 1.2, []
                    )

        assert new_budget == 120.0

    # === _apply_adaptive_target ===

    def test_apply_adaptive_target_when_enabled(self):
        """Test _apply_adaptive_target when enabled."""
        self.config.get_advanced_concept.side_effect = lambda key, default: {
            "adaptive_target_enabled": True,
            "adaptive_target_adjustment_factor": 0.3,
        }.get(key, default)

        with patch("src.adset.core.allocator.apply_adaptive_target_adj") as mock_adj:
            mock_adj.return_value = (110.0, ["adaptive_target_adjusted"])

            new_budget, decision_path = self.allocator._apply_adaptive_target(
                100.0, 2.5, 2.0, []
            )

            assert new_budget == 110.0
            assert "adaptive_target_adjusted" in decision_path
            mock_adj.assert_called_once()

    def test_apply_adaptive_target_when_disabled(self):
        """Test _apply_adaptive_target when disabled."""
        self.config.get_advanced_concept.return_value = False

        new_budget, decision_path = self.allocator._apply_adaptive_target(
            100.0, 2.5, 2.0, []
        )

        assert new_budget == 100.0
        assert decision_path == []

    # === _apply_marginal_roas_adjustment ===

    def test_apply_marginal_roas_adjustment_when_enabled(self):
        """Test _apply_marginal_roas_adjustment when enabled."""
        self.config.get_advanced_concept.side_effect = lambda key, default: {
            "marginal_roas_enabled": True,
            "marginal_roas_base_factor": 0.95,
            "marginal_roas_range_factor": 0.05,
        }.get(key, default)

        new_budget, decision_path = self.allocator._apply_marginal_roas_adjustment(
            100.0,  # new_budget
            1.5,  # marginal_roas
            2.0,  # roas_7d
            [],  # decision_path
        )

        # marginal_ratio = 1.5 / 2.0 = 0.75 (< 0.9)
        # factor = 0.95 + 0.75 * 0.05 = 0.95 + 0.0375 = 0.9875
        # new_budget = 100.0 * 0.9875 = 98.75
        expected = 100.0 * (0.95 + 0.75 * 0.05)
        assert abs(new_budget - expected) < 0.01
        assert "marginal_roas_adjustment" in decision_path[0]

    def test_apply_marginal_roas_adjustment_skipped_when_ratio_above_threshold(self):
        """Test _apply_marginal_roas_adjustment skipped when ratio >= 0.9."""
        self.config.get_advanced_concept.side_effect = lambda key, default: {
            "marginal_roas_enabled": True,
        }.get(key, default)

        new_budget, decision_path = self.allocator._apply_marginal_roas_adjustment(
            100.0,  # new_budget
            1.9,  # marginal_roas
            2.0,  # roas_7d (ratio = 0.95 >= 0.9)
            [],  # decision_path
        )

        assert new_budget == 100.0  # Unchanged
        assert decision_path == []

    def test_apply_marginal_roas_adjustment_disabled(self):
        """Test _apply_marginal_roas_adjustment when disabled."""
        self.config.get_advanced_concept.return_value = False

        new_budget, decision_path = self.allocator._apply_marginal_roas_adjustment(
            100.0, 1.5, 2.0, []
        )

        assert new_budget == 100.0
        assert decision_path == []

    # === _apply_budget_utilization ===

    def test_apply_budget_utilization_low_utilization(self):
        """Test _apply_budget_utilization with low utilization."""
        self.config.get_advanced_concept.side_effect = lambda key, default: {
            "budget_utilization_enabled": True,
            "low_utilization_threshold": 0.7,
            "low_utilization_adjustment_factor": 0.5,
            "high_utilization_threshold": 0.95,
            "high_utilization_boost_factor": 1.1,
        }.get(key, default)

        self.decision_rules.aggressive_increase_pct = 0.5

        metrics = BudgetAllocationMetrics.from_dict(
            {**self.default_metrics, "budget_utilization": 0.5}  # Below low threshold
        )

        new_budget, adjustment_factor, decision_path = (
            self.allocator._apply_budget_utilization(100.0, metrics, 1.2, [])
        )

        # increase_component = 0.2, adjusted = 0.2 * 0.5 = 0.1
        # adjustment_factor = 1.0 + 0.1 = 1.1
        # new_budget = 100.0 * 1.1 = 110.0
        assert new_budget == pytest.approx(110.0)
        assert adjustment_factor == pytest.approx(1.1)
        assert "low_utilization_gate" in decision_path[0]

    def test_apply_budget_utilization_high_utilization(self):
        """Test _apply_budget_utilization with high utilization."""
        self.config.get_advanced_concept.side_effect = lambda key, default: {
            "budget_utilization_enabled": True,
            "low_utilization_threshold": 0.7,
            "low_utilization_adjustment_factor": 0.5,
            "high_utilization_threshold": 0.95,
            "high_utilization_boost_factor": 1.1,
        }.get(key, default)

        self.decision_rules.aggressive_increase_pct = 0.5

        metrics = BudgetAllocationMetrics.from_dict(
            {**self.default_metrics, "budget_utilization": 0.98}  # Above high threshold
        )

        new_budget, adjustment_factor, decision_path = (
            self.allocator._apply_budget_utilization(100.0, metrics, 1.2, [])
        )

        # adjustment_factor = min(1.2 * 1.1, 1.5) = min(1.32, 1.5) = 1.32
        # new_budget = 100.0 * 1.32 = 132.0
        assert new_budget == 132.0
        assert adjustment_factor == 1.32
        assert "high_utilization_boost" in decision_path[0]

    def test_apply_budget_utilization_disabled(self):
        """Test _apply_budget_utilization when disabled."""
        self.config.get_advanced_concept.return_value = False

        metrics = BudgetAllocationMetrics.from_dict(
            {**self.default_metrics, "budget_utilization": 0.5}
        )

        new_budget, adjustment_factor, decision_path = (
            self.allocator._apply_budget_utilization(100.0, metrics, 1.2, [])
        )

        assert new_budget == 100.0
        assert adjustment_factor == 1.2
        assert decision_path == []

    def test_apply_budget_utilization_none(self):
        """Test _apply_budget_utilization with None budget_utilization."""
        self.config.get_advanced_concept.return_value = True

        metrics = BudgetAllocationMetrics.from_dict(
            {**self.default_metrics, "budget_utilization": None}
        )

        new_budget, adjustment_factor, decision_path = (
            self.allocator._apply_budget_utilization(100.0, metrics, 1.2, [])
        )

        assert new_budget == 100.0
        assert adjustment_factor == 1.2
        assert decision_path == []
