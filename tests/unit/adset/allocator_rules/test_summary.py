"""
Unit tests for summary.py module.
Tests the complete rule execution functions.
"""

from src.meta.adset.allocator.utils.summary import (
    apply_post_modifications,
    calculate_budget_adjustment,
    execute_all_rules,
)


class TestSummary:
    """Test suite for summary.py functions"""

    def test_calc_budget_adj_high(self):
        """Test budget adjustment for high performer"""
        adjustment_factor, reason, decision_path = calculate_budget_adjustment(
            roas_7d=3.0,
            roas_trend=0.10,
            adset_roas=2.8,
            campaign_roas=2.5,
            account_roas=2.2,
            health_score=0.85,
            days_active=25,
        )

        assert isinstance(adjustment_factor, (int, float))
        assert adjustment_factor > 1.0  # Should increase for high performer
        assert isinstance(reason, str)
        assert isinstance(decision_path, list)
        assert len(decision_path) > 0

    def test_calc_budget_adj_low(self):
        """Test budget adjustment for low performer"""
        adjustment_factor, reason, decision_path = calculate_budget_adjustment(
            roas_7d=1.0,
            roas_trend=-0.10,
            health_score=0.4,
            days_active=30,
        )

        assert isinstance(adjustment_factor, (int, float))
        assert adjustment_factor < 1.0  # Should decrease for low performer
        assert isinstance(reason, str)
        assert isinstance(decision_path, list)

    def test_calc_budget_adj_minimal(self):
        """Test budget adjustment with minimal parameters"""
        adjustment_factor, reason, decision_path = calculate_budget_adjustment(
            roas_7d=2.5,
            roas_trend=0.0,
        )

        assert isinstance(adjustment_factor, (int, float))
        assert adjustment_factor > 0
        assert isinstance(reason, str)
        assert isinstance(decision_path, list)

    def test_execute_all_rules_high(self):
        """Test execute_all_rules for high performing adset"""
        (
            final_adjustment,
            reason,
            decision_path,
            post_modifications,
        ) = execute_all_rules(
            roas_7d=3.5,
            roas_trend=0.12,
            current_budget=100.0,
            health_score=0.90,
            days_active=30,
        )

        assert isinstance(final_adjustment, (int, float))
        assert final_adjustment > 1.0  # Should increase
        assert isinstance(reason, str)
        assert isinstance(decision_path, list)
        assert isinstance(post_modifications, list)
        assert len(decision_path) > 0

    def test_execute_all_rules_low(self):
        """Test execute_all_rules for low performing adset"""
        current_budget = 100.0
        (
            final_budget,
            reason,
            decision_path,
            post_modifications,
        ) = execute_all_rules(
            roas_7d=1.0,
            roas_trend=-0.10,
            current_budget=current_budget,
            health_score=0.4,
            days_active=30,
        )

        assert isinstance(final_budget, (int, float))
        assert final_budget < current_budget  # Should decrease
        assert isinstance(reason, str)
        assert isinstance(decision_path, list)
        assert isinstance(post_modifications, list)

    def test_execute_all_rules_all(self, sample_metrics_dict):
        """Test execute_all_rules with all fields provided"""
        # Remove fields that execute_all_rules doesn't use
        metrics = {
            k: v
            for k, v in sample_metrics_dict.items()
            if k
            not in (
                "adset_id",
                "current_budget",
                "previous_budget",
                "total_budget_today",
            )
        }
        (
            final_adjustment,
            reason,
            decision_path,
            post_modifications,
        ) = execute_all_rules(**metrics)

        assert isinstance(final_adjustment, (int, float))
        assert final_adjustment > 0
        assert isinstance(reason, str)
        assert isinstance(decision_path, list)
        assert isinstance(post_modifications, list)
        assert len(decision_path) > 0

    def test_execute_all_returns_tuple(self):
        """Test that execute_all_rules returns a tuple of 4 elements"""
        result = execute_all_rules(
            roas_7d=2.5,
            roas_trend=0.0,
        )

        assert isinstance(result, tuple)
        assert len(result) == 4
        final_adjustment, reason, decision_path, post_modifications = result
        assert isinstance(final_adjustment, (int, float))
        assert isinstance(reason, str)
        assert isinstance(decision_path, list)
        assert isinstance(post_modifications, list)

    def test_calc_budget_adj_override(self):
        """Test threshold overrides in calculate_budget_adjustment"""
        # Test with custom thresholds
        adjustment_factor, reason, decision_path = calculate_budget_adjustment(
            roas_7d=2.5,
            roas_trend=0.0,
            high_roas_threshold=3.0,  # Override default 2.5
            low_roas_threshold=2.0,  # Override default 1.5
            excellent_roas_threshold=4.0,  # Override default 3.5
            aggressive_increase_pct=0.20,  # Override default 0.15
            moderate_increase_pct=0.12,  # Override default 0.10
            conservative_increase_pct=0.06,  # Override default 0.05
            moderate_decrease_pct=0.12,  # Override default 0.10
            aggressive_decrease_pct=0.25,  # Override default 0.20
        )

        assert isinstance(adjustment_factor, (int, float))
        assert isinstance(reason, str)
        assert isinstance(decision_path, list)

    def test_post_mods_cold_start_cap(self):
        """Test cold start protection capping large increases"""
        adjustment_factor, mods = apply_post_modifications(
            1.20,  # 20% increase
            roas_7d=2.5,
            days_active=2,  # Cold start
            cold_start_max_increase_pct=0.10,
        )

        # Should be capped at 1.10 (10% max)
        assert adjustment_factor <= 1.10
        assert any("cold_start_cap" in mod for mod in mods)

    def test_post_mods_cold_within_cap(self):
        """Test cold start protection when within cap"""
        adjustment_factor, mods = apply_post_modifications(
            1.08,  # 8% increase (within 10% cap)
            roas_7d=2.5,
            days_active=2,  # Cold start
            cold_start_max_increase_pct=0.10,
        )

        # Should not be capped
        assert adjustment_factor == 1.08
        assert any("cold_start_within_cap" in mod for mod in mods)

    def test_post_mods_cold_start_hold(self):
        """Test cold start protection with no change"""
        adjustment_factor, mods = apply_post_modifications(
            1.0,  # No change
            roas_7d=2.5,
            days_active=2,  # Cold start
            cold_start_max_increase_pct=0.10,
        )

        assert adjustment_factor == 1.0
        assert any("cold_start_hold" in mod for mod in mods)

    def test_post_mods_marginal_penalty(self):
        """Test marginal ROAS penalty application"""
        adjustment_factor, mods = apply_post_modifications(
            1.10,  # 10% increase
            roas_7d=2.5,
            marginal_roas=2.0,  # Lower than roas_7d (ratio < 0.9)
        )

        # Should have penalty applied
        assert adjustment_factor < 1.10
        assert any("marginal_roas_penalty" in mod for mod in mods)

    def test_post_mods_marginal_bonus(self):
        """Test marginal ROAS bonus application"""
        adjustment_factor, mods = apply_post_modifications(
            1.10,  # 10% increase
            roas_7d=2.5,
            marginal_roas=3.0,  # Higher than roas_7d (ratio > 1.1)
        )

        # Should have bonus applied
        assert adjustment_factor > 1.10
        assert any("marginal_roas_bonus" in mod for mod in mods)

    def test_post_mods_low_utilization(self):
        """Test low budget utilization gate"""
        adjustment_factor, mods = apply_post_modifications(
            1.15,  # 15% increase
            roas_7d=2.5,
            budget_utilization=0.5,  # Low utilization (< 0.7)
        )

        # Should reduce increase component by 50%
        # 1.0 + (0.15 * 0.5) = 1.075
        assert adjustment_factor == 1.075
        assert any("low_utilization_gate" in mod for mod in mods)

    def test_post_mods_high_utilization(self):
        """Test high budget utilization boost"""
        adjustment_factor, mods = apply_post_modifications(
            1.10,  # 10% increase
            roas_7d=2.5,
            budget_utilization=0.98,  # High utilization (> 0.95)
            aggressive_increase_pct=0.15,
        )

        # Should boost by 5%: min(1.10 * 1.05, 1.15) = 1.155
        assert adjustment_factor > 1.10
        assert any("high_utilization_boost" in mod for mod in mods)
