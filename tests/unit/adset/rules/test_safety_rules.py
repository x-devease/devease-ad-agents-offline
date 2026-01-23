"""
Unit tests for SafetyRules class.

Tests hard constraints for budget allocation including:
- Learning shock protection
- Freeze conditions
- Budget caps
- Cold start protection
"""

from unittest.mock import Mock
from src.adset.lib.safety_rules import SafetyRules


class TestSafetyRules:
    """Test suite for SafetyRules class"""

    def test_init_with_defaults(self):
        """Test initialization with default values"""
        rules = SafetyRules()

        assert rules.max_daily_increase_pct == 0.15
        assert rules.max_daily_decrease_pct == 0.15
        assert rules.freeze_roas_threshold == 0.5
        assert rules.freeze_health_threshold == 0.2
        assert rules.min_budget == 1.0
        assert rules.max_budget_pct_of_total == 0.40
        assert rules.cold_start_days == 3
        assert rules.cold_start_max_increase_pct == 0.10

    def test_init_with_config(self):
        """Test initialization with config parser"""
        config = Mock()

        def get_rule(key, default):
            return {
                "max_daily_increase_pct": 0.20,
                "max_daily_decrease_pct": 0.20,
                "freeze_roas_threshold": 0.6,
                "freeze_health_threshold": 0.3,
                "min_budget": 2.0,
                "max_budget_pct_of_total": 0.50,
                "cold_start_days": 5,
                "cold_start_max_increase_pct": 0.15,
            }.get(key, default)

        config.get_safety_rule = Mock(side_effect=get_rule)

        rules = SafetyRules(config)

        assert rules.max_daily_increase_pct == 0.20
        assert rules.max_daily_decrease_pct == 0.20
        assert rules.freeze_roas_threshold == 0.6
        assert rules.freeze_health_threshold == 0.3
        assert rules.min_budget == 2.0
        assert rules.max_budget_pct_of_total == 0.50
        assert rules.cold_start_days == 5
        assert rules.cold_start_max_increase_pct == 0.15

    def test_learning_shock_ok(self):
        """Test learning shock protection when change is within limits"""
        rules = SafetyRules()
        old_budget = 100.0
        new_budget = 110.0  # 10% increase, within 15% limit

        result = rules.apply_learning_shock_protection(new_budget, old_budget)
        assert result == 110.0

    def test_learning_shock_inc_exceeds(self):
        """Test learning shock protection when increase exceeds limit"""
        rules = SafetyRules()
        old_budget = 100.0
        new_budget = 120.0  # 20% increase, exceeds 15% limit

        result = rules.apply_learning_shock_protection(new_budget, old_budget)
        assert (
            abs(result - 115.0) < 0.01
        )  # Capped at 15% increase (with floating point tolerance)

    def test_learning_shock_dec_exceeds(self):
        """Test learning shock protection when decrease exceeds limit"""
        rules = SafetyRules()
        old_budget = 100.0
        new_budget = 80.0  # 20% decrease, exceeds 15% limit

        result = rules.apply_learning_shock_protection(new_budget, old_budget)
        assert result == 85.0  # Capped at 15% decrease

    def test_learning_shock_no_old(self):
        """Test learning shock protection with no previous budget"""
        rules = SafetyRules()
        new_budget = 100.0

        result = rules.apply_learning_shock_protection(new_budget, None)
        assert result == 100.0

        result = rules.apply_learning_shock_protection(new_budget, 0.0)
        assert result == 100.0

    def test_should_freeze_low_roas(self):
        """Test freeze condition when ROAS is too low"""
        rules = SafetyRules()

        assert rules.should_freeze(roas=0.4, health_score=0.5) is True
        assert rules.should_freeze(roas=0.5, health_score=0.5) is False
        assert rules.should_freeze(roas=0.6, health_score=0.5) is False

    def test_should_freeze_low_health(self):
        """Test freeze condition when health score is too low"""
        rules = SafetyRules()

        assert rules.should_freeze(roas=2.0, health_score=0.15) is True
        assert rules.should_freeze(roas=2.0, health_score=0.2) is False
        assert rules.should_freeze(roas=2.0, health_score=0.25) is False

    def test_should_freeze_both_low(self):
        """Test freeze condition when both ROAS and health are low"""
        rules = SafetyRules()

        assert rules.should_freeze(roas=0.4, health_score=0.15) is True

    def test_budget_caps_min_enforced(self):
        """Test budget caps enforce minimum budget"""
        rules = SafetyRules()
        total_budget = 1000.0

        result = rules.apply_budget_caps(budget=0.5, total_budget=total_budget)
        assert result == 1.0  # Enforced minimum

    def test_budget_caps_max_enforced(self):
        """Test budget caps enforce maximum budget percentage"""
        rules = SafetyRules()
        total_budget = 1000.0
        max_allowed = total_budget * 0.40  # 40% of total

        result = rules.apply_budget_caps(budget=500.0, total_budget=total_budget)
        # Capped at 40%
        assert result == max_allowed

    def test_budget_caps_within_limits(self):
        """Test budget caps when budget is within limits"""
        rules = SafetyRules()
        total_budget = 1000.0

        result = rules.apply_budget_caps(budget=100.0, total_budget=total_budget)
        # No change needed
        assert result == 100.0

    def test_cold_start_during_period(self):
        """Test cold start protection during cold start period"""
        rules = SafetyRules()
        old_budget = 100.0
        days_active = 2  # Within cold start period (3 days)
        new_budget = 120.0  # Would be 20% increase

        result = rules.apply_cold_start_protection(new_budget, old_budget, days_active)
        # Capped at 10% increase during cold start
        # (with floating point tolerance)
        assert abs(result - 110.0) < 0.01

    def test_cold_start_after_period(self):
        """Test cold start protection after cold start period"""
        rules = SafetyRules()
        old_budget = 100.0
        days_active = 5  # After cold start period
        new_budget = 120.0

        result = rules.apply_cold_start_protection(new_budget, old_budget, days_active)
        # No protection after cold start
        assert result == 120.0

    def test_cold_start_no_old_budget(self):
        """Test cold start protection with no previous budget"""
        rules = SafetyRules()
        days_active = 2
        new_budget = 100.0

        result = rules.apply_cold_start_protection(new_budget, None, days_active)
        # No protection if no old budget
        assert result == 100.0

        result = rules.apply_cold_start_protection(new_budget, 0.0, days_active)
        assert result == 100.0  # No protection if old budget is 0
