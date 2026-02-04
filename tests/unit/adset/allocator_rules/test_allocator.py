"""
Unit tests for Allocator class.

Tests the integration of SafetyRules and DecisionRules.
"""

from unittest.mock import Mock, patch, mock_open, MagicMock
import pytest
import yaml

from src.meta.adset.allocator.allocator import Allocator
from src.meta.adset.allocator.lib.safety_rules import SafetyRules
from src.meta.adset.allocator.lib.decision_rules import DecisionRules
from src.meta.adset.allocator.utils.parser import Parser


@pytest.fixture
def mock_config():
    """Create a mock config object with necessary attributes."""
    config = MagicMock()

    # Decision rules as a dict attribute (as expected by DecisionRules)
    config.decision_rules = {
        "low_roas_threshold": 0.5,
        "high_roas_threshold": 2.0,
        "freeze_roas_threshold": 0.5,
        "aggressive_increase_pct": 0.15,
        "marginal_roas_threshold": 1.5,
        "excellent_roas_threshold": 3.0,
        "cold_start_days": 3,
    }

    # Safety rules values for get_safety_rule method
    safety_rules_values = {
        "max_daily_increase_pct": 0.15,
        "max_daily_decrease_pct": 0.15,
        "freeze_roas_threshold": 0.5,
        "freeze_health_threshold": 0.2,
        "min_budget": 1.0,
        "max_budget_pct_of_total": 0.40,
        "cold_start_days": 3,
        "cold_start_max_increase_pct": 0.10,
        "low_utilization_threshold": 0.7,
        "high_utilization_threshold": 0.95,
    }

    # Advanced concepts values
    advanced_concepts_values = {
        "low_utilization_adjustment_factor": 0.5,
        "high_utilization_boost_factor": 1.1,
    }

    def get_safety_rule(key, default=None):
        return safety_rules_values.get(key, default)

    def get_advanced_concept(key, default=None):
        return advanced_concepts_values.get(key, default)

    config.get_safety_rule = get_safety_rule
    config.get_advanced_concept = get_advanced_concept
    config.safety_rules = safety_rules_values  # Also as dict attribute

    return config


class TestAllocator:
    """Test suite for Allocator class"""

    def test_init(self):
        """Test initialization"""
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        config = Mock()

        allocator = Allocator(safety_rules, decision_rules, config)

        assert allocator.safety_rules == safety_rules
        assert allocator.decision_rules == decision_rules
        assert allocator.config == config

    def test_alloc_frozen_by_safety(self):
        """Test allocation when adset should be frozen"""
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        allocator = Allocator(safety_rules, decision_rules)

        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_001",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=0.3,  # Below freeze threshold
            roas_trend=0.0,
            health_score=0.5,
        )

        assert new_budget == 0.0
        assert "frozen_by_safety_rules" in decision_path

    def test_alloc_high_performer(self):
        """Test allocation for high performing adset"""
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        allocator = Allocator(safety_rules, decision_rules)

        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_002",
            current_budget=100.0,
            previous_budget=95.0,
            roas_7d=3.5,  # High ROAS
            roas_trend=0.12,  # Rising trend
            health_score=0.85,
        )

        assert new_budget > 100.0  # Should increase
        assert len(decision_path) > 0
        assert any("decision_rule" in path for path in decision_path)

    def test_alloc_learning_shock(self):
        """Test that learning shock protection is applied"""
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        allocator = Allocator(safety_rules, decision_rules)

        # Decision rule would suggest large increase,
        # but safety rules should cap it
        new_budget, _ = allocator.allocate_budget(
            adset_id="test_003",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=4.0,  # Excellent ROAS
            roas_trend=0.20,  # Strong rising trend
            efficiency=0.15,
            health_score=0.95,
        )

        # Should be capped at 15% increase
        assert new_budget <= 115.0

    def test_alloc_applies_caps(self):
        """Test that budget caps are applied"""
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        allocator = Allocator(safety_rules, decision_rules)

        # Large budget that would exceed max percentage
        total_budget = 1000.0
        new_budget, _ = allocator.allocate_budget(
            adset_id="test_004",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=4.0,
            roas_trend=0.15,
            health_score=0.9,
            total_budget_today=total_budget,
        )

        # Should be capped at 40% of total budget
        max_allowed = total_budget * 0.40
        assert new_budget <= max_allowed

    def test_alloc_cold_start(self):
        """Test that cold start protection is applied"""
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        allocator = Allocator(safety_rules, decision_rules)

        new_budget, _ = allocator.allocate_budget(
            adset_id="test_005",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.5,  # High ROAS
            roas_trend=0.15,
            days_active=2,  # Cold start period
            health_score=0.8,
        )

        # Should be capped at 10% increase during cold start
        # (with floating point tolerance)
        assert new_budget <= 110.01

    def test_alloc_keeps_frozen(self, sample_frozen_metrics_dict):
        """Test that frozen adsets stay frozen if still underperforming"""
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        allocator = Allocator(safety_rules, decision_rules)

        metrics = sample_frozen_metrics_dict.copy()
        metrics["adset_id"] = "test_006"
        new_budget, decision_path = allocator.allocate_budget(**metrics)

        assert new_budget == 0.0
        # Decision path may be 'frozen_by_safety_rules' or
        # 'keep_frozen_still_underperforming'
        decision_path_str = " ".join(decision_path).lower()
        assert "frozen" in decision_path_str

    def test_alloc_all_features(self, sample_metrics_dict):
        """Test allocation with all 21 features provided"""
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        allocator = Allocator(safety_rules, decision_rules)

        metrics = sample_metrics_dict.copy()
        metrics["adset_id"] = "test_007"
        new_budget, decision_path = allocator.allocate_budget(**metrics)

        assert new_budget > 0
        assert len(decision_path) > 0

    def test_alloc_keep_frozen_under(self, mock_config):
        """Test that frozen adsets stay frozen if still underperforming"""
        safety_rules = SafetyRules(mock_config)
        decision_rules = DecisionRules(mock_config)
        allocator = Allocator(safety_rules, decision_rules, mock_config)

        # Adset was frozen (current_budget = 0) and still underperforming
        # ROAS must be < freeze_roas_threshold * 1.2 (0.5 * 1.2 = 0.6)
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_008",
            current_budget=0.0,  # Currently frozen
            previous_budget=0.0,  # Was frozen
            roas_7d=0.4,  # Still below threshold * 1.2 (0.5 * 1.2 = 0.6)
            roas_trend=0.0,
            health_score=0.3,
        )

        assert new_budget == 0.0
        # May be either frozen_by_safety_rules or
        # keep_frozen_still_underperforming
        decision_path_str = " ".join(decision_path).lower()
        assert "frozen" in decision_path_str

    def test_alloc_low_utilization_gate(self, mock_config):
        """Test low budget utilization gate"""
        safety_rules = SafetyRules(mock_config)
        decision_rules = DecisionRules(mock_config)
        allocator = Allocator(safety_rules, decision_rules, mock_config)

        _, decision_path = allocator.allocate_budget(
            adset_id="test_009",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.0,  # High ROAS
            roas_trend=0.10,
            health_score=0.85,
            budget_utilization=0.5,  # Low utilization (< 0.7)
        )

        # Should have low utilization gate applied
        assert any("low_utilization_gate" in path for path in decision_path)

    def test_alloc_high_util_boost(self, mock_config):
        """Test high budget utilization boost"""
        safety_rules = SafetyRules(mock_config)
        decision_rules = DecisionRules(mock_config)
        allocator = Allocator(safety_rules, decision_rules, mock_config)

        _, decision_path = allocator.allocate_budget(
            adset_id="test_010",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.0,  # High ROAS
            roas_trend=0.10,
            health_score=0.85,
            budget_utilization=0.98,  # High utilization (> 0.95)
        )

        # Should have high utilization boost applied
        assert any("high_utilization_boost" in path for path in decision_path)

    def test_alloc_marginal_roas_adj(self, mock_config):
        """Test marginal ROAS adjustment in allocator"""
        safety_rules = SafetyRules(mock_config)
        decision_rules = DecisionRules(mock_config)
        allocator = Allocator(safety_rules, decision_rules, mock_config)

        _, decision_path = allocator.allocate_budget(
            adset_id="test_011",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.5,
            roas_trend=0.10,
            health_score=0.85,
            marginal_roas=2.0,  # Lower than roas_7d (ratio < 0.9)
        )

        # Should have marginal ROAS adjustment applied
        assert any("marginal_roas_adjustment" in path for path in decision_path)
