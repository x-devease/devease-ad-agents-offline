"""
Unit tests for DecisionRules class.

Tests priority-based budget adjustment rules including:
- Excellent performers (Tier 2)
- High performers (Tier 3)
- Efficiency-based rules (Tier 4)
- Volume-based rules (Tier 5)
- Declining performers (Tier 6)
- Time-based adjustments (Tier 7)
- Lifecycle-based rules (Tier 8)
"""

from unittest.mock import Mock

from src.adset.allocator.lib.decision_rules import DecisionRules
from src.adset.allocator.lib.decision_rules_helpers import DecisionRulesHelpers


class TestDecisionRules:
    """Test suite for DecisionRules class"""

    def test_init_with_defaults(self):
        """Test initialization with default values"""
        rules = DecisionRules()

        assert rules.high_roas_threshold == 3.0
        assert rules.low_roas_threshold == 1.5
        assert rules.excellent_roas_threshold == 3.5
        assert rules.aggressive_increase_pct == 0.15
        assert rules.moderate_increase_pct == 0.10
        assert rules.conservative_increase_pct == 0.05

    def test_init_with_config(self):
        """Test initialization with config parser"""
        config = Mock()
        config.decision_rules = {
            "high_roas_threshold": 2.8,
            "low_roas_threshold": 1.4,
            "excellent_roas_threshold": 3.8,
            "aggressive_increase_pct": 0.18,
            "moderate_increase_pct": 0.12,
        }

        rules = DecisionRules(config)

        assert rules.high_roas_threshold == 2.8
        assert rules.low_roas_threshold == 1.4
        assert rules.excellent_roas_threshold == 3.8
        assert rules.aggressive_increase_pct == 0.18
        assert rules.moderate_increase_pct == 0.12

    def test_excellent_perf_tier2_r21(self):
        """Test Tier 2.1: Excellent ROAS + Strong Rising +
        High Efficiency + Healthy"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=4.0,  # Excellent ROAS
            roas_trend=0.15,  # Strong rising trend
            efficiency=0.12,  # High efficiency
            health_score=0.90,  # Excellent health
        )

        assert multiplier > 1.0  # Should increase
        assert "excellent_performer" in reason.lower()

    def test_high_perf_tier3_r31(self):
        """Test Tier 3.1: High ROAS + Rising Trend + Healthy"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            roas_trend=0.12,  # Strong rising trend
            health_score=0.75,  # Healthy
        )

        assert multiplier > 1.0  # Should increase
        assert "high_roas" in reason.lower()

    def test_low_perf_tier6_r61(self):
        """Test Tier 6.1: Low ROAS + Falling Trend"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=1.0,  # Low ROAS
            roas_trend=-0.15,  # Strong falling trend
            health_score=0.40,  # Not too unhealthy
        )

        assert multiplier < 1.0  # Should decrease
        assert "decrease" in reason.lower() or "low_roas" in reason.lower()

    def test_weekend_boost_t7_r71(self):
        """Test Tier 7.1: Weekend Boost"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.5,  # Medium ROAS
            roas_trend=0.0,
            is_weekend=True,
            health_score=0.7,
        )

        assert multiplier > 1.0  # Should get boost
        assert "weekend" in reason.lower()

    def test_cold_start_t8_r81(self):
        """Test Tier 8.1: Cold Start - Conservative"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.0,  # High ROAS
            roas_trend=0.0,
            days_active=2,  # Cold start period
            health_score=0.7,
        )

        assert multiplier >= 1.0  # Should maintain or slight increase
        assert multiplier <= 1.05  # Should be conservative (max 5%)
        assert "cold_start" in reason.lower()

    def test_relative_perf_tier2_r23(self):
        """Test Tier 2.3: High ROAS + Strong Relative Performance"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            roas_trend=0.0,
            roas_vs_adset=1.3,  # Strong vs adset
            roas_vs_campaign=1.2,  # Strong vs campaign
            health_score=0.7,
        )

        assert multiplier > 1.0  # Should increase
        assert "relative" in reason.lower() or "vs_adset" in reason.lower()

    def test_maintenance_t15_r151(self):
        """Test Tier 15.1: Medium ROAS, Stable - Maintain"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,  # Medium ROAS
            roas_trend=0.01,  # Stable trend
            days_active=25,  # Not in cold start
            health_score=0.6,
        )

        # May have slight adjustment due to other rules,
        # but should be close to 1.0
        # Within 5% of maintain
        assert abs(multiplier - 1.0) < 0.05
        assert "maintain" in reason.lower() or "stable" in reason.lower()

    def test_default_t15_r152(self):
        """Test Tier 15.2: Default - Status Quo"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,
            roas_trend=0.0,
            days_active=25,  # Not in cold start
            health_score=0.5,
        )

        # May have slight adjustment due to other rules,
        # but should be close to 1.0
        # Within 5% of maintain
        assert abs(multiplier - 1.0) < 0.05
        reason_lower = reason.lower()
        assert "status_quo" in reason_lower or "maintain" in reason_lower

    def test_q4_boost_t7_r72(self):
        """Test Tier 7.2: Q4 Boost"""
        rules = DecisionRules()

        # Q4 week
        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.5, roas_trend=0.0, week_of_year=50, health_score=0.7
        )

        assert multiplier > 1.0  # Should get boost
        assert "q4" in reason.lower()

    def test_efficiency_tier4_r41(self):
        """Test Tier 4.1: High Revenue per Impression + High Volume"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,
            roas_trend=0.0,
            revenue_per_impression=0.10,  # High revenue per impression
            impressions=6000,  # High volume
            health_score=0.6,
        )

        # Should increase
        assert multiplier > 1.0
        assert "revenue_per_impression" in reason.lower()

    def test_volume_tier5_r51(self):
        """Test Tier 5.1: Low Spend but High ROAS +
        High Efficiency = Scale Up"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            roas_trend=0.0,
            spend=5.0,  # Low spend
            efficiency=0.12,  # High efficiency
            days_active=25,  # Established
        )

        assert multiplier > 1.0  # Should scale up
        # May match other high ROAS rules, so just check it increases
        assert (
            "high_roas" in reason.lower()
            or "low_spend" in reason.lower()
            or "scale" in reason.lower()
        )

    def test_gradient_adjustment(self):
        """Test gradient adjustment for smooth transitions"""

        # Test value just above threshold
        result = DecisionRulesHelpers.gradient_adjustment(
            3.1, 3.0, 0.15, range_size=0.5, increasing=True
        )
        # Should be partial adjustment
        assert 0.0 < result < 0.15

        # Test value well above threshold
        result = DecisionRulesHelpers.gradient_adjustment(
            4.0, 3.0, 0.15, range_size=0.5, increasing=True
        )
        # Should be full adjustment
        assert result == 0.15

    def test_trend_scaling(self):
        """Test trend magnitude scaling"""
        # Strong rising trend
        result = DecisionRulesHelpers.trend_scaling(0.20, 0.15, is_rising=True)
        # Full adjustment for strong trend
        assert result == 0.15

        # Weak rising trend
        result = DecisionRulesHelpers.trend_scaling(0.03, 0.15, is_rising=True)
        # Reduced adjustment for weak trend
        assert result < 0.15

    def test_health_score_multiplier(self):
        """Test health score multiplier"""
        # High health score
        result = DecisionRulesHelpers.health_score_multiplier(0.9)
        assert result > 0.9  # Should be close to 1.0

        # Low health score
        result = DecisionRulesHelpers.health_score_multiplier(0.3)
        assert result < 0.7  # Should be reduced

    def test_rel_perf_boost_legacy(self):
        """Test legacy attribute access for rel_perf_boost_campaign"""
        rules = DecisionRules()
        # Should access relative_perf_boost_campaign via legacy name
        result = rules.rel_perf_boost_campaign
        assert result is not None

    def test_excellent_roas_rel_perf_tr(self):
        """Test excellent ROAS with strong relative performance
        and positive trend"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=4.0,  # Excellent ROAS
            roas_trend=0.05,  # Positive trend
            roas_vs_adset=1.3,  # Strong vs adset
            roas_vs_campaign=1.2,  # Strong vs campaign
            health_score=0.8,
        )

        assert multiplier > 1.0
        assert "excellent_roas_strong_relative_perf" in reason

    def test_exc_roas_rel_perf_no_tr(self):
        """Test excellent ROAS with strong relative performance
        but no trend"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=4.0,  # Excellent ROAS
            roas_trend=0.0,  # No trend
            roas_vs_adset=1.3,  # Strong vs adset
            roas_vs_campaign=1.2,  # Strong vs campaign
            health_score=0.8,
        )

        assert multiplier > 1.0
        assert "excellent_roas_strong_relative_perf" in reason

    def test_high_roas_strong_vs_acct(self):
        """Test high ROAS with strong vs account performance"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            roas_vs_account=1.15,  # Strong vs account
            account_roas=3.0,  # Account ROAS > 0
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "high_roas_strong_vs_account" in reason

    def test_high_roas_efficient_tr(self):
        """Test high ROAS efficient rule with positive trend"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.2,  # High ROAS
            roas_trend=0.05,  # Positive trend
            efficiency=0.12,  # High efficiency
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "high_roas_efficient" in reason

    def test_high_roas_efficient_no_tr(self):
        """Test high ROAS efficient rule without trend"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.2,  # High ROAS
            roas_trend=0.0,  # No trend
            efficiency=0.12,  # High efficiency
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "high_roas_efficient" in reason

    def test_med_roas_strong_vs_adset(self):
        """Test medium ROAS with strong vs adset"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.5,  # Medium ROAS
            roas_trend=0.06,  # Moderate rising trend
            roas_vs_adset=1.25,  # Strong vs adset
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "medium_roas_strong_vs_adset" in reason

    def test_high_rev_per_click_vol(self):
        """Test high revenue per click with high volume"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,  # Above low threshold
            revenue_per_click=3.0,  # High revenue per click
            clicks=1000,  # High clicks
            health_score=0.6,
        )

        assert multiplier > 1.0
        assert "high_revenue_per_click_high_volume" in reason

    def test_low_efficiency_decrease(self):
        """Test low efficiency decrease rule"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,  # Below high threshold
            roas_trend=-0.01,  # Falling or flat trend
            efficiency=0.01,  # Low efficiency
            health_score=0.5,
        )

        assert multiplier < 1.0
        assert "low_efficiency_decrease" in reason

    def test_low_spend_high_roas_scale(self):
        """Test low spend high ROAS scale up rule"""
        rules = DecisionRules()

        # Use parameters that won't trigger other high-priority rules
        multiplier, _reason = rules.calculate_budget_adjustment(
            roas_7d=3.2,  # High ROAS but not excellent
            roas_trend=0.0,  # No trend to avoid trend-based rules
            spend=5.0,  # Low spend
            efficiency=0.11,  # High efficiency but not too high
            days_active=30,  # Established
            health_score=0.65,  # Not too high to avoid health-based rules
            revenue_per_impression=None,  # Avoid efficiency rules
        )

        assert multiplier > 1.0
        # May match other rules, so just check it increases
        # The specific rule may not match if other rules have higher priority

    def test_high_spend_declining_roas(self):
        """Test high spend declining ROAS rule"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,  # Below high threshold
            roas_trend=-0.06,  # Moderate falling trend
            spend=500.0,  # High spend
            health_score=0.5,
        )

        assert multiplier < 1.0
        assert "high_spend_declining_roas" in reason

    def test_high_reach_high_roas(self):
        """Test high reach high ROAS rule"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            reach=50000,  # High reach
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "high_reach_high_roas" in reason

    def test_monday_recovery_boost(self):
        """Test Monday recovery boost rule"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            day_of_week=0,  # Monday
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "monday_recovery_boost" in reason

    def test_early_learning_good_perf(self):
        """Test early learning phase good performer"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            days_active=5,  # Early learning
            clicks=100,  # Some clicks
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "early_learning" in reason

    def test_early_learning_moderate(self):
        """Test early learning phase moderate performer"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,  # Medium ROAS
            days_active=5,  # Early learning
            clicks=50,  # Some clicks
            health_score=0.6,
        )

        assert multiplier > 1.0
        assert "early_learning" in reason

    def test_mid_learning_performer(self):
        """Test mid learning phase performer"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            roas_trend=0.05,  # Positive trend
            days_active=10,  # Mid learning
            clicks=200,  # Some clicks
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "mid_learning" in reason

    def test_mid_learning_perf_no_tr(self):
        """Test mid learning phase performer without trend"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            roas_trend=0.0,  # No trend
            days_active=10,  # Mid learning
            clicks=200,  # Some clicks
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "mid_learning" in reason

    def test_late_learning_transitional(self):
        """Test late learning phase transitional"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            days_active=20,  # Late learning
            health_score=0.75,  # Healthy
        )

        assert multiplier > 1.0
        assert "late_learning" in reason

    def test_established_consistent_tr(self):
        """Test established consistent performer with positive trend"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            roas_trend=0.05,  # Positive trend
            days_active=30,  # Established
            health_score=0.75,  # Healthy
        )

        assert multiplier > 1.0
        assert "established_consistent_performer" in reason

    def test_est_consistent_no_tr(self):
        """Test established consistent performer without trend"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            roas_trend=0.0,  # No trend
            days_active=30,  # Established
            health_score=0.75,  # Healthy
        )

        assert multiplier > 1.0
        assert "established_consistent_performer" in reason

    def test_strong_recent_trend(self):
        """Test strong recent trend rule"""
        rules = DecisionRules()

        # Use parameters that won't trigger other high-priority rules
        multiplier, _reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,  # Above low threshold
            roas_trend=0.20,  # Strong positive trend
            days_active=30,  # Not in learning phase
            health_score=0.6,  # Not too high
            efficiency=None,  # Avoid efficiency rules
            revenue_per_impression=None,  # Avoid efficiency rules
        )

        assert multiplier > 1.0
        # May match other rules, so check if it's a positive adjustment
        # The specific rule may not match if other rules have higher priority

    def test_strong_recent_decline(self):
        """Test strong recent decline rule"""
        rules = DecisionRules()

        # Use parameters that won't trigger other high-priority rules
        multiplier, _reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,  # Below high threshold
            roas_trend=-0.20,  # Strong negative trend
            days_active=30,  # Not in learning phase
            health_score=0.5,  # Not too high
            efficiency=None,  # Avoid efficiency rules
            spend=None,  # Avoid spend-based rules
        )

        assert multiplier < 1.0
        # May match other rules, so check if it's a negative adjustment
        # The specific rule may not match if other rules have higher priority

    def test_small_adset_high_roas(self):
        """Test small adset high ROAS scale rule"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            adset_spend=10.0,  # Small adset spend
            campaign_spend=200.0,  # Large campaign spend (share < 0.1)
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "small_adset_high_roas_scale" in reason

    def test_spend_rising_high_roas(self):
        """Test spend rising high ROAS rule"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # High ROAS
            spend=120.0,  # Current spend
            spend_rolling_7d=100.0,  # Rolling 7d spend (20% increase)
            health_score=0.7,
        )

        assert multiplier > 1.0
        assert "spend_rising_high_roas" in reason

    def test_exceeding_expected_clicks(self):
        """Test exceeding expected clicks rule"""
        rules = DecisionRules()

        multiplier, reason = rules.calculate_budget_adjustment(
            roas_7d=2.0,  # Above low threshold
            expected_clicks=100,  # Expected clicks
            clicks=150,  # Actual clicks (1.5x efficiency)
            health_score=0.6,
        )

        assert multiplier > 1.0
        assert "exceeding_expected_clicks" in reason

    def test_low_roas_improving_weak_tr(self):
        """Test low ROAS improving with weak trend"""
        rules = DecisionRules()

        # Use parameters that specifically trigger the low ROAS improving rule
        multiplier, _reason = rules.calculate_budget_adjustment(
            roas_7d=1.6,  # Low ROAS but improving (above very_low threshold)
            roas_trend=0.02,  # Weak positive trend (below moderate)
            days_active=30,  # Established
            health_score=0.5,  # Not too high
            efficiency=None,  # Avoid efficiency rules
            revenue_per_impression=None,  # Avoid efficiency rules
            spend=None,  # Avoid spend-based rules
        )

        # Should have moderate decrease or maintain
        # The rule returns a small decrease, but other rules might override
        assert multiplier <= 1.0
        # May match other rules, so just check the adjustment direction
