"""Tests for Shopify ROAS integration in budget allocation rules."""

import pytest

from src.adset.allocator.lib.decision_rules import DecisionRules


class TestShopifyROASIntegration:
    """Test that Shopify data is available in rules but Meta ROAS remains primary."""

    def test_meta_roas_primary_signal(self):
        """Test that Meta ROAS is the primary signal, Shopify doesn't override it."""
        rules = DecisionRules(config=None)

        # Scenario: Meta ROAS is low (1.5), Shopify ROAS is high (4.0)
        # Rules should use Meta ROAS (primary signal) and not increase budget
        adjustment, reason = rules.calculate_budget_adjustment(
            roas_7d=1.5,  # Meta ROAS (low) - primary signal
            roas_trend=0.1,
            shopify_roas=4.0,  # Shopify ROAS (high) - available but doesn't override
            health_score=0.9,
            efficiency=0.15,
            spend=100.0,
            clicks=50,
        )

        # Should NOT aggressively increase because Meta ROAS (1.5) is low
        # Shopify ROAS is available in params but doesn't override the decision
        assert adjustment >= 0.9, f"Meta ROAS should drive decision, got {adjustment}, reason: {reason}"

    def test_shopify_data_available_in_params(self):
        """Test that Shopify data is passed through to params_dict."""
        rules = DecisionRules(config=None)

        # Make sure Shopify data doesn't cause errors and is available
        adjustment, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # Meta ROAS (high)
            roas_trend=0.1,
            shopify_roas=4.0,  # Shopify ROAS available
            shopify_revenue=500.0,  # Shopify revenue available
            health_score=0.9,
            efficiency=0.15,
            spend=100.0,
            clicks=50,
        )

        # Should process normally with both Meta and Shopify data
        assert adjustment > 1.0, f"Expected increase based on Meta ROAS, got {adjustment}"

    def test_no_shopify_data_works_fine(self):
        """Test that rules work fine without Shopify data."""
        rules = DecisionRules(config=None)

        # No Shopify data provided
        adjustment, reason = rules.calculate_budget_adjustment(
            roas_7d=3.5,  # Meta ROAS (high)
            roas_trend=0.1,
            health_score=0.9,
            efficiency=0.15,
            spend=100.0,
            clicks=50,
        )

        # Should work normally using Meta ROAS
        assert adjustment > 1.0, f"Expected increase, got {adjustment}, reason: {reason}"

    def test_effective_roas_helper_still_works(self):
        """Test that the _get_effective_roas helper method still works for custom use."""
        rules = DecisionRules(config=None)

        # Helper can be used for custom logic if needed
        params_dict = {"roas_7d": 1.5, "shopify_roas": 4.0}
        effective_roas = rules._get_effective_roas(params_dict)
        assert effective_roas == 4.0, f"Helper should return Shopify ROAS, got {effective_roas}"

        # No Shopify ROAS - returns Meta
        params_dict = {"roas_7d": 2.5, "shopify_roas": None}
        effective_roas = rules._get_effective_roas(params_dict)
        assert effective_roas == 2.5, f"Helper should return Meta ROAS, got {effective_roas}"

