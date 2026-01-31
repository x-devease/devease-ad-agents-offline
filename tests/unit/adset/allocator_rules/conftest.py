"""
Shared pytest fixtures for unit tests in adset/rules.
"""

import pytest


@pytest.fixture
def sample_metrics_dict():
    """Sample metrics dictionary with all optional fields for testing."""
    return {
        "current_budget": 100.0,
        "previous_budget": 95.0,
        "roas_7d": 2.5,
        "roas_trend": 0.10,
        "adset_roas": 2.4,
        "campaign_roas": 2.3,
        "account_roas": 2.2,
        "roas_vs_adset": 1.04,
        "roas_vs_campaign": 1.09,
        "roas_vs_account": 1.14,
        "efficiency": 0.85,
        "revenue_per_impression": 0.025,
        "revenue_per_click": 2.5,
        "spend": 100.0,
        "spend_rolling_7d": 700.0,
        "impressions": 10000,
        "clicks": 100,
        "reach": 8000,
        "adset_spend": 100.0,
        "campaign_spend": 500.0,
        "expected_clicks": 110,
        "health_score": 0.75,
        "days_active": 25,
        "day_of_week": 1,
        "is_weekend": False,
        "week_of_year": 5,
        "num_ads": 3,
        "num_active_ads": 2,
        "marginal_roas": 2.3,
        "budget_utilization": 0.85,
        "total_budget_today": 1000.0,
    }


@pytest.fixture
def sample_frozen_metrics_dict():
    """Sample metrics dictionary for a frozen adset (underperforming)."""
    return {
        "current_budget": 0.0,
        "previous_budget": 0.0,
        "roas_7d": 0.3,  # Below freeze threshold
        "roas_trend": -0.20,
        "health_score": 0.15,  # Below freeze threshold
        "days_active": 15,
        "total_budget_today": 1000.0,
    }
