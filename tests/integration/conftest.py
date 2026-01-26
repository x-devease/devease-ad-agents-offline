"""
Shared pytest fixtures for integration tests.
"""

from pathlib import Path

import pytest
from src.meta.adset import DecisionRules, Allocator, SafetyRules
from src.meta.adset.allocator.utils.parser import Parser


@pytest.fixture
def config():
    """Load configuration from config/{customer}/{platform}/rules.yaml"""
    config_path = (
        Path(__file__).parent.parent.parent
        / "config"
        / "moprobo"
        / "meta"
        / "rules.yaml"
    )
    if not config_path.exists():
        pytest.skip(f"Configuration file not found: {config_path}")
    return Parser(
        config_path=str(config_path), customer_name="moprobo", platform="meta"
    )


@pytest.fixture
def allocator(request):
    """Create a fully configured allocator"""
    parser_config = request.getfixturevalue("config")
    safety_rules = SafetyRules(parser_config)
    decision_rules = DecisionRules(parser_config)
    return Allocator(safety_rules, decision_rules, parser_config)


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
