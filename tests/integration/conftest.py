"""
Shared pytest fixtures for integration tests.
"""

from pathlib import Path

import pytest
from src.adset import DecisionRules, Allocator, SafetyRules
from src.adset.utils.parser import Parser


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
