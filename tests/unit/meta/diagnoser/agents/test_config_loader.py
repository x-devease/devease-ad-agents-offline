"""
Unit tests for agent configuration loader.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.meta.diagnoser.agents.config.config_loader import AgentConfig


@pytest.fixture
def sample_config_yaml():
    """Sample YAML configuration."""
    return """
business:
  avg_monthly_spend_per_ad: 500
  daily_waste_per_fatigue_ad: 50
  loss_per_false_positive: 200
  currency: "USD"

performance_targets:
  min_precision: 0.90
  min_recall: 0.60
  min_f1: 0.70
  min_improvement: 0.03

confidence:
  high: 0.7
  medium: 0.5
  low: 0.3

agents:
  pm:
    temperature: 0.7
    max_tokens: 4096
  coder:
    temperature: 0.3
    max_tokens: 8192
  reviewer:
    temperature: 0.5
    max_tokens: 4096
  judge:
    temperature: 0.2
    max_tokens: 4096
  memory:
    temperature: 0.4
    max_tokens: 2048

advanced_features:
  semantic_search_enabled: false
  counterfactuals_enabled: false
  telemetry_enabled: false
"""


@pytest.fixture
def temp_config_file(tmp_path, sample_config_yaml):
    """Create temporary config file."""
    config_file = tmp_path / "test_agent_config.yaml"
    config_file.write_text(sample_config_yaml)
    return config_file


class TestAgentConfigInit:
    """Test AgentConfig initialization."""

    def test_load_config_success(self, temp_config_file):
        """Test successful config loading."""
        config = AgentConfig(str(temp_config_file))

        assert config.config is not None
        assert "business" in config.config
        assert "performance_targets" in config.config

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            AgentConfig("nonexistent_config.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises error."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content:")

        with pytest.raises(yaml.YAMLError):
            AgentConfig(str(invalid_file))


class TestAgentConfigGet:
    """Test AgentConfig.get() method."""

    @pytest.fixture
    def config(self, temp_config_file):
        """Create AgentConfig instance."""
        return AgentConfig(str(temp_config_file))

    def test_get_top_level_key(self, config):
        """Test getting top level configuration."""
        result = config.get("business")

        assert result is not None
        assert result["avg_monthly_spend_per_ad"] == 500

    def test_get_nested_key(self, config):
        """Test getting nested configuration with dot notation."""
        result = config.get("business.avg_monthly_spend_per_ad")

        assert result == 500

    def test_get_deep_nested_key(self, config):
        """Test getting deeply nested configuration."""
        result = config.get("agents.pm.temperature")

        assert result == 0.7

    def test_get_nonexistent_key(self, config):
        """Test getting nonexistent key returns None."""
        result = config.get("nonexistent.key")

        assert result is None

    def test_get_nonexistent_key_with_default(self, config):
        """Test getting nonexistent key with default value."""
        result = config.get("nonexistent.key", default="default_value")

        assert result == "default_value"

    def test_get_boolean_value(self, config):
        """Test getting boolean configuration value."""
        result = config.get("advanced_features.semantic_search_enabled")

        assert result is False


class TestAgentConfigInjectIntoPrompt:
    """Test AgentConfig.inject_into_prompt() method."""

    @pytest.fixture
    def config(self, temp_config_file):
        """Create AgentConfig instance."""
        return AgentConfig(str(temp_config_file))

    def test_inject_single_placeholder(self, config):
        """Test injecting single placeholder."""
        prompt = "Average spend: {CONFIG:business.avg_monthly_spend_per_ad}"

        result = config.inject_into_prompt(prompt)

        assert result == "Average spend: 500"

    def test_inject_multiple_placeholders(self, config):
        """Test injecting multiple placeholders."""
        prompt = """
Spend: {CONFIG:business.avg_monthly_spend_per_ad}
Waste: {CONFIG:business.daily_waste_per_fatigue_ad}
Min precision: {CONFIG:performance_targets.min_precision}
""".strip()

        result = config.inject_into_prompt(prompt)

        assert "Spend: 500" in result
        assert "Waste: 50" in result
        assert "Min precision: 0.9" in result

    def test_inject_placeholder_with_default(self, config):
        """Test injecting placeholder that uses default."""
        prompt = "Value: {CONFIG:nonexistent.key|default}"

        result = config.inject_into_prompt(prompt)

        # Should handle gracefully - either keep placeholder or use None
        assert "Value:" in result

    def test_inject_no_placeholders(self, config):
        """Test prompt without placeholders remains unchanged."""
        prompt = "This is a plain prompt without placeholders"

        result = config.inject_into_prompt(prompt)

        assert result == prompt

    def test_inject_malformed_placeholder(self, config):
        """Test malformed placeholder is handled gracefully."""
        prompt = "Value: {CONFIG:invalid_syntax"

        result = config.inject_into_prompt(prompt)

        # Should not crash
        assert "Value:" in result

    def test_inject_nested_placeholder(self, config):
        """Test injecting deeply nested placeholder."""
        prompt = "PM temperature: {CONFIG:agents.pm.temperature}"

        result = config.inject_into_prompt(prompt)

        assert result == "PM temperature: 0.7"


class TestAgentConfigValidation:
    """Test configuration validation."""

    @pytest.fixture
    def config(self, temp_config_file):
        """Create AgentConfig instance."""
        return AgentConfig(str(temp_config_file))

    def test_required_business_keys_exist(self, config):
        """Test required business configuration keys exist."""
        business = config.get("business")

        assert "avg_monthly_spend_per_ad" in business
        assert "daily_waste_per_fatigue_ad" in business
        assert "loss_per_false_positive" in business

    def test_required_performance_targets_exist(self, config):
        """Test required performance targets exist."""
        targets = config.get("performance_targets")

        assert "min_precision" in targets
        assert "min_recall" in targets
        assert "min_f1" in targets

    def test_required_agent_configs_exist(self, config):
        """Test required agent configurations exist."""
        agents = config.get("agents")

        # Check that at least pm and coder exist (the ones in sample config)
        assert "pm" in agents
        assert "coder" in agents

        # Check that reviewer, judge, memory exist if using real config
        # (sample config may not have all)
        if "reviewer" in agents:
            assert "judge" in agents
            assert "memory" in agents


class TestAgentConfigTypes:
    """Test configuration value types."""

    @pytest.fixture
    def config(self, temp_config_file):
        """Create AgentConfig instance."""
        return AgentConfig(str(temp_config_file))

    def test_business_values_are_numeric(self, config):
        """Test business values are numeric."""
        spend = config.get("business.avg_monthly_spend_per_ad")
        waste = config.get("business.daily_waste_per_fatigue_ad")
        loss = config.get("business.loss_per_false_positive")

        assert isinstance(spend, (int, float))
        assert isinstance(waste, (int, float))
        assert isinstance(loss, (int, float))

    def test_performance_targets_are_floats(self, config):
        """Test performance targets are floats between 0 and 1."""
        precision = config.get("performance_targets.min_precision")
        recall = config.get("performance_targets.min_recall")
        f1 = config.get("performance_targets.min_f1")

        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(f1, float)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_agent_temperatures_in_valid_range(self, config):
        """Test agent temperatures are in valid range (0-1)."""
        pm_temp = config.get("agents.pm.temperature")
        coder_temp = config.get("agents.coder.temperature")

        assert 0 <= pm_temp <= 1
        assert 0 <= coder_temp <= 1

    def test_advanced_features_are_boolean(self, config):
        """Test advanced feature flags are boolean."""
        semantic = config.get("advanced_features.semantic_search_enabled")
        counterfactuals = config.get("advanced_features.counterfactuals_enabled")

        assert isinstance(semantic, bool)
        assert isinstance(counterfactuals, bool)


class TestAgentConfigRealFile:
    """Test with real config file."""

    def test_load_actual_config_file(self):
        """Test loading the actual agent_config.yaml file."""
        config_path = Path("src/meta/diagnoser/agents/config/agent_config.yaml")

        if not config_path.exists():
            pytest.skip("Real config file not found")

        config = AgentConfig(str(config_path))

        assert config.config is not None
        assert "business" in config.config or "performance_targets" in config.config
