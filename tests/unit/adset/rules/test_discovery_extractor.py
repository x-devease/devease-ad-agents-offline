"""Unit tests for RuleExtractor."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.adset.allocator.lib.discovery_models import DiscoveredRule
from src.adset.allocator.lib.discovery_extractor import RuleExtractor


@pytest.fixture
def sample_rules():
    """Create sample discovered rules."""
    return [
        DiscoveredRule(
            rule_id="rule_1",
            conditions={
                "purchase_roas_7d": {"min": 3.0, "max": 10.0},
                "roas_trend": {"min": 0.05},
            },
            outcome="increase",
            adjustment_factor=1.15,
            support=100,
            confidence=0.85,
            lift=1.5,
            discovery_method="decision_tree",
        ),
        DiscoveredRule(
            rule_id="rule_2",
            conditions={"purchase_roas_7d": {"max": 1.5}},
            outcome="decrease",
            adjustment_factor=0.90,
            support=80,
            confidence=0.80,
            lift=1.2,
            discovery_method="decision_tree",
        ),
    ]


class TestRuleExtractor:
    """Test RuleExtractor class."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = RuleExtractor()
        assert extractor.config_schema == {}

    def test_convert_to_decision_rule(self, sample_rules):
        """Test converting DiscoveredRule to DecisionRules format."""
        extractor = RuleExtractor()
        rule = sample_rules[0]

        converted = extractor.convert_to_decision_rule(rule)

        # Check structure
        assert "rule_id" in converted
        assert "rule_name" in converted
        assert "conditions" in converted
        assert "action" in converted
        assert "metadata" in converted

        # Check conditions format
        assert "purchase_roas_7d" in converted["conditions"]
        assert "roas_trend" in converted["conditions"]

        # Check action
        assert converted["action"]["adjustment_factor"] == 1.15
        assert "reason" in converted["action"]

    def test_generate_rule_name(self, sample_rules):
        """Test rule name generation."""
        extractor = RuleExtractor()
        rule = sample_rules[0]

        name = extractor._generate_rule_name(rule)

        assert isinstance(name, str)
        assert len(name) > 0
        assert "high_performer" in name or "low_performer" in name

    def test_format_conditions(self, sample_rules):
        """Test condition formatting."""
        extractor = RuleExtractor()
        rule = sample_rules[0]

        formatted = extractor._format_conditions(rule.conditions)

        assert "purchase_roas_7d" in formatted
        assert "roas_trend" in formatted

        # Check min/max preserved
        assert formatted["purchase_roas_7d"]["min"] == 3.0
        assert formatted["purchase_roas_7d"]["max"] == 10.0
        assert formatted["roas_trend"]["min"] == 0.05

    def test_generate_yaml_config(self, sample_rules):
        """Test YAML config generation."""
        extractor = RuleExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rules.yaml"

            extractor.generate_yaml_config(sample_rules, str(output_path))

            # Check file exists
            assert output_path.exists()

            # Load and check content
            with open(output_path, "r") as f:
                yaml_content = yaml.safe_load(f)

            assert "discovered_decision_rules" in yaml_content
            assert len(yaml_content["discovered_decision_rules"]) == 2

    def test_prioritize_rules(self, sample_rules):
        """Test rule prioritization by tier."""
        extractor = RuleExtractor()

        prioritized = extractor.prioritize_rules(sample_rules)

        # All rules should have tiers
        for rule in prioritized:
            assert rule.tier is not None

        # Should be sorted by tier
        tiers = [r.tier for r in prioritized]
        assert tiers == sorted(tiers)

    def test_merge_with_existing_config(self, sample_rules):
        """Test merging with existing config."""
        extractor = RuleExtractor()

        existing_config = {
            "decision_rules": {
                "very_low_roas_threshold": 1.0,
                "high_roas_threshold": 3.0,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing config file
            existing_path = Path(tmpdir) / "existing.yaml"
            with open(existing_path, "w") as f:
                yaml.dump(existing_config, f)

            output_path = Path(tmpdir) / "merged.yaml"

            # Merge
            extractor.merge_with_existing_config(
                sample_rules, str(existing_path), str(output_path)
            )

            # Check merged file
            with open(output_path, "r") as f:
                merged = yaml.safe_load(f)

            # Should have original rules
            assert "decision_rules" in merged
            assert merged["decision_rules"]["very_low_roas_threshold"] == 1.0

            # Should have new discovered rules
            assert "discovered_decision_rules" in merged
            assert len(merged["discovered_decision_rules"]) == 2

    def test_generate_rule_name_empty_conditions(self):
        """Test rule name generation with empty conditions."""
        extractor = RuleExtractor()

        rule = DiscoveredRule(
            rule_id="test_empty",
            conditions={},
            outcome="increase",
            adjustment_factor=1.1,
            support=100,
            confidence=0.8,
            lift=1.2,
            discovery_method="decision_tree",
        )

        name = extractor._generate_rule_name(rule)
        assert name == "high_performer_discovered_rule"

    def test_generate_yaml_config_safety_rules(self, sample_rules):
        """Test generating YAML for safety rules config type."""
        extractor = RuleExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "safety_rules.yaml"
            extractor.generate_yaml_config(
                sample_rules, str(output_path), config_type="safety_rules"
            )

            with open(output_path, "r") as f:
                content = yaml.safe_load(f)

            assert "discovered_safety_rules" in content
            assert len(content["discovered_safety_rules"]) == 2

    def test_generate_yaml_config_custom_type(self, sample_rules):
        """Test generating YAML with custom config type."""
        extractor = RuleExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom.yaml"
            extractor.generate_yaml_config(
                sample_rules, str(output_path), config_type="custom_rules"
            )

            with open(output_path, "r") as f:
                content = yaml.safe_load(f)

            assert "discovered_custom_rules" in content
            assert len(content["discovered_custom_rules"]) == 2

    def test_prioritize_rules_all_tiers(self, sample_rules):
        """Test that rules get prioritized into different tiers."""
        extractor = RuleExtractor()

        # Create rules with different quality levels
        rules = [
            # Excellent performer
            DiscoveredRule(
                rule_id="excellent",
                conditions={"roas": {"min": 5.0}},
                outcome="increase",
                adjustment_factor=1.15,
                support=100,
                confidence=0.95,
                lift=2.0,
                discovery_method="decision_tree",
            ),
            # Good performer
            DiscoveredRule(
                rule_id="good",
                conditions={"roas": {"min": 3.0}},
                outcome="increase",
                adjustment_factor=1.1,
                support=80,
                confidence=0.85,
                lift=1.4,
                discovery_method="decision_tree",
            ),
            # Low performer
            DiscoveredRule(
                rule_id="low",
                conditions={"roas": {"max": 1.0}},
                outcome="decrease",
                adjustment_factor=0.9,
                support=50,
                confidence=0.9,
                lift=1.3,
                discovery_method="decision_tree",
            ),
            # Default
            DiscoveredRule(
                rule_id="default",
                conditions={"roas": {"min": 2.0}},
                outcome="increase",
                adjustment_factor=1.05,
                support=30,
                confidence=0.75,
                lift=1.1,
                discovery_method="decision_tree",
            ),
        ]

        prioritized = extractor.prioritize_rules(rules)

        # Check that different tiers are assigned
        tiers = [r.tier for r in prioritized]
        assert len(set(tiers)) > 1  # Should have multiple tiers

        # Should be sorted by tier
        for i in range(len(prioritized) - 1):
            assert prioritized[i].tier <= prioritized[i + 1].tier
