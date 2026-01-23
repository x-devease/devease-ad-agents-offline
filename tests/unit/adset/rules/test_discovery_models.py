"""Unit tests for pattern discovery data models."""

from src.adset.allocator.lib.discovery_models import DiscoveredRule, ValidationResult


class TestDiscoveredRule:
    """Test DiscoveredRule data model."""

    def test_create_rule(self):
        """Test creating a discovered rule."""
        rule = DiscoveredRule(
            rule_id="test_rule_1",
            conditions={"purchase_roas_7d": {"min": 3.0}, "roas_trend": {"min": 0.05}},
            outcome="increase",
            adjustment_factor=1.15,
            support=100,
            confidence=0.85,
            lift=1.5,
            discovery_method="decision_tree",
        )

        assert rule.rule_id == "test_rule_1"
        assert rule.outcome == "increase"
        assert rule.adjustment_factor == 1.15
        assert rule.support == 100
        assert rule.confidence == 0.85
        assert rule.lift == 1.5
        assert rule.discovery_method == "decision_tree"

    def test_to_dict(self):
        """Test converting rule to dictionary."""
        rule = DiscoveredRule(
            rule_id="test_rule_1",
            conditions={"purchase_roas_7d": {"min": 3.0}},
            outcome="increase",
            adjustment_factor=1.15,
            support=100,
            confidence=0.85,
            lift=1.5,
        )

        rule_dict = rule.to_dict()

        assert rule_dict["rule_id"] == "test_rule_1"
        assert rule_dict["outcome"] == "increase"
        assert rule_dict["adjustment_factor"] == 1.15
        assert rule_dict["support"] == 100
        assert "discovery_date" in rule_dict

    def test_from_dict(self):
        """Test creating rule from dictionary."""
        rule_dict = {
            "rule_id": "test_rule_1",
            "conditions": {"purchase_roas_7d": {"min": 3.0}},
            "outcome": "increase",
            "adjustment_factor": 1.15,
            "support": 100,
            "confidence": 0.85,
            "lift": 1.5,
            "feature_importance": {},
            "discovery_date": "2024-01-01T00:00:00",
            "discovery_method": "decision_tree",
            "validation_metric": None,
            "tier": None,
            "metadata": {},
        }

        rule = DiscoveredRule.from_dict(rule_dict)

        assert rule.rule_id == "test_rule_1"
        assert rule.outcome == "increase"
        assert rule.adjustment_factor == 1.15


class TestValidationResult:
    """Test ValidationResult data model."""

    def test_create_validation_result(self):
        """Test creating a validation result."""
        result = ValidationResult(
            rule_id="test_rule_1",
            mean_roas=2.5,
            std_roas=0.5,
            budget_utilization=0.8,
            safety_violations=0,
            recommendation="deploy",
            validation_samples=100,
        )

        assert result.rule_id == "test_rule_1"
        assert result.mean_roas == 2.5
        assert result.std_roas == 0.5
        assert result.budget_utilization == 0.8
        assert result.safety_violations == 0
        assert result.recommendation == "deploy"
        assert result.validation_samples == 100

    def test_to_dict(self):
        """Test converting validation result to dictionary."""
        result = ValidationResult(
            rule_id="test_rule_1",
            mean_roas=2.5,
            std_roas=0.5,
            budget_utilization=0.8,
            safety_violations=0,
            recommendation="deploy",
            validation_samples=100,
        )

        result_dict = result.to_dict()

        assert result_dict["rule_id"] == "test_rule_1"
        assert result_dict["mean_roas"] == 2.5
        assert result_dict["recommendation"] == "deploy"

    def test_from_dict(self):
        """Test creating validation result from dictionary."""
        result_dict = {
            "rule_id": "test_rule_1",
            "mean_roas": 2.5,
            "std_roas": 0.5,
            "budget_utilization": 0.8,
            "safety_violations": 0,
            "recommendation": "deploy",
            "validation_samples": 100,
            "forward_roas": None,
            "backtest_roas": None,
            "confidence_interval": None,
            "metadata": {},
        }

        result = ValidationResult.from_dict(result_dict)

        assert result.rule_id == "test_rule_1"
        assert result.mean_roas == 2.5
        assert result.recommendation == "deploy"
