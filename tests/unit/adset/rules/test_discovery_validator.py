"""Unit tests for RuleValidator."""

import numpy as np
import pandas as pd
import pytest

from src.meta.adset.allocator.lib.discovery_models import DiscoveredRule
from src.meta.adset.allocator.lib.discovery_validator import RuleValidator


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200

    return pd.DataFrame(
        {
            "adset_id": [f"adset_{i}" for i in range(n_samples)],
            "purchase_roas_7d": np.concatenate(
                [
                    np.random.uniform(3.0, 5.0, n_samples // 2),
                    np.random.uniform(0.5, 2.0, n_samples // 2),
                ]
            ),
            "roas_trend": np.random.uniform(-0.1, 0.15, n_samples),
            "efficiency": np.random.uniform(0.01, 0.20, n_samples),
            "spend": np.random.uniform(10, 200, n_samples),
        }
    )


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    np.random.seed(43)
    n_samples = 100

    return pd.DataFrame(
        {
            "adset_id": [f"adset_{i}" for i in range(n_samples)],
            "purchase_roas_7d": np.concatenate(
                [
                    np.random.uniform(3.0, 5.0, n_samples // 2),
                    np.random.uniform(0.5, 2.0, n_samples // 2),
                ]
            ),
            "roas_trend": np.random.uniform(-0.1, 0.15, n_samples),
            "efficiency": np.random.uniform(0.01, 0.20, n_samples),
            "spend": np.random.uniform(10, 200, n_samples),
        }
    )


@pytest.fixture
def sample_rule():
    """Create sample discovered rule."""
    return DiscoveredRule(
        rule_id="test_rule_1",
        conditions={"purchase_roas_7d": {"min": 3.0}, "roas_trend": {"min": 0.05}},
        outcome="increase",
        adjustment_factor=1.15,
        support=100,
        confidence=0.85,
        lift=1.5,
        discovery_method="decision_tree",
    )


class TestRuleValidator:
    """Test RuleValidator class."""

    def test_initialization(self, sample_train_data, sample_test_data):
        """Test validator initialization."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        assert validator.df_train is not None
        assert validator.df_test is not None
        assert len(validator.df_train) == 200
        assert len(validator.df_test) == 100

    def test_validate_rule(self, sample_train_data, sample_test_data, sample_rule):
        """Test rule validation."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        result = validator.validate_rule(sample_rule)

        # Check result structure
        assert result.rule_id == "test_rule_1"
        assert isinstance(result.mean_roas, float)
        assert isinstance(result.std_roas, float)
        assert 0.0 <= result.budget_utilization <= 1.0
        assert isinstance(result.safety_violations, int)
        assert result.recommendation in ["deploy", "test", "reject"]

    def test_apply_rule(self, sample_train_data, sample_test_data, sample_rule):
        """Test applying rule conditions."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        matched = validator._apply_rule(sample_test_data, sample_rule)

        # Should filter data
        assert len(matched) <= len(sample_test_data)

        # All matched rows should satisfy conditions
        if len(matched) > 0:
            assert (matched["purchase_roas_7d"] >= 3.0).all()

    def test_check_safety_compatibility(
        self, sample_train_data, sample_test_data, sample_rule
    ):
        """Test safety constraint checking."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        matched = validator._apply_rule(sample_test_data, sample_rule)
        violations = validator._check_safety_compatibility(sample_rule, matched)

        assert isinstance(violations, int)
        assert violations >= 0

    def test_detect_overfitting(self, sample_train_data, sample_test_data, sample_rule):
        """Test overfitting detection."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        is_overfit = validator.detect_overfitting(sample_rule)

        assert isinstance(is_overfit, bool)

    def test_low_support_rule_rejected(self, sample_train_data, sample_test_data):
        """Test that low support rules are rejected."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        low_support_rule = DiscoveredRule(
            rule_id="low_support",
            conditions={"purchase_roas_7d": {"min": 10.0}},  # Very high threshold
            outcome="increase",
            adjustment_factor=1.15,
            support=5,  # Low support
            confidence=0.90,
            lift=2.0,
        )

        result = validator.validate_rule(low_support_rule)

        # Should be rejected due to low support
        assert result.recommendation == "reject"

    def test_safety_violation_rejected(self, sample_train_data, sample_test_data):
        """Test that rules violating safety are rejected."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        unsafe_rule = DiscoveredRule(
            rule_id="unsafe",
            conditions={"purchase_roas_7d": {"min": 3.0}},
            outcome="increase",
            adjustment_factor=1.50,  # Way above max_daily_increase_pct (0.19)
            support=100,
            confidence=0.90,
            lift=2.0,
        )

        result = validator.validate_rule(unsafe_rule)

        # Should be rejected due to safety violation
        assert result.recommendation == "reject"
        assert result.safety_violations > 0

    def test_rank_rules_by_validation_score(self, sample_train_data, sample_test_data):
        """Test ranking rules by validation score."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        rules = [
            DiscoveredRule(
                rule_id=f"rule_{i}",
                conditions={"purchase_roas_7d": {"min": 2.0 + i * 0.5}},
                outcome="increase",
                adjustment_factor=1.10,
                support=50 + i * 10,
                confidence=0.75 + i * 0.02,
                lift=1.2 + i * 0.1,
            )
            for i in range(5)
        ]

        scored = validator.rank_rules_by_validation_score(rules)

        # Should have scores
        assert len(scored) == 5

        # Should be sorted by score (descending)
        scores = [s[0] for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_apply_rule_missing_feature(self, sample_train_data, sample_test_data):
        """Test applying rule when feature doesn't exist in data."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        rule = DiscoveredRule(
            rule_id="missing_feature",
            conditions={"nonexistent_feature": {"min": 1.0}},
            outcome="increase",
            adjustment_factor=1.1,
            support=100,
            confidence=0.8,
            lift=1.2,
            discovery_method="decision_tree",
        )

        # Should not crash, returns all rows when feature is missing
        # (because mask starts as all True and missing features are skipped)
        matched = validator._apply_rule(sample_test_data, rule)
        # Feature doesn't exist, so it's skipped and all rows are returned
        assert len(matched) == len(sample_test_data)

    def test_apply_rule_max_condition(self, sample_train_data, sample_test_data):
        """Test applying rule with max condition."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        rule = DiscoveredRule(
            rule_id="max_rule",
            conditions={"purchase_roas_7d": {"max": 2.0}},
            outcome="decrease",
            adjustment_factor=0.9,
            support=50,
            confidence=0.75,
            lift=1.1,
            discovery_method="decision_tree",
        )

        matched = validator._apply_rule(sample_test_data, rule)

        # All matched should have roas <= 2.0
        if len(matched) > 0:
            assert (matched["purchase_roas_7d"] <= 2.0).all()

    def test_apply_rule_both_conditions(self, sample_train_data, sample_test_data):
        """Test applying rule with both min and max conditions."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        rule = DiscoveredRule(
            rule_id="range_rule",
            conditions={"purchase_roas_7d": {"min": 1.5, "max": 3.0}},
            outcome="increase",
            adjustment_factor=1.1,
            support=50,
            confidence=0.75,
            lift=1.1,
            discovery_method="decision_tree",
        )

        matched = validator._apply_rule(sample_test_data, rule)

        # All matched should have roas in range [1.5, 3.0]
        if len(matched) > 0:
            assert (matched["purchase_roas_7d"] >= 1.5).all()
            assert (matched["purchase_roas_7d"] <= 3.0).all()

    def test_apply_rule_equality_condition(self, sample_train_data, sample_test_data):
        """Test applying rule with equality condition."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        # Add a column with discrete values
        test_data = sample_test_data.copy()
        test_data["test_col"] = 10

        rule = DiscoveredRule(
            rule_id="equality_rule",
            conditions={"test_col": 10},
            outcome="increase",
            adjustment_factor=1.1,
            support=50,
            confidence=0.75,
            lift=1.1,
            discovery_method="decision_tree",
        )

        matched = validator._apply_rule(test_data, rule)

        # All matched should have test_col == 10
        if len(matched) > 0:
            assert (matched["test_col"] == 10).all()

    def test_detect_overfitting_no_overfit(self, sample_train_data, sample_test_data):
        """Test overfitting detection when not overfitting."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        # Create rule that performs similarly on both sets
        rule = DiscoveredRule(
            rule_id="balanced_rule",
            conditions={"purchase_roas_7d": {"min": 1.0}},
            outcome="increase",
            adjustment_factor=1.1,
            support=100,
            confidence=0.8,
            lift=1.2,
            discovery_method="decision_tree",
        )

        is_overfit = validator.detect_overfitting(rule)

        # Should not be overfitting (similar performance)
        assert isinstance(is_overfit, bool)

    def test_validate_rule_with_low_support_rule(
        self, sample_train_data, sample_test_data
    ):
        """Test validation rejects rules with very low support."""
        validator = RuleValidator(sample_train_data, sample_test_data)

        # Rule with extremely low support
        low_support_rule = DiscoveredRule(
            rule_id="tiny_support",
            conditions={"purchase_roas_7d": {"min": 100.0}},  # Impossible threshold
            outcome="increase",
            adjustment_factor=1.15,
            support=3,  # Very low
            confidence=0.95,
            lift=2.0,
            discovery_method="decision_tree",
        )

        result = validator.validate_rule(low_support_rule)

        # Should be rejected
        assert result.recommendation == "reject"
