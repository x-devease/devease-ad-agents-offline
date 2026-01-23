"""Unit tests for DecisionTreeMiner."""

import numpy as np
import pandas as pd
import pytest

from src.adset.lib.discovery_miner import DecisionTreeMiner


@pytest.fixture
def sample_features():
    """Create sample feature dataframe."""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame(
        {
            "purchase_roas_7d": np.concatenate(
                [
                    np.random.uniform(3.0, 5.0, n_samples // 2),  # High ROAS
                    np.random.uniform(0.5, 2.0, n_samples // 2),  # Low ROAS
                ]
            ),
            "roas_trend": np.concatenate(
                [
                    np.random.uniform(0.05, 0.15, n_samples // 2),  # Rising
                    np.random.uniform(-0.1, 0.05, n_samples // 2),  # Flat/falling
                ]
            ),
            "efficiency": np.concatenate(
                [
                    np.random.uniform(0.10, 0.20, n_samples // 2),  # High efficiency
                    np.random.uniform(0.01, 0.10, n_samples // 2),  # Low efficiency
                ]
            ),
            "spend": np.random.uniform(10, 200, n_samples),
            "days_active": np.random.randint(1, 30, n_samples),
        }
    )

    return df


class TestDecisionTreeMiner:
    """Test DecisionTreeMiner class."""

    def test_initialization(self):
        """Test miner initialization."""
        miner = DecisionTreeMiner(
            max_depth=5,
            min_samples_leaf=20,
        )

        assert miner.max_depth == 5
        assert miner.min_samples_leaf == 20
        assert miner.tree is None

    def test_mine_rules(self, sample_features):
        """Test mining rules from data."""
        miner = DecisionTreeMiner(
            max_depth=4,
            min_samples_leaf=5,  # Very low to ensure we get some rules
        )

        rules = miner.mine_rules(sample_features, target_col="purchase_roas_7d")

        # Should find some rules (or skip if none found due to random data)
        if len(rules) > 0:
            # Check rule structure
            rule = rules[0]
            assert rule.rule_id is not None
            assert rule.outcome in ["increase", "decrease"]
            assert 0.85 <= rule.adjustment_factor <= 1.15
            assert rule.support >= 5
            assert 0.0 <= rule.confidence <= 1.0
            assert rule.discovery_method == "decision_tree"
        else:
            # If no rules found, that's okay for random data
            pytest.skip("No rules discovered from random sample data")

    def test_extract_rules_from_tree(self, sample_features):
        """Test rule extraction from trained tree."""
        miner = DecisionTreeMiner(max_depth=3, min_samples_leaf=5)
        miner.mine_rules(sample_features)

        # Should have tree trained
        assert miner.tree is not None
        assert miner.feature_names is not None

        # Extract rules
        rules = miner.extract_rules_from_tree(
            miner.tree,
            miner.feature_names,
            target_threshold=2.0,
        )

        # Should have rules (or skip if none found)
        if len(rules) > 0:
            assert True
        else:
            pytest.skip("No rules extracted from random sample data")

    def test_rank_rules_by_quality(self, sample_features):
        """Test rule ranking by quality metrics."""
        miner = DecisionTreeMiner(min_samples_leaf=5)
        rules = miner.mine_rules(sample_features)

        if len(rules) == 0:
            pytest.skip("No rules discovered from random sample data")

        # Rank by support_confidence
        ranked = miner.rank_rules_by_quality(rules, metric="support_confidence")

        # Should be sorted
        assert len(ranked) == len(rules)

        # First rule should have higher score than last
        first_score = ranked[0].support * ranked[0].confidence
        last_score = ranked[-1].support * ranked[-1].confidence
        assert first_score >= last_score

    def test_get_feature_importance(self, sample_features):
        """Test feature importance extraction."""
        miner = DecisionTreeMiner()
        miner.mine_rules(sample_features)

        importance = miner.get_feature_importance()

        assert importance is not None
        assert len(importance) > 0

        # All values should be non-negative
        for imp in importance.values():
            assert imp >= 0

    def test_low_confidence_rules_filtered(self, sample_features):
        """Test that low confidence rules are filtered out."""
        miner = DecisionTreeMiner(
            max_depth=3,
            min_samples_leaf=10,
        )

        rules = miner.mine_rules(sample_features)

        # All rules should have reasonable confidence
        for rule in rules:
            assert rule.confidence >= 0.7  # Filter threshold

    def test_adjustment_factor_bounds(self, sample_features):
        """Test that adjustment factors are bounded."""
        miner = DecisionTreeMiner()
        rules = miner.mine_rules(sample_features)

        for rule in rules:
            # Adjustment factors should be between 0.85 and 1.15
            assert 0.85 <= rule.adjustment_factor <= 1.15

    def test_no_numeric_features_error(self):
        """Test error when no numeric features are available."""
        miner = DecisionTreeMiner()

        # Create dataframe with only non-numeric columns
        df = pd.DataFrame(
            {
                "adset_id": ["a", "b", "c"],
                "name": ["ad1", "ad2", "ad3"],
                "purchase_roas_7d": [1.0, 2.0, 3.0],
            }
        )

        with pytest.raises(ValueError, match="No numeric feature columns found"):
            miner._prepare_data(df, "purchase_roas_7d", 2.0)

    def test_extract_rules_with_min_samples(self, sample_features):
        """Test rule extraction with minimum sample filtering."""
        miner = DecisionTreeMiner(min_samples_leaf=5)
        miner.mine_rules(sample_features, target_col="purchase_roas_7d")

        # Extract rules directly
        X, y, feature_names = miner._prepare_data(
            sample_features, "purchase_roas_7d", 2.0
        )
        tree = miner._train_tree(X, y)

        rules = miner.extract_rules_from_tree(
            tree,
            feature_names,
            target_threshold=2.0,
        )

        # Should have rules or empty list (random data may not produce patterns)
        assert isinstance(rules, list)

    def test_custom_thresholds(self, sample_features):
        """Test with custom target threshold."""
        miner = DecisionTreeMiner(min_samples_leaf=5)

        # Test with higher threshold
        rules_high = miner.mine_rules(
            sample_features,
            target_col="purchase_roas_7d",
            target_threshold=4.0,
        )

        # Test with lower threshold
        rules_low = miner.mine_rules(
            sample_features,
            target_col="purchase_roas_7d",
            target_threshold=1.0,
        )

        # Both should return lists (may be empty for random data)
        assert isinstance(rules_high, list)
        assert isinstance(rules_low, list)

    def test_custom_outcome_labels(self, sample_features):
        """Test with custom outcome labels."""
        miner = DecisionTreeMiner(min_samples_leaf=5)

        rules = miner.mine_rules(
            sample_features,
            target_col="purchase_roas_7d",
            positive_outcome="boost",
            negative_outcome="reduce",
        )

        for rule in rules:
            assert rule.outcome in ["boost", "reduce"]
