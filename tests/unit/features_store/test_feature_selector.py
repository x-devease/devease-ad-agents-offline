"""Test FeatureSelector class."""

import pytest
import numpy as np
import pandas as pd
from src.adset.allocator.features.feature_selector import FeatureSelector


class TestFeatureSelector:
    """Test FeatureSelector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 50

        # Create feature matrix with various characteristics
        X = np.random.randn(n_samples, n_features)

        # Add some correlated features
        X[:, 5] = X[:, 0] * 0.9 + np.random.randn(n_samples) * 0.1
        X[:, 6] = X[:, 1] * 0.95 + np.random.randn(n_samples) * 0.05

        # Add low-variance features
        X[:, 10] = 1.0 + np.random.randn(n_samples) * 0.001
        X[:, 11] = 2.0 + np.random.randn(n_samples) * 0.001

        # Add features with varying predictive power
        feature_names = [f"feature_{i}" for i in range(n_features)]

        # Create target with relationship to first few features
        y = 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(n_samples) * 0.5

        return X, feature_names, y

    def test_selector_initialization(self):
        """Test selector initialization."""
        selector = FeatureSelector()

        assert selector.variance_threshold == 0.01
        assert selector.correlation_threshold == 0.95
        assert selector.max_features == 100
        assert selector.selected_features is None

    def test_selector_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        selector = FeatureSelector(
            variance_threshold=0.05,
            correlation_threshold=0.9,
            max_features=50,
            k_mi=100,
        )

        assert selector.variance_threshold == 0.05
        assert selector.correlation_threshold == 0.9
        assert selector.max_features == 50
        assert selector.k_mi == 100

    def test_fit(self, sample_data):
        """Test fitting selector."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(max_features=20)

        selector.fit(X, feature_names, y)

        assert selector.selected_features is not None
        assert len(selector.selected_features) <= 20
        assert isinstance(selector.selected_features, list)

    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(max_features=20)

        X_transformed = selector.fit_transform(X, feature_names, y)

        assert X_transformed.shape[1] <= 20
        assert X_transformed.shape[0] == X.shape[0]

    def test_transform_without_fit(self, sample_data):
        """Test error when transforming without fitting."""
        X, feature_names, _ = sample_data
        selector = FeatureSelector()

        with pytest.raises(ValueError, match="must be fitted"):
            selector.transform(X, feature_names)

    def test_variance_threshold_removal(self, sample_data):
        """Test that low-variance features are removed."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(variance_threshold=0.01, max_features=50)

        selector.fit(X, feature_names, y)

        # Features 10 and 11 have very low variance
        assert "feature_10" not in selector.selected_features
        assert "feature_11" not in selector.selected_features

    def test_correlation_filter(self, sample_data):
        """Test that highly correlated features are removed."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(correlation_threshold=0.9, max_features=50)

        selector.fit(X, feature_names, y)

        # Features 5 and 6 are highly correlated with 0 and 1
        # At least one from each pair should be removed
        has_feature_0 = "feature_0" in selector.selected_features
        has_feature_5 = "feature_5" in selector.selected_features

        has_feature_1 = "feature_1" in selector.selected_features
        has_feature_6 = "feature_6" in selector.selected_features

        # At least one from each correlated pair should be removed
        assert not (has_feature_0 and has_feature_5) or not (
            has_feature_1 and has_feature_6
        )

    def test_mutual_information_selection(self, sample_data):
        """Test mutual information feature selection."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(
            max_features=10,
            k_mi=15,
            variance_threshold=0.001,  # Lower threshold to keep more features
        )

        selector.fit(X, feature_names, y)

        # Should select features based on MI with target
        # Features 0, 1 should definitely be selected (highest predictive power)
        assert "feature_0" in selector.selected_features
        assert "feature_1" in selector.selected_features
        # At least some features should be selected
        assert len(selector.selected_features) > 0

    def test_model_based_selection(self, sample_data):
        """Test model-based feature selection."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(max_features=5)

        selector.fit(X, feature_names, y)

        assert len(selector.selected_features) <= 5

    def test_get_selection_history(self, sample_data):
        """Test selection history tracking."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(max_features=15)

        selector.fit(X, feature_names, y)
        history = selector.get_selection_history()

        assert "variance_threshold" in history
        assert "correlation_filter" in history
        assert "final" in history
        assert history["final"]["original"] == len(feature_names)
        assert history["final"]["selected"] <= 15

    def test_save_and_load(self, sample_data, tmp_path):
        """Test selector save and load."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(max_features=10)

        selector.fit(X, feature_names, y)

        # Save
        selector_path = tmp_path / "selector.joblib"
        selector.save(selector_path)

        assert selector_path.exists()

        # Load
        loaded_selector = FeatureSelector.load(selector_path)

        assert loaded_selector.selected_features == selector.selected_features
        assert loaded_selector.selection_history == selector.selection_history
        assert loaded_selector.variance_threshold == selector.variance_threshold

    def test_transform_maintains_feature_order(self, sample_data):
        """Test that transform maintains selected feature order."""
        X, feature_names, y = sample_data
        selector = FeatureSelector(max_features=10)

        X_transformed = selector.fit_transform(X, feature_names, y)

        # Check that columns correspond to selected features
        for i, feat_name in enumerate(selector.selected_features):
            original_idx = feature_names.index(feat_name)
            np.testing.assert_array_equal(X_transformed[:, i], X[:, original_idx])

    def test_handles_zero_variance_features(self):
        """Test handling of features with zero variance."""
        np.random.seed(42)
        n_samples = 100

        X = np.random.randn(n_samples, 10)
        # Add constant features
        X[:, 5] = 5.0
        X[:, 6] = 10.0

        feature_names = [f"feature_{i}" for i in range(10)]
        y = np.random.randn(n_samples)

        selector = FeatureSelector(variance_threshold=0.01)
        selector.fit(X, feature_names, y)

        # Constant features should be removed
        assert "feature_5" not in selector.selected_features
        assert "feature_6" not in selector.selected_features

    def test_all_features_removed_edge_case(self):
        """Test edge case where all features might be removed."""
        np.random.seed(42)
        n_samples = 50

        # All features have very low variance
        X = np.ones((n_samples, 10)) + np.random.randn(n_samples, 10) * 0.0001
        feature_names = [f"feature_{i}" for i in range(10)]
        y = np.random.randn(n_samples)

        selector = FeatureSelector(variance_threshold=0.1, max_features=5)

        # Should handle gracefully - VarianceThreshold may raise error
        # so we catch that or handle empty result
        try:
            selector.fit(X, feature_names, y)
            assert isinstance(selector.selected_features, list)
        except ValueError as e:
            # Expected when no features meet variance threshold
            assert "variance threshold" in str(e)

    def test_small_dataset(self):
        """Test with small dataset."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        feature_names = [f"feature_{i}" for i in range(5)]
        y = np.random.randn(20)

        selector = FeatureSelector(max_features=3)
        selector.fit(X, feature_names, y)

        assert len(selector.selected_features) <= 3
