"""Test TargetTransformer class."""

import pytest
import numpy as np
from src.utils.target_transformer import TargetTransformer


class TestTargetTransformer:
    """Test TargetTransformer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample target data."""
        np.random.seed(42)
        # Positive values only (for boxcox compatibility)
        y = np.random.exponential(scale=2.0, size=100)
        return y

    @pytest.fixture
    def mixed_sign_data(self):
        """Create data with mixed positive and negative values."""
        np.random.seed(42)
        y = np.random.randn(100)
        return y

    def test_initialization(self):
        """Test transformer initialization."""
        transformer = TargetTransformer(method="log1p")
        assert transformer.method == "log1p"
        assert not transformer.is_fitted

    def test_fit_log1p(self, sample_data):
        """Test fitting with log1p method."""
        transformer = TargetTransformer(method="log1p")
        transformer.fit(sample_data)

        assert transformer.is_fitted

    def test_fit_sqrt(self, sample_data):
        """Test fitting with sqrt method."""
        transformer = TargetTransformer(method="sqrt")
        transformer.fit(sample_data)

        assert transformer.is_fitted

    def test_fit_boxcox(self, sample_data):
        """Test fitting with boxcox method."""
        transformer = TargetTransformer(method="boxcox")
        transformer.fit(sample_data)

        assert transformer.is_fitted
        assert transformer.lambda_ is not None

    def test_fit_yeo_johnson(self, mixed_sign_data):
        """Test fitting with yeo-johnson method."""
        transformer = TargetTransformer(method="yeo-johnson")
        transformer.fit(mixed_sign_data)

        assert transformer.is_fitted
        assert transformer.fitted_transformer is not None

    def test_fit_auto(self, sample_data):
        """Test automatic method selection."""
        transformer = TargetTransformer(method="auto")
        transformer.fit(sample_data)

        assert transformer.is_fitted
        assert transformer.method in ["log1p", "sqrt", "yeo-johnson"]

    def test_transform_log1p(self, sample_data):
        """Test log1p transform."""
        transformer = TargetTransformer(method="log1p")
        transformer.fit(sample_data)

        y_transformed = transformer.transform(sample_data)

        expected = np.log1p(sample_data)
        np.testing.assert_array_almost_equal(y_transformed, expected)

    def test_transform_sqrt(self, sample_data):
        """Test sqrt transform."""
        transformer = TargetTransformer(method="sqrt")
        transformer.fit(sample_data)

        y_transformed = transformer.transform(sample_data)

        expected = np.sqrt(sample_data)
        np.testing.assert_array_almost_equal(y_transformed, expected)

    def test_transform_boxcox(self, sample_data):
        """Test boxcox transform."""
        transformer = TargetTransformer(method="boxcox")
        transformer.fit(sample_data)

        y_transformed = transformer.transform(sample_data)

        # Box-Cox should produce roughly normal distribution
        assert y_transformed is not None
        assert len(y_transformed) == len(sample_data)
        assert np.all(np.isfinite(y_transformed))

    def test_transform_yeo_johnson(self, mixed_sign_data):
        """Test yeo-johnson transform."""
        transformer = TargetTransformer(method="yeo-johnson")
        transformer.fit(mixed_sign_data)

        y_transformed = transformer.transform(mixed_sign_data)

        assert y_transformed is not None
        assert len(y_transformed) == len(mixed_sign_data)
        assert np.all(np.isfinite(y_transformed))

    def test_inverse_transform_log1p(self, sample_data):
        """Test inverse log1p transform."""
        transformer = TargetTransformer(method="log1p")
        transformer.fit(sample_data)

        y_transformed = transformer.transform(sample_data)
        y_original = transformer.inverse_transform(y_transformed)

        np.testing.assert_array_almost_equal(y_original, sample_data, decimal=10)

    def test_inverse_transform_sqrt(self, sample_data):
        """Test inverse sqrt transform."""
        transformer = TargetTransformer(method="sqrt")
        transformer.fit(sample_data)

        y_transformed = transformer.transform(sample_data)
        y_original = transformer.inverse_transform(y_transformed)

        np.testing.assert_array_almost_equal(y_original, sample_data, decimal=10)

    def test_inverse_transform_boxcox(self, sample_data):
        """Test inverse boxcox transform."""
        transformer = TargetTransformer(method="boxcox")
        transformer.fit(sample_data)

        y_transformed = transformer.transform(sample_data)
        y_original = transformer.inverse_transform(y_transformed)

        np.testing.assert_array_almost_equal(y_original, sample_data, decimal=5)

    def test_inverse_transform_yeo_johnson(self, mixed_sign_data):
        """Test inverse yeo-johnson transform."""
        transformer = TargetTransformer(method="yeo-johnson")
        transformer.fit(mixed_sign_data)

        y_transformed = transformer.transform(mixed_sign_data)
        y_original = transformer.inverse_transform(y_transformed)

        np.testing.assert_array_almost_equal(y_original, mixed_sign_data, decimal=5)

    def test_error_transform_without_fit(self, sample_data):
        """Test error when transforming without fitting."""
        transformer = TargetTransformer(method="log1p")

        with pytest.raises(ValueError, match="must be fitted"):
            transformer.transform(sample_data)

    def test_error_inverse_transform_without_fit(self, sample_data):
        """Test error when inverse transforming without fitting."""
        transformer = TargetTransformer(method="log1p")

        with pytest.raises(ValueError, match="must be fitted"):
            transformer.inverse_transform(sample_data)

    def test_boxcox_handles_negative_values(self, sample_data):
        """Test that boxcox shifts negative values to positive."""
        # Add negative values
        y_negative = sample_data - sample_data.min() - 1

        transformer = TargetTransformer(method="boxcox")
        transformer.fit(y_negative)

        assert transformer.is_fitted
        assert hasattr(transformer, "shift")

        y_transformed = transformer.transform(y_negative)
        assert np.all(np.isfinite(y_transformed))

    def test_unknown_method_raises_error(self, sample_data):
        """Test that unknown method raises error."""
        transformer = TargetTransformer(method="unknown_method")
        transformer.fit(sample_data)

        with pytest.raises(ValueError, match="Unknown method"):
            transformer.transform(sample_data)

    def test_save_and_load(self, sample_data, tmp_path):
        """Test transformer save and load."""
        transformer = TargetTransformer(method="boxcox")
        transformer.fit(sample_data)

        # Save
        transformer_path = tmp_path / "transformer.joblib"
        transformer.save(transformer_path)

        assert transformer_path.exists()

        # Load
        loaded_transformer = TargetTransformer.load(transformer_path)

        assert loaded_transformer.is_fitted
        assert loaded_transformer.method == transformer.method
        assert loaded_transformer.lambda_ == transformer.lambda_

        # Transforms should match
        y_transformed_original = transformer.transform(sample_data)
        y_transformed_loaded = loaded_transformer.transform(sample_data)

        np.testing.assert_array_almost_equal(
            y_transformed_original, y_transformed_loaded
        )

    def test_zero_values_handled(self):
        """Test handling of zero values."""
        y = np.array([0, 1, 2, 3, 4, 5])

        # log1p should handle zeros
        transformer = TargetTransformer(method="log1p")
        transformer.fit(y)
        y_transformed = transformer.transform(y)

        assert np.all(np.isfinite(y_transformed))

        # sqrt should handle zeros
        transformer = TargetTransformer(method="sqrt")
        transformer.fit(y)
        y_transformed = transformer.transform(y)

        assert np.all(np.isfinite(y_transformed))

    def test_normality_test_in_auto_selection(self, sample_data):
        """Test that auto selection uses normality test."""
        transformer = TargetTransformer(method="auto")
        transformer.fit(sample_data)

        # Should have selected a method
        assert transformer.method != "auto"
        assert transformer.is_fitted
