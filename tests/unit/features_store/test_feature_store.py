"""Test FeatureStore class."""

import pytest
import pandas as pd
import numpy as np
from src.adset.features.feature_store import FeatureStore
from src.utils import Config


class TestFeatureStore:
    """Test FeatureStore class."""

    @pytest.fixture
    def feature_store(self):
        """Create feature store instance."""
        return FeatureStore()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "ad_id": [f"ad_{i}" for i in range(n_samples)],
            "adset_id": [f"adset_{i % 10}" for i in range(n_samples)],
            "spend": np.random.uniform(10, 1000, n_samples),
            "impressions": np.random.uniform(100, 10000, n_samples),
            "age_min": np.random.randint(18, 50, n_samples),
            "age_max": np.random.randint(50, 65, n_samples),
            "gender": np.random.choice(["M", "F", "Unknown"], n_samples),
            "purchase_roas": np.random.exponential(2.0, n_samples),
        }
        return pd.DataFrame(data)

    def test_initialization(self, feature_store):
        """Test feature store initialization."""
        assert feature_store.scalers == {}
        assert feature_store.encoders == {}
        assert feature_store.feature_names == []

    def test_filter_valid_data(self, feature_store, sample_dataframe):
        """Test data filtering."""
        filtered = feature_store._filter_valid_data(sample_dataframe)

        # All rows should have spend > MIN_SPEND and impressions > MIN_IMPRESSIONS
        assert all(filtered["spend"] > Config.MIN_SPEND())
        assert all(filtered["impressions"] > Config.MIN_IMPRESSIONS())

    def test_preprocess_features_fit(self, feature_store, sample_dataframe):
        """Test feature preprocessing with fit=True."""
        df = sample_dataframe.copy()

        # Add required columns (must be in Config.NUMERICAL_FEATURES)
        df["budget"] = df["spend"] * 1.2
        df["bid_amount"] = df["spend"] * 0.1

        # Use only features defined in Config
        feature_columns = ["age_min", "age_max", "budget", "bid_amount"]
        df_features = df[feature_columns]

        X, feature_names = feature_store.preprocess_features(df_features, fit=True)

        assert X.shape[0] == len(df_features)
        assert len(feature_names) == len(feature_columns)
        assert len(feature_store.scalers) > 0

    def test_preprocess_features_transform(self, feature_store, sample_dataframe):
        """Test feature preprocessing with fit=False."""
        df = sample_dataframe.copy()

        # Add required columns (must be in Config.NUMERICAL_FEATURES)
        df["budget"] = df["spend"] * 1.2
        df["bid_amount"] = df["spend"] * 0.1

        # Use only features defined in Config
        feature_columns = ["age_min", "age_max", "budget", "bid_amount"]
        df_features = df[feature_columns]

        # First fit
        X_train, _ = feature_store.preprocess_features(df_features, fit=True)

        # Then transform
        X_test, _ = feature_store.preprocess_features(df_features, fit=False)

        assert X_test.shape == X_train.shape

    def test_create_interaction_features(self, feature_store, sample_dataframe):
        """Test interaction feature creation."""
        df = sample_dataframe.copy()
        df["budget"] = df["spend"] * 1.2
        df["campaign_lifetime_spend"] = df["spend"] * 5

        result = feature_store.create_interaction_features(df)

        assert "budget_spend_ratio" in result.columns
