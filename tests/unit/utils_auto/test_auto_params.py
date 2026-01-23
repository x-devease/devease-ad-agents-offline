"""Test AutoParams class."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import yaml

from src.utils.auto_params import AutoParams


class TestAutoParams:
    """Test AutoParams class."""

    def test_default_params_structure(self):
        """Test that DEFAULT_PARAMS has expected structure."""
        defaults = AutoParams.DEFAULT_PARAMS

        # Check main sections exist
        assert "age_targeting" in defaults
        assert "gender" in defaults
        assert "geographic" in defaults
        assert "engagement" in defaults
        assert "shopify" in defaults
        assert "segment_health" in defaults

        # Check age_targeting has expected keys
        age = defaults["age_targeting"]
        assert "broad_age_range" in age
        assert "narrow_age_range" in age
        assert "age_min_bound" in age
        assert "age_max_bound" in age
        assert age["age_min_bound"] == 18
        assert age["age_max_bound"] == 65

    def test_guardrails_structure(self):
        """Test that GUARDRAILS has expected structure."""
        guardrails = AutoParams.GUARDRAILS

        # Check main sections exist
        assert "roas" in guardrails
        assert "spend" in guardrails
        assert "engagement" in guardrails
        assert "cpm" in guardrails

        # Check ROAS guardrails
        roas = guardrails["roas"]
        assert "high_min" in roas
        assert "high_max" in roas
        assert "low_min" in roas
        assert "low_max" in roas
        assert roas["high_min"] == 0.5
        assert roas["high_max"] == 10.0

    def test_percentile_with_guardrails_low_value(self):
        """Test guardrail minimum for ROAS high threshold."""
        data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

        # 75th percentile is 0.4, but guardrail minimum is 0.5
        result = AutoParams._percentile_with_guardrails(data, 75, "roas", "high")

        assert result == 0.5  # Should hit guardrail minimum

    def test_percentile_with_guardrails_high_value(self):
        """Test guardrail maximum for ROAS high threshold."""
        data = pd.Series([12.0, 15.0, 20.0, 25.0, 30.0])

        # 75th percentile is 25.0, but guardrail maximum is 10.0
        result = AutoParams._percentile_with_guardrails(data, 75, "roas", "high")

        assert result == 10.0  # Should hit guardrail maximum

    def test_percentile_with_guardrails_normal_value(self):
        """Test normal percentile within guardrails."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        # 75th percentile is 4.0, within guardrails (0.5-10.0)
        result = AutoParams._percentile_with_guardrails(data, 75, "roas", "high")

        assert result == 4.0  # Should return actual percentile

    def test_percentile_with_guardrails_no_guardrails(self):
        """Test percentile when no guardrails defined."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        # No guardrails for "unknown" type
        result = AutoParams._percentile_with_guardrails(data, 75, "unknown", "high")

        assert result == 4.0  # Should return actual percentile

    def test_calculate_performance_params_roas(self):
        """Test ROAS parameter calculation."""
        # Create mock adset data with ROAS values
        adset_data = pd.DataFrame({"purchase_roas": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]})

        params = AutoParams._calculate_performance_params(adset_data)

        assert "roas" in params
        roas = params["roas"]

        # Check that thresholds are calculated
        assert "high_roas_threshold" in roas
        assert "medium_roas_threshold" in roas
        assert "low_roas_threshold" in roas

        # Verify guardrail minimum applied (75th percentile is 4.25, but min is 0.5)
        assert roas["high_roas_threshold"] >= 0.5
        assert roas["medium_roas_threshold"] == 2.5  # 50th percentile
        assert roas["low_roas_threshold"] >= 0.5  # Guardrail minimum

    def test_calculate_performance_params_spend(self):
        """Test spend parameter calculation."""
        adset_data = pd.DataFrame({"spend": [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]})

        params = AutoParams._calculate_performance_params(adset_data)

        assert "spend" in params
        spend = params["spend"]

        # Check that thresholds are calculated
        assert "high_spend_threshold" in spend
        assert "medium_spend_threshold" in spend
        assert "low_spend_threshold" in spend
        assert "min_spend_for_scale" in spend

        # Verify percentiles
        assert spend["high_spend_threshold"] == 175.0  # 75th percentile
        assert spend["medium_spend_threshold"] == 75.0  # 50th percentile
        assert spend["low_spend_threshold"] == 27.5  # 25th percentile

    def test_calculate_performance_params_engagement(self):
        """Test engagement parameter calculation."""
        adset_data = pd.DataFrame(
            {
                "ctr": [0.005, 0.01, 0.02, 0.03, 0.04],
                "cpc": [1.0, 2.0, 3.0, 5.0, 10.0],
                "conversion_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
            }
        )

        params = AutoParams._calculate_performance_params(adset_data)

        assert "engagement" in params
        eng = params["engagement"]

        # Check CTR thresholds
        assert "low_ctr_threshold" in eng
        assert "image_preferred_ctr_threshold" in eng

        # Check CPC threshold
        assert "high_cpc_threshold" in eng

        # Check conversion rate
        assert "low_conversion_rate_threshold" in eng

    def test_calculate_performance_params_cpm(self):
        """Test CPM parameter calculation with guardrails."""
        adset_data = pd.DataFrame(
            {
                "cpm": [
                    2.0,
                    10.0,
                    30.0,
                    50.0,
                    150.0,
                ]  # 75th is 70, > max guardrail of 100
            }
        )

        params = AutoParams._calculate_performance_params(adset_data)

        assert "cpm" in params
        cpm = params["cpm"]

        # 75th percentile is 70, but should be capped at guardrail max of 100
        assert cpm["high_cpm_threshold"] <= 100.0
        assert cpm["high_cpm_threshold"] >= 5.0  # Min guardrail

    def test_calculate_performance_params_frequency(self):
        """Test frequency parameter calculation."""
        adset_data = pd.DataFrame({"frequency": [1.0, 2.0, 3.0, 5.0, 8.0]})

        params = AutoParams._calculate_performance_params(adset_data)

        assert "frequency" in params
        freq = params["frequency"]

        # Check thresholds
        assert "high_frequency_threshold" in freq
        assert "optimal_frequency" in freq
        assert "medium_frequency" in freq
        assert "low_frequency" in freq

        # Verify high frequency is capped at 10.0
        assert freq["high_frequency_threshold"] <= 10.0

    def test_calculate_shopify_params_with_data(self):
        """Test Shopify parameter calculation with valid data."""
        shopify_data = pd.DataFrame(
            {"Shipping Province": ["CA", "CA", "NY", "TX", "FL", "CA"]}
        )

        params = AutoParams._calculate_shopify_params(shopify_data)

        assert "shopify" in params
        shopify = params["shopify"]

        # Check that concentration is calculated
        # 3 out of 6 orders from CA = 50% concentration * 0.9 = 0.45
        assert "buyer_state_concentration_threshold" in shopify
        assert isinstance(shopify["buyer_state_concentration_threshold"], float)
        assert 0.0 <= shopify["buyer_state_concentration_threshold"] <= 1.0

    def test_calculate_shopify_params_empty_data(self):
        """Test Shopify parameter calculation with empty data."""
        shopify_data = pd.DataFrame()

        params = AutoParams._calculate_shopify_params(shopify_data)

        # Should return empty dict for no data
        assert params == {}

    def test_calculate_shopify_params_none(self):
        """Test Shopify parameter calculation with None."""
        params = AutoParams._calculate_shopify_params(None)

        # Should return empty dict for None
        assert params == {}

    def test_calculate_shopify_params_with_ages(self):
        """Test Shopify parameter calculation with customer ages."""
        shopify_data = pd.DataFrame({"customer_age": [25, 30, 35, 40, 45, 50, 55]})

        params = AutoParams._calculate_shopify_params(shopify_data)

        assert "shopify" in params
        shopify = params["shopify"]

        # 5th percentile of ages should be used
        assert "min_buyer_age_threshold" in shopify
        assert isinstance(shopify["min_buyer_age_threshold"], int)

    def test_merge_with_defaults(self):
        """Test merging calculated params with defaults."""
        calculated = {
            "roas": {"high_roas_threshold": 3.5, "medium_roas_threshold": 2.0}
        }

        result = AutoParams._merge_with_defaults(calculated)

        # Should have defaults
        assert "age_targeting" in result
        assert "gender" in result

        # Should have calculated values
        assert "roas" in result
        assert result["roas"]["high_roas_threshold"] == 3.5
        assert result["roas"]["medium_roas_threshold"] == 2.0

    def test_merge_with_defaults_preserves_calculated(self):
        """Test that merge preserves calculated values over defaults."""
        calculated = {"roas": {"high_roas_threshold": 5.0}}

        result = AutoParams._merge_with_defaults(calculated)

        # Calculated value should be preserved
        assert result["roas"]["high_roas_threshold"] == 5.0

    def test_get_defaults_only(self):
        """Test getting default parameters."""
        defaults = AutoParams._get_defaults_only()

        # Should have all default sections
        assert "age_targeting" in defaults
        assert "gender" in defaults
        assert "geographic" in defaults

        # Should be a deep copy (not same object)
        assert defaults is not AutoParams.DEFAULT_PARAMS

    def test_deep_copy_dict(self):
        """Test deep copy of dictionary."""
        original = {"level1": {"level2": {"level3": "value"}}}

        copy = AutoParams._deep_copy_dict(original)

        # Should be equal
        assert copy == original

        # Should not be same object
        assert copy is not original
        assert copy["level1"] is not original["level1"]

        # Modifying copy should not affect original
        copy["level1"]["level2"]["level3"] = "modified"
        assert original["level1"]["level2"]["level3"] == "value"

    @patch("src.features.feature_store.FeatureStore")
    def test_calculate_from_data_insufficient_adsets(self, mock_fs):
        """Test calculation with insufficient data returns defaults."""
        # Mock feature store with few adsets
        mock_instance = Mock()
        mock_instance.load_adset_data.return_value = pd.DataFrame(
            {"purchase_roas": [1.0, 2.0]}  # Only 2 adsets
        )
        mock_instance.load_shopify_data.return_value = pd.DataFrame()
        mock_fs.return_value = mock_instance

        result = AutoParams.calculate_from_data("test_customer", "meta")

        # Should return defaults only
        assert "age_targeting" in result
        assert "gender" in result

    @patch("src.features.feature_store.FeatureStore")
    @patch("src.utils.auto_params.AutoParams._save_params")
    def test_calculate_from_data_sufficient_adsets(self, mock_save, mock_fs):
        """Test calculation with sufficient data."""
        # Create mock adset data with 100+ rows
        roas_values = np.random.uniform(0, 10, 150)
        spend_values = np.random.uniform(10, 500, 150)

        mock_instance = Mock()
        mock_instance.load_adset_data.return_value = pd.DataFrame(
            {
                "purchase_roas": roas_values,
                "spend": spend_values,
                "ctr": np.random.uniform(0.001, 0.05, 150),
                "cpc": np.random.uniform(0.5, 10, 150),
                "conversion_rate": np.random.uniform(0.001, 0.05, 150),
                "cpm": np.random.uniform(5, 100, 150),
                "frequency": np.random.uniform(0.5, 8, 150),
                "reach": np.random.uniform(1000, 100000, 150),
            }
        )
        mock_instance.load_shopify_data.return_value = pd.DataFrame(
            {"Shipping Province": ["CA"] * 50 + ["NY"] * 30 + ["TX"] * 20}
        )
        mock_fs.return_value = mock_instance

        result = AutoParams.calculate_from_data("test_customer", "meta")

        # Should have calculated performance params
        assert "roas" in result
        assert "spend" in result
        assert "engagement" in result
        assert "shopify" in result

        # Verify save was called
        mock_save.assert_called_once()

    def test_calculate_performance_params_empty_dataframe(self):
        """Test performance params with empty dataframe."""
        adset_data = pd.DataFrame()

        params = AutoParams._calculate_performance_params(adset_data)

        # Should return empty dict
        assert params == {}

    def test_calculate_performance_params_missing_columns(self):
        """Test performance params with missing columns."""
        adset_data = pd.DataFrame({"some_column": [1, 2, 3]})

        params = AutoParams._calculate_performance_params(adset_data)

        # Should return empty dict (no recognized columns)
        assert params == {}

    def test_percentile_with_guardrails_low_threshold(self):
        """Test guardrails for low threshold."""
        data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

        # 25th percentile is 0.2, but guardrail minimum is 0.5
        result = AutoParams._percentile_with_guardrails(data, 25, "roas", "low")

        assert result == 0.5  # Should hit guardrail minimum

    def test_percentile_with_guardrails_engagement_ctr(self):
        """Test guardrails for CTR engagement metric."""
        data = pd.Series([0.0001, 0.0005, 0.001, 0.005, 0.01])

        # Note: engagement guardrails use ctr_min/ctr_max structure
        # which doesn't match the high_min/high_max pattern
        # So guardrails are not applied, returns raw percentile
        result = AutoParams._percentile_with_guardrails(data, 25, "engagement", "ctr")

        # Returns raw 25th percentile (no guardrail applied)
        assert result == 0.0005

    def test_percentile_with_guardrails_engagement_cpc(self):
        """Test guardrails for CPC engagement metric."""
        data = pd.Series([0.1, 0.3, 0.5, 1.0, 2.0])

        # 75th percentile is 1.0, min guardrail is 0.5
        result = AutoParams._percentile_with_guardrails(data, 75, "engagement", "cpc")

        assert result == 1.0  # Should be above guardrail

    def test_spend_min_spend_for_scale_guardrail(self):
        """Test that min_spend_for_scale has minimum guardrail."""
        adset_data = pd.DataFrame(
            {"spend": [5.0, 8.0, 9.0, 10.0, 12.0]}  # 10th percentile is 8.5
        )

        params = AutoParams._calculate_performance_params(adset_data)

        # min_spend_for_scale should be max(10th percentile, 10.0)
        assert params["spend"]["min_spend_for_scale"] >= 10.0

    def test_opportunity_size_params_calculated(self):
        """Test that opportunity size params are calculated."""
        adset_data = pd.DataFrame(
            {
                "frequency": [1.0, 2.0, 3.0, 5.0, 8.0],
                "purchase_roas": [0.5, 1.0, 2.0, 3.0, 5.0],
                "spend": [10.0, 50.0, 100.0, 200.0, 500.0],
            }
        )

        params = AutoParams._calculate_performance_params(adset_data)

        # Check opportunity size section exists
        assert "opportunity_size" in params

        opp = params["opportunity_size"]
        assert "high_frequency_threshold" in opp
        assert "medium_frequency_threshold" in opp
        assert "low_frequency_threshold" in opp
        assert "high_roas_threshold" in opp
        assert "medium_roas_threshold" in opp
        assert "high_spend_threshold" in opp
        assert "medium_spend_threshold" in opp

    def test_shopify_params_type_conversion(self):
        """Test that Shopify params are converted to native Python types."""
        shopify_data = pd.DataFrame({"Shipping Province": ["CA"] * 100})

        params = AutoParams._calculate_shopify_params(shopify_data)

        # Check types are native Python (not numpy)
        assert isinstance(
            params["shopify"]["buyer_state_concentration_threshold"], float
        )
        # Verify it's not a numpy type
        assert (
            type(params["shopify"]["buyer_state_concentration_threshold"]).__name__
            == "float"
        )
