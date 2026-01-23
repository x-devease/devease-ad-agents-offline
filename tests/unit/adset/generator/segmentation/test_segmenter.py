"""Unit tests for Segmenter with new audience types."""

import pytest
import pandas as pd
import numpy as np

from src.adset.generator.segmentation.segmenter import Segmenter

pytestmark = pytest.mark.filterwarnings(
    "ignore::UserWarning", "ignore::DeprecationWarning"
)


@pytest.fixture
def segmenter():
    """Create segmenter instance."""
    return Segmenter()


@pytest.fixture
def sample_rows():
    """Create sample rows for testing."""
    return {
        "exclusion": pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 3,
                "adset_targeting_custom_audiences_count": 2,
                "adset_targeting_advantage_audience": True,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 65,
            }
        ),
        "lookalike": pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 2,
                "adset_targeting_advantage_audience": False,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 65,
            }
        ),
        "advantage_geo": pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 0,
                "adset_targeting_advantage_audience": True,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 65,
                "adset_targeting_countries": "['US', 'TX', 'CA']",  # Top states
            }
        ),
        "advantage": pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 0,
                "adset_targeting_advantage_audience": True,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 65,
                "adset_targeting_countries": "['US']",  # National
            }
        ),
        "broad_geo": pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 0,
                "adset_targeting_advantage_audience": False,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 65,
                "adset_targeting_countries": "['US', 'FL', 'GA']",  # Top states
            }
        ),
        "broad": pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 0,
                "adset_targeting_advantage_audience": False,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 65,
                "adset_targeting_countries": "['US']",  # National
            }
        ),
        "interest": pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 0,
                "adset_targeting_advantage_audience": False,
                "adset_targeting_age_min": 25,
                "adset_targeting_age_max": 45,
            }
        ),
    }


class TestAudienceTypeClassification:
    """Test audience type classification with Shopify geo enrichment."""

    def test_exclusion_highest_priority(self, segmenter, sample_rows):
        """Test Exclusion type has highest priority."""
        result = segmenter.parse_audience_type(sample_rows["exclusion"])
        assert result == "Exclusion"

    def test_lookalike_second_priority(self, segmenter, sample_rows):
        """Test Lookalike type has second priority."""
        result = segmenter.parse_audience_type(sample_rows["lookalike"])
        assert result == "Lookalike"

    def test_advantage_geo_enriched(self, segmenter, sample_rows):
        """Test Advantage_Geo type (Advantage+ targeting top states)."""
        result = segmenter.parse_audience_type(sample_rows["advantage_geo"])
        assert result == "Advantage_Geo"

    def test_advantage_national(self, segmenter, sample_rows):
        """Test Advantage type (Advantage+ national targeting)."""
        result = segmenter.parse_audience_type(sample_rows["advantage"])
        assert result == "Advantage"

    def test_broad_geo_enriched(self, segmenter, sample_rows):
        """Test Broad_Geo type (Broad targeting top states)."""
        result = segmenter.parse_audience_type(sample_rows["broad_geo"])
        assert result == "Broad_Geo"

    def test_broad_national(self, segmenter, sample_rows):
        """Test Broad type (Broad national targeting)."""
        result = segmenter.parse_audience_type(sample_rows["broad"])
        assert result == "Broad"

    def test_interest_fallback(self, segmenter, sample_rows):
        """Test Interest type as fallback."""
        result = segmenter.parse_audience_type(sample_rows["interest"])
        assert result == "Interest"

    def test_age_threshold_45(self, segmenter):
        """Test that age threshold is 45 (not 40)."""
        # Age range = 44 should be Interest
        row = pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 0,
                "adset_targeting_advantage_audience": False,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 62,  # Range = 44
                "adset_targeting_countries": "['US']",
            }
        )
        result = segmenter.parse_audience_type(row)
        assert result == "Interest"  # Age range < 45

        # Age range = 45 should be Broad
        row["adset_targeting_age_max"] = 63  # Range = 45
        result = segmenter.parse_audience_type(row)
        assert result == "Broad"

    def test_nan_handling(self, segmenter):
        """Test handling of NaN values."""
        row = pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": np.nan,
                "adset_targeting_custom_audiences_count": np.nan,
                "adset_targeting_advantage_audience": np.nan,
                "adset_targeting_age_min": np.nan,
                "adset_targeting_age_max": np.nan,
                "adset_targeting_countries": np.nan,
            }
        )
        result = segmenter.parse_audience_type(row)
        assert result == "Interest"

    def test_geo_detection_with_state_abbreviation(self, segmenter):
        """Test geo detection with state abbreviation (TX)."""
        row = pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 0,
                "adset_targeting_advantage_audience": True,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 65,
                "adset_targeting_countries": "['US', 'TX']",  # TX abbreviation
            }
        )
        result = segmenter.parse_audience_type(row)
        assert result == "Advantage_Geo"

    def test_geo_detection_with_state_name(self, segmenter):
        """Test geo detection with full state name (Texas)."""
        row = pd.Series(
            {
                "adset_targeting_excluded_custom_audiences_count": 0,
                "adset_targeting_custom_audiences_count": 0,
                "adset_targeting_advantage_audience": False,
                "adset_targeting_age_min": 18,
                "adset_targeting_age_max": 65,
                "adset_targeting_countries": "United States, Texas",  # Full name
            }
        )
        result = segmenter.parse_audience_type(row)
        assert result == "Broad_Geo"


class TestGeoFocusedDetection:
    """Test _is_geo_focused helper method."""

    def test_detects_california(self, segmenter):
        """Test detects California in countries string."""
        row = pd.Series({"adset_targeting_countries": "United States, California"})
        assert segmenter._is_geo_focused(row) is True

    def test_detects_texas_abbreviation(self, segmenter):
        """Test detects Texas abbreviation."""
        row = pd.Series({"adset_targeting_countries": "['US', 'TX']"})
        assert segmenter._is_geo_focused(row) is True

    def test_detects_multiple_top_states(self, segmenter):
        """Test detects multiple top states."""
        row = pd.Series({"adset_targeting_countries": "['US', 'CA', 'TX', 'FL']"})
        assert segmenter._is_geo_focused(row) is True

    def test_returns_false_for_small_states(self, segmenter):
        """Test returns False for states not in top 10."""
        row = pd.Series({"adset_targeting_countries": "['US', 'Delaware', 'Vermont']"})
        assert segmenter._is_geo_focused(row) is False

    def test_returns_false_for_missing(self, segmenter):
        """Test returns False when countries is None."""
        row = pd.Series({"adset_targeting_countries": None})
        assert segmenter._is_geo_focused(row) is False

    def test_returns_false_for_empty(self, segmenter):
        """Test returns False when countries is empty string."""
        row = pd.Series({"adset_targeting_countries": ""})
        assert segmenter._is_geo_focused(row) is False

    def test_case_insensitive(self, segmenter):
        """Test detection is case-insensitive."""
        row = pd.Series(
            {"adset_targeting_countries": "['us', 'Ca', 'Fl']"}
        )  # Mixed case
        assert segmenter._is_geo_focused(row) is True


class TestCreativeFormatClassification:
    """Test creative format classification."""

    def test_video_detection(self, segmenter):
        """Test video format detection."""
        row = pd.Series(
            {
                "video_30_sec_watched_actions": 50,
                "video_p100_watched_actions": 0,
            }
        )
        result = segmenter.parse_creative_format(row)
        assert result == "Video"

    def test_video_p100_detection(self, segmenter):
        """Test video format with p100 metric."""
        row = pd.Series(
            {
                "video_30_sec_watched_actions": 0,
                "video_p100_watched_actions": 25,
            }
        )
        result = segmenter.parse_creative_format(row)
        assert result == "Video"

    def test_image_fallback(self, segmenter):
        """Test image format as fallback."""
        row = pd.Series(
            {
                "video_30_sec_watched_actions": 0,
                "video_p100_watched_actions": 0,
            }
        )
        result = segmenter.parse_creative_format(row)
        assert result == "Image"


class TestGeographyParsing:
    """Test geography parsing."""

    def test_us_parsing(self, segmenter):
        """Test US country parsing."""
        row = pd.Series({"adset_targeting_countries": "['US']"})
        result = segmenter.parse_geography(row)
        assert result == "US"

    def test_multiple_countries(self, segmenter):
        """Test multiple countries (returns first)."""
        row = pd.Series({"adset_targeting_countries": "['US', 'CA', 'GB']"})
        result = segmenter.parse_geography(row)
        assert result == "US"

    def test_string_fallback(self, segmenter):
        """Test string fallback for countries."""
        row = pd.Series({"adset_targeting_countries": "US, CA"})
        result = segmenter.parse_geography(row)
        assert "US" in result

    def test_unknown_handling(self, segmenter):
        """Test unknown/missing geography."""
        row = pd.Series({"adset_targeting_countries": None})
        result = segmenter.parse_geography(row)
        assert result == "Unknown"

        row = pd.Series({"adset_targeting_countries": ""})
        result = segmenter.parse_geography(row)
        assert result == "Unknown"


class TestSegmentationIntegration:
    """Test segmentation integration."""

    def test_segment_recommendations(self, segmenter):
        """Test segmenting recommendations DataFrame."""
        recommendations = pd.DataFrame(
            {
                "adset_id": ["adset_1", "adset_2", "adset_3"],
                "issue_type": ["age_range_too_wide", "missing_lal", "wasting"],
                "description": [
                    "Age range too wide",
                    "No LAL audiences",
                    "Low ROAS + high spend",
                ],
                "priority": ["MEDIUM", "MEDIUM", "CRITICAL"],
                "confidence": ["MEDIUM", "MEDIUM", "HIGH"],
                "current_spend": [50.0, 80.0, 200.0],
                "current_roas": [1.2, 0.8, 0.3],
                "suggested_action": [
                    "Test narrower ranges",
                    "Create LAL 1%",
                    "PAUSE immediately",
                ],
            }
        )

        adset_data = pd.DataFrame(
            {
                "adset_id": ["adset_1", "adset_2", "adset_3"],
                "adset_targeting_countries": ["['US']", "['US']", "['CA']"],
                "adset_targeting_custom_audiences_count": [0, 1, 0],
                "adset_targeting_excluded_custom_audiences_count": [0, 0, 2],
                "adset_targeting_age_min": [18, 25, 30],
                "adset_targeting_age_max": [65, 45, 50],
                "adset_targeting_advantage_audience": [False, True, False],
                "video_30_sec_watched_actions": [0, 50, 0],
                "video_p100_watched_actions": [0, 25, 0],
            }
        )

        segmented = segmenter.segment_recommendations(recommendations, adset_data)

        # Verify segmentation
        assert "geography" in segmented.columns
        assert "audience_type" in segmented.columns
        assert "creative_format" in segmented.columns
        assert len(segmented) == len(recommendations)

        # Verify audience types (enriched with Shopify geo data)
        valid_types = {
            "Exclusion",
            "Lookalike",
            "Advantage_Geo",
            "Advantage",
            "Broad_Geo",
            "Broad",
            "Interest",
        }
        assert segmented["audience_type"].iloc[0] in valid_types
        assert segmented["audience_type"].iloc[1] in valid_types
        assert segmented["audience_type"].iloc[2] in valid_types
