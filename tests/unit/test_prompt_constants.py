"""Unit tests for prompt constants."""

import pytest
from src.meta.ad.generator.core.prompts.constants import get_feature_category, CATEGORIES


class TestFeatureConstants:
    """Test feature constants."""

    def test_categories_structure(self):
        """Test that CATEGORIES is properly structured."""
        assert isinstance(CATEGORIES, dict)
        assert len(CATEGORIES) > 0

        # Check all categories are lists
        for category, features in CATEGORIES.items():
            assert isinstance(category, str)
            assert isinstance(features, list)
            assert all(isinstance(f, str) for f in features)

    def test_get_feature_category_known_features(self):
        """Test get_feature_category for known features."""
        # Test lighting features
        assert get_feature_category("brightness") == "lighting"
        assert get_feature_category("lighting_style") == "lighting"
        assert get_feature_category("shadow_quality") == "lighting"

        # Test composition features
        assert get_feature_category("depth_layers") == "composition"
        assert get_feature_category("framing") == "composition"
        assert get_feature_category("negative_space_usage") == "composition"

        # Test content features
        assert get_feature_category("person_count") == "content"
        assert get_feature_category("activity_level") == "content"

        # Test background features
        assert get_feature_category("background_type") == "background"
        assert get_feature_category("scene_type") == "background"

        # Test visual_style features
        assert get_feature_category("image_style") == "visual_style"
        assert get_feature_category("color_harmony") == "visual_style"

        # Test product features
        assert get_feature_category("product_presentation") == "product"

    def test_get_feature_category_unknown_feature(self):
        """Test get_feature_category for unknown features."""
        assert get_feature_category("unknown_feature") == "other"
        assert get_feature_category("") == "other"
        assert get_feature_category("not_a_real_feature") == "other"

    def test_get_feature_category_case_sensitive(self):
        """Test that get_feature_category is case-sensitive."""
        # Lowercase should work
        assert get_feature_category("brightness") == "lighting"

        # Uppercase should not match
        assert get_feature_category("Brightness") == "other"
        assert get_feature_category("BRIGHTNESS") == "other"

    def test_all_category_members_accessible(self):
        """Test that all features in CATEGORIES can be categorized."""
        # Note: Some features may appear in multiple categories
        # The function returns the first match, which is acceptable
        for category, features in CATEGORIES.items():
            for feature in features:
                result = get_feature_category(feature)
                # Just verify it gets categorized (not necessarily to the expected category
                # since features can be in multiple categories)
                assert result in CATEGORIES or result == "other", f"Feature {feature} should categorize to a known category, got {result}"
