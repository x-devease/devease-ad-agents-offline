"""
Feature constants shared across all converters.

Provides single source of truth for feature categorization.
"""

# Feature category mapping (superset from all converters)
CATEGORIES = {
    "lighting": [
        "brightness",
        "lighting_style",
        "mood_lighting",
        "brightness_distribution",
        "lighting_type",
        "temperature",
        "shadow_quality",
        "highlight_intensity",
    ],
    "composition": [
        "depth_layers",
        "framing",
        "product_angle",
        "visual_complexity",
        "focal_point_count",
        "rule_of_thirds",
        "symmetry_type",
        "perspective_type",
        "negative_space_usage",
        "product_placement",
        "leading_lines",
        "eye_tracking_path",
    ],
    "content": [
        "person_count",
        "person_gender",
        "person_activity",
        "activity_level",
        "relationship_depiction",
        "person_relationship_type",
        "person_age_group",
        "human_elements",
        "content_storytelling",
    ],
    "background": [
        "background_complexity",
        "background_type",
        "background_content_type",
        "scene_type",
        "specific_location",
    ],
    "visual_style": [
        "image_style",
        "visual_prominence",
        "object_diversity",
        "color_harmony",
        "visual_impact",
        "contrast_level",
    ],
    "product": [
        "product_presentation",
        "product_angle",
        "product_placement",
        "product_context",
    ],
}


def get_feature_category(feature_name: str) -> str:
    """
    Get category for a feature.

    Args:
        feature_name: Name of the feature to categorize

    Returns:
        Category name or "other" if not found
    """
    for category, features in CATEGORIES.items():
        if feature_name in features:
            return category
    return "other"
