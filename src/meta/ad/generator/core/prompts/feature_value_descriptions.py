"""
Feature Value Descriptions

Maps feature values to natural language descriptions for better prompt generation.
"""

# Comprehensive mapping of feature values to descriptive phrases
FEATURE_VALUE_DESCRIPTIONS = {
    "brightness_distribution": {
        "gradient": (
            "smooth gradient lighting with gradual transitions from bright "
            "highlights to soft shadows"
        ),
        "even": "uniform, flat lighting with minimal contrast and even illumination",
        "spotlight": "dramatic spotlight effect with focused illumination",
    },
    "visual_impact": {
        "weak": "subtle, understated visual presentation with muted tones and gentle contrast",
        "moderate": "balanced visual impact with moderate contrast and clear composition",
        "strong": "bold, high-contrast imagery with dramatic visual impact and vibrant colors",
    },
    "temperature": {
        "warm": "warm, golden-hour lighting with amber and orange tones creating a cozy atmosphere",
        "cool": "cool, blue-toned lighting reminiscent of overcast days or early morning",
        "neutral": "neutral, balanced color temperature with natural white light",
    },
    "negative_space_usage": {
        "generous": "ample breathing room around the product with generous negative space",
        "balanced": "balanced composition with appropriate negative space around the product",
        "minimal": "minimal negative space with product filling most of the frame",
        "cramped": "tight composition with minimal space around elements",
    },
    "product_placement": {
        "center": "product centered in the frame as the primary focal point",
        "top": "product positioned in the upper portion of the frame",
        "bottom": "product positioned in the lower portion of the frame",
        "left": "product positioned on the left side of the frame",
        "right": "product positioned on the right side of the frame",
    },
    "relationship_depiction": {
        "product-alone": "product shown in isolation without people or other objects",
        "product-with-people": "product shown with people interacting or nearby",
        "product-with-objects": "product shown with related objects or accessories",
        "product-in-environment": "product shown within its natural environment or setting",
    },
    "content_storytelling": {
        "strong": "strong narrative context with clear story elements and context",
        "moderate": "moderate storytelling elements that provide context without overwhelming",
        "weak": "minimal storytelling with focus on product presentation",
        "none": "no storytelling elements, pure product focus",
    },
    "leading_lines": {
        "strong": "strong leading lines that guide the viewer's eye to the product",
        "moderate": "moderate leading lines that subtly guide attention",
        "weak": "weak or subtle leading lines with natural eye flow",
        "none": "no prominent leading lines, natural composition",
    },
    "eye_tracking_path": {
        "linear": "linear eye-tracking path that guides attention naturally",
        "circular": "circular eye-tracking path creating visual flow",
        "z-pattern": "Z-pattern composition guiding eye movement",
        "random": "natural, unstructured eye movement",
    },
    "human_elements": {
        "lifestyle-context": "subtle lifestyle context elements suggesting product use",
        "face-visible": "human faces visible in the image",
        "silhouette": "human silhouettes or figures in background",
        "none": "no human elements present",
    },
}

# Category groupings for natural language organization
FEATURE_CATEGORIES = {
    "lighting": [
        "brightness_distribution",
        "temperature",
        "lighting_type",
        "shadow_quality",
        "highlight_intensity",
    ],
    "composition": [
        "product_placement",
        "negative_space_usage",
        "depth_layers",
        "leading_lines",
        "eye_tracking_path",
        "rule_of_thirds",
    ],
    "content": [
        "relationship_depiction",
        "human_elements",
        "content_storytelling",
        "product_context",
    ],
    "visual_style": [
        "visual_impact",
        "image_style",
        "color_harmony",
        "contrast_level",
    ],
}


def get_feature_description(feature_name: str, value: str) -> str:
    """
    Get natural language description for a feature value.

    Args:
        feature_name: Name of the feature
        value: Feature value

    Returns:
        Natural language description
    """
    # Normalize value
    value_lower = value.lower().replace("_", "-").replace(" ", "-")

    if feature_name in FEATURE_VALUE_DESCRIPTIONS:
        descriptions = FEATURE_VALUE_DESCRIPTIONS[feature_name]
        if value_lower in descriptions:
            return descriptions[value_lower]

    # Fallback: format value as readable text
    return value.replace("_", " ").replace("-", " ").title()


def get_feature_category(feature_name: str) -> str:
    """
    Get category for a feature.

    Args:
        feature_name: Name of the feature

    Returns:
        Category name or "other"
    """
    for category, features in FEATURE_CATEGORIES.items():
        if feature_name in features:
            return category
    return "other"


def build_natural_language_prompt(
    base_prompt: str,
    feature_values: dict,
    include_technical: bool = True,
) -> str:
    """
    Build natural language prompt from base prompt and feature values.

    Args:
        base_prompt: Base product description
        feature_values: Dict mapping feature names to values
        include_technical: Whether to include technical photography terms

    Returns:
        Natural language prompt string
    """
    # Group features by category
    category_features = {
        "lighting": [],
        "composition": [],
        "content": [],
        "visual_style": [],
        "other": [],
    }

    for feature_name, value in feature_values.items():
        category = get_feature_category(feature_name)
        description = get_feature_description(feature_name, value)
        category_features[category].append(description)

    # Build prompt parts
    prompt_parts = [base_prompt]

    # Add lighting description
    if category_features["lighting"]:
        lighting_text = ", ".join(category_features["lighting"])
        prompt_parts.append(f"The image features {lighting_text}.")

    # Add composition description
    if category_features["composition"]:
        composition_text = ", ".join(category_features["composition"])
        prompt_parts.append(f"Composition: {composition_text}.")

    # Add content description
    if category_features["content"]:
        content_text = ", ".join(category_features["content"])
        prompt_parts.append(f"Content: {content_text}.")

    # Add visual style
    if category_features["visual_style"]:
        style_text = ", ".join(category_features["visual_style"])
        prompt_parts.append(f"Visual style: {style_text}.")

    # Add technical specifications if requested
    if include_technical:
        prompt_parts.append(
            "Shot with a full-frame camera and 50mm lens at f/2.8 aperture, "
            "with realistic depth of field and natural material imperfections."
        )

    return " ".join(prompt_parts)
