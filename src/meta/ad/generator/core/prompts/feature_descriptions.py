"""
Feature Field Descriptions

Detailed descriptions of feature fields and their values, similar to analyzer.py.
These descriptions help GPT-4o understand feature meanings when generating prompts.
"""

# flake8: noqa
# pylint: disable=line-too-long

# Feature field descriptions for GPT-4o prompt generation
FEATURE_FIELD_DESCRIPTIONS = {
    "brightness_distribution": {
        "field_description": "How brightness is distributed across the image",
        "values": {
            "gradient": "Smooth gradient lighting with gradual transitions from bright highlights to soft shadows. Creates depth, dimension, and soft falloff.",
            "even": "Uniform, flat lighting with minimal contrast and even illumination across the image. Creates a clean, shadow-free commercial look.",
            "spotlight": "Dramatic spotlight effect with focused illumination on specific areas. Creates high contrast, dramatic emphasis, and deep shadows.",
        },
    },
    "visual_impact": {
        "field_description": "Overall visual intensity and energy of the image",
        "values": {
            "weak": "Subtle, understated visual presentation with muted tones and gentle contrast. Calm, minimalist, and professional appearance.",
            "moderate": "Balanced visual impact with moderate contrast and clear composition. Professional yet engaging commercial aesthetic.",
            "strong": "Bold, high-contrast imagery with dramatic visual impact, vibrant colors, and high dynamic range. Eye-catching and energetic.",
        },
    },
    "temperature": {
        "field_description": "Color temperature of the lighting",
        "values": {
            "warm": "Warm, Golden Hour lighting with amber and orange tones creating a cozy, inviting atmosphere. Typically 3000-4000K.",
            "cool": "Cool, Blue Hour or clean studio lighting reminiscent of overcast days or early morning. Typically 5000-6500K.",
            "neutral": "Neutral, balanced color temperature with natural white light. typically 4000-5000K. True-to-life color rendering.",
        },
    },
    "negative_space_usage": {
        "field_description": "Amount and use of empty space around the product",
        "values": {
            "generous": "Ample breathing room around the product with generous negative space (>40% of image). Minimalist, elegant, high-end look.",
            "balanced": "Balanced composition with appropriate negative space around the product (20-40% of image). Standard commercial spacing.",
            "minimal": "Minimal negative space with product filling most of the frame (10-20% empty space). Impactful, detailed product focus.",
            "cramped": "Tight composition with minimal space around elements (<10% empty space). Elements are crowded (avoid unless intentional).",
        },
    },
    "product_placement": {
        "field_description": "Position of the product within the frame",
        "values": {
            "center": "Product's center point is within 40% of image center. Symmetrical, balanced, hero-shot composition.",
            "top": "Product's center point is in top 40% of image height. Creates sense of elevation or floating.",
            "bottom": "Product's center point is in bottom 40% of image height. Grounded, stable appearance.",
            "left": "Product's center point is in left 40% of image width. Rule of thirds placement for natural reading flow.",
            "right": "Product's center point is in right 40% of image width. Rule of thirds placement for visual balance.",
        },
    },
    "relationship_depiction": {
        "field_description": "How the product is shown in relation to other elements",
        "values": {
            "product-alone": "Product shown without people. Keep the scene product-first: the product is the clear focal point, with at most minimal, non-distracting context props/surfaces (no accessory clutter).",
            "product-with-people": "Product shown with people interacting or nearby. Lifestyle context, human connection (ensure natural poses).",
            "product-with-objects": "Product shown with related objects or accessories. Contextual setting, product ecosystem (curated arrangement).",
            "product-in-environment": "Product shown within its natural environment or setting. Real-world context, practical use.",
        },
    },
    "content_storytelling": {
        "field_description": "Narrative elements and context in the image",
        "values": {
            "strong": "Strong narrative context with clear story elements. Tells a complete story about product use or benefit.",
            "moderate": "Moderate storytelling elements that provide context without overwhelming. Balanced narrative and product focus.",
            "weak": "Minimal storytelling with focus on product presentation. Product-first approach.",
            "none": "No storytelling elements, pure product focus. Minimal context or narrative.",
        },
    },
    "leading_lines": {
        "field_description": "Lines that guide the viewer's eye through the image",
        "values": {
            "strong": "Clear, bold lines (architecture, shadows, props) that strongly guide the eye to the product.",
            "moderate": "Some lines guide the eye but not prominently. Subtle directional guidance.",
            "weak": "Minimal lines that weakly guide the eye. Natural eye flow without strong direction.",
            "none": "No clear lines that guide the eye. Natural, unstructured composition.",
        },
    },
    "eye_tracking_path": {
        "field_description": "Pattern of eye movement through the image",
        "values": {
            "linear": "Gentle linear eye flow that guides attention from one point to another in a mostly straight path. Use layout/contrast/lighting cues.",
            "circular": "Circular eye-tracking path creating visual flow around the composition, keeping attention within the frame.",
            "z-pattern": "Z-pattern composition guiding eye movement in a Z-shape (top-left to top-right to bottom-left to bottom-right).",
            "random": "Natural, unstructured eye movement without clear pattern.",
        },
    },
    "human_elements": {
        "field_description": "Presence and type of human elements in the image",
        "values": {
            "lifestyle-context": "Subtle lifestyle context suggesting product use without showing identifiable people (no faces). Use environment/props/action cues rather than human subjects.",
            "face-visible": "Human faces visible in the image. Personal connection, human element prominent (ensure high quality faces).",
            "silhouette": "Human silhouettes or figures in background. Human presence without detail.",
            "none": "No human elements present. Pure product focus.",
        },
    },
    "depth_layers": {
        "field_description": "Perception of depth in the image",
        "values": {
            "flat": "Image appears two-dimensional, minimal depth perception. No clear foreground/background separation.",
            "shallow": "Slight depth perception. Shallow depth of field (f/2.8-f/4) with creamy bokeh blurring the background.",
            "moderate": "Clear depth perception, 3-4 distinct layers. Good foreground/midground/background separation.",
            "deep": "Strong depth perception, 4-5 distinct layers. Deep depth of field (f/8-f/11) with sharp focus throughout.",
            "very-deep": "Very strong depth perception, 5+ distinct layers. Maximum depth effect with deep perspective.",
        },
    },
}


def get_feature_field_description(feature_name: str) -> str:
    """
    Get field description for a feature.

    Args:
        feature_name: Name of the feature

    Returns:
        Field description or empty string
    """
    if feature_name in FEATURE_FIELD_DESCRIPTIONS:
        return FEATURE_FIELD_DESCRIPTIONS[feature_name].get(
            "field_description", ""
        )
    return ""


def get_feature_value_description(feature_name: str, value: str) -> str:
    """
    Get detailed description for a feature value.

    Args:
        feature_name: Name of the feature
        value: Feature value

    Returns:
        Detailed value description or empty string
    """
    value_lower = value.lower().replace("_", "-").replace(" ", "-")

    if feature_name in FEATURE_FIELD_DESCRIPTIONS:
        values = FEATURE_FIELD_DESCRIPTIONS[feature_name].get("values", {})
        if value_lower in values:
            return values[value_lower]

    return ""


def build_feature_descriptions_context(feature_values: dict) -> str:
    """
    Build context string with feature descriptions for GPT-4o.

    Args:
        feature_values: Dict mapping feature names to values

    Returns:
        Formatted context string with descriptions
    """
    context_parts = []

    for feature_name, value in feature_values.items():
        field_desc = get_feature_field_description(feature_name)
        value_desc = get_feature_value_description(feature_name, value)

        if field_desc or value_desc:
            part = f"- {feature_name.replace('_', ' ').title()}: {value}"
            if field_desc:
                part += f"\n  Field: {field_desc}"
            if value_desc:
                part += f"\n  Description: {value_desc}"
            context_parts.append(part)

    if context_parts:
        return "\n\n".join(context_parts)

    return ""
