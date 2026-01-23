"""
Default values for template placeholders when features are missing.

These defaults are based on:
1. Common high-ROAS patterns from data analysis
2. Neutral values that work across product categories
3. Product photography best practices
"""

# Default values for P0 Master Mask template placeholders
DEFAULT_VALUES = {
    # Product information
    "subject_name": "product",  # Rarely used (product_name usually available)
    "subject_description": (
        "product, complete assembled unit"
    ),  # Generic fallback
    "material_finish": (
        "premium finish"
    ),  # Generic default - should be extracted from product_context
    "completeness_instruction": (
        "Show the complete product as an assembled unit with all components "
        "connected. DO NOT show only individual parts separately"
    ),
    # Layout and composition
    "product_position": "center",  # Neutral, safe default
    "product_visibility": "full",  # Show product clearly
    # Interaction elements
    "interaction_context": "",  # No human elements by default (product-focused)
    # Color constraints
    "color_constraint": (
        ""
    ),  # Generic default - should be extracted from product_context if needed
    # Background
    "static_context": (
        "sunlit, modern minimalist home environment"
    ),  # Audience-driven
    # Lighting
    "color_balance": (
        "luminous tones, ultra-bright high-key studio lighting with natural "
        "sunlight-filled atmosphere, vibrant tones, and clean, vibrant highlights"
    ),  # Meta Ad optimization with high-luminance priority
    "brightness_distribution": "gradient",  # Common high-ROAS pattern
    # Focal point
    "visual_prominence": "dominant",  # Product as hero
    # Physical grounding
    "grounding_instruction": (
        "Product must be firmly grounded on the floor with natural contact shadows, "
        "preventing floating artifacts"
    ),
    # V2 Enhanced ROAS features
    "placement_target": "modern sofa or wall",  # Default placement context
    "composition_style": "",  # Empty default (will be filled from negative_space_usage)
    "lighting_detail": "",  # Empty default (will be filled from brightness_distribution)
    "environment_objects": "",  # Empty default (will be filled from relationship_depiction)
}

# Rationale documentation
DEFAULT_RATIONALE = {
    "product_position": "Center is neutral and works for most products. "
    "High-ROAS alternatives: bottom-right (4.51 ROAS), left (4.05 ROAS)",
    "product_visibility": "Full visibility ensures product clarity. "
    "High-ROAS alternative: partial (5.15 ROAS)",
    "color_balance": (
        "Balanced is neutral. High-ROAS alternative: warm-dominant (3.87 ROAS)"
    ),
    "brightness_distribution": (
        "Gradient is common in high performers (3.43 ROAS)"
    ),
    "visual_prominence": ("Dominant ensures product is hero (3.32 ROAS)"),
    "static_context": (
        "Sunlit, modern minimalist home environment for Meta Ad audience"
    ),
    "material_finish": (
        "Generic fallback - should be extracted from product_context"
    ),
}


def get_default(placeholder_name: str) -> str:
    """
    Get default value for a template placeholder.

    Args:
        placeholder_name: Name of the placeholder (e.g., "product_position")

    Returns:
        Default value string

    Raises:
        KeyError: If placeholder has no default defined
    """
    if placeholder_name not in DEFAULT_VALUES:
        raise KeyError(
            f"No default defined for placeholder '{placeholder_name}'. "
            f"Available defaults: {list(DEFAULT_VALUES.keys())}"
        )
    return DEFAULT_VALUES[placeholder_name]
