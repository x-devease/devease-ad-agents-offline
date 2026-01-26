"""
Multi-Perspective Generation Module.

Generates prompts for multiple product angles/views from a single recommendation.
Each perspective emphasizes different product aspects and uses appropriate templates.

Perspectives:
- 侧俯 (Side-oblique/45°): Shows depth and form
- 右侧45 (Right 45°): Classic product angle
- 180躺平 (Flat-lay 180°): Shows full top surface
- 正面 (Front view): Shows primary interface/face
- 背面 (Back view): Shows ports/connections
"""

import logging
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


# Perspective definitions
PERSPECTIVES = {
    "侧俯": {
        "name_en": "Side-oblique / 45° angle",
        "description": "Shows depth, form, and dimensionality",
        "best_for": ["Demonstrating product depth", "Showing side features", "Technical detail"],
        "template": "WIDE_SCENE",  # Hero lifestyle
        "camera_angle": "45° from horizontal",
        "emphasis": "Three-dimensional form and spatial relationships",
        "composition": "Product at slight angle showing depth",
        "lighting": "Three-point lighting to define form",
    },
    "右侧45": {
        "name_en": "Right 45° angle",
        "description": "Classic product photography angle",
        "best_for": ["Standard product shots", "E-commerce", "Catalog images"],
        "template": "MACRO_DETAIL",  # Close-up technical
        "camera_angle": "45° from horizontal, right side",
        "emphasis": "Product features and technical details",
        "composition": "Profile view showing design elements",
        "lighting": "Even illumination for detail visibility",
    },
    "180躺平": {
        "name_en": "Flat-lay 180° top-down",
        "description": "Shows full top surface area",
        "best_for": ["Demonstrating 180° capability", "Flat-lay arrangements", "Top features"],
        "template": "FLAT_TECH",  # Flat-lay demonstration
        "camera_angle": "90° from horizontal (top-down)",
        "emphasis": "Surface area, top features, layout",
        "composition": "Product lying flat, camera above",
        "lighting": "Even overhead lighting",
    },
    "正面": {
        "name_en": "Front view",
        "description": "Shows primary interface/face",
        "best_for": ["Controls and interfaces", "Primary features", "User-facing elements"],
        "template": "MACRO_DETAIL",  # Close-up
        "camera_angle": "0° from horizontal (straight on)",
        "emphasis": "Primary interface, controls, displays",
        "composition": "Direct front view",
        "lighting": "Even front lighting",
    },
    "背面": {
        "name_en": "Back view",
        "description": "Shows ports, connections, rear features",
        "best_for": ["Connectivity", "Ports and interfaces", "Technical documentation"],
        "template": "MACRO_DETAIL",  # Close-up
        "camera_angle": "180° from horizontal (from behind)",
        "emphasis": "Ports, connections, rear panels",
        "composition": "Rear view",
        "lighting": "Even illumination of rear features",
    },
}


def get_perspective_info(perspective: str) -> Dict:
    """
    Get perspective information.

    Args:
        perspective: Perspective name (Chinese or English)

    Returns:
        Perspective information dictionary
    """
    # Check both Chinese and English
    if perspective in PERSPECTIVES:
        return PERSPECTIVES[perspective]

    # Try to find by English name
    for key, info in PERSPECTIVES.items():
        if perspective.lower() in info["name_en"].lower():
            return info

    # Default to side-oblique
    return PERSPECTIVES["侧俯"]


def generate_multi_perspective_prompts(
    visual_formula: Dict,
    product_context: Dict,
    perspectives: List[str] = None,
) -> Dict[str, Dict]:
    """
    Generate prompts for multiple product perspectives.

    Args:
        visual_formula: Visual formula with features
        product_context: Product context information
        perspectives: List of perspectives to generate (default: common ones)

    Returns:
        Dict mapping perspective to prompt info:
        {
            "侧俯": {
                "template": "WIDE_SCENE",
                "emphasis": "Three-dimensional form",
                "camera_angle": "45°",
                "feature_priorities": ["form", "depth", "spatial"]
            },
            ...
        }

    Example:
        >>> prompts = generate_multi_perspective_prompts(formula, context, ["侧俯", "180躺平"])
    """
    if perspectives is None:
        perspectives = ["侧俯", "右侧45", "180躺平"]  # Default set

    result = {}

    for perspective in perspectives:
        info = get_perspective_info(perspective)

        # Determine feature priorities based on perspective
        feature_priorities = _get_feature_priorities(perspective)

        result[perspective] = {
            "template": info["template"],
            "perspective_en": info["name_en"],
            "description": info["description"],
            "best_for": info["best_for"],
            "camera_angle": info["camera_angle"],
            "emphasis": info["emphasis"],
            "composition": info["composition"],
            "lighting": info["lighting"],
            "feature_priorities": feature_priorities,
        }

        logger.info(
            "Perspective %s (%s): Template=%s, Emphasis=%s",
            perspective,
            info["name_en"],
            info["template"],
            info["emphasis"],
        )

    return result


def _get_feature_priorities(perspective: str) -> List[str]:
    """
    Get feature priorities for a given perspective.

    Args:
        perspective: Perspective name

    Returns:
        List of feature priority keywords
    """
    priority_map = {
        "侧俯": ["form", "depth", "spatial", "dimensionality", "3d_structure"],
        "右侧45": ["technical_detail", "features", "design", "profile"],
        "180躺平": ["surface_area", "top_features", "layout", "arrangement"],
        "正面": ["interface", "controls", "display", "primary_features"],
        "背面": ["ports", "connections", "rear_features", "connectivity"],
    }

    return priority_map.get(perspective, ["detail", "quality"])


def get_perspective_specific_instructions(
    perspective: str,
) -> str:
    """
    Get perspective-specific instructions for prompt.

    Args:
        perspective: Perspective name

    Returns:
        Formatted perspective-specific instructions
    """
    info = get_perspective_info(perspective)

    instructions = []

    instructions.append(f"[Perspective: {info['name_en']}]")
    instructions.append(f"[Description: {info['description']}]")
    instructions.append(f"[Camera Angle: {info['camera_angle']}]")
    instructions.append(f"[Emphasis: {info['emphasis']}]")
    instructions.append(f"[Composition: {info['composition']}]")
    instructions.append(f"[Lighting: {info['lighting']}]")
    instructions.append(f"[Best For: {', '.join(info['best_for'])}]")

    return " ".join(instructions)


# Perspective angle recommendations
ANGLE_RECOMMENDATIONS = {
    "hero_shot": {
        "perspectives": ["侧俯", "右侧45"],
        "template": "WIDE_SCENE",
        "purpose": "Primary marketing image",
    },
    "technical_detail": {
        "perspectives": ["右侧45", "正面", "背面"],
        "template": "MACRO_DETAIL",
        "purpose": "Show features and specifications",
    },
    "demonstration": {
        "perspectives": ["180躺平", "正面"],
        "template": "FLAT_TECH",
        "purpose": "Demonstrate capabilities",
    },
    "360_view": {
        "perspectives": ["正面", "右侧45", "背面", "左侧45"],
        "template": "MACRO_DETAIL",
        "purpose": "Complete product view",
    },
}


def get_perspective_set(use_case: str) -> List[str]:
    """
    Get recommended perspectives for a use case.

    Args:
        use_case: Type of shot (hero_shot, technical_detail, demonstration, 360_view)

    Returns:
        List of recommended perspectives
    """
    if use_case not in ANGLE_RECOMMENDATIONS:
        logger.warning(f"Unknown use case '{use_case}', returning default")
        return ["侧俯", "右侧45", "180躺平"]

    return ANGLE_RECOMMENDATIONS[use_case]["perspectives"]


def create_perspective_variation(
    base_prompt: str,
    perspective: str,
    visual_formula: Dict,
) -> str:
    """
    Create a prompt variation for a specific perspective.

    Args:
        base_prompt: Original prompt
        perspective: Target perspective
        visual_formula: Visual formula for features

    Returns:
        Modified prompt for perspective (placeholder for now)

    Note: This is a simplified version. Full implementation would
    re-render the template with perspective-specific adjustments.
    """
    info = get_perspective_info(perspective)

    # For now, append perspective instructions
    # In full implementation, would re-render with adjusted parameters
    perspective_instructions = get_perspective_specific_instructions(perspective)

    return f"{base_prompt} {perspective_instructions}"


# Perspective templates mapping
def get_template_for_perspective(perspective: str) -> str:
    """
    Get optimal template for a given perspective.

    Args:
        perspective: Perspective name

    Returns:
        Template name (WIDE_SCENE, MACRO_DETAIL, or FLAT_TECH)
    """
    info = get_perspective_info(perspective)
    return info["template"]


def generate_perspective_summary(
    perspectives: List[str],
) -> str:
    """
    Generate a summary of perspectives for logging/reporting.

    Args:
        perspectives: List of perspective names

    Returns:
        Formatted summary string
    """
    summary_parts = []

    summary_parts.append(f"Multi-Perspective Generation: {len(perspectives)} perspectives")

    for i, perspective in enumerate(perspectives, 1):
        info = get_perspective_info(perspective)
        summary_parts.append(
            f"\n{i}. {perspective} ({info['name_en']}): {info['description']}"
        )
        summary_parts.append(f"   Template: {info['template']}")
        summary_parts.append(f"   Emphasis: {info['emphasis']}")

    return "\n".join(summary_parts)


# Perspective combinations for different products
PRODUCT_PERSPECTIVE_RECOMMENDATIONS = {
    "electronics": {
        "primary": ["正面", "右侧45"],
        "secondary": ["180躺平", "背面"],
        "all": ["正面", "右侧45", "背面", "180躺平", "侧俯"],
        "reasoning": "Show controls, ports, and top surface",
    },
    "furniture": {
        "primary": ["侧俯", "右侧45"],
        "secondary": ["180躺平"],
        "all": ["侧俯", "右侧45", "正面"],
        "reasoning": "Show form, scale, and context in room",
    },
    "appliances": {
        "primary": ["正面", "右侧45"],
        "secondary": ["180躺平"],
        "all": ["正面", "右侧45", "背面"],
        "reasoning": "Show interface, controls, and features",
    },
    "flat_products": {
        "primary": ["180躺平", "正面"],
        "secondary": ["侧俯"],
        "all": ["180躺平", "正面", "侧俯", "右侧45"],
        "reasoning": "Emphasize flat-lay capability",
    },
}


def get_recommended_perspectives(
    product_type: str,
    depth: str = "primary",
) -> List[str]:
    """
    Get recommended perspectives for a product type.

    Args:
        product_type: Type of product (electronics, furniture, etc.)
        depth: Depth of recommendations (primary, secondary, all)

    Returns:
        List of recommended perspective names

    Example:
        >>> perspectives = get_recommended_perspectives("electronics", "all")
        >>> # Returns ["正面", "右侧45", "背面", "180躺平", "侧俯"]
    """
    product_type_lower = product_type.lower()

    # Match product type
    for key, recommendations in PRODUCT_PERSPECTIVE_RECOMMENDATIONS.items():
        if key in product_type_lower:
            return recommendations[depth]

    # Default fallback
    return PRODUCT_PERSPECTIVE_RECOMMENDATIONS["electronics"][depth]
