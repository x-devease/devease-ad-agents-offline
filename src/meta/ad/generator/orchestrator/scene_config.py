"""
Scene Overview Configuration System.

Allows different scenarios to be defined through configuration files or product context,
making the system universal for different products and use cases.
"""

import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)
# Default scene overview templates (can be overridden by product context)
# CRITICAL: All templates must include grounding information to prevent floating products
DEFAULT_SCENE_OVERVIEWS = {
    "golden_ratio": {
        "template": (
            "A wide-angle lifestyle photography scene showing an authentic "
            "high-end home environment ({static_context}), "
            "with the product naturally integrated into the environment, "
            "creating a harmonious balance between the product and its "
            "surroundings, "
            "capturing the entire unit silhouette from handle to base in a "
            "wide-angle view. "
            "The scene must feel like a real home, not a photography studio."
        ),
        "description": (
            "Wide-angle lifestyle scene with product integration in "
            "authentic home environment"
        ),
    },
    "high_efficiency": {
        "template": (
            "A professional close-up photography scene focusing exclusively on the product's "
            "mechanical details and CMF textures, with a cinematic high-angle view creating "
            "a shallow depth of field that isolates the product base and cleaning mechanism, "
            "emphasizing precision and craftsmanship through extreme detail. "
            "The product base must be firmly resting on the floor or surface "
            "with visible contact shadows, "
            "preventing any floating appearance."
        ),
        "description": (
            "Professional close-up focusing on mechanical details with "
            "grounding"
        ),
    },
    "cool_peak": {
        "template": (
            "A low-profile flat-lay product photography scene showing the product lying "
            "180-degree flat against the floor, demonstrating its slim profile and ability "
            "to slide under low-clearance furniture, creating a clean, minimalist composition "
            "that highlights the product's innovative design and space-saving capabilities. "
            "The product must be lying flat on the floor surface with visible contact shadows, "
            "preventing any floating or suspended appearance."
        ),
        "description": (
            "Low-profile flat-lay demonstrating space-saving capabilities "
            "with grounding"
        ),
    },
}


def get_scene_overview(
    branch_name: str,
    product_context: Dict[str, Any],
    placeholder_values: Dict[str, str],
) -> str:
    """
    Get scene overview for a specific branch, with support for custom scenarios.

    Priority:
    1. Custom scene overview from product_context.scene_overviews[branch_name]
    2. Custom scene overview from product_context.scene_overview (generic)
    3. Default template for branch

    Args:
        branch_name: Branch identifier (golden_ratio, high_efficiency, cool_peak)
        product_context: Product context dict (may contain custom scene_overviews)
        placeholder_values: Current placeholder values for template rendering

    Returns:
        Scene overview string
    """
    # Check for custom scene overviews in product context
    scene_overviews = product_context.get("scene_overviews", {})
    # Priority 1: Branch-specific custom scene overview
    if branch_name in scene_overviews:
        custom_template = scene_overviews[branch_name]
        logger.info(
            "Using custom scene overview for branch '%s' from product context",
            branch_name,
        )
        return _render_scene_template(custom_template, placeholder_values)
    # Priority 2: Generic custom scene overview
    if "scene_overview" in product_context:
        custom_template = product_context["scene_overview"]
        logger.info(
            "Using generic custom scene overview from product context for branch '%s'",
            branch_name,
        )
        return _render_scene_template(custom_template, placeholder_values)
    # Priority 3: Default template for branch
    if branch_name in DEFAULT_SCENE_OVERVIEWS:
        default_template = DEFAULT_SCENE_OVERVIEWS[branch_name]["template"]
        logger.debug(
            "Using default scene overview template for branch '%s'", branch_name
        )
        return _render_scene_template(default_template, placeholder_values)
    # Fallback: Generic description
    logger.warning(
        "No scene overview found for branch '%s', using fallback", branch_name
    )
    return (
        "A professional product photography scene showing the product "
        "in a well-composed setting with appropriate lighting and context."
    )


def _render_scene_template(
    template: str, placeholder_values: Dict[str, str]
) -> str:
    """
    Render scene template with placeholder values.

    Supports placeholders like:
    - {static_context}
    - {product_position}
    - {position_desc} (auto-generated from product_position)

    Args:
        template: Template string with placeholders
        placeholder_values: Values to substitute

    Returns:
        Rendered scene overview string
    """
    # Position is now only defined in [Layout] block, not in scene overview
    # No need to generate position_desc here to avoid duplication
    render_values = placeholder_values.copy()
    # Render template
    try:
        return template.format(**render_values)
    except KeyError as e:
        logger.warning(
            "Missing placeholder '%s' in scene template, using raw template",
            e.args[0],
        )
        # Fallback: Replace known placeholders manually
        result = template
        result = result.replace(
            "{static_context}",
            render_values.get("static_context", "modern home"),
        )
        return result
