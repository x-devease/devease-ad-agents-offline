"""
Dynamic Template Selection Module.

Intelligently selects the appropriate template based on product type,
category, and characteristics using rule-based logic.

Template types:
- WIDE_SCENE: Hero lifestyle ads with environment context
- MACRO_DETAIL: Close-up technical detail
- FLAT_TECH: Flat-lay demonstration (180-degree)
"""

import logging
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


# Template to ROAS mapping (based on historical performance)
TEMPLATE_ROAS_MAPPING = {
    "WIDE_SCENE": {
        "roas": 10.7,
        "branch": "golden_ratio",
        "best_for": ["furniture", "home_goods", "appliances", "lifestyle_context"],
        "strength": "Maximum environment integration",
        "use_case": "Hero ads requiring lifestyle context",
    },
    "MACRO_DETAIL": {
        "roas": 5.15,
        "branch": "high_efficiency",
        "best_for": ["electronics", "tech_accessories", "mechanical", "small_products"],
        "strength": "Extreme focus on mechanics and CMF",
        "use_case": "Technical detail and close-up shots",
    },
    "FLAT_TECH": {
        "roas": 8.34,
        "branch": "cool_peak",
        "best_for": ["apparel", "flat_lay_products", "demonstration", "180_capability"],
        "strength": "Shows 180-degree flat-lay capability",
        "use_case": "Demonstrating product from above",
    },
}


# Product type to template mapping rules
PRODUCT_TEMPLATE_RULES = {
    # Electronics → MACRO_DETAIL (show technical details)
    "electronics": "MACRO_DETAIL",
    "phone": "MACRO_DETAIL",
    "laptop": "MACRO_DETAIL",
    "tablet": "MACRO_DETAIL",
    "camera": "MACRO_DETAIL",
    "headphones": "MACRO_DETAIL",
    "speaker": "MACRO_DETAIL",
    "watch": "MACRO_DETAIL",
    "tech_accessory": "MACRO_DETAIL",

    # Furniture → WIDE_SCENE (lifestyle context)
    "furniture": "WIDE_SCENE",
    "sofa": "WIDE_SCENE",
    "chair": "WIDE_SCENE",
    "table": "WIDE_SCENE",
    "desk": "WIDE_SCENE",
    "bed": "WIDE_SCENE",
    "shelf": "WIDE_SCENE",
    "cabinet": "WIDE_SCENE",

    # Home goods → WIDE_SCENE (room context)
    "home_goods": "WIDE_SCENE",
    "lamp": "WIDE_SCENE",
    "appliance": "WIDE_SCENE",
    "rug": "WIDE_SCENE",
    "decor": "WIDE_SCENE",

    # Apparel → FLAT_TECH (flat-lay demonstration)
    "apparel": "FLAT_TECH",
    "clothing": "FLAT_TECH",
    "shirt": "FLAT_TECH",
    "pants": "FLAT_TECH",
    "dress": "FLAT_TECH",
    "shoes": "FLAT_TECH",
    "accessories": "FLAT_TECH",

    # Power tools → MACRO_DETAIL (technical)
    "power_station": "MACRO_DETAIL",
    "tool": "MACRO_DETAIL",
    "drill": "MACRO_DETAIL",
    "saw": "MACRO_DETAIL",
    "equipment": "MACRO_DETAIL",
}


# Feature-based template selection
FEATURE_TEMPLATE_RULES = {
    "lifestyle_context": "WIDE_SCENE",
    "technical_detail": "MACRO_DETAIL",
    "flat_lay": "FLAT_TECH",
    "hero_shot": "WIDE_SCENE",
    "close_up": "MACRO_DETAIL",
    "demonstration": "FLAT_TECH",
}


def select_template_by_product(
    product_name: str,
    product_category: Optional[str] = None,
    fallback: str = "WIDE_SCENE",
) -> str:
    """
    Select template based on product name and category.

    Args:
        product_name: Name of the product
        product_category: Optional category hint
        fallback: Fallback template if no match found

    Returns:
        Selected template name (WIDE_SCENE, MACRO_DETAIL, or FLAT_TECH)

    Example:
        >>> template = select_template_by_product("Power Station", "electronics")
        >>> # Returns "MACRO_DETAIL"
    """
    name_lower = product_name.lower()

    # Check product name keywords first
    for keyword, template in PRODUCT_TEMPLATE_RULES.items():
        if keyword in name_lower:
            logger.info(
                "Template selected by keyword '%s': %s for product '%s'",
                keyword,
                template,
                product_name,
            )
            return template

    # Check category if provided
    if product_category and product_category.lower() in PRODUCT_TEMPLATE_RULES:
        template = PRODUCT_TEMPLATE_RULES[product_category.lower()]
        logger.info(
            "Template selected by category '%s': %s for product '%s'",
            product_category,
            template,
            product_name,
        )
        return template

    # Use fallback
    logger.info(
        "No template match found for '%s', using fallback: %s",
        product_name,
        fallback,
    )
    return fallback


def select_template_by_features(
    visual_formula: Dict,
    fallback: str = "WIDE_SCENE",
) -> str:
    """
    Select template based on visual formula features.

    Args:
        visual_formula: Visual formula with entrance_features and headroom_features
        fallback: Fallback template if no match found

    Returns:
        Selected template name
    """
    # Check features for template indicators
    entrance_features = visual_formula.get("entrance_features", [])
    headroom_features = visual_formula.get("headroom_features", [])

    all_features = entrance_features + headroom_features

    # Look for feature names that suggest template
    for feature in all_features:
        feature_name = feature.get("feature_name", "").lower()
        feature_value = feature.get("feature_value", "").lower()

        # Check feature name and value
        for rule_key, template in FEATURE_TEMPLATE_RULES.items():
            if rule_key in feature_name or rule_key in feature_value:
                logger.info(
                    "Template selected by feature '%s': %s",
                    feature_name,
                    template,
                )
                return template

    logger.info("No feature-based template match, using fallback: %s", fallback)
    return fallback


def select_template_by_branch(
    branch_name: str,
) -> str:
    """
    Select template based on branch name.

    Args:
        branch_name: Branch identifier (golden_ratio, high_efficiency, cool_peak)

    Returns:
        Selected template name

    Example:
        >>> template = select_template_by_branch("high_efficiency")
        >>> # Returns "MACRO_DETAIL"
    """
    branch_to_template = {
        "golden_ratio": "WIDE_SCENE",
        "high_efficiency": "MACRO_DETAIL",
        "cool_peak": "FLAT_TECH",
    }

    template = branch_to_template.get(branch_name, "WIDE_SCENE")
    logger.info("Template selected by branch '%s': %s", branch_name, template)

    return template


def get_template_recommendation(
    product_name: str,
    product_category: Optional[str] = None,
    visual_formula: Optional[Dict] = None,
    branch_name: Optional[str] = None,
    confidence_threshold: float = 0.7,
) -> Dict[str, any]:
    """
    Get complete template recommendation with reasoning.

    Args:
        product_name: Name of the product
        product_category: Optional category hint
        visual_formula: Optional visual formula for feature-based selection
        branch_name: Optional branch name override
        confidence_threshold: Minimum confidence for auto-selection

    Returns:
        Dict with template recommendation and reasoning:
        {
            "template": "WIDE_SCENE",
            "branch": "golden_ratio",
            "roas": 10.7,
            "confidence": 0.85,
            "reasoning": "List of reasons for selection",
            "alternatives": ["MACRO_DETAIL", "FLAT_TECH"]
        }
    """
    recommendations = []
    reasoning = []

    # Product-based recommendation
    product_template = select_template_by_product(product_name, product_category)
    product_confidence = 0.8 if product_template != "WIDE_SCENE" else 0.5
    recommendations.append((product_template, product_confidence))
    reasoning.append(
        f"Product '{product_name}' suggests {product_template} (confidence: {product_confidence})"
    )

    # Feature-based recommendation (if formula provided)
    if visual_formula:
        feature_template = select_template_by_features(visual_formula)
        feature_confidence = 0.9 if feature_template != "WIDE_SCENE" else 0.5
        recommendations.append((feature_template, feature_confidence))
        reasoning.append(
            f"Features suggest {feature_template} (confidence: {feature_confidence})"
        )

    # Branch-based recommendation (if provided)
    if branch_name:
        branch_template = select_template_by_branch(branch_name)
        branch_confidence = 1.0  # Explicit branch selection = high confidence
        recommendations.append((branch_template, branch_confidence))
        reasoning.append(f"Branch '{branch_name}' explicitly selects {branch_template}")

    # Select highest confidence recommendation
    if not recommendations:
        selected_template = "WIDE_SCENE"
        confidence = 0.3
    else:
        selected_template, confidence = max(recommendations, key=lambda x: x[1])

    # Get template info
    template_info = TEMPLATE_ROAS_MAPPING.get(selected_template, {})
    roas = template_info.get("roas", 0)
    branch = template_info.get("branch", "golden_ratio")

    # Get alternatives (other templates)
    all_templates = ["WIDE_SCENE", "MACRO_DETAIL", "FLAT_TECH"]
    alternatives = [t for t in all_templates if t != selected_template]

    return {
        "template": selected_template,
        "branch": branch,
        "roas": roas,
        "confidence": confidence,
        "reasoning": reasoning,
        "alternatives": alternatives,
    }


def should_use_template(
    product_name: str,
    product_category: Optional[str] = None,
    visual_formula: Optional[Dict] = None,
    min_confidence: float = 0.7,
) -> tuple[bool, str, Dict]:
    """
    Determine if we should use auto-selected template.

    Args:
        product_name: Name of the product
        product_category: Optional category hint
        visual_formula: Optional visual formula
        min_confidence: Minimum confidence threshold

    Returns:
        Tuple of (should_use, template_name, recommendation_dict)

    Example:
        >>> should_use, template, rec = should_use_template("Power Station")
        >>> if should_use:
        ...     print(f"Using {template} with {rec['confidence']:.0%} confidence")
    """
    recommendation = get_template_recommendation(
        product_name=product_name,
        product_category=product_category,
        visual_formula=visual_formula,
    )

    should_use = recommendation["confidence"] >= min_confidence

    if should_use:
        logger.info(
            "Auto-selecting %s for '%s' (confidence: %.0f%%)",
            recommendation["template"],
            product_name,
            recommendation["confidence"] * 100,
        )
    else:
        logger.warning(
            "Low confidence (%.0f%%) for template selection of '%s', manual selection recommended",
            recommendation["confidence"] * 100,
            product_name,
        )

    return should_use, recommendation["template"], recommendation
