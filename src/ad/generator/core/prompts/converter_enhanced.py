"""
Enhanced Feature-to-Prompt Converter

Optimized prompt generation based on professional prompt engineering insights
from devease-image-gen-offline reference implementations.

Key improvements:
- Structured sections with clear organization
- Rich natural language feature value descriptions
- Critical constraints prioritized
- Variable-based template system
- Professional photography terminology
"""

import logging
from typing import Any, Dict, List, Optional

from .constants import CATEGORIES
from .constants import get_feature_category as _get_feature_category
from .feature_descriptions import get_feature_value_description


logger = logging.getLogger(__name__)


def _get_priority_level(
    feature_name: str,
    feature_importance: Optional[Dict[str, float]] = None,
    default_importance: float = 0.0,
) -> str:
    """Get priority level for a feature (critical, high, medium, low)."""
    if not feature_importance:
        return "medium"

    importance = feature_importance.get(feature_name, default_importance)

    if importance >= 20.0:
        return "critical"
    if importance >= 10.0:
        return "high"
    if importance >= 5.0:
        return "medium"
    return "low"


def _format_feature_description(
    feature_name: str,
    value: str,
    is_negative: bool = False,
    feature_importance: Optional[
        Dict[str, float]
    ] = None,  # noqa: ARG001 - Unused but kept for API compatibility
) -> str:
    """
    Format a feature as rich natural language description.

    Args:
        feature_name: Name of the feature
        value: Feature value
        is_negative: Whether this is a negative instruction (avoid)
        feature_importance: Optional importance scores (unused, kept for API compatibility)

    Returns:
        Formatted natural language description
    """
    # Get rich description of the value
    value_desc = get_feature_value_description(feature_name, value)
    # Build description
    if is_negative:
        if value_desc:
            return f"AVOID: {value_desc}"
        value_display = value.replace("_", " ").title()
        return f"AVOID: {value_display}"
    if value_desc:
        return value_desc
    value_display = value.replace("_", " ").title()
    return value_display


def _build_critical_constraints_section(
    features: List[Dict[str, Any]],
    feature_importance: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Build critical constraints section (high-priority features).

    Args:
        features: List of feature dicts with 'feature', 'value', 'instruction'
        feature_importance: Optional importance scores

    Returns:
        List of constraint lines
    """
    critical_features = []
    for feat in features:
        priority = _get_priority_level(
            feat["feature"], feature_importance, default_importance=0.0
        )
        if priority in ["critical", "high"]:
            critical_features.append(feat)

    if not critical_features:
        return []

    lines = ["CRITICAL REQUIREMENTS:"]
    for feat in critical_features:
        desc = _format_feature_description(
            feat["feature"],
            feat["value"],
            is_negative=False,
            feature_importance=feature_importance,
        )
        lines.append(f"  - {desc}")

    return lines


def _build_category_section(
    category: str,
    features: List[Dict[str, Any]],
    feature_importance: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Build a category section with natural language descriptions.

    Args:
        category: Category name (lighting, composition, etc.)
        features: List of feature dicts
        feature_importance: Optional importance scores

    Returns:
        List of section lines
    """
    if not features:
        return []
    category_title = category.replace("_", " ").title()
    title = f"{category_title.upper()} REQUIREMENTS:"

    lines = [title]

    for feat in features:
        desc = _format_feature_description(
            feat["feature"],
            feat["value"],
            is_negative=False,
            feature_importance=feature_importance,
        )
        lines.append(f"  - {desc}")

    return lines


def _build_negative_constraints_section(
    negative_features: List[Dict[str, Any]],
    negative_feature_importance: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Build negative constraints section (what to avoid).

    Args:
        negative_features: List of negative feature dicts
        negative_feature_importance: Optional importance scores

    Returns:
        List of constraint lines
    """
    if not negative_features:
        return []

    lines = ["AVOID THESE ELEMENTS:"]

    for feat in negative_features:
        desc = _format_feature_description(
            feat["feature"],
            feat["value"],
            is_negative=True,
            feature_importance=negative_feature_importance,
        )
        lines.append(f"  - {desc}")

    return lines


def _build_technical_specs_section() -> List[str]:
    """
    Build technical photography specifications section.

    Returns:
        List of technical spec lines
    """
    return [
        "TECHNICAL PHOTOGRAPHY SPECIFICATIONS:",
        "  - Shot with a full-frame camera, 50mm or 85mm lens",
        "  - Aperture: f/2.8–f/4 with natural depth of field",
        "  - Realistic lighting with believable direction and falloff",
        "  - Consistent shadows matching light position",
        "  - Natural material imperfections and surface textures",
        "  - Professional product photography quality",
    ]


def convert_features_to_enhanced_prompt(
    recommended_features: List[str],
    feature_importance: Optional[Dict[str, float]] = None,
    negative_features: Optional[List[str]] = None,
    negative_feature_importance: Optional[Dict[str, float]] = None,
    feature_values: Optional[Dict[str, str]] = None,
    negative_feature_values: Optional[Dict[str, str]] = None,
    min_importance: float = 0.0,
    include_technical_specs: bool = True,
) -> Dict[str, Any]:
    """
    Convert recommended features to enhanced structured prompt.

    Based on professional prompt engineering patterns from
    devease-image-gen-offline reference implementations.

    Args:
        recommended_features: List of feature names recommended
        feature_importance: Optional dict mapping feature names to
            importance scores
        negative_features: Optional list of feature names to avoid
        negative_feature_importance: Optional dict mapping negative feature
            names to importance scores
        feature_values: Dict mapping feature names to explicit values
        negative_feature_values: Dict mapping negative feature names to
            explicit values
        min_importance: Minimum importance score to include
        include_technical_specs: Whether to include technical photography
            specs

    Returns:
        Dict with enhanced prompt sections
    """
    # Filter and sort features by importance
    if feature_importance:
        filtered_features = [
            feat
            for feat in recommended_features
            if feature_importance.get(feat, 0.0) >= min_importance
        ]
        filtered_features = sorted(
            filtered_features,
            key=lambda f: feature_importance.get(f, 0.0),
            reverse=True,
        )
    else:
        filtered_features = recommended_features
    # Organize features by category
    category_features = {cat: [] for cat in CATEGORIES}
    category_features["other"] = []
    all_feature_dicts = []
    # Process positive features
    for feature_name in filtered_features:
        if not feature_values or feature_name not in feature_values:
            logger.warning(
                "No value provided for feature %s, skipping", feature_name
            )
            continue

        value = feature_values[feature_name]
        category = _get_feature_category(feature_name)

        feat_dict = {
            "feature": feature_name,
            "value": value,
            "instruction": _format_feature_description(
                feature_name,
                value,
                is_negative=False,
                feature_importance=feature_importance,
            ),
        }

        category_features[category].append(feat_dict)
        all_feature_dicts.append(feat_dict)
    # Process negative features
    negative_feature_dicts = []
    if negative_features:
        if negative_feature_importance:
            filtered_negative = [
                feat
                for feat in negative_features
                if negative_feature_importance.get(feat, 0.0) >= min_importance
            ]
            filtered_negative = sorted(
                filtered_negative,
                key=lambda f: negative_feature_importance.get(f, 0.0),
                reverse=True,
            )
        else:
            filtered_negative = negative_features

        for feature_name in filtered_negative:
            if (
                not negative_feature_values
                or feature_name not in negative_feature_values
            ):
                continue

            value = negative_feature_values[feature_name]
            feat_dict = {
                "feature": feature_name,
                "value": value,
                "instruction": _format_feature_description(
                    feature_name,
                    value,
                    is_negative=True,
                    feature_importance=negative_feature_importance,
                ),
            }
            negative_feature_dicts.append(feat_dict)
    # Build structured prompt sections
    prompt_sections = []
    # 1. Critical constraints (high-priority features)
    critical_section = _build_critical_constraints_section(
        all_feature_dicts, feature_importance
    )
    if critical_section:
        prompt_sections.extend(critical_section)
        prompt_sections.append("")  # Blank line separator
    # 2. Category-based sections
    category_order = [
        "lighting",
        "composition",
        "content",
        "visual_style",
        "product",
        "background",
        "other",
    ]
    for category in category_order:
        if category_features[category]:
            section = _build_category_section(
                category, category_features[category], feature_importance
            )
            if section:
                prompt_sections.extend(section)
                prompt_sections.append("")  # Blank line separator
    # 3. Negative constraints
    if negative_feature_dicts:
        negative_section = _build_negative_constraints_section(
            negative_feature_dicts, negative_feature_importance
        )
        if negative_section:
            prompt_sections.extend(negative_section)
            prompt_sections.append("")  # Blank line separator
    # 4. Technical specifications
    if include_technical_specs:
        tech_section = _build_technical_specs_section()
        prompt_sections.extend(tech_section)
    # Combine into final prompt
    enhanced_prompt = "\n".join(prompt_sections)

    return {
        "enhanced_prompt": enhanced_prompt,
        "sections": {
            "critical": critical_section if critical_section else [],
            "categories": {
                cat: category_features[cat]
                for cat in category_order
                if category_features.get(cat)
            },
            "negative": negative_feature_dicts,
            "technical": (
                _build_technical_specs_section()
                if include_technical_specs
                else []
            ),
        },
        "all_features": all_feature_dicts,
        "negative_features": negative_feature_dicts,
    }
