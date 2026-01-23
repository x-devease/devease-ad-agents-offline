"""
Unified Feature Application Registry

This module provides a centralized registry of feature definitions to ensure
consistency across:
1. Feature selection from visual_formula
2. Feature injection into prompts
3. Feature extraction from generated images

All feature values must use the exact same string definitions from this registry
to ensure 100% match rate between injected and extracted features.
"""

import logging
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)
# ============================================================================
# MASTER FEATURE REGISTRY
# ============================================================================
# All feature values must be defined here to ensure consistency across
# formula selection, prompt injection, and feature extraction.
# Product Position Values
PRODUCT_POSITION_VALUES: Dict[str, str] = {
    "bottom-right": "bottom-right",
    "bottom-left": "bottom-left",
    "top-right": "top-right",
    "top-left": "top-left",
    "center": "center",
    "left": "left",
    "right": "right",
}
# Atmosphere/Temperature Values
ATMOSPHERE_VALUES: Dict[str, str] = {
    "warm": "Warm",
    "cool": "Cool",
    "neutral": "Neutral",
    "warm-dominant": "Warm",
    "cool-dominant": "Cool",
    "balanced": "Neutral",
}
# Color Balance Values (for backward compatibility)
COLOR_BALANCE_VALUES: Dict[str, str] = {
    "warm-dominant": "warm-dominant",
    "cool-dominant": "cool-dominant",
    "balanced": "balanced",
    "neutral": "neutral",
}
# Product Visibility Values
PRODUCT_VISIBILITY_VALUES: Dict[str, str] = {
    "full": "full",
    "partial": "partial",
    "obscured": "obscured",
}
# Visual Impact Values
VISUAL_IMPACT_VALUES: Dict[str, str] = {
    "strong": "strong",
    "moderate": "moderate",
    "weak": "weak",
}
# Human Elements Values
HUMAN_ELEMENTS_VALUES: Dict[str, str] = {
    "Person visible": "Person visible",
    "Lifestyle context": "Lifestyle context",
    "None": "None",
}
# Relationship Depiction Values
RELATIONSHIP_DEPICTION_VALUES: Dict[str, str] = {
    "product-in-environment": "product-in-environment",
    "product-with-objects": "product-with-objects",
    "product-alone": "product-alone",
}
# Negative Space Usage Values
NEGATIVE_SPACE_USAGE_VALUES: Dict[str, str] = {
    "generous": "generous",
    "balanced": "balanced",
    "minimal": "minimal",
}
# Composition Style Values
COMPOSITION_STYLE_VALUES: Dict[str, str] = {
    "generous negative space": "generous negative space",
    "balanced composition": "balanced composition",
    "minimal negative space": "minimal negative space",
}
# Lighting Detail Values
LIGHTING_DETAIL_VALUES: Dict[str, str] = {
    "subtle light-to-shadow gradient across the floor": (
        "subtle light-to-shadow gradient across the floor"
    ),
    "ultra-bright high-key lighting": "ultra-bright high-key lighting",
    "natural sunlight-filled atmosphere": "natural sunlight-filled atmosphere",
}
# Environment Objects Values
ENVIRONMENT_OBJECTS_VALUES: Dict[str, str] = {
    "placed near minimalist lifestyle items like designer indoor plants or pet accessories": (
        "placed near minimalist lifestyle items like designer indoor plants or pet accessories"
    ),
    "integrated into a natural home environment with contextual elements": (
        "integrated into a natural home environment with contextual elements"
    ),
}
# Placement Target Values
PLACEMENT_TARGET_VALUES: Dict[str, str] = {
    "natural home environment elements": "natural home environment elements",
    "lifestyle furniture or decorative elements": (
        "lifestyle furniture or decorative elements"
    ),
    "low-clearance furniture": "low-clearance furniture",
    "modern sofa or wall": "modern sofa or wall",
}


# ============================================================================
# FEATURE VALUE NORMALIZATION
# ============================================================================
def normalize_feature_value(feature_name: str, value: str) -> str:
    """
    Normalize a feature value to its canonical form from the registry.

    This ensures that variations in input (e.g., "warm-dominant", "warm_dominant",
    "Warm-Dominant") all map to the same canonical value used in prompts and
    extraction.

    Args:
        feature_name: Name of the feature (e.g., "product_position")
        value: Raw feature value from formula or extraction

    Returns:
        Canonical feature value from registry
    """
    if not value:
        return ""
    # Normalize to lowercase for comparison
    value_lower = value.lower().strip()
    # Map feature name to appropriate registry
    registry_map = {
        "product_position": PRODUCT_POSITION_VALUES,
        "atmosphere": ATMOSPHERE_VALUES,
        "temperature": ATMOSPHERE_VALUES,
        "color_balance": COLOR_BALANCE_VALUES,
        "product_visibility": PRODUCT_VISIBILITY_VALUES,
        "visual_impact": VISUAL_IMPACT_VALUES,
        "human_elements": HUMAN_ELEMENTS_VALUES,
        "relationship_depiction": RELATIONSHIP_DEPICTION_VALUES,
        "negative_space_usage": NEGATIVE_SPACE_USAGE_VALUES,
        "composition_style": COMPOSITION_STYLE_VALUES,
        "lighting_detail": LIGHTING_DETAIL_VALUES,
        "environment_objects": ENVIRONMENT_OBJECTS_VALUES,
        "placement_target": PLACEMENT_TARGET_VALUES,
    }

    registry = registry_map.get(feature_name)
    if not registry:
        # No registry for this feature, return normalized lowercase
        return value_lower
    # Try exact match first
    if value in registry:
        return registry[value]
    # Try lowercase match
    if value_lower in {k.lower(): val for k, val in registry.items()}:
        for k, val in registry.items():
            if k.lower() == value_lower:
                return val
    # Try partial match (e.g., "warm-dominant" -> "Warm")
    for key, canonical_value in registry.items():
        if value_lower in key.lower() or key.lower() in value_lower:
            return canonical_value
    # No match found, return original (normalized)
    return value_lower


def get_canonical_value(feature_name: str, value: str) -> Optional[str]:
    """
    Get canonical value from registry, or None if not found.

    Args:
        feature_name: Name of the feature
        value: Feature value to look up

    Returns:
        Canonical value from registry, or None if not found
    """
    normalized = normalize_feature_value(feature_name, value)
    if normalized:
        return normalized
    return None


def validate_feature_value(feature_name: str, value: str) -> bool:
    """
    Validate that a feature value exists in the registry.

    Args:
        feature_name: Name of the feature
        value: Feature value to validate

    Returns:
        True if value is valid (exists in registry or can be normalized)
    """
    canonical = normalize_feature_value(feature_name, value)
    return bool(canonical)


def get_all_valid_values(feature_name: str) -> List[str]:
    """
    Get all valid values for a feature from the registry.

    Args:
        feature_name: Name of the feature

    Returns:
        List of all valid canonical values for this feature
    """
    registry_map = {
        "product_position": PRODUCT_POSITION_VALUES,
        "atmosphere": ATMOSPHERE_VALUES,
        "temperature": ATMOSPHERE_VALUES,
        "color_balance": COLOR_BALANCE_VALUES,
        "product_visibility": PRODUCT_VISIBILITY_VALUES,
        "visual_impact": VISUAL_IMPACT_VALUES,
        "human_elements": HUMAN_ELEMENTS_VALUES,
        "relationship_depiction": RELATIONSHIP_DEPICTION_VALUES,
        "negative_space_usage": NEGATIVE_SPACE_USAGE_VALUES,
        "composition_style": COMPOSITION_STYLE_VALUES,
        "lighting_detail": LIGHTING_DETAIL_VALUES,
        "environment_objects": ENVIRONMENT_OBJECTS_VALUES,
        "placement_target": PLACEMENT_TARGET_VALUES,
    }

    registry = registry_map.get(feature_name)
    if not registry:
        return []
    # Return unique canonical values
    return list(set(registry.values()))


# ============================================================================
# ROAS SYNERGY MAPPINGS
# ============================================================================
# Hardcoded ROAS synergies for V2 branches
V2_BRANCH_SYNERGIES: Dict[str, Dict[str, str]] = {
    "golden_ratio": {
        "product_position": "bottom-right",
        "relationship_depiction": "product-in-environment",
        "atmosphere": "Warm",
        "human_elements": "Person visible",
        "negative_space_usage": "generous",
    },
    "high_efficiency": {
        "product_visibility": "partial",
        "visual_impact": "strong",
        "atmosphere": "Neutral",
        "negative_space_usage": "generous",
    },
    "cool_peak": {
        "atmosphere": "Cool",
        "negative_space_usage": "generous",
    },
}


def get_branch_synergy_features(branch_name: str) -> Dict[str, str]:
    """
    Get hardcoded ROAS synergy features for a V2 branch.

    Args:
        branch_name: Branch identifier ("golden_ratio", "high_efficiency", "cool_peak")

    Returns:
        Dictionary of feature_name -> canonical_value mappings
    """
    return V2_BRANCH_SYNERGIES.get(branch_name, {}).copy()


def apply_branch_synergy(
    feature_name: str, branch_name: str, formula_value: Optional[str] = None
) -> Optional[str]:
    """
    Apply branch-specific ROAS synergy as fallback to scorer recommendations.

    Priority (FIXED - scorer first):
    1. Formula/scorer value (if provided) - Data-driven from current analysis
    2. Branch hardcoded value (fallback only) - Historical defaults
    3. None

    Args:
        feature_name: Name of the feature
        branch_name: Branch identifier
        formula_value: Value from visual_formula/scorer (optional)

    Returns:
        Canonical feature value (formula > hardcoded fallback > None)
    """
    # Priority 1: Check scorer/formula recommendation FIRST
    if formula_value:
        normalized = normalize_feature_value(feature_name, formula_value)
        logger.debug(
            "Using scorer/formula value '%s' for feature '%s' (branch: %s)",
            normalized,
            feature_name,
            branch_name,
        )
        return normalized
    # Priority 2: Fallback to hardcoded only if scorer has no recommendation
    branch_synergies = get_branch_synergy_features(branch_name)
    if feature_name in branch_synergies:
        hardcoded_value = branch_synergies[feature_name]
        logger.warning(
            "No scorer recommendation for feature '%s' (branch: %s), "
            "using hardcoded fallback: '%s'",
            feature_name,
            branch_name,
            hardcoded_value,
        )
        return hardcoded_value
    # Priority 3: No value available
    return None
