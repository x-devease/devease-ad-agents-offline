"""
Advanced Color Science Module.

Provides precise color accuracy specifications and advanced color rendering
descriptions for prompt generation.

All values are based on color science standards (CIE, sRGB, WCAG).
"""

from typing import Dict, List, Optional


# Color accuracy specifications
COLOR_ACCURACY = {
    "delta_e_tolerance": {
        "imperceptible": {
            "delta_e": "< 1.0",
            "description": "Imperceptible difference - expert level",
        },
        "acceptable": {
            "delta_e": "< 2.0",
            "description": "Acceptable commercial quality - imperceptible to most",
        },
        "noticeable": {
            "delta_e": "< 5.0",
            "description": "Noticeable difference but acceptable",
        },
    },
    "metamerism": {
        "description": "Colors match under specific illuminant",
        "standard_illuminant": "D50 (5000K) - standard viewing condition",
        "alternatives": ["D65 (6500K)", "A (2856K)", "F11 (fluorescent)"],
    },
    "bit_depth": {
        "8_bit": "256 levels per channel (standard sRGB)",
        "10_bit": "1024 levels per channel (professional grade)",
        "12_bit": "4096 levels per channel (high-end professional)",
        "16_bit": "65536 levels per channel (cinema grade)",
    },
    "color_space": {
        "srgb": "Standard RGB for web/screens (gamma 2.2)",
        "adobe_rgb": "Wider gamut for print",
        "prophoto_rgb": "Ultra-wide gamut for professional photography",
        "rec2020": "Ultra-wide gamut for HDR video",
    },
}


def get_color_accuracy_spec(
    tolerance_level: str = "acceptable",
    bit_depth: int = 10,
    color_space: str = "srgb",
) -> str:
    """
    Get color accuracy specification for prompt.

    Args:
        tolerance_level: Delta E tolerance (imperceptible, acceptable, noticeable)
        bit_depth: Bit depth for color precision (8, 10, 12, 16)
        color_space: Color space (srgb, adobe_rgb, prophoto_rgb, rec2020)

    Returns:
        Formatted color accuracy specification
    """
    specs = []

    # Delta E tolerance
    if tolerance_level in COLOR_ACCURACY["delta_e_tolerance"]:
        delta_e_info = COLOR_ACCURACY["delta_e_tolerance"][tolerance_level]
        specs.append(f"ΔE {delta_e_info['delta_e']} - {delta_e_info['description']}")

    # Metamerism
    meta_info = COLOR_ACCURACY["metamerism"]
    specs.append(f"{meta_info['description']}: {meta_info['standard_illuminant']}")

    # Bit depth
    bit_key = f"{bit_depth}_bit"
    if bit_key in COLOR_ACCURACY["bit_depth"]:
        specs.append(f"Bit depth: {COLOR_ACCURACY['bit_depth'][bit_key]}")

    # Color space
    if color_space in COLOR_ACCURACY["color_space"]:
        specs.append(f"Color space: {COLOR_ACCURACY['color_space'][color_space]}")

    return " ".join(f"[{spec}]" for spec in specs)


# Advanced color rendering
ADVANCED_COLOR_RENDERING = {
    "subsurface_scattering": {
        "description": "Light traveling through semi-transparent materials",
        "materials": ["skin", "wax", "milk", "flesh", "marble", "soap"],
        "effect": "Soft, diffused appearance with internal glow",
    },
    "thin_film_interference": {
        "description": "Rainbow effects from thin transparent layers",
        "materials": ["oil", "soap_bubbles", "coatings", "cd_surfaces"],
        "effect": "Colorful iridescent patterns",
    },
    "fluorescence": {
        "description": "Materials emitting light under UV excitation",
        "materials": ["optical_brighteners", "safety_vest", "certain_minerals"],
        "effect": "Glowing appearance under UV light",
    },
    "iridescence": {
        "description": "Color-changing effects with viewing angle",
        "materials": ["pearlescent", "mother_of_pearl", "cd_back", "certain_fabrics"],
        "effect": "Shifting colors with angle (goniochromism)",
    },
    "goniochromism": {
        "description": "Angle-dependent color appearance",
        "materials": ["metallic_flake", "pearl", "holographic"],
        "effect": "Color shifts with viewing angle",
    },
}


def get_advanced_color_rendering(
    effects: List[str] = None,
) -> str:
    """
    Get advanced color rendering effects description.

    Args:
        effects: List of effects to include (subsurface_scattering,
                thin_film_interference, fluorescence, iridescence, goniochromism)

    Returns:
        Formatted advanced color rendering description
    """
    if effects is None:
        return ""

    descriptions = []
    for effect in effects:
        if effect in ADVANCED_COLOR_RENDERING:
            effect_info = ADVANCED_COLOR_RENDERING[effect]
            descriptions.append(
                f"{effect_info['description']}. "
                f"Materials: {', '.join(effect_info['materials'])}. "
                f"Effect: {effect_info['effect']}."
            )

    return " ".join(f"[{desc}]" for desc in descriptions)


# Color interaction effects
COLOR_INTERACTION = {
    "simultaneous_contrast": {
        "description": "Colors appear different based on surrounding colors",
        "effect": "Gray appears darker on white, lighter on black",
    },
    "color_assimilation": {
        "description": "Adjacent colors blend at boundaries",
        "effect": "Subtle color bleeding between adjacent areas",
    },
    "bezold_effect": {
        "description": "Small areas take on hue of surrounding area",
        "effect": "Color assimilation for small color areas",
    },
}


def get_color_interaction_description(
    interactions: List[str] = None,
) -> str:
    """
    Get color interaction effect descriptions.

    Args:
        interactions: List of interactions (simultaneous_contrast,
                      color_assimilation, bezold_effect)

    Returns:
        Formatted color interaction description
    """
    if interactions is None:
        return ""

    descriptions = []
    for interaction in interactions:
        if interaction in COLOR_INTERACTION:
            interaction_info = COLOR_INTERACTION[interaction]
            descriptions.append(
                f"{interaction_info['description']}. "
                f"Effect: {interaction_info['effect']}."
            )

    return " ".join(f"[{desc}]" for desc in descriptions)


# WCAG accessibility requirements
WCAG_REQUIREMENTS = {
    "normal_text": {
        "AA": "4.5:1 contrast ratio (minimum)",
        "AAA": "7:1 contrast ratio (enhanced)",
    },
    "large_text": {
        "AA": "3:1 contrast ratio (minimum)",
        "AAA": "4.5:1 contrast ratio (enhanced)",
    },
    "ui_components": {
        "AA": "3:1 contrast ratio against adjacent colors",
    },
}


def get_wcag_compliance(
    level: str = "AA",
    text_type: str = "normal_text",
) -> str:
    """
    Get WCAG accessibility compliance specification.

    Args:
        level: Compliance level (AA, AAA)
        text_type: Type of text (normal_text, large_text, ui_components)

    Returns:
        Formatted WCAG compliance specification
    """
    if text_type not in WCAG_REQUIREMENTS:
        return ""

    if level not in WCAG_REQUIREMENTS[text_type]:
        return ""

    return (
        f"[WCAG {level} compliance: {WCAG_REQUIREMENTS[text_type][level]}] "
        f"Text must be readable and accessible"
    )


# Complete color specification for prompts
def get_complete_color_specification(
    tolerance_level: str = "acceptable",
    bit_depth: int = 10,
    color_space: str = "srgb",
    wcag_level: str = "AA",
    advanced_effects: List[str] = None,
    color_interactions: List[str] = None,
) -> str:
    """
    Get complete color specification for prompt injection.

    Args:
        tolerance_level: Delta E tolerance (imperceptible, acceptable, noticeable)
        bit_depth: Bit depth for color precision (8, 10, 12, 16)
        color_space: Color space (srgb, adobe_rgb, prophoto_rgb, rec2020)
        wcag_level: WCAG compliance level (AA, AAA)
        advanced_effects: List of advanced rendering effects
        color_interactions: List of color interaction effects

    Returns:
        Complete formatted color specification

    Example:
        >>> spec = get_complete_color_specification(
        ...     tolerance_level="acceptable",
        ...     bit_depth=10,
        ...     wcag_level="AA",
        ...     advanced_effects=["subsurface_scattering"]
        ... )
    """
    specs = []

    # Color accuracy
    color_accuracy = get_color_accuracy_spec(tolerance_level, bit_depth, color_space)
    if color_accuracy:
        specs.append(color_accuracy)

    # WCAG compliance
    wcag = get_wcag_compliance(wcag_level, "normal_text")
    if wcag:
        specs.append(wcag)

    # Advanced effects
    if advanced_effects:
        advanced = get_advanced_color_rendering(advanced_effects)
        if advanced:
            specs.append(advanced)

    # Color interactions
    if color_interactions:
        interactions = get_color_interaction_description(color_interactions)
        if interactions:
            specs.append(interactions)

    return " ".join(specs)
