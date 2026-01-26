"""
Anti-Hallucination Enhancement Layer.

Provides product integrity enforcement rules for prompt generation.
Ensures generated images maintain exact product geometry, components,
colors, text, and brand identifiers.

All rules follow KISS principle - deterministic, rules-based logic.
"""

from typing import Dict, List


# Enhanced anti-hallucination rules organized by category
ANTI_HALLUCINATION_RULES = {
    "text_preservation": [
        "ALL text/labels must be sharp and readable at 100% zoom",
        "NO missing letters, NO blur, NO distortion, NO partial occlusion",
        "Font weight must be bold enough for web/social media visibility",
        "Text contrast ratio must exceed 4.5:1 (WCAG AA standard)",
        "ALL brand identifiers, logos, labels preserved exactly",
    ],
    "component_integrity": [
        "Preserve ALL components exactly as in Image 1",
        "NO missing parts, NO added accessories, NO decorative elements",
        "Component layout and positions must match source exactly",
        "Component assembly and relationships must be preserved",
        "Show ONLY ONE product instance - no duplicates",
    ],
    "color_accuracy": [
        "Match ALL product colors exactly with <2% tolerance",
        "ΔE (Delta E) < 2.0 - imperceptible difference from source",
        "NO white balance shifts, NO tinting, NO color casts",
        "Product colors must remain accurate under all lighting",
        "Metamerism: Colors match under D50 illuminant (standard viewing)",
    ],
    "geometry_preservation": [
        "Strictly maintain exact geometric structure and proportions",
        "NO aspect ratio changes, NO stretching, NO distortion",
        "Product scale and sizing must match source exactly",
        "Perspective and viewing angle must be consistent",
        "Shape contours and silhouettes must be identical",
    ],
    "material_integrity": [
        "Material appearance must match source (metal, plastic, glass, etc.)",
        "Surface texture and finish quality preserved exactly",
        "Transparency/opacity levels maintained",
        "Reflectivity and gloss levels match source",
        "NO material substitutions or appearance changes",
    ],
    "prohibited_actions": [
        "Do NOT redesign, modify, or create variations of the product",
        "Do NOT add elements not visible in Image 1",
        "Do NOT remove or omit any visible components",
        "Do NOT change product proportions or assembly",
        "ONLY change background/scene - product 100% identical to Image 1",
    ],
}


def get_anti_hallucination_block(
    categories: List[str] = None,
    enhanced: bool = True,
) -> str:
    """
    Generate anti-hallucination constraint block for prompts.

    Args:
        categories: List of rule categories to include (default: all)
        enhanced: If True, use enhanced 18-line block; if False, use concise 6-line block

    Returns:
        Formatted anti-hallucination constraint string

    Example:
        >>> block = get_anti_hallucination_block(
        ...     categories=["text_preservation", "component_integrity"],
        ...     enhanced=True
        ... )
    """
    if categories is None:
        categories = list(ANTI_HALLUCINATION_RULES.keys())

    rules = []
    for category in categories:
        if category in ANTI_HALLUCINATION_RULES:
            rules.extend(ANTI_HALLUCINATION_RULES[category])

    if not enhanced:
        # Concise version - first rule from each category
        rules = [
            "Strictly maintain exact geometric structure and proportions of Image 1",
            "Preserve ALL components, text, logos, labels exactly as in Image 1",
            "Match ALL product colors exactly with <2% tolerance",
            "Do NOT redesign, modify, or create variations of the product",
            "Do NOT add elements not visible in Image 1",
            "ONLY change background/scene - product must be 100% identical to Image 1",
        ]

    return " ".join(f"[{rule}]" for rule in rules)


def get_brand_integrity_rules(brand: str = None) -> List[str]:
    """
    Get brand-specific integrity rules.

    Args:
        brand: Brand identifier (e.g., "moprobo", "ecoflow")

    Returns:
        List of brand-specific rules
    """
    base_rules = [
        "Logo placement: top-right or top-left (10% margin from edges)",
        "Logo size: 5-8% of image width",
        "Logo integrity: Sharp, fully readable, NO blur or distortion",
        "Logo protection: Clear space 2x logo width around it",
        "Brand colors: Must match brand palette (ΔE < 1.0 tolerance)",
        "NO color tinting or white balance shifts on brand elements",
    ]

    # Brand-specific rules
    brand_rules = {
        "moprobo": [
            "Primary colors: Red (#FF0000) and Black (#000000)",
            "Style: Minimalist tech aesthetic",
            "Logo always visible and readable",
        ],
        "ecoflow": [
            "Primary colors: Green and White",
            "Style: Eco-friendly, clean, natural",
            "Emphasis on sustainability messaging",
        ],
    }

    if brand and brand.lower() in brand_rules:
        return base_rules + brand_rules[brand.lower()]

    return base_rules


def format_anti_hallucination_for_template(
    enhanced: bool = True,
    brand: str = None,
) -> Dict[str, str]:
    """
    Format anti-hallucination rules for template injection.

    Args:
        enhanced: Use enhanced rules (18-line) vs concise (6-line)
        brand: Brand identifier for brand-specific rules

    Returns:
        Dict with formatted constraint blocks
    """
    if enhanced:
        block = get_anti_hallucination_block(enhanced=True)
    else:
        block = get_anti_hallucination_block(enhanced=False)

    brand_rules = get_brand_integrity_rules(brand)

    return {
        "anti_hallucination_block": block,
        "brand_integrity": " ".join(f"[{rule}]" for rule in brand_rules),
        "text_preservation": " ".join(
            f"[{rule}]"
            for rule in ANTI_HALLUCINATION_RULES["text_preservation"]
        ),
        "component_preservation": " ".join(
            f"[{rule}]"
            for rule in ANTI_HALLUCINATION_RULES["component_integrity"]
        ),
        "color_preservation": " ".join(
            f"[{rule}]"
            for rule in ANTI_HALLUCINATION_RULES["color_accuracy"]
        ),
    }
