"""
Brand Identity Enforcement Module.

Provides brand-specific guidelines and enforcement rules for prompt generation.
Ensures consistent brand representation across generated images.

Includes logo handling, color enforcement, typography rules, and visual style.
"""

from typing import Dict, List, Optional


# Brand-specific guidelines
BRAND_GUIDELINES = {
    "moprobo": {
        "name": "Moprobo",
        "primary_colors": ["#FF0000", "#000000"],  # Red, Black
        "secondary_colors": ["#FFFFFF", "#333333"],
        "style": "Minimalist tech aesthetic",
        "logo_placement": "top-right or top-left",
        "logo_size": "5-8% of image width",
        "logo_protection": "Clear space 2x logo width around it",
        "visual_language": "Clean, modern, technical precision",
        "tone": "Professional, confident, straightforward",
    },
    "ecoflow": {
        "name": "EcoFlow",
        "primary_colors": ["#00A651", "#FFFFFF"],  # Green, White
        "secondary_colors": ["#333333", "#E8F5E9"],
        "style": "Eco-friendly, clean, natural",
        "logo_placement": "top-left or centered top",
        "logo_size": "6-10% of image width",
        "logo_protection": "Clear space 1.5x logo width",
        "visual_language": "Nature-inspired, sustainable",
        "tone": "Friendly, environmentally conscious",
    },
    "generic": {
        "name": "Generic Brand",
        "primary_colors": ["#000000", "#FFFFFF"],  # Black, White
        "secondary_colors": ["#333333", "#CCCCCC"],
        "style": "Professional, neutral",
        "logo_placement": "top-right corner",
        "logo_size": "5-8% of image width",
        "logo_protection": "Clear space 2x logo width",
        "visual_language": "Clean, straightforward",
        "tone": "Professional",
    },
}


def get_brand_guidelines(brand: str = "generic") -> Dict:
    """
    Get brand guidelines by brand name.

    Args:
        brand: Brand identifier

    Returns:
        Brand guidelines dictionary
    """
    brand_lower = brand.lower()

    # Check for exact match
    if brand_lower in BRAND_GUIDELINES:
        return BRAND_GUIDELINES[brand_lower]

    # Fuzzy match
    for brand_key, guidelines in BRAND_GUIDELINES.items():
        if brand_key in brand_lower or brand_lower in brand_key:
            return guidelines

    # Default to generic
    return BRAND_GUIDELINES["generic"]


def get_brand_color_spec(brand: str = "generic") -> str:
    """
    Get brand color specification.

    Args:
        brand: Brand identifier

    Returns:
        Formatted brand color specification
    """
    guidelines = get_brand_guidelines(brand)
    components = []

    components.append(f"[Brand: {guidelines['name']}]")
    components.append(f"[Primary Colors: {', '.join(guidelines['primary_colors'])}]")
    components.append(f"[Secondary Colors: {', '.join(guidelines['secondary_colors'])}]")
    components.append(f"[ΔE < 1.0 tolerance - exact match required]")

    return " ".join(components)


# Logo handling specifications
LOGO_SPECIFICATIONS = {
    "placement": {
        "top_right": {
            "margin": "10% from right edge, 10% from top",
            "best_for": "Most products, standard placement",
        },
        "top_left": {
            "margin": "10% from left edge, 10% from top",
            "best_for": "Balanced composition",
        },
        "centered_top": {
            "margin": "10% from top, centered horizontally",
            "best_for": "Hero products, symmetrical designs",
        },
        "bottom_right": {
            "margin": "10% from right edge, 10% from bottom",
            "best_for": "Signature style, branding",
        },
    },
    "size": {
        "small": "3-5% of image width",
        "medium": "5-8% of image width (recommended)",
        "large": "8-12% of image width",
    },
    "integrity": {
        "sharpness": "100% sharp, no blur, no distortion",
        "readability": "Fully readable at 100% zoom",
        "color_accuracy": "Exact brand colors (ΔE < 1.0)",
        "preservation": "All elements intact, no missing parts",
    },
    "protection": {
        "clear_space": "2x logo width around logo",
        "no_clutter": "No text or graphics overlapping protection zone",
        "background": "Contrasting background for visibility",
        "priority": "Logo always visible, never obscured",
    },
}


def get_logo_specification(
    placement: str = "top_right",
    size: str = "medium",
    brand: str = "generic",
) -> str:
    """
    Get logo specification.

    Args:
        placement: Logo placement position
        size: Logo size category
        brand: Brand identifier

    Returns:
        Formatted logo specification
    """
    components = []

    # Get brand guidelines
    guidelines = get_brand_guidelines(brand)

    # Placement
    if placement in LOGO_SPECIFICATIONS["placement"]:
        place_spec = LOGO_SPECIFICATIONS["placement"][placement]
        components.append(f"[Logo Placement: {placement.replace('_', ' ').title()}]")
        components.append(f"[Position: {place_spec['margin']}]")
        components.append(f"[Best For: {place_spec['best_for']}]")

    # Size
    if size in LOGO_SPECIFICATIONS["size"]:
        size_spec = LOGO_SPECIFICATIONS["size"][size]
        components.append(f"[Logo Size: {size_spec}]")

    # Use brand-specific if available
    if "logo_placement" in guidelines:
        components.append(f"[Brand Preference: {guidelines['logo_placement']}]")
    if "logo_size" in guidelines:
        components.append(f"[Brand Size: {guidelines['logo_size']}]")

    # Integrity
    integrity = LOGO_SPECIFICATIONS["integrity"]
    components.append(f"[Logo Integrity: {integrity['sharpness']}]")
    components.append(f"[{integrity['readability']}]")
    components.append(f"[{integrity['color_accuracy']}]")
    components.append(f"[{integrity['preservation']}]")

    # Protection
    protection = LOGO_SPECIFICATIONS["protection"]
    components.append(f"[Logo Protection: {protection['clear_space']}]")
    components.append(f"[{protection['no_clutter']}]")
    components.append(f"[{protection['background']}]")
    components.append(f"[{protection['priority']}]")

    return " ".join(components)


# Typography specifications
TYPOGRAPHY_SPECS = {
    "legibility": {
        "minimum_size": "Readable at 100% zoom",
        "contrast_ratio": "> 4.5:1 (WCAG AA standard)",
        "font_weight": "Bold or heavy for visibility",
        "anti_aliasing": "Crisp edges, no pixelation",
    },
    "preservation": {
        "text_clarity": "All text must be sharp and crisp",
        "no_blur": "NO blur or distortion on any text",
        "no_missing_letters": "NO missing or occluded letters",
        "complete_integrity": "ALL text elements fully visible",
    },
    "brand_text": {
        "logo_text": "Logo text must be fully readable",
        "product_labels": "Product labels preserved exactly",
        "safety_warnings": "Warning text fully visible if present",
        "regulatory_marks": "Certification marks preserved",
    },
}


def get_typography_spec() -> str:
    """
    Get typography specification.

    Returns:
        Formatted typography specification
    """
    typo = []

    # Legibility
    leg = TYPOGRAPHY_SPECS["legibility"]
    typo.append(f"[Typography Legibility]")
    typo.append(f"[{leg['minimum_size']}]")
    typo.append(f"[Contrast Ratio: {leg['contrast_ratio']}]")
    typo.append(f"[Font Weight: {leg['font_weight']}]")
    typo.append(f"[{leg['anti_aliasing']}]")

    # Preservation
    pres = TYPOGRAPHY_SPECS["preservation"]
    typo.append(f"[Text Preservation: {pres['text_clarity']}]")
    typo.append(f"[{pres['no_blur']}]")
    typo.append(f"[{pres['no_missing_letters']}]")
    typo.append(f"[{pres['complete_integrity']}]")

    # Brand text
    brand = TYPOGRAPHY_SPECS["brand_text"]
    typo.append(f"[Brand Text: {brand['logo_text']}]")
    typo.append(f"[{brand['product_labels']}]")
    typo.append(f"[{brand.get('safety_warnings', 'N/A')}]")
    typo.append(f"[{brand.get('regulatory_marks', 'N/A')}]")

    return " ".join(typo)


# Visual style enforcement
VISUAL_STYLE = {
    "minimalist_tech": {
        "description": "Clean, modern, technical precision",
        "characteristics": [
            "Simple backgrounds (solid colors, subtle gradients)",
            "Focus on product (no clutter)",
            "Technical accuracy emphasized",
            "Crisp, sharp details",
        ],
        "avoid": [
            "Ornate backgrounds",
            "Excessive props or decorations",
            "Distorted angles",
            "Artistic filters",
        ],
    },
    "eco_friendly": {
        "description": "Nature-inspired, sustainable messaging",
        "characteristics": [
            "Natural environments",
            "Soft, organic lighting",
            "Earth tones and natural colors",
            "Lifestyle context",
        ],
        "avoid": [
            "Harsh industrial settings",
            "Artificial/sterile environments",
            "Overly dramatic lighting",
            "Synthetic appearance",
        ],
    },
    "luxury": {
        "description": "Premium, elegant presentation",
        "characteristics": [
            "Sophisticated backgrounds",
            "Dramatic lighting",
            "Rich colors and textures",
            "High-end aesthetic",
        ],
        "avoid": [
            "Cheap or casual elements",
            "Cluttered compositions",
            "Flat lighting",
            "Basic backgrounds",
        ],
    },
}


def get_visual_style_enforcement(
    style: str,
    brand: str = "generic",
) -> str:
    """
    Get visual style enforcement specification.

    Args:
        style: Visual style name
        brand: Brand identifier

    Returns:
        Formatted visual style specification
    """
    style_lower = style.lower().replace(" ", "_")

    # Check if style exists
    if style_lower not in VISUAL_STYLE:
        return ""

    style_info = VISUAL_STYLE[style_lower]
    components = []

    components.append(f"[Visual Style: {style_info['description']}]")
    components.append(f"[Characteristics: {', '.join(style_info['characteristics'])}]")
    components.append(f"[Avoid: {', '.join(style_info['avoid'])}]")

    return " ".join(components)


# Complete brand identity enforcement
def get_brand_identity_enforcement(
    brand: str = "generic",
    include_logo: bool = True,
    include_typography: bool = True,
    include_style: bool = True,
) -> str:
    """
    Get complete brand identity enforcement specification.

    Args:
        brand: Brand identifier
        include_logo: Include logo specifications
        include_typography: Include typography specifications
        include_style: Include visual style

    Returns:
        Complete formatted brand identity specification
    """
    components = []

    # Brand guidelines
    guidelines = get_brand_guidelines(brand)
    components.append(f"[{guidelines['name']} Brand Identity]")
    components.append(f"[Style: {guidelines['style']}]")
    components.append(f"[Tone: {guidelines['tone']}]")

    # Colors
    components.append(get_brand_color_spec(brand))

    # Logo
    if include_logo:
        components.append(get_logo_specification(brand=brand))

    # Typography
    if include_typography:
        components.append(get_typography_spec())

    # Visual style
    if include_style and "style" in guidelines:
        style_name = guidelines["style"].lower().replace(" ", "_")
        # Map style to visual style
        style_map = {
            "minimalist tech aesthetic": "minimalist_tech",
            "eco-friendly, clean, natural": "eco_friendly",
        }
        if style_name in style_map:
            components.append(get_visual_style_enforcement(style_map[style_name], brand))

    return " ".join(components)


# Brand detection from product name
BRAND_PATTERNS = {
    "moprobo": ["moprobo", "moprobo power", "moprobo station"],
    "ecoflow": ["ecoflow", "ecoflow delta", "ecoflow river"],
}


def detect_brand(product_name: str) -> str:
    """
    Detect brand from product name.

    Args:
        product_name: Name of the product

    Returns:
        Detected brand identifier (or "generic")
    """
    name_lower = product_name.lower()

    for brand, patterns in BRAND_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower:
                return brand

    return "generic"
