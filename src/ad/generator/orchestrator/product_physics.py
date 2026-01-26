"""
Product-Specific Physics Module.

Provides physics descriptions for different product types including
liquid dynamics, fabric behavior, flexible materials, and transparent materials.

Based on real-world physics and material science principles.
"""

from typing import Dict, List, Optional


# Liquid container physics
LIQUID_PHYSICS = {
    "fluid_dynamics": {
        "water": {
            "viscosity": "Low viscosity, flows freely",
            "clarity": "Clear, transparent",
            "surface_tension": "Curved meniscus at edges (concave)",
            "behavior": "Rapid movement, quick settling",
        },
        "oil": {
            "viscosity": "Medium viscosity, flows slowly",
            "clarity": "Subtle yellow tint",
            "surface_tension": "Curved meniscus (less pronounced than water)",
            "behavior": "Smooth, viscous flow",
        },
        "syrup": {
            "viscosity": "High viscosity, very slow flow",
            "clarity": "Amber color, translucent",
            "surface_tension": "Minimal surface tension effects",
            "behavior": "Very slow movement, clings to surfaces",
        },
    },
    "liquid_effects": {
        "bubbles": {
            "description": "Tiny air bubbles visible in liquid",
            "visibility": "Subtle, realistic distribution",
            "purpose": "Shows liquid transparency and volume",
        },
        "refraction": {
            "description": "Liquid bends light passing through",
            "index": "Refractive index 1.33 for water",
            "effect": "Objects appear shifted when viewed through liquid",
        },
        "caustics": {
            "description": "Light patterns cast on surfaces through liquid",
            "visibility": "Subtle caustic patterns from focused light",
            "purpose": "Shows liquid's optical properties",
        },
        "meniscus": {
            "description": "Curved liquid surface at container edges",
            "shape": "Concave for water (wets glass), convex for mercury",
            "visibility": "Visible at container walls",
        },
    },
}


def get_liquid_physics(
    liquid_type: str = "water",
    include_effects: bool = True,
) -> str:
    """
    Get liquid physics description.

    Args:
        liquid_type: Type of liquid (water, oil, syrup)
        include_effects: Include optical effects (bubbles, refraction, caustics)

    Returns:
        Formatted liquid physics description
    """
    components = []

    # Fluid dynamics
    if liquid_type in LIQUID_PHYSICS["fluid_dynamics"]:
        fluid = LIQUID_PHYSICS["fluid_dynamics"][liquid_type]
        components.append(f"[Liquid type: {liquid_type}]")
        components.append(f"[Viscosity: {fluid['viscosity']}]")
        components.append(f"[Clarity: {fluid['clarity']}]")
        components.append(f"[Surface tension: {fluid['surface_tension']}]")
        components.append(f"[Behavior: {fluid['behavior']}]")

    # Optical effects
    if include_effects:
        for effect_name, details in LIQUID_PHYSICS["liquid_effects"].items():
            components.append(f"[{effect_name}: {details['description']}]")
            if "index" in details:
                components.append(f"[Refractive index: {details['index']}]")
            if "visibility" in details:
                components.append(f"[Visibility: {details['visibility']}]")
            if "purpose" in details:
                components.append(f"[Purpose: {details['purpose']}]")
            if "shape" in details:
                components.append(f"[Shape: {details['shape']}]")

    return " ".join(components)


# Flexible material physics
FLEXIBLE_MATERIAL_PHYSICS = {
    "fabric_drape": {
        "description": "Natural folding and gravity effects on fabric",
        "gravity": "Natural downward sag due to weight",
        "tension": "Tight where stretched, loose elsewhere",
        "folds": "Soft, rounded creases (not sharp)",
        "wrinkles": "Random, organic distribution",
    },
    "elastic_deformation": {
        "description": "Material returns to original shape after stress",
        "behavior": "Temporary deformation under load",
        "recovery": "Full recovery when stress removed",
    },
    "plastic_deformation": {
        "description": "Permanent creases from folding",
        "behavior": "Material retains crease marks",
        "recovery": "No recovery - permanent change",
    },
    "flexibility_types": {
        "cotton": "Moderate drape, natural wrinkles",
        "silk": "Fluid drape, subtle sheen",
        "denim": "Stiff drape, sharp creases",
        "wool": "Heavy drape, warmth retention",
        "synthetic": "Variable drape based on weave",
    },
}


def get_flexible_material_physics(
    material_type: str = "cotton",
    behavior: str = "natural_drape",
) -> str:
    """
    Get flexible material physics description.

    Args:
        material_type: Type of material (cotton, silk, denim, wool, synthetic)
        behavior: Type of behavior (natural_drape, stretched, compressed)

    Returns:
        Formatted flexible material physics description
    """
    components = []

    # General drape behavior
    for key, details in FLEXIBLE_MATERIAL_PHYSICS.items():
        if key == "fabric_drape":
            components.append(f"[{details['description']}]")
            for subkey, value in details.items():
                if subkey != "description":
                    components.append(f"[{subkey}: {value}]")
        elif key == "flexibility_types" and material_type in details:
            components.append(f"[Material: {material_type}, Properties: {details[material_type]}]")

    # Behavior-specific
    if behavior == "stretched":
        components.append(f"[{FLEXIBLE_MATERIAL_PHYSICS['elastic_deformation']['description']}]")
        components.append(f"[Behavior: {FLEXIBLE_MATERIAL_PHYSICS['elastic_deformation']['behavior']}]")
    elif behavior == "compressed":
        components.append(f"[{FLEXIBLE_MATERIAL_PHYSICS['plastic_deformation']['description']}]")
        components.append(f"[Behavior: {FLEXIBLE_MATERIAL_PHYSICS['plastic_deformation']['behavior']}]")

    return " ".join(components)


# Transparent material physics
TRANSPARENT_MATERIAL_PHYSICS = {
    "glass": {
        "refraction": {
            "index": "1.5 (standard glass)",
            "description": "Light bending through material",
            "effect": "Objects appear shifted when viewed through",
        },
        "reflection": {
            "fresnel": "4-8% surface reflection (angle-dependent)",
            "description": "Reflectivity increases at grazing angles",
        },
        "transmission": {
            "percentage": "90% light transmission (clear glass)",
            "absorption": "Minimal absorption in clear glass",
        },
        "dispersion": {
            "description": "RGB color separation at edges",
            "effect": "Prism-like color fringing on edges",
        },
        "caustics": {
            "description": "Light patterns cast through glass",
            "effect": "Focused light beams on surfaces",
        },
    },
    "transparent_plastic": {
        "refraction": {
            "index": "1.4-1.5 (varies by type)",
            "description": "Slightly less refraction than glass",
        },
        "subsurface_scattering": {
            "description": "Light penetration into material",
            "effect": "Soft, diffuse appearance",
        },
        "transmission": {
            "percentage": "85-95% depending on formulation",
        },
    },
}


def get_transparent_material_physics(
    material_type: str = "glass",
    include_all_effects: bool = True,
) -> str:
    """
    Get transparent material physics description.

    Args:
        material_type: Type of transparent material (glass, transparent_plastic)
        include_all_effects: Include all optical effects

    Returns:
        Formatted transparent material physics description
    """
    components = []

    if material_type not in TRANSPARENT_MATERIAL_PHYSICS:
        return ""

    material = TRANSPARENT_MATERIAL_PHYSICS[material_type]

    if include_all_effects:
        for effect_name, details in material.items():
            components.append(f"[{effect_name}: {details['description']}]")
            if "index" in details:
                components.append(f"[Index: {details['index']}]")
            if "effect" in details:
                components.append(f"[Effect: {details['effect']}]")
            if "percentage" in details:
                components.append(f"[Transmission: {details['percentage']}]")
    else:
        # Just basic refraction and reflection
        components.append(f"[Refraction: {material['refraction']['description']}]")
        components.append(f"[Reflection: {material['reflection']['description']}]")

    return " ".join(components)


# Product type detection
PRODUCT_TYPE_PATTERNS = {
    "liquid_container": [
        "bottle", "jar", "cup", "glass", "mug", "pitcher", "vase",
        "container", "dispenser", "bottle of", "jug", "carafe"
    ],
    "fabric": [
        "shirt", "pants", "dress", "cloth", "textile", "fabric", "towel",
        "curtain", "upholstery", "linen", "blanket", "cloth"
    ],
    "flexible": [
        "hose", "cable", "cord", "tube", "pipe", "belt", "strap",
        "band", "ribbon", "flexible", "bendable"
    ],
    "transparent": [
        "glass", "window", "transparent", "clear", "lens", "mirror",
        "crystal", "optic", "see-through"
    ],
}


def detect_product_type(product_name: str) -> List[str]:
    """
    Detect product type from product name using keyword matching.

    Args:
        product_name: Name of the product

    Returns:
        List of detected product types
    """
    detected = []
    name_lower = product_name.lower()

    for product_type, keywords in PRODUCT_TYPE_PATTERNS.items():
        if any(kw in name_lower for kw in keywords):
            detected.append(product_type)

    return detected


def get_product_physics_description(
    product_name: str,
    include_all_effects: bool = True,
) -> str:
    """
    Get product-specific physics description based on product type.

    Args:
        product_name: Name of the product
        include_all_effects: Include all detailed effects

    Returns:
        Formatted product physics description

    Example:
        >>> desc = get_product_physics_description("Water Bottle")
        >>> # Returns liquid container physics
    """
    product_types = detect_product_type(product_name)
    components = []

    for product_type in product_types:
        if product_type == "liquid_container":
            components.append(get_liquid_physics("water", include_all_effects))
        elif product_type == "fabric":
            components.append(get_flexible_material_physics("cotton", "natural_drape"))
        elif product_type == "flexible":
            components.append(get_flexible_material_physics("synthetic", "natural_drape"))
        elif product_type == "transparent":
            components.append(get_transparent_material_physics("glass", include_all_effects))

    return " ".join(components) if components else ""


# Product state descriptions
PRODUCT_STATES = {
    "leaning": {
        "description": "Product leaning against surface",
        "angle": "60-75° from vertical",
        "contact": "Point contact along edge",
        "stability": "Firm contact shadow for grounding",
    },
    "standing": {
        "description": "Product standing upright on base",
        "angle": "90° from horizontal",
        "contact": "Full base contact",
        "stability": "Wide, stable contact shadow",
    },
    "lying_flat": {
        "description": "Product lying horizontally (180°)",
        "angle": "0° from horizontal",
        "contact": "Full surface contact",
        "stability": "Maximum surface area contact",
    },
    "suspended": {
        "description": "Product hanging or suspended",
        "support": "Supported from above",
        "gravity": "Hanging downward due to gravity",
        "contact": "No ground contact (floating)",
    },
}


def get_product_state_description(state: str) -> str:
    """
    Get product state description.

    Args:
        state: Product state (leaning, standing, lying_flat, suspended)

    Returns:
        Formatted product state description
    """
    if state not in PRODUCT_STATES:
        return ""

    state_info = PRODUCT_STATES[state]
    components = []

    components.append(f"[State: {state_info['description']}]")
    for key, value in state_info.items():
        if key != "description":
            components.append(f"[{key}: {value}]")

    return " ".join(components)
