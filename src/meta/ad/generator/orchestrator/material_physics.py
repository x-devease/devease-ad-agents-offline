"""
Material Physics Module.

Provides material-specific physics descriptions for enhanced prompt fidelity.
Includes optical properties, surface characteristics, and physical behaviors.

Rules-based, deterministic descriptions - no ML.
"""

from typing import Dict, List, Optional


# Material-specific physics
MATERIAL_PHYSICS = {
    "metal": {
        "anisotropic_reflections": (
            "Directional reflection patterns following grain structure. "
            "Brushed metals show directional highlights; polished metals show uniform reflections."
        ),
        "fresnel_effect": (
            "Stronger reflections at grazing angles. "
            "Reflectivity increases as viewing angle approaches surface parallel."
        ),
        "subsurface_scattering": "Minimal - metals are opaque with no light penetration.",
        "conductivity": "Thermally and electrically conductive (affects thermal imaging).",
    },
    "plastic": {
        "specular_highlights": "Sharp, concentrated specular highlights with distinct edges.",
        "subsurface_scattering": (
            "Light penetration into translucent areas. "
            "Soft glow effect in thin sections."
        ),
        "dispersion": "Light separation on curved edges creating chromatic effects.",
        "surface_finish": "Can be matte (diffuse) or glossy (specular).",
    },
    "glass": {
        "refraction": (
            "Light bending through material with refractive index 1.5 (standard glass). "
            "Objects appear shifted when viewed through."
        ),
        "caustics": "Light patterns cast on surfaces from focused light beams.",
        "chromatic_dispersion": "RGB color separation at edges (prism effect).",
        "reflection": "4-8% surface reflection (Fresnel equation).",
    },
    "wood": {
        "grain_direction": "Visible growth rings and texture direction patterns.",
        "anisotropic_reflectivity": "Direction-dependent shine following grain.",
        "subsurface_scattering": "Light penetration into grain structure.",
        "variability": "Natural color and texture variations between pieces.",
    },
    "fabric": {
        "fiber_structure": "Visible thread weave patterns (warp and weft).",
        "subsurface_scattering": "Soft light penetration creating diffuse appearance.",
        "drape_simulation": "Natural folding and gravity effects.",
        "surface_texture": "Visible nap, fuzz, or weave depending on fabric type.",
    },
    "ceramic": {
        "surface_properties": "Smooth, non-porous surface with diffuse reflectance.",
        "glaze_effects": "Glass-like surface coating with specular highlights.",
        "thermal_properties": "Poor thermal conductor.",
        "breakage_pattern": "Conchoidal fracture when broken.",
    },
    "rubber": {
        "elasticity": "Flexible material that returns to original shape.",
        "surface_texture": "Matte finish with diffuse reflectance.",
        "friction": "High coefficient of friction (grip).",
        "deformation": "Visible compression where weight applied.",
    },
    "leather": {
        "grain_structure": "Natural surface texture with pores and variations.",
        "patina": "Develops character with age and use.",
        "flexibility": "Bends and folds naturally.",
        "absorption": "Can absorb liquids (affects appearance).",
    },
}


def get_material_physics_description(
    material_type: str,
    properties: List[str] = None,
) -> str:
    """
    Get material physics description for prompt injection.

    Args:
        material_type: Type of material (metal, plastic, glass, wood, etc.)
        properties: List of specific properties to include (default: all)

    Returns:
        Formatted description string

    Example:
        >>> desc = get_material_physics_description("metal", ["anisotropic_reflections", "fresnel_effect"])
    """
    if material_type not in MATERIAL_PHYSICS:
        return ""

    material_data = MATERIAL_PHYSICS[material_type]

    if properties is None:
        properties = list(material_data.keys())

    descriptions = []
    for prop in properties:
        if prop in material_data:
            descriptions.append(material_data[prop])

    return " ".join(f"[{desc}]" for desc in descriptions)


def detect_material_from_product(
    product_name: str,
    product_context: Dict = None,
) -> List[str]:
    """
    Detect material types from product name and context.

    Args:
        product_name: Name of the product
        product_context: Additional product information

    Returns:
        List of detected material types

    Note: This is a simple rule-based detector - not ML-based.
    """
    materials = []
    name_lower = product_name.lower()

    # Simple keyword matching
    if any(kw in name_lower for kw in ["metal", "aluminum", "steel", "iron", "copper"]):
        materials.append("metal")
    if any(kw in name_lower for kw in ["plastic", "polymer", "acrylic", "polycarbonate"]):
        materials.append("plastic")
    if any(kw in name_lower for kw in ["glass", "mirror", "lens", "crystal"]):
        materials.append("glass")
    if any(kw in name_lower for kw in ["wood", "bamboo", "timber", "oak", "pine"]):
        materials.append("wood")
    if any(kw in name_lower for kw in ["fabric", "cloth", "textile", "cotton", "polyester"]):
        materials.append("fabric")
    if any(kw in name_lower for kw in ["ceramic", "porcelain", "pottery", "tile"]):
        materials.append("ceramic")
    if any(kw in name_lower for kw in ["rubber", "silicone", "elastic"]):
        materials.append("rubber")
    if any(kw in name_lower for kw in ["leather", "hide", "skin"]):
        materials.append("leather")

    return materials


def get_material_enhancement_for_prompt(
    product_name: str,
    product_context: Dict = None,
) -> str:
    """
    Generate material physics enhancement for prompt.

    Args:
        product_name: Name of the product
        product_context: Additional product information

    Returns:
        Formatted material physics description for prompt injection

    Example:
        >>> enhancement = get_material_enhancement_for_prompt("Power Station")
        >>> # Returns material physics description based on detected materials
    """
    materials = detect_material_from_product(product_name, product_context)

    if not materials:
        return ""

    descriptions = []
    for material in materials[:2]:  # Limit to 2 materials for prompt length
        desc = get_material_physics_description(material)
        if desc:
            descriptions.append(desc)

    return " ".join(descriptions) if descriptions else ""


# Surface micro-details for manufacturing and cleanliness
SURFACE_MICRO_DETAILS = {
    "manufacturing_marks": {
        "injection_molding": {
            "description": "Gate marks, parting lines, ejector pin circles",
            "visibility": "Subtle, requires close inspection",
        },
        "machining": {
            "description": "Tool marks, CNC patterns, surface finish Ra values",
            "visibility": "Visible directional patterns",
        },
        "casting": {
            "description": "Parting lines, slight surface texture variation",
            "visibility": "Subtle texture changes",
        },
        "3d_printing": {
            "description": "Layer lines, print bed texture",
            "visibility": "Visible stair-stepping on curves",
        },
    },
    "wear_and_tear": {
        "fingerprint_resistance": "Oleophobic coating reduces visible fingerprints",
        "scratch_resistance": "Micro-scratches from normal use (subtle)",
        "dust_accumulation": "Fine particles in recesses and crevices",
        "patina": "Subtle surface oxidation on exposed metals",
    },
    "cleanliness": {
        "dust_particles": "Subtle dust motes visible in light beams",
        "smudge_resistance": "No visible fingerprints or hand oils",
        "reflection_clarity": "Mirror surfaces show crisp reflections",
    },
}


def get_surface_micro_details(
    manufacturing_process: Optional[str] = None,
    include_wear: bool = False,
    include_cleanliness: bool = True,
) -> str:
    """
    Get surface micro-details description.

    Args:
        manufacturing_process: Type of manufacturing (injection_molding, machining, etc.)
        include_wear: Include wear and tear details
        include_cleanliness: Include cleanliness specifications

    Returns:
        Formatted surface details description
    """
    details = []

    # Manufacturing marks
    if manufacturing_process and manufacturing_process in SURFACE_MICRO_DETAILS["manufacturing_marks"]:
        mark_info = SURFACE_MICRO_DETAILS["manufacturing_marks"][manufacturing_process]
        details.append(f"Manufacturing: {mark_info['description']}. Visibility: {mark_info['visibility']}.")

    # Wear and tear
    if include_wear:
        for key, value in SURFACE_MICRO_DETAILS["wear_and_tear"].items():
            details.append(f"{key}: {value}")

    # Cleanliness
    if include_cleanliness:
        for key, value in SURFACE_MICRO_DETAILS["cleanliness"].items():
            details.append(f"{key}: {value}")

    return " ".join(f"[{detail}]" for detail in details)
