"""
Advanced Lighting Optics Module.

Provides precise lighting specifications including bounced light, ambient occlusion,
color temperature, and volumetric lighting effects.

Based on professional photography lighting principles.
"""

from typing import Dict, List, Optional


# Advanced lighting specifications
ADVANCED_LIGHTING = {
    "bounced_light": {
        "description": "Secondary illumination from light reflecting off surfaces",
        "floor_bounce": {
            "direction": "Soft upward fill from floor",
            "intensity": "0.3 stops below ambient",
            "effect": "Illuminates undersides of objects",
        },
        "wall_bounce": {
            "direction": "Colored reflection from environment walls",
            "intensity": "0.2-0.5 stops below ambient",
            "effect": "Adds color tint from wall surfaces",
        },
        "product_bounce": {
            "direction": "Inter-reflections between product components",
            "intensity": "Subtle, localized",
            "effect": "Enhances form and dimensionality",
        },
    },
    "ambient_occlusion": {
        "description": "Shadow accumulation in crevices and contact points",
        "contact_areas": {
            "darkening": "2 stops darker than ambient",
            "edges": "Dark, defined edges where product touches",
        },
        "crevice_shadows": {
            "darkening": "Darkest shadows in tight spaces",
            "edges": "Defined edges in concave areas",
        },
        "corner_darkening": {
            "description": "Natural shadow buildup in concave areas",
            "intensity": "Proportional to corner depth",
        },
    },
    "color_temperature": {
        "description": "Precise Kelvin temperature control for light sources",
        "key_light": {
            "temperature": "5600K (daylight balanced)",
            "purpose": "Primary illumination",
        },
        "fill_light": {
            "temperature": "6000K (slightly cooler)",
            "purpose": "Shadow fill with neutral balance",
        },
        "rim_light": {
            "temperature": "5000K (warmer for separation)",
            "purpose": "Edge separation from background",
        },
        "ambient": {
            "temperature": "4800K (warm overall base)",
            "purpose": "Base illumination level",
        },
    },
    "volumetric_lighting": {
        "description": "Atmospheric light scattering through particles",
        "dust_particles": {
            "visibility": "Subtle dust motes visible in light beams",
            "concentration": "5% density for depth perception",
        },
        "light_beams": {
            "visibility": "Faintly visible beams through dusty air",
            "source": "From key light direction",
        },
        "atmospheric_haze": {
            "density": "5% haze for depth",
            "backscatter": "Visible in dark areas from key light",
        },
    },
}


def get_lighting_specification(
    include_bounced: bool = True,
    include_ao: bool = True,
    include_color_temp: bool = True,
    include_volumetric: bool = False,
) -> str:
    """
    Get complete lighting specification for prompt.

    Args:
        include_bounced: Include bounced light descriptions
        include_ao: Include ambient occlusion descriptions
        include_color_temp: Include color temperature specifications
        include_volumetric: Include volumetric lighting effects

    Returns:
        Formatted lighting specification string

    Example:
        >>> spec = get_lighting_specification(
        ...     include_bounced=True,
        ...     include_ao=True,
        ...     include_color_temp=True
        ... )
    """
    specs = []

    # Bounced light
    if include_bounced:
        bounced = ADVANCED_LIGHTING["bounced_light"]
        for bounce_type, details in bounced.items():
            if bounce_type == "description":
                specs.append(f"[{details}]")
            else:
                specs.append(
                    f"[{bounce_type}: {details['direction']}, "
                    f"{details['intensity']}, Effect: {details['effect']}]"
                )

    # Ambient occlusion
    if include_ao:
        ao = ADVANCED_LIGHTING["ambient_occlusion"]
        specs.append(f"[{ao['description']}]")
        for occlusion_type, details in ao.items():
            if occlusion_type != "description":
                specs.append(f"[{occlusion_type}: {details}]")

    # Color temperature
    if include_color_temp:
        ct = ADVANCED_LIGHTING["color_temperature"]
        specs.append(f"[{ct['description']}]")
        for light_type, details in ct.items():
            if light_type != "description":
                specs.append(
                    f"[{light_type}: {details['temperature']}, {details['purpose']}]"
                )

    # Volumetric lighting
    if include_volumetric:
        vol = ADVANCED_LIGHTING["volumetric_lighting"]
        specs.append(f"[{vol['description']}]")
        for vol_type, details in vol.items():
            if vol_type != "description":
                specs.append(f"[{vol_type}: {details}]")

    return " ".join(specs)


# Three-point lighting enhancements
THREE_POINT_ENHANCED = {
    "key_light": {
        "position": "45° top-right",
        "intensity": "1.5 stops brighter than ambient",
        "quality": "Main directional light",
        "shadow": "Creates primary shadows",
    },
    "fill_light": {
        "position": "45° top-left",
        "intensity": "0.5 stops above ambient",
        "quality": "Softens key light shadows",
        "purpose": "Reduces shadow contrast",
    },
    "rim_light": {
        "position": "Back-left",
        "intensity": "1 stop above ambient",
        "quality": "Edge separation light",
        "purpose": "Separates product from background",
    },
    "product_separation": {
        "description": "Product illuminated brighter than background",
        "ratio": "1.5 stops brighter",
        "effect": "Clear visual separation",
    },
}


def get_three_point_lighting_enhanced() -> str:
    """
    Get enhanced three-point lighting specification.

    Returns:
        Formatted three-point lighting description
    """
    lighting_desc = []

    for light_name, details in THREE_POINT_ENHANCED.items():
        if light_name == "product_separation":
            lighting_desc.append(
                f"[{details['description']}: {details['ratio']}, {details['effect']}]"
            )
        else:
            lighting_desc.append(
                f"[{light_name.replace('_', ' ')}: "
                f"{details['position']}, {details['intensity']}, "
                f"{details['quality']}, {details.get('shadow', details.get('purpose', ''))}]"
            )

    return " ".join(lighting_desc)


# Lighting quality modifiers
LIGHTING_QUALITY = {
    "soft": "Diffuse, gentle shadows, wraparound light",
    "hard": "Sharp, defined shadows with crisp edges",
    "diffuse": "Scattered light, soft shadows",
    "specular": "Sharp highlights, strong reflections",
    "dramatic": "High contrast, strong shadow definition",
    "flat": "Even illumination, minimal shadows",
}


def get_lighting_quality_modifier(quality: str) -> str:
    """
    Get lighting quality description.

    Args:
        quality: Type of lighting quality (soft, hard, diffuse, specular, dramatic, flat)

    Returns:
        Lighting quality description
    """
    if quality in LIGHTING_QUALITY:
        return f"[Lighting quality: {LIGHTING_QUALITY[quality]}]"
    return ""


# Shadow specifications
SHADOW_SPECIFICATIONS = {
    "contact_shadows": {
        "hardness": "Hard, dark shadows",
        "rgb": "RGB(20,20,20) - near black",
        "edges": "Defined, sharp edges where product touches",
        "purpose": "Defines grounding and contact",
    },
    "cast_shadows": {
        "hardness": "Soft shadows",
        "rgb": "RGB(80,80,80) - medium gray",
        "direction": "Direction from key light (45°)",
        "falloff": "Natural exponential decay, no hard cutoffs",
    },
    "shadow_depth": {
        "contact": "2 stops darker than ambient",
        "cast": "1 stop darker than ambient",
        "terminator": "Gradual transition between light and shadow",
    },
}


def get_shadow_specification() -> str:
    """
    Get complete shadow specification.

    Returns:
        Formatted shadow specification description
    """
    shadow_desc = []

    for shadow_type, details in SHADOW_SPECIFICATIONS.items():
        if isinstance(details, dict):
            for key, value in details.items():
                shadow_desc.append(f"[{shadow_type}_{key}: {value}]")
        else:
            shadow_desc.append(f"[{shadow_type}: {details}]")

    return " ".join(shadow_desc)


# Complete lighting system
def get_complete_lighting_system(
    lighting_type: str = "three_point",
    include_advanced: bool = True,
    include_volumetric: bool = False,
) -> str:
    """
    Get complete lighting system description.

    Args:
        lighting_type: Type of lighting (three_point, natural, studio)
        include_advanced: Include advanced optics (bounced, AO, color temp)
        include_volumetric: Include volumetric effects

    Returns:
        Complete formatted lighting description
    """
    components = []

    # Base three-point lighting
    if lighting_type == "three_point":
        components.append(get_three_point_lighting_enhanced())

    # Shadow specs
    components.append(get_shadow_specification())

    # Advanced optics
    if include_advanced:
        components.append(
            get_lighting_specification(
                include_bounced=True,
                include_ao=True,
                include_color_temp=True,
                include_volumetric=include_volumetric,
            )
        )

    return " ".join(components)


def get_scene_lighting_for_template(
    scene_type: str = "studio",
    time_of_day: str = "day",
) -> str:
    """
    Get scene-appropriate lighting description.

    Args:
        scene_type: Type of scene (kitchen, living_room, office, outdoor)
        time_of_day: Time of day (day, evening, night)

    Returns:
        Scene-specific lighting description
    """
    lighting_presets = {
        "kitchen_day": (
            "[Natural window light (5000K) + overhead ambient (3500K)]. "
            "[Window light: Directional soft light from left]. "
            "[Countertop reflection: Product reflects in stainless steel surfaces]"
        ),
        "living_room_day": (
            "[Soft directional window light with shadows]. "
            "[Ambient room light (3500K)]. "
            "[Natural light filtering through curtains]"
        ),
        "office_day": (
            "[Cool daylight (5500K) from ceiling fixtures]. "
            "[Monitor glow: Subtle blue tint]. "
            "[Fluorescent ambient: Even illumination]"
        ),
        "outdoor_day": (
            "[Sunlight + skylight mixture (6000K)]. "
            "[Direct sunlight: Harsh shadows]. "
            "[Open sky: Soft fill light]"
        ),
        "outdoor_golden_hour": (
            "[Warm golden light (3500K)]. "
            "[Long shadows from low sun angle]. "
            "[Orange/yellow color cast]"
        ),
    }

    preset_key = f"{scene_type}_{time_of_day}"
    if preset_key in lighting_presets:
        return lighting_presets[preset_key]

    # Default fallback
    return get_complete_lighting_system(
        lighting_type="three_point",
        include_advanced=True,
    )
