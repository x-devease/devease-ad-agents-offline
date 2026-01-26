"""
Camera Physics Module.

Provides precise camera and lens specifications based on professional photography standards.
Includes optical characteristics, depth of field calculations, and lens-specific effects.

Based on real camera physics and optical principles.
"""

from typing import Dict, List, Optional


# Lens characteristics by model
LENS_CHARACTERISTICS = {
    "canon_85mm_f1.4": {
        "focal_length": "85mm",
        "max_aperture": "f/1.4",
        "bokeh_quality": "Circular, 9-blade aperture polygonal shapes",
        "vignetting": "Subtle darkening at corners (-0.5 EV)",
        "chromatic_aberration": "Minimal LCA, slight longitudinal CA at f/1.4",
        "distortion": "<1% barrel distortion (near-rectilinear)",
        "field_curvature": "Flat field across sensor",
        "best_for": "Portraits, product detail, shallow DOF",
    },
    "canon_50mm_f1.8": {
        "focal_length": "50mm",
        "max_aperture": "f/1.8",
        "bokeh_quality": "Decent circular bokeh, 7-blade",
        "vignetting": "Noticeable at wide apertures (-1.2 EV at f/1.8)",
        "chromatic_aberration": "Moderate LCA visible at high contrast edges",
        "distortion": "Minimal, <0.5% barrel",
        "field_curvature": "Slight field curvature",
        "best_for": "General purpose, normal perspective",
    },
    "canon_100mm_f2.8_macro": {
        "focal_length": "100mm",
        "max_aperture": "f/2.8",
        "magnification": "1:1 (life-size) macro capability",
        "working_distance": "15cm minimum distance",
        "focus_stacking": "Multiple focal planes merged for deep sharpness",
        "depth_of_field": "Extremely shallow at macro distances",
        "best_for": "Extreme close-ups, technical product details",
    },
    "canon_24-70mm_f2.8": {
        "focal_length": "24-70mm zoom",
        "max_aperture": "f/2.8 constant",
        "versatility": "Wide to short telephoto range",
        "distortion": "Barrel at 24mm, minimal at 50mm, pincushion at 70mm",
        "best_for": "Flexible shooting, varying compositions",
    },
}


def get_lens_characteristics(
    lens_model: str = "canon_85mm_f1.4",
) -> str:
    """
    Get lens characteristics description.

    Args:
        lens_model: Lens model identifier

    Returns:
        Formatted lens characteristics description
    """
    if lens_model not in LENS_CHARACTERISTICS:
        return ""

    lens = LENS_CHARACTERISTICS[lens_model]
    components = []

    components.append(f"[Lens: {lens['focal_length']} f/{lens['max_aperture'].replace('f/', '')}]")
    components.append(f"[Bokeh: {lens['bokeh_quality']}]")
    components.append(f"[Vignetting: {lens['vignetting']}]")
    components.append(f"[Chromatic aberration: {lens['chromatic_aberration']}]")
    components.append(f"[Distortion: {lens['distortion']}]")
    components.append(f"[Field curvature: {lens['field_curvature']}]")

    return " ".join(components)


# Camera body specifications
CAMERA_SPECIFICATIONS = {
    "canon_eos_r5": {
        "sensor_type": "Full-frame CMOS",
        "sensor_size": "36x24mm (135 format)",
        "resolution": "45MP",
        "bit_depth": "14-bit RAW",
        "dynamic_range": "12 stops usable",
        "iso_range": "100-51200",
        "best_iso": "ISO 100 for maximum quality",
    },
    "canon_eos_r3": {
        "sensor_type": "Full-frame stacked CMOS",
        "sensor_size": "36x24mm",
        "resolution": "24MP",
        "bit_depth": "14-bit RAW",
        "dynamic_range": "11 stops usable",
        "iso_range": "100-102400",
        "best_iso": "ISO 100-800 for optimal quality",
    },
}


def get_camera_specification(
    camera_model: str = "canon_eos_r5",
    lens_model: str = "canon_85mm_f1.4",
    iso: int = 100,
    aperture: str = "f/8",
) -> str:
    """
    Get complete camera specification.

    Args:
        camera_model: Camera body model
        lens_model: Lens model
        iso: ISO setting
        aperture: Aperture setting

    Returns:
        Formatted camera specification description
    """
    if camera_model not in CAMERA_SPECIFICATIONS:
        return ""

    camera = CAMERA_SPECIFICATIONS[camera_model]
    components = []

    components.append(f"[Camera: {camera_model.replace('_', ' ').title()}]")
    components.append(f"[Sensor: {camera['sensor_type']}, {camera['sensor_size']}]")
    components.append(f"[Resolution: {camera['resolution']}]")
    components.append(f"[Bit depth: {camera['bit_depth']}]")
    components.append(f"[Dynamic range: {camera['dynamic_range']}]")
    components.append(f"[ISO: {iso} (optimal quality)]")
    components.append(f"[Aperture: {aperture} for deep focus]")

    # Add lens specs
    lens_specs = get_lens_characteristics(lens_model)
    if lens_specs:
        components.append(lens_specs)

    return " ".join(components)


# Depth of field calculations
DEPTH_OF_FIELD = {
    "foreground": {
        "description": "Product in perfect focus",
        "aperture": "f/8",
        "sharpness": "100% sharp",
        "purpose": "Primary subject",
    },
    "midground": {
        "description": "Supporting elements",
        "aperture": "f/4",
        "sharpness": "70% sharp",
        "purpose": "Secondary elements",
    },
    "background": {
        "description": "Environment",
        "aperture": "f/2.8",
        "sharpness": "30% sharp, soft bokeh",
        "purpose": "Context without distraction",
    },
    "layering": {
        "description": "Lens-compressed background perspective",
        "effect": "Product in perfect focus, background softly blurred for depth separation",
    },
}


def get_depth_of_field_specification() -> str:
    """
    Get depth of field layering specification.

    Returns:
        Formatted DOF specification
    """
    dof_desc = []

    for layer, details in DEPTH_OF_FIELD.items():
        if layer == "layering":
            dof_desc.append(f"[Layering: {details['description']}, {details['effect']}]")
        else:
            dof_desc.append(
                f"[{layer.capitalize()}: {details['description']}, "
                f"f/{details['aperture']}, {details['sharpness']}, Purpose: {details['purpose']}]"
            )

    return " ".join(dof_desc)


# Photographic physics calculations
PHOTOGRAPHIC_PHYSICS = {
    "motion_blur": {
        "freeze_motion": {
            "shutter_speed": "1/250s or faster",
            "purpose": "Freeze fast action",
        },
        "motion_blur_intentional": {
            "shutter_speed": "1/60s or slower",
            "purpose": "Show motion direction",
        },
    },
    "diffraction_limit": {
        "description": "Optimal aperture before diffraction softness",
        "limit": "f/11 maximum before softness from diffraction",
        "optimal": "f/5.6 to f/8 for sharpness",
    },
    "circle_of_confusion": {
        "description": "Acceptable sharpness threshold",
        "value": "0.03mm for full-frame depth of field calculations",
    },
    "hyperfocal_distance": {
        "description": "Closest focus distance for infinity sharpness",
        "calculation": "Based on focal length and aperture",
        "use": "Maximize depth of field for landscape",
    },
}


def get_photographic_physics() -> str:
    """
    Get photographic physics descriptions.

    Returns:
        Formatted photographic physics description
    """
    physics_desc = []

    for category, details in PHOTOGRAPHIC_PHYSICS.items():
        if isinstance(details, dict):
            physics_desc.append(f"[{category.replace('_', ' ')}: {details['description']}]")
            if category == "motion_blur":
                for key, value in details.items():
                    if key != "description":
                        physics_desc.append(f"[{key}: {value['shutter_speed']}, {value['purpose']}]")
            elif category == "diffraction_limit":
                for key, value in details.items():
                    if key not in ["description", "optimal"]:
                        physics_desc.append(f"[{key}: {value}]")
                physics_desc.append(f"[Optimal: {details['optimal']}]")
            elif category == "circle_of_confusion":
                physics_desc.append(f"[Value: {details['value']}]")
            elif category == "hyperfocal_distance":
                physics_desc.append(f"[Use: {details['use']}]")

    return " ".join(physics_desc)


# Macro photography specifications
MACRO_SPECIFICATIONS = {
    "magnification_ratio": "1:2 (half life-size) for product details",
    "working_distance": "Minimum 15cm from front element",
    "focus_stacking": "Multiple focal planes merged for deep sharpness",
    "depth_of_field": "Extremely shallow at macro distances",
    "diffraction": "Becomes significant past f/11",
    "lighting": "Critical for macro - need even illumination",
}


def get_macro_specifications() -> str:
    """
    Get macro photography specifications.

    Returns:
        Formatted macro specification description
    """
    macro_desc = []

    for spec, value in MACRO_SPECIFICATIONS.items():
        macro_desc.append(f"[{spec.replace('_', ' ').title()}: {value}]")

    return " ".join(macro_desc)


# Complete camera system
def get_complete_camera_system(
    camera_model: str = "canon_eos_r5",
    lens_model: str = "canon_85mm_f1.4",
    iso: int = 100,
    aperture: str = "f/8",
    include_physics: bool = True,
    include_dof: bool = True,
    macro_mode: bool = False,
) -> str:
    """
    Get complete camera system description.

    Args:
        camera_model: Camera body model
        lens_model: Lens model
        iso: ISO setting
        aperture: Aperture setting
        include_physics: Include photographic physics
        include_dof: Include depth of field layering
        macro_mode: Use macro specifications

    Returns:
        Complete formatted camera system description
    """
    components = []

    # Camera and lens specs
    components.append(
        get_camera_specification(camera_model, lens_model, iso, aperture)
    )

    # Depth of field
    if include_dof:
        components.append(get_depth_of_field_specification())

    # Macro specs
    if macro_mode:
        components.append(get_macro_specifications())

    # Photographic physics
    if include_physics:
        components.append(get_photographic_physics())

    return " ".join(components)
