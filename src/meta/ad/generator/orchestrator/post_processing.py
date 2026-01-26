"""
Advanced Post-Processing Module.

Provides professional retouching workflow specifications including
color grading, sharpening, dodge and burn, and noise reduction.

Based on professional commercial photography post-processing standards.
"""

from typing import Dict, List, Optional


# Color grading specifications
COLOR_GRADING = {
    "s_curve": {
        "description": "Contrast enhancement using S-curve",
        "shadows": "-10 (lift darks slightly)",
        "midtones": "0 (preserve skin tones)",
        "highlights": "+15 (boost brights for punch)",
        "purpose": "Enhanced contrast while preserving midtones",
    },
    "selective_saturation": {
        "description": "Independent saturation control by area",
        "product": "+5 saturation (color pop)",
        "background": "-5 saturation (don't distract)",
        "skin_tones": "0 change (natural look)",
        "purpose": "Focus attention on product",
    },
    "color_balance": {
        "description": "Color balance adjustments by tonal range",
        "shadows": "Cool bias (+5 blue, -5 red)",
        "midtones": "Neutral (0,0,0)",
        "highlights": "Warm bias (+5 yellow, -5 blue)",
        "purpose": "Add depth and dimensionality",
    },
    "vibrance": {
        "description": "Smart saturation boost",
        "effect": "Boosts muted colors more than saturated ones",
        "intensity": "Subtle boost (+10-15)",
    },
}


def get_color_grading_spec(
    include_s_curve: bool = True,
    include_selective: bool = True,
    include_balance: bool = True,
    include_vibrance: bool = True,
) -> str:
    """
    Get color grading specification.

    Args:
        include_s_curve: Include S-curve contrast
        include_selective: Include selective saturation
        include_balance: Include color balance
        include_vibrance: Include vibrance

    Returns:
        Formatted color grading specification
    """
    grading = []

    if include_s_curve:
        s_curve = COLOR_GRADING["s_curve"]
        grading.append(f"[S-Curve: {s_curve['description']}]")
        grading.append(f"[Shadows {s_curve['shadows']}, Midtones {s_curve['midtones']}, Highlights {s_curve['highlights']}]")
        grading.append(f"[Purpose: {s_curve['purpose']}]")

    if include_selective:
        selective = COLOR_GRADING["selective_saturation"]
        grading.append(f"[Selective Saturation: {selective['description']}]")
        grading.append(f"[Product {selective['product']}, Background {selective['background']}, Skin tones {selective['skin_tones']}]")
        grading.append(f"[Purpose: {selective['purpose']}]")

    if include_balance:
        balance = COLOR_GRADING["color_balance"]
        grading.append(f"[Color Balance: {balance['description']}]")
        grading.append(f"[Shadows {balance['shadows']}, Midtones {balance['midtones']}, Highlights {balance['highlights']}]")
        grading.append(f"[Purpose: {balance['purpose']}]")

    if include_vibrance:
        vibrance = COLOR_GRADING["vibrance"]
        grading.append(f"[Vibrance: {vibrance['description']}, {vibrance['effect']}, {vibrance['intensity']}]")

    return " ".join(grading)


# Sharpening specifications
SHARPENING = {
    "high_pass_sharpen": {
        "radius": "2.0px",
        "amount": "75%",
        "blend_mode": "Overlay",
        "purpose": "Edge enhancement without halos",
    },
    "output_sharpening_web": {
        "radius": "0.6px",
        "amount": "150%",
        "threshold": "0",
        "purpose": "Web/screen output sharpening",
    },
    "output_sharpening_print": {
        "radius": "0.3px",
        "amount": "100%",
        "threshold": "0",
        "purpose": "Print output sharpening",
    },
    "unsharp_mask": {
        "description": "Traditional unsharp mask sharpening",
        "radius": "1.0px",
        "amount": "150%",
        "threshold": "0",
        "purpose": "Standard sharpening for general use",
    },
}


def get_sharpening_spec(output_type: str = "web") -> str:
    """
    Get sharpening specification.

    Args:
        output_type: Type of output (web, print, general)

    Returns:
        Formatted sharpening specification
    """
    sharpening = []

    # High pass sharpen (base layer)
    hp = SHARPENING["high_pass_sharpen"]
    sharpening.append(f"[High Pass Sharpen: Radius {hp['radius']}, Amount {hp['amount']}, Blend {hp['blend_mode']}]")
    sharpening.append(f"[Purpose: {hp['purpose']}]")

    # Output-specific sharpening
    if output_type == "web":
        output_spec = SHARPENING["output_sharpening_web"]
    elif output_type == "print":
        output_spec = SHARPENING["output_sharpening_print"]
    else:
        output_spec = SHARPENING["unsharp_mask"]

    sharpening.append(f"[Output Sharpening ({output_type}): Radius {output_spec['radius']}, Amount {output_spec['amount']}]")
    if "threshold" in output_spec:
        sharpening.append(f"[Threshold: {output_spec['threshold']}]")
    sharpening.append(f"[Purpose: {output_spec['purpose']}]")

    return " ".join(sharpening)


# Dodge and burn specifications
DODGE_AND_BURN = {
    "dodge": {
        "description": "Brighten shadows",
        "exposure": "+10",
        "range": "Midtones",
        "purpose": "Open up shadows for detail",
    },
    "burn": {
        "description": "Darken highlights",
        "exposure": "-10",
        "range": "Highlights",
        "purpose": "Recover highlight details",
    },
    "technique": {
        "description": "Selective exposure adjustment",
        "method": "Brush with gray (50% opacity) at 10-20% flow",
        "purpose": "Enhance local contrast, add dimension",
    },
}


def get_dodge_and_burn_spec() -> str:
    """
    Get dodge and burn specification.

    Returns:
        Formatted dodge and burn specification
    """
    dnb = []

    dodge = DODGE_AND_BURN["dodge"]
    dnb.append(f"[Dodge: {dodge['description']}, Exposure {dodge['exposure']}, Range {dodge['range']}]")
    dnb.append(f"[Purpose: {dodge['purpose']}]")

    burn = DODGE_AND_BURN["burn"]
    dnb.append(f"[Burn: {burn['description']}, Exposure {burn['exposure']}, Range {burn['range']}]")
    dnb.append(f"[Purpose: {burn['purpose']}]")

    technique = DODGE_AND_BURN["technique"]
    dnb.append(f"[Technique: {technique['description']}, {technique['method']}]")
    dnb.append(f"[Purpose: {technique['purpose']}]")

    return " ".join(dnb)


# Noise reduction specifications
NOISE_REDUCTION = {
    "luminance_noise": {
        "description": "Reduce brightness variation noise",
        "reduction": "50%",
        "detail_preservation": "Preserve edges while smoothing",
    },
    "color_noise": {
        "description": "Reduce color speckles",
        "reduction": "80%",
        "purpose": "Clean color without artifacts",
    },
    "method": {
        "description": "Edge-preserving noise reduction",
        "technique": "Bilateral filtering or AI-based denoising",
        "balance": "Maintain sharp detail while reducing noise",
    },
}


def get_noise_reduction_spec(
    reduce_luminance: bool = True,
    reduce_color: bool = True,
) -> str:
    """
    Get noise reduction specification.

    Args:
        reduce_luminance: Reduce luminance/brightness noise
        reduce_color: Reduce color noise

    Returns:
        Formatted noise reduction specification
    """
    nr = []

    method = NOISE_REDUCTION["method"]
    nr.append(f"[Noise Reduction: {method['description']}, {method['technique']}]")
    nr.append(f"[Balance: {method['balance']}]")

    if reduce_luminance:
        lum = NOISE_REDUCTION["luminance_noise"]
        nr.append(f"[Luminance Noise: {lum['description']}, {lum['reduction']}, {lum['detail_preservation']}]")

    if reduce_color:
        color = NOISE_REDUCTION["color_noise"]
        nr.append(f"[Color Noise: {color['description']}, {color['reduction']}, {color['purpose']}]")

    return " ".join(nr)


# Complete post-processing workflow
def get_complete_post_processing_workflow(
    output_type: str = "web",
    style: str = "commercial",
) -> str:
    """
    Get complete post-processing workflow specification.

    Args:
        output_type: Type of output (web, print)
        style: Style of post-processing (commercial, editorial, natural)

    Returns:
        Complete formatted post-processing workflow
    """
    components = []

    # Color grading
    components.append("[Color Grading]")
    components.append(get_color_grading_spec())

    # Sharpening
    components.append("[Sharpening]")
    components.append(get_sharpening_spec(output_type))

    # Dodge and burn
    if style == "commercial":
        components.append("[Dodge and Burn]")
        components.append(get_dodge_and_burn_spec())

    # Noise reduction
    components.append("[Noise Reduction]")
    components.append(get_noise_reduction_spec())

    return " ".join(components)


# Post-processing styles
POST_PROCESSING_STYLES = {
    "commercial": {
        "description": "High-gloss commercial finish",
        "characteristics": [
            "Punchy contrast (S-curve)",
            "Vibrant colors (selective saturation)",
            "Crisp details (aggressive sharpening)",
            "Clean (noise reduction)",
        ],
        "use_case": "Product advertising, e-commerce",
    },
    "editorial": {
        "description": "Editorial magazine style",
        "characteristics": [
            "Subtle contrast (gentle S-curve)",
            "Natural colors (minimal saturation boost)",
            "Fine details (moderate sharpening)",
            "Film-like grain (optional)",
        ],
        "use_case": "Editorial, lifestyle photography",
    },
    "natural": {
        "description": "Natural, minimal processing",
        "characteristics": [
            "Low contrast (preserve dynamic range)",
            "Accurate colors (no saturation boost)",
            "Subtle sharpening",
            "No noise reduction (preserve texture)",
        ],
        "use_case": "Realistic product representation",
    },
    "hdr": {
        "description": "High dynamic range look",
        "characteristics": [
            "Expanded dynamic range",
            "Crushed blacks slightly",
            "Boosted peak whites",
            "Vivid colors",
        ],
        "use_case": "Dramatic product showcase",
    },
}


def get_post_processing_style(style: str = "commercial") -> str:
    """
    Get post-processing style description.

    Args:
        style: Style name (commercial, editorial, natural, hdr)

    Returns:
        Formatted style description
    """
    if style not in POST_PROCESSING_STYLES:
        return ""

    style_info = POST_PROCESSING_STYLES[style]
    components = []

    components.append(f"[Style: {style_info['description']}]")
    components.append(f"[Characteristics: {', '.join(style_info['characteristics'])}]")
    components.append(f"[Use Case: {style_info['use_case']}]")

    return " ".join(components)


# Retouching workflow stages
RETOUCHING_WORKFLOW = {
    "stage_1_corrections": {
        "name": "Global Corrections",
        "steps": [
            "Exposure adjustment",
            "White balance correction",
            "Lens profile correction",
            "Chromatic aberration removal",
        ],
    },
    "stage_2_enhancement": {
        "name": "Selective Enhancement",
        "steps": [
            "Color grading (S-curve, selective saturation)",
            "Dodge and burn (local contrast)",
            "Sharpening (high pass + output sharpen)",
        ],
    },
    "stage_3_polish": {
        "name": "Final Polish",
        "steps": [
            "Noise reduction",
            "Spot retouching (dust, scratches)",
            "Final color adjustments",
            "Output sharpening",
        ],
    },
}


def get_retouching_workflow_description(
    include_stages: List[int] = None,
) -> str:
    """
    Get retouching workflow description.

    Args:
        include_stages: List of stage numbers to include (1, 2, 3)

    Returns:
        Formatted workflow description
    """
    if include_stages is None:
        include_stages = [1, 2, 3]

    workflow = []

    for stage_num in include_stages:
        stage_key = f"stage_{stage_num}_corrections"
        if stage_key in RETOUCHING_WORKFLOW:
            stage = RETOUCHING_WORKFLOW[stage_key]
            workflow.append(f"[Stage {stage_num}: {stage['name']}]")
            workflow.append(f"[Steps: {', '.join(stage['steps'])}]")

    return " ".join(workflow)
