"""
Photographic Chemistry Module.

Provides film emulation characteristics, grain structures, and photographic
chemical effects for authentic film-like appearance in generated images.

Based on real photographic film science and characteristics.
"""

from typing import Dict, List, Optional


# Film emulation profiles
FILM_EMULATION = {
    "kodak_portra_400": {
        "description": "Professional color negative film",
        "grain_size": "Fine, even distribution",
        "grain_characteristics": "Cubic grain structure, subtle in highlights",
        "color_response": "Neutral color balance with subtle warmth",
        "exposure_latitude": "High latitude - forgiving exposure",
        "best_for": "Portraits, lifestyle, natural skin tones",
        "contrast": "Medium contrast, gentle highlight rolloff",
    },
    "kodak_portra_800": {
        "description": "High-speed color negative film",
        "grain_size": "Moderate grain (more than 400)",
        "grain_characteristics": "Visible grain in shadows and midtones",
        "color_response": "Rich saturation, warm bias",
        "exposure_latitude": "Very high latitude",
        "best_for": "Low light, mood, atmosphere",
        "contrast": "Medium contrast",
    },
    "fuji_pro400h": {
        "description": "Professional color negative film",
        "grain_size": "Very fine, 4-layer sensitization",
        "grain_characteristics": "Smooth tonal transitions",
        "color_response": "Neutral balance, subtle warmth",
        "exposure_latitude": "Good latitude",
        "best_for": "General purpose, accurate colors",
        "contrast": "Medium contrast",
    },
    "fuji_velvia_50": {
        "description": "Vivid slide film",
        "grain_size": "Extremely fine",
        "grain_characteristics": "Barely visible grain",
        "color_response": "Highly saturated, vivid colors",
        "exposure_latitude": "Low latitude - requires precise exposure",
        "best_for": "Landscapes, products requiring color pop",
        "contrast": "High contrast, deep blacks",
    },
    "kodak_ektar_100": {
        "description": "Slide film with cool color balance",
        "grain_size": "Very fine",
        "grain_characteristics": "Minimal visible grain",
        "color_response": "Cool, blue bias, vibrant colors",
        "exposure_latitude": "Low latitude",
        "best_for": "Products with cool tones, technical subjects",
        "contrast": "High contrast",
    },
    "kodak_gold_200": {
        "description": "Consumer color negative film",
        "grain_size": "Moderate to large grain",
        "grain_characteristics": "Visible grain in all tonal ranges",
        "color_response": "Warm, nostalgic color balance",
        "exposure_latitude": "Very high latitude",
        "best_for": "Vintage look, nostalgia, warm atmosphere",
        "contrast": "Low to medium contrast",
    },
    "cinestill_800t": {
        "description": "Tungsten-balanced motion picture film",
        "grain_size": "Visible grain structure",
        "grain_characteristics": "Distinctive tri-colored grain (blue, green, red)",
        "color_response": "Warm, cinematic color balance",
        "exposure_latitude": "High latitude",
        "best_for": "Cinematic look, motion picture aesthetic",
        "contrast": "Medium contrast with soft highlights",
    },
    "black_and_white": {
        "description": "Classic black and white film",
        "grain_size": "Variable (based on ISO)",
        "grain_characteristics": "Traditional silver halide grain",
        "color_response": "Monochrome - no color information",
        "exposure_latitude": "High latitude",
        "best_for": "Timeless look, emphasis on form and texture",
        "contrast": "High contrast option available",
    },
}


def get_film_emulation_spec(
    film_stock: str = "kodak_portra_400",
    include_grain: bool = True,
    include_color: bool = True,
) -> str:
    """
    Get film emulation specification.

    Args:
        film_stock: Film stock name
        include_grain: Include grain characteristics
        include_color: Include color response information

    Returns:
        Formatted film emulation specification
    """
    film_key = film_stock.lower().replace(" ", "_")

    if film_key not in FILM_EMULATION:
        return f"[Film stock {film_stock} not found]"

    film = FILM_EMULATION[film_key]
    components = []

    components.append(f"[Film Stock: {film['description']}]")

    if include_grain:
        components.append(f"[Grain: {film['grain_size']}]")
        components.append(f"[Characteristics: {film['grain_characteristics']}]")

    if include_color and film_stock != "black_and_white":
        components.append(f"[Color Response: {film['color_response']}]")

    components.append(f"[Latitude: {film['exposure_latitude']}]")
    components.append(f"[Best For: {film['best_for']}]")
    components.append(f"[Contrast: {film['contrast']}]")

    return " ".join(components)


# Film grain characteristics
FILM_GRAIN = {
    "iso_dependent": {
        "description": "Grain increases with ISO setting",
        "iso_100": "Very fine grain, barely visible",
        "iso_200": "Fine grain, visible upon close inspection",
        "iso_400": "Moderate grain, subtly visible",
        "iso_800": "Visible grain throughout image",
        "iso_1600": "Heavy grain, dominant texture",
        "iso_3200": "Very heavy grain, strong texture",
    },
    "tonal_dependence": {
        "description": "Grain varies by tonal brightness",
        "highlights": "Grain barely visible in bright areas",
        "midtones": "Moderate grain visibility in middle tones",
        "shadows": "Most visible grain in dark areas",
    },
    "color_sensitive": {
        "description": "Different color layers have different grain",
        "blue_layer": "Finest grain (blue-sensitive layer)",
        "green_layer": "Medium grain (green-sensitive layer)",
        "red_layer": "Largest grain (red-sensitive layer)",
        "effect": "Subtle color variations due to grain differences",
    },
}


def get_grain_specification(
    iso: int = 400,
    film_type: str = "color_negative",
) -> str:
    """
    Get grain specification.

    Args:
        iso: ISO setting
        film_type: Type of film (color_negative, slide, black_and_white)

    Returns:
        Formatted grain specification
    """
    grain = []

    # ISO-dependent grain
    iso_range = "iso_100"
    if iso <= 200:
        iso_range = "iso_100"
    elif iso <= 400:
        iso_range = "iso_400"
    elif iso <= 800:
        iso_range = "iso_800"
    elif iso <= 1600:
        iso_range = "iso_1600"
    else:
        iso_range = "iso_3200"

    iso_info = FILM_GRAIN["iso_dependent"][iso_range]
    grain.append(f"[Grain at ISO {iso}: {iso_info}]")

    # Tonal dependence
    tonal = FILM_GRAIN["tonal_dependence"]
    grain.append(f"[{tonal['description']}]")
    grain.append(f"[Highlights: {tonal['highlights']}]")
    grain.append(f"[Midtones: {tonal['midtones']}]")
    grain.append(f"[Shadows: {tonal['shadows']}]")

    # Color-sensitive (only for color film)
    if film_type != "black_and_white":
        color = FILM_GRAIN["color_sensitive"]
        grain.append(f"[{color['description']}]")
        grain.append(f"[Effect: {color['effect']}]")

    return " ".join(grain)


# Halation effects
HALATION = {
    "description": "Red light bleed in bright areas from film exposure",
    "cause": "Light scattering in film base during exposure",
    "appearance": "Subtle red glow around specular highlights",
    "intensity": "5% opacity, 2px feather",
    "location": "Around bright highlights, specular reflections",
    "aesthetic": "Adds warmth and vintage character",
}


def get_halation_spec() -> str:
    """
    Get halation effect specification.

    Returns:
        Formatted halation specification
    """
    hal = []

    for key, value in HALATION.items():
        if key != "description":
            hal.append(f"[{key.replace('_', ' ').title()}: {value}]")
        else:
            hal.append(f"[Halation: {value}]")

    return " ".join(hal)


# Latitude characteristics
LATITUDE = {
    "highlight_recovery": {
        "description": "Detail retained in bright areas",
        "range": "3 stops overexposed still shows detail",
        "technique": "Shoulder of exposure curve",
    },
    "shadow_detail": {
        "description": "Detail visible in dark areas",
        "range": "4 stops underexposed still shows detail",
        "technique": "Toe of exposure curve with gradual falloff",
    },
    "dynamic_range": {
        "description": "Total usable range from darkest to lightest",
        "range": "12 stops printable range for most color negative films",
        "comparison": "Slide film: ~8 stops, Digital: ~14 stops",
    },
    "practical_benefit": {
        "description": "Forgiving exposure errors",
        "benefit": "Can recover detail from minor exposure mistakes",
        "use_case": "Valuable for challenging lighting conditions",
    },
}


def get_latitude_spec() -> str:
    """
    Get latitude specification.

    Returns:
        Formatted latitude specification
    """
    lat = []

    for category, details in LATITUDE.items():
        lat.append(f"[{category.replace('_', ' ').title()}: {details['description']}]")
        if "range" in details:
            lat.append(f"[Range: {details['range']}]")
        if "technique" in details:
            lat.append(f"[Technique: {details['technique']}]")
        if "comparison" in details:
            lat.append(f"[Comparison: {details['comparison']}]")
        if "benefit" in details:
            lat.append(f"[Benefit: {details['benefit']}]")
        if "use_case" in details:
            lat.append(f"[Use Case: {details['use_case']}]")

    return " ".join(lat)


# Complete photographic chemistry
def get_photographic_chemistry_spec(
    film_stock: str = "kodak_portra_400",
    iso: int = 400,
    include_halation: bool = False,
) -> str:
    """
    Get complete photographic chemistry specification.

    Args:
        film_stock: Film stock name
        iso: ISO setting
        include_halation: Include halation effects

    Returns:
        Complete formatted photographic chemistry specification
    """
    components = []

    # Film emulation
    components.append(get_film_emulation_spec(film_stock))

    # Grain
    film_type = "black_and_white" if "bw" in film_stock.lower() else "color_negative"
    components.append(get_grain_specification(iso, film_type))

    # Latitude
    components.append(get_latitude_spec())

    # Halation (optional)
    if include_halation and "slide" not in film_stock.lower():
        components.append(get_halation_spec())

    return " ".join(components)


# Aesthetics by use case
PHOTOGRAPHIC_AESTHETICS = {
    "vintage_nostalgia": {
        "film_stock": "kodak_gold_200",
        "characteristics": [
            "Warm, nostalgic colors",
            "Moderate visible grain",
            "Soft contrast",
            "Memory lane aesthetic",
        ],
    },
    "professional_clean": {
        "film_stock": "fuji_pro400h",
        "characteristics": [
            "Neutral, accurate colors",
            "Very fine grain",
            "Natural contrast",
            "Professional appearance",
        ],
    },
    "cinematic": {
        "film_stock": "cinestill_800t",
        "characteristics": [
            "Warm, cinematic color",
            "Visible tri-color grain",
            "Soft highlights",
            "Motion picture aesthetic",
        ],
    },
    "vivid_punchy": {
        "film_stock": "fuji_velvia_50",
        "characteristics": [
            "Highly saturated colors",
            "High contrast",
            "Deep blacks",
            "Color pop",
        ],
    },
    "timeless_classic": {
        "film_stock": "kodak_portra_400",
        "characteristics": [
            "Natural skin tones",
            "Fine grain",
            "Medium contrast",
            "Versatile aesthetic",
        ],
    },
}


def get_aesthetic_by_use_case(use_case: str) -> str:
    """
    Get photographic aesthetic by use case.

    Args:
        use_case: Type of aesthetic (vintage_nostalgia, professional_clean, etc.)

    Returns:
        Formatted aesthetic specification
    """
    use_case_key = use_case.lower().replace(" ", "_")

    if use_case_key not in PHOTOGRAPHIC_AESTHETICS:
        return f"[Use case '{use_case}' not found]"

    aesthetic = PHOTOGRAPHIC_AESTHETICS[use_case_key]
    components = []

    components.append(f"[Aesthetic: {use_case.replace('_', ' ').title()}]")
    components.append(f"[Film Stock: {aesthetic['film_stock']}]")
    components.append(f"[Characteristics: {', '.join(aesthetic['characteristics'])}]")

    return " ".join(components)
