"""
Atmosphere V2 Module.

Advanced atmospheric effects for enhanced environmental context and mood.

Based on physical atmospheric scattering and optical phenomena.
"""

from typing import Dict, List, Optional


# Atmospheric scattering (Rayleigh and Mie)
ATMOSPHERIC_SCATTERING = {
    "rayleigh": {
        "description": "Molecular scattering, wavelength-dependent",
        "effect": "Blue sky, red sunset",
        "particle_size": "< 0.1 μm (molecules)",
        "wavelength_dependence": "Scatters short wavelengths (blue) more than long (red)",
        "applications": [
            "Clear blue sky",
            "Red/orange sunset",
            "Deep atmospheric haze",
        ],
    },
    "mie": {
        "description": "Particle scattering, less wavelength-dependent",
        "effect": "White/gray haze, fog, clouds",
        "particle_size": "0.1 - 10 μm (aerosols, droplets)",
        "wavelength_dependence": "Scatters all wavelengths similarly",
        "applications": [
            "Fog and mist",
            "Cloud formation",
            "Industrial haze",
        ],
    },
}


def get_scattering_spec(
    scattering_type: str = "rayleigh",
    intensity: float = 0.5,
) -> str:
    """
    Get atmospheric scattering specification.

    Args:
        scattering_type: Type of scattering (rayleigh, mie)
        intensity: Intensity of scattering effect (0.0 to 1.0)

    Returns:
        Formatted scattering specification
    """
    if scattering_type not in ATMOSPHERIC_SCATTERING:
        return ""

    scattering = ATMOSPHERIC_SCATTERING[scattering_type]
    parts = []

    parts.append(f"[Atmospheric Scattering: {scattering['description']}]")
    parts.append(f"[Effect: {scattering['effect']}]")
    parts.append(f"[Particle Size: {scattering['particle_size']}]")

    if scattering_type == "rayleigh":
        if intensity > 0.7:
            parts.append("[Deep blue sky, strong wavelength separation]")
        elif intensity > 0.4:
            parts.append("[Clear sky, moderate blue tint]")
        else:
            parts.append("[Subtle atmospheric blueing]")
    else:  # mie
        if intensity > 0.7:
            parts.append("[Dense fog/haze, reduced visibility]")
        elif intensity > 0.4:
            parts.append("[Moderate haze, soft scattering]")
        else:
            parts.append("[Light mist, subtle scattering]")

    return " ".join(parts)


# Volumetric lighting
VOLUMETRIC_LIGHTING = {
    "god_rays": {
        "description": "Sunbeams through atmosphere",
        "cause": "Light scattering by particles/air",
        "intensity": "5-15% opacity",
        "angle": "Pointing from light source",
        "best_for": "Dramatic hero shots, shafts of light",
    },
    "crepuscular_rays": {
        "description": "Twilight rays through clouds/terrain",
        "cause": "Shadowing by clouds/mountains",
        "intensity": "3-10% opacity",
        "angle": "Radiating from sun position",
        "best_for": "Sunrise/sunset scenes, dramatic mood",
    },
    "tyndall_effect": {
        "description": "Light beam visible in particles",
        "cause": "Dust, smoke, fog scattering light",
        "intensity": "10-20% opacity for dense medium",
        "angle": "Directional from source",
        "best_for": "Atmospheric product photography",
    },
}


def get_volumetric_lighting_spec(
    effect_type: str = "god_rays",
    intensity: float = 0.5,
) -> str:
    """
    Get volumetric lighting specification.

    Args:
        effect_type: Type of volumetric effect
        intensity: Intensity of effect (0.0 to 1.0)

    Returns:
        Formatted volumetric lighting specification
    """
    if effect_type not in VOLUMETRIC_LIGHTING:
        return ""

    effect = VOLUMETRIC_LIGHTING[effect_type]
    parts = []

    parts.append(f"[Volumetric Lighting: {effect['description']}]")
    parts.append(f"[Cause: {effect['cause']}]")

    opacity = int(effect["intensity"].split("%")[0].split("-")[0])
    adjusted_opacity = opacity * (0.5 + intensity / 2)
    parts.append(f"[Opacity: {adjusted_opacity:.0f}%]")

    parts.append(f"[Angle: {effect['angle']}]")
    parts.append(f"[Best For: {effect['best_for']}]")

    return " ".join(parts)


# Atmospheric perspective (depth haze)
ATMOSPHERIC_PERSPECTIVE = {
    "description": "Objects appear bluer/less contrast with distance",
    "cause": "Atmospheric scattering between viewer and object",
    "gradient": "Foreground (clear) → Midground (slight haze) → Background (blue/low contrast)",
    "color_shift": "Warm foreground → Cool background",
    "contrast_reduction": "Near (high contrast) → Far (low contrast)",
    "applications": [
        "Deep depth in wide scenes",
        "Environmental context",
        "Spatial separation",
    ],
}


def get_atmospheric_perspective_spec(
    depth: str = "deep",
) -> str:
    """
    Get atmospheric perspective specification.

    Args:
        depth: Depth of scene (deep, medium, shallow)

    Returns:
        Formatted atmospheric perspective specification
    """
    parts = []

    parts.append(f"[Atmospheric Perspective: {ATMOSPHERIC_PERSPECTIVE['description']}]")
    parts.append(f"[Cause: {ATMOSPHERIC_PERSPECTIVE['cause']}]")

    if depth == "deep":
        parts.append("[Foreground: Full clarity, high contrast]")
        parts.append("[Midground: Slight cool shift, 10% contrast reduction]")
        parts.append("[Background: Strong blue tint, 40% contrast reduction, desaturated]")
    elif depth == "medium":
        parts.append("[Foreground: Full clarity]")
        parts.append("[Midground: 20% cool shift, 15% contrast reduction]")
        parts.append("[Background: Moderate blue tint, 25% contrast reduction]")
    else:  # shallow
        parts.append("[Foreground: Full clarity]")
        parts.append("[Background: Subtle cool shift, 10% contrast reduction]")

    return " ".join(parts)


# Weather conditions
WEATHER_CONDITIONS = {
    "clear": {
        "description": "Clear sky, direct sunlight",
        "lighting": "Hard shadows, high contrast",
        "atmosphere": "Minimal scattering, crisp details",
        "sky": "Blue (day) or clear (night)",
        "best_for": "Product visibility, technical shots",
    },
    "partly_cloudy": {
        "description": "Mix of sun and clouds",
        "lighting": "Varied shadows, some diffuse light",
        "atmosphere": "Moderate scattering, soft details",
        "sky": "Blue with white clouds",
        "best_for": "Lifestyle context, natural feel",
    },
    "overcast": {
        "description": "Full cloud cover, no direct sun",
        "lighting": "Soft shadows, low contrast",
        "atmosphere": "High diffuse scattering, very soft",
        "sky": "Gray/white uniform",
        "best_for": "Product photography, even illumination",
    },
    "foggy": {
        "description": "Fog or mist present",
        "lighting": "Very diffuse, reduced visibility",
        "atmosphere": "High Mie scattering, depth haze",
        "sky": "White/gray, obscured",
        "best_for": "Mood, atmosphere, mystery",
    },
    "rainy": {
        "description": "Rain falling",
        "lighting": "Diffuse with specular highlights on wet surfaces",
        "atmosphere": "High scattering, wet surface reflections",
        "sky": "Gray, dark",
        "best_for": "Dramatic mood, weather-resistant products",
    },
    "snowy": {
        "description": "Snow present",
        "lighting": "Very diffuse, bright (high albedo)",
        "atmosphere": "High scattering, cool white balance",
        "sky": "Gray or bright white",
        "best_for": "Winter context, cold weather gear",
    },
}


def get_weather_spec(
    weather: str = "clear",
    intensity: float = 0.5,
) -> str:
    """
    Get weather condition specification.

    Args:
        weather: Weather condition
        intensity: Intensity of weather effect (0.0 to 1.0)

    Returns:
        Formatted weather specification
    """
    weather_key = weather.lower().replace(" ", "_")

    if weather_key not in WEATHER_CONDITIONS:
        return ""

    condition = WEATHER_CONDITIONS[weather_key]
    parts = []

    parts.append(f"[Weather: {weather.title()}]")
    parts.append(f"[Description: {condition['description']}]")
    parts.append(f"[Lighting: {condition['lighting']}]")

    # Add intensity-specific adjustments
    if intensity > 0.7:
        parts.append(f"[Intensity: Strong {weather} effect]")
    elif intensity > 0.4:
        parts.append(f"[Intensity: Moderate {weather} effect]")
    else:
        parts.append(f"[Intensity: Subtle {weather} effect]")

    parts.append(f"[Atmosphere: {condition['atmosphere']}]")
    parts.append(f"[Sky: {condition['sky']}]")
    parts.append(f"[Best For: {condition['best_for']}]")

    return " ".join(parts)


# Time of day lighting
TIME_OF_DAY = {
    "golden_hour": {
        "description": "Hour after sunrise or before sunset",
        "sun_angle": "0-10° above horizon",
        "color_temp": "3000-3500K (warm)",
        "light_quality": "Soft, warm, directional",
        "shadow_character": "Long, soft, warm-tinted",
        "atmosphere": "Enhanced scattering, golden/red tones",
        "best_for": "Warm lifestyle, emotional connection",
    },
    "sunrise": {
        "description": "Sun breaking horizon",
        "sun_angle": "0° (horizon)",
        "color_temp": "2500-3000K (very warm)",
        "light_quality": "Dramatic, low angle",
        "shadow_character": "Very long, dramatic",
        "atmosphere": "Intense red/orange, scattering maximum",
        "best_for": "Dramatic hero shots, new beginnings",
    },
    "midday": {
        "description": "Sun at highest point",
        "sun_angle": "60-90° above horizon",
        "color_temp": "5500-6000K (neutral)",
        "light_quality": "Hard, bright, direct",
        "shadow_character": "Short, harsh, high contrast",
        "atmosphere": "Minimal scattering, blue sky",
        "best_for": "Product visibility, technical shots",
    },
    "sunset": {
        "description": "Sun dropping below horizon",
        "sun_angle": "0° (horizon)",
        "color_temp": "2500-3000K (very warm)",
        "light_quality": "Dramatic, directional",
        "shadow_character": "Long, warm-tinted",
        "atmosphere": "Intense red/orange/purple",
        "best_for": "Emotional lifestyle, dramatic mood",
    },
    "blue_hour": {
        "description": "Time before sunrise/after sunset",
        "sun_angle": "0-6° below horizon",
        "color_temp": "8000-12000K (cool)",
        "light_quality": "Soft, diffuse, cool",
        "shadow_character": "Very soft, minimal",
        "atmosphere": "Deep blue, tranquil",
        "best_for": "Mood, calmness, evening context",
    },
    "night": {
        "description": "Dark with artificial lighting",
        "sun_angle": "N/A (below horizon)",
        "color_temp": "Variable (artificial sources)",
        "light_quality": "Dark, directional artificial",
        "shadow_character": "Hard, contrasty (from artificial lights)",
        "atmosphere": "Dark, potential light pollution",
        "best_for": "Evening context, product with lights",
    },
}


def get_time_of_day_spec(
    time: str = "golden_hour",
) -> str:
    """
    Get time of day lighting specification.

    Args:
        time: Time of day

    Returns:
        Formatted time of day specification
    """
    time_key = time.lower().replace(" ", "_")

    if time_key not in TIME_OF_DAY:
        return ""

    tod = TIME_OF_DAY[time_key]
    parts = []

    parts.append(f"[Time of Day: {time.title().replace('_', ' ')}]")
    parts.append(f"[Description: {tod['description']}]")
    parts.append(f"[Sun Angle: {tod['sun_angle']}]")
    parts.append(f"[Color Temp: {tod['color_temp']}]")
    parts.append(f"[Light Quality: {tod['light_quality']}]")
    parts.append(f"[Shadows: {tod['shadow_character']}]")
    parts.append(f"[Atmosphere: {tod['atmosphere']}]")
    parts.append(f"[Best For: {tod['best_for']}]")

    return " ".join(parts)


# Haze and mist effects
HAZE_MIST = {
    "morning_mist": {
        "description": "Low-lying mist in morning",
        "cause": "Cooling overnight, ground-level condensation",
        "color": "White/gray, cool tint",
        "density": "Low-lying, dissipates with height",
        "best_for": "Tranquil scenes, early morning context",
    },
    "mountain_haze": {
        "description": "Haze at altitude",
        "cause": "Thin air, UV scattering",
        "color": "Blue/cool tint",
        "density": "Uniform with distance",
        "best_for": "Mountain scenes, depth",
    },
    "industrial_haze": {
        "description": "Pollution/urban haze",
        "cause": "Particulates, smog",
        "color": "Brown/gray, warm tint",
        "density": "Uniform, reduces contrast",
        "best_for": "Urban context (usually avoid for products)",
    },
    "sea_fret": {
        "description": "Sea fog/mist",
        "cause": "Warm air over cold water",
        "color": "White/gray",
        "density": "Can be dense, moving inland",
        "best_for": "Coastal context, maritime mood",
    },
}


def get_haze_mist_spec(
    haze_type: str = "morning_mist",
    density: float = 0.5,
) -> str:
    """
    Get haze/mist specification.

    Args:
        haze_type: Type of haze/mist
        density: Density of effect (0.0 to 1.0)

    Returns:
        Formatted haze/mist specification
    """
    haze_key = haze_type.lower().replace(" ", "_")

    if haze_key not in HAZE_MIST:
        return ""

    haze = HAZE_MIST[haze_key]
    parts = []

    parts.append(f"[Haze/Mist: {haze['description']}]")
    parts.append(f"[Cause: {haze['cause']}]")
    parts.append(f"[Color: {haze['color']}]")

    if density > 0.7:
        parts.append("[Density: Dense haze, high scattering]")
    elif density > 0.4:
        parts.append("[Density: Moderate haze, visible scattering]")
    else:
        parts.append("[Density: Light haze, subtle scattering]")

    parts.append(f"[Characteristics: {haze['density']}]")
    parts.append(f"[Best For: {haze['best_for']}]")

    return " ".join(parts)


# Complete atmosphere specification
def get_complete_atmosphere_spec(
    weather: str = "clear",
    time_of_day: str = "golden_hour",
    scattering: str = "rayleigh",
    volumetric: bool = False,
    atmospheric_perspective: bool = False,
    depth: str = "medium",
) -> str:
    """
    Get complete atmosphere specification.

    Args:
        weather: Weather condition
        time_of_day: Time of day
        scattering: Scattering type
        volumetric: Include volumetric lighting
        atmospheric_perspective: Include atmospheric perspective
        depth: Depth for perspective (deep, medium, shallow)

    Returns:
        Complete formatted atmosphere specification
    """
    components = []

    # Weather
    if weather:
        components.append(get_weather_spec(weather))

    # Time of day
    if time_of_day:
        components.append(get_time_of_day_spec(time_of_day))

    # Scattering
    if scattering:
        components.append(get_scattering_spec(scattering))

    # Volumetric lighting
    if volumetric:
        components.append(get_volumetric_lighting_spec("god_rays", 0.6))

    # Atmospheric perspective
    if atmospheric_perspective:
        components.append(get_atmospheric_perspective_spec(depth))

    return " \n".join(components)


# Atmosphere presets for different moods
ATMOSPHERE_PRESETS = {
    "warm_golden": {
        "weather": "clear",
        "time_of_day": "golden_hour",
        "scattering": "rayleigh",
        "volumetric": True,
        "atmospheric_perspective": True,
        "depth": "deep",
        "mood": "Warm, inviting, emotional",
    },
    "dramatic_storm": {
        "weather": "rainy",
        "time_of_day": "sunset",
        "scattering": "mie",
        "volumetric": False,
        "atmospheric_perspective": True,
        "depth": "medium",
        "mood": "Dramatic, intense, moody",
    },
    "crisp_clean": {
        "weather": "clear",
        "time_of_day": "midday",
        "scattering": "rayleigh",
        "volumetric": False,
        "atmospheric_perspective": False,
        "depth": "shallow",
        "mood": "Clean, bright, visible",
    },
    "soft_morning": {
        "weather": "partly_cloudy",
        "time_of_day": "sunrise",
        "scattering": "rayleigh",
        "volumetric": True,
        "atmospheric_perspective": True,
        "depth": "medium",
        "mood": "Soft, fresh, new beginnings",
    },
    "moody_blue": {
        "weather": "overcast",
        "time_of_day": "blue_hour",
        "scattering": "mie",
        "volumetric": False,
        "atmospheric_perspective": True,
        "depth": "deep",
        "mood": "Moody, calm, serene",
    },
}


def get_atmosphere_preset(
    preset: str = "warm_golden",
) -> str:
    """
    Get atmosphere specification from preset.

    Args:
        preset: Preset name

    Returns:
        Complete formatted atmosphere specification
    """
    if preset not in ATMOSPHERE_PRESETS:
        return ""

    preset_config = ATMOSPHERE_PRESETS[preset]

    return get_complete_atmosphere_spec(
        weather=preset_config["weather"],
        time_of_day=preset_config["time_of_day"],
        scattering=preset_config["scattering"],
        volumetric=preset_config["volumetric"],
        atmospheric_perspective=preset_config["atmospheric_perspective"],
        depth=preset_config["depth"],
    )
