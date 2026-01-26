"""
Temporal Elements Module.

Time-based effects and motion-related elements for dynamic image generation.

Based on real-world physics of motion, time, and temporal phenomena.
"""

from typing import Dict, List, Optional


# Motion blur effects
MOTION_BLUR = {
    "description": "Blurring caused by relative motion between camera and subject",
    "direction": "Blurred in direction of motion",
    "intensity": "Proportional to speed and exposure time",
    "types": {
        "linear": "Straight-line motion (cars, running)",
        "radial": "Rotational motion (wheels, spinning)",
        "zoom": "Zoom during exposure (explosion effect)",
        "pan": "Camera follows subject (sharp subject, blurred background)",
    },
    "creative_use": "Convey speed, action, energy, time passage",
}


def get_motion_blur_spec(
    blur_type: str = "linear",
    intensity: float = 0.5,
    direction: str = "horizontal",
) -> str:
    """
    Get motion blur specification.

    Args:
        blur_type: Type of motion blur
        intensity: Intensity of blur (0.0 to 1.0)
        direction: Direction of blur (horizontal, vertical, radial, zoom)

    Returns:
        Formatted motion blur specification
    """
    parts = []

    parts.append(f"[Motion Blur: {MOTION_BLUR['description']}]")
    parts.append(f"[Type: {blur_type.title()}]")

    if blur_type == "linear":
        parts.append(f"[Direction: {direction.title()} motion blur]")
        if intensity > 0.7:
            parts.append("[Intensity: Strong blur, fast motion, long trails]")
        elif intensity > 0.4:
            parts.append("[Intensity: Moderate blur, medium speed]")
        else:
            parts.append("[Intensity: Subtle blur, slow motion]")
    elif blur_type == "radial":
        parts.append("[Direction: Radial blur from center outward]")
        parts.append("[Use: Spinning wheels, rotating objects, explosion effect]")
    elif blur_type == "zoom":
        parts.append("[Direction: Radial zoom blur toward/away from center]")
        parts.append("[Use: Dramatic effect, speed, impact]")
    elif blur_type == "pan":
        parts.append("[Technique: Panning motion - camera follows subject]")
        parts.append("[Effect: Sharp subject, motion-blurred background]")
        parts.append("[Use: Sports photography, moving vehicles, action shots]")

    parts.append(f"[Creative Use: {MOTION_BLUR['creative_use']}]")

    return " ".join(parts)


# Shutter speed effects
SHUTTER_SPEED = {
    "description": "Duration of camera shutter being open (exposure time)",
    "effects": {
        "fast": {
            "range": "1/500s to 1/8000s",
            "effect": "Freeze motion, sharp details",
            "best_for": "Sports, action, fast-moving subjects",
            "tradeoff": "Requires more light or wider aperture",
        },
        "medium": {
            "range": "1/60s to 1/250s",
            "effect": "Natural motion rendering, slight blur on fast movement",
            "best_for": "General photography, portraits, everyday scenes",
            "tradeoff": "Balanced exposure",
        },
        "slow": {
            "range": "1/15s to 1/60s",
            "effect": "Motion blur, smooth movement",
            "best_for": "Panning, intentional motion blur, low light",
            "tradeoff": "Risk of camera shake, may need tripod",
        },
        "very_slow": {
            "range": "1s to 30s",
            "effect": "Heavy motion blur, light trails, silky water",
            "best_for": "Long exposure, night photography, artistic effects",
            "tradeoff": "Requires tripod, stationary camera",
        },
    },
}


def get_shutter_speed_spec(
    speed_category: str = "medium",
    specific_speed: Optional[str] = None,
) -> str:
    """
    Get shutter speed specification.

    Args:
        speed_category: Category of shutter speed (fast, medium, slow, very_slow)
        specific_speed: Optional specific shutter speed (e.g., "1/250s")

    Returns:
        Formatted shutter speed specification
    """
    parts = []

    if specific_speed:
        parts.append(f"[Shutter Speed: {specific_speed}]")
    else:
        if speed_category in SHUTTER_SPEED["effects"]:
            spec = SHUTTER_SPEED["effects"][speed_category]
            parts.append(f"[Shutter Speed: {spec['range']}]")
            parts.append(f"[Effect: {spec['effect']}]")
            parts.append(f"[Best For: {spec['best_for']}]")
            parts.append(f"[Tradeoff: {spec['tradeoff']}]")
        else:
            return ""

    return " ".join(parts)


# Time-lapse and long exposure
LONG_EXPOSURE = {
    "description": "Extended exposure time to capture temporal changes",
    "effects": {
        "light_trails": {
            "description": "Streaks of light from moving sources",
            "subjects": "Car headlights, stars, fireworks",
            "exposure": "1s to 30s or longer",
            "technique": "Stationary camera, moving lights",
        },
        "silky_water": {
            "description": "Smooth, mist-like water surface",
            "subjects": "Waterfalls, rivers, ocean waves",
            "exposure": "0.5s to 5s",
            "technique": "Neutral density filter often needed",
        },
        "cloud_streaks": {
            "description": "Streaked, wispy cloud motion",
            "subjects": "Moving clouds across sky",
            "exposure": "10s to several minutes",
            "technique": "Very long exposure, tripod essential",
        },
        "crowd_blur": {
            "description": "People disappear or become ghost-like",
            "subjects": "Busy public spaces, crowds",
            "exposure": "1s to 10s",
            "technique": "Moving people blur, stationary elements sharp",
        },
    },
    "requirements": [
        "Stable camera (tripod essential)",
        "Often requires neutral density (ND) filters",
        "Remote shutter release recommended",
        "Low light or ND filters to avoid overexposure",
    ],
}


def get_long_exposure_spec(
    effect_type: str = "light_trails",
) -> str:
    """
    Get long exposure specification.

    Args:
        effect_type: Type of long exposure effect

    Returns:
        Formatted long exposure specification
    """
    if effect_type not in LONG_EXPOSURE["effects"]:
        return ""

    effect = LONG_EXPOSURE["effects"][effect_type]
    parts = []

    parts.append(f"[Long Exposure Effect: {effect['description']}]")
    parts.append(f"[Subjects: {effect['subjects']}]")
    parts.append(f"[Exposure Time: {effect['exposure']}]")
    parts.append(f"[Technique: {effect['technique']}]")

    parts.append("[Requirements]")
    for req in LONG_EXPOSURE["requirements"]:
        parts.append(f"- {req}")

    return " ".join(parts)


# Time of day transitions
TIME_TRANSITIONS = {
    "sunrise": {
        "colors": "Cool blues → Warm oranges/golds → Bright yellows",
        "duration": "30-60 minutes",
        "light_quality": "Soft, rapidly changing, dramatic",
        "shadows": "Long, soft, rapidly shortening",
        "mood": "Hope, new beginnings, fresh starts",
    },
    "sunset": {
        "colors": "Bright yellows → Warm oranges/reds → Deep purples/blues",
        "duration": "30-60 minutes",
        "light_quality": "Warm, golden, dramatic",
        "shadows": "Long, warm-tinted, lengthening",
        "mood": "Nostalgia, ending, warmth",
    },
    "golden_hour": {
        "colors": "Warm golden tones, soft highlights",
        "duration": "1 hour after sunrise or before sunset",
        "light_quality": "Soft, warm, flattering",
        "shadows": "Long but soft, golden-tinted",
        "mood": "Warmth, beauty, ideal photography",
    },
    "blue_hour": {
        "colors": "Deep blues, purples, cool tones",
        "duration": "20-30 minutes before sunrise/after sunset",
        "light_quality": "Soft, cool, diffuse",
        "shadows": "Very soft, minimal contrast",
        "mood": "Calm, tranquility, serenity",
    },
}


def get_time_transition_spec(
    transition: str = "golden_hour",
) -> str:
    """
    Get time of day transition specification.

    Args:
        transition: Type of time transition

    Returns:
        Formatted time transition specification
    """
    if transition not in TIME_TRANSITIONS:
        return ""

    trans = TIME_TRANSITIONS[transition]
    parts = []

    parts.append(f"[Time Transition: {transition.title().replace('_', ' ')}]")
    parts.append(f"[Color Progression: {trans['colors']}]")
    parts.append(f"[Duration: {trans['duration']}]")
    parts.append(f"[Light Quality: {trans['light_quality']}]")
    parts.append(f"[Shadows: {trans['shadows']}]")
    parts.append(f"[Mood: {trans['mood']}]")

    return " ".join(parts)


# Seasonal time indicators
SEASONAL_INDICATORS = {
    "spring": {
        "visual_cues": [
            "Fresh green buds and leaves",
            "Flowering trees and plants",
            "Pastel color palette",
            "Soft, fresh lighting",
        ],
        "atmosphere": "Renewal, growth, freshness",
        "colors": "Greens, pinks, yellows, light blues",
    },
    "summer": {
        "visual_cues": [
            "Lush full foliage",
            "Bright, direct sunlight",
            "Vibrant saturated colors",
            "Long shadows",
        ],
        "atmosphere": "Energy, warmth, abundance",
        "colors": "Deep greens, bright blues, warm yellows",
    },
    "autumn": {
        "visual_cues": [
            "Changing leaf colors (red, orange, yellow)",
            "Falling leaves",
            "Warm golden light",
            "Harvest themes",
        ],
        "atmosphere": "Change, transition, warmth",
        "colors": "Oranges, reds, browns, golds",
    },
    "winter": {
        "visual_cues": [
            "Snow or frost (if applicable)",
            "Bare trees or evergreens",
            "Cool, crisp lighting",
            "Warm indoor contrasts",
        ],
        "atmosphere": "Quiet, stillness, reflection",
        "colors": "Whites, cool blues, grays, warm accent colors",
    },
}


def get_seasonal_indicators_spec(
    season: str = "summer",
) -> str:
    """
    Get seasonal indicators specification.

    Args:
        season: Season name

    Returns:
        Formatted seasonal indicators specification
    """
    if season not in SEASONAL_INDICATORS:
        return ""

    seasonal = SEASONAL_INDICATORS[season]
    parts = []

    parts.append(f"[Seasonal Indicators: {season.title()}]")

    parts.append("[Visual Cues]")
    for cue in seasonal["visual_cues"]:
        parts.append(f"- {cue}")

    parts.append(f"[Atmosphere: {seasonal['atmosphere']}]")
    parts.append(f"[Color Palette: {seasonal['colors']}]")

    return " ".join(parts)


# Action and dynamics
ACTION_DYNAMICS = {
    "frozen_action": {
        "description": "Split-second moment frozen in time",
        "shutter": "Fast (1/500s or faster)",
        "examples": "Water droplets, splashing, athlete in mid-air",
        "effect": "Sharp detail, reveals invisible moments",
    },
    "motion_blur_action": {
        "description": "Intentional blur to convey speed and movement",
        "shutter": "Slow (1/60s or slower)",
        "examples": "Runner, moving car, flowing fabric",
        "effect": "Dynamic, energetic, sense of speed",
    },
    "panning": {
        "description": "Camera follows moving subject",
        "shutter": "Medium (1/60s to 1/125s)",
        "examples": "Cyclist, race car, athlete",
        "effect": "Sharp subject, blurred background, speed",
    },
    "explosion_impact": {
        "description": "Capturing moment of impact or explosion",
        "shutter": "Very fast (1/1000s or faster)",
        "examples": "Water balloon burst, powder explosion, collision",
        "effect": "Dramatic, frozen debris, dynamic energy",
    },
}


def get_action_dynamics_spec(
    action_type: str = "frozen_action",
) -> str:
    """
    Get action dynamics specification.

    Args:
        action_type: Type of action capture

    Returns:
        Formatted action dynamics specification
    """
    if action_type not in ACTION_DYNAMICS:
        return ""

    action = ACTION_DYNAMICS[action_type]
    parts = []

    parts.append(f"[Action Dynamics: {action['description']}]")
    parts.append(f"[Shutter Speed: {action['shutter']}]")
    parts.append(f"[Examples: {action['examples']}]")
    parts.append(f"[Effect: {action['effect']}]")

    return " ".join(parts)


# Time stamps and dating
TIMESTAMP_ELEMENTS = {
    "clocks": {
        "description": "Clock or watch showing specific time",
        "uses": "Establish time of day, create urgency, mark moment",
        "placement": "Background prop or product feature",
    },
    "shadows": {
        "description": "Shadow length and angle indicate time",
        "uses": "Natural time indicator, adds depth",
        "morning": "Long shadows to west",
        "midday": "Short shadows, directly below",
        "evening": "Long shadows to east",
    },
    "light_quality": {
        "description": "Quality of light indicates time",
        "morning": "Soft, cool to warm transition",
        "midday": "Bright, harsh, direct",
        "evening": "Warm, golden, low angle",
    },
    "activity": {
        "description": "Human activity suggests time",
        "morning": "Commuting, breakfast routines",
        "midday": "Work, school, active",
        "evening": "Relaxing, dinner, winding down",
    },
}


def get_timestamp_spec(
    method: str = "shadows",
    time_of_day: str = "morning",
) -> str:
    """
    Get timestamp element specification.

    Args:
        method: Method of indicating time
        time_of_day: Time of day to indicate

    Returns:
        Formatted timestamp specification
    """
    if method not in TIMESTAMP_ELEMENTS:
        return ""

    method_info = TIMESTAMP_ELEMENTS[method]
    parts = []

    parts.append(f"[Time Indicator: {method_info['description']}]")
    parts.append(f"[Uses: {method_info['uses']}]")

    # Add time-specific information
    if time_of_day == "morning":
        if method == "shadows":
            parts.append("[Morning Shadows: Long, pointing west, soft edges]")
        elif method == "light_quality":
            parts.append("[Morning Light: Soft, cool warming to golden, low angle]")
        elif method == "activity":
            parts.append("[Morning Activity: Commuting, breakfast, starting day]")
    elif time_of_day == "midday":
        if method == "shadows":
            parts.append("[Midday Shadows: Short, directly beneath objects, harsh]")
        elif method == "light_quality":
            parts.append("[Midday Light: Bright, direct, overhead, harsh contrast]")
        elif method == "activity":
            parts.append("[Midday Activity: Work, school, peak activity]")
    elif time_of_day == "evening":
        if method == "shadows":
            parts.append("[Evening Shadows: Long, pointing east, warm-tinted]")
        elif method == "light_quality":
            parts.append("[Evening Light: Warm, golden, low angle, dramatic]")
        elif method == "activity":
            parts.append("[Evening Activity: Relaxing, dinner, ending day]")

    return " ".join(parts)


# Complete temporal specification
def get_complete_temporal_spec(
    motion_blur: bool = False,
    shutter_speed: str = "medium",
    long_exposure: Optional[str] = None,
    time_transition: Optional[str] = None,
    season: Optional[str] = None,
    action: Optional[str] = None,
) -> str:
    """
    Get complete temporal specification.

    Args:
        motion_blur: Include motion blur
        shutter_speed: Shutter speed category
        long_exposure: Optional long exposure effect
        time_transition: Optional time of day transition
        season: Optional seasonal indicators
        action: Optional action dynamics

    Returns:
        Complete formatted temporal specification
    """
    components = []

    # Motion blur
    if motion_blur:
        components.append(get_motion_blur_spec())

    # Shutter speed
    components.append(get_shutter_speed_spec(shutter_speed))

    # Long exposure
    if long_exposure:
        components.append(get_long_exposure_spec(long_exposure))

    # Time transition
    if time_transition:
        components.append(get_time_transition_spec(time_transition))

    # Seasonal indicators
    if season:
        components.append(get_seasonal_indicators_spec(season))

    # Action dynamics
    if action:
        components.append(get_action_dynamics_spec(action))

    return "\n\n".join(components)


# Temporal presets
TEMPORAL_PRESETS = {
    "frozen_sports": {
        "motion_blur": False,
        "shutter_speed": "fast",
        "action": "frozen_action",
        "mood": "Dynamic, sharp, energetic",
    },
    "motion_blur_speed": {
        "motion_blur": True,
        "shutter_speed": "slow",
        "action": "motion_blur_action",
        "mood": "Fast, energetic, sense of speed",
    },
    "panning_follow": {
        "motion_blur": True,
        "shutter_speed": "medium",
        "action": "panning",
        "mood": "Sharp subject, speed, dynamic background",
    },
    "serene_long_exposure": {
        "motion_blur": False,
        "shutter_speed": "very_slow",
        "long_exposure": "silky_water",
        "mood": "Calm, smooth, ethereal",
    },
    "golden_hour_warmth": {
        "motion_blur": False,
        "shutter_speed": "medium",
        "time_transition": "golden_hour",
        "mood": "Warm, beautiful, ideal light",
    },
}


def get_temporal_preset(
    preset: str = "frozen_sports",
) -> str:
    """
    Get temporal specification from preset.

    Args:
        preset: Preset name

    Returns:
        Complete formatted temporal specification
    """
    if preset not in TEMPORAL_PRESETS:
        return ""

    preset_config = TEMPORAL_PRESETS[preset]

    return get_complete_temporal_spec(
        motion_blur=preset_config["motion_blur"],
        shutter_speed=preset_config["shutter_speed"],
        long_exposure=preset_config.get("long_exposure"),
        time_transition=preset_config.get("time_transition"),
        action=preset_config.get("action"),
    )
