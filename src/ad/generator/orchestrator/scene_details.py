"""
Scene-Specific Details Module.

Provides detailed scene configurations for different product contexts
and environments with appropriate props, backgrounds, and atmospheric elements.
"""

from typing import Dict, List, Optional


# Indoor scene types
INDOOR_SCENES = {
    "modern_living_room": {
        "description": "Contemporary home environment",
        "background": "Clean wall, subtle texture, neutral colors",
        "props": [
            "Modern sofa (optional, blurred)",
            "Houseplant (Monstera or Ficus)",
            "Coffee table with magazine",
            "Floor lamp or pendant light",
            "Rug or carpet",
        ],
        "lighting": "Soft natural light from window + ambient room lighting",
        "colors": "Neutrals (white, gray, beige) with accent colors",
        "surface": "Wooden table, marble countertop, or fabric surface",
        "best_for": "Home electronics, lifestyle products, smart devices",
    },
    "office_workspace": {
        "description": "Professional office environment",
        "background": "Clean office wall, bookshelf, or city view through window",
        "props": [
            "Laptop or monitor (blurred)",
            "Keyboard and mouse",
            "Notebook or documents",
            "Desk lamp",
            "Coffee mug or water bottle",
        ],
        "lighting": "Bright office lighting, fluorescent or LED panels",
        "colors": "Professional tones (blue, gray, white)",
        "surface": "Desk (wood, glass, or white laminate)",
        "best_for": "Tech accessories, productivity tools, office equipment",
    },
    "kitchen": {
        "description": "Modern kitchen environment",
        "background": "Kitchen cabinets, backsplash tiles, countertop",
        "props": [
            "Cutting board",
            "Knife or utensil",
            "Ingredients or food items",
            "Appliances (blender, toaster in background)",
            "Herbs or spices",
        ],
        "lighting": "Bright task lighting + under-cabinet lighting",
        "colors": "Warm tones (wood, stainless steel, white)",
        "surface": "Countertop (marble, granite, or quartz)",
        "best_for": "Kitchen appliances, cooking tools, food products",
    },
    "bedroom": {
        "description": "Restful bedroom environment",
        "background": "Bed, nightstand, or bedroom wall",
        "props": [
            "Bedding (sheets, pillows)",
            "Nightstand with lamp",
            "Book or alarm clock",
            "Window with curtains",
            "Small plant or vase",
        ],
        "lighting": "Soft warm lighting, bedside lamp, or natural light",
        "colors": "Calming tones (pastels, neutrals, soft blues)",
        "surface": "Nightstand, bed linens, or dresser top",
        "best_for": "Personal devices, sleep products, bedroom electronics",
    },
    "garage_workshop": {
        "description": "DIY workshop environment",
        "background": "Workbench, tools on wall pegboard",
        "props": [
            "Tools (wrenches, screwdrivers)",
            "Toolbox",
            "Work light",
            "Safety equipment",
            "Project materials (wood, parts)",
        ],
        "lighting": "Bright overhead fluorescent or LED shop lights",
        "colors": "Industrial tones (gray, metal, wood)",
        "surface": "Workbench (wood, metal, or composite)",
        "best_for": "Power tools, DIY equipment, workshop gear",
    },
}


# Outdoor scene types
OUTDOOR_SCENES = {
    "urban_street": {
        "description": "City street environment",
        "background": "City street, buildings, pedestrians (blurred)",
        "props": [
            "Street signs or traffic lights",
            "Urban vegetation (planters, trees)",
            "Bench or seating",
            "City infrastructure (lamp posts, bike rack)",
        ],
        "lighting": "Natural daylight or city evening lighting",
        "colors": "Urban tones (gray, concrete, brick, glass)",
        "surface": "Sidewalk, street furniture, or urban plaza",
        "best_for": "Portable electronics, urban gear, everyday carry items",
    },
    "park_nature": {
        "description": "Park or natural environment",
        "background": "Trees, grass, sky, park landscape",
        "props": [
            "Park bench or picnic table",
            "Trees and foliage",
            "Path or trail",
            "Flowers or natural plants",
            "Distant people enjoying park",
        ],
        "lighting": "Natural sunlight, dappled through trees",
        "colors": "Natural tones (greens, browns, blues)",
        "surface": "Grass, wooden bench, or picnic table",
        "best_for": "Outdoor gear, recreational products, lifestyle items",
    },
    "beach_coastal": {
        "description": "Beach or coastal environment",
        "background": "Ocean, sand, sky, coastal landscape",
        "props": [
            "Sand and shells",
            "Driftwood or rocks",
            "Umbrella or beach towel",
            "Water ripples or waves",
            "Coastal vegetation",
        ],
        "lighting": "Bright sunlight, warm golden hour, or overcast",
        "colors": "Coastal tones (blue, sand, white, aqua)",
        "surface": "Sand, rock, or beach towel",
        "best_for": "Water sports gear, beach accessories, summer products",
    },
    "mountain_hiking": {
        "description": "Mountain trail or hiking environment",
        "background": "Mountain peaks, trail, forest, sky",
        "props": [
            "Trail markers or signs",
            "Rocks and boulders",
            "Mountain vegetation",
            "Distant hikers",
            "Backpack or gear",
        ],
        "lighting": "Natural light (bright sun or dramatic clouds)",
        "colors": "Mountain tones (gray rock, green forest, blue sky)",
        "surface": "Rock, trail dirt, or mountain grass",
        "best_for": "Hiking gear, outdoor equipment, adventure products",
    },
    "suburban_backyard": {
        "description": "Subordinate backyard or garden",
        "background": "Backyard fence, garden, patio",
        "props": [
            "Lawn furniture",
            "Garden plants or flowers",
            "Grass or landscaping",
            "Patio or deck",
            "Outdoor toys or grill",
        ],
        "lighting": "Natural daylight or evening patio lights",
        "colors": "Garden tones (green, floral colors, wood)",
        "surface": "Patio, grass, or outdoor table",
        "best_for": "Home and garden products, outdoor living items",
    },
}


# Studio/abstract scene types
STUDIO_SCENES = {
    "infinity_cove": {
        "description": "Seamless curved background",
        "background": "Infinity curve (white, gray, or color)",
        "props": [
            "Minimal or no props",
            "Optional reflectors or diffusion panels",
        ],
        "lighting": "Controlled studio lighting (3-point setup)",
        "colors": "Neutral or brand colors (seamless)",
        "surface": "Sweep from floor to background (same material)",
        "best_for": "Product shots, catalog images, clean presentations",
    },
    "colored_backdrop": {
        "description": "Solid colored background",
        "background": "Paper or fabric sweep (color)",
        "props": [
            "Minimal props",
            "Optional color-coordinated elements",
        ],
        "lighting": "Even studio lighting, soft shadows",
        "colors": "Brand colors or complementary colors",
        "surface": "Sweep (same as background)",
        "best_for": "Brand consistency, catalog shots, e-commerce",
    },
    "textured_surface": {
        "description": "Interesting surface material",
        "background": "Solid color or gradient (blurred)",
        "props": [
            "Surface texture (wood, concrete, fabric)",
            "Optional complementary objects",
        ],
        "lighting": "Directional lighting to enhance texture",
        "colors": "Neutral tones with surface material colors",
        "surface": "Textured material (reclaimed wood, concrete, fabric)",
        "best_for": "Artistic shots, texture-focused products, premium positioning",
    },
}


def get_scene_spec(
    scene_type: str,
    environment: str = "indoor",
) -> str:
    """
    Get scene specification.

    Args:
        scene_type: Specific scene type
        environment: Environment category (indoor, outdoor, studio)

    Returns:
        Formatted scene specification
    """
    # Select scene database
    if environment == "indoor":
        scenes = INDOOR_SCENES
    elif environment == "outdoor":
        scenes = OUTDOOR_SCENES
    elif environment == "studio":
        scenes = STUDIO_SCENES
    else:
        return ""

    # Get scene config
    scene_key = scene_type.lower().replace(" ", "_")
    if scene_key not in scenes:
        return ""

    scene = scenes[scene_key]
    parts = []

    # Header
    parts.append(f"[Scene: {scene_type.title().replace('_', ' ')}]")
    parts.append(f"[Environment: {environment.title()}]")

    # Description
    parts.append(f"[Description: {scene['description']}]")

    # Background
    parts.append(f"[Background: {scene['background']}]")

    # Props
    if scene.get("props"):
        parts.append(f"[Props: {', '.join(scene['props'][:3])}]")
        if len(scene["props"]) > 3:
            parts.append(f"[Additional Props: {', '.join(scene['props'][3:])}]")

    # Lighting
    parts.append(f"[Lighting: {scene['lighting']}]")

    # Colors
    parts.append(f"[Color Palette: {scene['colors']}]")

    # Surface
    parts.append(f"[Surface: {scene['surface']}]")

    # Best for
    parts.append(f"[Best For: {scene['best_for']}]")

    return " ".join(parts)


# Scene elements by product type
PRODUCT_SCENE_MAPPING = {
    "electronics": {
        "primary_scenes": ["modern_living_room", "office_workspace"],
        "secondary_scenes": ["studio_colored_backdrop", "infinity_cove"],
        "surfaces": ["wooden_table", "marble_countertop", "desk_surface"],
        "props": ["laptop", "coffee_mug", "notebook", "houseplant"],
    },
    "kitchen_appliances": {
        "primary_scenes": ["kitchen"],
        "secondary_scenes": ["studio_colored_backdrop"],
        "surfaces": ["countertop", "kitchen_island"],
        "props": ["cutting_board", "ingredients", "utensils", "bowls"],
    },
    "power_tools": {
        "primary_scenes": ["garage_workshop"],
        "secondary_scenes": ["studio_textured_surface"],
        "surfaces": ["workbench", "concrete_floor"],
        "props": ["tools", "project_materials", "work_light", "safety_gear"],
    },
    "outdoor_gear": {
        "primary_scenes": ["park_nature", "mountain_hiking", "beach_coastal"],
        "secondary_scenes": ["studio_textured_surface"],
        "surfaces": ["grass", "rock", "trail", "sand"],
        "props": ["backpack", "trail_elements", "natural_vegetation"],
    },
    "furniture": {
        "primary_scenes": ["modern_living_room", "bedroom"],
        "secondary_scenes": ["studio_colored_backdrop"],
        "surfaces": ["hardwood_floor", "rug", "carpet"],
        "props": ["decor_items", "lighting", "plants", "art"],
    },
}


def get_recommended_scene(
    product_type: str,
    environment_preference: Optional[str] = None,
) -> Dict:
    """
    Get recommended scene for product type.

    Args:
        product_type: Type of product
        environment_preference: Optional environment preference

    Returns:
        Dictionary with scene recommendation
    """
    product_key = product_type.lower().replace(" ", "_")

    if product_key not in PRODUCT_SCENE_MAPPING:
        return {
            "scene_type": "studio_colored_backdrop",
            "environment": "studio",
            "reason": "Default studio setup for unknown product type",
        }

    mapping = PRODUCT_SCENE_MAPPING[product_key]

    # Use preferred environment if provided
    if environment_preference:
        if environment_preference == "indoor" and mapping["primary_scenes"]:
            return {
                "scene_type": mapping["primary_scenes"][0],
                "environment": "indoor",
                "reason": f"Primary indoor scene for {product_type}",
            }
        elif environment_preference == "outdoor" and mapping["primary_scenes"]:
            outdoor_scene = next(
                (s for s in mapping["primary_scenes"] if s in OUTDOOR_SCENES),
                None,
            )
            if outdoor_scene:
                return {
                    "scene_type": outdoor_scene,
                    "environment": "outdoor",
                    "reason": f"Primary outdoor scene for {product_type}",
                }

    # Default to first primary scene
    return {
        "scene_type": mapping["primary_scenes"][0],
        "environment": "indoor" if mapping["primary_scenes"][0] in INDOOR_SCENES else "studio",
        "reason": f"Primary scene for {product_type}",
    }


def get_scene_elements(
    scene_type: str,
    include_props: bool = True,
    include_lighting: bool = True,
    include_background: bool = True,
) -> Dict[str, any]:
    """
    Get scene elements for building custom scene.

    Args:
        scene_type: Type of scene
        include_props: Include props
        include_lighting: Include lighting
        include_background: Include background

    Returns:
        Dictionary with scene elements
    """
    # Search all scene databases
    all_scenes = {**INDOOR_SCENES, **OUTDOOR_SCENES, **STUDIO_SCENES}
    scene_key = scene_type.lower().replace(" ", "_")

    if scene_key not in all_scenes:
        return {}

    scene = all_scenes[scene_key]
    elements = {}

    if include_background:
        elements["background"] = scene.get("background", "")

    if include_props:
        elements["props"] = scene.get("props", [])

    if include_lighting:
        elements["lighting"] = scene.get("lighting", "")

    elements["colors"] = scene.get("colors", "")
    elements["surface"] = scene.get("surface", "")

    return elements


# Seasonal variations
SEASONAL_ADJUSTMENTS = {
    "spring": {
        "lighting": "Soft, fresh natural light, pastel tones",
        "colors": "Greens, pinks, yellows (fresh growth)",
        "atmosphere": "Renewal, freshness, lightness",
        "props": ["flowers", "fresh_leaves", "blossoms"],
    },
    "summer": {
        "lighting": "Bright, strong sunlight, long shadows",
        "colors": "Vibrant greens, blues, warm tones",
        "atmosphere": "Energy, warmth, abundance",
        "props": ["green_foliage", "bright_flowers", "shade_elements"],
    },
    "autumn": {
        "lighting": "Warm golden light, softer angles",
        "colors": "Oranges, reds, browns, golds",
        "atmosphere": "Cozy, change, harvest",
        "props": ["fall_leaves", "warm_textures", "earth_tones"],
    },
    "winter": {
        "lighting": "Cool, soft light, or warm indoor lighting",
        "colors": "Whites, blues, cool grays, warm accents",
        "atmosphere": "Cozy, crisp, quiet",
        "props": ["snow", "evergreen", "warm_textiles", "indoor_comfort"],
    },
}


def apply_seasonal_adjustment(
    scene_spec: str,
    season: str,
) -> str:
    """
    Apply seasonal adjustment to scene specification.

    Args:
        scene_spec: Original scene specification
        season: Season (spring, summer, autumn, winter)

    Returns:
        Adjusted scene specification with seasonal elements
    """
    if season not in SEASONAL_ADJUSTMENTS:
        return scene_spec

    seasonal = SEASONAL_ADJUSTMENTS[season]
    additions = []

    additions.append(f"\n[Seasonal Adjustment: {season.title()}]")
    additions.append(f"[Lighting: {seasonal['lighting']}]")
    additions.append(f"[Colors: {seasonal['colors']}]")
    additions.append(f"[Atmosphere: {seasonal['atmosphere']}]")

    if seasonal.get("props"):
        additions.append(f"[Seasonal Props: {', '.join(seasonal['props'][:3])}]")

    return scene_spec + "\n" + " ".join(additions)


# Time of day adjustments for scenes
TIME_ADJUSTMENTS = {
    "morning": {
        "lighting": "Soft, low-angle light, cool to warm transition",
        "atmosphere": "Fresh, new beginnings, quiet",
        "shadows": "Long, soft, directional",
    },
    "midday": {
        "lighting": "Bright, overhead, strong illumination",
        "atmosphere": "Active, energetic, clear",
        "shadows": "Short, harsh, high contrast",
    },
    "afternoon": {
        "lighting": "Warm, directional, golden tones approaching",
        "atmosphere": "Productive, warm, active",
        "shadows": "Lengthening, still defined",
    },
    "evening": {
        "lighting": "Warm golden hour, dramatic, or artificial lighting",
        "atmosphere": "Relaxed, ending day, cozy",
        "shadows": "Long, warm-tinted, dramatic",
    },
}


def apply_time_adjustment(
    scene_spec: str,
    time_of_day: str,
) -> str:
    """
    Apply time of day adjustment to scene specification.

    Args:
        scene_spec: Original scene specification
        time_of_day: Time of day (morning, midday, afternoon, evening)

    Returns:
        Adjusted scene specification with time elements
    """
    if time_of_day not in TIME_ADJUSTMENTS:
        return scene_spec

    time_adj = TIME_ADJUSTMENTS[time_of_day]
    additions = []

    additions.append(f"\n[Time of Day: {time_of_day.title()}]")
    additions.append(f"[Lighting: {time_adj['lighting']}]")
    additions.append(f"[Atmosphere: {time_adj['atmosphere']}]")
    additions.append(f"[Shadows: {time_adj['shadows']}]")

    return scene_spec + "\n" + " ".join(additions)


# Complete scene builder
def build_complete_scene(
    scene_type: str,
    environment: str = "indoor",
    season: Optional[str] = None,
    time_of_day: Optional[str] = None,
) -> str:
    """
    Build complete scene specification with all adjustments.

    Args:
        scene_type: Type of scene
        environment: Environment category
        season: Optional seasonal adjustment
        time_of_day: Optional time adjustment

    Returns:
        Complete formatted scene specification
    """
    # Base scene
    scene_spec = get_scene_spec(scene_type, environment)

    # Apply seasonal adjustment
    if season:
        scene_spec = apply_seasonal_adjustment(scene_spec, season)

    # Apply time adjustment
    if time_of_day:
        scene_spec = apply_time_adjustment(scene_spec, time_of_day)

    return scene_spec
