"""
Advanced Composition Module.

Professional composition techniques for visually balanced and impactful images.

Based on established photographic and art composition principles.
"""

from typing import Dict, List, Optional, Tuple


# Rule of Thirds
RULE_OF_THIRDS = {
    "description": "Divide frame into 3x3 grid, place key elements at intersections",
    "grid": "2 horizontal lines + 2 vertical lines at 1/3 and 2/3 positions",
    "intersections": "4 points where lines cross (power points)",
    "usage": "Place primary subject at one or more power points",
    "best_for": "General product photography, balanced compositions",
    "balance": "Asymmetrical balance, more dynamic than center",
}


def get_rule_of_thirds_spec(
    subject_position: str = "right_third",
) -> str:
    """
    Get rule of thirds composition specification.

    Args:
        subject_position: Position of subject (left_third, right_third, top_third, bottom_third, center)

    Returns:
        Formatted rule of thirds specification
    """
    parts = []

    parts.append(f"[Rule of Thirds Composition]")
    parts.append(f"[Description: {RULE_OF_THIRDS['description']}]")
    parts.append(f"[Grid: {RULE_OF_THIRDS['grid']}]")
    parts.append(f"[Power Points: {RULE_OF_THIRDS['intersections']}]")

    position_map = {
        "left_third": "Place subject at left vertical third (1/3 from left)",
        "right_third": "Place subject at right vertical third (2/3 from left)",
        "top_third": "Place subject at top horizontal third (1/3 from top)",
        "bottom_third": "Place subject at bottom horizontal third (2/3 from top)",
        "center": "Place subject at center intersection (for symmetry)",
    }

    if subject_position in position_map:
        parts.append(f"[Subject Position: {position_map[subject_position]}]")

    parts.append(f"[Best For: {RULE_OF_THIRDS['best_for']}]")
    parts.append(f"[Balance: {RULE_OF_THIRDS['balance']}]")

    return " ".join(parts)


# Golden Ratio
GOLDEN_RATIO = {
    "description": "Spiral based on φ (phi) ≈ 1.618, natural proportions",
    "spiral": "Logarithmic spiral expanding outward",
    "placement": "Place subject along spiral curve or at focal point",
    "proportions": "1:1.618 (width:height or element ratios)",
    "usage": "Create natural flow, lead eye through composition",
    "best_for": "Hero shots, premium products, elegant layouts",
    "aesthetic": "More organic and dynamic than rule of thirds",
}


def get_golden_ratio_spec(
    orientation: str = "landscape",
) -> str:
    """
    Get golden ratio composition specification.

    Args:
        orientation: Image orientation (landscape, portrait)

    Returns:
        Formatted golden ratio specification
    """
    parts = []

    parts.append(f"[Golden Ratio Composition]")
    parts.append(f"[Description: {GOLDEN_RATIO['description']}]")
    parts.append(f"[Spiral: {GOLDEN_RATIO['spiral']}]")

    if orientation == "landscape":
        parts.append("[Frame Ratio: 1:1.618 (width:height)]")
        parts.append("[Subject Placement: Along golden spiral, focal point at top-right intersection]")
    else:  # portrait
        parts.append("[Frame Ratio: 1.618:1 (height:width)]")
        parts.append("[Subject Placement: Along ascending spiral, focal point at upper-third]")

    parts.append(f"[Usage: {GOLDEN_RATIO['usage']}]")
    parts.append(f"[Best For: {GOLDEN_RATIO['best_for']}]")
    parts.append(f"[Aesthetic: {GOLDEN_RATIO['aesthetic']}]")

    return " ".join(parts)


# Leading Lines
LEADING_LINES = {
    "description": "Lines that guide viewer's eye through composition",
    "types": [
        "Straight lines (roads, edges, horizons)",
        "Diagonal lines (dynamic, create movement)",
        "Curved lines (organic, gentle flow)",
        "Converging lines (draw eye to vanishing point)",
    ],
    "placement": "Direct eye from foreground to subject",
    "usage": "Create depth, guide attention, tell visual story",
    "best_for": "Product in environment, lifestyle context",
}


def get_leading_lines_spec(
    line_type: str = "diagonal",
    direction: str = "to_subject",
) -> str:
    """
    Get leading lines composition specification.

    Args:
        line_type: Type of leading lines (straight, diagonal, curved, converging)
        direction: Direction lines lead (to_subject, to_background, across)

    Returns:
        Formatted leading lines specification
    """
    parts = []

    parts.append(f"[Leading Lines Composition]")
    parts.append(f"[Description: {LEADING_LINES['description']}]")
    parts.append(f"[Line Type: {line_type.title()}]")

    if line_type == "straight":
        parts.append("[Use: Horizontal/vertical lines for stability, structure]")
    elif line_type == "diagonal":
        parts.append("[Use: Dynamic diagonals for energy, movement, tension]")
    elif line_type == "curved":
        parts.append("[Use: S-curves for organic flow, gentle visual journey]")
    elif line_type == "converging":
        parts.append("[Use: Lines converging to vanishing point for depth]")

    parts.append(f"[Direction: Lines lead {direction}]")
    parts.append(f"[Usage: {LEADING_LINES['usage']}]")
    parts.append(f"[Best For: {LEADING_LINES['best_for']}]")

    return " ".join(parts)


# Symmetry and Patterns
SYMMETRY = {
    "description": "Balanced repetition of elements across central axis",
    "types": [
        "Bilateral symmetry (left-right mirror)",
        "Radial symmetry (circular, radiating from center)",
        "Translational symmetry (repeating pattern)",
    ],
    "usage": "Create order, stability, visual satisfaction",
    "best_for": "Technical products, precision engineering, minimalist design",
    "balance": "Perfect balance, calm, professional",
}


def get_symmetry_spec(
    symmetry_type: str = "bilateral",
) -> str:
    """
    Get symmetry composition specification.

    Args:
        symmetry_type: Type of symmetry (bilateral, radial, translational)

    Returns:
        Formatted symmetry specification
    """
    parts = []

    parts.append(f"[Symmetrical Composition]")
    parts.append(f"[Description: {SYMMETRY['description']}]")
    parts.append(f"[Type: {symmetry_type.title()} Symmetry]")

    if symmetry_type == "bilateral":
        parts.append("[Axis: Vertical center line, left-right mirror]")
        parts.append("[Subject: Centered, balanced on both sides]")
    elif symmetry_type == "radial":
        parts.append("[Axis: Central point, radiating outward]")
        parts.append("[Subject: Center, elements arranged circularly]")
    elif symmetry_type == "translational":
        parts.append("[Pattern: Repeating elements across frame]")
        parts.append("[Subject: Part of repeating sequence, or breaks pattern for emphasis]")

    parts.append(f"[Usage: {SYMMETRY['usage']}]")
    parts.append(f"[Best For: {SYMMETRY['best_for']}]")
    parts.append(f"[Balance: {SYMMETRY['balance']}]")

    return " ".join(parts)


# Depth and Layering
DEPTH_LAYERING = {
    "description": "Organize elements into foreground, midground, background",
    "layers": {
        "foreground": "Close to camera, provides context, frame within frame",
        "midground": "Primary subject area, main focus",
        "background": "Distant elements, environment, context",
    },
    "usage": "Create 3D depth, separation, visual hierarchy",
    "best_for": "Wide scenes, product in environment, hero shots",
    "techniques": [
        "Selective focus (blur foreground/background)",
        "Atmospheric perspective (haze with distance)",
        "Size scaling (foreground larger, background smaller)",
        "Overlapping elements (occlusion creates depth)",
    ],
}


def get_depth_layering_spec(
    num_layers: int = 3,
    focus_layer: str = "midground",
) -> str:
    """
    Get depth layering composition specification.

    Args:
        num_layers: Number of layers (2, 3, or 4)
        focus_layer: Which layer is in focus (foreground, midground, background)

    Returns:
        Formatted depth layering specification
    """
    parts = []

    parts.append(f"[Depth Layering Composition]")
    parts.append(f"[Description: {DEPTH_LAYERING['description']}]")

    if num_layers >= 2:
        parts.append("[Foreground: Contextual elements, frame edges, depth cue]")
    if num_layers >= 3:
        parts.append("[Midground: Primary subject, sharp focus, main visual]")
    if num_layers >= 4:
        parts.append("[Background: Environment, context, atmospheric haze]")

    parts.append(f"[Focus Layer: {focus_layer.title()}]")

    if focus_layer == "foreground":
        parts.append("[DOF: Shallow, background blurred (f/2.8)]")
    elif focus_layer == "midground":
        parts.append("[DOF: Moderate, foreground soft, background slightly soft (f/5.6)]")
    else:  # background
        parts.append("[DOF: Deep, all layers sharp (f/11)]")

    parts.append(f"[Usage: {DEPTH_LAYERING['usage']}]")
    parts.append(f"[Best For: {DEPTH_LAYERING['best_for']}]")

    return " ".join(parts)


# Negative Space
NEGATIVE_SPACE = {
    "description": "Empty area around subject, breathing room",
    "proportions": {
        "minimal": "10-20% negative space (tight)",
        "moderate": "30-40% negative space (balanced)",
        "generous": "50-60% negative space (airy)",
        "extreme": "70%+ negative space (minimalist)",
    },
    "usage": "Emphasize subject, create focus, reduce clutter",
    "best_for": "Minimalist products, luxury branding, clean aesthetics",
    "psychology": "Negative space = luxury, premium, sophistication",
}


def get_negative_space_spec(
    proportion: str = "moderate",
    placement: str = "surrounding",
) -> str:
    """
    Get negative space composition specification.

    Args:
        proportion: Amount of negative space (minimal, moderate, generous, extreme)
        placement: Where negative space is (surrounding, one_side, top, bottom)

    Returns:
        Formatted negative space specification
    """
    parts = []

    parts.append(f"[Negative Space Composition]")
    parts.append(f"[Description: {NEGATIVE_SPACE['description']}]")

    if proportion in NEGATIVE_SPACE["proportions"]:
        parts.append(f"[Proportion: {NEGATIVE_SPACE['proportions'][proportion]}]")

    parts.append(f"[Placement: {placement.title()}]")

    if placement == "one_side":
        parts.append("[Subject on one side, empty space on other (rule of thirds)]")
    elif placement == "top":
        parts.append("[Subject at bottom, empty space above (breathing room)]")
    elif placement == "bottom":
        parts.append("[Subject at top, empty space below (grounding)]")
    else:  # surrounding
        parts.append("[Subject surrounded by empty space (isolation, emphasis)]")

    parts.append(f"[Usage: {NEGATIVE_SPACE['usage']}]")
    parts.append(f"[Best For: {NEGATIVE_SPACE['best_for']}]")
    parts.append(f"[Psychology: {NEGATIVE_SPACE['psychology']}]")

    return " ".join(parts)


# Framing
FRAMING = {
    "description": "Use elements to create frame around subject",
    "types": [
        "Natural framing (windows, arches, branches)",
        "Geometric framing (rectangles, circles, triangles)",
        "Environmental framing (foreground elements)",
        "Light framing (vignette, light shafts)",
    ],
    "usage": "Focus attention, create depth, add context",
    "best_for": "Product in context, lifestyle shots, environmental portraiture",
    "technique": "Frame should not compete with subject for attention",
}


def get_framing_spec(
    frame_type: str = "natural",
) -> str:
    """
    Get framing composition specification.

    Args:
        frame_type: Type of framing (natural, geometric, environmental, light)

    Returns:
        Formatted framing specification
    """
    parts = []

    parts.append(f"[Framing Composition]")
    parts.append(f"[Description: {FRAMING['description']}]")
    parts.append(f"[Type: {frame_type.title()} Framing]")

    if frame_type == "natural":
        parts.append("[Use: Windows, doorways, arches, branches as natural frame]")
    elif frame_type == "geometric":
        parts.append("[Use: Rectangles, circles, triangles to create geometric frame]")
    elif frame_type == "environmental":
        parts.append("[Use: Foreground elements (people, objects) to frame subject]")
    elif frame_type == "light":
        parts.append("[Use: Vignette, light shafts, shadow edges to create light frame]")

    parts.append(f"[Usage: {FRAMING['usage']}]")
    parts.append(f"[Best For: {FRAMING['best_for']}]")
    parts.append(f"[Technique: {FRAMING['technique']}]")

    return " ".join(parts)


# Viewpoint and Perspective
VIEWPOINT = {
    "description": "Camera angle and height relative to subject",
    "types": {
        "eye_level": "Horizontal to subject, neutral, realistic",
        "high_angle": "Looking down, diminishes subject, shows context",
        "low_angle": "Looking up, empowers subject, dramatic",
        "birds_eye": "Straight down, flat-lay, overview",
        "worms_eye": "Straight up, dramatic, emphasis on height",
        "dutch_angle": "Tilted horizon, dynamic, tension",
    },
    "usage": "Create mood, establish relationship, show perspective",
    "best_for": "Product demonstration, dramatic effect, unique angle",
}


def get_viewpoint_spec(
    viewpoint: str = "eye_level",
    angle_degrees: Optional[float] = None,
) -> str:
    """
    Get viewpoint composition specification.

    Args:
        viewpoint: Type of viewpoint
        angle_degrees: Optional specific angle in degrees

    Returns:
        Formatted viewpoint specification
    """
    parts = []

    parts.append(f"[Viewpoint: {viewpoint.title().replace('_', ' ')}]")

    if viewpoint in VIEWPOINT["types"]:
        parts.append(f"[Description: {VIEWPOINT['types'][viewpoint]}]")

    if angle_degrees is not None:
        parts.append(f"[Angle: {angle_degrees:.0f}° from horizontal]")

    parts.append(f"[Usage: {VIEWPOINT['usage']}]")
    parts.append(f"[Best For: {VIEWPOINT['best_for']}]")

    return " ".join(parts)


# Scale and Proportion
SCALE_PROPORTION = {
    "description": "Relative size of elements to show scale",
    "techniques": [
        "Include human element for scale reference",
        "Show product in hand or held by person",
        "Place product next to familiar objects",
        "Use environmental context (room, furniture)",
    ],
    "usage": "Communicate size, create relatability, show context",
    "best_for": "Products where size matters (small electronics, furniture, appliances)",
}


def get_scale_spec(
    reference: str = "human_hand",
) -> str:
    """
    Get scale and proportion composition specification.

    Args:
        reference: Reference for scale (human_hand, person, object, environment)

    Returns:
        Formatted scale specification
    """
    parts = []

    parts.append(f"[Scale and Proportion]")
    parts.append(f"[Description: {SCALE_PROPORTION['description']}]")

    if reference == "human_hand":
        parts.append("[Reference: Product held in hand, shows true scale]")
    elif reference == "person":
        parts.append("[Reference: Person with product or using product]")
    elif reference == "object":
        parts.append("[Reference: Familiar object (coin, pen, phone) for comparison]")
    elif reference == "environment":
        parts.append("[Reference: Product in environment (room, table) for context]")

    parts.append(f"[Usage: {SCALE_PROPORTION['usage']}]")
    parts.append(f"[Best For: {SCALE_PROPORTION['best_for']}]")

    return " ".join(parts)


# Visual Balance
VISUAL_BALANCE = {
    "description": "Distribution of visual weight across frame",
    "types": {
        "symmetrical": "Equal visual weight on both sides, stable",
        "asymmetrical": "Different elements with equal weight, dynamic",
        "radial": "Elements balanced around central point",
        "crystallographic": "Repeated pattern across entire frame",
    },
    "considerations": [
        "Size: Larger elements have more weight",
        "Color: Bright/warm colors appear heavier",
        "Position: Elements further from center have more leverage",
        "Texture: Detailed elements draw more attention",
    ],
    "usage": "Create stability, harmony, visual satisfaction",
    "best_for": "All compositions, essential for pleasing images",
}


def get_visual_balance_spec(
    balance_type: str = "asymmetrical",
) -> str:
    """
    Get visual balance composition specification.

    Args:
        balance_type: Type of balance

    Returns:
        Formatted visual balance specification
    """
    parts = []

    parts.append(f"[Visual Balance: {balance_type.title()}]")

    if balance_type in VISUAL_BALANCE["types"]:
        parts.append(f"[Description: {VISUAL_BALANCE['types'][balance_type]}]")

    if balance_type == "asymmetrical":
        parts.append("[Technique: Balance large subject on one side with small detailed elements on other]")

    parts.append(f"[Usage: {VISUAL_BALANCE['usage']}]")
    parts.append(f"[Best For: {VISUAL_BALANCE['best_for']}]")

    return " ".join(parts)


# Composition presets
COMPOSITION_PRESETS = {
    "hero_product": {
        "composition": "golden_ratio",
        "viewpoint": "slightly_low_angle",
        "depth": "3_layers",
        "negative_space": "moderate",
        "balance": "asymmetrical",
        "best_for": "Flagship product, premium positioning",
    },
    "technical_detail": {
        "composition": "symmetrical",
        "viewpoint": "eye_level",
        "depth": "shallow",
        "negative_space": "minimal",
        "balance": "symmetrical",
        "best_for": "Close-ups, features, specifications",
    },
    "lifestyle_context": {
        "composition": "rule_of_thirds",
        "viewpoint": "eye_level",
        "depth": "deep",
        "negative_space": "generous",
        "balance": "asymmetrical",
        "best_for": "Product in use, environmental context",
    },
    "minimalist_elegant": {
        "composition": "negative_space",
        "viewpoint": "eye_level",
        "depth": "shallow",
        "negative_space": "extreme",
        "balance": "symmetrical",
        "best_for": "Luxury products, clean aesthetics",
    },
}


def get_composition_preset(
    preset: str = "hero_product",
) -> Dict[str, str]:
    """
    Get complete composition preset.

    Args:
        preset: Preset name

    Returns:
        Dictionary with composition specifications
    """
    if preset not in COMPOSITION_PRESETS:
        return {}

    preset_config = COMPOSITION_PRESETS[preset]

    return {
        "composition": preset_config["composition"],
        "viewpoint": preset_config["viewpoint"],
        "depth": preset_config["depth"],
        "negative_space": preset_config["negative_space"],
        "balance": preset_config["balance"],
        "best_for": preset_config["best_for"],
    }


# Complete composition specification
def get_complete_composition_spec(
    composition: str = "rule_of_thirds",
    viewpoint: str = "eye_level",
    depth_layers: int = 3,
    negative_space: str = "moderate",
    balance: str = "asymmetrical",
) -> str:
    """
    Get complete composition specification.

    Args:
        composition: Primary composition technique
        viewpoint: Camera viewpoint
        depth_layers: Number of depth layers
        negative_space: Amount of negative space
        balance: Type of visual balance

    Returns:
        Complete formatted composition specification
    """
    components = []

    # Main composition
    if composition == "rule_of_thirds":
        components.append(get_rule_of_thirds_spec())
    elif composition == "golden_ratio":
        components.append(get_golden_ratio_spec())
    elif composition == "leading_lines":
        components.append(get_leading_lines_spec())
    elif composition == "symmetry":
        components.append(get_symmetry_spec())

    # Viewpoint
    components.append(get_viewpoint_spec(viewpoint))

    # Depth
    components.append(get_depth_layering_spec(depth_layers))

    # Negative space
    components.append(get_negative_space_spec(negative_space))

    # Balance
    components.append(get_visual_balance_spec(balance))

    return "\n\n".join(components)
