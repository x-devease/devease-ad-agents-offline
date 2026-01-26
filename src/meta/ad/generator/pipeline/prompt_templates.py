"""
Prompt Templates: Structured, actionable prompts for image generation.

This module replaces the abstract feature-to-prompt approach with
structured templates that provide concrete, actionable instructions.

Key principles (from devease-image-gen-offline patterns):
1. Concrete specifications (dimensions, weights, placement rules)
2. Structured variables (not free-form features)
3. Explicit MUST/NEVER constraints
4. Three-point lighting setup
5. Specific composition requirements

The image generation model can follow concrete instructions like:
  "Product positioned at foreground-center, 25% of frame"
But struggles with abstract features like:
  "visual_prominence: dominant"
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class PhysicalSpecs:
    """Physical specifications for product."""

    dimensions: Optional[str] = None  # e.g., "27 x 19 x 21 inches"
    weight: Optional[str] = None  # e.g., "215 lbs"
    size_reference: Optional[str] = None  # e.g., "waist-high to adult"


@dataclass
class PlacementConstraints:
    """Placement constraints for product."""

    valid_surfaces: List[str] = field(
        default_factory=lambda: ["floor", "ground"]
    )
    invalid_surfaces: List[str] = field(
        default_factory=lambda: ["counter", "table", "shelf"]
    )


@dataclass
class BrandRequirements:
    """Brand and UI requirements."""

    brand: str
    brand_style: str = "exact"  # exact, lowercase, uppercase
    logo_position: Optional[str] = None  # e.g., "front panel"
    display_content: Optional[str] = None  # e.g., "50%"


@dataclass
class ProductSpec:
    """Concrete product specifications."""

    name: str
    brand_req: BrandRequirements
    physical: Optional[PhysicalSpecs] = None
    placement: Optional[PlacementConstraints] = None


@dataclass
class CompositionSpec:
    """Composition specifications for scene."""

    product_position: str = "foreground-center"
    product_size_pct: int = 25  # percentage of frame
    background_elements: int = 3  # max number of elements


@dataclass
class HumanElementSpec:
    """Human element specifications for scene."""

    include_person: bool = False
    person_action: Optional[str] = None  # e.g., "using product"


@dataclass
class SceneSpec:
    """Concrete scene specifications."""

    environment: str  # e.g., "modern living room", "backyard"
    time_of_day: str = "daylight"  # daylight, golden_hour, evening, night
    mood: str = "aspirational"  # aspirational, urgent, cozy, professional
    composition: Optional[CompositionSpec] = None
    human: Optional[HumanElementSpec] = None

    def __post_init__(self):
        """Initialize optional sub-specs with defaults."""
        if self.composition is None:
            self.composition = CompositionSpec()
        if self.human is None:
            self.human = HumanElementSpec()


@dataclass
class LightingSpec:
    """Three-point lighting specification."""

    key_light: str = "45° angle, high-intensity studio strobe"
    fill_light: str = "1.5:1 ratio"
    rim_light: str = "strong crisp rim light for sharp silhouette definition"
    product_brightness: str = "2-3 stops brighter than background"
    overall_mood: str = (
        "vibrant high-key advertising lighting, ultra-bright atmosphere"
    )


@dataclass
class CameraSettings:
    """Camera and lens settings."""

    camera: str = "full-frame camera"
    lens: str = "85mm f/2.8"  # 85mm for compression, f/2.8 for controlled blur
    depth_of_field: str = "shallow depth of field"
    bokeh: str = "natural bokeh with polygonal aperture shapes"


@dataclass
class PhysicalInteractions:
    """Physical interaction settings."""

    contact_shadows: bool = True  # Objects touching surfaces show shadows
    surface_compression: bool = True  # Weight creates indentations


@dataclass
class MaterialImperfections:
    """Material imperfection settings (avoid plastic/AI look)."""

    surface_texture: str = "visible surface texture"
    dust_particles: str = "subtle dust particles in light"
    micro_variations: str = "micro surface variations"
    film_grain: str = "slight film grain"


@dataclass
class LightingRealism:
    """Lighting realism settings."""

    shadow_consistency: str = "shadows consistent with light direction"
    light_falloff: str = "believable light direction and falloff"


@dataclass
class RealismSpec:
    """
    Photographic realism specifications to avoid the "AI look".

    Based on patterns from devease-image-gen-offline/gpt_nano_processor.py.

    Camera/Lens Rationale:
    - 85mm: Classic portrait/product lens, flattering compression,
      comfortable working distance, beautiful bokeh
    - 50mm: "Normal" lens, slight compression, versatile
    - 35mm: Wide angle, shows more context, some distortion

    Aperture Rationale:
    - f/1.8: Very shallow DOF, dreamy, risk of product blur
    - f/2.8: Shallow but controlled DOF, product sharp, bg soft (DEFAULT)
    - f/4: Moderate DOF, more context visible
    - f/8+: Deep DOF, everything sharp, good for technical shots
    """

    camera: Optional[CameraSettings] = None
    physical: Optional[PhysicalInteractions] = None
    material: Optional[MaterialImperfections] = None
    lighting: Optional[LightingRealism] = None

    def __post_init__(self):
        """Initialize optional sub-specs with defaults."""
        if self.camera is None:
            self.camera = CameraSettings()
        if self.physical is None:
            self.physical = PhysicalInteractions()
        if self.material is None:
            self.material = MaterialImperfections()
        if self.lighting is None:
            self.lighting = LightingRealism()


# Pre-defined realism presets for common scenarios
REALISM_PRESETS = {
    # Hero product shot - shallow DOF, product is star
    "hero_shot": RealismSpec(
        camera=CameraSettings(
            lens="85mm f/2.8",
            depth_of_field="shallow depth of field",
            bokeh="natural bokeh with polygonal aperture shapes",
        ),
    ),
    # Lifestyle scene - moderate DOF, context matters
    "lifestyle": RealismSpec(
        camera=CameraSettings(
            lens="50mm f/4",
            depth_of_field="moderate depth of field",
            bokeh="soft background blur",
        ),
    ),
    # Technical/detail shot - everything sharp
    "technical": RealismSpec(
        camera=CameraSettings(
            lens="50mm f/8",
            depth_of_field="deep focus, everything sharp",
            bokeh="minimal blur, sharp throughout",
        ),
    ),
    # Close-up detail - very shallow DOF
    "close_up": RealismSpec(
        camera=CameraSettings(
            lens="85mm f/1.8",
            depth_of_field="very shallow depth of field",
            bokeh="creamy bokeh, dreamy background",
        ),
    ),
    # Wide environment shot - shows context
    "environment": RealismSpec(
        camera=CameraSettings(
            lens="35mm f/5.6",
            depth_of_field="moderate to deep depth of field",
            bokeh="subtle background softness",
        ),
    ),
}


class PromptBuilder:
    """
    Build structured, actionable prompts for image generation.

    Instead of translating abstract features, this provides concrete
    instructions that image generation models can reliably follow.
    """

    def __init__(
        self,
        product_spec: ProductSpec,
        scene_spec: Optional[SceneSpec] = None,
        lighting_spec: Optional[LightingSpec] = None,
        realism_spec: Optional[RealismSpec] = None,
    ):
        self.product = product_spec
        self.scene = scene_spec or SceneSpec(environment="studio")
        self.lighting = lighting_spec or LightingSpec()
        self.realism = realism_spec or RealismSpec()

    def build_system_prompt(self) -> str:
        """Build system prompt with role and constraints."""
        brand_instruction = self._get_brand_instruction()
        placement_constraints = self._get_placement_constraints()

        return f"""You are a Nano Banana prompt generator for \
{self.product.name} marketing images.
Maintain professional advertising photography standards.

CORE OBJECTIVES:
1. Create visually compelling product photography with perfect lighting
2. Show product as premium, reliable solution
3. Generate emotional connection through storytelling
4. Maintain consistent professional quality

PRODUCT SPECIFICATIONS:
- Product: {self.product.name}
- Brand: {self.product.brand_req.brand}
{self._format_physical_specs()}

{placement_constraints}

{brand_instruction}

MANDATORY THREE-POINT LIGHTING:
1. KEY LIGHT: {self.lighting.key_light}
2. FILL LIGHT: {self.lighting.fill_light}
3. RIM LIGHT: {self.lighting.rim_light}
4. PRODUCT BRIGHTNESS: {self.lighting.product_brightness}

MUST INCLUDE:
- Complete three-point lighting setup on product
- EXACT SAME product angle as input image
- Product positioned at {self.scene.composition.product_position}
- Product size: approximately {self.scene.composition.product_size_pct}% of frame
- Clear brand visibility (never obscured or overexposed)
- Realistic product scale and positioning
- Maximum {self.scene.composition.background_elements} background elements (less is more)

NEVER INCLUDE:
- Different product angle than input image
- Product in shadow or poorly lit
- Overexposed logos or details
- Unrealistic size (too big or too small)
- Floating or impossible positioning
- Dark, gloomy atmosphere
- Cluttered background with too many elements
"""

    def build_user_prompt(self) -> str:
        """Build user prompt with scene requirements."""
        person_instruction = ""
        if self.scene.human.include_person:
            person_instruction = f"""
HUMAN ELEMENT:
- Include person {self.scene.human.person_action or 'interacting with product'}
- Person should complement, not overshadow product
"""

        return f"""Create a Nano Banana prompt for this {self.product.name} image.

SCENE REQUIREMENTS:
- Environment: {self.scene.environment}
- Time of day: {self.scene.time_of_day}
- Mood: {self.scene.mood}
- Product as hero with breathing room
- Simple, uncluttered background
{person_instruction}
HERO SHOT REQUIREMENTS:
- Product as absolute focal point
- {self.lighting.overall_mood}
- Include size reference if helpful for realism
- PRESERVE EXACT PRODUCT ANGLE from input

OUTPUT FORMAT:
Start with: "Professional product photography of {self.product.name} with..."
Include: Complete lighting setup, background scenario, story context
End with: Mood descriptors and technical quality markers (8K, ultra-detailed, etc.)
"""

    def build_combined_prompt(self) -> str:
        """Build a single combined prompt for direct image generation."""
        brand_name = self._format_brand_name()

        return f"""Professional product photography of {self.product.name} with
{self.lighting.overall_mood} lighting.

SCENE: {self.scene.environment}, {self.scene.time_of_day}, {self.scene.mood} atmosphere.

COMPOSITION:
- Product positioned at {self.scene.composition.product_position},
  {self.scene.composition.product_size_pct}% of frame
- Three-point lighting: key at 45°, fill at 2:1 ratio, subtle rim light
- Product 1-2 stops brighter than background
- Maximum {self.scene.composition.background_elements} background elements
- Shallow depth of field, background softly out of focus

BRAND: {brand_name} logo clearly visible on product, sharp and undistorted.

QUALITY: 8K, ultra-detailed, professional advertising photography, cinematic lighting.
"""

    def build_nano_banana_prompt(self) -> str:
        """
        Build prompt optimized for Nano Banana (Gemini) using the formula:
        [Action] + [Subject] + [Setting] + [Style] + [Technical Details]

        Uses spatial conjunctions (where, nearby, at the same time) to
        establish clear relationships between elements.

        Includes photographic realism details to avoid AI look.
        """
        brand_name = self._format_brand_name()
        # Build spatial relationships using conjunctions
        lighting_clause = (
            f"where {self.lighting.key_light} creates defining shadows"
        )
        background_clause = (
            f"nearby {self.scene.environment} elements softly out of focus"
        )
        # Person clause if applicable
        person_clause = ""
        if self.scene.human.include_person and self.scene.human.person_action:
            person_clause = (
                f", at the same time a person {self.scene.human.person_action} "
                "in the background"
            )
        # Build realism details (avoid AI look)
        realism_details = (
            f"Shot on {self.realism.camera.lens}, {self.realism.camera.depth_of_field}. "
            f"Contact shadows visible where product touches surface. "
            f"{self.realism.material.surface_texture}, {self.realism.material.dust_particles}. "
            f"{self.realism.material.film_grain}, {self.realism.lighting.shadow_consistency}."
        )
        # Build the prompt following the formula
        prompt = (
            # [Action]
            f"Place the {self.product.name} "
            # [Subject position with spatial conjunction]
            f"positioned at {self.scene.composition.product_position}, "
            f"occupying {self.scene.composition.product_size_pct}% of frame, "
            # [Setting with spatial conjunctions]
            f"{lighting_clause}, "
            f"{background_clause}{person_clause}. "
            # [Style]
            f"{self.scene.mood.capitalize()} {self.scene.time_of_day} atmosphere. "
            # [Technical Details with realism]
            f"Product {self.lighting.product_brightness}. "
            f"{brand_name} logo sharp and visible. "
            # [Realism details]
            f"{realism_details} "
            "8K resolution."
        )

        return prompt

    def build_nano_banana_system_prompt(self) -> str:
        """Build system prompt for Nano Banana generation."""
        brand_instruction = self._get_brand_instruction()
        placement_constraints = self._get_placement_constraints()

        return f"""You are a Nano Banana prompt generator for \
{self.product.name} marketing images.

THE NANO BANANA FORMULA:
[Action] + [Subject] + [Setting] + [Style] + [Technical Details]

PRODUCT: {self.product.name}
BRAND: {self.product.brand}
{self._format_physical_specs()}

{placement_constraints}

{brand_instruction}

SPATIAL KEYWORDS (use these to establish relationships):
- "where" - connects action to location
- "nearby" - establishes proximity
- "positioned at" - explicit placement
- "occupying X% of frame" - size specification

LIGHTING SETUP:
- Key: {self.lighting.key_light}
- Fill: {self.lighting.fill_light}
- Rim: {self.lighting.rim_light}
- Product brightness: {self.lighting.product_brightness}

CONSTRAINTS:
- Show ONLY ONE product (not multiple copies)
- Show the product as a COMPLETE, ASSEMBLED UNIT with all components connected
- DO NOT show only individual parts separately from the main product body
- All components should appear as one integrated, functional product
- Maximum {self.scene.composition.background_elements} background elements
- Preserve exact product angle from input image
- Logo must remain sharp and undistorted
"""

    def _get_brand_instruction(self) -> str:
        """Get brand casing instruction."""
        brand = self.product.brand_req.brand
        brand_style = self.product.brand_req.brand_style
        if brand_style == "lowercase":
            return f"""BRAND REQUIREMENTS:
- Brand name MUST appear as "{brand.lower()}" (lowercase only)
- Logo text must be sharp and readable
- Never obscure or distort brand elements"""
        if brand_style == "uppercase":
            return f"""BRAND REQUIREMENTS:
- Brand name MUST appear as "{brand.upper()}" (uppercase only)
- Logo text must be sharp and readable
- Never obscure or distort brand elements"""
        return f"""BRAND REQUIREMENTS:
- Brand name appears as "{brand}" (preserve exact casing)
- Logo text must be sharp and readable
- Never obscure or distort brand elements"""

    def _format_brand_name(self) -> str:
        """Format brand name according to style."""
        brand = self.product.brand_req.brand
        brand_style = self.product.brand_req.brand_style
        if brand_style == "lowercase":
            return brand.lower()
        if brand_style == "uppercase":
            return brand.upper()
        return brand

    def _format_physical_specs(self) -> str:
        """Format physical specifications if available."""
        specs = []
        if self.product.physical:
            if self.product.physical.dimensions:
                specs.append(f"- Dimensions: {self.product.physical.dimensions}")
            if self.product.physical.weight:
                specs.append(f"- Weight: {self.product.physical.weight}")
            if self.product.physical.size_reference:
                specs.append(f"- Size reference: {self.product.physical.size_reference}")
        return (
            "\n".join(specs)
            if specs
            else "- (Use input image for size reference)"
        )

    def _get_placement_constraints(self) -> str:
        """Get placement constraints if weight/surface rules exist."""
        if not self.product.physical or not self.product.physical.weight:
            return ""

        weight = self.product.physical.weight
        valid_surfaces = (
            self.product.placement.valid_surfaces
            if self.product.placement
            else ["floor", "ground"]
        )
        invalid_surfaces = (
            self.product.placement.invalid_surfaces
            if self.product.placement
            else ["counter", "table", "shelf"]
        )
        valid = ", ".join(valid_surfaces)
        invalid = ", ".join(invalid_surfaces)

        return f"""CRITICAL PLACEMENT CONSTRAINT:
- Product weighs {weight} - MUST be placed on: {valid}
- NEVER place on: {invalid}"""


def create_prompt_from_context(
    product_context: Dict[str, Any],
    scene_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a structured prompt from product and scene context.

    This is a convenience function for the pipeline.
    """
    brand_req = BrandRequirements(
        brand=product_context.get("brand", ""),
        brand_style=product_context.get("brand_display_style", "exact"),
    )
    physical = PhysicalSpecs(
        dimensions=product_context.get("dimensions"),
        weight=product_context.get("weight"),
        size_reference=product_context.get("size_reference"),
    )
    placement = PlacementConstraints()

    product_spec = ProductSpec(
        name=product_context.get("product_name", "product"),
        brand_req=brand_req,
        physical=physical,
        placement=placement,
    )

    scene_spec = None
    if scene_context:
        composition = CompositionSpec(
            product_position=scene_context.get("position", "foreground-center"),
            product_size_pct=scene_context.get("size_pct", 25),
        )
        human = HumanElementSpec(
            include_person=scene_context.get("include_person", False),
            person_action=scene_context.get("person_action"),
        )
        scene_spec = SceneSpec(
            environment=scene_context.get("environment", "studio"),
            time_of_day=scene_context.get("time_of_day", "daylight"),
            mood=scene_context.get("mood", "aspirational"),
            composition=composition,
            human=human,
        )

    builder = PromptBuilder(product_spec, scene_spec)
    return builder.build_combined_prompt()


# Pre-defined scene templates for common use cases
SCENE_TEMPLATES = {
    "lifestyle_home": SceneSpec(
        environment="modern living room with natural light",
        time_of_day="daylight",
        mood="cozy, aspirational",
        composition=CompositionSpec(
            product_position="foreground-left",
            product_size_pct=25,
        ),
        human=HumanElementSpec(
            include_person=True,
            person_action="using product naturally",
        ),
    ),
    "hero_studio": SceneSpec(
        environment="clean studio with gradient background",
        time_of_day="studio lighting",
        mood="professional, premium",
        composition=CompositionSpec(
            product_position="foreground-center",
            product_size_pct=30,
        ),
        human=HumanElementSpec(include_person=False),
    ),
    "outdoor_active": SceneSpec(
        environment="outdoor natural setting",
        time_of_day="golden_hour",
        mood="adventurous, aspirational",
        composition=CompositionSpec(
            product_position="foreground-center",
            product_size_pct=25,
        ),
        human=HumanElementSpec(
            include_person=True,
            person_action="enjoying outdoor activity",
        ),
    ),
    "emergency_power": SceneSpec(
        environment="home during power outage, warm lamp light",
        time_of_day="evening",
        mood="secure, reliable",
        composition=CompositionSpec(
            product_position="foreground-center",
            product_size_pct=25,
        ),
        human=HumanElementSpec(
            include_person=True,
            person_action="relying on product for power",
        ),
    ),
}
