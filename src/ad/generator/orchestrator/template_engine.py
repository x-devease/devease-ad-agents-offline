"""
Template Engine: Render P0 Master Mask template with mapped values.

This module handles template rendering with validation and error handling.

The P0 Master Mask template is designed to be compatible with Nano Banana
(Gemini) image generation models. For prompt engineering best practices,
see nano_banana_guide.md.

Reference: https://ai.google.dev/gemini-api/docs/image-generation#prompt-guide
"""

import logging
import re
from typing import Dict, Optional


logger = logging.getLogger(__name__)
# P0 Master Mask Template (from li_suggestion_v1.md)
# Compatible with Nano Banana (Gemini) prompt format:
# [Action] + [Subject] + [Setting] + [Style] + [Technical Details]
# Updated to include product completeness description
P0_MASTER_MASK = (
    "Professional studio photography of {subject_description}, {material_finish}. "
    "{interaction_context} "
    "Layout: {product_position}, {product_visibility} view. "
    "Background: {static_context}. "
    "Lighting: {color_balance} with {brightness_distribution} gradient. "
    "Focal Point: {visual_prominence}. "
    "Constraint: Based on Image 1, maintain exact geometric structure. "
    "Color Constraint: {color_constraint}. "
    "Product Completeness: {completeness_instruction}."
)
# Lean Anchor template: Trust pixels, not words. Focus on what's NOT in source image.
# Structure: [Anchor] -> [Core CMF] -> [Lifestyle] -> [Ad Logic]
# Target: < 600 characters
P0_LEAN_MASK = (
    "Strictly maintain the exact geometric structure and proportions of Image 1. "
    "Professional photography of {subject_description}. "
    "Physical: {grounding_instruction}. "
    "CMF: {material_finish}; {color_constraint}. "
    "Layout: {product_position}. "
    "Interaction: {interaction_context}. "
    "Context: {static_context} with {color_balance}."
)
# Enhanced ROAS template: Expanded feature injection while maintaining anchor integrity
# Structure: [Constraints] -> [Anchor] -> [Physical] -> [CMF] -> [Focus] ->
# [Layout+Composition] -> [Interaction] -> [Context+Environment]
# HIGH-FIDELITY: Anti-hallucination, camera specs, material textures,
# three-point lighting, shadow spec, post-processing, frame occupancy
# Note: Template uses conditional formatting to handle empty values gracefully
# Note: This template is kept for backward compatibility but
# tri-template architecture is preferred
ENHANCED_ROAS_MASK = (
    "{global_view_definition} of {subject_description}, strictly maintaining "
    "the color accuracy, metallic textures, and geometric profile of the "
    "provided reference image. "
    "CRITICAL ANTI-HALLUCINATION: Do NOT add elements not in Image 1. "
    "Do NOT create variations, redesigns, or add extra text/accessories. "
    "ONLY change background/scene - product 100% identical to Image 1. "
    "Physical: mop shown in realistic leaning position against "
    "{placement_target}, firmly grounded with contact shadows. "
    "Shadow specification: Contact shadows hard, dark; cast shadows soft, key light (45°). "
    "CMF: {material_finish}; {color_constraint}. "
    "Material textures: Visible surface grain, micro-imperfections, anisotropic highlights. "
    "Color accuracy: Match product colors exactly as in Image 1, <2% tolerance. "
    "[Focus] Shot on Canon EOS R5 full-frame, 85mm f/1.4, ISO 100, f/8 deep focus. "
    "Depth of field: Product f/8; midground f/4; background f/2.8 bokeh. "
    "Layout: {product_position}{composition_style_suffix}{frame_occupancy}. "
    "Professional three-point lighting: Key (45° top-right, 1.5 stops), "
    "Fill (45° top-left, 0.5 stops), Rim (back-left, 1 stop). "
    "Post-processing: Subtle sharpening, S-curve contrast. "
    "Interaction: {interaction_context}. "
    "Context: {static_context} with {color_balance}"
    "{lighting_detail_suffix}{environment_objects_suffix}."
)
# Tri-Template Architecture: Branch-specific optimized templates
# Template 1: WIDE_SCENE (10.7 ROAS Hero - golden_ratio)
# Target: Maximum environment integration for lifestyle ads
# Image: 侧俯.png or 右侧45.png
# CONSISTENCY-FIRST: [Constraints] and [Product] at top to prioritize
# geometry and product integrity
# HIGH-FIDELITY: Camera specs, material textures, anti-hallucination,
# three-point lighting, depth layering, shadow spec, post-processing,
# frame occupancy, composition/visual flow (HIGH_FIDELITY_IMPROVEMENTS)
# STRICT MODULAR TEMPLATE: [Constraints] -> [Product] -> [Scene Overview] ->
# [CMF] -> [Focus] -> [Physical] -> [Layout] -> [Interaction] -> [Environment]
# CMF_Core must be at position 200-300 for subject primacy
WIDE_SCENE = (
    "[Constraints] Strictly maintain the exact geometric structure and proportions of Image 1. "
    "Product integrity is non-negotiable. The product MUST be properly "
    "grounded with visible contact shadows. "
    "Preserve ALL components exactly as in Image 1: same design, same shape, "
    "same features, same component layout. "
    "Preserve ALL product text, logos, labels, and brand identifiers exactly "
    "as in Image 1 - NO blur, NO distortion, NO missing letters. "
    "ALL text must be sharp and readable at 100% zoom with contrast ratio > 4.5:1. "
    "Preserve ALL product colors exactly as in Image 1 - NO color shifts, "
    "NO tinting, NO alterations. ΔE (Delta E) < 2.0 - imperceptible difference. "
    "Metamerism: Colors match under D50 illuminant (standard viewing conditions). "
    "Do NOT redesign, modify, or create variations of the product. "
    "Show ONLY ONE product instance - no duplicates, no additional products, "
    "no accessories. "
    "CRITICAL ANTI-HALLUCINATION: Do NOT add any elements not visible in Image 1. "
    "Do NOT create product variations, redesigns, or modifications. "
    "Do NOT add extra text, labels, or branding not in source. "
    "Do NOT change product proportions, component positions, or assembly. "
    "Do NOT add accessories, props, or decorative elements not in source. "
    "Do NOT modify product colors, materials, or finishes. "
    "Material appearance must match source exactly - NO substitutions. "
    "ONLY change background/scene - product must remain 100% identical to Image 1. "
    "[Product] {subject_description}. "
    "[Scene Overview] {scene_overview} "
    "[CMF] {cmf_core}{metallic_texture_enhancement}. "
    "Material textures: Visible surface grain, subtle micro-imperfections, "
    "realistic material properties. Metallic surfaces show anisotropic highlights "
    "with directional reflection patterns. Avoid uniform, plastic-like surfaces. "
    "Include subtle dust particles in light, natural surface variations, "
    "slight film grain for photographic realism. "
    "Color accuracy: Match ALL product colors exactly as in Image 1 with <2% tolerance. "
    "NO color tinting, NO white balance shifts, NO saturation changes. "
    "Product colors must remain 100% accurate regardless of lighting temperature. "
    "[Focus] Shot on Canon EOS R5 full-frame camera with 85mm f/1.4 lens. "
    "ISO 100, f/8 aperture for deep focus ensuring all text, labels, "
    "metallic textures, and component details are sharp and crisp. "
    "Background: f/2.8 with natural bokeh (polygonal aperture shapes). "
    "Depth of field layering: Foreground (product) f/8 perfect focus; "
    "midground (supporting elements) f/4, 70% sharpness; "
    "background (environment) f/2.8, soft bokeh, 30% sharpness. "
    "Lens-compressed background perspective. Product in perfect focus, "
    "background softly blurred for depth separation. "
    "[Physical] {physical_state_description}. "
    "Shadow specification: Contact shadows hard, dark (RGB 20,20,20), "
    "defined edges where product touches. Cast shadows soft, medium gray "
    "(RGB 80,80,80), direction from key light (45°). Shadow falloff: "
    "natural exponential decay, no hard cutoffs. "
    "[Layout] {product_position}{composition_style_suffix}{frame_occupancy}. "
    "Composition: Product at primary focal point (highest visual weight). "
    "Visual flow: Z-pattern eye tracking from top-left → product → bottom-right. "
    "[Interaction] {interaction_context}{interaction_scene_enhancement}. "
    "[Environment] {static_context} with {atmosphere_description}. "
    "Professional three-point lighting: Key light (45° top-right, 1.5 stops brighter "
    "than ambient), Fill light (45° top-left, 0.5 stops), "
    "Rim light (back-left, 1 stop for edge separation). "
    "Product illuminated 1.5 stops brighter than background for clear separation. "
    "Contact shadows: Dark, defined where product touches (darker than cast shadows). "
    "Physical indentations visible where weight is applied. "
    "{lighting_enhancement}{lighting_detail_suffix}. "
    "CRITICAL: Completely transparent lighting aesthetic with highlights overflowing, "
    "creating a luminous glow that spills over edges. "
    "Vivid high-dynamic-range (HDR) rendering with expanded contrast; "
    "crush the blacks slightly and boost the peak whites for a punchy look. "
    "Vivid-pop color science achieving a high-gloss commercial finish. "
    "Post-processing: Professional commercial retouching with subtle sharpening "
    "(Unsharp Mask 150%, 1.0px, 0 threshold). Color grading: S-curve for contrast, "
    "slight vibrance boost, selective saturation on product vs background. "
    "NO studio equipment, NO visible soft-boxes, NO photographic backdrops. "
    "NO color tinting - product colors must remain accurate and unchanged. "
    "{environment_objects_suffix}. {render_quality} "
    "Completely transparent lighting with highlights overflowing, luminous glow spilling over edges."
)
# Template 2: MACRO_DETAIL (5.15 ROAS Close-up - high_efficiency)
# Target: Extreme focus on mechanics and CMF textures
# CONSISTENCY-FIRST: [Constraints] and [Product] at top to prioritize
# geometry and product integrity
# HIGH-FIDELITY: Anti-hallucination, camera specs, material textures,
# color accuracy, shadow spec, post-processing (HIGH_FIDELITY_IMPROVEMENTS)
# SCENE-FIRST NARRATIVE: [Constraints] -> [Product] -> [Scene Overview] ->
# [Action] -> [Product Consistency] -> [Technical] -> [Aesthetics]
# FEATURE-EMBEDDED: Keywords mapped to Feature Tracking registry for
# extractor visibility
# CRITICAL: Explicit Feature Exclusion List - strips Person, Background
# Decor, Layout even if Formula requests them
# Image: 侧俯.png (cropped to base) - clean geometry for macro detail
MACRO_DETAIL = (
    "[Constraints] {geometric_constraint} "
    "Product integrity is non-negotiable. Firm contact shadows define the "
    "unit's shape and grounding. "
    "Preserve ALL components exactly as in Image 1: same design, same shape, "
    "same features, same component layout. "
    "Preserve ALL product text, logos, labels, and brand identifiers exactly "
    "as in Image 1 - NO blur, NO distortion, NO missing letters. "
    "ALL text must be sharp and readable at 100% zoom with contrast ratio > 4.5:1. "
    "Preserve ALL product colors exactly as in Image 1 - NO color shifts, "
    "NO tinting, NO alterations. ΔE (Delta E) < 2.0 - imperceptible difference. "
    "Metamerism: Colors match under D50 illuminant (standard viewing conditions). "
    "Do NOT redesign, modify, or create variations of the product. "
    "Show ONLY ONE product instance - no duplicates, no additional products, "
    "no accessories. "
    "CRITICAL ANTI-HALLUCINATION: Do NOT add any elements not visible in Image 1. "
    "Do NOT create product variations, redesigns, or modifications. "
    "Do NOT add extra text, labels, or branding not in source. "
    "Do NOT change product proportions, component positions, or assembly. "
    "Do NOT add accessories, props, or decorative elements not in source. "
    "Do NOT modify product colors, materials, or finishes. "
    "Material appearance must match source exactly - NO substitutions. "
    "ONLY change background/scene - product must remain 100% identical to Image 1. "
    "[Product] {subject_description}. "
    "[Scene Overview] {scene_overview} "
    "[Action] {action_description} "
    "[Product Consistency] {consistency_anchor}. "
    "[Technical] Shot on Canon EOS R5 full-frame, 85mm f/1.4, ISO 100, f/8 deep focus. "
    "Material textures: Visible surface grain, subtle micro-imperfections, "
    "anisotropic highlights. Avoid plastic-like surfaces. Slight film grain. "
    "Color accuracy: Match product colors exactly as in Image 1, <2% tolerance. "
    "Shadow specification: Contact shadows hard, dark; cast shadows soft, "
    "direction from key light (45°). Natural shadow falloff. "
    "Post-processing: Subtle sharpening, S-curve contrast, selective saturation. "
    "[Aesthetics] {aesthetic_polish}"
)
# Template 3: FLAT_TECH (8.34 ROAS Differentiator - cool_peak)
# Target: Demonstrating 180-degree flat-lay capability
# CONSISTENCY-FIRST: [Constraints] and [Product] at top to prioritize
# geometry and product integrity
# HIGH-FIDELITY: Anti-hallucination, camera specs, material textures,
# three-point lighting, depth layering, shadow spec, post-processing,
# frame occupancy, composition (HIGH_FIDELITY_IMPROVEMENTS)
# STRICT MODULAR TEMPLATE: [Constraints] -> [Product] -> [Scene Overview] ->
# [CMF] -> [Focus] -> [Physical] -> [Layout] -> [Environment]
# Physical Logic: Hardcoded flat-lay state (no leaning/standing conflict)
# Image: 180躺平.png
FLAT_TECH = (
    "[Constraints] Strictly maintain the exact geometric structure and proportions of Image 1. "
    "Product integrity is non-negotiable. Firm contact shadows define the "
    "unit's shape and grounding. "
    "Preserve ALL components exactly as in Image 1: same design, same shape, "
    "same features, same component layout. "
    "Preserve ALL product text, logos, labels, and brand identifiers exactly "
    "as in Image 1 - NO blur, NO distortion, NO missing letters. "
    "Preserve ALL product colors exactly as in Image 1 - NO color shifts, "
    "NO tinting, NO alterations. "
    "Do NOT redesign, modify, or create variations of the product. "
    "Show ONLY ONE product instance - no duplicates, no additional products, "
    "no accessories. "
    "CRITICAL ANTI-HALLUCINATION: Do NOT add any elements not visible in Image 1. "
    "Do NOT create product variations, redesigns, or modifications. "
    "Do NOT add extra text, labels, or branding not in source. "
    "Do NOT change product proportions, component positions, or assembly. "
    "Do NOT add accessories, props, or decorative elements not in source. "
    "Do NOT modify product colors, materials, or finishes. "
    "ONLY change background/scene - product must remain 100% identical to Image 1. "
    "[Product] {subject_description}. "
    "[Scene Overview] {scene_overview} "
    "[CMF] {cmf_core}{metallic_texture_enhancement}. "
    "Material textures: Visible surface grain, subtle micro-imperfections, "
    "anisotropic highlights. Avoid plastic-like surfaces. Slight film grain. "
    "Color accuracy: Match product colors exactly as in Image 1, <2% tolerance. "
    "[Focus] Shot on Canon EOS R5 full-frame, 85mm f/1.4, ISO 100, f/8 deep focus. "
    "Depth of field: Product f/8 perfect focus; midground f/4; background f/2.8 bokeh. "
    "[Physical] {physical_state_description}. "
    "Shadow specification: Contact shadows hard, dark; cast shadows soft, key light (45°). "
    "[Layout]{composition_style_suffix}{frame_occupancy}. "
    "Visual flow: Z-pattern eye tracking; product at primary focal point. "
    "[Environment] {static_context} with {atmosphere_description}. "
    "Professional three-point lighting: Key (45° top-right, 1.5 stops), "
    "Fill (45° top-left, 0.5 stops), Rim (back-left, 1 stop). "
    "Product 1.5 stops brighter than background. "
    "{lighting_enhancement}{lighting_detail_suffix}{environment_objects_suffix}. "
    "CRITICAL: Completely transparent lighting aesthetic with highlights "
    "overflowing, luminous glow that spills over edges. "
    "Vivid high-dynamic-range (HDR) rendering; crush blacks, boost peak whites. "
    "Post-processing: Subtle sharpening, S-curve contrast, selective saturation. "
    "NO studio equipment, NO visible soft-boxes, NO photographic backdrops. "
    "NO color tinting - product colors must remain accurate and unchanged. "
    "{render_quality} Completely transparent lighting with highlights "
    "overflowing, luminous glow spilling over edges."
)


class TemplateEngine:
    """
    Renders P0 Master Mask template with provided values.

    All templates are professional quality with optional sections controlled
    by feature flags in PromptBuilderConfig.

    Handles:
    - Template rendering with Python str.format()
    - Placeholder validation
    - Error handling for missing/invalid placeholders
    """

    def __init__(
        self,
        template: Optional[str] = None,
        branch_name: Optional[str] = None,
    ):
        """
        Initialize template engine.

        Args:
            template: Template string (uses WIDE_SCENE if None)
            branch_name: Branch identifier for tri-template selection:
                - "golden_ratio" -> WIDE_SCENE (hero lifestyle, 10.7x ROAS)
                - "high_efficiency" -> MACRO_DETAIL (close-up, 5.15x ROAS)
                - "cool_peak" -> FLAT_TECH (flat-lay, 8.34x ROAS)
                - None -> WIDE_SCENE (default)
        """
        if template is None:
            # Tri-template architecture: branch-specific templates
            if branch_name == "golden_ratio":
                self.template = WIDE_SCENE
            elif branch_name == "high_efficiency":
                self.template = MACRO_DETAIL
            elif branch_name == "cool_peak":
                self.template = FLAT_TECH
            else:
                # Default to wide scene template
                self.template = WIDE_SCENE
        else:
            self.template = template
        self.required_placeholders = self._extract_placeholders()

    def _extract_placeholders(self) -> set:
        """
        Extract placeholder names from template.

        Returns:
            Set of placeholder names (without braces)
        """
        # Find all {placeholder_name} patterns
        pattern = r"\{(\w+)\}"
        placeholders = set(re.findall(pattern, self.template))
        return placeholders

    def render(
        self,
        values: Dict[str, str],
        strict: bool = True,
    ) -> str:
        """
        Render template with provided values.

        Args:
            values: Dictionary mapping placeholder names to values
            strict: If True, raise error for missing required placeholders.
                   If False, use placeholder name as fallback.

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If strict=True and required placeholders are missing
            KeyError: If template format is invalid
        """
        # Validate all required placeholders are provided
        missing = self.required_placeholders - set(values.keys())

        if missing and strict:
            raise ValueError(
                f"Missing required placeholders: {missing}. "
                f"Provided: {list(values.keys())}"
            )

        if missing:
            logger.warning(
                "Missing placeholders (using names as fallback): %s",
                missing,
            )
            # Use placeholder name as fallback
            for placeholder in missing:
                values[placeholder] = f"{{{placeholder}}}"

        try:
            rendered = self.template.format(**values)
            logger.debug(
                "Template rendered successfully (%d chars)", len(rendered)
            )
            return rendered
        except KeyError as e:
            # This shouldn't happen if validation passed, but handle gracefully
            logger.error("Template rendering error: %s", e)
            raise ValueError(
                f"Template rendering failed: missing placeholder {e}"
            ) from e
        except ValueError as e:
            # Format error (e.g., invalid format specifier)
            logger.error("Template format error: %s", e)
            raise ValueError(f"Template format error: {e}") from e

    def validate_values(self, values: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate that all required placeholders have values.

        Args:
            values: Dictionary mapping placeholder names to values

        Returns:
            Dictionary mapping placeholder names to validation status (True/False)
        """
        validation = {}
        for placeholder in self.required_placeholders:
            has_value = (
                placeholder in values
                and values[placeholder] is not None
                and str(values[placeholder]).strip() != ""
            )
            validation[placeholder] = has_value

        return validation
