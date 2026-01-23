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
# V2 Enhanced ROAS template: Expanded feature injection while maintaining anchor integrity
# Structure: [Anchor] -> [Physical] -> [CMF] -> [Layout+Composition] ->
# [Interaction] -> [Context+Environment]
# New placeholders: placement_target, composition_style, lighting_detail, environment_objects
# Note: Template uses conditional formatting to handle empty values gracefully
# Note: This template is kept for backward compatibility but
# tri-template architecture is preferred
P0_V2_ENHANCED_ROAS_MASK = (
    "{global_view_definition} of {subject_description}, strictly maintaining "
    "the color accuracy, metallic textures, and geometric profile of the "
    "provided reference image. "
    "Physical: mop shown in realistic leaning position against "
    "{placement_target}, firmly grounded with contact shadows. "
    "CMF: {material_finish}; {color_constraint}. "
    "Layout: {product_position}{composition_style_suffix}. "
    "Interaction: {interaction_context}. "
    "Context: {static_context} with {color_balance}"
    "{lighting_detail_suffix}{environment_objects_suffix}."
)
# Tri-Template Architecture: Branch-specific optimized templates
# Template 1: V2_WIDE_SCENE (10.7 ROAS Hero - golden_ratio)
# Target: Maximum environment integration for lifestyle ads
# Image: 侧俯.png or 右侧45.png
# CONSISTENCY-FIRST: [Constraints] and [Product] at top to prioritize
# geometry and product integrity
# STRICT MODULAR TEMPLATE: [Constraints] -> [Product] -> [Scene Overview] ->
# [CMF] -> [Physical] -> [Layout] -> [Interaction] -> [Environment]
# CMF_Core must be at position 200-300 for subject primacy
P0_V2_WIDE_SCENE = (
    "[Constraints] Strictly maintain the exact geometric structure and proportions of Image 1. "
    "Product integrity is non-negotiable. The product MUST be properly "
    "grounded with visible contact shadows. "
    "Preserve ALL components exactly as in Image 1: same design, same shape, "
    "same features, same component layout. "
    "Preserve ALL product text, logos, labels, and brand identifiers exactly "
    "as in Image 1 - NO blur, NO distortion, NO missing letters. "
    "Preserve ALL product colors exactly as in Image 1 - NO color shifts, "
    "NO tinting, NO alterations. "
    "Do NOT redesign, modify, or create variations of the product. "
    "Show ONLY ONE product instance - no duplicates, no additional products, "
    "no accessories. "
    "[Product] {subject_description}. "
    "[Scene Overview] {scene_overview} "
    "[CMF] {cmf_core}{metallic_texture_enhancement}. "
    "[Focus] Product in deep focus (f/8+) ensuring all text, labels, "
    "metallic textures, and component details are sharp and crisp. "
    "[Physical] {physical_state_description}. "
    "[Layout] {product_position}{composition_style_suffix}. "
    "[Interaction] {interaction_context}{interaction_scene_enhancement}. "
    "[Environment] {static_context} with {atmosphere_description}, "
    "{lighting_enhancement}{lighting_detail_suffix}. CRITICAL: Completely "
    "transparent lighting aesthetic with highlights overflowing, creating a "
    "luminous glow that spills over edges. Vivid high-dynamic-range (HDR) "
    "rendering with expanded contrast; crush the blacks slightly and boost "
    "the peak whites for a punchy look. Vivid-pop color science achieving a "
    "high-gloss commercial finish. NO studio equipment, NO visible "
    "soft-boxes, NO photographic backdrops. NO color tinting - product "
    "colors must remain accurate and unchanged."
    "{environment_objects_suffix}. {render_quality} Completely transparent "
    "lighting with highlights overflowing, luminous glow spilling over edges."
)
# Template 2: V2_MACRO_DETAIL (5.15 ROAS Close-up - high_efficiency)
# Target: Extreme focus on mechanics and CMF textures
# CONSISTENCY-FIRST: [Constraints] and [Product] at top to prioritize
# geometry and product integrity
# SCENE-FIRST NARRATIVE: [Constraints] -> [Product] -> [Scene Overview] ->
# [Action] -> [Product Consistency] -> [Aesthetics]
# FEATURE-EMBEDDED: Keywords mapped to Feature Tracking registry for
# extractor visibility
# CRITICAL: Explicit Feature Exclusion List - strips Person, Background
# Decor, Layout even if Formula requests them
# Image: 侧俯.png (cropped to base) - clean geometry for macro detail
P0_V2_MACRO_DETAIL = (
    "[Constraints] {geometric_constraint} "
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
    "[Product] {subject_description}. "
    "[Scene Overview] {scene_overview} "
    "[Action] {action_description} "
    "[Product Consistency] {consistency_anchor}. "
    "[Aesthetics] {aesthetic_polish}"
)
# Template 3: V2_FLAT_TECH (8.34 ROAS Differentiator - cool_peak)
# Target: Demonstrating 180-degree flat-lay capability
# CONSISTENCY-FIRST: [Constraints] and [Product] at top to prioritize
# geometry and product integrity
# STRICT MODULAR TEMPLATE: [Constraints] -> [Product] -> [Scene Overview] ->
# [CMF] -> [Physical] -> [Layout] -> [Environment]
# Physical Logic: Hardcoded flat-lay state (no leaning/standing conflict)
# Image: 180躺平.png
P0_V2_FLAT_TECH = (
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
    "[Product] {subject_description}. "
    "[Scene Overview] {scene_overview} "
    "[CMF] {cmf_core}{metallic_texture_enhancement}. "
    "[Physical] {physical_state_description}. "
    "[Layout]{composition_style_suffix}. "
    "[Environment] {static_context} with {atmosphere_description}, "
    "{lighting_enhancement}{lighting_detail_suffix}{environment_objects_suffix}. "
    "CRITICAL: Completely transparent lighting aesthetic with highlights "
    "overflowing, creating a luminous glow that spills over edges. "
    "Vivid high-dynamic-range (HDR) rendering with expanded contrast; "
    "crush the blacks slightly and boost the peak whites for a punchy look. "
    "Vivid-pop color science achieving a high-gloss commercial finish. "
    "NO studio equipment, NO visible soft-boxes, NO photographic backdrops. "
    "NO color tinting - product colors must remain accurate and unchanged. "
    "{render_quality} Completely transparent lighting with highlights "
    "overflowing, luminous glow spilling over edges."
)


class TemplateEngine:
    """
    Renders P0 Master Mask template with provided values.

    Handles:
    - Template rendering with Python str.format()
    - Placeholder validation
    - Error handling for missing/invalid placeholders
    """

    def __init__(
        self,
        template: Optional[str] = None,
        lean_mode: bool = False,
        v2_mode: bool = False,
        branch_name: Optional[str] = None,
    ):
        """
        Initialize template engine.

        Args:
            template: Template string (uses P0_MASTER_MASK if None)
            lean_mode: If True, use P0_LEAN_MASK instead
            v2_mode: If True, use V2 template (overrides lean_mode)
            branch_name: Branch identifier for tri-template selection:
                - "golden_ratio" -> V2_WIDE_SCENE
                - "high_efficiency" -> V2_MACRO_DETAIL
                - "cool_peak" -> V2_FLAT_TECH
        """
        if template is None:
            if v2_mode and branch_name:
                # Tri-template architecture: branch-specific templates
                if branch_name == "golden_ratio":
                    self.template = P0_V2_WIDE_SCENE
                elif branch_name == "high_efficiency":
                    self.template = P0_V2_MACRO_DETAIL
                elif branch_name == "cool_peak":
                    self.template = P0_V2_FLAT_TECH
                else:
                    # Fallback to legacy V2 template
                    self.template = P0_V2_ENHANCED_ROAS_MASK
            elif v2_mode:
                # Legacy V2 mode (no branch specified)
                self.template = P0_V2_ENHANCED_ROAS_MASK
            elif lean_mode:
                self.template = P0_LEAN_MASK
            else:
                self.template = P0_MASTER_MASK
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
