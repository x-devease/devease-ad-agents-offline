"""
Natural Language Prompt Builder for Nano Banana Pro Agent.

Converts generic prompts into conversational, descriptive natural language
following Nano Banana Pro best practices.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from src.agents.nano.core.types import (
    PromptCategory,
    PromptIntent,
    ProductContext,
    BrandGuidelines,
    IntermediatePrompt,
)


logger = logging.getLogger(__name__)


class NaturalLanguagePromptBuilder:
    """
    Build conversational, descriptive natural language prompts.

    Converts "make an ad" into detailed, natural language descriptions
    that Nano Banana Pro understands best.
    """

    # Natural language vocabulary for describing concepts
    VOCABULARY = {
        "lighting": {
            "natural": "soft, diffused natural light",
            "studio": "professional studio lighting with soft boxes",
            "golden_hour": "warm golden hour light streaming through windows",
            "dramatic": "dramatic lighting with strong shadows",
            "bright": "bright, cheerful lighting filling the scene",
        },
        "mood": {
            "professional": "polished, professional appearance",
            "authentic": "authentic, lived-in feeling",
            "aspirational": "elevated, aspirational quality",
            "friendly": "warm, approachable atmosphere",
            "urgent": "energetic, immediate feeling",
        },
        "composition": {
            "centered": "positioned at the center of the frame",
            "off_center": "positioned slightly off-center for visual interest",
            "foreground": "in the foreground, prominently featured",
            "background": "in the background, providing context",
        },
        "quality": {
            "photorealistic": "photorealistic rendering with accurate materials",
            "high_quality": "professional quality with fine details",
            "detailed": "intricate details and textures visible",
        },
    }

    def __init__(self):
        """Initialize the natural language builder."""
        pass

    def build(
        self,
        agent_input,
        category: PromptCategory,
        intent: PromptIntent,
        analysis: Dict[str, Any],
    ) -> IntermediatePrompt:
        """
        Build natural language prompt from generic input.

        Args:
            agent_input: The enriched agent input
            category: Detected prompt category
            intent: Detected intent
            analysis: Deep analysis results

        Returns:
            IntermediatePrompt with natural language content
        """

        # Start with the core description
        description = self._build_core_description(
            agent_input, intent, analysis
        )

        # Add visual details
        visual_details = self._add_visual_details(
            agent_input, intent, analysis
        )

        # Add environmental context
        environment = self._add_environment(
            agent_input, intent, analysis
        )

        # Combine into full prompt
        prompt_content = self._combine_sections([
            description,
            visual_details,
            environment,
        ])

        logger.info(f"Built natural language prompt ({len(prompt_content)} chars)")

        return IntermediatePrompt(
            stage="natural_language_build",
            prompt_content=prompt_content,
            metadata={
                "category": category.value,
                "intent": intent.value,
                "sections": ["description", "visual_details", "environment"],
            },
        )

    def _build_core_description(self, agent_input, intent: PromptIntent, analysis: Dict) -> str:
        """Build the core description section."""

        parts = []

        # Product description
        if agent_input.product_context:
            product_desc = self._describe_product(
                agent_input.product_context, intent
            )
            parts.append(product_desc)

        # Human element (if applicable)
        if analysis.get("needs_human_element"):
            human_desc = self._describe_human_element(intent)
            parts.append(human_desc)

        # Action/scene
        action_desc = self._describe_action(intent, agent_input.generic_prompt)
        parts.append(action_desc)

        return " ".join(parts)

    def _describe_product(self, product_context: ProductContext, intent: PromptIntent) -> str:
        """Describe the product in natural language."""

        if intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
            # For product photos, be precise and technical but natural
            return (
                f"A professional product photograph showing the {product_context.name} "
                f"in detail. The {product_context.name} displays {self._list_features(product_context.key_features)}. "
                f"Materials include {', '.join(product_context.materials[:-1])} "
                f"and {product_context.materials[-1] if product_context.materials else ''} "
                f"with accurate color representation of {', '.join(product_context.colors[:3])}."
            )

        elif intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            # For lifestyle ads, focus on benefits and context
            return (
                f"The {product_context.name} sits naturally in a home environment, "
                f"showcasing its {product_context.key_features[0] if product_context.key_features else 'design'} "
                f"in an authentic setting."
            )

        else:
            # Generic description
            return (
                f"The {product_context.name} features "
                f"{', '.join(product_context.key_features[:2] if product_context.key_features else [])}, "
                f"displaying {', '.join(product_context.materials)} construction."
            )

    def _list_features(self, features: List[str]) -> str:
        """List features naturally in a sentence."""

        if len(features) == 0:
            return "its design"

        elif len(features) == 1:
            return features[0]

        elif len(features) == 2:
            return f"{features[0]} and {features[1]}"

        else:
            # Oxford comma style
            return ', '.join(features[:-1]) + f", and {features[-1]}"

    def _describe_human_element(self, intent: PromptIntent) -> str:
        """Describe the human element naturally."""

        if intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            return "A person interacts naturally with the product, their expression showing genuine satisfaction and comfort."

        return "A person appears in the scene, adding human scale and relatability."

    def _describe_action(self, intent: PromptIntent, original_prompt: str) -> str:
        """Describe what's happening in the scene."""

        prompt_lower = original_prompt.lower()

        if "clean" in prompt_lower or "mop" in prompt_lower:
            return "The scene captures a moment of cleaning action, with the product in use."

        elif "happy" in prompt_lower or "satisfied" in prompt_lower:
            return "The composition conveys a sense of satisfaction and success."

        elif "compare" in prompt_lower or "versus" in prompt_lower:
            return "The arrangement clearly shows comparative differences between items."

        return "The scene presents the subject in a clear, engaging manner."

    def _add_visual_details(self, agent_input, intent: PromptIntent, analysis: Dict) -> str:
        """Add specific visual details to the prompt."""

        details = []

        # Materiality
        if agent_input.product_context and agent_input.product_context.materials:
            details.append(
                f"Material textures are clearly visible: "
                f"{self._describe_materials(agent_input.product_context.materials)}."
            )

        # Colors
        if agent_input.product_context and agent_input.product_context.colors:
            details.append(
                f"Colors are accurate and vibrant: "
                f"{', '.join(agent_input.product_context.colors[:3])}."
            )

        # Brand colors
        if agent_input.brand_guidelines and agent_input.brand_guidelines.primary_colors:
            details.append(
                f"Brand colors {', '.join(agent_input.brand_guidelines.primary_colors[:2])} "
                f"are prominent in the composition."
            )

        # Quality
        details.append("The rendering is photorealistic with fine details and textures.")

        return " ".join(details) if details else ""

    def _describe_materials(self, materials: List[str]) -> str:
        """Describe materials naturally."""

        descriptions = {
            "microfiber": "soft, absorbent microfiber with visible texture",
            "plastic": "smooth plastic with subtle surface variations",
            "metal": "polished metal with realistic reflections",
            "wood": "natural wood grain with visible texture",
            "cotton": "natural cotton fabric with soft weave",
        }

        material_descs = []
        for material in materials:
            material_lower = material.lower()
            if material_lower in descriptions:
                material_descs.append(descriptions[material_lower])
            else:
                material_descs.append(f"{material} material with realistic appearance")

        return ", ".join(material_descs)

    def _add_environment(self, agent_input, intent: PromptIntent, analysis: Dict) -> str:
        """Add environmental context to the prompt."""

        env_parts = []

        # Setting with rich background details
        if intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            env_parts.append(
                "The setting is a modern, clean home environment that feels authentic and lived-in."
            )
            # Add more lifestyle background details
            env_parts.append(
                "Soft natural light creates gentle shadows, adding depth and dimensionality."
            )

        elif intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
            # Enhanced background for product photography
            env_parts.append(
                "The background features a polished white marble surface with subtle grey veining patterns."
            )
            env_parts.append(
                "Luxurious marble backdrop extends seamlessly to edges, creating an elegant setting."
            )
            env_parts.append(
                "Soft reflections are visible on the marble surface, adding sophistication."
            )
            env_parts.append(
                "A soft contact shadow grounds the product, while gentle vignette at edges frames the composition."
            )

        # Lighting with details
        lighting = self.VOCABULARY["lighting"].get("natural", "soft, natural light")
        env_parts.append(f"Lighting is {lighting}.")

        # Mood
        if agent_input.emotion_goal:
            mood = agent_input.emotion_goal
        else:
            mood = self.VOCABULARY["mood"].get("professional", "polished and professional")

        env_parts.append(f"The overall mood is {mood}.")

        return " ".join(env_parts)

    def _combine_sections(self, sections: List[str]) -> str:
        """Combine sections into a coherent prompt."""

        # Filter out empty sections
        sections = [s for s in sections if s.strip()]

        # Join with paragraph breaks
        return "\n\n".join(sections)

    def enhance_for_nano_banana(self, prompt: str) -> str:
        """
        Enhance a prompt specifically for Nano Banana Pro.

        Applies NB-specific optimizations:
        - Conversational tone
        - Descriptive adjectives
        - Spatial language
        - Context provision

        Args:
            prompt: The base prompt to enhance

        Returns:
            Enhanced prompt ready for NB Pro
        """

        enhanced = prompt

        # Ensure it starts naturally
        if not enhanced[0].isupper():
            enhanced = enhanced[0].upper() + enhanced[1:]

        # Add conversational markers if missing
        conversational_markers = [
            "situated in",
            "positioned at",
            "resting on",
            "featuring",
            "displaying",
            "showcasing",
        ]

        # Add descriptive words if prompt is too short
        words = enhanced.split()
        if len(words) < 30:
            # Add more descriptive language
            enhanced = self._add_descriptive_language(enhanced)

        return enhanced

    def _add_descriptive_language(self, prompt: str) -> str:
        """Add descriptive language to a short prompt."""

        # Add descriptive words before key terms
        enhancements = {
            "product": "high-quality",
            "light": "soft, diffused",
            "background": "clean, neutral",
            "scene": "professional",
            "colors": "vibrant, accurate",
        }

        for term, enhancement in enhancements.items():
            if term in prompt.lower() and enhancement not in prompt.lower():
                # Find and replace first occurrence
                idx = prompt.lower().find(term)
                if idx != -1:
                    prompt = prompt[:idx] + enhancement + " " + prompt[idx:]

        return prompt
