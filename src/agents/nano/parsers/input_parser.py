"""
Input Parser & Intent Analyzer for Nano Banana Pro Agent.

Analyzes generic input prompts to detect intent, category, and missing elements.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from src.agents.nano.core.types import (
    PromptCategory,
    PromptIntent,
)


logger = logging.getLogger(__name__)


class InputParser:
    """
    Parse generic input prompts and detect user intent.

    Analyzes what the user wants and what's missing from their request.
    """

    # Keyword patterns for intent detection
    INTENT_PATTERNS = {
        PromptIntent.PRODUCT_PHOTOGRAPHY: [
            r"product photo",
            r"product shot",
            r"catalog",
            r"white background",
            r"studio shot",
            r"ecommerce",
        ],
        PromptIntent.LIFESTYLE_ADVERTISEMENT: [
            r"lifestyle",
            r"advertisement",
            r"ad for",
            r"commercial",
            r"show someone using",
            r"in use",
            r"happy with",
        ],
        PromptIntent.COMPARATIVE_INFOGRAPHIC: [
            r"compare",
            r"comparison",
            r"versus",
            r"vs\.?",
            r"chart",
            r"infographic",
            r"graph",
        ],
        PromptIntent.STORYBOARD_SEQUENCE: [
            r"story",
            r"sequence",
            r"before.*after",
            r"narrative",
            r"storyboard",
            r"first.*then.*finally",
        ],
        PromptIntent.TECHNICAL_DIAGRAM: [
            r"technical diagram",
            r"blueprint",
            r"schematic",
            r"mechanism",
            r"how it works",
            r"exploded view",
            r"technical drawing",
        ],
    }

    # Keyword patterns for category detection
    CATEGORY_PATTERNS = {
        PromptCategory.ULTRA_SIMPLE: [
            r"^make (an? )?(ad|advertisement|image|photo)$",
            r"^create (an? )?(ad|advertisement|image|photo)$",
            r"^generate (an? )?(ad|advertisement|image|photo)$",
        ],
        PromptCategory.BASIC_DIRECTION: [
            r"show",
            r"with (someone|people|person)",
            r"in (a|the) (kitchen|living room|bathroom|home)",
        ],
        PromptCategory.SPECIFIC_REQUEST: [
            r"\d+K",
            r"white background",
            r"studio lighting",
            r"\d+.*degrees",
        ],
        PromptCategory.COMPARATIVE: [
            r"compare",
            r"versus",
            r"vs\.?",
            r"better than",
            r"difference",
        ],
        PromptCategory.SEQUENTIAL: [
            r"story",
            r"sequence",
            r"before.*after",
            r"narrative",
            r"part \d+",
        ],
        PromptCategory.TECHNICAL: [
            r"diagram",
            r"blueprint",
            r"schematic",
            r"mechanism",
            r"technical",
        ],
    }

    def parse(self, generic_prompt: str) -> Tuple[PromptCategory, PromptIntent]:
        """
        Parse generic prompt and detect category/intent.

        Args:
            generic_prompt: The user's input prompt

        Returns:
            Tuple of (detected_category, detected_intent)
        """
        category = self._detect_category(generic_prompt)
        intent = self._detect_intent(generic_prompt)

        logger.info(f"Detected category: {category.value}, intent: {intent.value}")

        return category, intent

    def _detect_category(self, prompt: str) -> PromptCategory:
        """Detect the category of the input prompt."""

        prompt_lower = prompt.lower()

        # Check each category's patterns
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return category

        # Default: treat as basic direction if no pattern matches
        return PromptCategory.BASIC_DIRECTION

    def _detect_intent(self, prompt: str) -> PromptIntent:
        """Detect the intent of the input prompt."""

        prompt_lower = prompt.lower()

        # Check each intent's patterns
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return intent

        # Default: infer from keywords
        if "photo" in prompt_lower or "image" in prompt_lower:
            return PromptIntent.PRODUCT_PHOTOGRAPHY
        elif "ad" in prompt_lower or "advertisement" in prompt_lower:
            return PromptIntent.LIFESTYLE_ADVERTISEMENT

        # Fallback
        return PromptIntent.LIFESTYLE_ADVERTISEMENT

    def identify_missing_elements(self, generic_prompt: str) -> List[str]:
        """
        Identify what's missing from the user's request.

        Args:
            generic_prompt: The user's input prompt

        Returns:
            List of missing elements (e.g., "lighting", "style", "emotion")
        """

        missing = []
        prompt_lower = generic_prompt.lower()

        # Check for common missing elements
        if not any(word in prompt_lower for word in ["light", "lighting", "sun", "bright"]):
            missing.append("lighting description")

        if not any(word in prompt_lower for word in ["style", "look like", "feel"]):
            missing.append("visual style")

        if not any(word in prompt_lower for word in ["happy", "sad", "excited", "emotional", "satisfied"]):
            missing.append("emotion/mood")

        if not any(word in prompt_lower for word in ["4k", "2k", "high res", "resolution"]):
            missing.append("resolution")

        if not any(word in prompt_lower for word in ["natural", "conversational", "authentic", "realistic"]):
            missing.append("natural language description")

        return missing

    def extract_keywords(self, generic_prompt: str) -> List[str]:
        """
        Extract key nouns and concepts from the prompt.

        Args:
            generic_prompt: The user's input prompt

        Returns:
            List of keywords found in the prompt
        """

        # Simple extraction: look for common patterns
        keywords = []

        # Extract product mentions (simple heuristic)
        product_patterns = [
            r"(?:for|of|with|our) ([a-z]+(?: [a-z]+){0,2})",
        ]

        for pattern in product_patterns:
            matches = re.findall(pattern, generic_prompt.lower())
            keywords.extend(matches)

        # Remove duplicates and common words
        stop_words = {"a", "an", "the", "for", "of", "with", "in", "on"}
        keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 2]

        return list(set(keywords))


class IntentAnalyzer:
    """
    Analyze the intent behind user prompts and suggest enhancements.

    Goes beyond simple pattern matching to understand what the user
    really wants from the output.
    """

    def analyze_request_type(self, prompt: str, category: PromptCategory) -> dict:
        """
        Perform deep analysis of the request type.

        Args:
            prompt: The user's input prompt
            category: Detected prompt category

        Returns:
            Dictionary with analysis results
        """

        analysis = {
            "is_sequential": False,
            "needs_text_overlay": False,
            "needs_human_element": False,
            "needs_physics": False,
            "needs_brand_consistency": False,
            "complexity": "simple",
        }

        prompt_lower = prompt.lower()

        # Detect sequential requirements
        if category == PromptCategory.SEQUENTIAL:
            analysis["is_sequential"] = True
            analysis["complexity"] = "complex"

        # Detect text overlay needs
        if any(word in prompt_lower for word in ["text", "title", "label", "chart", "infographic"]):
            analysis["needs_text_overlay"] = True
            analysis["complexity"] = "medium"

        # Detect human element
        if any(word in prompt_lower for word in ["person", "someone", "woman", "man", "people", "using", "holding"]):
            analysis["needs_human_element"] = True

        # Detect physics requirements
        if any(word in prompt_lower for word in ["liquid", "water", "spill", "wet", "pour", "spray", "falling"]):
            analysis["needs_physics"] = True
            analysis["complexity"] = "complex"

        # Detect brand consistency needs
        if any(word in prompt_lower for word in ["brand", "logo", "consistent", "match"]):
            analysis["needs_brand_consistency"] = True

        return analysis

    def suggest_enhancement_strategy(
        self,
        category: PromptCategory,
        intent: PromptIntent,
        missing_elements: List[str],
        analysis: dict,
    ) -> List[str]:
        """
        Suggest what enhancements should be applied.

        Args:
            category: Detected prompt category
            intent: Detected intent
            missing_elements: List of missing elements
            analysis: Deep analysis results

        Returns:
            List of enhancement strategies to apply
        """

        strategies = []

        # Base strategies based on category
        if category == PromptCategory.ULTRA_SIMPLE:
            strategies.extend([
                "add_complete_context",
                "add_visual_description",
                "add_technical_specs",
                "add_emotional_context",
            ])

        elif category == PromptCategory.BASIC_DIRECTION:
            strategies.extend([
                "enhance_visual_details",
                "add_lighting_description",
                "add_style_declaration",
            ])

        elif category == PromptCategory.COMPARATIVE:
            strategies.extend([
                "add_data_visualization",
                "add_text_rendering_specs",
                "add_layout_structure",
            ])

        elif category == PromptCategory.SEQUENTIAL:
            strategies.extend([
                "add_storyboard_structure",
                "add_consistency_requirements",
                "add_emotional_progression",
            ])

        # Intent-based strategies
        if intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
            strategies.append("add_photorealism_specs")

        elif intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            strategies.extend([
                "add_human_emotion",
                "add_authenticity_details",
            ])

        # Analysis-based strategies
        if analysis.get("needs_physics"):
            strategies.append("add_phics_simulation")

        if analysis.get("needs_text_overlay"):
            strategies.append("add_text_rendering")

        if analysis.get("is_sequential"):
            strategies.append("add_storyboard_consistency")

        return strategies
