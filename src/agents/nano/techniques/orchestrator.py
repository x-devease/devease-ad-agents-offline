"""
Technique Orchestrator for Nano Banana Pro Agent.

Applies Nano Banana Pro techniques to the prompt.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from src.agents.nano.core.types import AppliedTechnique


logger = logging.getLogger(__name__)


class TechniqueOrchestrator:
    """
    Orchestrate the application of Nano Banana Pro techniques.

    Applies techniques like text rendering, character consistency, physics,
    etc. based on the request analysis.
    """

    def apply_techniques(
        self,
        prompt_content: str,
        techniques_list: List[str],
        analysis: Dict[str, Any],
    ) -> tuple[str, List[AppliedTechnique]]:
        """
        Apply NB techniques to the prompt.

        Args:
            prompt_content: The current prompt content
            techniques_list: List of techniques to apply
            analysis: Request analysis results

        Returns:
            Tuple of (enhanced_prompt, list_of_applied_techniques)
        """

        enhanced_prompt = prompt_content
        applied_techniques = []

        for technique in techniques_list:
            technique_addition, applied = self._apply_technique(
                technique, enhanced_prompt, analysis
            )

            if applied:
                enhanced_prompt += "\n\n" + technique_addition
                applied_techniques.append(AppliedTechnique(
                    technique_name=technique,
                    description=self._get_technique_description(technique),
                    prompt_addition=technique_addition,
                ))

        logger.info(f"Applied {len(applied_techniques)} techniques")

        return enhanced_prompt, applied_techniques

    def _apply_technique(self, technique: str, prompt: str, analysis: Dict) -> tuple[str, bool]:
        """Apply a single technique."""

        technique_lower = technique.lower()

        if "text rendering" in technique_lower:
            addition = self._apply_text_rendering(analysis)
            return addition, True

        elif "physics" in technique_lower:
            addition = self._apply_physics(analysis)
            return addition, True

        elif "character consistency" in technique_lower:
            addition = self._apply_character_consistency(analysis)
            return addition, True

        elif "storyboard" in technique_lower:
            addition = self._apply_storyboard(analysis)
            return addition, True

        elif "natural language" in technique_lower:
            return "", False  # Already applied in NaturalLanguagePromptBuilder

        else:
            return "", False

    def _apply_text_rendering(self, analysis: Dict) -> str:
        """Apply text rendering technique."""

        return (
            "Text Rendering Specifications:\n"
            "- Use clean, modern sans-serif font (Inter, Roboto, or Arial)\n"
            "- Text should be highly legible with proper contrast\n"
            "- Add subtle drop shadows for text readability\n"
            "- Position text in upper regions with proper spacing"
        )

    def _apply_physics(self, analysis: Dict) -> str:
        """Apply physics understanding technique."""

        return (
            "Physics Requirements:\n"
            "- Realistic material properties (weight, density, friction)\n"
            "- Proper contact shadows where objects touch surfaces\n"
            "- Accurate light interaction (reflections, refraction)\n"
            "- Natural cloth behavior and soft body dynamics if applicable"
        )

    def _apply_character_consistency(self, analysis: Dict) -> str:
        """Apply character consistency technique."""

        return (
            "Character Consistency:\n"
            "- Maintain exact facial features and identity across all images\n"
            "- Keep clothing, accessories, and props consistent\n"
            "- Preserve proportions and silhouette\n"
            "- Use reference images for identity locking"
        )

    def _apply_storyboard(self, analysis: Dict) -> str:
        """Apply storyboard technique."""

        return (
            "Storyboard Requirements:\n"
            "- Maintain visual consistency across all images in the sequence\n"
            "- Create emotional progression through the narrative\n"
            "- Keep lighting and color grading cohesive\n"
            "- Vary camera angles for visual interest while maintaining consistency"
        )

    def _get_technique_description(self, technique: str) -> str:
        """Get human-readable description of a technique."""

        descriptions = {
            "text rendering": "Optimized text rendering for legibility and style",
            "physics": "Realistic physics simulation for materials and forces",
            "character consistency": "Identity locking for consistent characters/products",
            "storyboard": "Sequential narrative with consistency management",
            "natural language": "Conversational, descriptive prompting",
        }

        for key, desc in descriptions.items():
            if key in technique.lower():
                return desc

        return f"Applied {technique}"
