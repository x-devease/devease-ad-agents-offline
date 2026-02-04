"""
Thinking Engine (Strategy) for Nano Banana Pro Agent.

Generates thinking blocks and plans which NB techniques to apply.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from src.agents.nano.core.types import (
    ThinkingBlock,
    PromptCategory,
    PromptIntent,
    AgentInput,
)


logger = logging.getLogger(__name__)


class ThinkingEngine:
    """
    Generate strategic thinking and plan technique application.

    Analyzes the request and decides:
    - Which NB techniques to apply
    - What risks exist
    - How to mitigate risks
    - What the output should accomplish
    """

    # Risk patterns to check for
    RISK_PATTERNS = {
        "hallucination": [
            "creative", "artistic", "interpretive", "stylized",
        ],
        "inconsistency": [
            "sequence", "story", "series", "multiple",
        ],
        "loss_brand_integrity": [
            "creative", "unique", "different",
        ],
        "unrealistic_physics": [
            "liquid", "water", "pouring", "falling", "floating",
        ],
        "text_illegibility": [
            "text", "title", "label", "chart", "infographic",
        ],
    }

    def __init__(self):
        """Initialize the thinking engine."""
        self.technique_priority = {
            "natural_language": 1,
            "text_rendering": 2,
            "character_consistency": 2,
            "physics_understanding": 3,
            "dimensional_translation": 3,
            "search_grounding": 4,
            "storyboarding": 4,
            "structural_control": 5,
        }

    def generate_thinking(
        self,
        agent_input: AgentInput,
        category: PromptCategory,
        intent: PromptIntent,
        missing_elements: List[str],
        analysis: Dict[str, Any],
    ) -> ThinkingBlock:
        """
        Generate a thinking block for the prompt.

        Args:
            agent_input: The enriched agent input
            category: Detected prompt category
            intent: Detected intent
            missing_elements: List of what's missing from input
            analysis: Deep analysis results

        Returns:
            ThinkingBlock with analysis, techniques, risks, mitigation
        """

        # What is this request about?
        analysis_text = self._generate_analysis(
            agent_input, category, intent, analysis
        )

        # Which techniques should we apply?
        techniques = self._select_techniques(
            category, intent, missing_elements, analysis
        )

        # What could go wrong?
        risks = self._assess_risks(
            agent_input.generic_prompt, category, intent
        )

        # How do we prevent issues?
        mitigation = self._plan_mitigation(risks, category)

        # For whom and why?
        context = self._generate_context(agent_input, intent)

        return ThinkingBlock(
            analysis=analysis_text,
            techniques=techniques,
            risks=risks,
            mitigation=mitigation,
            context=context,
        )

    def _generate_analysis(
        self,
        agent_input: AgentInput,
        category: PromptCategory,
        intent: PromptIntent,
        analysis: Dict[str, Any],
    ) -> str:
        """Generate the analysis section of the thinking block."""

        parts = []

        # Request type
        parts.append(f"Request: {intent.value.replace('_', ' ')}")

        # Category
        parts.append(f"Input Category: {category.value} ({self._get_category_description(category)})")

        # Product context
        if agent_input.product_context:
            parts.append(
                f"Product: {agent_input.product_context.name} "
                f"({agent_input.product_context.category})"
            )
            if agent_input.product_context.key_features:
                parts.append(
                    f"Key Features: {', '.join(agent_input.product_context.key_features[:3])}"
                )

        # Brand context
        if agent_input.brand_guidelines:
            parts.append(
                f"Brand: {agent_input.brand_guidelines.brand_name} "
                f"- {agent_input.brand_guidelines.visual_language}"
            )

        # Complexity assessment
        complexity = analysis.get("complexity", "simple")
        parts.append(f"Complexity: {complexity}")

        return ". ".join(parts) + "."

    def _select_techniques(
        self,
        category: PromptCategory,
        intent: PromptIntent,
        missing_elements: List[str],
        analysis: Dict[str, Any],
    ) -> List[str]:
        """Select which NB techniques to apply."""

        techniques = []

        # Always use natural language
        techniques.append("Natural language prompting (conversational, descriptive)")

        # Category-based technique selection
        if category == PromptCategory.COMPARATIVE:
            techniques.extend([
                "Text rendering & infographics",
                "Data visualization",
                "Structural control (layout precision)",
            ])

        elif category == PromptCategory.SEQUENTIAL:
            techniques.extend([
                "One-shot storyboarding",
                "Character consistency (identity locking)",
                "Emotional progression",
            ])

        elif category == PromptCategory.TECHNICAL:
            techniques.extend([
                "Text rendering (technical fonts)",
                "Dimensional translation (2D → 3D)",
                "Structural control (blueprints)",
            ])

        # Intent-based technique selection
        if intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
            techniques.extend([
                "Character consistency (product)",
                "High-resolution output (4K)",
                "Physics understanding (material properties)",
            ])

        elif intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            techniques.extend([
                "Natural environment setting",
                "Human emotion rendering",
                "Authenticity over perfection",
            ])

        # Analysis-based technique selection
        if analysis.get("needs_physics"):
            techniques.append("Physics understanding (liquids, gravity, cloth)")

        if analysis.get("needs_text_overlay"):
            techniques.append("Text rendering optimization")

        if analysis.get("needs_human_element"):
            techniques.append("Character consistency (person)")

        if analysis.get("is_sequential"):
            techniques.append("Storyboard consistency management")

        # Sort by priority
        techniques = self._sort_by_priority(techniques)

        return techniques

    def _sort_by_priority(self, techniques: List[str]) -> List[str]:
        """Sort techniques by priority (lower number = higher priority)."""

        def get_priority(technique: str) -> int:
            for key, priority in self.technique_priority.items():
                if key in technique.lower():
                    return priority
            return 999  # Unknown techniques get lowest priority

        return sorted(techniques, key=get_priority)

    def _assess_risks(
        self,
        prompt: str,
        category: PromptCategory,
        intent: PromptIntent,
    ) -> List[str]:
        """Assess potential risks for this request."""

        risks = []
        prompt_lower = prompt.lower()

        # Check for risk patterns
        for risk_type, patterns in self.RISK_PATTERNS.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    risks.append(f"{risk_type.replace('_', ' ')} - '{pattern}' detected")
                    break

        # Category-specific risks
        if category == PromptCategory.ULTRA_SIMPLE:
            risks.append("vague_input - prompt lacks specific details")

        if category == PromptCategory.SEQUENTIAL:
            risks.append("consistency - maintaining consistency across multiple images is challenging")

        if category == PromptCategory.COMPARATIVE:
            risks.append("data_clarity - comparative data may not be visually clear")

        # Intent-specific risks
        if intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
            risks.append("stock_photo_appearance - may look inauthentic or staged")

        # Default risk if nothing detected
        if not risks:
            risks.append("generic - prompt may produce generic results without specific direction")

        return risks

    def _plan_mitigation(
        self,
        risks: List[str],
        category: PromptCategory,
    ) -> List[str]:
        """Plan how to mitigate identified risks."""

        mitigations = []

        for risk in risks:
            risk_lower = risk.lower()

            if "hallucination" in risk_lower:
                mitigations.append("Add anti-hallucination guards (Do NOT add elements)")

            elif "consistency" in risk_lower:
                mitigations.append("Use character consistency with reference images")

            elif "brand_integrity" in risk_lower:
                mitigations.append("Strict brand guideline enforcement")

            elif "physics" in risk_lower:
                mitigations.append("Specify realistic physics parameters")

            elif "text" in risk_lower:
                mitigations.append("Use explicit text rendering specifications")

            elif "vague" in risk_lower:
                mitigations.append("Add specific descriptive details and context")

            elif "stock_photo" in risk_lower:
                mitigations.append("Add authenticity imperfections and lived-in details")

            elif "data_clarity" in risk_lower:
                mitigations.append("Use clear data visualization with proper hierarchy")

        # Remove duplicates
        mitigations = list(set(mitigations))

        # Add default mitigations based on category
        if category == PromptCategory.ULTRA_SIMPLE:
            if "add specific descriptive details" not in mitigations:
                mitigations.append("Add comprehensive visual descriptions and context")

        return mitigations

    def _generate_context(self, agent_input: AgentInput, intent: PromptIntent) -> str:
        """Generate the 'for whom, why' context."""

        context_parts = []

        # Who is this for?
        if agent_input.target_audience:
            context_parts.append(f"Target audience: {agent_input.target_audience}")
        else:
            # Default audience based on intent
            if intent == PromptIntent.PRODUCT_PHOTOGRAPHY:
                context_parts.append("Target audience: E-commerce customers, potential buyers")
            elif intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
                context_parts.append("Target audience: Homeowners, families")
            else:
                context_parts.append("Target audience: General consumers")

        # What emotion should it evoke?
        if agent_input.emotion_goal:
            context_parts.append(f"Emotional goal: {agent_input.emotion_goal}")
        else:
            # Default emotion based on intent
            if intent == PromptIntent.LIFESTYLE_ADVERTISEMENT:
                context_parts.append("Emotional goal: Trust, satisfaction, aspiration")
            elif intent == PromptIntent.COMPARATIVE_INFOGRAPHIC:
                context_parts.append("Emotional goal: Confidence, clarity, decision-making")
            else:
                context_parts.append("Emotional goal: Professional, trustworthy")

        # Why create this?
        context_parts.append(f"Purpose: {self._get_purpose(intent)}")

        return ". ".join(context_parts) + "."

    def _get_purpose(self, intent: PromptIntent) -> str:
        """Get the purpose based on intent."""

        purposes = {
            PromptIntent.PRODUCT_PHOTOGRAPHY: "Show product clearly for e-commerce or catalog",
            PromptIntent.LIFESTYLE_ADVERTISEMENT: "Demonstrate product benefits in authentic lifestyle setting",
            PromptIntent.COMPARATIVE_INFOGRAPHIC: "Compare and highlight competitive advantages visually",
            PromptIntent.STORYBOARD_SEQUENCE: "Tell a narrative story with emotional arc",
            PromptIntent.TECHNICAL_DIAGRAM: "Explain technical details with precision",
            PromptIntent.EDIT_REFINEMENT: "Improve existing image quality",
            PromptIntent.BRAND_ASSET_GENERATION: "Create consistent brand assets",
        }

        return purposes.get(intent, "Generate professional visual content")

    def _get_category_description(self, category: PromptCategory) -> str:
        """Get human-readable description of category."""

        descriptions = {
            PromptCategory.ULTRA_SIMPLE: "Very basic input, needs complete enhancement",
            PromptCategory.BASIC_DIRECTION: "Has some direction, needs detail and style",
            PromptCategory.SPECIFIC_REQUEST: "Specific requirements given, needs execution",
            PromptCategory.COMPARATIVE: "Comparison requested, needs data visualization",
            PromptCategory.SEQUENTIAL: "Story/sequence requested, needs consistency management",
            PromptCategory.TECHNICAL: "Technical/diagram requested, needs precision",
        }

        return descriptions.get(category, "Standard request")
