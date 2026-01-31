"""
Nano Adapter for the Prompt Enhancement Framework.

Adapts the generic framework to work with Nano Banana Pro.
"""

from __future__ import annotations

import logging
from typing import Tuple, List, Dict, Any
import re

from src.agents.framework.adapters.base import BaseAdapter, AdapterConfig
from src.agents.framework.core.types import AgentInput
from src.agents.nano.parsers.input_parser import InputParser, IntentAnalyzer
from src.agents.nano.core.context_enrichment import ContextEnrichmentEngine
from src.agents.nano.core.thinking_engine import ThinkingEngine
from src.agents.nano.formatters.natural_language_builder import NaturalLanguagePromptBuilder
from src.agents.nano.techniques.orchestrator import TechniqueOrchestrator


logger = logging.getLogger(__name__)


class NanoAdapter(BaseAdapter):
    """
    Nano Banana Pro specific adapter.

    Bridges the generic framework with Nano's domain-specific logic.
    """

    def __init__(self, config: AdapterConfig):
        """
        Initialize Nano adapter.

        Args:
            config: Adapter configuration
        """
        super().__init__(config)

        # Set domain identifier
        self._domain = "nano"

        # Initialize Nano components
        self.parser = InputParser()
        self.intent_analyzer = IntentAnalyzer()
        self.context_enrichment = ContextEnrichmentEngine()
        self.thinking_engine = ThinkingEngine()
        self.prompt_builder = NaturalLanguagePromptBuilder()
        self.technique_orchestrator = TechniqueOrchestrator()

        # Load Nano config if provided
        if config.config_path:
            # Could load Nano-specific config here
            pass

        logger.info(f"Initialized Nano adapter with domain 'nano'")

    @property
    def domain(self) -> str:
        """Domain identifier."""
        return self._domain

    def parse_input(self, generic_prompt: str) -> Tuple[str, str]:
        """
        Parse input into category and intent.

        Args:
            generic_prompt: User's generic input prompt

        Returns:
            (category, intent) - Nano-specific categorization
        """
        category, intent = self.parser.parse(generic_prompt)
        return category, intent

    def enrich_context(self, agent_input: AgentInput) -> AgentInput:
        """
        Enrich agent input with Nano-specific context.

        Args:
            agent_input: Original agent input

        Returns:
            Enriched agent input with product/brand context
        """
        # Use existing context enrichment
        enriched = self.context_enrichment.enrich(agent_input)
        return enriched

    def generate_thinking(
        self,
        agent_input: AgentInput,
        category: str,
        intent: str,
        examples: List,
    ) -> str:
        """
        Generate thinking block for the enhancement.

        Args:
            agent_input: User input
            category: Detected category
            intent: Detected intent
            examples: Relevant grounding examples (not used in current Nano implementation)

        Returns:
            Thinking block as string
        """
        # Get missing elements
        missing_elements = self.parser.identify_missing_elements(
            agent_input.generic_prompt
        )

        # Analyze request type
        analysis = self.intent_analyzer.analyze_request_type(
            agent_input.generic_prompt, category
        )

        # Use existing thinking engine
        thinking_block = self.thinking_engine.generate_thinking(
            agent_input, category, intent, missing_elements, analysis
        )
        # Return formatted thinking as a string
        return thinking_block.format()

    def build_prompt(self, agent_input: AgentInput, category: str, intent: str) -> str:
        """
        Build Nano-specific prompt.

        Args:
            agent_input: User input with context
            category: Detected category
            intent: Detected intent

        Returns:
            Built prompt as string
        """
        # Get analysis for prompt builder
        analysis = self.intent_analyzer.analyze_request_type(
            agent_input.generic_prompt, category
        )

        # Use existing prompt builder
        intermediate = self.prompt_builder.build(
            agent_input, category, intent, analysis
        )
        return intermediate.prompt_content

    def apply_techniques(self, prompt: str, thinking: str) -> str:
        """
        Apply Nano-specific techniques to prompt.

        Args:
            prompt: Base prompt
            thinking: Thinking block with technique selection

        Returns:
            Enhanced prompt with techniques applied
        """
        # Extract techniques from thinking
        techniques = self._extract_techniques_from_thinking(thinking)

        # Apply techniques
        analysis = {
            "category": "nano",
            "intent": "enhancement",
        }

        enhanced_prompt, applied_techniques = self.technique_orchestrator.apply_techniques(
            prompt, techniques, analysis
        )

        return enhanced_prompt

    def refine_prompt(
        self, prompt: str, critique: str, agent_input: AgentInput
    ) -> str:
        """
        Refine prompt based on critique (Nano-specific logic).

        Args:
            prompt: Current prompt
            critique: Critique from Reflexion engine
            agent_input: Original user input

        Returns:
            Refined prompt
        """
        # Nano-specific refinement logic
        refined = prompt

        # Add more specificity if critique mentions it
        if "specific" in critique.lower() or "detail" in critique.lower():
            refined = self._add_specificity(refined, agent_input)

        # Improve natural language if critique mentions it
        if "natural" in critique.lower() or "conversational" in critique.lower():
            refined = self._improve_natural_language(refined)

        # Add visual details if critique mentions length or completeness
        if "short" in critique.lower() or "incomplete" in critique.lower():
            refined = self._add_visual_details(refined)

        return refined

    def _extract_techniques_from_thinking(self, thinking: str) -> List[str]:
        """
        Extract technique names from thinking block.

        Args:
            thinking: Thinking block text

        Returns:
            List of technique names
        """
        techniques = []

        # Look for common Nano technique patterns
        technique_patterns = [
            r"text rendering",
            r"character consistency",
            r"physics",
            r"storyboard",
            r"natural language",
        ]

        thinking_lower = thinking.lower()
        for pattern in technique_patterns:
            if pattern in thinking_lower:
                techniques.append(pattern)

        return techniques

    def _add_specificity(self, prompt: str, agent_input: AgentInput) -> str:
        """Add more specific descriptors to prompt."""
        additions = []

        # Add product context if available
        if agent_input.product_context:
            if agent_input.product_context.key_features:
                features = ", ".join(agent_input.product_context.key_features[:2])
                additions.append(f"featuring {features}")

            if agent_input.product_context.materials:
                materials = ", ".join(agent_input.product_context.materials[:2])
                additions.append(f"crafted from {materials}")

        if additions:
            return prompt + "\n\n" + " | ".join(additions)

        return prompt

    def _improve_natural_language(self, prompt: str) -> str:
        """Improve natural language flow of prompt."""
        # Remove excessive colons
        improved = re.sub(r":\s*", ": ", prompt)

        # Convert bullet points to flowing text
        improved = re.sub(r"\n-\s*", ", with ", improved)

        # Ensure it starts naturally
        if improved.lower().startswith("a "):
            improved = "Show " + improved[2:].lower()

        return improved

    def _add_visual_details(self, prompt: str) -> str:
        """Add visual details to prompt."""
        visual_additions = [
            "with professional lighting",
            "high resolution",
            "sharp details",
        ]

        # Add 1-2 visual details that aren't already present
        for addition in visual_additions:
            if addition.lower() not in prompt.lower():
                return prompt + " " + addition

        return prompt

    def validate_domain_specific(self, prompt: str) -> List[str]:
        """
        Validate Nano-specific requirements.

        Args:
            prompt: Prompt to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check for Nano Banana Pro techniques
        nb_techniques = [
            "text rendering",
            "character consistency",
            "physics",
            "resolution",
        ]

        has_any_technique = any(
            technique in prompt.lower() for technique in nb_techniques
        )

        if not has_any_technique:
            issues.append("Missing Nano Banana Pro techniques")

        return issues

    def compute_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Compute similarity between prompts for memory retrieval.

        Uses Nano-specific similarity that considers product names and features.

        Args:
            prompt1: First prompt
            prompt2: Second prompt

        Returns:
            Similarity score between 0 and 1
        """
        # Base keyword similarity
        base_score = super().compute_similarity(prompt1, prompt2)

        # Boost if prompts mention similar product features
        feature_words = [
            "mop",
            "cleaning",
            "360",
            "rotation",
            "spray",
            "microfiber",
        ]

        features1 = sum(1 for word in feature_words if word in prompt1.lower())
        features2 = sum(1 for word in feature_words if word in prompt2.lower())

        if features1 > 0 and features2 > 0:
            feature_overlap = min(features1, features2) / max(features1, features2)
            # Boost score by up to 20% based on feature overlap
            base_score += feature_overlap * 0.2

        return min(base_score, 1.0)
