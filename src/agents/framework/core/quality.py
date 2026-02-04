"""
Quality Verifier for prompt enhancement.

Verifies the quality of enhanced prompts.
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any
from dataclasses import dataclass

from src.agents.framework.core.types import AgentInput, QualityCheck, GroundingExample
from src.agents.framework.adapters.base import BaseAdapter


logger = logging.getLogger(__name__)


class QualityVerifier:
    """
    Verifies the quality of enhanced prompts.

    Quality checks ensure prompts meet standards for:
    - Specificity
    - Pattern consistency
    - Natural language quality
    - Completeness
    - Domain-specific requirements
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialize QualityVerifier.

        Args:
            threshold: Minimum quality threshold (default: 0.7)
        """
        self.threshold = threshold

    def verify(
        self,
        prompt: str,
        agent_input: AgentInput,
        adapter: BaseAdapter,
        examples: List[GroundingExample] = None,
    ) -> QualityCheck:
        """
        Verify quality of an enhanced prompt.

        Args:
            prompt: Enhanced prompt to verify
            agent_input: Original user input
            adapter: Domain-specific adapter
            examples: Grounding examples (for pattern consistency check)

        Returns:
            QualityCheck with verification results
        """
        issues = []

        # Generic checks
        specificity = self._check_specificity(prompt)
        pattern_consistency = (
            self._check_pattern_consistency(prompt, examples) if examples else 1.0
        )
        natural_language = self._check_natural_language(prompt)
        completeness = self._check_completeness(prompt, agent_input)

        # Domain-specific checks via adapter
        domain_issues = adapter.validate_domain_specific(prompt)
        issues.extend(domain_issues)

        # Calculate overall scores
        scores = {
            "specificity": specificity,
            "pattern_consistency": pattern_consistency,
            "natural_language": natural_language,
            "completeness": completeness,
        }

        # Calculate overall confidence
        confidence = sum(scores.values()) / len(scores)

        # Add generic issues if scores are low
        if specificity < 0.5:
            issues.append("Prompt lacks specific descriptive details")

        if pattern_consistency < 0.5 and examples:
            issues.append("Prompt doesn't follow example patterns")

        if natural_language < 0.5:
            issues.append("Prompt doesn't use natural language")

        if completeness < 0.5:
            issues.append("Prompt is incomplete relative to input")

        # Determine if passes threshold
        passes = confidence >= self.threshold

        return QualityCheck(
            passes=passes,
            confidence=confidence,
            issues=issues,
            specificity_score=specificity,
            pattern_consistency=pattern_consistency,
            natural_language_score=natural_language,
            completeness_score=completeness,
            metadata={"scores": scores},
        )

    def _check_specificity(self, prompt: str) -> float:
        """
        Check prompt specificity.

        Specific prompts have:
        - Concrete descriptors
        - Visual details
        - Technical specifications

        Args:
            prompt: Prompt to check

        Returns:
            Specificity score between 0 and 1
        """
        score = 0.0

        # Length check (longer is usually more specific)
        if len(prompt) >= 500:
            score += 0.3
        elif len(prompt) >= 300:
            score += 0.2
        elif len(prompt) >= 200:
            score += 0.1

        # Count specific descriptors
        specific_patterns = [
            r"\d+K|\d+p",  # Resolution like 2K, 1080p
            r"\d+\s*(degrees|°|inches|ft|cm)",  # Measurements
            r"(soft|hard|diffuse|natural)\s+light",  # Lighting
            r"(modern|classic|vintage|minimalist)\s+style",  # Style
        ]

        for pattern in specific_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                score += 0.1

        # Count adjectives and adverbs
        words = prompt.split()
        descriptive_words = [
            word for word in words if len(word) > 6 and word.isalpha()
        ]  # Longer words tend to be more descriptive

        if len(descriptive_words) >= 10:
            score += 0.2
        elif len(descriptive_words) >= 5:
            score += 0.1

        return min(score, 1.0)

    def _check_pattern_consistency(
        self, prompt: str, examples: List[GroundingExample]
    ) -> float:
        """
        Check consistency with example patterns.

        Args:
            prompt: Prompt to check
            examples: Grounding examples

        Returns:
            Pattern consistency score between 0 and 1
        """
        if not examples:
            return 1.0  # No examples to compare against

        # Extract patterns from examples
        example_patterns = set()
        for example in examples:
            words = set(example.output_prompt.lower().split())
            example_patterns.update(words)

        # Check how many patterns from examples are in prompt
        prompt_words = set(prompt.lower().split())
        overlap = example_patterns.intersection(prompt_words)

        if not example_patterns:
            return 1.0

        consistency = len(overlap) / len(example_patterns)

        return consistency

    def _check_natural_language(self, prompt: str) -> float:
        """
        Check if prompt uses natural language.

        Natural language prompts:
        - Avoid excessive colons/bullets
        - Use conversational phrasing
        - Flow like sentences

        Args:
            prompt: Prompt to check

        Returns:
            Natural language score between 0 and 1
        """
        score = 1.0

        # Penalize excessive formatting
        colon_count = prompt.count(":")
        bullet_count = prompt.count("-") + prompt.count("•")

        if colon_count > 5:
            score -= 0.3
        elif colon_count > 3:
            score -= 0.1

        if bullet_count > 10:
            score -= 0.3
        elif bullet_count > 5:
            score -= 0.1

        # Check for sentence-like structures
        sentences = re.split(r"[.!?]+", prompt)
        if len(sentences) >= 3:
            score += 0.1

        # Check for conversational words
        conversational_words = ["show", "create", "with", "featuring", "including"]
        conversational_count = sum(
            1 for word in conversational_words if word in prompt.lower()
        )

        if conversational_count >= 2:
            score += 0.1

        return max(0.0, min(score, 1.0))

    def _check_completeness(self, prompt: str, agent_input: AgentInput) -> float:
        """
        Check if prompt completely addresses the input.

        Args:
            prompt: Enhanced prompt
            agent_input: Original input

        Returns:
            Completeness score between 0 and 1
        """
        score = 0.0

        # Check if input keywords are present
        input_words = set(agent_input.generic_prompt.lower().split())
        prompt_words = set(prompt.lower().split())

        overlap = input_words.intersection(prompt_words)
        if input_words:
            coverage = len(overlap) / len(input_words)
            score += coverage * 0.5

        # Check if prompt has substantial content
        if len(prompt) >= len(agent_input.generic_prompt) * 2:
            score += 0.3
        elif len(prompt) >= len(agent_input.generic_prompt) * 1.5:
            score += 0.2

        # Check for product context if provided
        if agent_input.product_context:
            product_name = agent_input.product_context.name.lower()
            if product_name in prompt.lower():
                score += 0.2

        return min(score, 1.0)
