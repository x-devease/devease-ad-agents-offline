"""
Quality Verifier for Nano Banana Pro Agent.

Verifies that prompts meet quality standards before output.
"""

from __future__ import annotations

import logging
from typing import List, Tuple


logger = logging.getLogger(__name__)


class QualityVerifier:
    """
    Verify prompt quality before output.

    Checks that prompts meet Nano Banana Pro best practices.
    """

    # Quality thresholds
    MIN_LENGTH = 100  # Minimum characters
    MIN_TECHNIQUES = 1  # At least 1 NB technique

    def verify(
        self,
        prompt: str,
        techniques_applied: List[str],
        has_thinking: bool = False,
    ) -> Tuple[bool, float, List[str]]:
        """
        Verify prompt quality.

        Args:
            prompt: The enhanced prompt to verify
            techniques_applied: List of techniques applied
            has_thinking: Whether thinking block is included

        Returns:
            Tuple of (passes_verification, confidence_score, list_of_issues)
        """

        issues = []
        score = 1.0

        # Check 1: Minimum length
        if len(prompt) < self.MIN_LENGTH:
            issues.append(f"Prompt too short ({len(prompt)} < {self.MIN_LENGTH} chars)")
            score -= 0.3

        # Check 2: Natural language
        if self._is_technical(prompt):
            issues.append("Prompt uses technical jargon instead of natural language")
            score -= 0.2

        # Check 3: Specific details
        if not self._has_specific_details(prompt):
            issues.append("Prompt lacks specific descriptive details")
            score -= 0.15

        # Check 4: Context
        if not self._has_context(prompt):
            issues.append("Prompt lacks context (why/for whom)")
            score -= 0.1

        # Check 5: Technical specs
        if not self._has_technical_specs(prompt):
            issues.append("Prompt lacks technical specifications")
            score -= 0.1

        # Check 6: NB techniques
        if len(techniques_applied) < self.MIN_TECHNIQUES:
            issues.append(f"Too few techniques applied ({len(techniques_applied)} < {self.MIN_TECHNIQUES})")
            score -= 0.2

        # Bonus for thinking block
        if has_thinking:
            score += 0.1

        # Ensure score is in [0, 1]
        score = max(0.0, min(1.0, score))

        passes = len(issues) == 0 or score >= 0.6

        logger.info(f"Quality verification: passes={passes}, score={score:.2f}, issues={len(issues)}")

        return passes, score, issues

    def _is_technical(self, prompt: str) -> bool:
        """Check if prompt uses too much technical jargon."""

        technical_keywords = [
            "85mm", "f/2.8", "f/4", "ISO", "shutter speed",
            "DOF", "depth of field", "bokeh",
            "HDR", "RAW",
        ]

        prompt_upper = prompt.upper()

        # Count technical keywords
        tech_count = sum(1 for kw in technical_keywords if kw.upper() in prompt_upper)

        # If more than 3 technical keywords, it's too technical
        return tech_count > 3

    def _has_specific_details(self, prompt: str) -> bool:
        """Check if prompt has specific, descriptive details."""

        specific_indicators = [
            "color", "texture", "material", "lighting", "shadow",
            "positioned", "placed", "situated", "resting",
            "displays", "features", "shows",
        ]

        prompt_lower = prompt.lower()

        # Need at least 3 specific indicators
        count = sum(1 for indicator in specific_indicators if indicator in prompt_lower)

        return count >= 3

    def _has_context(self, prompt: str) -> bool:
        """Check if prompt provides context."""

        context_indicators = [
            "for", "audience", "customer", "target",
            "purpose", "goal", "objective",
            "emotional", "mood", "feeling",
        ]

        prompt_lower = prompt.lower()

        return any(indicator in prompt_lower for indicator in context_indicators)

    def _has_technical_specs(self, prompt: str) -> bool:
        """Check if prompt includes technical specifications."""

        spec_indicators = [
            "resolution", "lighting", "camera", "style",
            "K", "4K", "2K", "1K",
        ]

        prompt_upper = prompt.upper()

        return any(indicator.upper() in prompt_upper for indicator in spec_indicators)
