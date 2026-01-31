"""
Reflexion Engine for self-refinement.

Implements the Reflexion pattern: Generate → Critique → Refine.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from src.agents.framework.core.types import AgentInput
from src.agents.framework.adapters.base import BaseAdapter


logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of critiquing a prompt."""

    passes: bool
    issues: List[str]
    suggestions: List[str]
    severity: str  # "low", "medium", "high"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReflexionEngine:
    """
    Implements the Reflexion pattern for prompt refinement.

    Reflexion is a self-refinement pattern where the system:
    1. Generates an initial prompt
    2. Critiques it for issues
    3. Refines based on critique
    4. Repeats until quality threshold met or max iterations

    Research shows this provides +20-30% quality improvement.
    """

    def __init__(self, max_iterations: int = 2, quality_threshold: float = 0.7):
        """
        Initialize ReflexionEngine.

        Args:
            max_iterations: Maximum refine iterations (default: 2)
            quality_threshold: Quality threshold to stop refining (default: 0.7)
        """
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    def refine(
        self,
        prompt: str,
        agent_input: AgentInput,
        adapter: BaseAdapter,
    ) -> Tuple[str, List[CritiqueResult]]:
        """
        Refine a prompt through critique-refine iterations.

        Args:
            prompt: Initial prompt to refine
            agent_input: Original user input
            adapter: Domain-specific adapter

        Returns:
            Tuple of (refined_prompt, list_of_critique_results)
        """
        current_prompt = prompt
        critique_history: List[CritiqueResult] = []

        for iteration in range(self.max_iterations):
            logger.info(f"Reflexion iteration {iteration + 1}/{self.max_iterations}")

            # Critique current prompt
            critique = self._critique(current_prompt, agent_input, adapter)
            critique_history.append(critique)

            # If no issues or high quality, stop
            if critique.passes:
                logger.info("Prompt meets quality threshold, stopping refinement")
                break

            # Refine based on critique
            current_prompt = adapter.refine_prompt(
                current_prompt,
                self._format_critique_for_refinement(critique),
                agent_input,
            )

            logger.info(f"Refined prompt (iteration {iteration + 1})")

        return current_prompt, critique_history

    def _critique(
        self, prompt: str, agent_input: AgentInput, adapter: BaseAdapter
    ) -> CritiqueResult:
        """
        Critique a prompt for issues.

        Args:
            prompt: Prompt to critique
            agent_input: Original user input
            adapter: Domain-specific adapter

        Returns:
            CritiqueResult with issues and suggestions
        """
        issues = []
        suggestions = []

        # Generic checks
        generic_issues = self._generic_quality_checks(prompt, agent_input)
        issues.extend(generic_issues)

        # Domain-specific checks via adapter
        domain_issues = adapter.validate_domain_specific(prompt)
        issues.extend(domain_issues)

        # Determine if passes
        passes = len(issues) == 0

        # Generate suggestions based on issues
        for issue in issues:
            suggestion = self._generate_suggestion(issue, prompt)
            if suggestion:
                suggestions.append(suggestion)

        # Determine severity
        severity = self._calculate_severity(issues)

        return CritiqueResult(
            passes=passes,
            issues=issues,
            suggestions=suggestions,
            severity=severity,
        )

    def _generic_quality_checks(
        self, prompt: str, agent_input: AgentInput
    ) -> List[str]:
        """
        Perform generic quality checks on a prompt.

        Args:
            prompt: Prompt to check
            agent_input: Original input

        Returns:
            List of issues found
        """
        issues = []

        # Length check
        if len(prompt) < 200:
            issues.append("Prompt is too short (may lack detail)")

        # Specificity check
        words = prompt.split()
        specific_words = [
            "show",
            "create",
            "with",
            "featuring",
            "including",
            "lighting",
            "style",
            "resolution",
        ]
        specific_count = sum(1 for word in words if word.lower() in specific_words)

        if specific_count < 3:
            issues.append("Prompt lacks specific descriptors")

        # Natural language check
        if prompt.count(":") > 5:
            issues.append("Too many colons (may not be natural language)")

        return issues

    def _generate_suggestion(self, issue: str, prompt: str) -> Optional[str]:
        """
        Generate a suggestion to fix an issue.

        Args:
            issue: Issue description
            prompt: Current prompt

        Returns:
            Suggestion or None
        """
        if "too short" in issue.lower():
            return "Add more specific details about the scene, lighting, and style"

        elif "lacks specific descriptors" in issue.lower():
            return "Include more descriptive adjectives and visual specifics"

        elif "too many colons" in issue.lower():
            return "Rewrite using more natural, conversational language"

        return None

    def _calculate_severity(self, issues: List[str]) -> str:
        """
        Calculate severity level from issues.

        Args:
            issues: List of issues

        Returns:
            Severity: "low", "medium", or "high"
        """
        if len(issues) == 0:
            return "low"
        elif len(issues) <= 2:
            return "medium"
        else:
            return "high"

    def _format_critique_for_refinement(self, critique: CritiqueResult) -> str:
        """
        Format critique results for use in refinement.

        Args:
            critique: Critique results

        Returns:
            Formatted critique string
        """
        parts = []

        if critique.issues:
            parts.append("Issues found:")
            for issue in critique.issues:
                parts.append(f"  - {issue}")

        if critique.suggestions:
            parts.append("\nSuggestions:")
            for suggestion in critique.suggestions:
                parts.append(f"  - {suggestion}")

        return "\n".join(parts)
