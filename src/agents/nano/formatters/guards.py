"""
Anti-Hallucination Guard for Nano Banana Pro Agent.

Adds constraints to prevent unwanted additions and hallucinations.
"""

from __future__ import annotations

import logging
from typing import List

from src.agents.nano.core.types import PromptConstraint


logger = logging.getLogger(__name__)


class AntiHallucinationGuard:
    """
    Add anti-hallucination constraints to prompts.

    Prevents NB Pro from adding unwanted elements or making
    unauthorized changes.
    """

    def generate_constraints(
        self,
        has_product_reference: bool = False,
        has_person_reference: bool = False,
        intent = None,
    ) -> List[PromptConstraint]:
        """Generate appropriate anti-hallucination constraints."""

        constraints = []

        # Product preservation
        if has_product_reference:
            constraints.append(PromptConstraint(
                constraint_type="preserve_exact",
                description="Do NOT modify product appearance, colors, or features",
                subject="product",
            ))

        # Person preservation
        if has_person_reference:
            constraints.append(PromptConstraint(
                constraint_type="preserve_exact",
                description="Keep person's facial features exactly the same",
                subject="person",
            ))

        # No unauthorized additions
        constraints.append(PromptConstraint(
            constraint_type="do_not_add",
            description="Do NOT add elements not specified in the prompt",
        ))

        # No redesigns
        constraints.append(PromptConstraint(
            constraint_type="do_not_add",
            description="Do NOT create variations or redesigns",
        ))

        # Brand compliance
        constraints.append(PromptConstraint(
            constraint_type="brand_compliance",
            description="Follow brand guidelines exactly",
        ))

        return constraints

    def format_constraints(self, constraints: List[PromptConstraint]) -> str:
        """Format constraints as text for the prompt."""

        if not constraints:
            return ""

        lines = ["ANTI-HALLUCINATION CONSTRAINTS:"]

        for constraint in constraints:
            if constraint.subject:
                lines.append(f"  - [{constraint.subject}] {constraint.description}")
            else:
                lines.append(f"  - {constraint.description}")

        return "\n".join(lines)
