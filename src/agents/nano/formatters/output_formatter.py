"""
Output Formatter for Nano Banana Pro Agent.

Assembles the final prompt with all components and generates metadata.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any
from datetime import datetime

from src.agents.nano.core.types import (
    AgentOutput,
    AgentInput,
    ThinkingBlock,
    AppliedTechnique,
    PromptConstraint,
    IntermediatePrompt,
)


logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Format the final output prompt and generate metadata.

    Assembles all components into the final high-fidelity prompt.
    """

    def format(
        self,
        thinking_block: ThinkingBlock,
        intermediate_prompt: IntermediatePrompt,
        techniques_applied: List[AppliedTechnique],
        technical_specs_text: str,
        constraints_text: str,
        agent_input: AgentInput,
        confidence: float,
        processing_time_ms: int,
    ) -> AgentOutput:
        """
        Format the final output.

        Args:
            thinking_block: The thinking block
            intermediate_prompt: The intermediate prompt with NL content
            techniques_applied: List of applied techniques
            technical_specs_text: Formatted technical specifications
            constraints_text: Formatted anti-hallucination constraints
            agent_input: Original input
            confidence: Quality confidence score
            processing_time_ms: Processing time in milliseconds

        Returns:
            Complete AgentOutput ready for NB Pro
        """

        # Assemble final prompt
        enhanced_prompt = self._assemble_prompt(
            thinking_block if agent_input.enable_thinking else None,
            intermediate_prompt.prompt_content,
            technical_specs_text,
            constraints_text,
        )

        # Extract technique names
        techniques_used = [t.technique_name for t in techniques_applied]

        # Generate explanation
        explanation = self._generate_explanation(
            agent_input,
            techniques_applied,
            confidence,
        )

        return AgentOutput(
            enhanced_prompt=enhanced_prompt,
            thinking_block=thinking_block,
            applied_techniques=techniques_applied,
            constraints=self._extract_constraints(constraints_text),
            confidence=confidence,
            explanation=explanation,
            processing_time_ms=processing_time_ms,
            techniques_used=techniques_used,
            request_id=agent_input.request_id,
            timestamp=datetime.now(),
        )

    def _assemble_prompt(
        self,
        thinking_block: ThinkingBlock,
        nl_content: str,
        technical_specs: str,
        constraints: str,
    ) -> str:
        """Assemble all components into final prompt."""

        parts = []

        # Add thinking block (if enabled)
        if thinking_block:
            parts.append(thinking_block.format())

        # Add main natural language content
        parts.append(nl_content)

        # Add technical specifications
        if technical_specs:
            parts.append(technical_specs)

        # Add constraints
        if constraints:
            parts.append(constraints)

        return "\n\n".join(parts)

    def _extract_constraints(self, constraints_text: str) -> List:
        """Extract constraints from formatted text."""

        # Return empty list for now
        return []

    def _generate_explanation(
        self,
        agent_input: AgentInput,
        techniques_applied: List[AppliedTechnique],
        confidence: float,
    ) -> str:
        """Generate human-readable explanation of what was done."""

        parts = []

        # What was transformed
        parts.append(
            f"Transformed generic prompt into high-fidelity Nano Banana Pro prompt."
        )

        # Techniques applied
        if techniques_applied:
            technique_names = [t.technique_name for t in techniques_applied]
            parts.append(f"Applied {len(technique_names)} techniques: {', '.join(technique_names)}")

        # Confidence
        confidence_pct = int(confidence * 100)
        parts.append(f"Quality confidence: {confidence_pct}%")

        return " ".join(parts) + "."
