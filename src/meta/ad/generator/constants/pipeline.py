"""
Pipeline constants.

Centralized constants for prompt types, recommendations, and other
commonly-used strings throughout the pipeline.
"""

from enum import Enum


class PromptType(str, Enum):
    """Prompt generation types."""

    ORCHESTRATOR = "orchestrator"
    LLM_ENHANCED = "llm_enhanced"
    TECHNICAL = "technical"


class Recommendation(str, Enum):
    """Scoring recommendation values."""

    APPROVE = "approve"
    REJECT = "reject"
