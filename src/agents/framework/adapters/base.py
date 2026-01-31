"""
Base adapter interface for domain-specific prompt enhancement.

Domain adapters must implement this interface to work with the generic framework.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

from src.agents.framework.core.types import AgentInput


logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for adapter."""

    domain: str
    config_path: Optional[str] = None
    options: Dict[str, Any] = None


class BaseAdapter(ABC):
    """
    Abstract base class for domain-specific adapters.

    Each domain provides an adapter that implements domain-specific logic
    while using the generic framework for orchestration.
    """

    def __init__(self, config: AdapterConfig):
        """
        Initialize adapter with configuration.

        Args:
            config: Adapter configuration
        """
        self.config = config
        self.domain = config.domain
        logger.info(f"Initialized {self.domain} adapter")

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier (e.g., 'nano')."""
        pass

    @domain.setter
    def domain(self, value: str):
        """Set domain identifier."""
        self._domain = value

    @abstractmethod
    def parse_input(self, generic_prompt: str) -> Tuple[str, str]:
        """
        Parse input into category and intent.

        Args:
            generic_prompt: User's generic input prompt

        Returns:
            (category, intent) - Domain-specific categorization

        Example:
            Nano: ("ultra_simple", "product_photography")
        """
        pass

    @abstractmethod
    def enrich_context(self, agent_input: AgentInput) -> AgentInput:
        """
        Enrich agent input with domain-specific context.

        Args:
            agent_input: Original agent input

        Returns:
            Enriched agent input with context added

        Example:
            Nano: Add product/brand context from database
        """
        pass

    @abstractmethod
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
            examples: Relevant grounding examples

        Returns:
            Thinking block as string
        """
        pass

    @abstractmethod
    def build_prompt(self, agent_input: AgentInput, category: str, intent: str) -> str:
        """
        Build domain-specific prompt.

        Args:
            agent_input: User input with context
            category: Detected category
            intent: Detected intent

        Returns:
            Built prompt as string
        """
        pass

    @abstractmethod
    def apply_techniques(self, prompt: str, thinking: str) -> str:
        """
        Apply domain-specific techniques to prompt.

        Args:
            prompt: Base prompt
            thinking: Thinking block with technique selection

        Returns:
            Enhanced prompt with techniques applied
        """
        pass

    @abstractmethod
    def refine_prompt(
        self, prompt: str, critique: str, agent_input: AgentInput
    ) -> str:
        """
        Refine prompt based on critique (domain-specific logic).

        Args:
            prompt: Current prompt
            critique: Critique from Reflexion engine
            agent_input: Original user input

        Returns:
            Refined prompt
        """
        pass

    def validate_domain_specific(self, prompt: str) -> List[str]:
        """
        Validate domain-specific requirements.

        Override in subclasses for domain-specific validation.

        Args:
            prompt: Prompt to validate

        Returns:
            List of validation issues (empty if valid)
        """
        return []

    def compute_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Compute similarity between prompts (for memory retrieval).

        Override in subclasses for domain-specific similarity.

        Args:
            prompt1: First prompt
            prompt2: Second prompt

        Returns:
            Similarity score between 0 and 1
        """
        return self._keyword_similarity(prompt1, prompt2)

    def _keyword_similarity(self, prompt1: str, prompt2: str) -> float:
        """Default keyword-based similarity."""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0
