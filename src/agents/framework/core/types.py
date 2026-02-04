"""
Framework core types and data structures.

This module defines the core data structures used across the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import hashlib


class Resolution(Enum):
    """Output resolution specifications."""

    K1 = "1K"  # 1024x1024
    K2 = "2K"  # 2048x1080 or similar
    K4 = "4K"  # 3840x2160


@dataclass
class AgentInput:
    """
    Generic input to the Prompt Enhancement Agent.

    This is a framework-level type that can be used by any domain adapter.
    Domain-specific adapters may extend this with additional context.
    """

    # Core input
    generic_prompt: str  # The user's generic prompt

    # Optional context (can be enriched if not provided)
    product_context: Optional[Any] = None
    brand_guidelines: Optional[Any] = None
    reference_images: Optional[List[str]] = None  # Paths to reference images

    # Optional preferences
    preferred_resolution: Optional[Resolution] = None
    enable_thinking: bool = True
    target_audience: Optional[str] = None
    emotion_goal: Optional[str] = None

    # Metadata
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentOutput:
    """
    Generic output from the Prompt Enhancement Agent.

    This is a framework-level type that can be used by any domain adapter.
    """

    # Core output
    enhanced_prompt: str  # The enhanced prompt

    # What was applied
    techniques_used: List[str] = field(default_factory=list)

    # Classification (domain-specific)
    detected_category: Optional[str] = None
    detected_intent: Optional[str] = None

    # Metadata
    confidence: float = 0.0  # 0.0 to 1.0
    processing_time_ms: int = 0
    thinking_block: Optional[Any] = None

    # Legacy backward compatibility fields
    applied_techniques: List[Any] = field(default_factory=list)  # Legacy: List[AppliedTechnique]
    explanation: Optional[str] = None  # Legacy: explanation text
    constraints: Optional[str] = None  # Legacy: constraints text

    # Request tracking
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FrameworkConfig:
    """Configuration for the framework."""

    examples_db_path: str = "data/agents/examples.json"
    memory_db_path: str = "data/agents/memory.json"
    enable_reflexion: bool = True
    enable_memory: bool = True
    max_reflexion_iterations: int = 2
    memory_max_entries: int = 1000
    quality_threshold: float = 0.7
    log_level: str = "INFO"


@dataclass
class GroundingExample:
    """
    A grounding example for few-shot learning.

    Examples show the model what good outputs look like.
    """

    input_prompt: str
    output_prompt: str
    domain: str
    category: str
    intent: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    example_id: str = field(default="")

    def __post_init__(self):
        """Generate example ID if not provided."""
        if not self.example_id:
            content = f"{self.input_prompt}:{self.output_prompt}:{self.domain}"
            self.example_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class MemoryEntry:
    """
    A memory entry from past prompt enhancements.

    Memory enables the framework to learn from past interactions.
    """

    input_prompt: str
    enhanced_prompt: str
    domain: str
    detected_category: str
    detected_intent: str
    confidence: float
    techniques_used: List[str]
    user_feedback: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    entry_id: str = field(default="")

    def __post_init__(self):
        """Generate entry ID if not provided."""
        if not self.entry_id:
            content = f"{self.input_prompt}:{self.enhanced_prompt}:{self.created_at}"
            self.entry_id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class QualityCheck:
    """
    Results of a quality verification check.

    Contains issues found and overall quality metrics.
    """

    passes: bool
    confidence: float
    issues: List[str]
    specificity_score: float = 0.0
    pattern_consistency: float = 0.0
    natural_language_score: float = 0.0
    completeness_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: str):
        """Add an issue to the check results."""
        self.issues.append(issue)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passes": self.passes,
            "confidence": self.confidence,
            "issues": self.issues,
            "specificity_score": self.specificity_score,
            "pattern_consistency": self.pattern_consistency,
            "natural_language_score": self.natural_language_score,
            "completeness_score": self.completeness_score,
            "metadata": self.metadata,
        }
