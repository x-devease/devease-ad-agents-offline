"""
Core data types for Nano Banana Pro Prompt Enhancement Agent.

Defines all input/output types and internal data structures.

Note: AgentInput and AgentOutput are now defined in the framework
types module for reuse across different agent domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import generic types from framework
from src.agents.framework.core.types import AgentInput, AgentOutput


class PromptCategory(Enum):
    """Categories of input prompts."""

    ULTRA_SIMPLE = "ultra_simple"  # "Make an ad for our mop"
    BASIC_DIRECTION = "basic_direction"  # "Show someone cleaning..."
    SPECIFIC_REQUEST = "specific_request"  # "Product photo on white, 4K"
    COMPARATIVE = "comparative"  # "Compare our mop to competitors"
    SEQUENTIAL = "sequential"  # "Show before/during/after story"
    TECHNICAL = "technical"  # "Generate technical diagram"


class PromptIntent(Enum):
    """Intent of the prompt - what type of output is desired."""

    PRODUCT_PHOTOGRAPHY = "product_photography"
    LIFESTYLE_ADVERTISEMENT = "lifestyle_advertisement"
    COMPARATIVE_INFOGRAPHIC = "comparative_infographic"
    STORYBOARD_SEQUENCE = "storyboard_sequence"
    TECHNICAL_DIAGRAM = "technical_diagram"
    EDIT_REFINEMENT = "edit_refinement"
    BRAND_ASSET_GENERATION = "brand_asset_generation"


class Resolution(Enum):
    """Output resolution specifications."""

    K1 = "1K"  # 1024x1024
    K2 = "2K"  # 2048x1080 or similar
    K4 = "4K"  # 3840x2160


class LightingStyle(Enum):
    """Common lighting styles."""

    NATURAL_DAYLIGHT = "natural_daylight"
    MORNING_SUNLIGHT = "morning_sunlight"
    GOLDEN_HOUR = "golden_hour"
    OVERCAST = "overcast"
    STUDIO_SOFT = "studio_soft"
    STUDIO_DRAMATIC = "studio_dramatic"
    NEON_CYBERPUNK = "neon_cyberpunk"


class CameraStyle(Enum):
    """Camera and perspective styles."""

    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    BIRDS_EYE = "birds_eye"
    DUTCH_ANGLE = "dutch_angle"
    MACRO_CLOSEUP = "macro_closeup"
    WIDE_ESTABLISHING = "wide_establishing"


@dataclass
class ProductContext:
    """Product information from context enrichment."""

    name: str
    category: str
    key_features: List[str]
    materials: List[str]
    colors: List[str]
    brand_colors: Optional[List[str]] = None
    reference_images: Optional[List[str]] = None


@dataclass
class BrandGuidelines:
    """Brand identity and style guidelines."""

    brand_name: str
    primary_colors: List[str]
    secondary_colors: List[str]
    typography_system: str
    visual_language: str
    logo_requirements: Optional[str] = None


@dataclass
class TechnicalSpecs:
    """Technical specifications for the output."""

    resolution: Resolution = Resolution.K2
    lighting_style: Optional[LightingStyle] = None
    camera_style: Optional[CameraStyle] = None
    style_declaration: Optional[str] = None  # e.g., "professional product photography"
    aspect_ratio: str = "16:9"


@dataclass
class ThinkingBlock:
    """Thinking/reasoning block for the prompt."""

    analysis: str  # What is this request about?
    techniques: List[str]  # Which NB techniques to apply
    risks: List[str]  # What could go wrong?
    mitigation: List[str]  # How to prevent issues
    context: str  # For whom, why

    def format(self) -> str:
        """Format thinking block as text."""
        parts = [
            f"<thinking>",
            f"Analysis: {self.analysis}",
            f"",
            f"Techniques to apply:",
            *[f"  - {t}" for t in self.techniques],
            f"",
            f"Risks:",
            *[f"  - {r}" for r in self.risks],
            f"",
            f"Mitigation:",
            *[f"  - {m}" for m in self.mitigation],
            f"",
            f"Context: {self.context}",
            f"</thinking>",
        ]
        return "\n".join(parts)


@dataclass
class PromptConstraint:
    """Anti-hallucination constraint."""

    constraint_type: str  # "do_not_add", "preserve_exact", etc.
    description: str
    subject: Optional[str] = None  # What this applies to


@dataclass
class AppliedTechnique:
    """A Nano Banana Pro technique applied to the prompt."""

    technique_name: str
    description: str
    prompt_addition: str  # The actual text added to the prompt


@dataclass
class IntermediatePrompt:
    """Intermediate prompt during the enhancement pipeline."""

    stage: str  # Which pipeline stage created this
    prompt_content: str  # The prompt at this stage
    metadata: Dict[str, Any] = field(default_factory=dict)
    techniques_applied: List[str] = field(default_factory=list)
