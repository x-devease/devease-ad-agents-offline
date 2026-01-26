"""
Feature-to-Prompt Module

Converts recommended features to prompt instructions.
"""

from .feature_loader import load_recommended_features
from .feature_validation import (
    ValidationResult,
    compare_features,
    get_failing_features,
    validate_generation_batch,
)
from .output_formatter import format_output


# Use simplified converter that works with recommendations.json
try:
    from .converter import convert_features_to_prompts
except ImportError:
    # Fall back to simplified version if converter_core not available
    from .converter_simple import convert_features_to_prompts

__all__ = [
    "load_recommended_features",
    "convert_features_to_prompts",
    "format_output",
    "ValidationResult",
    "compare_features",
    "validate_generation_batch",
    "get_failing_features",
]
