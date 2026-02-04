"""
Recommendation-based Creative Generation Module.

This module provides:
- RecommendationLoader: Load visual recommendations from the scorer repository
- create_product_context: Create product context for prompt generation
- CreativePipeline: End-to-end orchestrator for recommendation-based generation
- FeatureReproductionTracker: Track and validate feature reproduction (formula → prompt → image)
"""

from .ad_miner_adapter import (
    convert_patterns_to_visual_formula,
    load_patterns_as_visual_formula,
    # Backward compatibility aliases
    convert_recommendations_to_visual_formula,
    load_recommendations_as_visual_formula,
)
from .feature_reproduction import FeatureReproductionTracker
from .pipeline import CreativePipeline
from .product_context import create_product_context
from .recommendation_loader import (
    RecommendationLoader,
    load_visual_recommendation,
)


__all__ = [
    "RecommendationLoader",
    "load_visual_recommendation",
    "create_product_context",
    "CreativePipeline",
    "FeatureReproductionTracker",
    # New function names
    "convert_patterns_to_visual_formula",
    "load_patterns_as_visual_formula",
    # Backward compatibility
    "convert_recommendations_to_visual_formula",
    "load_recommendations_as_visual_formula",
]
