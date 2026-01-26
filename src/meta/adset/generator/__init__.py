"""Recommenders module."""

# Core recommender system
from src.meta.adset.generator.core import (
    ConfigurableRecommender,
    ROASRecommender,
    MetricRecommender,
    MetricConfig,
    RecommendationStrategy,
    create_recommender,
    create_roas_recommender,
    create_cpa_recommender,
    create_cpc_recommender,
    create_ctr_recommender,
    create_percentile_recommender,
    create_custom_recommender,
    default_opportunity_maximize,
    default_opportunity_minimize,
    roi_opportunity,
)

# Detection
from src.meta.adset.generator.detection import MistakeDetector, Issue

# Analyzers
from src.meta.adset.generator.analyzers import (
    OpportunitySizer,
    ShopifyAnalyzer,
    AdvantageConstraintGenerator,
)

# Segmentation
from src.meta.adset.generator.segmentation import Segmenter

# Generation
from src.meta.adset.generator.generation import (
    AudienceRecommender,
    AudienceAggregator,
    AudienceSegment,
    CreativeCompatibility,
)

__all__ = [
    # Core
    "ConfigurableRecommender",
    "ROASRecommender",
    "MetricRecommender",
    "MetricConfig",
    "RecommendationStrategy",
    "create_recommender",
    "create_roas_recommender",
    "create_cpa_recommender",
    "create_cpc_recommender",
    "create_ctr_recommender",
    "create_percentile_recommender",
    "create_custom_recommender",
    "default_opportunity_maximize",
    "default_opportunity_minimize",
    "roi_opportunity",
    # Detection
    "MistakeDetector",
    "Issue",
    # Analyzers
    "OpportunitySizer",
    "ShopifyAnalyzer",
    "AdvantageConstraintGenerator",
    # Segmentation
    "Segmenter",
    # Generation
    "AudienceRecommender",
    "AudienceAggregator",
    "AudienceSegment",
    "CreativeCompatibility",
]
