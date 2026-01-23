"""Core recommender system."""

from src.adset.generator.core.recommender import (
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

__all__ = [
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
]
