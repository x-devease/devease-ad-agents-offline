"""
Feature Extractor Module

A modular package for extracting image features using GPT-4 Vision API.
"""

from .extract import (
    add_roas_to_features,
    extract_batch_features,
    extract_single_image_features,
)
from .extractors.gpt4_feature_extractor import GPT4FeatureExtractor

# pylint: disable=import-error
from .lib import (
    create_synthetic_roas,
    load_ad_data,
    load_feature_data,
    merge_features_with_roas,
    parse_creative_info_from_filename,
    parse_creative_info_from_filenames,
    parse_roas_value,
)
from .interactions import (
    create_interaction_features,
    discover_interactions,
    get_interaction_features,
)
from .transformers.gpt4_feature_transformer import (
    apply_feature_weights,
    convert_to_features,
)

__all__ = [
    # Feature Extractor
    "GPT4FeatureExtractor",
    # Feature transformers
    "apply_feature_weights",
    "convert_to_features",
    # Feature extraction
    "extract_batch_features",
    "extract_single_image_features",
    # ROAS integration
    "add_roas_to_features",
    # Feature linking
    "create_synthetic_roas",
    "load_ad_data",
    "load_feature_data",
    "merge_features_with_roas",
    "parse_creative_info_from_filename",
    "parse_creative_info_from_filenames",
    "parse_roas_value",
    # Interaction features
    "create_interaction_features",
    "discover_interactions",
    "get_interaction_features",
]

__version__ = "1.0.0"
