"""GPT-4 Feature Transformer for converting API responses to feature vectors.

This module provides functions to transform GPT-4 Vision API responses into
feature vectors and then into weighted feature values for model prediction.

The transformation process consists of two main steps:
1. Convert GPT-4 Vision API response to feature dictionary (29 features)
2. Convert feature dictionary to weighted values using feature weights
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def convert_to_features(gpt_response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert GPT-4 Vision API response to feature dictionary.

    This function maps the structured GPT-4 Vision API response (with sections
    like visual_elements, composition, lighting, etc.) to a flat dictionary
    of 29 binary/categorical features used by the model.

    Args:
        gpt_response: Dictionary containing GPT-4 Vision API response with
            sections like visual_elements, composition, lighting,
            content_elements, technical_quality, overall_assessment,
            social_proof, scene_context.

    Returns:
        Dictionary with 29 feature names as keys and their corresponding
        values (typically 0.0 or 1.0 for binary features, or actual values
        for numeric/categorical features).

    Raises:
        ValueError: If the response format is invalid or missing required
            fields.
        KeyError: If required configuration or response fields are missing.
    """
    # Load feature configuration
    features_config = ConfigManager.get_config(None, "ad/recommender/gpt4/features.yaml")
    features_list = features_config.get("features", [])

    feature_dict = {}

    for feature_config in features_list:
        feature_name = feature_config.get("name")
        source_section = feature_config.get("source_section")
        source_field = feature_config.get("source_field")
        mapping_type = feature_config.get("mapping_type")

        if not all([feature_name, source_section, source_field, mapping_type]):
            logger.warning(
                "Skipping feature %s: missing configuration", feature_name
            )
            continue

        # Get source data from GPT response
        source_data = gpt_response.get(source_section, {})
        source_value = source_data.get(source_field)

        # Handle missing source data
        if source_value is None:
            feature_dict[feature_name] = 0.0
            continue

        # Apply mapping based on mapping_type
        if mapping_type == "value_match":
            match_value = feature_config.get("match_value")
            feature_dict[feature_name] = (
                1.0 if source_value == match_value else 0.0
            )

        elif mapping_type == "value_match_any":
            match_values = feature_config.get("match_values", [])
            feature_dict[feature_name] = (
                1.0 if source_value in match_values else 0.0
            )

        elif mapping_type == "text_contains":
            search_text = feature_config.get("search_text", "")
            if isinstance(source_value, list):
                feature_dict[feature_name] = (
                    1.0
                    if any(
                        search_text.lower() in str(item).lower()
                        for item in source_value
                    )
                    else 0.0
                )
            else:
                feature_dict[feature_name] = (
                    1.0
                    if search_text.lower() in str(source_value).lower()
                    else 0.0
                )

        elif mapping_type == "text_contains_all":
            search_texts = feature_config.get("search_texts", [])
            if isinstance(source_value, list):
                source_str = " ".join(
                    str(item).lower() for item in source_value
                )
                feature_dict[feature_name] = (
                    1.0
                    if all(text.lower() in source_str for text in search_texts)
                    else 0.0
                )
            else:
                source_str = str(source_value).lower()
                feature_dict[feature_name] = (
                    1.0
                    if all(text.lower() in source_str for text in search_texts)
                    else 0.0
                )

        elif mapping_type == "list_length":
            if isinstance(source_value, list):
                feature_dict[feature_name] = float(len(source_value))
            else:
                feature_dict[feature_name] = 0.0

        elif mapping_type == "numeric":
            try:
                feature_dict[feature_name] = float(source_value)
            except (ValueError, TypeError):
                feature_dict[feature_name] = 0.0

        elif mapping_type == "direct_value":
            feature_dict[feature_name] = source_value

        else:
            logger.warning(
                "Unknown mapping_type '%s' for feature %s",
                mapping_type,
                feature_name,
            )
            feature_dict[feature_name] = 0.0

    return feature_dict


def apply_feature_weights(
    features: Dict[str, Any], weights_file: str, cpc_value: float = None
) -> Dict[str, Any]:
    """Apply feature weights to feature dictionary.

    This function loads feature weights from a JSON file and applies them to
    the feature dictionary. It also handles CPC transformation if a CPC value
    is provided.

    Args:
        features: Dictionary of feature names and values.
        weights_file: Path to JSON file containing feature weights.
        cpc_value: Optional CPC (Cost Per Click) value for CPC_transformed
            feature calculation. If provided, will calculate and add
            CPC_transformed feature.

    Returns:
        Dictionary with the same feature names as input, but with weighted
        values applied. Values are typically in the range [0.0, 1.0] after
        weight application.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        KeyError: If required fields are missing from the weights file.
    """
    weights_path = Path(weights_file)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    with open(weights_path, "r", encoding="utf-8") as weights_file_handle:
        weights_data = json.load(weights_file_handle)

    weights = weights_data.get("weights", {})

    # Apply weights to features
    weighted_features = {}
    for feature_name, feature_value in features.items():
        weight = weights.get(feature_name, 1.0)
        weighted_features[feature_name] = feature_value * weight

    # Handle CPC transformation if CPC value is provided
    if cpc_value is not None:
        # CPC_transformed = 1 / (1 + CPC)
        cpc_transformed = 1.0 / (1.0 + cpc_value)
        weighted_features["CPC_transformed"] = cpc_transformed

    return weighted_features
