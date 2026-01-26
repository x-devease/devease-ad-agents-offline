"""
Feature Loader Module

Loads recommended features from various sources (JSON files, dicts, etc.).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .recommendations_loader import load_recommendations_json


logger = logging.getLogger(__name__)


def load_recommended_features(
    source: Union[str, Path, Dict[str, Any]],
    scorer_repo_path: Optional[Path] = None,
    min_importance: float = 0.0,
    min_confidence: Optional[str] = None,
    include_interactions: bool = False,
) -> Dict[str, Any]:
    """
    Load recommended features from various sources.

    Args:
        source: Can be:
            - Path to JSON file (str or Path)
            - Dictionary with feature data
        scorer_repo_path: Optional path to creative scorer repo (for finding default files)

    Returns:
        Dict with:
            - 'recommended_features': List[str] - List of feature names (positive features)
            - 'negative_features': List[str] - List of feature names to avoid (optional)
            - 'feature_importance': Dict[str, float] - Feature importance scores
            - 'negative_feature_importance': Dict[str, float] - Negative
                feature importance scores (optional)
            - 'feature_values': Dict[str, str] - Explicit feature value
                mappings (optional, overrides auto-determination)
            - 'negative_feature_values': Dict[str, str] - Explicit negative
                feature value mappings (optional)
            - 'source': str - Source identifier

    Raises:
        FileNotFoundError: If source is a file path that doesn't exist
        ValueError: If source format is invalid
    """
    # Handle dict input
    if isinstance(source, dict):
        # Check if it's recommendations.json format
        if "recommendations" in source and isinstance(
            source["recommendations"], list
        ):
            logger.info("Detected recommendations.json format")
            return load_recommendations_json(
                source,
                min_importance=min_importance,
                min_confidence=min_confidence,
                include_interactions=include_interactions,
            )
        return _validate_and_normalize_feature_dict(source)
    # Handle file path
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        # If file doesn't exist, try to find it in scorer repo
        if not file_path.exists() and scorer_repo_path:
            possible_paths = [
                scorer_repo_path
                / "data"
                / "roas_analysis"
                / "reduced_model_results.json",
                scorer_repo_path
                / "data"
                / "unified_analysis"
                / "reduced_model_results.json",
                scorer_repo_path / file_path.name,
            ]

            for path in possible_paths:
                if path.exists():
                    file_path = path
                    break
            else:
                raise FileNotFoundError(
                    f"Feature file not found: {source}\n"
                    f"Tried paths:\n"
                    + "\n".join(f"  - {p}" for p in possible_paths)
                )

        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        logger.info("Loading recommended features from: %s", file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}") from e
        # Check if it's recommendations.json format
        if "recommendations" in data and isinstance(
            data["recommendations"], list
        ):
            logger.info("Detected recommendations.json format")
            result = load_recommendations_json(
                data,
                min_importance=min_importance,
                min_confidence=min_confidence,
                include_interactions=include_interactions,
            )
            result["source"] = str(file_path)
            return result
        # Use existing format handler
        result = _validate_and_normalize_feature_dict(data)
        result["source"] = str(file_path)
        return result

    raise ValueError(f"Unsupported source type: {type(source)}")


def _validate_and_normalize_feature_dict(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and normalize feature data dictionary.

    Args:
        data: Raw feature data dict

    Returns:
        Normalized feature data dict

    Raises:
        ValueError: If data format is invalid
    """
    # Extract recommended features
    recommended_features = (
        data.get("recommended_features")
        or data.get("selected_features")
        or data.get("features", [])
    )

    if not isinstance(recommended_features, list):
        raise ValueError(
            f"recommended_features must be a list, got {type(recommended_features)}"
        )

    if not all(isinstance(f, str) for f in recommended_features):
        raise ValueError("All feature names must be strings")
    # Extract feature importance
    feature_importance = data.get("feature_importance") or data.get(
        "importance", {}
    )

    if not isinstance(feature_importance, dict):
        # Try to convert from list format if present
        if isinstance(feature_importance, list):
            feature_importance = {}
            logger.warning(
                "feature_importance is a list, converting to dict (using indices)"
            )
        else:
            logger.warning(
                "feature_importance must be a dict, got %s. Using empty dict.",
                type(feature_importance),
            )
            feature_importance = {}
    # Ensure all importance values are floats
    normalized_importance = {}
    for feature, importance in feature_importance.items():
        try:
            normalized_importance[str(feature)] = float(importance)
        except (ValueError, TypeError):
            logger.warning(
                "Invalid importance value for %s: %s. Skipping.",
                feature,
                importance,
            )
    # Extract negative features (features to avoid)
    negative_features = (
        data.get("negative_features")
        or data.get("features_to_avoid")
        or data.get("avoid_features", [])
    )

    if not isinstance(negative_features, list):
        negative_features = []

    if not all(isinstance(f, str) for f in negative_features):
        logger.warning(
            "Some negative feature names are not strings, filtering them out"
        )
        negative_features = [f for f in negative_features if isinstance(f, str)]
    # Extract negative feature importance
    negative_feature_importance = data.get(
        "negative_feature_importance"
    ) or data.get("negative_importance", {})

    if not isinstance(negative_feature_importance, dict):
        negative_feature_importance = {}
    # Ensure all negative importance values are floats
    normalized_negative_importance = {}
    for feature, importance in negative_feature_importance.items():
        try:
            normalized_negative_importance[str(feature)] = float(importance)
        except (ValueError, TypeError):
            logger.warning(
                "Invalid negative importance value for %s: %s. Skipping.",
                feature,
                importance,
            )
    # Extract explicit feature value mappings (optional)
    feature_values = data.get("feature_values") or data.get(
        "feature_value_mapping", {}
    )
    if not isinstance(feature_values, dict):
        feature_values = {}
    # Extract explicit negative feature value mappings (optional)
    negative_feature_values = data.get("negative_feature_values") or data.get(
        "negative_feature_value_mapping", {}
    )
    if not isinstance(negative_feature_values, dict):
        negative_feature_values = {}

    return {
        "recommended_features": recommended_features,
        "negative_features": negative_features,
        "feature_importance": normalized_importance,
        "negative_feature_importance": normalized_negative_importance,
        "feature_values": feature_values,
        "negative_feature_values": negative_feature_values,
        "source": data.get("source", "unknown"),
    }
