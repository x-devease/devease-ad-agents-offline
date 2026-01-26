"""
Recommendations.json Loader (Legacy Scorer Format)

Loads and converts recommendations.json format from creative scorer repository
into the format expected by convert_features_to_prompts().

Note:
    This module is for the creative scorer offline repository JSON format.
    For ad/recommender markdown format (primary), use the ad_recommender_adapter
    which handles MD to visual formula conversion.

Path formats:
    - Scorer repo (JSON): devease-creative-scorer-offline/data/headroom_analysis/recommendations.json
    - Ad/recommender (MD, primary): config/ad/recommender/{customer}/{platform}/{date}/recommendations.md
"""

# flake8: noqa
import logging
from typing import Any, Dict, Optional


# Import centralized normalization from feature registry
try:
    from src.orchestrator.feature_registry import normalize_feature_value
except ImportError:
    # Fallback if module structure changes
    normalize_feature_value = None


logger = logging.getLogger(__name__)


def _select_negative_value_from_value_comparison(
    *,
    value_comparison: Any,
    recommended_value: str,
    feature_name: Optional[str] = None,
    min_count: int = 3,
) -> Optional[str]:
    """
    Pick a robust "avoid" value from scorer value_comparison.

    Strategy:
    - Consider only values with count >= min_count (default: 3, less restrictive than before)
    - Pick the value with the lowest `best_metric` (worst performing)
    - Do not return the recommended_value itself
    - If no candidates meet min_count, retry with min_count=1 as fallback

    Args:
        value_comparison: Dict of value -> {count, best_metric, ...}
        recommended_value: The recommended value to exclude
        feature_name: Optional feature name for registry-based normalization
        min_count: Minimum count threshold for statistical robustness (default: 3)

    If data is sparse or malformed, return None and let callers fall back
    to negative_signals.
    """
    if not isinstance(value_comparison, dict) or not value_comparison:
        return None

    rec_norm = normalize_value(str(recommended_value), feature_name)

    def _extract_candidates(min_threshold: int) -> list[tuple[float, str]]:
        """Extract candidates with given minimum count threshold."""
        candidates: list[tuple[float, str]] = []
        for k, val_info in value_comparison.items():
            try:
                val_norm = normalize_value(str(k), feature_name)
                if val_norm == rec_norm:
                    continue
                if not isinstance(val_info, dict):
                    continue
                count = int(val_info.get("count", 0) or 0)
                if count < min_threshold:
                    continue
                best_metric = val_info.get("best_metric", None)
                if best_metric is None:
                    continue
                score = float(best_metric)
                candidates.append((score, val_norm))
            except Exception:  # pylint: disable=broad-exception-caught
                continue
        return candidates

    # Try with requested min_count first
    candidates = _extract_candidates(min_count)
    # Fallback: try with min_count=1 if no candidates found
    if not candidates and min_count > 1:
        logger.debug(
            "No candidates found for %s with min_count=%d, trying min_count=1",
            feature_name or "unknown",
            min_count,
        )
        candidates = _extract_candidates(1)

    if not candidates:
        logger.debug(
            "No valid negative value candidates found for %s",
            feature_name or "unknown",
        )
        return None

    candidates.sort(key=lambda x: x[0])  # smallest metric = worst performing
    selected = candidates[0][1]
    logger.debug(
        "Selected negative value '%s' for %s (metric: %.2f, %d candidates)",
        selected,
        feature_name or "unknown",
        candidates[0][0],
        len(candidates),
    )
    return selected


def normalize_value(value: str, feature_name: Optional[str] = None) -> str:
    """
    Normalize feature value using centralized feature registry.

    This function provides a unified interface for feature value normalization.
    It delegates to the feature registry when available, with a fallback to
    basic normalization for backward compatibility.

    Args:
        value: Feature value to normalize
        feature_name: Optional feature name for registry-based normalization

    Returns:
        Normalized value (canonical form from registry or basic normalization)
    """
    if not value:
        return value
    # Use centralized registry if available and feature_name is provided
    if normalize_feature_value is not None and feature_name:
        try:
            return normalize_feature_value(feature_name, value)
        except (ValueError, KeyError, AttributeError, TypeError):
            # Callback may raise these exceptions for invalid input
            # Fall through to basic normalization on error
            pass
        except BaseException:
            # Callback raised an unexpected exception
            # Fall through to basic normalization on error
            pass
    # Fallback: basic normalization for backward compatibility
    normalized = value.lower().strip()
    # Replace underscores with hyphens consistently
    normalized = normalized.replace("_", "-")
    # Handle specific legacy variations
    variations = {
        "product-alone": "product-alone",
        "product-with-people": "product-with-people",
        "product-with-objects": "product-with-objects",
        "product-in-environment": "product-in-environment",
        "lifestyle context": "lifestyle-context",
        "lifestyle-context": "lifestyle-context",
    }

    return variations.get(normalized, normalized)


def load_recommendations_json(
    data: Dict[str, Any],
    min_importance: float = 0.0,
    min_confidence: Optional[str] = None,
    include_interactions: bool = False,
) -> Dict[str, Any]:
    """
    Load recommendations from recommendations.json format.

    Args:
        data: Dict with recommendations data (from JSON file)
        min_importance: Minimum importance_score to include
        min_confidence: Minimum confidence level ("high", "medium", or None)
        include_interactions: Whether to include feature interactions

    Returns:
        Dict in format expected by convert_features_to_prompts():
        {
            "recommended_features": List[str],
            "feature_importance": Dict[str, float],
            "feature_values": Dict[str, str],
            "negative_feature_values": Dict[str, str],
            "source": str
        }
    """
    # Extract recommendations array
    recommendations = data.get("recommendations", [])
    if not isinstance(recommendations, list):
        raise ValueError(
            f"recommendations must be a list, got {type(recommendations)}"
        )
    # Separate single-feature recommendations from interaction blocks
    feature_recommendations = []
    interactions = []
    interaction_optimized = []

    for rec in recommendations:
        rec_type = rec.get("type")
        if rec_type == "feature_interaction":
            if include_interactions:
                interactions.append(rec)
        elif rec_type == "feature_interaction_optimized":
            # New scorer schema: "interaction optimized" blocks live alongside
            # single-feature recommendations.
            if include_interactions:
                interaction_optimized.append(rec)
        else:
            # Feature recommendation (may or may not have type field)
            feature_recommendations.append(rec)

    logger.info(
        "Found %s feature recommendations and %s interactions",
        len(feature_recommendations),
        len(interactions),
    )
    # Build feature lists and dicts
    recommended_features = []
    feature_importance = {}
    feature_values = {}
    negative_features = []
    negative_feature_importance = {}
    negative_feature_values = {}

    for rec in feature_recommendations:
        # Extract feature name
        feature_name = rec.get("feature")
        if not feature_name:
            logger.warning("Skipping recommendation without 'feature' field")
            continue
        # Extract recommended value
        recommended_value = rec.get("recommended_value")
        if recommended_value is None:
            # Try to use first positive signal as fallback
            positive_signals = rec.get("positive_signals", [])
            if positive_signals:
                recommended_value = positive_signals[0]
                logger.debug(
                    "Using positive_signals[0] for %s: %s",
                    feature_name,
                    recommended_value,
                )  # noqa: E501
            else:
                logger.warning(
                    "Skipping %s: no recommended_value or positive_signals",
                    feature_name,
                )
                continue
        # Normalize value using centralized registry
        normalized_value = normalize_value(str(recommended_value), feature_name)
        # Extract importance score
        importance_score = rec.get("importance_score")
        if importance_score is None:
            logger.warning(
                "No importance_score for %s, using 0.0",
                feature_name,
            )
            importance_score = 0.0
        # Extract confidence
        confidence = rec.get("confidence", "low")
        # Filter by importance
        if importance_score < min_importance:
            logger.debug(
                "Skipping %s: importance %s < %s",
                feature_name,
                importance_score,
                min_importance,
            )
            continue
        # Filter by confidence
        if min_confidence:

            def _confidence_to_level(val: str) -> int:
                """
                Normalize confidence to a comparable level.

                Supports:
                - scorer variants: very_high / medium_high
                - english: high / medium / low
                """
                if val is None:
                    return 0
                s = str(val).strip().lower()
                # Common scorer variants (english)
                if s in {"very_high", "very-high", "veryhigh"}:
                    return 4
                if s in {"high"}:
                    return 3
                if s in {"medium_high", "medium-high", "mediumhigh"}:
                    return 2
                if s in {"medium"}:
                    return 2
                if s in {"low", "very_low", "very-low", "verylow"}:
                    return 1
                # Best-effort fallback on substrings
                if "high" in s:
                    return 3
                if "medium" in s:
                    return 2
                if "low" in s:
                    return 1
                return 0

            min_level = _confidence_to_level(min_confidence)
            rec_level = _confidence_to_level(confidence)
            if rec_level < min_level:
                logger.debug(
                    "Skipping %s: confidence %s < %s",
                    feature_name,
                    confidence,
                    min_confidence,
                )
                continue
        # Extract negative value (values to avoid)
        # Prefer robust selection from value_comparison (if available),
        # otherwise fall back to the first negative_signals item.
        negative_value = _select_negative_value_from_value_comparison(
            value_comparison=rec.get("value_comparison"),
            recommended_value=normalized_value,
            feature_name=feature_name,
        )
        if not negative_value:
            negative_signals = rec.get("negative_signals", [])
            if negative_signals:
                negative_value = normalize_value(
                    str(negative_signals[0]), feature_name
                )
        # Add to lists/dicts
        recommended_features.append(feature_name)
        feature_importance[feature_name] = float(importance_score)
        feature_values[feature_name] = normalized_value

        if negative_value:
            negative_feature_values[feature_name] = negative_value
            negative_feature_importance[feature_name] = float(importance_score)
            if feature_name not in negative_features:
                negative_features.append(feature_name)
    # Sort by importance (descending)
    recommended_features.sort(
        key=lambda f: feature_importance.get(f, 0.0), reverse=True
    )

    logger.info(
        "Loaded %s features after filtering (min_importance=%s, min_confidence=%s)",
        len(recommended_features),
        min_importance,
        min_confidence,
    )

    result = {
        "recommended_features": recommended_features,
        "negative_features": negative_features,
        "feature_importance": feature_importance,
        "negative_feature_importance": negative_feature_importance,
        "feature_values": feature_values,
        "negative_feature_values": negative_feature_values,
        "source": data.get("source", "recommendations.json"),
    }
    # Pass through new top-level metadata (for advanced prompt logic and audit).
    summary = data.get("summary")
    if isinstance(summary, dict):
        result["summary"] = summary

        model_comp = summary.get("model_comparison", {})
        if isinstance(model_comp, dict):
            result["recommended_model"] = model_comp.get("recommended_model")

        prevalence = summary.get("prevalence_analysis", {})
        if isinstance(prevalence, dict):
            top_necessary = prevalence.get("top_necessary_conditions", [])
            if isinstance(top_necessary, list):
                result["necessary_conditions"] = top_necessary

    feature_selection = data.get("feature_selection")
    if isinstance(feature_selection, dict):
        result["feature_selection"] = feature_selection
    # Add interactions if requested
    if include_interactions and interactions:
        result["interactions"] = interactions
    elif include_interactions:
        result["interactions"] = []

    if include_interactions:
        result["interaction_optimized"] = interaction_optimized
    # Add full feature recommendations for advanced converter
    result["feature_recommendations"] = feature_recommendations

    return result
