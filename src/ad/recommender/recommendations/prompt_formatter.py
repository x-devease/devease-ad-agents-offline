"""Format recommendation engine output as prompts for creative generation.

This module provides functionality to convert recommendation engine outputs
into formatted prompts (positive and negative) that can be used for
creative generation or as human-readable recommendations.
"""

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _process_recommendations(recommendations):
    """Process recommendations into positive and negative parts.

    Args:
        recommendations: List of recommendation dictionaries

    Returns:
        Tuple of (positive_parts, negative_parts, features_used,
        features_avoided)
    """
    positive_parts = []
    negative_parts = []
    features_used = []
    features_avoided = []

    for rec in recommendations:
        feature = rec["feature"]
        source = rec.get("source", "")
        rec_type = rec.get("type", "")
        recommended = rec.get("recommended", "")
        current = rec.get("current", "")

        # Positive improvements (rule-based, MD-loaded, AI counterfactuals)
        positive_sources = source in ["rule", "ai_counterfactual", "md"]
        if positive_sources and rec_type != "anti_pattern":
            feature_prompt = _feature_to_positive_prompt(feature, recommended)
            if feature_prompt:
                positive_parts.append(feature_prompt)
                features_used.append(feature)

        # Anti-patterns (what to avoid)
        if rec_type == "anti_pattern" or source == "ai_shap":
            feature_prompt = _feature_to_negative_prompt(
                feature, current, recommended
            )
            if feature_prompt:
                negative_parts.append(feature_prompt)
                features_avoided.append(feature)

    return positive_parts, negative_parts, features_used, features_avoided


def format_recs_as_prompts(
    recommendation_output: Dict[str, Any],
    base_positive: str = (
        "Professional product photography, sharp focus, studio lighting, "
        "high resolution, clean composition, commercial quality, 4K"
    ),
    base_negative: str = (
        "low quality, blurry, distorted, watermark, oversaturated, "
        "plastic look, flat lighting, cluttered, amateur"
    ),
) -> Dict[str, Any]:
    """Format recommendation engine output as positive and negative prompts.

    Args:
        recommendation_output: Output from HybridRecommendationEngine
        base_positive: Base positive prompt for quality
        base_negative: Base negative prompt for quality

    Returns:
        Dictionary with formatted prompts and metadata
    """
    recommendations = recommendation_output.get("recommendations", [])
    positive_parts, negative_parts, features_used, features_avoided = (
        _process_recommendations(recommendations)
    )

    # Build prompts
    positive_additions = ", ".join(positive_parts)
    final_prompt = (
        f"{base_positive}, {positive_additions}"
        if positive_additions
        else base_positive
    )
    neg_additions = ", ".join(negative_parts)
    negative_prompt = (
        f"{base_negative}, {neg_additions}" if neg_additions else base_negative
    )

    # Get metadata
    confidence_scores = recommendation_output.get("confidence_scores", {})
    current_roas = recommendation_output.get("current_roas", 0.0)

    return {
        "creative_id": recommendation_output.get("creative_id", "unknown"),
        "current_roas": current_roas,
        "predicted_roas": recommendation_output.get(
            "predicted_roas", current_roas
        ),
        "final_prompt": final_prompt,
        "negative_prompt": negative_prompt,
        "metadata": {
            "confidence": confidence_scores.get("combined_confidence", 0.5),
            "features_used": features_used,
            "features_avoided": features_avoided,
            "num_recommendations": len(recommendations),
        },
        # Image generation parameters
        "width": 1080,
        "height": 1080,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "seed": None,
    }


def _feature_to_positive_prompt(feature: str, recommended: str) -> str:
    """Convert feature recommendation to positive prompt phrase.

    Args:
        feature: Feature name
        recommended: Recommended value

    Returns:
        Prompt phrase for positive prompt
    """
    # Map features to prompt language (rich, model-friendly phrases)
    feature_mappings = {
        "has_logo": {
            "True": "company logo in top-right corner, brand visible",
            "true": "company logo in top-right corner, brand visible",
            "yes": "company logo in top-right corner, brand visible",
        },
        "cta_button": {
            "yes": "prominent call-to-action button, clear CTA",
            "no": "",
        },
        "text_overlay": {
            "True": "clear, readable text overlay",
            "true": "clear, readable text overlay",
        },
        "dominant_color": {
            "blue": "blue color palette, cool tones",
            "green": "green color palette, natural tones",
            "black": "black color palette, minimal contrast",
            "white": "white color palette, clean background",
            "red": "red accent palette, bold contrast",
            "neutral": "neutral color palette, soft tones",
        },
        "layout": {
            "center": "centered composition, product as focal point",
            "left": "left-aligned composition, rule of thirds",
            "right": "right-aligned composition, balanced negative space",
        },
        "direction": {
            "Overhead": "overhead product shot, flat lay",
            "overhead": "overhead product shot, flat lay",
            "side": "side angle, product profile",
            "front": "front-facing product shot",
        },
        "visual_prominence": {
            "dominant": "product dominant, 40-50% of frame",
            "balanced": "product balanced, 25-30% of frame",
            "subtle": "product subtle, 15-20% of frame, lifestyle context",
        },
    }

    # Check exact matches
    if feature in feature_mappings:
        if recommended in feature_mappings[feature]:
            return feature_mappings[feature][recommended]

    # Handle specific CTA button text
    standard_cta_values = ["yes", "no", "True", "true"]
    if feature == "cta_button" and recommended not in standard_cta_values:
        return f"prominent '{recommended}' call-to-action button"

    # Handle color values
    if "color" in feature.lower():
        return f"{recommended} color scheme"

    # Handle position features
    if "position" in feature.lower():
        return f"{recommended} position"

    # Handle size features
    if "size" in feature.lower():
        return f"{recommended} size"

    # Default: return generic description
    return f"{feature.replace('_', ' ')}: {recommended}"


def _feature_to_negative_prompt(
    feature: str, current: str, recommended: str
) -> str:
    """Convert anti-pattern feature to negative prompt phrase.

    Args:
        feature: Feature name
        current: Current value
        recommended: Recommended value (what to do instead)

    Returns:
        Prompt phrase for negative prompt
    """
    # Map anti-patterns to negative prompt language
    anti_pattern_mappings = {
        "dominant_color": {
            "red": "red color scheme, harsh tones",
            "dark": "dark color scheme, underexposed",
        },
        "layout": {
            "cluttered": "cluttered design, busy composition",
            "busy": "busy composition, distracting elements",
            "left": "off-center left, unbalanced",
            "right": "off-center right, unbalanced",
        },
    }

    # Check exact matches first (use mapping over raw "NOT x" when available)
    if feature in anti_pattern_mappings:
        if current in anti_pattern_mappings[feature]:
            return anti_pattern_mappings[feature][current]
        # Normalize for lookup (e.g. "Left" -> "left")
        cur_lower = str(current).strip().lower()
        if cur_lower in anti_pattern_mappings[feature]:
            return anti_pattern_mappings[feature][cur_lower]

    # If recommended starts with "NOT", extract the bad value
    if isinstance(recommended, str) and recommended.startswith("NOT "):
        return recommended[4:].strip().lower()

    # Handle feature-specific anti-patterns
    feature_lower = feature.lower()
    if "color" in feature_lower:
        return f"{current} color scheme"
    if "layout" in feature_lower or "structure" in feature_lower:
        return "confusing layout"
    if "cta" in feature_lower or "logo" in feature_lower:
        return "no call-to-action" if "cta" in feature_lower else "no logo"

    # Default: use current value
    return f"{current}"


def batch_format_as_prompts(
    recommendation_outputs: List[Dict[str, Any]], **kwargs
) -> List[Dict[str, Any]]:
    """Format multiple recommendation outputs as prompts.

    Args:
        recommendation_outputs: List of recommendation outputs
        **kwargs: Additional arguments passed to
            format_recs_as_prompts()

    Returns:
        List of formatted prompt dictionaries
    """
    results = []

    for output in recommendation_outputs:
        try:
            # Validate required fields
            if not isinstance(output, dict):
                logger.warning("Skipping non-dict output")
                continue

            if "creative_id" not in output:
                logger.warning("Skipping output missing creative_id")
                continue

            if "recommendations" not in output:
                logger.warning(
                    "Skipping output %s: missing recommendations",
                    output.get("creative_id"),
                )
                continue

            formatted = format_recs_as_prompts(output, **kwargs)
            results.append(formatted)

        except (KeyError, ValueError, TypeError, AttributeError) as error:
            creative_id = (
                output.get("creative_id", "unknown")
                if isinstance(output, dict)
                else "unknown"
            )
            logger.error("Failed to format creative %s: %s", creative_id, error)

    total_converted = len(results)
    total_inputs = len(recommendation_outputs)
    logger.info(
        "Formatted %d/%d creatives as prompts", total_converted, total_inputs
    )
    return results


def export_prompts_to_json(
    formatted_prompts: List[Dict[str, Any]], output_path: str
) -> None:
    """Export formatted prompts to JSON file.

    Args:
        formatted_prompts: List of formatted prompt dictionaries
        output_path: Path to output JSON file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_prompts, f, indent=2)

    logger.info(
        "Exported %d prompts to %s", len(formatted_prompts), output_path
    )


def export_prompts_batch(
    formatted_prompts: List[Dict[str, Any]], output_path: str
) -> None:
    """Export formatted prompts to JSON file for batch processing.

    Extracts just the prompt/negative_prompt fields for external tools.

    Args:
        formatted_prompts: List of formatted prompt dictionaries
        output_path: Path to output JSON file
    """
    # Extract just the prompts for external tools
    batch_prompts = [
        {
            "creative_id": p.get("creative_id"),
            "prompt": p.get("final_prompt"),
            "negative_prompt": p.get("negative_prompt"),
        }
        for p in formatted_prompts
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch_prompts, f, indent=2)

    logger.info(
        "Exported %d prompts for batch processing to %s",
        len(batch_prompts),
        output_path,
    )
