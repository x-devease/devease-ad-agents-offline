"""
Simplified Feature-to-Prompt Converter

Converts recommended features to prompt instructions using explicit values
from recommendations.json (no dependency on converter_core).
"""

import logging
from typing import Any, Dict, List, Optional

from .constants import CATEGORIES
from .constants import get_feature_category as _get_feature_category


logger = logging.getLogger(__name__)


def _format_feature_instruction(
    feature_name: str, value: str, is_negative: bool = False
) -> str:
    """
    Format a feature instruction as natural language.

    Args:
        feature_name: Name of the feature
        value: Feature value
        is_negative: Whether this is a negative instruction (avoid)

    Returns:
        Formatted instruction string
    """

    def _norm(s: str) -> str:
        return str(s or "").strip().lower().replace("_", "-")

    feat = _norm(feature_name)
    val = _norm(value)
    # High-signal, feature-specific instructions for the most important
    # moprobo-style recommendations. These are intentionally more concrete
    # than the generic "should be X" phrasing, because vague instructions
    # often fail to transfer into image generation.
    positive_templates: Dict[tuple[str, str], str] = {
        (
            "brightness-distribution",
            "gradient",
        ): (
            "CRITICAL: Use smooth gradient lighting with clear falloff "
            "(bright highlights → soft shadows) to create depth and dimension."
        ),
        (
            "visual-impact",
            "weak",
        ): (
            "MUST INCLUDE: Keep visual impact subtle and understated: "
            "muted tones, gentle contrast, calm professional look."
        ),
        (
            "leading-lines",
            "weak",
        ): (
            "MUST INCLUDE: Use minimal, weak leading lines only (no bold "
            "arrows/strong diagonal guides). Eye flow should feel natural."
        ),
        (
            "relationship-depiction",
            "product-alone",
        ): (
            "MUST INCLUDE: Depict the product alone as the clear "
            "focal point (no people; minimal context props only)."
        ),
        (
            "product-placement",
            "center",
        ): (
            "MUST INCLUDE: Place the product centrally (center point within "
            "the central 40% of the frame in both axes)."
        ),
        (
            "eye-tracking-path",
            "linear",
        ): (
            "MUST INCLUDE: Guide attention with a gentle linear eye path "
            "using layout/contrast/lighting cues "
            "(not aggressive leading lines)."
        ),
        (
            "human-elements",
            "lifestyle-context",
        ): (
            "MUST INCLUDE: Show a real home-cleaning context via the "
            "environment (not people): place the product on a clean tiled "
            "floor "
            "or a simple kitchen/laundry-room surface with a subtle wet sheen "
            "and a few small water droplets near the mop head. Keep the "
            "product as the clear focal point. Do NOT show people, faces, "
            "hands, or any "
            "prominent extra cleaning tools/bottles that could be mistaken as "
            "separate products."
        ),
    }

    negative_templates: Dict[tuple[str, str], str] = {
        (
            "brightness-distribution",
            "even",
        ): (
            "AVOID: Flat, uniform even lighting with minimal contrast. "
            "Do not light the scene evenly across the whole frame."
        ),
        (
            "visual-impact",
            "strong",
        ): (
            "AVOID: Strong visual impact (dramatic contrast, bold colors, "
            "high-energy, attention-grabbing style)."
        ),
        (
            "relationship-depiction",
            "product-with-people",
        ): ("AVOID: Product shown with people (especially interacting)."),
        (
            "product-placement",
            "bottom",
        ): ("AVOID: Product placed in the bottom 40% of the frame."),
        (
            "human-elements",
            "face visible",
        ): ("AVOID: Any visible human faces."),
        (
            "human-elements",
            "face-visible",
        ): ("AVOID: Any visible human faces."),
    }

    key = (feat, val)
    if is_negative and key in negative_templates:
        return negative_templates[key]
    if not is_negative and key in positive_templates:
        return positive_templates[key]
    # Fallback: generic instruction (kept for broad feature coverage)
    value_display = str(value).replace("_", " ").replace("-", " ").title()
    feature_display = (
        str(feature_name).replace("_", " ").replace("-", " ").title()
    )
    if is_negative:
        return f"NEVER INCLUDE: {feature_display} should NOT be {value_display}"
    return f"MUST INCLUDE: {feature_display} should be {value_display}"


def convert_features_to_prompts(
    recommended_features: List[str],
    feature_importance: Optional[Dict[str, float]] = None,
    negative_features: Optional[List[str]] = None,
    negative_feature_importance: Optional[Dict[str, float]] = None,
    feature_values: Optional[Dict[str, str]] = None,
    negative_feature_values: Optional[Dict[str, str]] = None,
    min_importance: float = 0.0,
    group_by_category: bool = True,
) -> Dict[str, Any]:
    """
    Convert recommended features to prompt instructions.

    Args:
        recommended_features: List of feature names recommended
        feature_importance: Optional dict mapping feature names to importance
            scores
        negative_features: Optional list of feature names to avoid
        negative_feature_importance: Optional dict mapping negative feature
            names to importance scores
        feature_values: Dict mapping feature names to explicit values
        negative_feature_values: Dict mapping negative feature names to
            explicit values
        min_importance: Minimum importance score to include
        group_by_category: Whether to group instructions by category

    Returns:
        Dict with prompt instructions, grouped by category if requested
    """
    # Filter features by importance if provided
    if feature_importance:
        filtered_features = [
            feat
            for feat in recommended_features
            if feature_importance.get(feat, 0.0) >= min_importance
        ]
        # Sort by importance (descending)
        filtered_features = sorted(
            filtered_features,
            key=lambda feat: feature_importance.get(feat, 0.0),
            reverse=True,
        )
    else:
        filtered_features = recommended_features
    # Convert features to instructions
    all_instructions = []
    category_instructions = {cat: [] for cat in CATEGORIES}
    category_instructions["other"] = []
    skipped_features = []
    # Process positive features (recommended features)
    for feature_name in filtered_features:
        if not feature_values or feature_name not in feature_values:
            # Recovery mechanism: use generic instruction based on feature name
            importance = (
                feature_importance.get(feature_name, 0.0)
                if feature_importance
                else 0.0
            )
            logger.warning(
                "No value provided for feature %s (importance: %.2f), using generic instruction",
                feature_name,
                importance,
            )
            # Create generic instruction
            feature_display = (
                str(feature_name).replace("_", " ").replace("-", " ").title()
            )
            instruction = f"MUST INCLUDE: {feature_display} should be optimized based on creative best practices"
            skipped_features.append(feature_name)

            category = _get_feature_category(feature_name)
            category_instructions[category].append(
                {
                    "feature": feature_name,
                    "instruction": instruction,
                    "value": "generic",
                    "is_negative": False,
                    "recovered": True,
                }
            )
            all_instructions.append(
                {
                    "feature": feature_name,
                    "instruction": instruction,
                    "value": "generic",
                    "is_negative": False,
                    "recovered": True,
                }
            )
            continue

        value = feature_values[feature_name]
        instruction = _format_feature_instruction(
            feature_name, value, is_negative=False
        )

        category = _get_feature_category(feature_name)
        category_instructions[category].append(
            {
                "feature": feature_name,
                "instruction": instruction,
                "value": value,
                "is_negative": False,
                "recovered": False,
            }
        )
        all_instructions.append(
            {
                "feature": feature_name,
                "instruction": instruction,
                "value": value,
                "is_negative": False,
                "recovered": False,
            }
        )
    # Process negative features (features to avoid)
    if negative_features:
        filtered_negative = negative_features
        if negative_feature_importance:
            filtered_negative = [
                feat
                for feat in negative_features
                if negative_feature_importance.get(feat, 0.0) >= min_importance
            ]
            filtered_negative = sorted(
                filtered_negative,
                key=lambda f: negative_feature_importance.get(f, 0.0),
                reverse=True,
            )

        for feature_name in filtered_negative:
            if (
                not negative_feature_values
                or feature_name not in negative_feature_values
            ):
                continue

            value = negative_feature_values[feature_name]
            instruction = _format_feature_instruction(
                feature_name, value, is_negative=True
            )

            category = _get_feature_category(feature_name)
            category_instructions[category].append(
                {
                    "feature": feature_name,
                    "instruction": instruction,
                    "value": value,
                    "is_negative": True,
                }
            )
            all_instructions.append(
                {
                    "feature": feature_name,
                    "instruction": instruction,
                    "value": value,
                    "is_negative": True,
                }
            )
    # Build result
    result: Dict[str, Any] = {
        "all_instructions": all_instructions,
        "category_instructions": {
            cat: insts for cat, insts in category_instructions.items() if insts
        },
    }

    if group_by_category:
        # Combine instructions by category into prompt strings
        category_prompts = {}
        for category, instructions in category_instructions.items():
            if instructions:
                prompt_lines = [inst["instruction"] for inst in instructions]
                category_prompts[category] = "\n".join(prompt_lines)

        result["category_prompts"] = category_prompts
    # Create combined prompt
    combined_prompt = "\n".join(
        [inst["instruction"] for inst in all_instructions]
    )
    result["combined_prompt"] = combined_prompt
    # Lightweight metrics for status management and tests.
    features_with_values = sum(
        1 for f in filtered_features if feature_values and f in feature_values
    )
    result["features_processed"] = features_with_values
    result["features_recovered"] = len(skipped_features)
    result["negative_features_processed"] = sum(
        1
        for f in (filtered_negative if negative_features else [])
        if negative_feature_values and f in negative_feature_values
    )
    result["instructions_generated"] = len(all_instructions)

    # Log recovery summary
    if skipped_features:
        logger.info(
            "Recovered %d features with missing values: %s",
            len(skipped_features),
            skipped_features,
        )

    return result
