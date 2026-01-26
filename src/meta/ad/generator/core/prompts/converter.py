"""
Feature-to-Prompt Converter

Main converter module that wraps the core conversion logic.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils.optional_imports import optional_import


# Try to import from converter_core, fall back if not available
get_optimal_feature_value = optional_import(
    "feature_to_prompt.converter_core", "get_optimal_feature_value"
)
get_prompt_instruction = optional_import(
    "feature_to_prompt.converter_core", "get_prompt_instruction"
)
PROMPT_INSTRUCTIONS = optional_import(
    "feature_to_prompt.converter_core", "PROMPT_INSTRUCTIONS"
)
load_analysis_data = optional_import(
    "feature_to_prompt.data_loader", "load_analysis_data"
)

from .constants import CATEGORIES


logger = logging.getLogger(__name__)


def convert_features_to_prompts(
    recommended_features: List[str],
    feature_importance: Optional[Dict[str, float]] = None,
    negative_features: Optional[List[str]] = None,
    negative_feature_importance: Optional[Dict[str, float]] = None,
    feature_values: Optional[Dict[str, str]] = None,
    negative_feature_values: Optional[Dict[str, str]] = None,
    analysis_data_path: Optional[Path] = None,
    min_importance: float = 0.0,
    group_by_category: bool = True,
    correlation_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Convert recommended features to prompt instructions.

    Args:
        recommended_features: List of feature names recommended by scorer (positive features)
        feature_importance: Optional dict mapping feature names to importance scores
        negative_features: Optional list of feature names to avoid
        negative_feature_importance: Optional dict mapping negative feature
            names to importance scores
        feature_values: Optional dict mapping feature names to explicit values
            (overrides auto-determination)
        negative_feature_values: Optional dict mapping negative feature names to explicit values
        analysis_data_path: Path to analysis results directory.
                          If None, uses default path relative to package.
        min_importance: Minimum importance score to include (if importance provided)
        group_by_category: Whether to group instructions by category
        correlation_threshold: Minimum correlation to consider directional

    Returns:
        Dict with prompt instructions, grouped by category if requested,
        including validation results
    """
    # Load analysis data
    optimal_values, correlations = load_analysis_data(analysis_data_path)
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
            key=lambda f: feature_importance.get(f, 0.0),
            reverse=True,
        )
    else:
        filtered_features = recommended_features
    # Convert features to instructions
    all_instructions = []
    category_instructions = {cat: [] for cat in CATEGORIES}
    category_instructions["other"] = []
    # Process positive features (recommended features)
    for feature_name in filtered_features:
        # Get optimal value (use explicit value if provided, otherwise auto-determine)
        if feature_values and feature_name in feature_values:
            optimal_value = feature_values[feature_name]
            logger.debug(
                "Using explicit value for %s: %s", feature_name, optimal_value
            )
        else:
            optimal_value = get_optimal_feature_value(
                feature_name,
                optimal_values,
                correlations,
                correlation_threshold,
            )
        # Get prompt instruction
        instruction = get_prompt_instruction(
            feature_name, optimal_value, is_negative=False
        )

        if instruction:
            instruction_entry = {
                "feature": feature_name,
                "value": optimal_value,
                "instruction": instruction,
                "importance": (
                    feature_importance.get(feature_name, 0.0)
                    if feature_importance
                    else None
                ),
                "is_negative": False,
            }
            all_instructions.append(instruction_entry)
            # Categorize
            categorized = False
            for category, features in CATEGORIES.items():
                if feature_name in features:
                    category_instructions[category].append(instruction_entry)
                    categorized = True
                    break

            if not categorized:
                category_instructions["other"].append(instruction_entry)
        else:
            logger.warning(
                "No prompt instruction found for feature: %s (value: %s)",
                feature_name,
                optimal_value,
            )
    # Process negative features (features to avoid)
    filtered_negative_features = []
    if negative_features:
        # Filter negative features by importance if provided
        if negative_feature_importance:
            filtered_negative_features = [
                feat
                for feat in negative_features
                if negative_feature_importance.get(feat, 0.0) >= min_importance
            ]
            # Sort by importance (descending)
            filtered_negative_features = sorted(
                filtered_negative_features,
                key=lambda f: negative_feature_importance.get(f, 0.0),
                reverse=True,
            )
        else:
            filtered_negative_features = negative_features

        for feature_name in filtered_negative_features:
            # Get the value to avoid (use explicit value if provided, otherwise auto-determine)
            if (
                negative_feature_values
                and feature_name in negative_feature_values
            ):
                worst_value = negative_feature_values[feature_name]
                logger.debug(
                    "Using explicit negative value for %s: %s",
                    feature_name,
                    worst_value,
                )
            else:
                # Get the value to avoid (opposite of optimal or worst performing value)
                worst_value = _get_worst_feature_value(
                    feature_name,
                    optimal_values,
                    correlations,
                    correlation_threshold,
                )
            # Get negative prompt instruction
            instruction = get_prompt_instruction(
                feature_name, worst_value, is_negative=True
            )

            if instruction:
                instruction_entry = {
                    "feature": feature_name,
                    "value": worst_value,
                    "instruction": instruction,
                    "importance": (
                        negative_feature_importance.get(feature_name, 0.0)
                        if negative_feature_importance
                        else None
                    ),
                    "is_negative": True,
                }
                all_instructions.append(instruction_entry)
                # Categorize
                categorized = False
                for category, features in CATEGORIES.items():
                    if feature_name in features:
                        category_instructions[category].append(
                            instruction_entry
                        )
                        categorized = True
                        break

                if not categorized:
                    category_instructions["other"].append(instruction_entry)
            else:
                logger.warning(
                    "No prompt instruction found for negative feature: %s",
                    feature_name,
                )
    # Build result
    result = {
        "features_processed": len(filtered_features),
        "negative_features_processed": (
            len(filtered_negative_features) if negative_features else 0
        ),
        "instructions_generated": len(all_instructions),
        "all_instructions": all_instructions,
        "validation": {
            "is_valid": True,
            "warnings": [],
            "missing_context_features": [],
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
        result["category_instructions"] = {
            cat: insts for cat, insts in category_instructions.items() if insts
        }
    # Create combined prompt
    combined_prompt = "\n".join(
        [inst["instruction"] for inst in all_instructions]
    )
    result["combined_prompt"] = combined_prompt

    return result


def _get_worst_feature_value(
    feature_name: str,
    optimal_values: Dict[str, str],
    correlations: Dict[str, float],
    correlation_threshold: float = 0.2,
) -> str:
    """
    Get the worst performing value for a feature (opposite of optimal).

    Args:
        feature_name: Name of the feature
        optimal_values: Dict of optimal values
        correlations: Dict of correlation coefficients
        correlation_threshold: Minimum correlation to consider

    Returns:
        Worst feature value as string (must be a valid value from PROMPT_INSTRUCTIONS)
    """
    if PROMPT_INSTRUCTIONS is None:
        return "default"
    # Get available values for this feature
    if feature_name not in PROMPT_INSTRUCTIONS:
        return "default"

    available_values = list(PROMPT_INSTRUCTIONS[feature_name].keys())
    if not available_values:
        return "default"
    # If we have optimal value, find opposite
    if feature_name in optimal_values:
        optimal_value = optimal_values[feature_name]
        # Find opposite value based on common patterns
        opposite_map = {
            "bright": "dark",
            "dark": "bright",
            "medium": "dark",  # or bright, but dark is more problematic
            "simple": "complex",
            "complex": "simple",
            "moderate": "complex",
            "minimal": "complex",
            "flat": "deep",
            "deep": "flat",
            "none": "multiple",  # for person_count
            "single": "multiple",
            "multiple": "none",
            "static": "active",
            "active": "static",
            "professional": "editorial",
            "lifestyle": "editorial",
            "45-degree": "front",  # less optimal angle
            "front": "top-down",
        }
        # Try direct opposite
        if optimal_value in opposite_map:
            opposite = opposite_map[optimal_value]
            if opposite in available_values:
                return opposite
        # Try to find a different value (not optimal, not default)
        for value in available_values:
            if value not in (optimal_value, "default"):
                return value
    # Use correlation to determine worst value
    correlation = correlations.get(feature_name, 0.0)
    if abs(correlation) >= correlation_threshold:
        # For negative correlation: lower values are better, so higher values are worse
        # For positive correlation: higher values are better, so lower values are worse
        # This is heuristic - we'll pick a value that's likely opposite
        if correlation < 0:
            # Negative correlation: prefer lower values, so worst is likely a "high" value
            worst_candidates = ["complex", "deep", "multiple", "active", "dark"]
        else:
            # Positive correlation: prefer higher values, so worst is likely a "low" value
            worst_candidates = ["simple", "flat", "none", "static", "minimal"]

        for candidate in worst_candidates:
            if candidate in available_values:
                return candidate
    # Default: return first non-default value (or default if that's all we have)
    for value in available_values:
        if value != "default":
            return value

    return "default"
