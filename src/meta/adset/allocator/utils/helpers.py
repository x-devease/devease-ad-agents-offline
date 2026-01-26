"""
Helper functions for rule-based allocation.

Provides utility functions for post-adjustment modifications
and adaptive target ROAS adjustments.
"""

from typing import Dict, List, Optional, Tuple

# Default constants
DEFAULT_ADJUSTMENT_FACTOR = 0.3
DEFAULT_ADAPTIVE_TARGET_OPTIONS = {
    "enable_penalty": True,
    "use_bonus_cap": True,
}


def apply_adaptive_target_adj(
    value: float,
    adaptive_target_roas: Optional[float],
    static_target_roas: Optional[float],
    adjustment_factor: float = DEFAULT_ADJUSTMENT_FACTOR,
    options: Optional[Dict[str, bool]] = None,
) -> Tuple[float, List[str]]:
    """
    Apply adaptive target ROAS adjustment to budget value.

    Adjusts the budget value based on the difference between
    adaptive_target_roas and static_target_roas.

    Args:
        value: Current budget adjustment value
        adaptive_target_roas: Adaptive target ROAS value
        static_target_roas: Static target ROAS value
        adjustment_factor: Factor to apply (default: 0.3)
        options: Optional dict with:
            - enable_penalty: If True, penalize when adaptive < static
            - use_bonus_cap: If True, cap bonus adjustments

    Returns:
        Tuple of (adjusted_value, list_of_modifications)
    """
    if options is None:
        options = DEFAULT_ADAPTIVE_TARGET_OPTIONS.copy()

    modifications: List[str] = []

    # Handle None or invalid values
    if (
        adaptive_target_roas is None
        or static_target_roas is None
        or static_target_roas <= 0
    ):
        return value, modifications

    # Calculate ratio
    ratio = adaptive_target_roas / static_target_roas if static_target_roas > 0 else 1.0

    # Apply adjustment
    if ratio > 1.0:
        # Adaptive is higher than static - bonus
        bonus = (ratio - 1.0) * adjustment_factor
        if options.get("use_bonus_cap", True):
            bonus = min(bonus, 0.5)  # Cap at 50% bonus
        adjusted_value = value * (1.0 + bonus)
        modifications.append("adaptive_target_bonus")
    elif ratio < 1.0 and options.get("enable_penalty", True):
        # Adaptive is lower than static - penalty
        penalty = (1.0 - ratio) * adjustment_factor
        adjusted_value = value * (1.0 - penalty)
        modifications.append("adaptive_target_penalty")
    else:
        # No adjustment needed
        adjusted_value = value

    return adjusted_value, modifications
