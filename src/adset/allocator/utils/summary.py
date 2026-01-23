"""
Summary functions for rule execution.

Provides high-level wrapper functions for executing rules
and calculating budget adjustments.
"""

from typing import Dict, List, Tuple

from src.adset.allocator.lib.decision_rules import DecisionRules
from src.adset.allocator.lib.safety_rules import SafetyRules
from src.adset.allocator.utils.helpers import apply_adaptive_target_adj


def calculate_budget_adjustment(
    roas_7d: float,
    roas_trend: float,
    **kwargs,
) -> Tuple[float, str, List[str]]:
    """
    Calculate budget adjustment factor using decision rules.

    Args:
        roas_7d: 7-day ROAS
        roas_trend: ROAS trend
        **kwargs: Additional metrics (adset_roas, campaign_roas, etc.)

    Returns:
        Tuple of (adjustment_factor, reason, decision_path)
    """
    decision_rules = DecisionRules(config=None)
    adjustment_factor, reason = decision_rules.calculate_budget_adjustment(
        roas_7d=roas_7d,
        roas_trend=roas_trend,
        **kwargs,
    )
    decision_path = [reason] if reason else []
    return adjustment_factor, reason, decision_path


def apply_post_modifications(
    adjustment_factor: float,
    roas_7d: float,
    days_active: int = None,
    marginal_roas: float = None,
    budget_utilization: float = None,
    adaptive_target_roas: float = None,
    static_target_roas: float = None,
    cold_start_max_increase_pct: float = 0.10,
    aggressive_increase_pct: float = 0.15,
    **kwargs,
) -> Tuple[float, List[str]]:
    """
    Apply post-modification adjustments to budget factor.

    Args:
        adjustment_factor: Base adjustment factor
        roas_7d: 7-day ROAS
        days_active: Number of days adset has been active
        marginal_roas: Marginal ROAS value
        budget_utilization: Budget utilization rate
        adaptive_target_roas: Adaptive target ROAS
        static_target_roas: Static target ROAS
        cold_start_max_increase_pct: Max increase for cold start
        aggressive_increase_pct: Aggressive increase percentage
        **kwargs: Additional parameters

    Returns:
        Tuple of (adjusted_factor, list_of_modifications)
    """
    modifications: List[str] = []
    adjusted_factor = adjustment_factor

    # Cold start protection
    if days_active is not None and days_active < 3:
        if adjustment_factor > 1.0:
            max_factor = 1.0 + cold_start_max_increase_pct
            if adjustment_factor > max_factor:
                adjusted_factor = max_factor
                modifications.append("cold_start_cap")
            else:
                modifications.append("cold_start_within_cap")
        else:
            modifications.append("cold_start_hold")

    # Marginal ROAS adjustment
    if marginal_roas is not None and roas_7d > 0:
        ratio = marginal_roas / roas_7d
        if ratio < 0.9:
            penalty = (0.9 - ratio) * 0.1
            adjusted_factor = adjusted_factor * (1.0 - penalty)
            modifications.append("marginal_roas_penalty")
        elif ratio > 1.1:
            bonus = (ratio - 1.1) * 0.1
            adjusted_factor = adjusted_factor * (1.0 + bonus)
            modifications.append("marginal_roas_bonus")

    # Budget utilization gate
    if budget_utilization is not None:
        if budget_utilization < 0.7:
            # Low utilization - reduce increase component by 50%
            increase_component = adjusted_factor - 1.0
            adjusted_factor = 1.0 + (increase_component * 0.5)
            modifications.append("low_utilization_gate")
        elif budget_utilization > 0.95:
            # High utilization - boost by 5%
            boost = min(adjusted_factor * 0.05, aggressive_increase_pct * 0.05)
            adjusted_factor = min(adjusted_factor + boost, 1.0 + aggressive_increase_pct)
            modifications.append("high_utilization_boost")

    # Adaptive target adjustment
    if adaptive_target_roas is not None and static_target_roas is not None:
        adjusted_factor, target_mods = apply_adaptive_target_adj(
            adjusted_factor,
            adaptive_target_roas,
            static_target_roas,
            adjustment_factor=0.3,
            options={"enable_penalty": True, "use_bonus_cap": True},
        )
        modifications.extend(target_mods)

    return adjusted_factor, modifications


def execute_all_rules(
    roas_7d: float,
    roas_trend: float,
    current_budget: float = 100.0,
    **kwargs,
) -> Tuple[float, str, List[str], List[str]]:
    """
    Execute all rules and return final adjustment.

    Args:
        roas_7d: 7-day ROAS
        roas_trend: ROAS trend
        current_budget: Current budget
        **kwargs: Additional metrics

    Returns:
        Tuple of (final_adjustment, reason, decision_path, post_modifications)
    """
    # Calculate base adjustment
    adjustment_factor, reason, decision_path = calculate_budget_adjustment(
        roas_7d=roas_7d,
        roas_trend=roas_trend,
        **kwargs,
    )

    # Apply post-modifications
    final_factor, post_modifications = apply_post_modifications(
        adjustment_factor,
        roas_7d=roas_7d,
        **kwargs,
    )

    # Calculate final budget
    final_budget = current_budget * final_factor

    return final_budget, reason, decision_path, post_modifications
