"""
Rule-Based Budget Allocator

Combines Safety Rules and Decision Rules to allocate budget
based on all 21 features.
"""

from typing import List
from src.meta.adset.allocator.lib.safety_rules import (
    SafetyRules,
)
from src.meta.adset.allocator.lib.decision_rules import DecisionRules
from src.meta.adset.allocator.lib.models import BudgetAllocationMetrics, BudgetAdjustmentParams
from src.meta.adset.allocator.utils.helpers import apply_adaptive_target_adj


class Allocator:  # pylint: disable=too-few-public-methods
    """
    Rule-based budget allocator that combines:
    - Layer 1: Safety Rules (hard constraints)
    - Layer 2: Decision Rules (priority-based)
    - Layer 3: Post-adjustment modifications (Section 14 advanced concepts)
    """

    def __init__(
        self,
        safety_rules: SafetyRules,
        decision_rules: DecisionRules,
        config=None,
    ):
        """
        Initialize rule-based allocator.

        Args:
            safety_rules: SafetyRules instance
            decision_rules: DecisionRules instance
            config: Parser instance (optional, for advanced concepts)
        """
        self.safety_rules = safety_rules
        self.decision_rules = decision_rules
        self.config = config

    def allocate_budget(  # pylint: disable=too-many-statements,too-many-branches,too-many-locals
        self,
        **kwargs,
    ) -> tuple:
        """
        Allocate budget using rule-based system.

        Accepts all parameters as keyword arguments. For type safety,
        you can use BudgetAllocationMetrics.from_dict() to create the
        metrics object, then pass it as **metrics.to_dict().

        Args:
            **kwargs: All metrics as keyword arguments:
                Required: adset_id, current_budget, roas_7d, roas_trend
                Optional: All other fields from BudgetAllocationMetrics

        Returns:
            (new_budget, decision_path) tuple
        """
        metrics = BudgetAllocationMetrics.from_dict(kwargs)
        decision_path: List[str] = []

        # Layer 1: Check freeze conditions (safety)
        result = self._check_safety_freeze(metrics, decision_path)
        if result:
            return result

        # Layer 2: Calculate adjustment based on decision rules
        adjustment_factor, decision_path = self._calculate_decision_adjustment(
            metrics, decision_path
        )

        # Calculate new budget from adjustment factor
        new_budget = metrics.current_budget * adjustment_factor

        # Apply Section 14 advanced concepts (post-adjustment modifications)
        new_budget, decision_path = self._apply_post_mods(
            new_budget, metrics, adjustment_factor, decision_path
        )

        # Apply safety constraints
        new_budget = self._apply_safety_constraints(new_budget, metrics, decision_path)

        return (new_budget, decision_path)

    def _check_safety_freeze(self, metrics, decision_path):
        """Check if budget should be frozen by safety rules"""
        if self.safety_rules.should_freeze(metrics.roas_7d, metrics.health_score):
            decision_path.append("frozen_by_safety_rules")
            return (0.0, decision_path)

        if (
            metrics.current_budget == 0
            and metrics.roas_7d < self.safety_rules.freeze_roas_threshold * 1.2
        ):
            decision_path.append("keep_frozen_still_underperforming")
            return (0.0, decision_path)

        return None

    def _calculate_decision_adjustment(self, metrics, decision_path):
        """Calculate adjustment based on decision rules"""
        decision_params = BudgetAdjustmentParams(
            roas_7d=metrics.roas_7d,
            roas_trend=metrics.roas_trend,
            current_budget=metrics.current_budget,
            adset_roas=metrics.adset_roas,
            campaign_roas=metrics.campaign_roas,
            account_roas=metrics.account_roas,
            roas_vs_adset=metrics.roas_vs_adset,
            roas_vs_campaign=metrics.roas_vs_campaign,
            roas_vs_account=metrics.roas_vs_account,
            efficiency=metrics.efficiency,
            revenue_per_impression=metrics.revenue_per_impression,
            revenue_per_click=metrics.revenue_per_click,
            spend=metrics.spend,
            spend_rolling_7d=metrics.spend_rolling_7d,
            impressions=metrics.impressions,
            clicks=metrics.clicks,
            reach=metrics.reach,
            adset_spend=metrics.adset_spend,
            campaign_spend=metrics.campaign_spend,
            expected_clicks=metrics.expected_clicks,
            health_score=metrics.health_score,
            days_active=metrics.days_active,
            day_of_week=metrics.day_of_week,
            is_weekend=metrics.is_weekend,
            week_of_year=metrics.week_of_year,
        )
        adjustment_factor, reason = self.decision_rules.calculate_budget_adjustment(
            **decision_params.to_dict()
        )
        decision_path.append(f"decision_rule: {reason}")

        # Apply budget-relative scaling to adjustment factor
        if metrics.current_budget is not None:
            adjustment_pct = adjustment_factor - 1.0
            adjusted_pct = self.decision_rules.budget_relative_scaling(
                metrics.current_budget, adjustment_pct
            )
            adjustment_factor = 1.0 + adjusted_pct

        return adjustment_factor, decision_path

    def _apply_safety_constraints(self, new_budget, metrics, decision_path):
        """Apply safety constraints to budget"""
        previous_budget = metrics.previous_budget or metrics.current_budget

        # Learning shock protection
        new_budget = self.safety_rules.apply_learning_shock_protection(
            new_budget, previous_budget
        )

        # Cold start protection
        new_budget = self.safety_rules.apply_cold_start_protection(
            new_budget, previous_budget, metrics.days_active
        )

        # Post-adjustment smoothing
        if (
            self.config
            and self.config.get_advanced_concept("smoothing_enabled", True)
            and metrics.previous_budget is not None
        ):
            smoothing_alpha = self.config.get_advanced_concept("smoothing_alpha", 0.7)
            smoothed_budget = (
                smoothing_alpha * new_budget
                + (1 - smoothing_alpha) * metrics.previous_budget
            )
            decision_path.append(
                f"smoothing_alpha{smoothing_alpha}_"
                f"from_{new_budget:.2f}_to_{smoothed_budget:.2f}"
            )
            new_budget = smoothed_budget

        # Budget caps (applied after smoothing to ensure final budget
        # respects cap)
        if metrics.total_budget_today is not None:
            new_budget = self.safety_rules.apply_budget_caps(
                new_budget, metrics.total_budget_today
            )

        return new_budget

    def _apply_post_mods(self, new_budget, metrics, adjustment_factor, decision_path):
        """Apply Section 14 advanced concepts (post-adjustment modifications)"""
        new_budget, decision_path = self._apply_adaptive_target(
            new_budget,
            metrics.adaptive_target_roas,
            metrics.static_target_roas,
            decision_path,
        )
        new_budget, decision_path = self._apply_marginal_roas_adjustment(
            new_budget, metrics.marginal_roas, metrics.roas_7d, decision_path
        )
        new_budget, adjustment_factor, decision_path = self._apply_budget_utilization(
            new_budget, metrics, adjustment_factor, decision_path
        )
        return new_budget, decision_path

    def _apply_adaptive_target(
        self,
        new_budget,
        adaptive_target_roas,
        static_target_roas,
        decision_path,
    ):
        """Apply adaptive target ROAS adjustment."""
        if self.config and self.config.get_advanced_concept(
            "adaptive_target_enabled", True
        ):
            adj_factor = self.config.get_advanced_concept(
                "adaptive_target_adjustment_factor", 0.3
            )
            new_budget, target_mods = apply_adaptive_target_adj(
                new_budget,
                adaptive_target_roas,
                static_target_roas,
                adjustment_factor=adj_factor,
                options={"enable_penalty": False, "use_bonus_cap": False},
            )
            decision_path.extend(target_mods)
        return new_budget, decision_path

    def _apply_marginal_roas_adjustment(
        self, new_budget, marginal_roas, roas_7d, decision_path
    ):
        """Apply marginal ROAS adjustment."""
        if self.config and self.config.get_advanced_concept(
            "marginal_roas_enabled", True
        ):
            if marginal_roas is not None and roas_7d > 0:
                marginal_roas_base_factor = self.config.get_advanced_concept(
                    "marginal_roas_base_factor", 0.95
                )
                marginal_roas_range_factor = self.config.get_advanced_concept(
                    "marginal_roas_range_factor", 0.05
                )
                marginal_ratio = marginal_roas / roas_7d
                if marginal_ratio < 0.9:
                    new_budget *= (
                        marginal_roas_base_factor
                        + marginal_ratio * marginal_roas_range_factor
                    )
                    decision_path.append(
                        f"marginal_roas_adjustment_ratio{marginal_ratio:.2f}"
                    )
        return new_budget, decision_path

    def _apply_budget_utilization(
        self, new_budget, metrics, adjustment_factor, decision_path
    ):
        """Apply budget utilization adjustments."""
        if self.config and self.config.get_advanced_concept(
            "budget_utilization_enabled", True
        ):
            budget_utilization = metrics.budget_utilization
            if budget_utilization is not None:
                low_threshold = self.config.get_advanced_concept(
                    "low_utilization_threshold", 0.7
                )
                high_threshold = self.config.get_advanced_concept(
                    "high_utilization_threshold", 0.95
                )
                low_adj_factor = self.config.get_advanced_concept(
                    "low_utilization_adjustment_factor", 0.5
                )
                high_boost_factor = self.config.get_advanced_concept(
                    "high_utilization_boost_factor", 1.1
                )

                if budget_utilization < low_threshold and adjustment_factor > 1.0:
                    increase_component = adjustment_factor - 1.0
                    adjustment_factor = 1.0 + increase_component * low_adj_factor
                    new_budget = metrics.current_budget * adjustment_factor
                    decision_path.append(
                        f"low_utilization_gate_util{budget_utilization:.2f}"
                    )
                elif budget_utilization > high_threshold and adjustment_factor > 1.0:
                    adjustment_factor = min(
                        adjustment_factor * high_boost_factor,
                        1 + self.decision_rules.aggressive_increase_pct,
                    )
                    new_budget = metrics.current_budget * adjustment_factor
                    decision_path.append(
                        f"high_utilization_boost_util{budget_utilization:.2f}"
                    )
        return new_budget, adjustment_factor, decision_path
