"""
Safety Rules - Hard constraints for budget allocation

These rules must be followed to prevent learning shock
and ensure system stability.
"""

from typing import Optional


class SafetyRules:
    """Hard constraints that must be followed"""

    def __init__(self, config=None):
        """
        Initialize safety rules.

        Args:
            config: Parser instance. If None, uses default values.
        """
        if config:
            # Load from config
            self._settings = {
                "max_daily_increase_pct": config.get_safety_rule(
                    "max_daily_increase_pct", 0.15
                ),
                "max_daily_decrease_pct": config.get_safety_rule(
                    "max_daily_decrease_pct", 0.15
                ),
                "freeze_roas_threshold": config.get_safety_rule(
                    "freeze_roas_threshold", 0.5
                ),
                "freeze_health_threshold": config.get_safety_rule(
                    "freeze_health_threshold", 0.2
                ),
                "min_budget": config.get_safety_rule("min_budget", 1.0),
                "max_budget_pct_of_total": config.get_safety_rule(
                    "max_budget_pct_of_total", 0.40
                ),
                "cold_start_days": config.get_safety_rule("cold_start_days", 3),
                "cold_start_max_increase_pct": config.get_safety_rule(
                    "cold_start_max_increase_pct", 0.10
                ),
            }
        else:
            # Default values
            self._settings = {
                "max_daily_increase_pct": 0.15,  # 15% max increase per day
                "max_daily_decrease_pct": 0.15,  # 15% max decrease per day
                "freeze_roas_threshold": 0.5,  # Freeze if ROAS < 0.5
                "freeze_health_threshold": 0.2,  # Freeze if health < 0.2
                "min_budget": 1.0,  # Minimum daily budget per adset
                "max_budget_pct_of_total": 0.40,  # Max 40% of total budget
                "cold_start_days": 3,  # First 3 days
                "cold_start_max_increase_pct": 0.10,  # Only 10% increase
            }

    @property
    def max_daily_increase_pct(self):
        """Max daily increase percentage."""
        return self._settings["max_daily_increase_pct"]

    @property
    def max_daily_decrease_pct(self):
        """Max daily decrease percentage."""
        return self._settings["max_daily_decrease_pct"]

    @property
    def freeze_roas_threshold(self):
        """ROAS threshold for freezing."""
        return self._settings["freeze_roas_threshold"]

    @property
    def freeze_health_threshold(self):
        """Health threshold for freezing."""
        return self._settings["freeze_health_threshold"]

    @property
    def min_budget(self):
        """Minimum budget."""
        return self._settings["min_budget"]

    @property
    def max_budget_pct_of_total(self):
        """Max budget percentage of total."""
        return self._settings["max_budget_pct_of_total"]

    @property
    def cold_start_days(self):
        """Cold start days."""
        return self._settings["cold_start_days"]

    @property
    def cold_start_max_increase_pct(self):
        """Cold start max increase percentage."""
        return self._settings["cold_start_max_increase_pct"]

    def apply_learning_shock_protection(
        self, new_budget: float, old_budget: Optional[float]
    ) -> float:
        """
        Apply learning shock protection (max change limit).

        Prevents large budget changes that could disrupt
        Meta's learning algorithm.

        Args:
            new_budget: Proposed new budget
            old_budget: Previous budget (None if new adset)

        Returns:
            Adjusted budget respecting max change limits
        """
        if old_budget is None or old_budget == 0:
            return new_budget

        max_increase = old_budget * (1 + self.max_daily_increase_pct)
        max_decrease = old_budget * (1 - self.max_daily_decrease_pct)

        return max(max_decrease, min(new_budget, max_increase))

    def should_freeze(self, roas: float, health_score: float) -> bool:
        """
        Determine if adset should be frozen (budget set to 0).

        Args:
            roas: Current ROAS
            health_score: Current health score (0-1)

        Returns:
            True if adset should be frozen
        """
        return (roas < self.freeze_roas_threshold) or (
            health_score < self.freeze_health_threshold
        )

    def apply_budget_caps(self, budget: float, total_budget: float) -> float:
        """
        Apply minimum and maximum budget constraints.

        Args:
            budget: Proposed budget
            total_budget: Total available budget

        Returns:
            Adjusted budget respecting min/max constraints
        """
        budget = max(budget, self.min_budget)
        max_allowed = total_budget * self.max_budget_pct_of_total
        return min(budget, max_allowed)

    def apply_cold_start_protection(
        self, budget: float, old_budget: Optional[float], days_active: int
    ) -> float:
        """
        Protect new adsets from large budget changes.

        Applies stricter limits during cold start period
        to prevent learning shock.

        Args:
            budget: Proposed new budget
            old_budget: Previous budget
            days_active: Number of days since adset started

        Returns:
            Adjusted budget with cold start protection
        """
        if (
            days_active <= self.cold_start_days
            and old_budget is not None
            and old_budget > 0
        ):
            max_increase = old_budget * (1 + self.cold_start_max_increase_pct)
            return min(budget, max_increase)
        return budget
