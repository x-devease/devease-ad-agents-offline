"""
Helper functions for DecisionRules calculations.

These utility functions are used by DecisionRules for various
calculations including gradient adjustments, trend scaling, etc.
"""

from typing import Optional


class DecisionRulesHelpers:
    """Helper methods for DecisionRules calculations."""

    @staticmethod
    def gradient_adjustment(
        value: float,
        threshold: float,
        base_pct: float,
        gradient_enabled: bool = True,
        **kwargs,
    ) -> float:
        """
        Calculate gradient-based adjustment instead of binary threshold.

        Args:
            value: Current value (e.g., ROAS)
            threshold: Threshold value
            base_pct: Base adjustment percentage
            gradient_enabled: Whether gradient adjustment is enabled
            range_size: Size of range for scaling
                (if None, uses threshold * gradient_range_size_factor)
            increasing: True if higher values should increase adjustment
            gradient_range_size_factor: Factor for calculating range_size
            (default: 0.5)
            gradient_start_factor: Start factor for gradient (default: 0.5)
            gradient_end_factor: End factor for gradient (default: 1.0)

        Returns:
            Scaled adjustment percentage
        """
        if not gradient_enabled:
            return base_pct

        range_size = kwargs.get("range_size")
        increasing = kwargs.get("increasing", True)
        gradient_range_size_factor = kwargs.get("gradient_range_size_factor", 0.5)
        gradient_start_factor = kwargs.get("gradient_start_factor", 0.5)
        gradient_end_factor = kwargs.get("gradient_end_factor", 1.0)

        if range_size is None:
            range_size = threshold * gradient_range_size_factor

        result = 0.0
        if increasing:
            if value >= threshold + range_size:
                result = base_pct  # Full adjustment
            elif value >= threshold:
                # Linear interpolation from threshold to threshold + range_size
                factor = min(1.0, (value - threshold) / range_size)
                # Start at gradient_start_factor, scale to gradient_end_factor
                result = base_pct * (
                    gradient_start_factor
                    + (gradient_end_factor - gradient_start_factor) * factor
                )
        else:
            if value <= threshold - range_size:
                result = base_pct  # Full adjustment
            elif value <= threshold:
                factor = min(1.0, (threshold - value) / range_size)
                result = base_pct * (
                    gradient_start_factor
                    + (gradient_end_factor - gradient_start_factor) * factor
                )
        return result

    @staticmethod
    def trend_scaling(
        trend: float,
        base_pct: float,
        trend_scaling_enabled: bool = True,
        **kwargs,
    ) -> float:
        """
        Scale adjustment based on trend magnitude.

        Args:
            trend: Trend value (e.g., 0.15 for 15% increase)
            base_pct: Base adjustment percentage
            trend_scaling_enabled: Whether trend scaling is enabled
            trend_strong_threshold: Threshold for strong trend (default: 0.20)
            trend_moderate_threshold: Threshold for moderate trend
            (default: 0.10)
            trend_weak_threshold: Threshold for weak trend (default: 0.05)
            trend_strong_factor: Factor for strong trend (default: 1.0)
            trend_moderate_start_factor: Start factor for moderate trend
            (default: 0.7)
            trend_moderate_range_factor: Range factor for moderate trend
            (default: 0.3)
            trend_weak_start_factor: Start factor for weak trend (default: 0.4)
            trend_weak_range_factor: Range factor for weak trend (default: 0.3)
            trend_min_factor: Minimum factor for weak trend (default: 0.3)

        Returns:
            Scaled adjustment percentage
        """
        if not trend_scaling_enabled:
            return base_pct

        trend_abs = abs(trend)

        # Get configurable parameters with defaults - grouped to reduce locals
        config = {
            "strong_threshold": kwargs.get("trend_strong_threshold", 0.20),
            "moderate_threshold": kwargs.get("trend_moderate_threshold", 0.10),
            "weak_threshold": kwargs.get("trend_weak_threshold", 0.05),
            "strong_factor": kwargs.get("trend_strong_factor", 1.0),
            "moderate_start": kwargs.get("trend_moderate_start_factor", 0.7),
            "moderate_range": kwargs.get("trend_moderate_range_factor", 0.3),
            "weak_start": kwargs.get("trend_weak_start_factor", 0.4),
            "weak_range": kwargs.get("trend_weak_range_factor", 0.3),
            "min_factor": kwargs.get("trend_min_factor", 0.3),
        }

        result = base_pct * config["min_factor"]
        if trend_abs >= config["strong_threshold"]:
            result = base_pct * config["strong_factor"]
        elif trend_abs >= config["moderate_threshold"]:
            factor = config["moderate_start"] + config["moderate_range"] * (
                (trend_abs - config["moderate_threshold"])
                / (config["strong_threshold"] - config["moderate_threshold"])
            )
            result = base_pct * factor
        elif trend_abs >= config["weak_threshold"]:
            factor = config["weak_start"] + config["weak_range"] * (
                (trend_abs - config["weak_threshold"])
                / (config["moderate_threshold"] - config["weak_threshold"])
            )
            result = base_pct * factor
        return result

    @staticmethod
    def relative_performance_gradient(
        ratio: float,
        threshold: float,
        base_pct: float,
        is_above: bool = True,
        **kwargs,
    ) -> float:
        """
        Calculate proportional adjustment based on relative performance ratio.

        Args:
            ratio: Relative performance ratio (e.g., roas_vs_adset = 1.3)
            threshold: Threshold value (e.g., 1.2)
            base_pct: Base adjustment percentage
            is_above: True if ratio > threshold means increase
            relative_performance_max_scale: Max scale factor (default: 1.5)
            relative_performance_multiplier: Multiplier for excess/deficit
            (default: 2.0)

        Returns:
            Scaled adjustment percentage
        """
        relative_performance_max_scale = kwargs.get(
            "relative_performance_max_scale", 1.5
        )
        relative_performance_multiplier = kwargs.get(
            "relative_performance_multiplier", 2.0
        )

        if is_above:
            if ratio < threshold:
                return 0.0
            # Scale by how much above threshold
            # e.g., (1.3 - 1.2) / 1.2 = 0.083
            excess = (ratio - threshold) / threshold
            # Cap at relative_performance_max_scale
            scale_factor = min(
                relative_performance_max_scale,
                1.0 + excess * relative_performance_multiplier,
            )
            return base_pct * scale_factor
        if ratio > threshold:
            return 0.0
        # Scale by how much below threshold
        # e.g., (0.8 - 0.7) / 0.8 = 0.125
        deficit = (threshold - ratio) / threshold
        scale_factor = min(
            relative_performance_max_scale,
            1.0 + deficit * relative_performance_multiplier,
        )
        return base_pct * scale_factor

    @staticmethod
    def health_score_multiplier(
        health_score: float,
        health_score_multiplier_enabled: bool = True,
        **kwargs,
    ) -> float:
        """
        Apply health score as multiplier to adjustment.

        Args:
            health_score: Health score (0-1)
            health_score_multiplier_enabled: Whether multiplier is enabled
            health_score_min_multiplier: Minimum multiplier (default: 0.5)
            health_score_max_multiplier: Maximum multiplier (default: 1.0)

        Returns:
            Multiplier (health_score_min_multiplier to
            health_score_max_multiplier based on health)
        """
        if not health_score_multiplier_enabled:
            return 1.0
        health_score_min_multiplier = kwargs.get("health_score_min_multiplier", 0.5)
        health_score_max_multiplier = kwargs.get("health_score_max_multiplier", 1.0)
        # Map health score (0-1) to multiplier (min to max)
        return (
            health_score_min_multiplier
            + (health_score_max_multiplier - health_score_min_multiplier) * health_score
        )

    @staticmethod
    def budget_relative_scaling(
        current_budget: float,
        base_pct: float,
        budget_relative_scaling_enabled: bool = True,
        **kwargs,
    ) -> float:
        """
        Scale adjustment based on budget size.

        Args:
            current_budget: Current daily budget
            base_pct: Base adjustment percentage
            budget_relative_scaling_enabled: Whether scaling is enabled
            large_budget_threshold: Threshold for large budgets
                (default: 500.0)
            medium_budget_threshold: Threshold for medium budgets
                (default: 100.0)
            large_budget_max_increase: Max increase for large budgets
                (default: 0.15)
            medium_budget_max_increase: Max increase for medium budgets
                (default: 0.20)
            small_budget_max_increase: Max increase for small budgets
                (default: 0.25)

        Returns:
            Scaled adjustment percentage with budget-relative cap
        """
        if not budget_relative_scaling_enabled:
            return base_pct

        large_threshold = kwargs.get("large_budget_threshold", 500.0)
        medium_threshold = kwargs.get("medium_budget_threshold", 100.0)
        large_max = kwargs.get("large_budget_max_increase", 0.15)
        medium_max = kwargs.get("medium_budget_max_increase", 0.20)
        small_max = kwargs.get("small_budget_max_increase", 0.25)

        if current_budget >= large_threshold:
            max_pct = large_max
        elif current_budget >= medium_threshold:
            max_pct = medium_max
        else:
            max_pct = small_max

        # Cap the adjustment but maintain sign
        if base_pct > 0:
            return min(base_pct, max_pct)
        return max(base_pct, -max_pct)

    @staticmethod
    def sample_size_confidence(
        clicks: Optional[int],
        base_pct: float,
        **kwargs,
    ) -> float:
        """
        Adjust confidence based on sample size (clicks).

        Args:
            clicks: Number of clicks
            base_pct: Base adjustment percentage
            low_clicks_threshold: Threshold for low clicks (default: 50)
            medium_clicks_threshold: Threshold for medium clicks (default: 200)
            low_clicks_multiplier: Multiplier for low clicks (default: 0.5)
            medium_clicks_multiplier: Multiplier for medium clicks
                (default: 0.75)

        Returns:
            Scaled adjustment percentage
        """
        low_threshold = kwargs.get("low_clicks_threshold", 50)
        medium_threshold = kwargs.get("medium_clicks_threshold", 200)
        low_mult = kwargs.get("low_clicks_multiplier", 0.5)
        medium_mult = kwargs.get("medium_clicks_multiplier", 0.75)

        if clicks is None:
            return base_pct * low_mult

        if clicks < low_threshold:
            return base_pct * low_mult
        if clicks < medium_threshold:
            return base_pct * medium_mult
        return base_pct  # Full confidence

    @staticmethod
    def q4_dynamic_boost(
        week_of_year: Optional[int],
        q4_week_48_boost: float = 0.01,
        q4_week_49_boost: float = 0.02,
        q4_week_50_51_boost: float = 0.03,
        q4_week_52_boost: float = 0.025,
    ) -> float:
        """
        Get dynamic Q4 boost based on week number.

        Args:
            week_of_year: Week of year (48-52 for Q4)
            q4_week_48_boost: Boost for week 48
            q4_week_49_boost: Boost for week 49
            q4_week_50_51_boost: Boost for weeks 50-51
            q4_week_52_boost: Boost for week 52

        Returns:
            Boost factor (0.0 to 0.03)
        """
        if week_of_year is None:
            return 0.0

        if week_of_year == 48:
            return q4_week_48_boost
        if week_of_year == 49:
            return q4_week_49_boost
        if week_of_year in [50, 51]:
            return q4_week_50_51_boost
        if week_of_year == 52:
            return q4_week_52_boost
        return 0.0
