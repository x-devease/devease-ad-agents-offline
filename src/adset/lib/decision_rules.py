"""
Decision Rules - Priority-based budget adjustment rules

Comprehensive decision rules based on ALL 21 important features.
Rules are organized by priority tiers and executed in optimal order.
"""

from types import SimpleNamespace
from typing import Optional, Tuple
from src.adset.lib.models import BudgetAdjustmentParams
from src.adset.lib.decision_rules_helpers import DecisionRulesHelpers


class DecisionRules:
    """
    Comprehensive decision rules based on 21 important features.

    Features: roas_7d, revenue_per_impression, roas_vs_adset, adset_roas,
    roas_vs_campaign, revenue_per_click, campaign_roas, reach, spend_rolling_7d,
    spend, impressions, clicks, days_active, week_of_year, is_weekend,
    day_of_week, roas_vs_account, account_roas, adset_spend, campaign_spend,
    expected_clicks.
    """

    def __init__(self, config=None):
        """
        Initialize decision rules.

        Args:
            config: Parser instance. If None, uses default values.
        """
        decision_rules = config.decision_rules if config else {}
        self._config = self._load_config(decision_rules)

    @staticmethod
    def _load_config(decision_rules):  # pylint: disable=too-many-statements
        """Load configuration from decision_rules dict"""
        cfg = SimpleNamespace()
        cfg.very_low_roas_threshold = decision_rules.get("very_low_roas_threshold", 1.0)
        cfg.low_roas_threshold = decision_rules.get("low_roas_threshold", 1.5)
        cfg.medium_roas_threshold = decision_rules.get("medium_roas_threshold", 2.0)
        cfg.medium_high_roas_threshold = decision_rules.get(
            "medium_high_roas_threshold", 2.5
        )
        cfg.high_roas_threshold = decision_rules.get("high_roas_threshold", 3.0)
        cfg.very_high_roas_threshold = decision_rules.get(
            "very_high_roas_threshold", 3.5
        )
        cfg.excellent_roas_threshold = decision_rules.get(
            "excellent_roas_threshold", 3.5
        )
        cfg.high_vs_adset_threshold = decision_rules.get("high_vs_adset_threshold", 1.2)
        cfg.high_vs_campaign_threshold = decision_rules.get(
            "high_vs_campaign_threshold", 1.15
        )
        cfg.high_vs_account_threshold = decision_rules.get(
            "high_vs_account_threshold", 1.10
        )
        cfg.low_vs_adset_threshold = decision_rules.get("low_vs_adset_threshold", 0.8)
        cfg.strong_rising_trend = decision_rules.get("strong_rising_trend", 0.10)
        cfg.strong_falling_trend = decision_rules.get("strong_falling_trend", -0.10)
        cfg.moderate_rising_trend = decision_rules.get("moderate_rising_trend", 0.05)
        cfg.moderate_falling_trend = decision_rules.get("moderate_falling_trend", -0.05)
        cfg.high_revenue_per_impression = decision_rules.get(
            "high_revenue_per_impression", 0.08
        )
        cfg.low_revenue_per_impression = decision_rules.get(
            "low_revenue_per_impression", 0.02
        )
        cfg.high_revenue_per_click = decision_rules.get("high_revenue_per_click", 2.5)
        cfg.low_revenue_per_click = decision_rules.get("low_revenue_per_click", 1.0)
        cfg.high_efficiency_threshold = decision_rules.get(
            "high_efficiency_threshold", 0.10
        )
        cfg.low_efficiency_threshold = decision_rules.get(
            "low_efficiency_threshold", 0.02
        )
        cfg.healthy_threshold = decision_rules.get("healthy_threshold", 0.7)
        cfg.unhealthy_threshold = decision_rules.get("unhealthy_threshold", 0.3)
        cfg.excellent_health_threshold = decision_rules.get(
            "excellent_health_threshold", 0.85
        )
        cfg.cold_start_days = decision_rules.get("cold_start_days", 3)
        cfg.learning_phase_days_early = decision_rules.get(
            "learning_phase_days_early", 7
        )
        cfg.learning_phase_days_mid = decision_rules.get("learning_phase_days_mid", 14)
        cfg.learning_phase_days_late = decision_rules.get(
            "learning_phase_days_late", 21
        )
        cfg.learning_phase_days = decision_rules.get("learning_phase_days", 14)
        cfg.established_days = decision_rules.get("established_days", 21)
        cfg.high_spend_threshold = decision_rules.get("high_spend_threshold", 100)
        cfg.low_spend_threshold = decision_rules.get("low_spend_threshold", 10)
        cfg.high_impressions_threshold = decision_rules.get(
            "high_impressions_threshold", 5000
        )
        cfg.spend_trend_rising = decision_rules.get("spend_trend_rising", 0.15)
        cfg.spend_trend_falling = decision_rules.get("spend_trend_falling", -0.20)
        cfg.high_clicks_threshold = decision_rules.get("high_clicks_threshold", 50)
        cfg.high_reach_threshold = decision_rules.get("high_reach_threshold", 1000)
        cfg.weekend_boost_factor = decision_rules.get("weekend_boost_factor", 1.02)
        cfg.weekday_factor = decision_rules.get("weekday_factor", 1.0)
        cfg.q4_boost_factor = decision_rules.get("q4_boost_factor", 1.03)
        cfg.aggressive_increase_pct = decision_rules.get(
            "aggressive_increase_pct", 0.15
        )
        cfg.moderate_increase_pct = decision_rules.get("moderate_increase_pct", 0.10)
        cfg.conservative_increase_pct = decision_rules.get(
            "conservative_increase_pct", 0.05
        )
        cfg.moderate_decrease_pct = decision_rules.get("moderate_decrease_pct", 0.10)
        cfg.aggressive_decrease_pct = decision_rules.get(
            "aggressive_decrease_pct", 0.20
        )
        cfg.maintenance_pct = decision_rules.get("maintenance_pct", 0.0)
        cfg.gradient_enabled = decision_rules.get("gradient_enabled", True)
        cfg.gradient_smoothing_factor = decision_rules.get(
            "gradient_smoothing_factor", 0.2
        )
        cfg.gradient_range_size_factor = decision_rules.get(
            "gradient_range_size_factor", 0.5
        )
        cfg.gradient_start_factor = decision_rules.get("gradient_start_factor", 0.5)
        cfg.gradient_end_factor = decision_rules.get("gradient_end_factor", 1.0)
        cfg.trend_scaling_enabled = decision_rules.get("trend_scaling_enabled", True)
        cfg.trend_strong_threshold = decision_rules.get("trend_strong_threshold", 0.20)
        cfg.trend_moderate_threshold = decision_rules.get(
            "trend_moderate_threshold", 0.10
        )
        cfg.trend_weak_threshold = decision_rules.get("trend_weak_threshold", 0.05)
        cfg.trend_strong_factor = decision_rules.get("trend_strong_factor", 1.0)
        cfg.trend_moderate_start_factor = decision_rules.get(
            "trend_moderate_start_factor", 0.7
        )
        cfg.trend_moderate_range_factor = decision_rules.get(
            "trend_moderate_range_factor", 0.3
        )
        cfg.trend_weak_start_factor = decision_rules.get("trend_weak_start_factor", 0.4)
        cfg.trend_weak_range_factor = decision_rules.get("trend_weak_range_factor", 0.3)
        cfg.trend_min_factor = decision_rules.get("trend_min_factor", 0.3)
        cfg.rel_perf_max_scale = decision_rules.get(
            "relative_performance_max_scale", 1.5
        )
        cfg.rel_perf_multiplier = decision_rules.get(
            "relative_performance_multiplier", 2.0
        )
        cfg.health_score_min_mult = decision_rules.get(
            "health_score_min_multiplier", 0.5
        )
        cfg.health_score_max_mult = decision_rules.get(
            "health_score_max_multiplier", 1.0
        )
        cfg.rel_perf_boost_base = decision_rules.get(
            "relative_performance_boost_base", 0.05
        )
        cfg.relative_perf_boost_campaign = decision_rules.get(
            "relative_performance_boost_campaign", 0.03
        )
        cfg.relative_perf_boost_medium = decision_rules.get(
            "relative_performance_boost_medium", 0.3
        )
        cfg.click_efficiency_threshold = decision_rules.get(
            "click_efficiency_threshold", 1.2
        )
        cfg.default_health_score = decision_rules.get("default_health_score", 0.5)
        cfg.default_efficiency = decision_rules.get("default_efficiency", 0.05)
        cfg.health_score_mult_enabled = decision_rules.get(
            "health_score_multiplier_enabled", True
        )
        cfg.budget_rel_scaling_enabled = decision_rules.get(
            "budget_relative_scaling_enabled", True
        )
        cfg.large_budget_threshold = decision_rules.get("large_budget_threshold", 100)
        cfg.medium_budget_threshold = decision_rules.get("medium_budget_threshold", 20)
        cfg.small_budget_max_increase = decision_rules.get(
            "small_budget_max_increase", 0.20
        )
        cfg.medium_budget_max_increase = decision_rules.get(
            "medium_budget_max_increase", 0.15
        )
        cfg.large_budget_max_increase = decision_rules.get(
            "large_budget_max_increase", 0.10
        )
        cfg.low_clicks_threshold = decision_rules.get("low_clicks_threshold", 20)
        cfg.medium_clicks_threshold = decision_rules.get("medium_clicks_threshold", 100)
        cfg.low_clicks_multiplier = decision_rules.get("low_clicks_multiplier", 0.8)
        cfg.medium_clicks_multiplier = decision_rules.get(
            "medium_clicks_multiplier", 0.9
        )
        cfg.q4_week_48_boost = decision_rules.get("q4_week_48_boost", 0.01)
        cfg.q4_week_49_boost = decision_rules.get("q4_week_49_boost", 0.02)
        cfg.q4_week_50_51_boost = decision_rules.get("q4_week_50_51_boost", 0.03)
        cfg.q4_week_52_boost = decision_rules.get("q4_week_52_boost", 0.02)

        # Ad-level statistics rules (newly added)
        cfg.min_diversity_threshold = decision_rules.get("min_diversity_threshold", 5)
        cfg.min_active_ads_threshold = decision_rules.get("min_active_ads_threshold", 3)
        cfg.diversity_bonus_pct = decision_rules.get("diversity_bonus_pct", 0.10)
        cfg.max_spend_concentration = decision_rules.get(
            "max_spend_concentration", 0.80
        )
        cfg.spend_concentration_penalty_pct = decision_rules.get(
            "spend_concentration_penalty_pct", 0.20
        )
        cfg.max_roas_std = decision_rules.get("max_roas_std", 0.5)
        cfg.consistency_bonus_pct = decision_rules.get("consistency_bonus_pct", 0.10)
        cfg.min_format_diversity = decision_rules.get("min_format_diversity", 3)
        cfg.format_diversity_bonus_pct = decision_rules.get(
            "format_diversity_bonus_pct", 0.05
        )
        cfg.outlier_multiplier_threshold = decision_rules.get(
            "outlier_multiplier_threshold", 3.0
        )
        cfg.outlier_penalty_pct = decision_rules.get("outlier_penalty_pct", 0.15)
        return cfg

    def __getattr__(self, name):
        """Delegate attribute access to config object."""
        # Handle legacy attribute name
        if name == "rel_perf_boost_campaign":
            return self._config.relative_perf_boost_campaign
        return getattr(self._config, name)

    def _gradient_adjustment(
        self, value: float, threshold: float, base_pct: float, **kwargs
    ) -> float:
        """Calculate gradient-based adjustment."""
        return DecisionRulesHelpers.gradient_adjustment(
            value,
            threshold,
            base_pct,
            self.gradient_enabled,
            gradient_range_size_factor=self.gradient_range_size_factor,
            gradient_start_factor=self.gradient_start_factor,
            gradient_end_factor=self.gradient_end_factor,
            **kwargs,
        )

    def _trend_scaling(
        self, trend: float, base_pct: float, _is_rising: bool = True
    ) -> float:
        """Scale adjustment based on trend magnitude."""
        return DecisionRulesHelpers.trend_scaling(
            trend,
            base_pct,
            self.trend_scaling_enabled,
            trend_strong_threshold=self.trend_strong_threshold,
            trend_moderate_threshold=self.trend_moderate_threshold,
            trend_weak_threshold=self.trend_weak_threshold,
            trend_strong_factor=self.trend_strong_factor,
            trend_moderate_start_factor=self.trend_moderate_start_factor,
            trend_moderate_range_factor=self.trend_moderate_range_factor,
            trend_weak_start_factor=self.trend_weak_start_factor,
            trend_weak_range_factor=self.trend_weak_range_factor,
            trend_min_factor=self.trend_min_factor,
        )

    def _relative_performance_gradient(
        self,
        ratio: float,
        threshold: float,
        base_pct: float,
        is_above: bool = True,
    ) -> float:
        """Calculate proportional adjustment based on relative performance."""
        return DecisionRulesHelpers.relative_performance_gradient(
            ratio,
            threshold,
            base_pct,
            is_above,
            relative_performance_max_scale=self.rel_perf_max_scale,
            relative_performance_multiplier=self.rel_perf_multiplier,
        )

    def _health_score_multiplier(self, health_score: float) -> float:
        """Apply health score as multiplier to adjustment."""
        return DecisionRulesHelpers.health_score_multiplier(
            health_score,
            self.health_score_mult_enabled,
            health_score_min_multiplier=self.health_score_min_mult,
            health_score_max_multiplier=self.health_score_max_mult,
        )

    def _get_effective_roas(self, params_dict: dict) -> float:
        """Get effective ROAS, prioritizing Shopify ROAS over Meta ROAS.

        Shopify ROAS is based on actual revenue from orders, making it more
        accurate than Meta's platform ROAS. When Shopify ROAS is available,
        use it; otherwise fall back to Meta's purchase_roas_rolling_7d.

        Args:
            params_dict: Dictionary of parameters including roas_7d and shopify_roas.

        Returns:
            Effective ROAS value (Shopify if available, else Meta).
        """
        # Prioritize Shopify ROAS (actual revenue-based)
        shopify_roas = params_dict.get("shopify_roas")
        if shopify_roas is not None and shopify_roas > 0:
            return shopify_roas

        # Fall back to Meta ROAS
        return params_dict.get("roas_7d", 0.0) or 0.0

    def budget_relative_scaling(self, current_budget: float, base_pct: float) -> float:
        """Scale adjustment based on budget size."""
        return DecisionRulesHelpers.budget_relative_scaling(
            current_budget,
            base_pct,
            budget_relative_scaling_enabled=self.budget_rel_scaling_enabled,
            large_budget_threshold=self.large_budget_threshold,
            medium_budget_threshold=self.medium_budget_threshold,
            large_budget_max_increase=self.large_budget_max_increase,
            medium_budget_max_increase=self.medium_budget_max_increase,
            small_budget_max_increase=self.small_budget_max_increase,
        )

    def _sample_size_confidence(self, clicks: Optional[int], base_pct: float) -> float:
        """Adjust confidence based on sample size (clicks)."""
        return DecisionRulesHelpers.sample_size_confidence(
            clicks,
            base_pct,
            low_clicks_threshold=self.low_clicks_threshold,
            medium_clicks_threshold=self.medium_clicks_threshold,
            low_clicks_multiplier=self.low_clicks_multiplier,
            medium_clicks_multiplier=self.medium_clicks_multiplier,
        )

    def _q4_dynamic_boost(self, week_of_year: Optional[int]) -> float:
        """Get dynamic Q4 boost based on week number."""
        return DecisionRulesHelpers.q4_dynamic_boost(
            week_of_year,
            self.q4_week_48_boost,
            self.q4_week_49_boost,
            self.q4_week_50_51_boost,
            self.q4_week_52_boost,
        )

    def calculate_budget_adjustment(
        self,
        **kwargs,
    ) -> Tuple[float, str]:
        """
        Calculate budget adjustment based on ALL 21 important features.

        Args:
            **kwargs: All parameters as keyword arguments:
                Required: roas_7d, roas_trend
                Optional: All other fields from BudgetAdjustmentParams

        Rule Priority (applied in order):
        1. Safety checks (freeze conditions) - handled by SafetyRules
        2. Excellent performers (multi-factor confirmation)
        3. High performers (strong signals)
        4. Relative performance (vs adset/campaign/account)
        5. Efficiency-based rules
        6. Volume-based rules
        7. Time-based adjustments
        8. Default/maintenance

        Returns:
            (adjustment_factor, reason) tuple
        """
        params = BudgetAdjustmentParams.from_dict(kwargs)
        params_dict = self._extract_and_normalize_params(params)
        result = (
            self._check_excellent_performers(params_dict)
            or self._check_high_performers(params_dict)
            or self._check_efficiency_rules(params_dict)
            or self._check_volume_rules(params_dict)
            or self._check_ad_diversity_rules(params_dict)
            or self._check_declining_performers(params_dict)
            or self._check_time_based_adjustments(params_dict)
            or self._check_lifecycle_rules(params_dict)
            or self._check_time_weighted_smoothing(params_dict)
        )
        adjustment_factor, reason = (
            result if result else self._default_adjustment(params_dict)
        )

        # Down-weight adjustments when rolling window metrics are low quality
        # This prevents aggressive decisions based on insufficient historical data
        if params_dict["rolling_low_quality"] == 1:
            # Convert to percentage change, reduce by 50%, convert back
            # This ensures both increases and decreases are made more conservative
            pct_change = adjustment_factor - 1.0
            pct_change_reduced = pct_change * 0.5  # Reduce magnitude by 50%
            adjustment_factor = 1.0 + pct_change_reduced
            reason = f"{reason}_low_quality_rolling"

        return adjustment_factor, reason

    def _extract_and_normalize_params(self, params):
        """Extract and normalize parameters from BudgetAdjustmentParams"""
        params_dict = {
            "roas_7d": params.roas_7d,
            "roas_trend": params.roas_trend,
            "adset_roas": params.adset_roas,
            "campaign_roas": params.campaign_roas,
            "account_roas": params.account_roas,
            "roas_vs_adset": params.roas_vs_adset,
            "roas_vs_campaign": params.roas_vs_campaign,
            "roas_vs_account": params.roas_vs_account,
            "efficiency": params.efficiency,
            "revenue_per_impression": params.revenue_per_impression,
            "revenue_per_click": params.revenue_per_click,
            "spend": params.spend,
            "spend_rolling_7d": params.spend_rolling_7d,
            "impressions": params.impressions,
            "clicks": params.clicks,
            "reach": params.reach,
            "adset_spend": params.adset_spend,
            "campaign_spend": params.campaign_spend,
            "expected_clicks": params.expected_clicks,
            "health_score": params.health_score,
            "days_active": params.days_active,
            "day_of_week": params.day_of_week,
            "is_weekend": params.is_weekend,
            "week_of_year": params.week_of_year,
            # Ad-level statistics (newly added)
            "num_ads": params.num_ads,
            "num_active_ads": params.num_active_ads,
            "ad_diversity": params.ad_diversity,
            "ad_roas_mean": params.ad_roas_mean,
            "ad_roas_std": params.ad_roas_std,
            "ad_roas_range": params.ad_roas_range,
            "ad_spend_gini": params.ad_spend_gini,
            "top_ad_spend_pct": params.top_ad_spend_pct,
            "video_ads_ratio": params.video_ads_ratio,
            "format_diversity_score": params.format_diversity_score,
            "rolling_low_quality": params.rolling_low_quality,
            # Shopify integration: actual revenue-based ROAS
            "shopify_roas": params.shopify_roas,
            "shopify_revenue": params.shopify_revenue,
        }
        if params_dict["health_score"] is None:
            params_dict["health_score"] = self.default_health_score
        if params_dict["days_active"] is None:
            params_dict["days_active"] = 0
        if params_dict["rolling_low_quality"] is None:
            params_dict["rolling_low_quality"] = 0
        if params_dict["efficiency"] is None:
            params_dict["efficiency"] = self.default_efficiency

        # NOTE: Shopify ROAS is available in params_dict["shopify_roas"]
        # but rules continue to use Meta's roas_7d as the primary signal.
        # Shopify data can be used for custom rules or analysis.

        # Set defaults for ad-level statistics if not present
        if "num_ads" not in params_dict or params_dict["num_ads"] is None:
            params_dict["num_ads"] = 1
        if "num_active_ads" not in params_dict or params_dict["num_active_ads"] is None:
            params_dict["num_active_ads"] = 0
        if "ad_diversity" not in params_dict or params_dict["ad_diversity"] is None:
            params_dict["ad_diversity"] = 1
        if "ad_roas_mean" not in params_dict or params_dict["ad_roas_mean"] is None:
            params_dict["ad_roas_mean"] = 0.0
        if "ad_roas_std" not in params_dict or params_dict["ad_roas_std"] is None:
            params_dict["ad_roas_std"] = 0.0
        if "ad_roas_range" not in params_dict or params_dict["ad_roas_range"] is None:
            params_dict["ad_roas_range"] = 0.0
        if "ad_spend_gini" not in params_dict or params_dict["ad_spend_gini"] is None:
            params_dict["ad_spend_gini"] = 0.0
        if (
            "top_ad_spend_pct" not in params_dict
            or params_dict["top_ad_spend_pct"] is None
        ):
            params_dict["top_ad_spend_pct"] = 1.0
        if (
            "video_ads_ratio" not in params_dict
            or params_dict["video_ads_ratio"] is None
        ):
            params_dict["video_ads_ratio"] = 0.0
        if (
            "format_diversity_score" not in params_dict
            or params_dict["format_diversity_score"] is None
        ):
            params_dict["format_diversity_score"] = 1
        return params_dict

    def _check_excellent_performers(  # pylint: disable=too-many-locals
        self, params_dict
    ) -> Optional[Tuple[float, str]]:
        """Check Tier 2: Excellent performers rules"""
        roas_7d = params_dict["roas_7d"]
        roas_trend = params_dict["roas_trend"]
        efficiency = params_dict["efficiency"]
        health_score = params_dict["health_score"]
        roas_vs_adset = params_dict["roas_vs_adset"]
        roas_vs_campaign = params_dict["roas_vs_campaign"]
        roas_vs_account = params_dict["roas_vs_account"]
        account_roas = params_dict["account_roas"]

        if (
            roas_7d >= self.excellent_roas_threshold
            and roas_trend >= self.strong_rising_trend
            and (efficiency is None or efficiency >= self.high_efficiency_threshold)
            and health_score >= self.excellent_health_threshold
        ):
            base_pct = self._gradient_adjustment(
                roas_7d,
                self.excellent_roas_threshold,
                self.aggressive_increase_pct,
                range_size=0.5,
            )
            trend_pct = self._trend_scaling(roas_trend, base_pct, _is_rising=True)
            if (
                roas_vs_adset is not None
                and roas_vs_adset >= self.high_vs_adset_threshold
            ):
                rel_boost = self._relative_performance_gradient(
                    roas_vs_adset,
                    self.high_vs_adset_threshold,
                    self.rel_perf_boost_base,
                    is_above=True,
                )
                trend_pct += rel_boost
            if (
                roas_vs_campaign is not None
                and roas_vs_campaign >= self.high_vs_campaign_threshold
            ):
                rel_boost = self._relative_performance_gradient(
                    roas_vs_campaign,
                    self.high_vs_campaign_threshold,
                    self.relative_perf_boost_campaign,
                    is_above=True,
                )
                trend_pct += rel_boost
            final_pct = trend_pct * self._health_score_multiplier(health_score)
            return (
                1 + final_pct,
                f"excellent_performer_roas{roas_7d:.2f}_"
                f"trend{roas_trend:.2%}_health{health_score:.2f}",
            )

        # Excellent ROAS with strong relative performance
        # (even without strong rising trend)
        if (
            roas_7d >= self.excellent_roas_threshold
            and roas_vs_adset is not None
            and roas_vs_adset >= self.high_vs_adset_threshold
            and roas_vs_campaign is not None
            and roas_vs_campaign >= self.high_vs_campaign_threshold
        ):
            base_pct = self._gradient_adjustment(
                roas_7d,
                self.excellent_roas_threshold,
                self.aggressive_increase_pct,
                range_size=0.5,
            )
            if roas_trend > 0:
                trend_pct = self._trend_scaling(roas_trend, base_pct, _is_rising=True)
            else:
                trend_pct = base_pct
            vs_adset_boost = self._relative_performance_gradient(
                roas_vs_adset,
                self.high_vs_adset_threshold,
                self.rel_perf_boost_base,
                is_above=True,
            )
            vs_campaign_boost = self._relative_performance_gradient(
                roas_vs_campaign,
                self.high_vs_campaign_threshold,
                self.relative_perf_boost_campaign,
                is_above=True,
            )
            final_pct = trend_pct + vs_adset_boost + vs_campaign_boost
            final_pct = final_pct * self._health_score_multiplier(health_score)
            return (
                1 + final_pct,
                f"excellent_roas_strong_relative_perf_roas{roas_7d:.2f}_"
                f"vs_adset{roas_vs_adset:.2f}_"
                f"vs_campaign{roas_vs_campaign:.2f}",
            )

        revenue_per_impression = params_dict["revenue_per_impression"]
        if (
            roas_7d >= self.high_roas_threshold
            and efficiency is not None
            and efficiency >= self.high_efficiency_threshold
            and revenue_per_impression is not None
            and revenue_per_impression >= self.high_revenue_per_impression
        ):
            base_pct = self._gradient_adjustment(
                roas_7d,
                self.high_roas_threshold,
                self.aggressive_increase_pct,
                range_size=0.5,
            )
            final_pct = base_pct * self._health_score_multiplier(health_score)
            return (
                1 + final_pct,
                f"high_roas_high_efficiency_high_rpi_roas{roas_7d:.2f}_"
                f"eff{efficiency:.3f}_rpi{revenue_per_impression:.4f}",
            )

        if (
            roas_7d >= self.high_roas_threshold
            and roas_vs_adset is not None
            and roas_vs_adset >= self.high_vs_adset_threshold
            and roas_vs_campaign is not None
            and roas_vs_campaign >= self.high_vs_campaign_threshold
        ):
            base_pct = self._gradient_adjustment(
                roas_7d,
                self.high_roas_threshold,
                self.aggressive_increase_pct,
                range_size=0.5,
            )
            vs_adset_boost = self._relative_performance_gradient(
                roas_vs_adset,
                self.high_vs_adset_threshold,
                self.rel_perf_boost_base,
                is_above=True,
            )
            vs_campaign_boost = self._relative_performance_gradient(
                roas_vs_campaign,
                self.high_vs_campaign_threshold,
                self.relative_perf_boost_campaign,
                is_above=True,
            )
            final_pct = base_pct + vs_adset_boost + vs_campaign_boost
            final_pct = final_pct * self._health_score_multiplier(health_score)
            return (
                1 + final_pct,
                f"high_roas_strong_relative_perf_roas{roas_7d:.2f}_"
                f"vs_adset{roas_vs_adset:.2f}_"
                f"vs_campaign{roas_vs_campaign:.2f}",
            )

        if (
            roas_7d >= self.high_roas_threshold
            and roas_vs_account is not None
            and roas_vs_account >= self.high_vs_account_threshold
            and account_roas is not None
            and account_roas > 0
        ):
            return (
                1 + self.aggressive_increase_pct,
                "high_roas_strong_vs_account",
            )

        return None

    def _check_high_performers(self, params_dict) -> Optional[Tuple[float, str]]:
        """Check Tier 3: High performers rules"""
        roas_7d = params_dict["roas_7d"]
        roas_trend = params_dict["roas_trend"]
        health_score = params_dict["health_score"]
        efficiency = params_dict["efficiency"]
        revenue_per_click = params_dict["revenue_per_click"]
        clicks = params_dict["clicks"]
        roas_vs_adset = params_dict["roas_vs_adset"]

        if (
            roas_7d >= self.high_roas_threshold
            and roas_trend >= self.strong_rising_trend
            and health_score >= self.healthy_threshold
        ):
            base_pct = self._gradient_adjustment(
                roas_7d,
                self.high_roas_threshold,
                self.aggressive_increase_pct,
                range_size=0.5,
            )
            trend_pct = self._trend_scaling(roas_trend, base_pct, _is_rising=True)
            final_pct = trend_pct * self._health_score_multiplier(health_score)
            return (
                1 + final_pct,
                f"high_roas_rising_healthy_roas{roas_7d:.2f}_"
                f"trend{roas_trend:.2%}_health{health_score:.2f}",
            )

        if (
            roas_7d >= self.high_roas_threshold
            and roas_trend >= 0
            and efficiency is not None
            and efficiency >= self.high_efficiency_threshold
        ):
            base_pct = self._gradient_adjustment(
                roas_7d,
                self.high_roas_threshold,
                self.moderate_increase_pct,
                range_size=0.5,
            )
            if roas_trend > 0:
                base_pct = self._trend_scaling(roas_trend, base_pct, _is_rising=True)
            final_pct = base_pct * self._health_score_multiplier(health_score)
            return (
                1 + final_pct,
                f"high_roas_efficient_roas{roas_7d:.2f}_"
                f"trend{roas_trend:.2%}_eff{efficiency:.3f}",
            )

        if (
            self.low_roas_threshold <= roas_7d < self.high_roas_threshold
            and efficiency is not None
            and efficiency >= self.high_efficiency_threshold
            and revenue_per_click is not None
            and revenue_per_click >= self.high_revenue_per_click
        ):
            if roas_trend >= 0:
                base_pct = self.moderate_increase_pct
                base_pct = self._sample_size_confidence(clicks, base_pct)
                final_pct = base_pct * self._health_score_multiplier(health_score)
                return (
                    1 + final_pct,
                    f"medium_roas_high_efficiency_high_rpc_"
                    f"roas{roas_7d:.2f}_eff{efficiency:.3f}_"
                    f"rpc{revenue_per_click:.2f}",
                )

        if (
            self.low_roas_threshold <= roas_7d < self.high_roas_threshold
            and roas_vs_adset is not None
            and roas_vs_adset >= self.high_vs_adset_threshold
            and roas_trend >= self.moderate_rising_trend
        ):
            base_pct = self.moderate_increase_pct
            vs_adset_boost = self._relative_performance_gradient(
                roas_vs_adset,
                self.high_vs_adset_threshold,
                base_pct * self.relative_perf_boost_medium,
                is_above=True,
            )
            trend_pct = self._trend_scaling(roas_trend, base_pct, _is_rising=True)
            final_pct = trend_pct + vs_adset_boost
            final_pct = final_pct * self._health_score_multiplier(health_score)
            return (
                1 + final_pct,
                f"medium_roas_strong_vs_adset_roas{roas_7d:.2f}_"
                f"vs_adset{roas_vs_adset:.2f}_trend{roas_trend:.2%}",
            )

        return None

    def _check_efficiency_rules(self, params_dict) -> Optional[Tuple[float, str]]:
        """Check Tier 4: Efficiency-based rules"""
        revenue_per_impression = params_dict["revenue_per_impression"]
        roas_7d = params_dict["roas_7d"]
        impressions = params_dict["impressions"]
        revenue_per_click = params_dict["revenue_per_click"]
        clicks = params_dict["clicks"]
        efficiency = params_dict["efficiency"]
        roas_trend = params_dict["roas_trend"]

        if (
            revenue_per_impression is not None
            and revenue_per_impression >= self.high_revenue_per_impression
            and roas_7d >= self.low_roas_threshold
            and impressions is not None
            and impressions >= self.high_impressions_threshold
        ):
            return (
                1 + self.moderate_increase_pct,
                f"high_revenue_per_impression_high_volume_"
                f"rpi{revenue_per_impression:.4f}_imps{impressions:.0f}",
            )

        if (
            revenue_per_click is not None
            and revenue_per_click >= self.high_revenue_per_click
            and clicks is not None
            and clicks >= self.high_clicks_threshold
            and roas_7d >= self.low_roas_threshold
        ):
            return (
                1 + self.conservative_increase_pct,
                f"high_revenue_per_click_high_volume_"
                f"rpc{revenue_per_click:.2f}_clicks{clicks:.0f}",
            )

        if (
            efficiency is not None
            and efficiency < self.low_efficiency_threshold
            and roas_7d < self.high_roas_threshold
            and roas_trend <= 0
        ):
            return (
                1 - self.moderate_decrease_pct,
                f"low_efficiency_decrease_eff{efficiency:.3f}_" f"roas{roas_7d:.2f}",
            )

        return None

    def _check_volume_rules(self, params_dict) -> Optional[Tuple[float, str]]:
        """Check Tier 5: Volume-based rules"""
        spend = params_dict["spend"]
        roas_7d = params_dict["roas_7d"]
        efficiency = params_dict["efficiency"]
        days_active = params_dict["days_active"]
        roas_trend = params_dict["roas_trend"]
        spend_rolling_7d = params_dict["spend_rolling_7d"]
        reach = params_dict["reach"]
        expected_clicks = params_dict["expected_clicks"]
        clicks = params_dict["clicks"]
        adset_spend = params_dict["adset_spend"]
        campaign_spend = params_dict["campaign_spend"]

        return (
            self._check_low_spend_high_roas(spend, roas_7d, efficiency, days_active)
            or self._check_high_spend_declining(spend, roas_trend, roas_7d)
            or self._check_spend_trend(spend, spend_rolling_7d, roas_7d)
            or self._check_high_reach_high_roas(reach, roas_7d)
            or self._check_expected_clicks(expected_clicks, clicks, roas_7d)
            or self._check_adset_spend_share(adset_spend, campaign_spend, roas_7d)
        )

    def _check_low_spend_high_roas(
        self, spend, roas_7d, efficiency, days_active
    ) -> Optional[Tuple[float, str]]:
        """Check low spend but high ROAS rule"""
        has_low_spend = spend is not None and spend < self.low_spend_threshold
        has_high_roas = roas_7d >= self.high_roas_threshold
        has_high_efficiency = (
            efficiency is not None and efficiency >= self.high_efficiency_threshold
        )
        is_established = days_active >= self.established_days
        if has_low_spend and has_high_roas and has_high_efficiency and is_established:
            return (
                1 + self.moderate_increase_pct,
                f"low_spend_high_roas_scale_up_spend{spend:.2f}_"
                f"roas{roas_7d:.2f}_eff{efficiency:.3f}",
            )
        return None

    def _check_high_spend_declining(
        self, spend, roas_trend, roas_7d
    ) -> Optional[Tuple[float, str]]:
        """Check high spend but declining ROAS rule"""
        if (
            spend is not None
            and spend >= self.high_spend_threshold
            and roas_trend <= self.moderate_falling_trend
            and roas_7d < self.high_roas_threshold
        ):
            return (
                1 - self.moderate_decrease_pct,
                f"high_spend_declining_roas_spend{spend:.2f}_" f"trend{roas_trend:.2%}",
            )
        return None

    def _check_high_reach_high_roas(
        self, reach, roas_7d
    ) -> Optional[Tuple[float, str]]:
        """Check high reach and high ROAS rule"""
        if (
            reach is not None
            and reach >= self.high_reach_threshold
            and roas_7d >= self.high_roas_threshold
        ):
            return (
                1 + self.conservative_increase_pct,
                f"high_reach_high_roas_reach{reach:.0f}_roas{roas_7d:.2f}",
            )
        return None

    def _check_adset_spend_share(
        self, adset_spend, campaign_spend, roas_7d
    ) -> Optional[Tuple[float, str]]:
        """Check adset spend share rule"""
        if (
            adset_spend is not None
            and campaign_spend is not None
            and campaign_spend > 0
            and roas_7d >= self.high_roas_threshold
        ):
            adset_share = adset_spend / campaign_spend
            if adset_share < 0.1:
                return (
                    1 + self.moderate_increase_pct,
                    "small_adset_high_roas_scale",
                )
        return None

    def _check_spend_trend(
        self, spend, spend_rolling_7d, roas_7d
    ) -> Optional[Tuple[float, str]]:
        """Check spend trend rule"""
        if spend_rolling_7d is not None and spend is not None and spend_rolling_7d > 0:
            spend_trend = (spend - spend_rolling_7d) / spend_rolling_7d
            if (
                spend_trend >= self.spend_trend_rising
                and roas_7d >= self.high_roas_threshold
            ):
                return (
                    1 + self.conservative_increase_pct,
                    f"spend_rising_high_roas_spend_trend{spend_trend:.2%}_"
                    f"roas{roas_7d:.2f}",
                )
        return None

    def _check_expected_clicks(
        self, expected_clicks, clicks, roas_7d
    ) -> Optional[Tuple[float, str]]:
        """Check expected clicks rule"""
        if expected_clicks is not None and clicks is not None and expected_clicks > 0:
            click_efficiency = clicks / expected_clicks
            if (
                click_efficiency > self.click_efficiency_threshold
                and roas_7d >= self.low_roas_threshold
            ):
                return (
                    1 + self.conservative_increase_pct,
                    f"exceeding_expected_clicks_click_eff"
                    f"{click_efficiency:.2f}_roas{roas_7d:.2f}",
                )
        return None

    def _check_ad_diversity_rules(self, params_dict) -> Optional[Tuple[float, str]]:
        """Check Tier 5.5: Ad-level diversity and performance rules"""
        roas_7d = params_dict["roas_7d"]
        ad_diversity = params_dict["ad_diversity"]
        num_active_ads = params_dict["num_active_ads"]
        top_ad_spend_pct = params_dict["top_ad_spend_pct"]
        ad_roas_mean = params_dict["ad_roas_mean"]
        ad_roas_std = params_dict["ad_roas_std"]
        ad_roas_range = params_dict["ad_roas_range"]
        format_diversity_score = params_dict["format_diversity_score"]
        ad_spend_gini = params_dict["ad_spend_gini"]

        # Skip ad-level diversity rules if statistics weren't provided
        # (indicated by default values: num_active_ads=0, top_ad_spend_pct=1.0)
        if num_active_ads == 0 and top_ad_spend_pct == 1.0:
            return None

        # Rule 1: Spend Concentration Penalty (highest priority)
        # Penalize adsets with too much spend concentrated in top ad
        if (
            top_ad_spend_pct >= self.max_spend_concentration
            and roas_7d < self.high_roas_threshold
        ):
            # Scale penalty based on concentration level
            concentration_over_limit = top_ad_spend_pct - self.max_spend_concentration
            penalty = self.spend_concentration_penalty_pct + (
                concentration_over_limit * 0.5
            )
            penalty = min(penalty, 0.40)  # Cap at 40%
            return (
                1 - penalty,
                f"spend_concentration_penalty_top_ad_{top_ad_spend_pct:.1%}",
            )

        # Rule 2: Outlier Detection
        # Penalize if one ad significantly outperforms others (indicates luck)
        if (
            ad_roas_mean > 0
            and ad_roas_range > 0
            and roas_7d < self.high_roas_threshold
        ):
            # Check if max is 3x the mean (outlier detection)
            # Using range as proxy: if range >> mean, likely has outlier
            if (
                ad_roas_mean > 0
                and (ad_roas_range / ad_roas_mean) >= self.outlier_multiplier_threshold
            ):
                return (
                    1 - self.outlier_penalty_pct,
                    f"outlier_ad_penalty_range_{ad_roas_range:.2f}_mean_{ad_roas_mean:.2f}",
                )

        # Rule 3: Ad Diversity Bonus
        # Reward adsets with diverse active ads (only if ROAS is good)
        if (
            roas_7d >= self.low_roas_threshold
            and ad_diversity >= self.min_diversity_threshold
            and num_active_ads >= self.min_active_ads_threshold
        ):
            bonus = self.diversity_bonus_pct
            # Extra bonus for very high diversity
            if ad_diversity >= 10 and num_active_ads >= 5:
                bonus = self.diversity_bonus_pct * 1.5
            return (
                1 + bonus,
                f"ad_diversity_bonus_diversity_{ad_diversity}_active_{num_active_ads}",
            )

        # Rule 4: Ad Performance Consistency
        # Reward adsets with consistent ad performance
        if (
            roas_7d >= self.low_roas_threshold
            and ad_roas_mean >= self.medium_roas_threshold
            and ad_roas_std <= self.max_roas_std
            and num_active_ads >= 2  # Need at least 2 ads to measure consistency
        ):
            return (
                1 + self.consistency_bonus_pct,
                f"ad_consistency_bonus_std_{ad_roas_std:.3f}_mean_{ad_roas_mean:.2f}",
            )

        # Rule 5: Format Diversity Bonus
        # Reward adsets with diverse ad formats
        if (
            roas_7d >= self.low_roas_threshold
            and format_diversity_score >= self.min_format_diversity
        ):
            return (
                1 + self.format_diversity_bonus_pct,
                f"format_diversity_bonus_formats_{format_diversity_score}",
            )

        # Rule 6: Balanced Spend Distribution (Low Gini)
        # Reward adsets with balanced spend across ads
        if (
            roas_7d >= self.low_roas_threshold
            and ad_spend_gini < 0.4  # Well-distributed spend
            and num_active_ads >= 3  # Only if has multiple ads
        ):
            return (
                1 + self.conservative_increase_pct,
                f"balanced_spend_distribution_gini_{ad_spend_gini:.3f}",
            )

        return None

    def _check_declining_performers(self, params_dict) -> Optional[Tuple[float, str]]:
        """Check Tier 6: Declining performers rules"""
        roas_7d = params_dict["roas_7d"]
        roas_trend = params_dict["roas_trend"]
        health_score = params_dict["health_score"]
        days_active = params_dict["days_active"]
        roas_vs_adset = params_dict["roas_vs_adset"]

        result = self._check_low_roas_falling(roas_7d, roas_trend, health_score)
        if result:
            return result

        result = self._check_low_roas_improving(roas_7d, roas_trend, days_active)
        if result:
            return result

        result = self._check_medium_roas_falling(roas_7d, roas_trend, health_score)
        if result:
            return result

        return self._check_rel_underperformance(
            roas_vs_adset, roas_7d, roas_trend, health_score
        )

    def _check_low_roas_falling(
        self, roas_7d, roas_trend, health_score
    ) -> Optional[Tuple[float, str]]:
        """Check low ROAS with falling trend"""
        if (
            roas_7d < self.low_roas_threshold
            and roas_trend <= self.strong_falling_trend
            and health_score >= self.unhealthy_threshold
        ):
            base_pct = self.aggressive_decrease_pct
            base_pct = self._gradient_adjustment(
                roas_7d,
                self.low_roas_threshold,
                base_pct,
                range_size=0.5,
                increasing=False,
            )
            base_pct = self._trend_scaling(roas_trend, base_pct, _is_rising=False)
            health_mult = 1.0 + (1.0 - health_score) * 0.3
            final_pct = base_pct * health_mult
            return (
                1 - final_pct,
                f"decrease_low_roas_falling_roas{roas_7d:.2f}_"
                f"trend{roas_trend:.2%}",
            )
        return None

    def _check_low_roas_improving(
        self, roas_7d, roas_trend, days_active
    ) -> Optional[Tuple[float, str]]:
        """Check low ROAS but improving"""
        if roas_7d < self.low_roas_threshold and roas_trend > 0:
            if days_active < 7:
                return (
                    1.0,
                    f"hold_low_roas_improving_new_roas{roas_7d:.2f}_"
                    f"trend{roas_trend:.2%}_days{days_active}",
                )
            if days_active < 14:
                return (
                    1.0,
                    f"hold_low_roas_improving_learning_roas{roas_7d:.2f}_"
                    f"trend{roas_trend:.2%}_days{days_active}",
                )
            if roas_trend >= self.moderate_rising_trend:
                return (
                    1.0,
                    f"hold_low_roas_improving_established_strong_"
                    f"trend_roas{roas_7d:.2f}_trend{roas_trend:.2%}",
                )
            return (
                1 - self.conservative_increase_pct * 0.5,
                f"moderate_decrease_low_roas_improving_weak_"
                f"trend_roas{roas_7d:.2f}_trend{roas_trend:.2%}",
            )
        return None

    def _check_medium_roas_falling(
        self, roas_7d, roas_trend, health_score
    ) -> Optional[Tuple[float, str]]:
        """Check medium ROAS with falling trend"""
        if (
            self.low_roas_threshold <= roas_7d < self.high_roas_threshold
            and roas_trend <= self.strong_falling_trend
        ):
            trend_abs = abs(roas_trend)
            base_pct = (
                self.aggressive_decrease_pct
                if trend_abs >= 0.15
                else self.moderate_decrease_pct
            )
            final_pct = base_pct * self._health_score_multiplier(health_score)
            return (
                1 - final_pct,
                f"decrease_medium_roas_falling_roas{roas_7d:.2f}_"
                f"trend{roas_trend:.2%}",
            )
        return None

    def _check_rel_underperformance(
        self, roas_vs_adset, roas_7d, roas_trend, health_score
    ) -> Optional[Tuple[float, str]]:
        """Check relative underperformance"""
        if (
            roas_vs_adset is not None
            and roas_vs_adset < self.low_vs_adset_threshold
            and roas_7d < self.high_roas_threshold
            and roas_trend <= 0
        ):
            base_pct = self.moderate_decrease_pct
            base_pct = self._relative_performance_gradient(
                roas_vs_adset,
                self.low_vs_adset_threshold,
                base_pct,
                is_above=False,
            )
            final_pct = base_pct * self._health_score_multiplier(health_score)
            return (
                1 - final_pct,
                f"underperforming_vs_adset_roas_vs_adset{roas_vs_adset:.2f}_"
                f"roas{roas_7d:.2f}",
            )
        return None

    def _check_time_based_adjustments(self, params_dict) -> Optional[Tuple[float, str]]:
        """Check Tier 7: Time-based adjustments"""
        is_weekend = params_dict["is_weekend"]
        roas_7d = params_dict["roas_7d"]
        week_of_year = params_dict["week_of_year"]
        day_of_week = params_dict["day_of_week"]

        if is_weekend is not None and is_weekend and roas_7d >= self.low_roas_threshold:
            boost = (
                0.03
                if roas_7d >= self.high_roas_threshold
                else 0.02 if roas_7d >= self.medium_roas_threshold else 0.01
            )
            return (1 + boost, f"weekend_boost_roas{roas_7d:.2f}")

        if (
            week_of_year is not None
            and 48 <= week_of_year <= 52
            and roas_7d >= self.low_roas_threshold
        ):
            base_boost = self._q4_dynamic_boost(week_of_year)
            boost = (
                base_boost * 1.2 if roas_7d >= self.high_roas_threshold else base_boost
            )
            return (1 + boost, f"q4_boost_week{week_of_year}_roas{roas_7d:.2f}")

        if day_of_week is not None and roas_7d >= self.low_roas_threshold:
            if day_of_week == 0 and roas_7d >= self.high_roas_threshold:
                return (1 + 0.01, "monday_recovery_boost")

        return None

    def _check_lifecycle_rules(self, params_dict) -> Optional[Tuple[float, str]]:
        """Check Tier 8: Lifecycle-based rules"""
        days_active = params_dict["days_active"]
        roas_7d = params_dict["roas_7d"]
        roas_trend = params_dict["roas_trend"]
        health_score = params_dict["health_score"]
        clicks = params_dict["clicks"]

        result = self._check_cold_start(days_active, roas_7d)
        if result:
            return result

        result = self._check_early_learning(days_active, roas_7d, clicks)
        if result:
            return result

        result = self._check_mid_learning(days_active, roas_7d, roas_trend, clicks)
        if result:
            return result

        result = self._check_late_learning(days_active, roas_7d, health_score)
        if result:
            return result

        return self._check_established_adset(
            days_active, roas_7d, roas_trend, health_score
        )

    def _check_cold_start(self, days_active, roas_7d) -> Optional[Tuple[float, str]]:
        """Check cold start lifecycle rules"""
        if days_active <= self.cold_start_days:
            max_increase = 0.05
            if roas_7d >= self.high_roas_threshold:
                increase = min(0.03, max_increase)
                return (
                    1 + increase,
                    f"cold_start_excellent_start_days{days_active}_"
                    f"roas{roas_7d:.2f}",
                )
            if roas_7d >= self.low_roas_threshold:
                increase = min(0.01, max_increase * 0.5)
                return (
                    1 + increase,
                    f"cold_start_good_start_days{days_active}_" f"roas{roas_7d:.2f}",
                )
            return (1.0, f"cold_start_wait_days{days_active}_roas{roas_7d:.2f}")
        return None

    def _check_early_learning(
        self, days_active, roas_7d, clicks
    ) -> Optional[Tuple[float, str]]:
        """Check early learning phase lifecycle rules"""
        if days_active <= self.learning_phase_days_early:
            max_increase = 0.10
            if roas_7d >= self.high_roas_threshold:
                base_pct = min(self.moderate_increase_pct, max_increase)
                base_pct = self._sample_size_confidence(clicks, base_pct)
                return (
                    1 + base_pct,
                    f"early_learning_good_performer_days{days_active}_"
                    f"roas{roas_7d:.2f}",
                )
            if roas_7d >= self.low_roas_threshold:
                base_pct = min(self.conservative_increase_pct, max_increase * 0.5)
                return (
                    1 + base_pct,
                    f"early_learning_moderate_days{days_active}_" f"roas{roas_7d:.2f}",
                )
        return None

    def _check_mid_learning(
        self, days_active, roas_7d, roas_trend, clicks
    ) -> Optional[Tuple[float, str]]:
        """Check mid learning phase lifecycle rules"""
        if days_active <= self.learning_phase_days_mid:
            max_increase = 0.15
            if roas_7d >= self.high_roas_threshold and roas_trend >= 0:
                base_pct = min(self.moderate_increase_pct, max_increase)
                trend_pct = (
                    self._trend_scaling(roas_trend, base_pct, _is_rising=True)
                    if roas_trend > 0
                    else base_pct
                )
                trend_pct = self._sample_size_confidence(clicks, trend_pct)
                return (
                    1 + trend_pct,
                    f"mid_learning_performer_days{days_active}_" f"roas{roas_7d:.2f}",
                )
        return None

    def _check_late_learning(
        self, days_active, roas_7d, health_score
    ) -> Optional[Tuple[float, str]]:
        """Check late learning phase lifecycle rules"""
        if days_active <= self.learning_phase_days_late:
            if (
                roas_7d >= self.high_roas_threshold
                and health_score >= self.healthy_threshold
            ):
                base_pct = self.moderate_increase_pct
                base_pct = base_pct * self._health_score_multiplier(health_score)
                return (
                    1 + base_pct,
                    f"late_learning_transitional_days{days_active}_"
                    f"roas{roas_7d:.2f}",
                )
        return None

    def _check_established_adset(
        self, days_active, roas_7d, roas_trend, health_score
    ) -> Optional[Tuple[float, str]]:
        """Check established adset lifecycle rules"""
        if (
            days_active >= self.established_days
            and roas_7d >= self.high_roas_threshold
            and roas_trend >= 0
            and health_score >= self.healthy_threshold
        ):
            base_pct = self.conservative_increase_pct
            trend_pct = (
                self._trend_scaling(roas_trend, base_pct, _is_rising=True)
                if roas_trend > 0
                else base_pct
            )
            final_pct = trend_pct * self._health_score_multiplier(health_score)
            return (
                1 + final_pct,
                f"established_consistent_performer_days{days_active}_"
                f"roas{roas_7d:.2f}",
            )
        return None

    def _check_time_weighted_smoothing(
        self, params_dict
    ) -> Optional[Tuple[float, str]]:
        """Check Tier 14: Time-weighted & smoothing rules"""
        roas_trend = params_dict["roas_trend"]
        roas_7d = params_dict["roas_7d"]

        if abs(roas_trend) > 0.15:
            if roas_trend > 0 and roas_7d >= self.low_roas_threshold:
                return (1 + self.moderate_increase_pct, "strong_recent_trend")
            if roas_trend < 0 and roas_7d < self.high_roas_threshold:
                return (1 - self.moderate_decrease_pct, "strong_recent_decline")

        return None

    def _default_adjustment(self, params_dict) -> Tuple[float, str]:
        """Tier 15: Default/maintenance adjustment"""
        roas_7d = params_dict["roas_7d"]
        roas_trend = params_dict["roas_trend"]
        health_score = params_dict["health_score"]

        if (
            self.low_roas_threshold <= roas_7d < self.high_roas_threshold
            and -0.03 <= roas_trend <= 0.03
        ):
            return (
                1.0,
                f"maintain_stable_medium_roas{roas_7d:.2f}_" f"trend{roas_trend:.2%}",
            )

        return (
            1.0,
            f"maintain_status_quo_roas{roas_7d:.2f}_"
            f"trend{roas_trend:.2%}_health{health_score:.2f}",
        )
