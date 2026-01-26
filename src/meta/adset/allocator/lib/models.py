"""
Data models for rule-based budget allocation.

This module defines dataclasses for type-safe parameter passing.
"""


class BudgetAllocationMetrics:
    """
    Metrics for budget allocation calculation.

    Required fields:
        - adset_id: Adset identifier
        - current_budget: Current budget for the adset
        - roas_7d: 7-day rolling ROAS
        - roas_trend: ROAS trend (change rate)

    All other fields are optional and default to None.
    """

    def __init__(
        self,
        adset_id: str,
        current_budget: float,
        roas_7d: float,
        roas_trend: float,
        **kwargs,
    ):
        # Required fields
        self.adset_id = adset_id
        self.current_budget = current_budget
        self.roas_7d = roas_7d
        self.roas_trend = roas_trend
        # Optional fields stored in dict to reduce instance attributes
        self._optional = {
            "previous_budget": kwargs.get("previous_budget"),
            "adset_roas": kwargs.get("adset_roas"),
            "campaign_roas": kwargs.get("campaign_roas"),
            "account_roas": kwargs.get("account_roas"),
            "roas_vs_adset": kwargs.get("roas_vs_adset"),
            "roas_vs_campaign": kwargs.get("roas_vs_campaign"),
            "roas_vs_account": kwargs.get("roas_vs_account"),
            "efficiency": kwargs.get("efficiency"),
            "revenue_per_impression": kwargs.get("revenue_per_impression"),
            "revenue_per_click": kwargs.get("revenue_per_click"),
            "spend": kwargs.get("spend"),
            "spend_rolling_7d": kwargs.get("spend_rolling_7d"),
            "impressions": kwargs.get("impressions"),
            "clicks": kwargs.get("clicks"),
            "reach": kwargs.get("reach"),
            "adset_spend": kwargs.get("adset_spend"),
            "campaign_spend": kwargs.get("campaign_spend"),
            "expected_clicks": kwargs.get("expected_clicks"),
            "health_score": kwargs.get("health_score", 0.5),
            "days_active": kwargs.get("days_active", 0),
            "day_of_week": kwargs.get("day_of_week"),
            "is_weekend": kwargs.get("is_weekend"),
            "week_of_year": kwargs.get("week_of_year"),
            "adaptive_target_roas": kwargs.get("adaptive_target_roas"),
            "static_target_roas": kwargs.get("static_target_roas"),
            "budget_utilization": kwargs.get("budget_utilization"),
            "marginal_roas": kwargs.get("marginal_roas"),
            "total_budget_today": kwargs.get("total_budget_today"),
            # P0-2: Rolling coverage metrics for confidence weighting
            "rolling_7d_coverage": kwargs.get("rolling_7d_coverage", 1.0),
            "rolling_14d_coverage": kwargs.get("rolling_14d_coverage", 1.0),
            # Ad-level statistics (newly added)
            "num_ads": kwargs.get("num_ads", 1),
            "num_active_ads": kwargs.get("num_active_ads", 0),
            "ad_diversity": kwargs.get("ad_diversity", 1),
            "ad_roas_mean": kwargs.get("ad_roas_mean", 0.0),
            "ad_roas_std": kwargs.get("ad_roas_std", 0.0),
            "ad_roas_range": kwargs.get("ad_roas_range", 0.0),
            "ad_spend_gini": kwargs.get("ad_spend_gini", 0.0),
            "top_ad_spend_pct": kwargs.get("top_ad_spend_pct", 1.0),
            "video_ads_ratio": kwargs.get("video_ads_ratio", 0.0),
            "format_diversity_score": kwargs.get("format_diversity_score", 1),
            # Shopify integration: Actual revenue-based ROAS
            "shopify_roas": kwargs.get("shopify_roas"),
            "shopify_revenue": kwargs.get("shopify_revenue"),
        }

    @property
    def previous_budget(self):
        """Get the previous budget value."""
        return self._optional["previous_budget"]

    @property
    def adset_roas(self):
        """Get the adset-level ROAS value."""
        return self._optional["adset_roas"]

    @property
    def campaign_roas(self):
        """Get the campaign-level ROAS value."""
        return self._optional["campaign_roas"]

    @property
    def account_roas(self):
        """Get the account-level ROAS value."""
        return self._optional["account_roas"]

    @property
    def roas_vs_adset(self):
        """Get the ROAS comparison against adset average."""
        return self._optional["roas_vs_adset"]

    @property
    def roas_vs_campaign(self):
        """Get the ROAS comparison against campaign average."""
        return self._optional["roas_vs_campaign"]

    @property
    def roas_vs_account(self):
        """Get the ROAS comparison against account average."""
        return self._optional["roas_vs_account"]

    @property
    def efficiency(self):
        """Get the efficiency metric value."""
        return self._optional["efficiency"]

    @property
    def revenue_per_impression(self):
        """Get the revenue per impression value."""
        return self._optional["revenue_per_impression"]

    @property
    def revenue_per_click(self):
        """Get the revenue per click value."""
        return self._optional["revenue_per_click"]

    @property
    def spend(self):
        """Get the spend value."""
        return self._optional["spend"]

    @property
    def spend_rolling_7d(self):
        """Get the 7-day rolling spend value."""
        return self._optional["spend_rolling_7d"]

    @property
    def impressions(self):
        """Get the impressions value."""
        return self._optional["impressions"]

    @property
    def clicks(self):
        """Get the clicks value."""
        return self._optional["clicks"]

    @property
    def reach(self):
        """Get the reach value."""
        return self._optional["reach"]

    @property
    def adset_spend(self):
        """Get the adset-level spend value."""
        return self._optional["adset_spend"]

    @property
    def campaign_spend(self):
        """Get the campaign-level spend value."""
        return self._optional["campaign_spend"]

    @property
    def expected_clicks(self):
        """Get the expected clicks value."""
        return self._optional["expected_clicks"]

    @property
    def health_score(self):
        """Get the health score value."""
        return self._optional["health_score"]

    @property
    def days_active(self):
        """Get the days active value."""
        return self._optional["days_active"]

    @property
    def day_of_week(self):
        """Get the day of week value."""
        return self._optional["day_of_week"]

    @property
    def is_weekend(self):
        """Get whether it's a weekend (boolean value)."""
        return self._optional["is_weekend"]

    @property
    def week_of_year(self):
        """Get the week of year value."""
        return self._optional["week_of_year"]

    @property
    def adaptive_target_roas(self):
        """Get the adaptive target ROAS value."""
        return self._optional["adaptive_target_roas"]

    @property
    def static_target_roas(self):
        """Get the static target ROAS value."""
        return self._optional["static_target_roas"]

    @property
    def budget_utilization(self):
        """Get the budget utilization value."""
        return self._optional["budget_utilization"]

    @property
    def marginal_roas(self):
        """Get the marginal ROAS value."""
        return self._optional["marginal_roas"]

    @property
    def total_budget_today(self):
        """Get the total budget today value."""
        return self._optional["total_budget_today"]

    # P0-2: Rolling coverage properties
    @property
    def rolling_7d_coverage(self):
        """Get the 7-day rolling window coverage (0-1)."""
        return self._optional["rolling_7d_coverage"]

    @property
    def rolling_14d_coverage(self):
        """Get the 14-day rolling window coverage (0-1)."""
        return self._optional["rolling_14d_coverage"]

    # Ad-level statistics properties (newly added)
    @property
    def num_ads(self):
        """Get the number of ads in adset."""
        return self._optional["num_ads"]

    @property
    def num_active_ads(self):
        """Get the number of active ads (with spend) in adset."""
        return self._optional["num_active_ads"]

    @property
    def ad_diversity(self):
        """Get the ad diversity (unique ad names) in adset."""
        return self._optional["ad_diversity"]

    @property
    def ad_roas_mean(self):
        """Get the mean ROAS across ads in adset."""
        return self._optional["ad_roas_mean"]

    @property
    def ad_roas_std(self):
        """Get the standard deviation of ROAS across ads in adset."""
        return self._optional["ad_roas_std"]

    @property
    def ad_roas_range(self):
        """Get the range (max-min) of ROAS across ads in adset."""
        return self._optional["ad_roas_range"]

    @property
    def ad_spend_gini(self):
        """Get the Gini coefficient of spend distribution (0=equal, 1=concentrated)."""
        return self._optional["ad_spend_gini"]

    @property
    def top_ad_spend_pct(self):
        """Get the percentage of spend on top-performing ad."""
        return self._optional["top_ad_spend_pct"]

    @property
    def video_ads_ratio(self):
        """Get the ratio of video ads to total ads."""
        return self._optional["video_ads_ratio"]

    @property
    def format_diversity_score(self):
        """Get the number of different ad formats in adset."""
        return self._optional["format_diversity_score"]

    @property
    def shopify_roas(self):
        """Get the Shopify ROAS (actual revenue-based)."""
        return self._optional.get("shopify_roas")

    @property
    def shopify_revenue(self):
        """Get the Shopify revenue for this date."""
        return self._optional.get("shopify_revenue")

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "adset_id": self.adset_id,
            "current_budget": self.current_budget,
            "previous_budget": self.previous_budget,
            "roas_7d": self.roas_7d,
            "roas_trend": self.roas_trend,
            "adset_roas": self.adset_roas,
            "campaign_roas": self.campaign_roas,
            "account_roas": self.account_roas,
            "roas_vs_adset": self.roas_vs_adset,
            "roas_vs_campaign": self.roas_vs_campaign,
            "roas_vs_account": self.roas_vs_account,
            "efficiency": self.efficiency,
            "revenue_per_impression": self.revenue_per_impression,
            "revenue_per_click": self.revenue_per_click,
            "spend": self.spend,
            "spend_rolling_7d": self.spend_rolling_7d,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "reach": self.reach,
            "adset_spend": self.adset_spend,
            "campaign_spend": self.campaign_spend,
            "expected_clicks": self.expected_clicks,
            "health_score": self.health_score,
            "days_active": self.days_active,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "week_of_year": self.week_of_year,
            "adaptive_target_roas": self.adaptive_target_roas,
            "static_target_roas": self.static_target_roas,
            "budget_utilization": self.budget_utilization,
            "marginal_roas": self.marginal_roas,
            "total_budget_today": self.total_budget_today,
            # P0-2: Rolling coverage metrics
            "rolling_7d_coverage": self.rolling_7d_coverage,
            "rolling_14d_coverage": self.rolling_14d_coverage,
            # Ad-level statistics (newly added)
            "num_ads": self.num_ads,
            "num_active_ads": self.num_active_ads,
            "ad_diversity": self.ad_diversity,
            "ad_roas_mean": self.ad_roas_mean,
            "ad_roas_std": self.ad_roas_std,
            "ad_roas_range": self.ad_roas_range,
            "ad_spend_gini": self.ad_spend_gini,
            "top_ad_spend_pct": self.top_ad_spend_pct,
            "video_ads_ratio": self.video_ads_ratio,
            "format_diversity_score": self.format_diversity_score,
            # Shopify integration
            "shopify_roas": self.shopify_roas,
            "shopify_revenue": self.shopify_revenue,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BudgetAllocationMetrics":
        """Create from dictionary."""
        # Provide defaults for required fields if missing
        adset_id = data.get("adset_id", "")
        current_budget = data.get("current_budget", 0.0)
        roas_7d = data.get("roas_7d", 0.0)
        roas_trend = data.get("roas_trend", 0.0)
        # Extract optional fields
        optional_fields = {
            "previous_budget",
            "adset_roas",
            "campaign_roas",
            "account_roas",
            "roas_vs_adset",
            "roas_vs_campaign",
            "roas_vs_account",
            "efficiency",
            "revenue_per_impression",
            "revenue_per_click",
            "spend",
            "spend_rolling_7d",
            "impressions",
            "clicks",
            "reach",
            "adset_spend",
            "campaign_spend",
            "expected_clicks",
            "health_score",
            "days_active",
            "day_of_week",
            "is_weekend",
            "week_of_year",
            "adaptive_target_roas",
            "static_target_roas",
            "budget_utilization",
            "marginal_roas",
            "total_budget_today",
            # P0-2: Rolling coverage metrics
            "rolling_7d_coverage",
            "rolling_14d_coverage",
            # Ad-level statistics (newly added)
            "num_ads",
            "num_active_ads",
            "ad_diversity",
            "ad_roas_mean",
            "ad_roas_std",
            "ad_roas_range",
            "ad_spend_gini",
            "top_ad_spend_pct",
            "video_ads_ratio",
            "format_diversity_score",
            "rolling_low_quality",
            # Shopify integration
            "shopify_roas",
            "shopify_revenue",
        }
        kwargs = {k: data.get(k) for k in optional_fields if k in data}
        # Set defaults for health_score and days_active
        if "health_score" not in kwargs:
            kwargs["health_score"] = 0.5
        if "days_active" not in kwargs:
            kwargs["days_active"] = 0
        # P0-2: Set defaults for rolling coverage
        if "rolling_7d_coverage" not in kwargs:
            kwargs["rolling_7d_coverage"] = 1.0
        if "rolling_14d_coverage" not in kwargs:
            kwargs["rolling_14d_coverage"] = 1.0
        # Set defaults for rolling_low_quality
        if "rolling_low_quality" not in kwargs:
            kwargs["rolling_low_quality"] = 0
        # Set defaults for ad-level statistics
        if "num_ads" not in kwargs:
            kwargs["num_ads"] = 1
        if "num_active_ads" not in kwargs:
            kwargs["num_active_ads"] = 0
        if "ad_diversity" not in kwargs:
            kwargs["ad_diversity"] = 1
        if "ad_roas_mean" not in kwargs:
            kwargs["ad_roas_mean"] = 0.0
        if "ad_roas_std" not in kwargs:
            kwargs["ad_roas_std"] = 0.0
        if "ad_roas_range" not in kwargs:
            kwargs["ad_roas_range"] = 0.0
        if "ad_spend_gini" not in kwargs:
            kwargs["ad_spend_gini"] = 0.0
        if "top_ad_spend_pct" not in kwargs:
            kwargs["top_ad_spend_pct"] = 1.0
        if "video_ads_ratio" not in kwargs:
            kwargs["video_ads_ratio"] = 0.0
        if "format_diversity_score" not in kwargs:
            kwargs["format_diversity_score"] = 1
        return cls(
            adset_id=adset_id,
            current_budget=current_budget,
            roas_7d=roas_7d,
            roas_trend=roas_trend,
            **kwargs,
        )


class BudgetAdjustmentParams:
    """
    Parameters for budget adjustment calculation.

    Required fields:
        - roas_7d: 7-day rolling ROAS
        - roas_trend: ROAS trend (change rate)

    All other fields are optional and default to None.
    """

    def __init__(
        self,
        roas_7d: float,
        roas_trend: float,
        **kwargs,
    ):
        # Required fields
        self.roas_7d = roas_7d
        self.roas_trend = roas_trend
        # Optional fields stored in dict to reduce instance attributes
        self._optional = {
            "current_budget": kwargs.get("current_budget"),
            "adset_roas": kwargs.get("adset_roas"),
            "campaign_roas": kwargs.get("campaign_roas"),
            "account_roas": kwargs.get("account_roas"),
            "roas_vs_adset": kwargs.get("roas_vs_adset"),
            "roas_vs_campaign": kwargs.get("roas_vs_campaign"),
            "roas_vs_account": kwargs.get("roas_vs_account"),
            "efficiency": kwargs.get("efficiency"),
            "revenue_per_impression": kwargs.get("revenue_per_impression"),
            "revenue_per_click": kwargs.get("revenue_per_click"),
            "spend": kwargs.get("spend"),
            "spend_rolling_7d": kwargs.get("spend_rolling_7d"),
            "impressions": kwargs.get("impressions"),
            "clicks": kwargs.get("clicks"),
            "reach": kwargs.get("reach"),
            "adset_spend": kwargs.get("adset_spend"),
            "campaign_spend": kwargs.get("campaign_spend"),
            "expected_clicks": kwargs.get("expected_clicks"),
            "health_score": kwargs.get("health_score", 0.5),
            "days_active": kwargs.get("days_active", 0),
            "day_of_week": kwargs.get("day_of_week"),
            "is_weekend": kwargs.get("is_weekend"),
            "week_of_year": kwargs.get("week_of_year"),
            # Ad-level statistics (newly added)
            "num_ads": kwargs.get("num_ads", 1),
            "num_active_ads": kwargs.get("num_active_ads", 0),
            "ad_diversity": kwargs.get("ad_diversity", 1),
            "ad_roas_mean": kwargs.get("ad_roas_mean", 0.0),
            "ad_roas_std": kwargs.get("ad_roas_std", 0.0),
            "ad_roas_range": kwargs.get("ad_roas_range", 0.0),
            "ad_spend_gini": kwargs.get("ad_spend_gini", 0.0),
            "top_ad_spend_pct": kwargs.get("top_ad_spend_pct", 1.0),
            "video_ads_ratio": kwargs.get("video_ads_ratio", 0.0),
            "format_diversity_score": kwargs.get("format_diversity_score", 1),
            "rolling_low_quality": kwargs.get("rolling_low_quality", 0),
            # Shopify integration: actual revenue-based ROAS
            "shopify_roas": kwargs.get("shopify_roas"),
            "shopify_revenue": kwargs.get("shopify_revenue"),
        }

    @property
    def current_budget(self):
        """Get the current budget value."""
        return self._optional["current_budget"]

    @property
    def adset_roas(self):
        """Get the adset-level ROAS value."""
        return self._optional["adset_roas"]

    @property
    def campaign_roas(self):
        """Get the campaign-level ROAS value."""
        return self._optional["campaign_roas"]

    @property
    def account_roas(self):
        """Get the account-level ROAS value."""
        return self._optional["account_roas"]

    @property
    def roas_vs_adset(self):
        """Get the ROAS comparison against adset average."""
        return self._optional["roas_vs_adset"]

    @property
    def roas_vs_campaign(self):
        """Get the ROAS comparison against campaign average."""
        return self._optional["roas_vs_campaign"]

    @property
    def roas_vs_account(self):
        """Get the ROAS comparison against account average."""
        return self._optional["roas_vs_account"]

    @property
    def efficiency(self):
        """Get the efficiency metric value."""
        return self._optional["efficiency"]

    @property
    def revenue_per_impression(self):
        """Get the revenue per impression value."""
        return self._optional["revenue_per_impression"]

    @property
    def revenue_per_click(self):
        """Get the revenue per click value."""
        return self._optional["revenue_per_click"]

    @property
    def spend(self):
        """Get the spend value."""
        return self._optional["spend"]

    @property
    def spend_rolling_7d(self):
        """Get the 7-day rolling spend value."""
        return self._optional["spend_rolling_7d"]

    @property
    def impressions(self):
        """Get the impressions value."""
        return self._optional["impressions"]

    @property
    def clicks(self):
        """Get the clicks value."""
        return self._optional["clicks"]

    @property
    def reach(self):
        """Get the reach value."""
        return self._optional["reach"]

    @property
    def adset_spend(self):
        """Get the adset-level spend value."""
        return self._optional["adset_spend"]

    @property
    def campaign_spend(self):
        """Get the campaign-level spend value."""
        return self._optional["campaign_spend"]

    @property
    def expected_clicks(self):
        """Get the expected clicks value."""
        return self._optional["expected_clicks"]

    @property
    def health_score(self):
        """Get the health score value."""
        return self._optional["health_score"]

    @property
    def days_active(self):
        """Get the days active value."""
        return self._optional["days_active"]

    @property
    def day_of_week(self):
        """Get the day of week value."""
        return self._optional["day_of_week"]

    @property
    def is_weekend(self):
        """Get whether it's a weekend (boolean value)."""
        return self._optional["is_weekend"]

    @property
    def week_of_year(self):
        """Get the week of year value."""
        return self._optional["week_of_year"]

    # Ad-level statistics properties (newly added)
    @property
    def num_ads(self):
        """Get the number of ads in adset."""
        return self._optional["num_ads"]

    @property
    def num_active_ads(self):
        """Get the number of active ads (with spend) in adset."""
        return self._optional["num_active_ads"]

    @property
    def ad_diversity(self):
        """Get the ad diversity (unique ad names) in adset."""
        return self._optional["ad_diversity"]

    @property
    def ad_roas_mean(self):
        """Get the mean ROAS across ads in adset."""
        return self._optional["ad_roas_mean"]

    @property
    def ad_roas_std(self):
        """Get the standard deviation of ROAS across ads in adset."""
        return self._optional["ad_roas_std"]

    @property
    def ad_roas_range(self):
        """Get the range (max-min) of ROAS across ads in adset."""
        return self._optional["ad_roas_range"]

    @property
    def ad_spend_gini(self):
        """Get the Gini coefficient of spend distribution (0=equal, 1=concentrated)."""
        return self._optional["ad_spend_gini"]

    @property
    def top_ad_spend_pct(self):
        """Get the percentage of spend on top-performing ad."""
        return self._optional["top_ad_spend_pct"]

    @property
    def video_ads_ratio(self):
        """Get the ratio of video ads to total ads."""
        return self._optional["video_ads_ratio"]

    @property
    def format_diversity_score(self):
        """Get the number of different ad formats in adset."""
        return self._optional["format_diversity_score"]

    @property
    def rolling_low_quality(self):
        """Get flag indicating if rolling window metrics are low quality (< 50% coverage)."""
        return self._optional["rolling_low_quality"]

    @property
    def shopify_roas(self):
        """Get the Shopify ROAS (actual revenue-based)."""
        return self._optional.get("shopify_roas")

    @property
    def shopify_revenue(self):
        """Get the Shopify revenue for this date."""
        return self._optional.get("shopify_revenue")

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "roas_7d": self.roas_7d,
            "roas_trend": self.roas_trend,
            "current_budget": self.current_budget,
            "adset_roas": self.adset_roas,
            "campaign_roas": self.campaign_roas,
            "account_roas": self.account_roas,
            "roas_vs_adset": self.roas_vs_adset,
            "roas_vs_campaign": self.roas_vs_campaign,
            "roas_vs_account": self.roas_vs_account,
            "efficiency": self.efficiency,
            "revenue_per_impression": self.revenue_per_impression,
            "revenue_per_click": self.revenue_per_click,
            "spend": self.spend,
            "spend_rolling_7d": self.spend_rolling_7d,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "reach": self.reach,
            "adset_spend": self.adset_spend,
            "campaign_spend": self.campaign_spend,
            "expected_clicks": self.expected_clicks,
            "health_score": self.health_score,
            "days_active": self.days_active,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "week_of_year": self.week_of_year,
            # Ad-level statistics (newly added)
            "num_ads": self.num_ads,
            "num_active_ads": self.num_active_ads,
            "ad_diversity": self.ad_diversity,
            "ad_roas_mean": self.ad_roas_mean,
            "ad_roas_std": self.ad_roas_std,
            "ad_roas_range": self.ad_roas_range,
            "ad_spend_gini": self.ad_spend_gini,
            "top_ad_spend_pct": self.top_ad_spend_pct,
            "video_ads_ratio": self.video_ads_ratio,
            "format_diversity_score": self.format_diversity_score,
            "rolling_low_quality": self.rolling_low_quality,
            # Shopify integration
            "shopify_roas": self.shopify_roas,
            "shopify_revenue": self.shopify_revenue,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BudgetAdjustmentParams":
        """Create from dictionary."""
        # Provide defaults for required fields if missing
        roas_7d = data.get("roas_7d", 0.0)
        roas_trend = data.get("roas_trend", 0.0)
        # Extract optional fields
        optional_fields = {
            "current_budget",
            "adset_roas",
            "campaign_roas",
            "account_roas",
            "roas_vs_adset",
            "roas_vs_campaign",
            "roas_vs_account",
            "efficiency",
            "revenue_per_impression",
            "revenue_per_click",
            "spend",
            "spend_rolling_7d",
            "impressions",
            "clicks",
            "reach",
            "adset_spend",
            "campaign_spend",
            "expected_clicks",
            "health_score",
            "days_active",
            "day_of_week",
            "is_weekend",
            "week_of_year",
            # Ad-level statistics (newly added)
            "num_ads",
            "num_active_ads",
            "ad_diversity",
            "ad_roas_mean",
            "ad_roas_std",
            "ad_roas_range",
            "ad_spend_gini",
            "top_ad_spend_pct",
            "video_ads_ratio",
            "format_diversity_score",
            "rolling_low_quality",
            # Shopify integration
            "shopify_roas",
            "shopify_revenue",
        }
        kwargs = {k: data.get(k) for k in optional_fields if k in data}
        # Set defaults for health_score and days_active
        if "health_score" not in kwargs:
            kwargs["health_score"] = 0.5
        if "days_active" not in kwargs:
            kwargs["days_active"] = 0
        # Set defaults for rolling_low_quality
        if "rolling_low_quality" not in kwargs:
            kwargs["rolling_low_quality"] = 0
        # Set defaults for ad-level statistics
        if "num_ads" not in kwargs:
            kwargs["num_ads"] = 1
        if "num_active_ads" not in kwargs:
            kwargs["num_active_ads"] = 0
        if "ad_diversity" not in kwargs:
            kwargs["ad_diversity"] = 1
        if "ad_roas_mean" not in kwargs:
            kwargs["ad_roas_mean"] = 0.0
        if "ad_roas_std" not in kwargs:
            kwargs["ad_roas_std"] = 0.0
        if "ad_roas_range" not in kwargs:
            kwargs["ad_roas_range"] = 0.0
        if "ad_spend_gini" not in kwargs:
            kwargs["ad_spend_gini"] = 0.0
        if "top_ad_spend_pct" not in kwargs:
            kwargs["top_ad_spend_pct"] = 1.0
        if "video_ads_ratio" not in kwargs:
            kwargs["video_ads_ratio"] = 0.0
        if "format_diversity_score" not in kwargs:
            kwargs["format_diversity_score"] = 1
        return cls(roas_7d=roas_7d, roas_trend=roas_trend, **kwargs)
