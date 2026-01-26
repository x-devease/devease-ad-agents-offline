"""
Detect human mistakes in audience setup.

Finds obvious errors that waste money or miss opportunities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from src.utils import Config


@dataclass
class Issue:
    """Represents a detected issue with an adset."""

    adset_id: str
    issue_type: str
    description: str
    opportunity: str
    priority: str
    confidence: str  # HIGH/MEDIUM/LOW
    current_spend: float
    current_roas: float
    suggested_action: str
    evidence: Dict  # Supporting data


class MistakeDetector:
    """
    Detects audience and region targeting issues.

    Comprehensive detection of audience optimization and performance issues:
    1. Too broad targeting (age range too wide)
    2. Missing lookalike audiences
    3. Geographic targeting too broad (too many countries)
    4. Audience saturation (high frequency - people seeing ad too often)
    5. Underfunded winners (high ROAS but low spend)
    6. Too narrow age range (limiting reach)
    7. Low CTR (poor creative/audience match)
    8. High CPC (expensive clicks)
    9. Low conversion rate (poor quality traffic)
    10. Low reach (very small audience)
    11. High CPM (expensive impressions)
    12. Low video engagement (poor video performance)
    13. Narrow gender targeting (limiting reach)
    14. Missing video format (image ads when video could perform better)
    15. Wrong format for audience (format mismatch with audience type)
    """

    def __init__(
        self,
        customer: str = "moprobo",
        platform: str = "meta",
        shopify_df: pd.DataFrame = None,
    ):
        """
        Initialize MistakeDetector with parameters from config.

        Args:
            customer: Customer name (default: "moprobo")
            platform: Platform name (default: "meta")
            shopify_df: Pre-loaded Shopify order data (optional)
        """
        # Load parameters from config
        params = Config.get_customer_params(customer, platform)

        # Load Shopify-based thresholds FIRST (before analyzing data)
        shopify_params = params.get("shopify", {})
        self.min_buyer_age_threshold = shopify_params.get("min_buyer_age_threshold", 35)
        self.top_buyer_states_count = shopify_params.get("top_buyer_states_count", 5)
        self.buyer_state_concentration_threshold = shopify_params.get(
            "buyer_state_concentration_threshold", 0.4
        )  # If 40%+ from top states → geo-focused buyers

        # Load or use provided Shopify data
        self.shopify_df = shopify_df
        self.shopify_buyer_states = None
        self.shopify_min_buyer_age = None
        self.shopify_geo_concentration = 0.0

        if self.shopify_df is not None and len(self.shopify_df) > 0:
            # Extract buyer patterns from Shopify data
            self._analyze_shopify_buyers()

        # Age targeting thresholds
        age_params = params.get("age_targeting", {})
        self.broad_age_range = age_params.get("broad_age_range", 30)
        self.narrow_age_range = age_params.get("narrow_age_range", 5)
        self.age_min_bound = age_params.get("age_min_bound", 18)
        self.age_max_bound = age_params.get("age_max_bound", 65)
        self.test_segment_size = age_params.get("test_segment_size", 20)
        self.test_segment_mid_offset = age_params.get("test_segment_mid_offset", 10)
        self.narrow_range_expansion = age_params.get("narrow_range_expansion", 5)

        # Lookalike thresholds
        lal_params = params.get("lookalike", {})
        self.lal_roas_threshold = lal_params.get("roas_threshold", 1.5)

        # Geographic thresholds
        geo_params = params.get("geographic", {})
        self.max_countries = geo_params.get("max_countries", 5)
        self.max_countries_display = geo_params.get("max_countries_display", 10)
        self.recommended_countries_to_test = geo_params.get(
            "recommended_countries_to_test", 3
        )

        # Gender thresholds
        gender_params = params.get("gender", {})
        self.total_genders_count = gender_params.get("total_genders_count", 2)
        self.min_custom_audiences_count = gender_params.get(
            "min_custom_audiences_count", 0
        )

        # Frequency thresholds
        freq_params = params.get("frequency", {})
        self.high_frequency_threshold = freq_params.get("high_frequency_threshold", 6.0)

        # ROAS thresholds
        roas_params = params.get("roas", {})
        self.low_roas_threshold = roas_params.get("low_roas_threshold", 1.0)
        self.medium_roas_threshold = roas_params.get("medium_roas_threshold", 1.5)
        self.high_roas_threshold = roas_params.get("high_roas_threshold", 2.5)
        self.high_roas_low_spend_threshold = roas_params.get(
            "high_roas_low_spend_threshold", 2.0
        )
        self.video_preferred_roas_threshold = roas_params.get(
            "video_preferred_roas_threshold", 1.5
        )
        self.warm_audience_roas_threshold = roas_params.get(
            "warm_audience_roas_threshold", 2.0
        )

        # Spend thresholds
        spend_params = params.get("spend", {})
        self.min_spend_for_scale = spend_params.get("min_spend_for_scale", 50.0)
        self.min_spend_for_video_format_check = spend_params.get(
            "min_spend_for_video_format_check", 20.0
        )

        # Engagement thresholds
        engagement_params = params.get("engagement", {})
        self.low_ctr_threshold = engagement_params.get("low_ctr_threshold", 0.005)
        self.high_cpc_threshold = engagement_params.get("high_cpc_threshold", 5.0)
        self.low_conversion_rate_threshold = engagement_params.get(
            "low_conversion_rate_threshold", 0.01
        )
        self.low_video_engagement_threshold = engagement_params.get(
            "low_video_engagement_threshold", 0.1
        )
        self.image_preferred_ctr_threshold = engagement_params.get(
            "image_preferred_ctr_threshold", 0.01
        )
        self.min_impressions_for_ctr = engagement_params.get(
            "min_impressions_for_ctr", 100
        )
        self.min_impressions_for_video = engagement_params.get(
            "min_impressions_for_video", 100
        )
        self.min_clicks_for_conversion = engagement_params.get(
            "min_clicks_for_conversion", 10
        )
        self.min_clicks_for_format_check = engagement_params.get(
            "min_clicks_for_format_check", 10
        )

        # Reach thresholds
        reach_params = params.get("reach", {})
        self.low_reach_threshold = reach_params.get("low_reach_threshold", 1000)

        # CPM thresholds
        cpm_params = params.get("cpm", {})
        self.high_cpm_threshold = cpm_params.get("high_cpm_threshold", 50.0)

    def _analyze_shopify_buyers(self):
        """
        Analyze Shopify order data to extract buyer patterns.

        Extracts:
        - Top buyer states
        - Minimum buyer age
        - Geographic concentration
        """
        if self.shopify_df is None or len(self.shopify_df) == 0:
            return

        # Get top buyer states
        state_counts = (
            self.shopify_df["Shipping Province"]
            .value_counts()
            .head(self.top_buyer_states_count)
        )
        self.shopify_buyer_states = set(state_counts.index.tolist())

        # Calculate geographic concentration
        total_orders = len(self.shopify_df)
        top_states_orders = state_counts.sum()
        self.shopify_geo_concentration = (
            top_states_orders / total_orders if total_orders > 0 else 0
        )

    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all issues in adsets.

        Args:
            df: Adset-level data (may have multiple rows per adset)

        Returns:
            DataFrame with detected issues (one row per issue per adset)
        """
        issues = []

        # Drop duplicates to process each adset only once
        # Take first row for each adset_id
        df_unique = df.drop_duplicates(subset=["adset_id"])

        for _, row in df_unique.iterrows():
            adset_issues = self.detect_adset_issues(row)
            issues.extend(adset_issues)

        if not issues:
            return pd.DataFrame()

        return pd.DataFrame(issues)

    def detect_adset_issues(self, adset: pd.Series) -> List[Dict]:
        """
        Detect audience and region targeting issues for a single adset.

        NO budget controls - only audience optimization suggestions.

        Args:
            adset: Row from adset dataframe

        Returns:
            List of issue dictionaries
        """
        issues = []
        roas = adset.get("purchase_roas", adset.get("roas", 0))
        spend = adset.get("spend", 0)

        # 1. TOO BROAD AGE TARGETING
        age_min = adset.get("adset_targeting_age_min", 0)
        age_max = adset.get("adset_targeting_age_max", 0)

        if age_max > 0 and age_min > 0:
            age_range = age_max - age_min
            # Only flag if broad AND performance is poor
            if age_range > self.broad_age_range and roas < self.medium_roas_threshold:
                # Estimate improvement from narrowing

                # Confidence: MEDIUM (we know it's too broad, but don't know best range)
                confidence = "MEDIUM"

                # Suggest testing multiple ranges instead of assuming one
                test_ranges = []
                mid_point = (age_min + age_max) // 2
                test_ranges.append(
                    f"{age_min}-{age_min + self.test_segment_size}"
                )  # Lower segment
                test_ranges.append(
                    f"{mid_point-self.test_segment_mid_offset}-{mid_point+self.test_segment_mid_offset}"
                )  # Middle segment
                test_ranges.append(
                    f"{age_max-self.test_segment_size}-{age_max}"
                )  # Upper segment

                issues.append(
                    {
                        "adset_id": adset.get("adset_id"),
                        "issue_type": "age_range_too_wide",
                        "description": f"Age {age_min}-{age_max} too wide ({age_range} years) - test narrower segments",
                        "opportunity": f"Testing narrower segments could improve ROAS by finding best fit",
                        "priority": "MEDIUM",
                        "confidence": confidence,
                        "current_spend": spend,
                        "current_roas": roas,
                        "suggested_action": f"Test narrower age ranges: {', '.join(test_ranges[:2])} (current: {age_min}-{age_max}, threshold: >{self.broad_age_range} years is too wide, ROAS {roas:.2f} < {self.medium_roas_threshold:.2f})",
                        "evidence": {
                            "age_min": age_min,
                            "age_max": age_max,
                            "age_range": age_range,
                            "current_value": age_range,
                            "threshold_value": self.broad_age_range,
                            "roas": roas,
                            "roas_threshold": self.medium_roas_threshold,
                            "threshold_used": f"age_range > {self.broad_age_range} AND ROAS < {self.medium_roas_threshold}",
                            "comparison": f"{age_range} > {self.broad_age_range} (exceeds by {age_range - self.broad_age_range} years), ROAS {roas:.2f} < {self.medium_roas_threshold:.2f} (below medium)",
                            "calculation": f"age_range = {age_max} - {age_min} = {age_range}",
                            "test_ranges": test_ranges,
                            "note": "Conservative approach: test before committing to specific range. Broad targeting with good ROAS is left alone.",
                        },
                    }
                )

        # 2. MISSING LOOKALIKE AUDIENCES
        lal_count = adset.get("adset_targeting_custom_audiences_count", 0)

        # Handle NaN values
        if pd.isna(lal_count):
            lal_count = 0

        if lal_count == 0 and roas < self.lal_roas_threshold:

            # Confidence: MEDIUM (estimate based on typical LAL performance)
            confidence = "MEDIUM"

            issues.append(
                {
                    "adset_id": adset.get("adset_id"),
                    "issue_type": "missing_lal",
                    "description": "No lookalike audiences (only interest targeting)",
                    "opportunity": "LALs typically 2-3x ROAS vs interest targeting",
                    "priority": "MEDIUM",
                    "confidence": confidence,
                    "current_spend": spend,
                    "current_roas": roas,
                    "suggested_action": f"Create LAL 1% from best customers (current: {lal_count} LAL audiences, ROAS {roas:.2f} < threshold {self.lal_roas_threshold})",
                    "evidence": {
                        "custom_audiences_count": lal_count,
                        "roas": roas,
                        "current_value": lal_count,
                        "threshold_value": self.min_custom_audiences_count,
                        "roas_threshold": self.lal_roas_threshold,
                        "threshold_used": f"custom_audiences_count == 0, ROAS < {self.lal_roas_threshold}",
                        "comparison": f"LAL count: {lal_count} == 0 (no lookalikes), ROAS: {roas:.2f} < {self.lal_roas_threshold} (below LAL threshold)",
                        "note": "Lookalike audiences typically outperform interest targeting by 2-3x. Start with 1% LAL from best customers.",
                    },
                }
            )

        # 3. TOO MANY COUNTRIES (Geographic targeting too broad)
        countries = adset.get("adset_targeting_countries", None)
        if countries is not None and not pd.isna(countries):
            try:
                import ast

                if isinstance(countries, str):
                    # Handle string representation like "['US', 'CA', 'MX']"
                    if countries.startswith("["):
                        country_list = ast.literal_eval(countries)
                    else:
                        country_list = [c.strip() for c in countries.split(",")]
                else:
                    country_list = (
                        countries if isinstance(countries, list) else [countries]
                    )

                country_count = (
                    len(country_list) if isinstance(country_list, list) else 1
                )

                if country_count > self.max_countries:
                    confidence = "MEDIUM"

                    issues.append(
                        {
                            "adset_id": adset.get("adset_id"),
                            "issue_type": "geographic_targeting_too_broad",
                            "description": f"Targeting {country_count} countries - too broad for effective optimization",
                            "opportunity": "Focusing on top-performing countries can improve ROAS",
                            "priority": "MEDIUM",
                            "confidence": confidence,
                            "current_spend": spend,
                            "current_roas": roas,
                            "suggested_action": f"Test top {self.recommended_countries_to_test} countries separately (current: {country_count} countries, threshold: >{self.max_countries} is too broad)",
                            "evidence": {
                                "country_count": country_count,
                                "countries": country_list[
                                    : self.max_countries_display
                                ],  # Limit display
                                "current_value": country_count,
                                "threshold_value": self.max_countries,
                                "threshold_used": f"country_count > {self.max_countries}",
                                "comparison": f"{country_count} > {self.max_countries} (exceeds by {country_count - self.max_countries} countries)",
                                "note": f"Test top {self.recommended_countries_to_test} countries separately to identify best performers before consolidating.",
                            },
                        }
                    )
            except (ValueError, SyntaxError):
                pass  # Skip if can't parse countries

        # 4. AUDIENCE SATURATION (High frequency = over-exposure)
        frequency = adset.get("frequency", 0)
        if pd.notna(frequency) and frequency > self.high_frequency_threshold:
            confidence = "HIGH"  # High confidence - frequency is measurable

            issues.append(
                {
                    "adset_id": adset.get("adset_id"),
                    "issue_type": "audience_saturation",
                    "description": f"Frequency {frequency:.1f} is high - audience is saturated (seeing ad too often)",
                    "opportunity": "Reducing frequency can improve efficiency and reduce ad fatigue",
                    "priority": "HIGH",
                    "confidence": confidence,
                    "current_spend": spend,
                    "current_roas": roas,
                    "suggested_action": f"Reduce budget or expand audience to lower frequency (current: {frequency:.1f}, threshold: >{self.high_frequency_threshold} indicates saturation)",
                    "evidence": {
                        "frequency": float(frequency),
                        "current_value": float(frequency),
                        "threshold_value": self.high_frequency_threshold,
                        "threshold_used": f"frequency > {self.high_frequency_threshold}",
                        "comparison": f"{frequency:.1f} > {self.high_frequency_threshold} (exceeds by {frequency - self.high_frequency_threshold:.1f})",
                        "saturation_risk": "high",
                        "note": "High frequency indicates ad fatigue. Reduce budget or expand audience to lower frequency and improve efficiency.",
                    },
                }
            )

        # 5. HIGH ROAS BUT LOW SPEND (Underfunded winner)
        if (
            pd.notna(roas)
            and roas >= self.high_roas_low_spend_threshold
            and spend < self.min_spend_for_scale
            and spend > 0
        ):
            # Estimate potential if scaled (conservative - assume some degradation)
            confidence = "MEDIUM"  # Medium - scaling might reduce ROAS

            issues.append(
                {
                    "adset_id": adset.get("adset_id"),
                    "issue_type": "underfunded_winner",
                    "description": f"High ROAS {roas:.2f} but low spend ${spend:.2f} - opportunity to scale",
                    "opportunity": f"Scaling budget could capture more value (with some ROAS degradation)",
                    "priority": "HIGH",
                    "confidence": confidence,
                    "current_spend": spend,
                    "current_roas": roas,
                    "suggested_action": f"Gradually increase budget to ${self.min_spend_for_scale}+ and monitor ROAS (current: ROAS {roas:.2f} >= {self.high_roas_low_spend_threshold}, spend ${spend:.2f} < ${self.min_spend_for_scale})",
                    "evidence": {
                        "roas": float(roas),
                        "spend": float(spend),
                        "current_roas": float(roas),
                        "current_spend": float(spend),
                        "roas_threshold": self.high_roas_low_spend_threshold,
                        "spend_threshold": self.min_spend_for_scale,
                        "threshold_used": f"ROAS >= {self.high_roas_low_spend_threshold}, spend < {self.min_spend_for_scale}",
                        "comparison": f"ROAS {roas:.2f} >= {self.high_roas_low_spend_threshold} (meets threshold), spend ${spend:.2f} < ${self.min_spend_for_scale} (below scaling threshold by ${self.min_spend_for_scale - spend:.2f})",
                        "scaling_potential": "high",
                        "note": "High-performing adset with low spend. Scale gradually and monitor ROAS as it may decrease with increased budget.",
                    },
                }
            )

        # 7. TOO NARROW AGE RANGE (Limiting reach unnecessarily)
        if age_max > 0 and age_min > 0:
            age_range = age_max - age_min
            if age_range < self.narrow_age_range and age_range > 0:
                confidence = "MEDIUM"

                issues.append(
                    {
                        "adset_id": adset.get("adset_id"),
                        "issue_type": "age_range_too_narrow",
                        "description": f"Age range {age_min}-{age_max} ({age_range} years) may be too narrow - limiting reach",
                        "opportunity": "Expanding age range could increase reach and scale",
                        "priority": "LOW",
                        "confidence": confidence,
                        "current_spend": spend,
                        "current_roas": roas,
                        "suggested_action": f"Test expanding to {max(self.age_min_bound, age_min-self.narrow_range_expansion)}-{min(self.age_max_bound, age_max+self.narrow_range_expansion)} (current: {age_min}-{age_max} = {age_range} years, threshold: <{self.narrow_age_range} years may limit reach)",
                        "evidence": {
                            "age_min": age_min,
                            "age_max": age_max,
                            "age_range": age_range,
                            "current_value": age_range,
                            "threshold_value": self.narrow_age_range,
                            "threshold_used": f"age_range < {self.narrow_age_range}",
                            "comparison": f"{age_range} < {self.narrow_age_range} (below threshold by {self.narrow_age_range - age_range} years)",
                            "calculation": f"age_range = {age_max} - {age_min} = {age_range}",
                            "note": "Narrow age range may limit reach unnecessarily. Test expanding gradually to find optimal range.",
                        },
                    }
                )

        # 8. LOW CTR (Poor creative/audience match)
        impressions = adset.get("impressions", 0)
        clicks = adset.get("clicks", 0)
        if (
            pd.notna(impressions)
            and impressions > self.min_impressions_for_ctr
            and pd.notna(clicks)
        ):
            ctr = clicks / impressions if impressions > 0 else 0
            if ctr < self.low_ctr_threshold and ctr > 0:
                confidence = "MEDIUM"

                issues.append(
                    {
                        "adset_id": adset.get("adset_id"),
                        "issue_type": "low_ctr",
                        "description": f"CTR {ctr:.2%} is very low - creative may not resonate with audience",
                        "opportunity": "Improving creative or audience targeting could increase engagement",
                        "priority": "MEDIUM",
                        "confidence": confidence,
                        "current_spend": spend,
                        "current_roas": roas,
                        "suggested_action": f"Test new creative or refine audience targeting (current CTR: {ctr:.2%}, threshold: <{self.low_ctr_threshold:.2%})",
                        "evidence": {
                            "ctr": float(ctr),
                            "clicks": int(clicks),
                            "impressions": int(impressions),
                            "current_value": float(ctr),
                            "threshold_value": self.low_ctr_threshold,
                            "threshold_used": f"CTR < {self.low_ctr_threshold}",
                            "comparison": f"{ctr:.2%} < {self.low_ctr_threshold:.2%} (below by {self.low_ctr_threshold - ctr:.2%})",
                            "calculation": f"CTR = {clicks} / {impressions} = {ctr:.4f}",
                            "note": "Low CTR suggests creative may not resonate with audience. Test new creative variations or refine audience targeting.",
                        },
                    }
                )

        # 9. HIGH CPC (Expensive clicks)
        if pd.notna(clicks) and clicks > 0 and pd.notna(spend):
            cpc = spend / clicks
            if cpc > self.high_cpc_threshold:
                confidence = "MEDIUM"

                issues.append(
                    {
                        "adset_id": adset.get("adset_id"),
                        "issue_type": "high_cpc",
                        "description": f"CPC ${cpc:.2f} is high - clicks are expensive",
                        "opportunity": "Optimizing targeting or creative could reduce cost per click",
                        "priority": "MEDIUM",
                        "confidence": confidence,
                        "current_spend": spend,
                        "current_roas": roas,
                        "suggested_action": f"Optimize targeting to reach more relevant audience or test lower-cost ad formats (current CPC: ${cpc:.2f}, threshold: >${self.high_cpc_threshold:.2f})",
                        "evidence": {
                            "cpc": float(cpc),
                            "clicks": int(clicks),
                            "spend": float(spend),
                            "current_value": float(cpc),
                            "threshold_value": self.high_cpc_threshold,
                            "threshold_used": f"CPC > {self.high_cpc_threshold}",
                            "comparison": f"${cpc:.2f} > ${self.high_cpc_threshold:.2f} (exceeds by ${cpc - self.high_cpc_threshold:.2f})",
                            "calculation": f"CPC = ${spend:.2f} / {clicks} clicks = ${cpc:.2f}",
                            "note": "High CPC indicates expensive clicks. Optimize targeting to reach more relevant audience or test lower-cost ad formats.",
                        },
                    }
                )

        # 10. LOW CONVERSION RATE (Poor quality traffic)
        conversions = adset.get("conversions", adset.get("purchases", 0))
        if (
            pd.notna(clicks)
            and clicks > self.min_clicks_for_conversion
            and pd.notna(conversions)
        ):
            conversion_rate = conversions / clicks if clicks > 0 else 0
            if (
                conversion_rate < self.low_conversion_rate_threshold
                and conversion_rate > 0
            ):
                confidence = "MEDIUM"

                issues.append(
                    {
                        "adset_id": adset.get("adset_id"),
                        "issue_type": "low_conversion_rate",
                        "description": f"Conversion rate {conversion_rate:.2%} is low - traffic quality may be poor",
                        "opportunity": "Improving audience targeting or landing page could increase conversions",
                        "priority": "MEDIUM",
                        "confidence": confidence,
                        "current_spend": spend,
                        "current_roas": roas,
                        "suggested_action": f"Refine audience targeting or optimize landing page experience (current conversion rate: {conversion_rate:.2%}, threshold: <{self.low_conversion_rate_threshold:.2%})",
                        "evidence": {
                            "conversion_rate": float(conversion_rate),
                            "conversions": int(conversions),
                            "clicks": int(clicks),
                            "current_value": float(conversion_rate),
                            "threshold_value": self.low_conversion_rate_threshold,
                            "threshold_used": f"conversion_rate < {self.low_conversion_rate_threshold}",
                            "comparison": f"{conversion_rate:.2%} < {self.low_conversion_rate_threshold:.2%} (below by {self.low_conversion_rate_threshold - conversion_rate:.2%})",
                            "calculation": f"conversion_rate = {conversions} / {clicks} = {conversion_rate:.4f}",
                            "note": "Low conversion rate suggests poor traffic quality. Refine audience targeting or optimize landing page experience.",
                        },
                    }
                )

        # 11. LOW REACH (Very small audience)
        reach = adset.get("reach", 0)
        if pd.notna(reach) and reach > 0 and reach < self.low_reach_threshold:
            confidence = "MEDIUM"

            issues.append(
                {
                    "adset_id": adset.get("adset_id"),
                    "issue_type": "low_reach",
                    "description": f"Reach {int(reach):,} is very small - limiting scale potential",
                    "opportunity": "Expanding targeting could increase reach and scale",
                    "priority": "LOW",
                    "confidence": confidence,
                    "current_spend": spend,
                    "current_roas": roas,
                    "suggested_action": f"Consider broadening targeting or testing lookalike audiences (current: {int(reach):,} people, threshold: <{self.low_reach_threshold:,} is too small)",
                    "evidence": {
                        "reach": int(reach),
                        "current_value": int(reach),
                        "threshold_value": self.low_reach_threshold,
                        "threshold_used": f"reach < {self.low_reach_threshold}",
                        "comparison": f"{int(reach):,} < {self.low_reach_threshold:,} (below by {self.low_reach_threshold - int(reach):,} people)",
                        "note": "Very small reach limits scale potential. Consider broadening targeting or testing lookalike audiences to expand reach.",
                    },
                }
            )

        # 12. HIGH CPM (Expensive impressions)
        if pd.notna(impressions) and impressions > 0 and pd.notna(spend):
            cpm = (spend / impressions) * 1000 if impressions > 0 else 0
            if cpm > self.high_cpm_threshold:
                confidence = "MEDIUM"

                issues.append(
                    {
                        "adset_id": adset.get("adset_id"),
                        "issue_type": "high_cpm",
                        "description": f"CPM ${cpm:.2f} is high - impressions are expensive",
                        "opportunity": "Optimizing targeting or ad placement could reduce cost per impression",
                        "priority": "MEDIUM",
                        "confidence": confidence,
                        "current_spend": spend,
                        "current_roas": roas,
                        "suggested_action": f"Refine targeting or test different ad placements (current CPM: ${cpm:.2f}, threshold: >${self.high_cpm_threshold:.2f})",
                        "evidence": {
                            "cpm": float(cpm),
                            "impressions": int(impressions),
                            "spend": float(spend),
                            "current_value": float(cpm),
                            "threshold_value": self.high_cpm_threshold,
                            "threshold_used": f"CPM > {self.high_cpm_threshold}",
                            "comparison": f"${cpm:.2f} > ${self.high_cpm_threshold:.2f} (exceeds by ${cpm - self.high_cpm_threshold:.2f})",
                            "calculation": f"CPM = (${spend:.2f} / {impressions}) * 1000 = ${cpm:.2f}",
                            "note": "High CPM indicates expensive impressions. Refine targeting to reach more relevant audience or test different ad placements.",
                        },
                    }
                )

        # 13. LOW VIDEO ENGAGEMENT (Poor video performance)
        video_30_sec = adset.get("video_30_sec_watched_actions", 0)
        video_p100 = adset.get("video_p100_watched_actions", 0)

        # Convert to numeric if needed
        try:
            video_30_sec = float(video_30_sec) if pd.notna(video_30_sec) else 0
        except (ValueError, TypeError):
            video_30_sec = 0
        try:
            video_p100 = float(video_p100) if pd.notna(video_p100) else 0
        except (ValueError, TypeError):
            video_p100 = 0

        if pd.notna(impressions) and impressions > self.min_impressions_for_video:
            # Check if this is a video ad (has video engagement metrics)
            if (video_30_sec > 0) or (video_p100 > 0):
                video_views = max(
                    float(video_30_sec) if pd.notna(video_30_sec) else 0,
                    float(video_p100) if pd.notna(video_p100) else 0,
                )
                video_completion_rate = (
                    video_views / impressions if impressions > 0 else 0
                )

                if video_completion_rate < self.low_video_engagement_threshold:
                    confidence = "MEDIUM"

                    issues.append(
                        {
                            "adset_id": adset.get("adset_id"),
                            "issue_type": "low_video_engagement",
                            "description": f"Video completion rate {video_completion_rate:.1%} is low - video may not be engaging",
                            "opportunity": "Creating more engaging video content could improve performance",
                            "priority": "MEDIUM",
                            "confidence": confidence,
                            "current_spend": spend,
                            "current_roas": roas,
                            "suggested_action": f"Test shorter videos, better hooks, or different video creative (current completion rate: {video_completion_rate:.1%}, threshold: <{self.low_video_engagement_threshold:.1%})",
                            "evidence": {
                                "video_completion_rate": float(video_completion_rate),
                                "video_views": int(video_views),
                                "impressions": int(impressions),
                                "current_value": float(video_completion_rate),
                                "threshold_value": self.low_video_engagement_threshold,
                                "threshold_used": f"video_completion_rate < {self.low_video_engagement_threshold}",
                                "comparison": f"{video_completion_rate:.1%} < {self.low_video_engagement_threshold:.1%} (below by {self.low_video_engagement_threshold - video_completion_rate:.1%})",
                                "calculation": f"completion_rate = {video_views} / {impressions} = {video_completion_rate:.4f}",
                                "note": "Low video completion rate suggests video may not be engaging. Test shorter videos, better hooks, or different video creative.",
                            },
                        }
                    )

        # 14. NARROW GENDER TARGETING (Limiting reach)
        genders = adset.get("adset_targeting_genders", None)
        if genders is not None and not pd.isna(genders):
            try:
                import ast

                if isinstance(genders, str):
                    if genders.startswith("["):
                        gender_list = ast.literal_eval(genders)
                    else:
                        gender_list = [g.strip() for g in genders.split(",")]
                else:
                    gender_list = genders if isinstance(genders, list) else [genders]

                # Check if only targeting one gender (1=Male, 2=Female)
                if len(gender_list) == 1:
                    gender_str = str(gender_list[0]).strip()
                    if gender_str in ["1", "2", "male", "female", "Male", "Female"]:
                        confidence = (
                            "LOW"  # Low confidence - single gender might be intentional
                        )

                        issues.append(
                            {
                                "adset_id": adset.get("adset_id"),
                                "issue_type": "narrow_gender_targeting",
                                "description": f"Targeting only one gender - may be limiting reach unnecessarily",
                                "opportunity": "Testing both genders could expand reach and scale",
                                "priority": "LOW",
                                "confidence": confidence,
                                "current_spend": spend,
                                "current_roas": roas,
                                "suggested_action": f"Test expanding to both genders if product is gender-neutral (current: targeting {len(gender_list)} gender(s): {', '.join(str(g) for g in gender_list)})",
                                "evidence": {
                                    "targeted_genders": gender_list,
                                    "current_value": len(gender_list),
                                    "threshold_value": self.total_genders_count,
                                    "threshold_used": "single gender targeting",
                                    "comparison": f"Targeting {len(gender_list)} gender(s) (both genders = {self.total_genders_count}, current = {len(gender_list)})",
                                    "note": "Single gender targeting may limit reach unnecessarily. Test both genders to expand reach and scale, unless gender-specific targeting is intentional.",
                                },
                            }
                        )
            except (ValueError, SyntaxError):
                pass  # Skip if can't parse genders

        # 15. MISSING VIDEO FORMAT (Image ads when video could work better)
        # Check if this is an image ad (no video metrics) but performance suggests video might help
        video_30_sec = adset.get("video_30_sec_watched_actions", 0)
        video_p100 = adset.get("video_p100_watched_actions", 0)

        # Convert to numeric if needed
        try:
            video_30_sec = float(video_30_sec) if pd.notna(video_30_sec) else 0
        except (ValueError, TypeError):
            video_30_sec = 0
        try:
            video_p100 = float(video_p100) if pd.notna(video_p100) else 0
        except (ValueError, TypeError):
            video_p100 = 0

        has_video_metrics = (video_30_sec > 0) or (video_p100 > 0)

        # If no video metrics (likely image ad) but good ROAS, video might work better
        if (
            not has_video_metrics
            and pd.notna(roas)
            and roas >= self.video_preferred_roas_threshold
            and pd.notna(spend)
            and spend > self.min_spend_for_video_format_check
        ):
            # Check if this is a warm audience (lookalike or retargeting) where video typically performs well
            lal_count = adset.get("adset_targeting_custom_audiences_count", 0)
            if pd.isna(lal_count):
                lal_count = 0

            is_warm_audience = (
                lal_count > 0 or roas >= self.warm_audience_roas_threshold
            )  # Lookalike or high ROAS = warm

            if is_warm_audience:
                confidence = "MEDIUM"

                issues.append(
                    {
                        "adset_id": adset.get("adset_id"),
                        "issue_type": "missing_video_format",
                        "description": f"Using image format but ROAS {roas:.2f} suggests video might perform better",
                        "opportunity": "Video typically performs better for warm audiences (lookalike/retargeting)",
                        "priority": "MEDIUM",
                        "confidence": confidence,
                        "current_spend": spend,
                        "current_roas": roas,
                        "suggested_action": f"Test video creative for this audience (current: Image format, ROAS {roas:.2f} >= {self.video_preferred_roas_threshold}, warm audience: {is_warm_audience})",
                        "evidence": {
                            "current_format": "Image",
                            "roas": float(roas),
                            "is_warm_audience": is_warm_audience,
                            "current_roas": float(roas),
                            "roas_threshold": self.video_preferred_roas_threshold,
                            "threshold_used": f"ROAS >= {self.video_preferred_roas_threshold}, no video metrics",
                            "comparison": f"ROAS {roas:.2f} >= {self.video_preferred_roas_threshold} (meets threshold), format: Image (no video metrics detected)",
                            "note": "Video typically performs better for warm audiences (lookalike/retargeting). Test video creative to potentially improve performance.",
                        },
                    }
                )

        # 16. WRONG FORMAT FOR AUDIENCE (Format mismatch)
        # Video for cold audiences when image might be better, or vice versa
        if has_video_metrics:
            # This is a video ad
            if (
                pd.notna(roas)
                and roas < self.low_roas_threshold
                and pd.notna(clicks)
                and clicks > self.min_clicks_for_format_check
            ):
                # Video with low ROAS and low engagement - might be wrong for cold audience
                ctr = (
                    clicks / impressions
                    if pd.notna(impressions) and impressions > 0
                    else 0
                )
                if ctr < self.image_preferred_ctr_threshold:
                    confidence = "LOW"  # Low confidence - format preference varies

                    issues.append(
                        {
                            "adset_id": adset.get("adset_id"),
                            "issue_type": "wrong_format_for_audience",
                            "description": f"Video format with low CTR {ctr:.2%} and ROAS {roas:.2f} - image might work better for cold audience",
                            "opportunity": "Image ads often perform better for cold/prospecting audiences",
                            "priority": "LOW",
                            "confidence": confidence,
                            "current_spend": spend,
                            "current_roas": roas,
                            "suggested_action": f"Test image format for this cold audience (current: Video format, CTR {ctr:.2%} < {self.image_preferred_ctr_threshold:.2%}, ROAS {roas:.2f} < {self.low_roas_threshold})",
                            "evidence": {
                                "current_format": "Video",
                                "ctr": float(ctr),
                                "roas": float(roas),
                                "current_ctr": float(ctr),
                                "current_roas": float(roas),
                                "ctr_threshold": self.image_preferred_ctr_threshold,
                                "roas_threshold": self.low_roas_threshold,
                                "threshold_used": f"Video with CTR < {self.image_preferred_ctr_threshold} and ROAS < {self.low_roas_threshold}",
                                "comparison": f"CTR {ctr:.2%} < {self.image_preferred_ctr_threshold:.2%} (below threshold), ROAS {roas:.2f} < {self.low_roas_threshold} (below break-even)",
                                "note": "Video format may not be optimal for cold/prospecting audiences. Test image ads which often perform better for cold audiences.",
                            },
                        }
                    )

        # 17. AGE TARGETING MISMATCH WITH BUYERS (Shopify data)
        # If targeting starts at 18 but buyers are mostly 35+, suggest narrowing
        if (
            self.shopify_df is not None
            and len(self.shopify_df) > 0
            and age_min > 0
            and age_min < self.min_buyer_age_threshold
        ):
            # Targeting is too young compared to actual buyers
            confidence = "HIGH"  # High confidence - based on actual buyer data

            issues.append(
                {
                    "adset_id": adset.get("adset_id"),
                    "issue_type": "age_targeting_mismatch_buyers",
                    "description": f"Targeting starts at {age_min} but Shopify buyers are {self.min_buyer_age_threshold}+ - wasting budget on non-buyers",
                    "opportunity": "Narrowing to buyer age range could improve efficiency and ROAS",
                    "priority": "HIGH",
                    "confidence": confidence,
                    "current_spend": spend,
                    "current_roas": roas,
                    "suggested_action": f"Raise age_min to {self.min_buyer_age_threshold}+ to match actual buyer demographics (current: {age_min}, buyers: {self.min_buyer_age_threshold}+)",
                    "evidence": {
                        "age_min": age_min,
                        "buyer_age_threshold": self.min_buyer_age_threshold,
                        "current_value": age_min,
                        "threshold_value": self.min_buyer_age_threshold,
                        "threshold_used": f"age_min < {self.min_buyer_age_threshold}",
                        "comparison": f"age_min {age_min} < {self.min_buyer_age_threshold} (targeting too young)",
                        "shopify_orders_analyzed": len(self.shopify_df),
                        "note": f"Shopify data shows buyers are {self.min_buyer_age_threshold}+ years old. Targeting younger ages wastes budget on non-converters.",
                    },
                }
            )

        # 18. GEO TARGETING MISMATCH WITH BUYERS (Shopify data)
        # If targeting national but buyers are concentrated in top states, suggest geo focus
        if (
            self.shopify_df is not None
            and len(self.shopify_df) > 0
            and self.shopify_geo_concentration
            >= self.buyer_state_concentration_threshold
            and countries is not None
            and not pd.isna(countries)
        ):
            # Buyers are geo-concentrated but targeting is national/broad
            try:
                import ast

                if isinstance(countries, str):
                    if countries.startswith("["):
                        country_list = ast.literal_eval(countries)
                    else:
                        country_list = [c.strip() for c in countries.split(",")]
                else:
                    country_list = (
                        countries if isinstance(countries, list) else [countries]
                    )

                # Check if targeting is broad (national US, no state-specific targeting)
                is_national_targeting = (
                    any("US" in str(c).upper() for c in country_list)
                    and len(country_list) <= 2
                )

                if is_national_targeting:
                    confidence = "HIGH"  # High confidence - based on actual buyer data

                    top_states_list = list(self.shopify_buyer_states)[:5]
                    states_str = ", ".join(top_states_list)

                    issues.append(
                        {
                            "adset_id": adset.get("adset_id"),
                            "issue_type": "geo_targeting_mismatch_buyers",
                            "description": f"Targeting national but {self.shopify_geo_concentration:.1%} of Shopify buyers from {len(top_states_list)} states",
                            "opportunity": "Geo-focused campaigns in top buyer states could improve ROAS",
                            "priority": "MEDIUM",
                            "confidence": confidence,
                            "current_spend": spend,
                            "current_roas": roas,
                            "suggested_action": f"Test separate campaigns for top states: {states_str} (current: National, buyer concentration: {self.shopify_geo_concentration:.1%})",
                            "evidence": {
                                "current_targeting": country_list,
                                "top_buyer_states": top_states_list,
                                "buyer_concentration": float(
                                    self.shopify_geo_concentration
                                ),
                                "current_value": "National",
                                "threshold_value": f"Top {len(top_states_list)} states",
                                "threshold_used": f"buyer_concentration >= {self.buyer_state_concentration_threshold}",
                                "comparison": f"{self.shopify_geo_concentration:.1%} of buyers from {len(top_states_list)} states (geo-focused)",
                                "shopify_orders_analyzed": len(self.shopify_df),
                                "note": f"Shopify data shows {self.shopify_geo_concentration:.1%} of orders from {len(top_states_list)} states. Test geo-focused campaigns for these high-performing states.",
                            },
                        }
                    )
            except (ValueError, SyntaxError):
                pass  # Skip if can't parse countries

        return issues
