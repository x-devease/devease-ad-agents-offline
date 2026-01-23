"""
Segment recommendations by geography, audience type, and creative format.

Generates actionable insights for each segment.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict

from src.utils import Config


class Segmenter:
    """
    Segment and analyze recommendations by multiple dimensions.

    Dimensions:
    1. Geography: Country/region from adset targeting
    2. Audience Type: Interest, Lookalike, Broad (inferred from targeting)
    3. Creative Format: Video, Image, UGC (inferred from engagement metrics)
    """

    # Top 10 states by revenue from Shopify analysis (60% of revenue)
    # Used to enrich audience types with geo focus signal
    TOP_STATES = {
        "Texas",
        "California",
        "Florida",
        "New York",
        "Georgia",
        "Ohio",
        "North Carolina",
        "Pennsylvania",
        "Washington",
        "Illinois",
    }

    def __init__(self, customer: str = "moprobo", platform: str = "meta"):
        """
        Initialize Segmenter with parameters from config.

        Args:
            customer: Customer name (default: "moprobo")
            platform: Platform name (default: "meta")
        """
        # Load parameters from config
        params = Config.get_customer_params(customer, platform)

        # Age targeting thresholds
        age_params = params.get("age_targeting", {})
        self.broad_classification_threshold = age_params.get(
            "broad_classification_threshold", 45
        )

        # Segment health thresholds
        health_params = params.get("segment_health", {})
        self.critical_ratio_threshold = health_params.get(
            "critical_ratio_threshold", 0.3
        )
        self.high_ratio_threshold = health_params.get("high_ratio_threshold", 0.3)
        self.medium_ratio_threshold = health_params.get("medium_ratio_threshold", 0.5)

        self.segment_columns = {
            "geography": "country",
            "audience_type": "audience_type",
            "creative_format": "creative_format",
        }

    def parse_audience_type(self, row: pd.Series) -> str:
        """
        Infer audience type from targeting configuration.

        Enriched with Shopify buyer data:
        - Adds "_Geo" suffix for campaigns targeting top 10 states
        - Base types: Exclusion, Lookalike, Advantage, Broad, Interest

        Priority Order (Most Specific → Least Specific):
        1. Exclusion - Has excluded custom audiences (highest specificity)
        2. Lookalike - Has custom audiences (LAL targeting)
        3. Advantage_Geo - Advantage+ targeting top states
        4. Advantage - Advantage+ national targeting
        5. Broad_Geo - Broad targeting top states
        6. Broad - Broad national targeting
        7. Interest - Fallback for interest-based targeting
        """
        # Get excluded count (highest priority)
        excluded_count = (
            row.get("adset_targeting_excluded_custom_audiences_count", 0) or 0
        )

        # Get custom count
        custom_count = row.get("adset_targeting_custom_audiences_count", 0)

        # Handle NaN
        if pd.isna(custom_count):
            custom_count = 0

        # Get Advantage+ flag
        advantage = row.get("adset_targeting_advantage_audience", False)
        # Handle NaN for advantage flag
        if pd.isna(advantage):
            advantage = False
        advantage = advantage or False

        # Get age range
        age_min = row.get("adset_targeting_age_min", 0) or 0
        age_max = row.get("adset_targeting_age_max", 0) or 0
        age_range = age_max - age_min if age_max > 0 else 0

        # Priority 1: Exclusion (most specific targeting constraint)
        if excluded_count > 0:
            return "Exclusion"

        # Priority 2: Lookalike (custom audiences = LAL targeting)
        if custom_count > 0:
            return "Lookalike"

        # Check if geo-focused (targets top 10 states)
        is_geo_focused = self._is_geo_focused(row)

        # Priority 3: Advantage+ (auto-targeting mode)
        if advantage:
            if is_geo_focused:
                return "Advantage_Geo"
            return "Advantage"

        # Priority 4: Broad (default 18-65 age range)
        if age_range >= self.broad_classification_threshold:
            if is_geo_focused:
                return "Broad_Geo"
            return "Broad"

        # Priority 5: Interest (fallback)
        return "Interest"

    def _is_geo_focused(self, row: pd.Series) -> bool:
        """
        Check if campaign targets top 10 states by Shopify revenue.

        Returns True if targeting configuration includes top states.
        """
        import re

        # Get countries field
        countries = row.get("adset_targeting_countries", None)

        if pd.isna(countries) or countries == "":
            return False

        # Convert to string for checking
        countries_str = str(countries).upper()

        # Check for top state names or abbreviations using word boundaries
        # Must check longer state names first to avoid partial matches
        state_keywords = [
            # Full names (check first to avoid "US" matching "UNITED STATES")
            "NORTH CAROLINA",
            "NEW YORK",
            "CALIFORNIA",
            "FLORIDA",
            "TEXAS",
            "GEORGIA",
            "PENNSYLVANIA",
            "WASHINGTON",
            "ILLINOIS",
            "OHIO",
            # Abbreviations (exclude "US" to avoid matching country code)
            "TX",
            "CA",
            "FL",
            "NY",
            "GA",
            "OH",
            "NC",
            "PA",
            "WA",
            "IL",
        ]

        for keyword in state_keywords:
            # Use word boundaries to avoid matching "VA" in "VERMONT" or "IL" in "ILLINOIS"
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, countries_str):
                return True

        return False

    def parse_creative_format(self, row: pd.Series) -> str:
        """
        Infer creative format from engagement metrics.

        Rules:
        - video_30_sec_watched_actions > 0 → Video
        - video_p100_watched_actions > 0 → Video
        - high video engagement rate → Video
        - otherwise → Image (default)
        """
        video_30 = row.get("video_30_sec_watched_actions", 0) or 0
        video_p100 = row.get("video_p100_watched_actions", 0) or 0

        # Convert to numeric if string
        try:
            video_30 = float(video_30) if video_30 else 0
        except (ValueError, TypeError):
            video_30 = 0

        try:
            video_p100 = float(video_p100) if video_p100 else 0
        except (ValueError, TypeError):
            video_p100 = 0

        # If there are video metrics, it's likely video format
        if video_30 > 0 or video_p100 > 0:
            return "Video"

        # Check for other indicators
        # For now, default to Image if no clear video signals
        return "Image"

    def parse_geography(self, row: pd.Series) -> str:
        """
        Parse geography from adset targeting countries.

        Handles JSON string format like "['US']" or "['US', 'CA']".
        """
        countries = row.get("adset_targeting_countries", None)

        # Handle missing values
        if pd.isna(countries) or countries == "":
            return "Unknown"

        # Handle string representation of list like "['US']"
        if isinstance(countries, str):
            try:
                # Use ast.literal_eval for safer evaluation
                import ast

                countries_list = ast.literal_eval(countries)
                if isinstance(countries_list, list) and len(countries_list) > 0:
                    return str(countries_list[0])  # Return first country
            except:
                pass

            # Fallback: try to extract country code from string
            if "US" in countries:
                return "US"
            elif "CA" in countries:
                return "CA"
            elif "GB" in countries:
                return "GB"
            elif "AU" in countries:
                return "AU"
            else:
                return countries[:20]  # Truncate long strings

        # Handle direct value
        if pd.notna(countries):
            return str(countries)

        return "Unknown"

    def segment_recommendations(
        self, recommendations: pd.DataFrame, adset_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add segment columns to recommendations.

        Args:
            recommendations: Output from MistakeDetector (one row per issue per adset)
            adset_data: Original adset features data (may have multiple rows per adset)

        Returns:
            Recommendations with segment columns added (one row per issue per adset)
        """
        # Get relevant columns for segmentation
        segment_cols = ["adset_id"] + [
            col
            for col in adset_data.columns
            if any(
                term in col.lower()
                for term in [
                    "countr",
                    "custom_audiences_count",
                    "age_min",
                    "age_max",
                    "video_30_sec",
                    "video_p100",
                ]
            )
        ]

        # Aggregate to one row per adset_id (take first value for each column)
        # This handles the case where adset_data has multiple rows per adset
        adset_agg = adset_data[segment_cols].groupby("adset_id").first().reset_index()

        # Merge with recommendations (one-to-one on adset_id)
        merged = recommendations.merge(adset_agg, on="adset_id", how="left")

        # Add segment columns
        merged["geography"] = merged.apply(self.parse_geography, axis=1)
        merged["audience_type"] = merged.apply(self.parse_audience_type, axis=1)
        merged["creative_format"] = merged.apply(self.parse_creative_format, axis=1)

        # Verify no duplicates
        duplicate_count = merged.duplicated(subset=["adset_id", "issue_type"]).sum()
        if duplicate_count > 0:
            print(
                f"  Warning: Found {duplicate_count} duplicate adset_id + issue_type combinations, removing..."
            )
            merged = merged.drop_duplicates(subset=["adset_id", "issue_type"])

        return merged

    def generate_segment_insights(
        self, segmented_recs: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Generate insights for each segment combination.

        Returns:
            Dictionary with segment insights
        """
        segments = segmented_recs.groupby(
            ["geography", "audience_type", "creative_format"]
        )

        insights = {}

        for (geo, audience, creative), segment_df in segments:
            segment_key = f"{geo}_{audience}_{creative}"

            # Calculate metrics
            total_issues = len(segment_df)
            critical_issues = (segment_df["priority"] == "CRITICAL").sum()
            high_issues = (segment_df["priority"] == "HIGH").sum()
            medium_issues = (segment_df["priority"] == "MEDIUM").sum()

            # opportunity_value removed - not reliable
            total_opportunity = 0
            total_spend = segment_df["current_spend"].sum()
            avg_roas = segment_df["current_roas"].mean()

            # Issue breakdown
            issue_types = segment_df["issue_type"].value_counts().to_dict()

            # Top issues by issue type (opportunity_value removed)
            top_issues = segment_df.head(5)[["issue_type", "suggested_action"]].to_dict(
                "records"
            )

            insights[segment_key] = {
                "geography": geo,
                "audience_type": audience,
                "creative_format": creative,
                "metrics": {
                    "total_adsets": total_issues,
                    "total_spend": float(total_spend),
                    "avg_roas": float(avg_roas),
                    "total_opportunity": float(total_opportunity),
                },
                "issues": {
                    "total": total_issues,
                    "critical": int(critical_issues),
                    "high": int(high_issues),
                    "medium": int(medium_issues),
                    "breakdown": issue_types,
                },
                "top_recommendations": top_issues,
            }

        return insights

    def rank_segments(self, insights: Dict[str, Dict]) -> List[Dict]:
        """
        Rank segments by opportunity and urgency.

        Returns:
            Sorted list of segments by priority
        """
        segments = []

        for segment_key, segment_data in insights.items():
            segments.append(
                {
                    "segment": segment_key,
                    "geography": segment_data["geography"],
                    "audience_type": segment_data["audience_type"],
                    "creative_format": segment_data["creative_format"],
                    "total_opportunity": segment_data["metrics"]["total_opportunity"],
                    "critical_issues": segment_data["issues"]["critical"],
                    "high_issues": segment_data["issues"]["high"],
                    "total_issues": segment_data["issues"]["total"],
                    "avg_roas": segment_data["metrics"]["avg_roas"],
                    "total_spend": segment_data["metrics"]["total_spend"],
                }
            )

        # Sort by total opportunity (descending), then by critical issues
        segments.sort(
            key=lambda x: (
                -x["total_opportunity"],
                -x["critical_issues"],
                -x["high_issues"],
            )
        )

        return segments

    def generate_segment_recommendations(self, insights: Dict[str, Dict]) -> List[Dict]:
        """
        Generate high-level recommendations for each segment.

        Returns:
            List of segment-level recommendations
        """
        segment_recs = []

        for segment_key, segment_data in insights.items():
            metrics = segment_data["metrics"]
            issues = segment_data["issues"]

            # Determine segment health
            if issues["critical"] > issues["total"] * self.critical_ratio_threshold:
                health = "CRITICAL"
                action = "Immediate cleanup required"
            elif issues["high"] > issues["total"] * self.high_ratio_threshold:
                health = "WARNING"
                action = "Optimize underperforming adsets"
            elif issues["medium"] > issues["total"] * self.medium_ratio_threshold:
                health = "NEEDS_ATTENTION"
                action = "Review targeting settings"
            else:
                health = "HEALTHY"
                action = "Maintain current strategy"

            # Identify top issue type
            top_issue = (
                max(issues["breakdown"].items(), key=lambda x: x[1])
                if issues["breakdown"]
                else ("none", 0)
            )

            segment_recs.append(
                {
                    "segment": segment_key,
                    "geography": segment_data["geography"],
                    "audience_type": segment_data["audience_type"],
                    "creative_format": segment_data["creative_format"],
                    "health_status": health,
                    "total_opportunity": metrics["total_opportunity"],
                    "avg_roas": metrics["avg_roas"],
                    "total_adsets": metrics["total_adsets"],
                    "top_issue_type": top_issue[0],
                    "top_issue_count": top_issue[1],
                    "recommended_action": action,
                }
            )

        return segment_recs

    def save_segment_report(
        self,
        insights: Dict[str, Dict],
        ranked_segments: List[Dict],
        segment_recs: List[Dict],
        output_path: str,
    ) -> None:
        """
        Save comprehensive segment report to JSON.

        Args:
            insights: Detailed segment insights
            ranked_segments: Ranked segments by opportunity
            segment_recs: Segment-level recommendations
            output_path: Path to save report
        """
        import json
        from pathlib import Path

        report = {
            "segments": insights,
            "ranked_segments": ranked_segments,
            "segment_recommendations": segment_recs,
            "summary": {
                "total_segments": len(insights),
                "top_segment": ranked_segments[0] if ranked_segments else None,
                "total_opportunity_all_segments": sum(
                    s["total_opportunity"] for s in ranked_segments
                ),
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    def print_segment_summary(
        self, ranked_segments: List[Dict], segment_recs: List[Dict], top_n: int = 10
    ) -> None:
        """
        Print formatted segment summary to console.

        Args:
            ranked_segments: Ranked segments by opportunity
            segment_recs: Segment-level recommendations
            top_n: Number of top segments to show
        """
        print("\n" + "=" * 80)
        print("SEGMENT RECOMMENDATIONS BY GEOGRAPHY, AUDIENCE TYPE, CREATIVE FORMAT")
        print("=" * 80)

        print(
            f"\n{'RANK':<5} {'SEGMENT':<40} {'OPPORTUNITY':>15} {'ISSUES':>10} {'STATUS':>15}"
        )
        print("-" * 80)

        for i, segment in enumerate(ranked_segments[:top_n], 1):
            seg_name = f"{segment['geography']} / {segment['audience_type']} / {segment['creative_format']}"
            opportunity = f"${segment['total_opportunity']:,.0f}"
            issues = f"{segment['critical_issues']}C/{segment['high_issues']}H"

            # Find matching recommendation
            rec = next(
                (r for r in segment_recs if r["segment"] == segment["segment"]),
                {"health_status": "UNKNOWN"},
            )

            print(
                f"{i:<5} {seg_name:<40} {opportunity:>15} {issues:>10} {rec['health_status']:<15}"
            )

        print("\n" + "=" * 80)
        print("TOP SEGMENT INSIGHTS")
        print("=" * 80)

        if ranked_segments:
            top = ranked_segments[0]
            print(
                f"\n🎯 TOP SEGMENT: {top['geography']} / {top['audience_type']} / {top['creative_format']}"
            )
            print(f"   Total Opportunity: ${top['total_opportunity']:,.2f}")
            print(f"   Total Issues: {top['total_issues']}")
            print(f"   Avg ROAS: {top['avg_roas']:.2f}")
            print(
                f"   Issues: {top['critical_issues']} Critical, {top['high_issues']} High"
            )
