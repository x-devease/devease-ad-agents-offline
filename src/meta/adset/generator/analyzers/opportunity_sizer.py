"""
Calculate opportunity size for adset scaling.

Estimates opportunity size based on frequency, ROAS, and budget.
Helps prioritize which adsets have the most scaling potential.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from src.utils import Config


class OpportunitySizer:
    """
    Calculate opportunity size for adset budget scaling.

    Opportunity size = qualitative assessment of scaling potential based on:
    - Frequency: How saturated the audience is
    - ROAS: Performance level (higher = more opportunity)
    - Budget: Spend level (higher = more opportunity)
    """

    def __init__(self):
        # Frequency thresholds
        self.target_frequency = 2.5  # Optimal frequency (2-3 impressions/person)
        self.max_frequency = 4.0  # Diminishing returns after this
        self.saturation_frequency = 6.0  # Complete saturation

        # Default audience size (if not provided)
        self.default_audience_size = 100000  # 100K people

        # Cost per impression/view assumptions
        self.default_cpm = 20  # $20 CPM

    def calculate_headroom(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate headroom for all adsets.

        Args:
            df: Adset-level data with columns:
                - adset_id
                - spend
                - impressions
                - frequency
                - reach (optional, for audience size estimation)

        Returns:
            DataFrame with headroom calculations:
                - adset_id
                - safe_headroom: Budget up to target frequency
                - max_headroom: Budget up to max frequency
                - current_spend
                - current_frequency
                - estimated_audience_size
                - max_spend
                - saturation_status
        """
        results = []

        for _, row in df.iterrows():
            headroom = self.calculate_adset_headroom(row)
            results.append(headroom)

        return pd.DataFrame(results)

    def calculate_adset_headroom(self, adset: pd.Series) -> dict:
        """
        Calculate headroom for a single adset.

        Args:
            adset: Row from adset dataframe

        Returns:
            Dictionary with headroom metrics
        """
        # Extract metrics
        adset_id = adset.get("adset_id", "")
        spend = adset.get("spend", 0)
        impressions = adset.get("impressions", 0)
        frequency = adset.get("frequency", 0)
        reach = adset.get("reach", 0)

        # Calculate audience size if not provided
        if reach > 0:
            audience_size = reach
        elif impressions > 0 and frequency > 0:
            audience_size = impressions / frequency
        else:
            # Estimate from spend and CPM
            estimated_impressions = (spend / self.default_cpm) * 1000
            audience_size = estimated_impressions / 2  # Assume avg 2 impressions

        # Recalculate frequency if missing
        if frequency <= 0 and impressions > 0 and audience_size > 0:
            frequency = impressions / audience_size

        # Calculate cost per person
        if audience_size > 0:
            cost_per_person = spend / audience_size
        else:
            cost_per_person = self.default_cpm / 1000  # CPM to cost per impression

        # Determine saturation status
        if frequency >= self.saturation_frequency:
            saturation_status = "saturated"
            safe_headroom = 0
            max_headroom = 0
        elif frequency >= self.max_frequency:
            saturation_status = "diminishing_returns"
            # Can still increase up to saturation, but not recommended
            safe_headroom = 0
            max_headroom = (
                audience_size
                * (self.saturation_frequency - frequency)
                * cost_per_person
            )
        elif frequency >= self.target_frequency:
            saturation_status = "optimal"
            # Past optimal, approaching diminishing returns
            safe_headroom = (
                audience_size * (self.max_frequency - frequency) * cost_per_person
            )
            max_headroom = (
                audience_size
                * (self.saturation_frequency - frequency)
                * cost_per_person
            )
        else:
            saturation_status = "room_to_grow"
            # Below optimal, plenty of room
            safe_headroom = (
                audience_size * (self.target_frequency - frequency) * cost_per_person
            )
            max_headroom = (
                audience_size * (self.max_frequency - frequency) * cost_per_person
            )

        # Calculate max spend (current + max headroom)
        max_spend = spend + max_headroom

        # Ensure non-negative headroom
        safe_headroom = max(0, safe_headroom)
        max_headroom = max(0, max_headroom)
        max_spend = max(spend, max_spend)

        return {
            "adset_id": adset_id,
            "safe_headroom": safe_headroom,
            "max_headroom": max_headroom,
            "current_spend": spend,
            "current_frequency": frequency,
            "estimated_audience_size": audience_size,
            "max_spend": max_spend,
            "saturation_status": saturation_status,
            "cost_per_person": cost_per_person,
        }

    def prioritize_by_headroom(
        self, df: pd.DataFrame, headroom_df: pd.DataFrame, min_roas: float = 1.5
    ) -> pd.DataFrame:
        """
        Prioritize adsets by headroom opportunity.

        Args:
            df: Original adset data
            headroom_df: Headroom calculations
            min_roas: Minimum ROAS to consider for scaling

        Returns:
            DataFrame ranked by opportunity score
        """
        # Merge adset data with headroom
        merged = df.merge(headroom_df, on="adset_id", how="left")

        # Filter to adsets with ROAS above threshold
        roas_col = "purchase_roas" if "purchase_roas" in merged.columns else "roas"
        merged = merged[merged[roas_col] >= min_roas]

        # Calculate opportunity score
        merged["opportunity_score"] = (
            merged[roas_col]
            * merged["safe_headroom"]
            * (merged[roas_col] - 1)  # Profit margin
        )

        # Sort by opportunity
        merged = merged.sort_values("opportunity_score", ascending=False)

        return merged

    def get_scaling_recommendation(self, adset: pd.Series, headroom: dict) -> str:
        """
        Get human-readable scaling recommendation.

        Args:
            adset: Adset data
            headroom: Headroom calculation

        Returns:
            Recommendation string
        """
        current_spend = headroom["current_spend"]
        safe_headroom = headroom["safe_headroom"]
        status = headroom["saturation_status"]

        if status == "saturated":
            return f"Saturated at ${current_spend:.0f}, no scaling recommended"
        elif status == "diminishing_returns":
            return f"Approaching saturation, maintain current spend of ${current_spend:.0f}"
        elif safe_headroom < current_spend * 0.5:
            return f"Limited headroom (${safe_headroom:.0f}), maintain ${current_spend:.0f}"
        elif safe_headroom >= current_spend * 5:
            # Can scale 5x or more
            recommended_spend = min(current_spend * 5, current_spend + safe_headroom)
            return f"Strong headroom, scale from ${current_spend:.0f} to ${recommended_spend:.0f}"
        else:
            # Moderate headroom
            recommended_spend = current_spend + safe_headroom
            return f"Moderate headroom, scale from ${current_spend:.0f} to ${recommended_spend:.0f}"
