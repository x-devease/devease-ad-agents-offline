"""
Day & Hour Performance Analyzer - Mixed Data Analysis (Score-Based).

Identifies low-performance time periods using:
- 24 hours hourly data for time-slot analysis
- 30 days daily data for day-of-week analysis

Outputs efficiency scores instead of monetary waste.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from src.meta.diagnoser.schemas.models import Issue, IssueSeverity, IssueCategory
from src.meta.diagnoser.core.issue_detector import BaseDetector


logger = logging.getLogger(__name__)


class DarkHoursDetector(BaseDetector):
    """
    Detects low-performance time periods using mixed data.

    Algorithm:
    1. Hourly Analysis (if 24h hourly data available):
       - Aggregate by hour (0-23)
       - Identify dead zones by hour
       - Calculate hourly efficiency score

    2. Day-of-Week Analysis (using 30d daily data):
       - Aggregate by day of week (0-6)
       - Identify weak days
       - Calculate weekly efficiency score

    3. Combined scoring:
       - Hourly score: 0-100
       - Weekly score: 0-100
       - Overall efficiency: weighted average

    Scoring:
    - 80-100: Excellent optimization
    - 60-79: Good - minor inefficiencies
    - 40-59: Moderate - significant inefficiencies
    - 20-39: Poor - major waste
    - 0-19: Critical - severe waste
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "target_roas": 2.5,
        "cvr_threshold_ratio": 0.2,  # 20% of average
        "min_spend_ratio_hourly": 0.05,  # 5% for hourly
        "min_spend_ratio_daily": 0.10,  # 10% for daily
        "min_days": 21,  # At least 3 weeks of data
    }

    # Day of week names
    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize day & hour performance analyzer."""
        super().__init__(config)
        self.thresholds = {
            **self.DEFAULT_THRESHOLDS,
            **self.config.get("thresholds", {}),
        }

    def detect(
        self,
        data: pd.DataFrame,
        entity_id: str,
        hourly_data: Optional[pd.DataFrame] = None,
    ) -> List[Issue]:
        """
        Detect weak time period issues.

        Args:
            data: AdSet-level daily data (30 days) with columns:
                - date_start, spend, purchase_roas
            entity_id: AdSet ID
            hourly_data: Optional hourly data (last 24h) with columns:
                - date_start, hour, spend, purchase_roas

        Returns:
            List of detected performance issues with efficiency scores
        """
        issues = []

        # Check for required columns in daily data
        required_cols = ["date_start", "spend", "purchase_roas"]
        if not all(col in data.columns for col in required_cols):
            logger.warning(f"Missing required columns for performance analysis: {entity_id}")
            return issues

        # Skip if insufficient daily data
        unique_days = data["date_start"].nunique()
        if unique_days < self.thresholds["min_days"]:
            logger.info(
                f"Insufficient data for analysis: "
                f"{unique_days} days < {self.thresholds['min_days']}"
            )
            return issues

        # Run day-of-week analysis (always available)
        weekly_result = self._analyze_weekly_performance(data, entity_id)

        # Run hourly analysis (if data available)
        hourly_result = None
        if hourly_data is not None and len(hourly_data) > 0:
            if all(col in hourly_data.columns for col in ["date_start", "hour", "spend", "purchase_roas"]):
                hourly_result = self._analyze_hourly_performance(hourly_data, entity_id)

        # Create issues based on available analysis
        if hourly_result and hourly_result["dead_zones"]:
            issues.append(self._create_hourly_issue(entity_id, hourly_result))

        if weekly_result["weak_days"]:
            issues.append(self._create_weekly_issue(entity_id, weekly_result))

        return issues

    def _analyze_hourly_performance(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze hourly performance and calculate efficiency score.

        Returns:
            Dict with dead_zones, efficiency_score, etc.
        """
        # Aggregate by hour (0-23)
        hourly_agg = data.groupby("hour").agg({
            "spend": "sum",
            "purchase_roas": "mean",
        }).reset_index()

        # Calculate total metrics
        total_spend = hourly_agg["spend"].sum()
        hourly_agg["spend_ratio"] = hourly_agg["spend"] / total_spend

        # Estimate conversions (if not provided)
        if "conversions" in data.columns:
            conversions_by_hour = data.groupby("hour")["conversions"].sum().reset_index()
            hourly_agg = hourly_agg.merge(conversions_by_hour, on="hour", how="left")
            hourly_agg["conversions"] = hourly_agg["conversions"].fillna(0)
        else:
            # Estimate conversions from ROAS
            avg_aov = 50
            hourly_agg["conversions"] = (
                (hourly_agg["purchase_roas"] * hourly_agg["spend"]) / avg_aov
            ).clip(lower=0)

        # Estimate clicks (if not provided)
        if "clicks" in data.columns:
            clicks_by_hour = data.groupby("hour")["clicks"].sum().reset_index()
            hourly_agg = hourly_agg.merge(clicks_by_hour, on="hour", how="left")
        else:
            avg_cpc = 1.0
            hourly_agg["clicks"] = (hourly_agg["spend"] / avg_cpc).clip(lower=1)

        hourly_agg["cvr"] = hourly_agg["conversions"] / hourly_agg["clicks"]

        # Calculate average CVR
        avg_cvr = hourly_agg[hourly_agg["cvr"] > 0]["cvr"].mean()

        # Identify dead zones
        cvr_threshold = avg_cvr * self.thresholds["cvr_threshold_ratio"]
        spend_ratio_threshold = self.thresholds["min_spend_ratio_hourly"]

        dead_zones = hourly_agg[
            (hourly_agg["cvr"] < cvr_threshold)
            & (hourly_agg["spend_ratio"] > spend_ratio_threshold)
        ]

        # Calculate efficiency score (0-100)
        efficiency_score = self._calculate_hourly_efficiency_score(hourly_agg, dead_zones)

        # Group consecutive dead zones into ranges
        dead_zone_ranges = self._group_consecutive_hours(
            dead_zones["hour"].tolist()
        )

        return {
            "dead_zones": dead_zone_ranges,
            "dead_zones_df": dead_zones,
            "hourly_agg": hourly_agg,
            "efficiency_score": efficiency_score,
            "avg_cvr": avg_cvr,
            "target_roas": self.thresholds["target_roas"],
            "total_spend": total_spend,
            "analysis_type": "hourly",
        }

    def _analyze_weekly_performance(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze day-of-week performance and calculate efficiency score.

        Returns:
            Dict with weak_days, efficiency_score, etc.
        """
        # Prepare data
        data = data.copy()
        data["date_start"] = pd.to_datetime(data["date_start"])
        data["day_of_week"] = data["date_start"].dt.dayofweek

        # Aggregate by day of week (0-6)
        dow_agg = data.groupby("day_of_week").agg({
            "spend": "sum",
            "purchase_roas": "mean",
        }).reset_index()

        # Calculate total metrics
        total_spend = dow_agg["spend"].sum()
        dow_agg["spend_ratio"] = dow_agg["spend"] / total_spend

        # Estimate conversions (if not provided)
        if "conversions" in data.columns:
            conversions_by_dow = data.groupby("day_of_week")["conversions"].sum().reset_index()
            dow_agg = dow_agg.merge(conversions_by_dow, on="day_of_week", how="left")
            dow_agg["conversions"] = dow_agg["conversions"].fillna(0)
        else:
            # Estimate conversions from ROAS
            avg_aov = 50
            dow_agg["conversions"] = (
                (dow_agg["purchase_roas"] * dow_agg["spend"]) / avg_aov
            ).clip(lower=0)

        # Estimate clicks (if not provided)
        if "clicks" in data.columns:
            clicks_by_dow = data.groupby("day_of_week")["clicks"].sum().reset_index()
            dow_agg = dow_agg.merge(clicks_by_dow, on="day_of_week", how="left")
        else:
            avg_cpc = 1.0
            dow_agg["clicks"] = (dow_agg["spend"] / avg_cpc).clip(lower=1)

        dow_agg["cvr"] = dow_agg["conversions"] / dow_agg["clicks"]

        # Calculate average CVR
        avg_cvr = dow_agg[dow_agg["cvr"] > 0]["cvr"].mean()

        # Identify weak days
        cvr_threshold = avg_cvr * self.thresholds["cvr_threshold_ratio"]
        spend_ratio_threshold = self.thresholds["min_spend_ratio_daily"]

        weak_days = dow_agg[
            (dow_agg["cvr"] < cvr_threshold)
            & (dow_agg["spend_ratio"] > spend_ratio_threshold)
        ]

        # Calculate efficiency score (0-100)
        efficiency_score = self._calculate_weekly_efficiency_score(dow_agg, weak_days)

        # Format weak day names
        weak_day_names = [
            self.DAY_NAMES[dow] for dow in weak_days["day_of_week"].tolist()
        ]

        return {
            "weak_days": weak_day_names,
            "weak_days_df": weak_days,
            "dow_agg": dow_agg,
            "efficiency_score": efficiency_score,
            "avg_cvr": avg_cvr,
            "target_roas": self.thresholds["target_roas"],
            "total_spend": total_spend,
            "analysis_days": data["date_start"].nunique(),
            "analysis_type": "weekly",
        }

    def _calculate_hourly_efficiency_score(
        self,
        hourly_agg: pd.DataFrame,
        dead_zones: pd.DataFrame,
    ) -> float:
        """
        Calculate hourly efficiency score (0-100).

        Components:
        - Dead zone penalty: 0-40 points
        - ROAS variance penalty: 0-30 points
        - Peak hours bonus: 0-30 points

        Returns:
            Efficiency score (0-100, higher = better)
        """
        score = 100

        # Dead zone penalty (0-40 points)
        if len(dead_zones) > 0:
            dead_zone_hours = len(dead_zones)
            dead_zone_spend_ratio = dead_zones["spend_ratio"].sum()

            # Penalty based on number of dead zone hours
            if dead_zone_hours <= 2:
                dead_penalty = dead_zone_hours * 5
            elif dead_zone_hours <= 4:
                dead_penalty = 10 + (dead_zone_hours - 2) * 5
            else:
                dead_penalty = 20 + min(20, (dead_zone_hours - 4) * 10)

            # Additional penalty for spend in dead zones
            spend_penalty = min(20, dead_zone_spend_ratio * 100)

            total_dead_penalty = min(40, dead_penalty + spend_penalty)
            score -= total_dead_penalty

        # ROAS variance penalty (0-30 points)
        roas_std = hourly_agg["purchase_roas"].std()
        roas_mean = hourly_agg["purchase_roas"].mean()

        if roas_mean > 0:
            roas_cv = roas_std / roas_mean
            if roas_cv < 0.5:
                variance_penalty = 0
            elif roas_cv < 1.0:
                variance_penalty = (roas_cv - 0.5) / 0.5 * 10
            else:
                variance_penalty = 10 + min(20, (roas_cv - 1.0) * 20)
            score -= variance_penalty

        # Peak hours bonus (0-30 points)
        peak_hours = hourly_agg[
            hourly_agg["purchase_roas"] >= self.thresholds["target_roas"]
        ]
        if len(peak_hours) > 0:
            peak_count = len(peak_hours)
            if peak_count >= 12:
                peak_bonus = 30
            elif peak_count >= 8:
                peak_bonus = 20
            elif peak_count >= 4:
                peak_bonus = 10
            else:
                peak_bonus = peak_count * 2.5
            score += min(30, peak_bonus)

        return min(100, max(0, score))

    def _calculate_weekly_efficiency_score(
        self,
        dow_agg: pd.DataFrame,
        weak_days: pd.DataFrame,
    ) -> float:
        """
        Calculate weekly efficiency score (0-100).

        Components:
        - Weak day penalty: 0-40 points
        - ROAS variance penalty: 0-30 points
        - Strong days bonus: 0-30 points

        Returns:
            Efficiency score (0-100, higher = better)
        """
        score = 100

        # Weak day penalty (0-40 points)
        if len(weak_days) > 0:
            weak_day_count = len(weak_days)
            weak_day_spend_ratio = weak_days["spend_ratio"].sum()

            # Penalty based on number of weak days
            if weak_day_count == 1:
                weak_penalty = 5
            elif weak_day_count == 2:
                weak_penalty = 15
            else:
                weak_penalty = min(40, 25 + (weak_day_count - 2) * 10)

            # Additional penalty for spend in weak days
            spend_penalty = min(20, weak_day_spend_ratio * 100)

            total_weak_penalty = min(40, weak_penalty + spend_penalty)
            score -= total_weak_penalty

        # ROAS variance penalty (0-30 points)
        roas_std = dow_agg["purchase_roas"].std()
        roas_mean = dow_agg["purchase_roas"].mean()

        if roas_mean > 0:
            roas_cv = roas_std / roas_mean
            if roas_cv < 0.3:
                variance_penalty = 0
            elif roas_cv < 0.6:
                variance_penalty = (roas_cv - 0.3) / 0.3 * 10
            else:
                variance_penalty = 10 + min(20, (roas_cv - 0.6) * 30)
            score -= variance_penalty

        # Strong days bonus (0-30 points)
        strong_days = dow_agg[
            dow_agg["purchase_roas"] >= self.thresholds["target_roas"]
        ]
        if len(strong_days) > 0:
            strong_count = len(strong_days)
            if strong_count >= 5:
                strong_bonus = 30
            elif strong_count >= 3:
                strong_bonus = 20
            else:
                strong_bonus = strong_count * 5
            score += min(30, strong_bonus)

        return min(100, max(0, score))

    def _group_consecutive_hours(self, hours: List[int]) -> List[str]:
        """
        Group consecutive hours into ranges.

        Example: [2, 3, 4, 22, 23] -> ["02:00-05:00", "22:00-00:00"]
        """
        if not hours:
            return []

        hours = sorted(hours)
        ranges = []
        start = hours[0]
        end = hours[0]

        for i in range(1, len(hours)):
            if hours[i] == end + 1:
                end = hours[i]
            else:
                ranges.append(f"{start:02d}:00-{(end + 1) % 24:02d}:00")
                start = hours[i]
                end = hours[i]

        ranges.append(f"{start:02d}:00-{(end + 1) % 24:02d}:00")
        return ranges

    def _create_hourly_issue(
        self,
        entity_id: str,
        result: Dict[str, Any],
    ) -> Issue:
        """Create hourly performance issue from analysis result."""
        dead_zones = result["dead_zones"]
        efficiency_score = result["efficiency_score"]

        # Calculate peak hours (best ROAS hours)
        hourly_agg = result["hourly_agg"]
        peak_hours = hourly_agg.nlargest(3, "purchase_roas")["hour"].tolist()
        peak_hours_str = ", ".join([f"{h:02d}:00" for h in peak_hours])

        # Calculate dead zone ROAS
        dead_zone_roas = result["dead_zones_df"]["purchase_roas"].mean()

        # Determine severity based on efficiency score
        if efficiency_score < 20:
            severity = IssueSeverity.CRITICAL
            status = "Critical time slot waste - urgent optimization needed"
        elif efficiency_score < 40:
            severity = IssueSeverity.HIGH
            status = "Significant dead zones detected"
        elif efficiency_score < 60:
            severity = IssueSeverity.MEDIUM
            status = "Moderate time slot inefficiency"
        else:
            severity = IssueSeverity.LOW
            status = "Minor dead zones detected"

        return Issue(
            id=f"hourly_performance_{entity_id}",
            category=IssueCategory.PERFORMANCE,
            severity=severity,
            title=f"Dead Hours Detected (Efficiency: {efficiency_score:.0f}/100)",
            description=(
                f"Identified {len(dead_zones)} low-performance time periods (dead zones). {status}. "
                f"Dead zones: {', '.join(dead_zones)}. "
                f"Dead zone ROAS: {dead_zone_roas:.2f} vs target {result['target_roas']}. "
                f"Peak hours: {peak_hours_str}. "
                f"Hourly efficiency score: {efficiency_score:.0f}/100."
            ),
            affected_entities=[entity_id],
            metrics={
                "efficiency_score": efficiency_score,
                "dead_zones": dead_zones,
                "dead_zone_roas": dead_zone_roas,
                "target_roas": result["target_roas"],
                "peak_hours": peak_hours,
                "total_spend_analysis_period": result["total_spend"],
                "analysis_type": "hourly",
            },
        )

    def _create_weekly_issue(
        self,
        entity_id: str,
        result: Dict[str, Any],
    ) -> Issue:
        """Create weekly performance issue from analysis result."""
        weak_days = result["weak_days"]
        efficiency_score = result["efficiency_score"]

        # Calculate strong days (best ROAS days)
        dow_agg = result["dow_agg"]
        strong_days = dow_agg.nlargest(3, "purchase_roas")["day_of_week"].tolist()
        strong_days_str = ", ".join([self.DAY_NAMES[d] for d in strong_days])

        # Calculate weak day ROAS
        weak_day_roas = result["weak_days_df"]["purchase_roas"].mean()

        # Determine severity based on efficiency score
        if efficiency_score < 20:
            severity = IssueSeverity.CRITICAL
            status = "Critical day performance waste - urgent optimization needed"
        elif efficiency_score < 40:
            severity = IssueSeverity.HIGH
            status = "Significant weak days detected"
        elif efficiency_score < 60:
            severity = IssueSeverity.MEDIUM
            status = "Moderate day performance inefficiency"
        else:
            severity = IssueSeverity.LOW
            status = "Minor weak days detected"

        return Issue(
            id=f"weekly_performance_{entity_id}",
            category=IssueCategory.PERFORMANCE,
            severity=severity,
            title=f"Weak Days Detected (Efficiency: {efficiency_score:.0f}/100)",
            description=(
                f"Identified {len(weak_days)} low-performance days. {status}. "
                f"Weak days: {', '.join(weak_days)}. "
                f"Weak day ROAS: {weak_day_roas:.2f} vs target {result['target_roas']}. "
                f"Strong days: {strong_days_str}. "
                f"Weekly efficiency score: {efficiency_score:.0f}/100."
            ),
            affected_entities=[entity_id],
            metrics={
                "efficiency_score": efficiency_score,
                "weak_days": weak_days,
                "weak_day_roas": weak_day_roas,
                "target_roas": result["target_roas"],
                "strong_days": strong_days,
                "total_spend_analysis_period": result["total_spend"],
                "analysis_days": result["analysis_days"],
                "analysis_type": "weekly",
            },
        )


def recommend_dayparting(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate hour-of-day scheduling recommendations based on hourly analysis.

    Args:
        analysis_result: Result from _analyze_hourly_performance

    Returns:
        Dict with recommended schedules and bid adjustments
    """
    hourly_agg = analysis_result["hourly_agg"]
    target_roas = analysis_result["target_roas"]

    # Classify hours
    high_performance = hourly_agg[
        hourly_agg["purchase_roas"] >= target_roas * 0.8
    ]["hour"].tolist()

    low_performance = hourly_agg[
        hourly_agg["purchase_roas"] < target_roas * 0.4
    ]["hour"].tolist()

    medium_performance = [
        h for h in range(24)
        if h not in high_performance and h not in low_performance
    ]

    return {
        "high_performance_hours": high_performance,
        "low_performance_hours": low_performance,
        "medium_performance_hours": medium_performance,
        "recommendations": {
            "high_performance": {
                "action": "increase_bid",
                "adjustment": "+20%",
                "hours": [f"{h:02d}:00-{(h+1)%24:02d}:00" for h in high_performance],
            },
            "low_performance": {
                "action": "decrease_bid",
                "adjustment": "-90%",
                "hours": [f"{h:02d}:00-{(h+1)%24:02d}:00" for h in low_performance],
            },
            "medium_performance": {
                "action": "maintain_bid",
                "adjustment": "0%",
                "hours": [f"{h:02d}:00-{(h+1)%24:02d}:00" for h in medium_performance],
            },
        },
    }


def recommend_day_scheduling(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate day-of-week scheduling recommendations based on weekly analysis.

    Args:
        analysis_result: Result from _analyze_weekly_performance

    Returns:
        Dict with recommended schedules and bid adjustments
    """
    dow_agg = analysis_result["dow_agg"]
    target_roas = analysis_result["target_roas"]

    # Classify days
    high_performance = dow_agg[
        dow_agg["purchase_roas"] >= target_roas * 0.8
    ]["day_of_week"].tolist()

    low_performance = dow_agg[
        dow_agg["purchase_roas"] < target_roas * 0.4
    ]["day_of_week"].tolist()

    medium_performance = [
        d for d in range(7)
        if d not in high_performance and d not in low_performance
    ]

    return {
        "high_performance_days": [DarkHoursDetector.DAY_NAMES[d] for d in high_performance],
        "low_performance_days": [DarkHoursDetector.DAY_NAMES[d] for d in low_performance],
        "medium_performance_days": [DarkHoursDetector.DAY_NAMES[d] for d in medium_performance],
        "recommendations": {
            "high_performance": {
                "action": "increase_bid",
                "adjustment": "+20%",
                "days": [DarkHoursDetector.DAY_NAMES[d] for d in high_performance],
            },
            "low_performance": {
                "action": "decrease_bid",
                "adjustment": "-90%",
                "days": [DarkHoursDetector.DAY_NAMES[d] for d in low_performance],
            },
            "medium_performance": {
                "action": "maintain_bid",
                "adjustment": "0%",
                "days": [DarkHoursDetector.DAY_NAMES[d] for d in medium_performance],
            },
        },
    }
