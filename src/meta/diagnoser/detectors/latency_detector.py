"""
Latency Detector - Response Delay Detection (Daily, Score-Based).

Detects delays in responding to performance drops using daily data.
Outputs responsiveness scores instead of monetary loss.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from src.meta.diagnoser.schemas.models import Issue, IssueSeverity, IssueCategory
from src.meta.diagnoser.core.issue_detector import BaseDetector


logger = logging.getLogger(__name__)


class LatencyDetector(BaseDetector):
    """
    Detects response latency to performance drops using daily data.

    Algorithm:
    1. Calculate rolling average ROAS (3-day window)
    2. Identify breakdown days:
       - Daily ROAS < threshold
       - Daily spend > min_spend
       - ROAS drop > 20% from rolling average

    3. Track days until intervention:
       - First PAUSED status after breakdown
       - OR ROAS recovery to threshold level

    4. Calculate responsiveness score (0-100)

    Scoring:
    - 80-100: Excellent response (same day or next day)
    - 60-79: Good response (2 days delay)
    - 40-59: Moderate delay (3-4 days)
    - 20-39: Poor response (5-7 days)
    - 0-19: Critical delay (>7 days)
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "roas_threshold": 1.0,
        "rolling_window_days": 3,
        "min_daily_spend": 50,
        "min_drop_ratio": 0.2,  # 20% drop from average
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize latency detector."""
        super().__init__(config)
        self.thresholds = {
            **self.DEFAULT_THRESHOLDS,
            **self.config.get("thresholds", {}),
        }

    def detect(
        self,
        data: pd.DataFrame,
        entity_id: str,
        status_changes: Optional[pd.DataFrame] = None,
    ) -> List[Issue]:
        """
        Detect response latency issues using daily data.

        Args:
            data: AdSet-level daily data with columns:
                - date_start, spend, purchase_roas, (optional) conversions
            entity_id: AdSet ID
            status_changes: DataFrame with status change events

        Returns:
            List of detected latency issues with responsiveness scores
        """
        issues = []

        # Check for required columns
        required_cols = ["date_start", "spend", "purchase_roas"]
        if not all(col in data.columns for col in required_cols):
            logger.warning(f"Missing required columns for latency detection: {entity_id}")
            return issues

        # Skip if insufficient data
        if len(data) < self.thresholds["rolling_window_days"] + 1:
            logger.info(
                f"Insufficient data for latency analysis: "
                f"{len(data)} days < {self.thresholds['rolling_window_days'] + 1}"
            )
            return issues

        # Run latency analysis
        incidents = self._detect_latency_incidents(data, entity_id, status_changes)

        if incidents:
            issues.append(self._create_latency_issue(entity_id, incidents))

        return issues

    def _detect_latency_incidents(
        self,
        data: pd.DataFrame,
        entity_id: str,
        status_changes: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect latency incidents using daily data.

        Returns:
            List of incident dicts with breakdown, intervention, and scores
        """
        # Prepare data
        data = data.sort_values("date_start").reset_index(drop=True)
        data["date_start"] = pd.to_datetime(data["date_start"])

        # Replace NaN/inf ROAS
        data["purchase_roas"] = data["purchase_roas"].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate rolling average ROAS
        data["rolling_roas"] = data["purchase_roas"].rolling(
            window=self.thresholds["rolling_window_days"],
            min_periods=1
        ).mean()

        incidents = []
        is_bleeding = False
        t_break = None
        bleed_start_idx = None

        for i in range(self.thresholds["rolling_window_days"], len(data)):
            row = data.iloc[i]
            rolling_roas = data.loc[i - 1, "rolling_roas"]  # Previous day's rolling avg

            # Calculate drop ratio
            if rolling_roas > 0:
                drop_ratio = (rolling_roas - row["purchase_roas"]) / rolling_roas
            else:
                drop_ratio = 1.0  # Assume 100% drop if rolling avg is 0

            # Check breakdown conditions
            is_breakdown = (
                row["purchase_roas"] < self.thresholds["roas_threshold"]
                and row["spend"] > self.thresholds["min_daily_spend"]
                and drop_ratio >= self.thresholds["min_drop_ratio"]
            )

            if not is_bleeding and is_breakdown:
                # Breakdown detected
                is_bleeding = True
                t_break = row["date_start"]
                bleed_start_idx = i
                logger.debug(
                    f"Breakdown detected for {entity_id} at {t_break}, "
                    f"ROAS: {row['purchase_roas']:.2f}, "
                    f"Drop: {drop_ratio:.1%}, "
                    f"Spend: ${row['spend']:.2f}"
                )
                continue

            # If bleeding, check for intervention or recovery
            if is_bleeding:
                # Check for status change to PAUSED
                t_action = self._find_intervention_time(
                    t_break, entity_id, status_changes
                )

                # Check for ROAS recovery (natural improvement)
                # Recovery is when ROAS returns to threshold level
                if row["purchase_roas"] >= self.thresholds["roas_threshold"]:
                    t_recovery = row["date_start"]
                else:
                    t_recovery = None

                # Determine actual intervention time
                if t_action is not None and t_recovery is not None:
                    # Use whichever came first
                    t_intervention = min(t_action, t_recovery)
                    intervention_type = "pause" if t_action <= t_recovery else "recovery"
                elif t_action is not None:
                    t_intervention = t_action
                    intervention_type = "pause"
                elif t_recovery is not None:
                    t_intervention = t_recovery
                    intervention_type = "recovery"
                else:
                    # Still bleeding, no intervention yet
                    continue

                # Calculate bleeding period
                if bleed_start_idx is not None:
                    bleeding_data = data.loc[bleed_start_idx:i]

                    # Calculate days delayed
                    delta_days = (t_intervention - t_break).days

                    if delta_days > 0:
                        # Calculate responsiveness score for this incident
                        incident_score = self._calculate_responsiveness_score(delta_days)

                        incidents.append({
                            "t_break": t_break,
                            "t_action": t_intervention,
                            "intervention_type": intervention_type,
                            "delta_days": delta_days,
                            "responsiveness_score": incident_score,
                            "avg_roas_during_bleed": bleeding_data["purchase_roas"].mean(),
                            "bleeding_days": len(bleeding_data),
                            "bleeding_spend": bleeding_data["spend"].sum(),
                        })

                # Reset bleeding state
                is_bleeding = False
                t_break = None
                bleed_start_idx = None

        return incidents

    def _find_intervention_time(
        self,
        t_break: datetime,
        entity_id: str,
        status_changes: Optional[pd.DataFrame] = None,
    ) -> Optional[datetime]:
        """
        Find intervention time from status changes.

        If no status changes available, return None (no intervention detected).
        """
        if status_changes is None or len(status_changes) == 0:
            return None

        # Filter for this adset
        entity_changes = status_changes[status_changes["adset_id"] == entity_id]

        if len(entity_changes) == 0:
            return None

        # Find first PAUSED after t_break
        paused_changes = entity_changes[
            (entity_changes["change_date"] > t_break)
            & (entity_changes["new_status"] == "PAUSED")
        ]

        if len(paused_changes) > 0:
            return paused_changes.iloc[0]["change_date"]

        return None

    def _calculate_responsiveness_score(self, delay_days: int) -> float:
        """
        Calculate responsiveness score (0-100) based on delay in days.

        Scoring:
        - 0-1 days: 80-100 points (excellent)
        - 2 days: 60-79 points (good)
        - 3-4 days: 40-59 points (moderate)
        - 5-7 days: 20-39 points (poor)
        - >7 days: 0-19 points (critical)

        Returns:
            Responsiveness score (0-100, higher = better)
        """
        if delay_days <= 1:
            # 0-1 day = 80-100 points
            score = 100 - (delay_days * 20)
        elif delay_days == 2:
            # 2 days = 60-80 points
            score = 80
        elif delay_days <= 4:
            # 3-4 days = 40-60 points
            score = 60 - ((delay_days - 2) * 20)
        elif delay_days <= 7:
            # 5-7 days = 20-40 points
            score = 40 - ((delay_days - 4) * 6.67)
        else:
            # >7 days = 0-20 points
            score = max(0, 20 - (delay_days - 7) * 3)

        return min(100, max(0, score))

    def _create_latency_issue(
        self,
        entity_id: str,
        incidents: List[Dict[str, Any]],
    ) -> Issue:
        """Create latency issue from detected incidents."""
        avg_delay = np.mean([inc["delta_days"] for inc in incidents])
        max_delay = max([inc["delta_days"] for inc in incidents])
        avg_score = np.mean([inc["responsiveness_score"] for inc in incidents])

        # Count intervention types
        pause_count = sum(1 for inc in incidents if inc["intervention_type"] == "pause")
        recovery_count = sum(1 for inc in incidents if inc["intervention_type"] == "recovery")

        # Determine severity based on average responsiveness score
        if avg_score < 20:
            severity = IssueSeverity.CRITICAL
            status = "Critical response delays detected"
        elif avg_score < 40:
            severity = IssueSeverity.HIGH
            status = "Poor response times"
        elif avg_score < 60:
            severity = IssueSeverity.MEDIUM
            status = "Moderate response delays"
        else:
            severity = IssueSeverity.LOW
            status = "Response delays acceptable"

        return Issue(
            id=f"latency_{entity_id}",
            category=IssueCategory.PERFORMANCE,
            severity=severity,
            title=f"Response Latency Detected (Responsiveness: {avg_score:.0f}/100)",
            description=(
                f"Detected {len(incidents)} performance drop events with delayed response. {status}. "
                f"Average response delay: {avg_delay:.1f} days. "
                f"Longest delay: {max_delay:.0f} days. "
                f"Responsiveness score: {avg_score:.0f}/100. "
                f"Interventions: {pause_count} manual pauses, {recovery_count} natural recoveries."
            ),
            affected_entities=[entity_id],
            metrics={
                "num_incidents": len(incidents),
                "avg_delay_days": avg_delay,
                "max_delay_days": max_delay,
                "avg_responsiveness_score": avg_score,
                "manual_pause_interventions": pause_count,
                "natural_recovery_interventions": recovery_count,
            },
        )


def infer_status_changes(adset_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer status change events from adset_daily data.

    Args:
        adset_daily_df: DataFrame with columns:
            - adset_id, date_start, adset_status

    Returns:
        DataFrame with columns:
            - adset_id, change_date, old_status, new_status
    """
    df = adset_daily_df.sort_values(["adset_id", "date_start"]).copy()

    # Detect status changes within each adset
    df["old_status"] = df.groupby("adset_id")["adset_status"].shift(1)
    df["status_changed"] = df["adset_status"] != df["old_status"]

    # Filter to changes only
    changes = df[df["status_changed"] & df["old_status"].notna()].copy()

    # Convert date_start to datetime
    changes["change_date"] = pd.to_datetime(changes["date_start"])

    # Select and rename columns
    result = changes[["adset_id", "change_date", "old_status", "adset_status"]].rename(
        columns={"adset_status": "new_status"}
    )
    return result
