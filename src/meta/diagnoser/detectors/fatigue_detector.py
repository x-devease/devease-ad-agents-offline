"""
Fatigue Detector - Creative Fatigue Detection (Score-Based, No Lookahead Bias).

Detects creative fatigue using rolling window approach.
Only uses historical data available at prediction time.
Outputs fatigue severity scores instead of monetary loss.
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


class FatigueDetector(BaseDetector):
    """
    Detects creative fatigue using rolling window (no lookahead bias).

    Algorithm:
    1. For each day t, only use data from day [t-window_size : t-1]
    2. Calculate cumulative frequency within the window
    3. Identify golden period in the window
    4. Check if current day shows fatigue signals
    5. Report fatigue only if consecutive days detected

    Key Feature:
    - Uses rolling window approach to avoid lookahead bias
    - Only uses historical data available at prediction time
    - Suitable for real-time prediction scenarios

    Scoring:
    - 0-30: Healthy (no fatigue)
    - 30-60: Early signs (monitor)
    - 60-80: Moderate fatigue (consider refresh)
    - 80-100: Severe fatigue (urgent action needed)
    """

    # Default thresholds (optimized via Iteration 10 - Improved recall)
    DEFAULT_THRESHOLDS = {
        "window_size_days": 23,  # Optimized: 21 → 23
        "golden_min_freq": 1.0,
        "golden_max_freq": 2.5,
        "fatigue_freq_threshold": 3.0,
        "cpa_increase_threshold": 1.10,  # Optimized: 1.2 → 1.15 → 1.10 (improved recall)
        "consecutive_days": 1,  # Need 1 consecutive day to confirm (optimized from 2)
        "min_golden_days": 1,  # Optimized: 2 → 1
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fatigue detector."""
        super().__init__(config)
        self.thresholds = {
            **self.DEFAULT_THRESHOLDS,
            **self.config.get("thresholds", {}),
        }

    def detect(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> List[Issue]:
        """
        Detect creative fatigue using rolling window.

        Args:
            data: Ad-level daily data with columns:
                - date_start, spend, impressions, reach, conversions (preprocessed)
                - OR: actions JSON (will be parsed internally)
            entity_id: Ad/Creative ID

        Returns:
            List of detected fatigue issues with severity scores (latest detection only)
        """
        issues = []

        # Check for required columns
        required_cols = ["date_start", "spend", "impressions", "reach"]
        if not all(col in data.columns for col in required_cols):
            logger.warning(f"Entity {entity_id}: Missing required columns")
            return issues

        # Handle conversions column (preprocessed or from JSON)
        if "conversions" not in data.columns:
            if "actions" in data.columns:
                # Parse actions JSON
                data = data.copy()
                data["conversions"] = data["actions"].apply(self._parse_conversions_from_json)
            else:
                logger.warning(f"Entity {entity_id}: No conversion data")
                return issues

        # Sort by date
        data = data.sort_values("date_start").reset_index(drop=True)

        # Skip if insufficient data
        min_required = self.thresholds["window_size_days"] + self.thresholds["consecutive_days"]
        if len(data) < min_required:
            logger.debug(
                f"Entity {entity_id}: Insufficient data: {len(data)} < {min_required}"
            )
            return issues

        # Log for debugging
        if str(entity_id) == '120215767837920310':
            logger.info(f"Entity {entity_id}: Running fatigue analysis on {len(data)} days...")
            logger.info(f"  Conversions: {data['conversions'].sum():.1f}")
            logger.info(f"  Spend: {data['spend'].sum():.2f}")

        # Run rolling window fatigue analysis
        analysis_result = self._analyze_fatigue_rolling(data, entity_id)

        if analysis_result["is_fatigued"]:
            logger.info(f"Entity {entity_id}: Fatigue detected!")
            issues.append(self._create_fatigue_issue(entity_id, data, analysis_result))
        else:
            if str(entity_id) == '120215767837920310':
                logger.info(f"Entity {entity_id}: No fatigue detected - {analysis_result.get('reason', 'unknown')}")

        return issues

    def _analyze_fatigue_rolling(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze creative fatigue using rolling window approach.

        Uses rolling window to avoid lookahead bias - only uses historical data.

        Returns:
            Dictionary with fatigue analysis results and severity score (0-100)
        """
        window_size = self.thresholds["window_size_days"]
        consecutive_days = self.thresholds["consecutive_days"]

        detections = []

        # Start from day where we have enough data
        min_days = window_size + consecutive_days
        logger.debug(f"Analyzing {len(data)} days, starting from day {min_days}")

        for i in range(min_days, len(data)):
            # ✅ ONLY use historical data (days [i-window_size : i-1])
            window = data.iloc[i - window_size : i].copy()

            # Calculate cumulative frequency within the window
            window = self._calculate_cumulative_frequency(window)

            # Find golden period in the window
            golden_period = self._find_golden_period(window)

            if len(golden_period) < self.thresholds["min_golden_days"]:
                # Not enough golden period data
                logger.debug(f"Day {i}: Golden period has {len(golden_period)} days, need {self.thresholds['min_golden_days']}")
                continue

            # Calculate CPA in golden period
            total_conversions = golden_period["conversions"].sum()
            if total_conversions == 0:
                # No conversions in golden period, skip this window
                logger.debug(f"Day {i}: No conversions in golden period")
                continue

            cpa_gold = (
                golden_period["spend"].sum() / total_conversions
            )

            # Check current day (day i) for fatigue
            current = data.iloc[i]
            current_freq = window.iloc[-1]["cum_freq"]
            current_cpa = current["spend"] / current["conversions"] if current["conversions"] > 0 else np.inf

            # Check fatigue conditions
            is_fatigued = (
                current_freq > self.thresholds["fatigue_freq_threshold"] and
                current_cpa > cpa_gold * self.thresholds["cpa_increase_threshold"]
            )

            logger.debug(f"Day {i}: freq={current_freq:.2f}, cpa={current_cpa:.2f}, cpa_gold={cpa_gold:.2f}, fatigued={is_fatigued}")

            detections.append({
                "date": current["date_start"],
                "is_fatigued": is_fatigued,
                "current_freq": current_freq,
                "current_cpa": current_cpa,
                "cpa_gold": cpa_gold,
            })

        # Check for consecutive detections
        consecutive_count = self._count_consecutive_detections(detections)
        logger.debug(f"Consecutive detections: {consecutive_count}, need {consecutive_days}")

        if consecutive_count >= consecutive_days and detections:
            # Get the most recent detection
            latest_detection = detections[-1]

            # Calculate post-fatigue metrics
            post_fatigue_start = len(data) - consecutive_count
            post_fatigue = data.iloc[post_fatigue_start:]

            current_cpa = post_fatigue["spend"].sum() / post_fatigue["conversions"].sum()
            cpa_increase_pct = ((current_cpa - latest_detection["cpa_gold"]) / latest_detection["cpa_gold"] * 100) if latest_detection["cpa_gold"] > 0 else 0

            # Calculate severity score
            severity_score = self._calculate_fatigue_severity(
                fatigue_freq=latest_detection["current_freq"],
                cpa_increase_pct=cpa_increase_pct,
                post_fatigue_days=len(post_fatigue),
            )

            health_score = 100 - severity_score

            return {
                "is_fatigued": True,
                "fatigue_date": data.iloc[post_fatigue_start]["date_start"],
                "fatigue_freq": latest_detection["current_freq"],
                "cpa_gold": latest_detection["cpa_gold"],
                "current_cpa": current_cpa,
                "cpa_increase_pct": cpa_increase_pct,
                "post_fatigue_days": len(post_fatigue),
                "severity_score": severity_score,
                "health_score": health_score,
                "consecutive_days": consecutive_count,
            }

        # No fatigue detected
        health_score = self._calculate_health_score_no_fatigue(data)
        return {
            "is_fatigued": False,
            "reason": "no_fatigue_detected",
            "health_score": health_score,
        }

    def _calculate_cumulative_frequency(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative frequency for given data."""
        data = data.copy()
        data["cum_impressions"] = data["impressions"].cumsum()
        data["cum_reach"] = data["reach"].expanding().max()
        data["cum_freq"] = data["cum_impressions"] / data["cum_reach"].replace(0, np.nan)
        data["daily_cpa"] = data["spend"] / data["conversions"].replace(0, np.nan)
        return data

    def _find_golden_period(self, data: pd.DataFrame) -> pd.DataFrame:
        """Find golden period in the given data."""
        golden_mask = (
            (data["cum_freq"] > self.thresholds["golden_min_freq"]) &
            (data["cum_freq"] < self.thresholds["golden_max_freq"])
        )
        return data[golden_mask]

    def _count_consecutive_detections(
        self,
        detections: List[Dict[str, Any]],
    ) -> int:
        """Count consecutive fatigue detections at the end of the list."""
        if not detections:
            return 0

        consecutive_count = 0
        for detection in reversed(detections):
            if detection["is_fatigued"]:
                consecutive_count += 1
            else:
                break

        return consecutive_count

    def _calculate_fatigue_severity(
        self,
        fatigue_freq: float,
        cpa_increase_pct: float,
        post_fatigue_days: int,
    ) -> float:
        """
        Calculate fatigue severity score (0-100).

        Components:
        - Frequency penalty: 0-30 points (based on how much freq exceeds threshold)
        - CPA penalty: 0-50 points (based on CPA increase percentage)
        - Duration penalty: 0-20 points (based on days in fatigue)

        Returns:
            Severity score (0-100, higher = more severe)
        """
        score = 0

        # Frequency penalty (0-30 points)
        freq_excess = max(0, fatigue_freq - self.thresholds["fatigue_freq_threshold"])
        freq_penalty = min(30, (freq_excess / 3.0) * 30)
        score += freq_penalty

        # CPA penalty (0-50 points)
        if cpa_increase_pct < 30:
            cpa_penalty = 0
        elif cpa_increase_pct < 100:
            cpa_penalty = ((cpa_increase_pct - 30) / 70) * 30
        else:
            cpa_penalty = 30 + min(20, (cpa_increase_pct - 100) / 100 * 20)
        score += cpa_penalty

        # Duration penalty (0-20 points)
        if post_fatigue_days <= 7:
            duration_penalty = (post_fatigue_days / 7) * 5
        elif post_fatigue_days <= 14:
            duration_penalty = 5 + ((post_fatigue_days - 7) / 7) * 5
        else:
            duration_penalty = 10 + min(10, (post_fatigue_days - 14) / 14 * 10)
        score += duration_penalty

        return min(100, max(0, score))

    def _calculate_health_score_no_fatigue(
        self,
        data: pd.DataFrame,
    ) -> float:
        """Calculate health score when no fatigue is detected."""
        if "purchase_roas" in data.columns:
            # Ensure purchase_roas is numeric
            roas_values = pd.to_numeric(data["purchase_roas"], errors="coerce").fillna(0)
            avg_roas = roas_values.replace([np.inf, -np.inf], 0).mean()
            # ROAS < 1.0 = 0-30 points, ROAS 1.0-3.0 = 30-70 points, ROAS > 3.0 = 70-100 points
            if avg_roas < 1.0:
                health_score = avg_roas * 30
            elif avg_roas < 3.0:
                health_score = 30 + ((avg_roas - 1.0) / 2.0) * 40
            else:
                health_score = 70 + min(30, (avg_roas - 3.0) * 5)
        else:
            health_score = 70.0  # Default if no ROAS data

        return min(100, max(0, health_score))

    def _create_fatigue_issue(
        self,
        entity_id: str,
        data: pd.DataFrame,
        result: Dict[str, Any],
    ) -> Issue:
        """Create fatigue issue from analysis result."""
        severity_score = result["severity_score"]
        health_score = result["health_score"]

        # Calculate premium loss and missed conversions
        post_fatigue_start = len(data) - result["post_fatigue_days"]
        post_fatigue_data = data.iloc[post_fatigue_start:]

        total_spend_fatigue = post_fatigue_data["spend"].sum()
        total_conversions_fatigue = post_fatigue_data["conversions"].sum()

        # What we would have spent at golden CPA
        expected_spend = total_conversions_fatigue * result["cpa_gold"]
        premium_loss = total_spend_fatigue - expected_spend

        # What conversions we would have gotten at same spend
        expected_conversions = total_spend_fatigue / result["cpa_gold"] if result["cpa_gold"] > 0 else 0
        missed_conversions = expected_conversions - total_conversions_fatigue

        # Determine severity level based on score
        if severity_score >= 80:
            severity = IssueSeverity.CRITICAL
            action = "PAUSE this ad immediately and refresh creative"
            explanation = "Ad is severely fatigued and wasting budget"
        elif severity_score >= 60:
            severity = IssueSeverity.HIGH
            action = "Consider pausing or reducing budget significantly"
            explanation = "Ad performance has declined substantially"
        elif severity_score >= 30:
            severity = IssueSeverity.MEDIUM
            action = "Monitor closely and plan creative refresh"
            explanation = "Early signs of fatigue detected"
        else:
            severity = IssueSeverity.LOW
            action = "Plan rotation schedule"
            explanation = "Minor fatigue beginning to show"

        # Calculate business impact in plain language
        premium_loss_formatted = f"${premium_loss:,.2f}" if premium_loss > 0 else "$0.00"
        missed_conversions_formatted = f"{missed_conversions:.0f} conversions" if missed_conversions > 1 else f"{missed_conversions:.1f} conversions"

        return Issue(
            id=f"fatigue_{entity_id}",
            category=IssueCategory.FATIGUE,
            severity=severity,
            title=f"Creative Fatigue: {explanation} (Health: {health_score:.0f}/100)",
            description=(
                f"**What's happening:** {explanation}. This ad has been shown to the same audience {result['fatigue_freq']:.1f}x, "
                f"causing CPA to increase by {result['cpa_increase_pct']:.0f}% since its best-performing period.\n\n"
                f"**Business impact:** You've lost {premium_loss_formatted} ({missed_conversions_formatted}) during the fatigue period "
                f"({result['post_fatigue_days']} days).\n\n"
                f"**Action recommended:** {action}.\n\n"
                f"**Metrics:** Health score: {health_score:.0f}/100 (higher is better). "
                f"Current CPA: ${result['current_cpa']:.2f} vs best: ${result['cpa_gold']:.2f}."
            ),
            affected_entities=[entity_id],
            metrics={
                "severity_score": severity_score,
                "health_score": health_score,
                "fatigue_freq": result["fatigue_freq"],
                "cpa_increase_pct": result["cpa_increase_pct"],
                "post_fatigue_days": result["post_fatigue_days"],
                "cpa_gold": result["cpa_gold"],
                "current_cpa": result["current_cpa"],
                "consecutive_days": result["consecutive_days"],
                "window_size_days": self.thresholds["window_size_days"],
                "premium_loss": max(0, premium_loss),
                "missed_conversions": max(0, missed_conversions),
                "action_recommendation": action,
                "business_impact": f"{premium_loss_formatted} wasted ({missed_conversions_formatted} missed)",
            },
        )

    def _parse_conversions_from_json(self, actions_str):
        """Parse conversions from actions JSON string."""
        import json
        if pd.isna(actions_str) or actions_str == "":
            return 0

        try:
            cleaned = actions_str.replace("'", '"')
            actions = json.loads(cleaned)

            purchase_keys = [
                "offsite_conversion.fb_pixel_purchase",
                "omni_purchase",
                "purchase",
                "onsite_web_purchase",
            ]

            total = 0.0
            for action in actions:
                action_type = action.get("action_type", "")
                if action_type in purchase_keys:
                    total += float(action.get("value", 0))

            return total
        except:
            return 0
