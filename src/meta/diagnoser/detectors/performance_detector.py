"""
Performance Detector - Detects performance-related issues.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import pandas as pd

from src.meta.diagnoser.schemas.models import Issue, IssueSeverity, IssueCategory
from src.meta.diagnoser.core.issue_detector import BaseDetector


logger = logging.getLogger(__name__)


class PerformanceDetector(BaseDetector):
    """
    Detects performance-related issues.

    Checks for:
    - Low ROAS
    - High CPA
    - Low CTR
    - Poor conversion rate
    """

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "min_roas": 1.5,
        "max_cpa": None,  # Will be calculated from data
        "min_ctr": 0.01,  # 1%
        "min_conversion_rate": 0.02,  # 2%
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance detector."""
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
        Detect performance issues.

        Args:
            data: Entity performance data with columns:
                - spend, impressions, reach, actions, roas
            entity_id: Entity identifier

        Returns:
            List of detected performance issues
        """
        issues = []

        # Check for low ROAS
        if "roas" in data.columns:
            roas_issue = self._check_roas(data, entity_id)
            if roas_issue:
                issues.append(roas_issue)

        # Check for high CPA
        if "spend" in data.columns and "actions" in data.columns:
            cpa_issue = self._check_cpa(data, entity_id)
            if cpa_issue:
                issues.append(cpa_issue)

        # Check for low CTR
        if "clicks" in data.columns and "impressions" in data.columns:
            ctr_issue = self._check_ctr(data, entity_id)
            if ctr_issue:
                issues.append(ctr_issue)

        # Check for declining trend
        if len(data) > 7:
            trend_issue = self._check_declining_trend(data, entity_id)
            if trend_issue:
                issues.append(trend_issue)

        return issues

    def _check_roas(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Optional[Issue]:
        """Check for low ROAS."""
        avg_roas = data["roas"].mean()

        if avg_roas < self.thresholds["min_roas"]:
            return Issue(
                id=f"low_roas_{entity_id}",
                category=IssueCategory.PERFORMANCE,
                severity=IssueSeverity.HIGH if avg_roas < 1.0 else IssueSeverity.MEDIUM,
                title="Low ROAS Detected",
                description=(
                    f"Average ROAS of {avg_roas:.2f} is below threshold "
                    f"of {self.thresholds['min_roas']}. "
                    f"Consider optimizing targeting or creative."
                ),
                affected_entities=[entity_id],
                metrics={"avg_roas": avg_roas, "threshold": self.thresholds["min_roas"]},
            )
        return None

    def _check_cpa(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Optional[Issue]:
        """Check for high CPA."""
        total_spend = data["spend"].sum()
        total_actions = data["actions"].sum()

        if total_actions > 0:
            avg_cpa = total_spend / total_actions

            # Use 2x median CPA as threshold if not set
            threshold = self.thresholds.get("max_cpa")
            if threshold is None:
                threshold = avg_cpa * 2

            if avg_cpa > threshold:
                return Issue(
                    id=f"high_cpa_{entity_id}",
                    category=IssueCategory.PERFORMANCE,
                    severity=IssueSeverity.MEDIUM,
                    title="High CPA Detected",
                    description=(
                        f"Average CPA of ${avg_cpa:.2f} exceeds threshold "
                        f"of ${threshold:.2f}. "
                        f"Review audience targeting and bid strategy."
                    ),
                    affected_entities=[entity_id],
                    metrics={"avg_cpa": avg_cpa, "threshold": threshold},
                )
        return None

    def _check_ctr(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Optional[Issue]:
        """Check for low CTR."""
        total_clicks = data["clicks"].sum()
        total_impressions = data["impressions"].sum()

        if total_impressions > 0:
            ctr = total_clicks / total_impressions

            if ctr < self.thresholds["min_ctr"]:
                return Issue(
                    id=f"low_ctr_{entity_id}",
                    category=IssueCategory.PERFORMANCE,
                    severity=IssueSeverity.MEDIUM,
                    title="Low CTR Detected",
                    description=(
                        f"CTR of {ctr:.2%} is below threshold "
                        f"of {self.thresholds['min_ctr']:.2%}. "
                        f"Consider improving creative or audience relevance."
                    ),
                    affected_entities=[entity_id],
                    metrics={"ctr": ctr, "threshold": self.thresholds["min_ctr"]},
                )
        return None

    def _check_declining_trend(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Optional[Issue]:
        """Check for declining performance trend."""
        if "roas" not in data.columns:
            return None

        # Calculate 7-day vs prior 7-day ROAS
        recent_roas = data.tail(7)["roas"].mean()
        prior_roas = data.head(max(1, len(data) - 7))["roas"].mean()

        if prior_roas > 0:
            decline_pct = (recent_roas - prior_roas) / prior_roas

            if decline_pct < -0.20:  # 20% decline
                return Issue(
                    id=f"declining_trend_{entity_id}",
                    category=IssueCategory.PERFORMANCE,
                    severity=IssueSeverity.HIGH if decline_pct < -0.40 else IssueSeverity.MEDIUM,
                    title="Declining Performance Trend",
                    description=(
                        f"ROAS declined by {abs(decline_pct):.1%} "
                        f"from {prior_roas:.2f} to {recent_roas:.2f}. "
                        f"Investigate cause: fatigue, competition, or seasonality."
                    ),
                    affected_entities=[entity_id],
                    metrics={
                        "prior_roas": prior_roas,
                        "recent_roas": recent_roas,
                        "decline_pct": decline_pct,
                    },
                )
        return None
