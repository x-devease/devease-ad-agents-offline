"""
Configuration Detector - Detects configuration issues.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import pandas as pd

from src.meta.diagnoser.schemas.models import Issue, IssueSeverity, IssueCategory
from src.meta.diagnoser.core.issue_detector import BaseDetector


logger = logging.getLogger(__name__)


class ConfigurationDetector(BaseDetector):
    """
    Detects configuration-related issues.

    Checks for:
    - Misaligned budgets
    - Incompatible audience settings
    - Bid strategy issues
    - Missing conversion tracking
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize configuration detector."""
        super().__init__(config)

    def detect(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> List[Issue]:
        """
        Detect configuration issues.

        Args:
            data: Entity configuration and performance data
            entity_id: Entity identifier

        Returns:
            List of detected configuration issues
        """
        issues = []

        # Check for budget misalignment
        budget_issue = self._check_budget_alignment(data, entity_id)
        if budget_issue:
            issues.append(budget_issue)

        # Check for bid strategy issues
        bid_issue = self._check_bid_strategy(data, entity_id)
        if bid_issue:
            issues.append(bid_issue)

        # TODO: Add audience overlap check when implemented
        # overlap_issue = self._check_audience_overlap(data, entity_id)
        # if overlap_issue:
        #     issues.append(overlap_issue)

        return issues

    def _check_budget_alignment(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Optional[Issue]:
        """Check for budget misalignment."""
        if "daily_budget" not in data.columns or "spend" not in data.columns:
            return None

        avg_spend = data["spend"].mean()
        daily_budget = data["daily_budget"].iloc[0]

        # Check if consistently under/over budget
        utilization = avg_spend / daily_budget if daily_budget > 0 else 0

        if utilization < 0.5:
            return Issue(
                id=f"budget_underutilized_{entity_id}",
                category=IssueCategory.CONFIGURATION,
                severity=IssueSeverity.MEDIUM,
                title="Budget Underutilization",
                description=(
                    f"Average spend (${avg_spend:.2f}) is only {utilization:.1%} "
                    f"of daily budget (${daily_budget:.2f}). "
                    f"Consider reducing budget or increasing targets."
                ),
                affected_entities=[entity_id],
                metrics={"utilization": utilization, "avg_spend": avg_spend},
            )
        elif utilization > 1.2:
            return Issue(
                id=f"budget_exceeded_{entity_id}",
                category=IssueCategory.CONFIGURATION,
                severity=IssueSeverity.HIGH,
                title="Budget Exceeded",
                description=(
                    f"Average spend (${avg_spend:.2f}) exceeds "
                    f"daily budget (${daily_budget:.2f}) by {utilization - 1:.1%}. "
                    f"Increase budget to avoid throttling."
                ),
                affected_entities=[entity_id],
                metrics={"utilization": utilization, "avg_spend": avg_spend},
            )
        return None

    def _check_bid_strategy(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Optional[Issue]:
        """Check for bid strategy issues."""
        if "bid_strategy" not in data.columns:
            return None

        bid_strategy = data["bid_strategy"].iloc[0]
        roas = data.get("roas", pd.Series([0])).mean()

        # Check for Lowest Cost without bids when ROAS is high
        if bid_strategy == "LOWEST_COST_WITHOUT_CAP" and roas > 3.0:
            return Issue(
                id=f"bid_strategy_suboptimal_{entity_id}",
                category=IssueCategory.CONFIGURATION,
                severity=IssueSeverity.LOW,
                title="Bid Strategy May Be Suboptimal",
                description=(
                    f"Using LOWEST_COST_WITHOUT_CAP with high ROAS ({roas:.2f}). "
                    f"Consider TARGET_COST to scale efficiently."
                ),
                affected_entities=[entity_id],
                metrics={"bid_strategy": bid_strategy, "roas": roas},
            )
        return None

    def _check_audience_overlap(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> Optional[Issue]:
        """Check for audience overlap (placeholder)."""
        # This would require comparing multiple adsets
        # For now, return None
        return None
