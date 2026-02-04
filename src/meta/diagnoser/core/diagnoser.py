"""
Main Diagnoser - Coordinates diagnosis process for Meta ad accounts.

This module provides the primary Diagnoser class that orchestrates the
entire diagnosis workflow, from data ingestion through issue detection
to report generation.

Key Classes:
    Diagnoser: Main coordinator for diagnosis workflow

Key Features:
    - Multi-level diagnosis: account, campaign, adset
    - Automatic detector registration and configuration
    - Comprehensive report generation with IssueSeverity ratings
    - Integration with IssueDetector for pluggable detection strategies

Usage:
    >>> from src.meta.diagnoser.core import Diagnoser
    >>> diagnoser = Diagnoser(config={...})
    >>> report = diagnoser.diagnose_account(account_id="123", data=data)
    >>> print(report.summary)

Workflow:
    1. Initialize Diagnoser with optional configuration
    2. Load entity performance data
    3. Run diagnosis at desired level (account/campaign/adset)
    4. Receive DiagnosisReport with issues and recommendations

See Also:
    - IssueDetector: Core detection abstraction
    - ReportGenerator: Report formatting and generation
    - DetectorFactory: Dynamic detector creation
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd

from src.meta.diagnoser.core.issue_detector import IssueDetector
from src.meta.diagnoser.schemas.models import (
    DiagnosisReport,
    IssueSeverity,
)
from src.meta.diagnoser.detectors.performance_detector import PerformanceDetector
from src.meta.diagnoser.detectors.configuration_detector import ConfigurationDetector
from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector


logger = logging.getLogger(__name__)


class Diagnoser:
    """
    Main diagnoser for Meta ad accounts and campaigns.

    Coordinates detection and analysis to generate comprehensive
    diagnosis reports.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize diagnoser."""
        self.config = config or {}
        self.issue_detector = IssueDetector(config)

        # Register default detectors
        self._register_default_detectors()

    def _register_default_detectors(self) -> None:
        """Register default issue detectors."""
        self.issue_detector.register_detector(PerformanceDetector())
        self.issue_detector.register_detector(ConfigurationDetector())
        self.issue_detector.register_detector(FatigueDetector())

    def diagnose_account(
        self,
        account_id: str,
        data: pd.DataFrame,
    ) -> DiagnosisReport:
        """
        Diagnose entire account.

        Args:
            account_id: Meta account ID
            data: Account performance data

        Returns:
            Complete diagnosis report
        """
        logger.info(f"Diagnosing account: {account_id}")

        # Detect issues
        issues = self.issue_detector.detect_all(data, account_id)

        # Calculate health score
        health_score = self._calculate_health_score(issues)

        # Generate summary
        summary = self._generate_summary(issues, health_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        return DiagnosisReport(
            account_id=account_id,
            entity_type="account",
            entity_id=account_id,
            issues=issues,
            recommendations=recommendations,
            summary=summary,
            overall_health_score=health_score,
        )

    def diagnose_campaign(
        self,
        account_id: str,
        campaign_id: str,
        data: pd.DataFrame,
    ) -> DiagnosisReport:
        """
        Diagnose specific campaign.

        Args:
            account_id: Meta account ID
            campaign_id: Campaign ID
            data: Campaign performance data

        Returns:
            Campaign diagnosis report
        """
        logger.info(f"Diagnosing campaign: {campaign_id}")

        # Detect issues
        issues = self.issue_detector.detect_all(data, campaign_id)

        # Calculate health score
        health_score = self._calculate_health_score(issues)

        # Generate summary
        summary = self._generate_summary(issues, health_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        return DiagnosisReport(
            account_id=account_id,
            entity_type="campaign",
            entity_id=campaign_id,
            issues=issues,
            recommendations=recommendations,
            summary=summary,
            overall_health_score=health_score,
        )

    def _calculate_health_score(self, issues: List) -> float:
        """
        Calculate overall health score (0-100).

        Score reduces based on issue severity:
        - Critical: -25 points
        - High: -15 points
        - Medium: -8 points
        - Low: -3 points
        - Info: -1 point
        """
        score = 100.0

        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                score -= 25
            elif issue.severity == IssueSeverity.HIGH:
                score -= 15
            elif issue.severity == IssueSeverity.MEDIUM:
                score -= 8
            elif issue.severity == IssueSeverity.LOW:
                score -= 3
            elif issue.severity == IssueSeverity.INFO:
                score -= 1

        return max(0.0, score)

    def _generate_summary(self, issues: List, health_score: float) -> str:
        """Generate diagnosis summary."""
        critical_count = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        high_count = len([i for i in issues if i.severity == IssueSeverity.HIGH])

        if critical_count > 0:
            status = "Critical"
        elif high_count > 0:
            status = "Needs Attention"
        elif health_score >= 80:
            status = "Healthy"
        else:
            status = "Fair"

        summary = (
            f"Health Score: {health_score:.1f}/100 ({status}). "
            f"Found {len(issues)} issues: "
            f"{critical_count} critical, {high_count} high priority."
        )

        return summary

    def _generate_recommendations(self, issues: List) -> List:
        """Generate recommendations from issues."""
        from src.meta.diagnoser.schemas.models import Recommendation

        recommendations = []

        for idx, issue in enumerate(issues):
            # Skip INFO level issues
            if issue.severity == IssueSeverity.INFO:
                continue

            rec = Recommendation(
                id=f"rec_{idx}",
                issue_id=issue.id,
                priority=5 - [  # Invert severity to priority
                    IssueSeverity.CRITICAL,
                    IssueSeverity.HIGH,
                    IssueSeverity.MEDIUM,
                    IssueSeverity.LOW,
                ].index(issue.severity),
                action=f"Address: {issue.title}",
                expected_impact="Improve performance",
                effort="Medium",
                implementation=[
                    "Review issue details",
                    "Implement recommended fix",
                    "Monitor results",
                ],
            )
            recommendations.append(rec)

        return recommendations
