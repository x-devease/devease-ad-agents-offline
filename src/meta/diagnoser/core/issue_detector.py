"""
Core issue detection abstractions and orchestrator.

This module provides the base abstractions for issue detection in the
diagnoser system, along with the main IssueDetector orchestrator that
coordinates multiple specialized detectors.

Key Classes:
    BaseDetector: Abstract base class for all detector implementations
    IssueDetector: Main orchestrator that coordinates multiple detectors

Key Features:
    - Pluggable detector architecture via BaseDetector
    - Multi-detector orchestration with configurable detector types
    - Unified issue aggregation across different detection strategies
    - Support for both performance and configuration issue detection

Usage:
    >>> from src.meta.diagnoser.core import IssueDetector
    >>> detector = IssueDetector()
    >>> issues = detector.detect_all(data, entity_id="123")
    >>> for issue in issues:
    ...     print(f"{issue.category}: {issue.description}")

Extending:
    To create a custom detector, inherit from BaseDetector:

    >>> class CustomDetector(BaseDetector):
    ...     def detect(self, data, entity_id):
    ...         # Your detection logic here
    ...         return [Issue(...)]

See Also:
    - src.meta.diagnoser.detectors.FatigueDetector: Creative fatigue detection
    - src.meta.diagnoser.detectors.LatencyDetector: Conversion latency detection
    - src.meta.diagnoser.detectors.DarkHoursDetector: Hourly performance issues
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import pandas as pd

from src.meta.diagnoser.schemas.models import Issue, IssueSeverity, IssueCategory


logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Base class for issue detectors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize detector with configuration."""
        self.config = config or {}

    @abstractmethod
    def detect(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> List[Issue]:
        """
        Detect issues in the provided data.

        Args:
            data: Entity performance data
            entity_id: Entity identifier

        Returns:
            List of detected issues
        """
        pass


class IssueDetector:
    """
    Main issue detection orchestrator.

    Coordinates multiple specialized detectors to identify
    performance and configuration issues.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize issue detector."""
        self.config = config or {}
        self.detectors: List[BaseDetector] = []

    def register_detector(self, detector: BaseDetector) -> None:
        """Register a specialized detector."""
        self.detectors.append(detector)
        logger.info(f"Registered detector: {detector.__class__.__name__}")

    def detect_all(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> List[Issue]:
        """
        Run all registered detectors.

        Args:
            data: Entity performance data
            entity_id: Entity identifier

        Returns:
            List of all detected issues
        """
        all_issues = []

        for detector in self.detectors:
            try:
                issues = detector.detect(data, entity_id)
                all_issues.extend(issues)
                logger.info(
                    f"{detector.__class__.__name__}: "
                    f"detected {len(issues)} issues"
                )
            except Exception as e:
                logger.error(
                    f"Error in {detector.__class__.__name__}: {e}",
                    exc_info=True,
                )

        # Sort by severity (critical first)
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
            IssueSeverity.INFO: 4,
        }
        all_issues.sort(key=lambda i: severity_order.get(i.severity, 5))

        return all_issues

    def detect_performance_issues(
        self,
        data: pd.DataFrame,
        entity_id: str,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[Issue]:
        """
        Detect performance-related issues.

        Args:
            data: Entity performance data
            entity_id: Entity identifier
            thresholds: Custom threshold values

        Returns:
            List of performance issues
        """
        from src.meta.diagnoser.detectors.performance_detector import (
            PerformanceDetector,
        )

        detector = PerformanceDetector(config={"thresholds": thresholds})
        return detector.detect(data, entity_id)

    def detect_configuration_issues(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> List[Issue]:
        """
        Detect configuration-related issues.

        Args:
            data: Entity performance data
            entity_id: Entity identifier

        Returns:
            List of configuration issues
        """
        from src.meta.diagnoser.detectors.configuration_detector import (
            ConfigurationDetector,
        )

        detector = ConfigurationDetector()
        return detector.detect(data, entity_id)

    def detect_fatigue_issues(
        self,
        data: pd.DataFrame,
        entity_id: str,
    ) -> List[Issue]:
        """
        Detect ad fatigue issues.

        Args:
            data: Entity performance data
            entity_id: Entity identifier

        Returns:
            List of fatigue issues
        """
        from src.meta.diagnoser.detectors.fatigue_detector import (
            FatigueDetector,
        )

        detector = FatigueDetector()
        return detector.detect(data, entity_id)
