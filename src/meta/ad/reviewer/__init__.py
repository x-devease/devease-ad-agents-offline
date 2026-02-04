"""
Ad Reviewer Module - Visual QA & Risk Matrix (4-Guard System).

This module provides quality assurance for generated ad images through
four sequential guards:
1. GeometricGuard - Product integrity validation
2. AestheticGuard - Visual quality checking
3. CulturalGuard - Regional compliance checking
4. PerformanceGuard - Optimization scoring

Usage:
    from src.meta.ad.reviewer import VisualQAMatrix

    # Initialize reviewer with config
    reviewer = VisualQAMatrix(config_path="config/moprobo/facebook/config.yaml")

    # Audit a generator session
    reports = reviewer.audit_session(session_path="path/to/session.json")

    # Check results
    for report in reports:
        if report.status == GuardStatus.PASS:
            print(f"PASS: Score {report.performance_score}")
        else:
            print(f"FAIL: {report.fail_reason}")
"""

from .pipeline import VisualQAMatrix
from .schemas.audit_report import (
    AuditReport,
    GuardStatus,
    RiskLevel,
    GuardResult,
    GeometricResult,
    AestheticResult,
    CulturalResult,
    PerformanceScore
)

__all__ = [
    # Main API
    "VisualQAMatrix",

    # Result types
    "AuditReport",
    "GuardStatus",
    "RiskLevel",
    "GuardResult",
    "GeometricResult",
    "AestheticResult",
    "CulturalResult",
    "PerformanceScore",
]
