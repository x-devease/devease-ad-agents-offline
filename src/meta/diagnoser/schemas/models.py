"""
Data models for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime


class IssueSeverity(Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(Enum):
    """Categories of issues."""
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    FATIGUE = "fatigue"
    BUDGET = "budget"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"


@dataclass
class Issue:
    """
    Represents a single diagnosed issue.

    Attributes:
        id: Unique issue identifier
        category: Issue category
        severity: Severity level
        title: Issue title
        description: Detailed description
        affected_entities: List of affected entity IDs (campaigns, adsets, ads)
        metrics: Relevant metrics
        detected_at: Detection timestamp
    """
    id: str
    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    affected_entities: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_entities": self.affected_entities,
            "metrics": self.metrics,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class Recommendation:
    """
    Actionable recommendation for an issue.

    Attributes:
        id: Unique recommendation identifier
        issue_id: Related issue ID
        priority: Priority level (1-5, 1=highest)
        action: Recommended action
        expected_impact: Expected impact description
        effort: Effort required (low/medium/high)
        implementation: Implementation steps
    """
    id: str
    issue_id: str
    priority: int
    action: str
    expected_impact: str
    effort: str
    implementation: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "issue_id": self.issue_id,
            "priority": self.priority,
            "action": self.action,
            "expected_impact": self.expected_impact,
            "effort": self.effort,
            "implementation": self.implementation,
        }


@dataclass
class DiagnosisReport:
    """
    Complete diagnosis report for an account or campaign.

    Attributes:
        account_id: Meta account ID
        entity_type: Type of entity (account, campaign, adset)
        entity_id: Entity ID
        issues: List of detected issues
        recommendations: List of recommendations
        summary: Diagnosis summary
        overall_health_score: Overall health score (0-100)
        metadata: Additional metadata
        generated_at: Report generation timestamp
    """
    account_id: str
    entity_type: str
    entity_id: str
    issues: List[Issue] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    summary: str = ""
    overall_health_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def get_issues_by_severity(self, severity: IssueSeverity) -> List[Issue]:
        """Get issues filtered by severity."""
        return [i for i in self.issues if i.severity == severity]

    def get_issues_by_category(self, category: IssueCategory) -> List[Issue]:
        """Get issues filtered by category."""
        return [i for i in self.issues if i.category == category]

    def get_critical_issues(self) -> List[Issue]:
        """Get all critical issues."""
        return self.get_issues_by_severity(IssueSeverity.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "issues": [i.to_dict() for i in self.issues],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "summary": self.summary,
            "overall_health_score": self.overall_health_score,
            "metadata": self.metadata,
            "generated_at": self.generated_at.isoformat(),
        }
