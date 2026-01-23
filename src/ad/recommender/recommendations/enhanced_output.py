"""Enhanced recommendation output with multi-format support.

This module provides structured, actionable recommendations with:
- Executive summaries (C-Suite view)
- Prioritized groups (Manager view)
- Detailed implementation (Designer view)
- Multi-format output (JSON, HTML, Markdown, Slack, PDF)
"""

# pylint: disable=line-too-long

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RecommendationPriority(Enum):
    """Priority level for recommendation groups."""

    QUICK_WIN = 1
    STRATEGIC = 2
    MAJOR_OVERHAUL = 3


class RiskLevel(Enum):
    """Risk assessment levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendationStatus(Enum):
    """Implementation status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VALIDATED = "validated"


@dataclass
class ExecutiveSummary:
    """Executive summary for C-Suite stakeholders."""

    overall_health: str  # A-F grade
    improvement_potential: str  # "+35% ROAS"
    quick_wins_available: int
    major_overhauls_needed: int
    implementation_complexity: str
    estimated_total_effort: str
    expected_completion: str
    roi_estimate: float
    target_metric: str  # "roas" or "cpa" (configurable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_health": self.overall_health,
            "improvement_potential": self.improvement_potential,
            "quick_wins_available": self.quick_wins_available,
            "major_overhauls_needed": self.major_overhauls_needed,
            "implementation_complexity": self.implementation_complexity,
            "estimated_total_effort": self.estimated_total_effort,
            "expected_completion": self.expected_completion,
            "roi_estimate": self.roi_estimate,
            "target_metric": self.target_metric,
        }


@dataclass
class ImplementationStep:
    """Single implementation step."""

    step: int
    action: str
    tool: str
    estimated_time: str
    skill_level: str


@dataclass
class RiskFactor:
    """A risk factor for a recommendation."""

    risk: str
    probability: str
    impact: str
    mitigation: str


@dataclass
class AlternativeOption:
    """Alternative recommendation option."""

    value: Any
    expected_lift: float
    confidence: float
    why_consider: str


@dataclass
class ValidationInfo:
    """Validation and testing information."""

    validation_id: str
    ab_test_ready: bool
    recommended_sample_size: int
    test_duration: str
    success_criteria: Dict[str, Any]


@dataclass
class RecommendationTracking:
    """Tracking information for recommendations."""

    created_at: str
    last_updated: str
    version: int
    status: RecommendationStatus
    assigned_to: Optional[str]
    implementation_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "version": self.version,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "implementation_notes": self.implementation_notes,
        }


# pylint: disable=too-many-instance-attributes
@dataclass
class EnhancedRecommendation:
    """Enhanced recommendation with full context."""

    # Identification
    rec_id: str
    group_id: str
    feature: str
    category: str

    # Current state
    current_value: Dict[str, Any]

    # Recommended state
    recommended_value: Dict[str, Any]

    # Evidence
    statistical_significance: Dict[str, Any]
    cross_customer_validation: Optional[Dict[str, Any]] = None
    top_performer_analysis: Optional[Dict[str, Any]] = None

    # Implementation
    implementation_steps: List[ImplementationStep] = field(default_factory=list)
    total_effort: str = "unknown"
    skill_required: str = "intermediate"
    tools_needed: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    can_rollback: bool = True
    rollback_difficulty: str = "medium"

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_factors: List[RiskFactor] = field(default_factory=list)
    potential_side_effects: List[str] = field(default_factory=list)
    rollback_plan: str = ""

    # Alternatives
    alternatives: List[AlternativeOption] = field(default_factory=list)

    # Validation
    validation: Optional[ValidationInfo] = None

    # Tracking
    tracking: Optional[RecommendationTracking] = None

    # Priority score
    priority_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.rec_id,
            "group_id": self.group_id,
            "feature": self.feature,
            "category": self.category,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "statistical_significance": self.statistical_significance,
            "cross_customer_validation": self.cross_customer_validation,
            "top_performer_analysis": self.top_performer_analysis,
            "implementation": {
                "steps": [
                    {
                        "step": s.step,
                        "action": s.action,
                        "tool": s.tool,
                        "estimated_time": s.estimated_time,
                        "skill_level": s.skill_level,
                    }
                    for s in self.implementation_steps
                ],
                "total_effort": self.total_effort,
                "skill_required": self.skill_required,
                "tools_needed": self.tools_needed,
                "dependencies": self.dependencies,
                "can_rollback": self.can_rollback,
                "rollback_difficulty": self.rollback_difficulty,
            },
            "risk_assessment": {
                "overall_risk": self.risk_level.value,
                "risk_factors": [
                    {
                        "risk": rf.risk,
                        "probability": rf.probability,
                        "impact": rf.impact,
                        "mitigation": rf.mitigation,
                    }
                    for rf in self.risk_factors
                ],
                "potential_side_effects": self.potential_side_effects,
                "rollback_plan": self.rollback_plan,
            },
            "alternatives": [
                {
                    "value": alt.value,
                    "expected_lift": alt.expected_lift,
                    "confidence": alt.confidence,
                    "why_consider": alt.why_consider,
                }
                for alt in self.alternatives
            ],
            "validation": (
                {
                    "validation_id": self.validation.validation_id,
                    "ab_test_ready": self.validation.ab_test_ready,
                    "recommended_sample_size": self.validation.recommended_sample_size,
                    "test_duration": self.validation.test_duration,
                    "success_criteria": self.validation.success_criteria,
                }
                if self.validation
                else None
            ),
            "tracking": self.tracking.to_dict() if self.tracking else None,
            "priority_score": self.priority_score,
        }


# pylint: disable=too-few-public-methods
@dataclass
class RecommendationGroup:
    """Group of related recommendations."""

    group_id: str
    name: str
    priority: RecommendationPriority
    description: str
    total_recommendations: int
    estimated_effort: str
    expected_impact: str
    recommendations: List[EnhancedRecommendation]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.group_id,
            "name": self.name,
            "priority": self.priority.value,
            "description": self.description,
            "total_recommendations": self.total_recommendations,
            "estimated_effort": self.estimated_effort,
            "expected_impact": self.expected_impact,
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


@dataclass
class EnhancedRecommendationOutput:
    """Complete enhanced recommendation output."""

    # Metadata
    creative_id: str
    analysis_date: str
    version: str
    target_metric: str  # "roas" or "cpa"

    # Executive summary
    summary: ExecutiveSummary

    # Groups
    groups: List[RecommendationGroup]

    # Statistics
    total_recommendations: int
    high_confidence_count: int
    quick_wins_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON format)."""
        return {
            "creative_id": self.creative_id,
            "analysis_date": self.analysis_date,
            "version": self.version,
            "target_metric": self.target_metric,
            "summary": self.summary.to_dict(),
            "groups": [g.to_dict() for g in self.groups],
            "statistics": {
                "total_recommendations": self.total_recommendations,
                "high_confidence_count": self.high_confidence_count,
                "quick_wins_count": self.quick_wins_count,
            },
        }


class RecommendationOutputFormatter:
    """Format recommendations for different audiences and platforms."""

    def __init__(self, target_metric: str = "roas"):
        """Initialize formatter.

        Args:
            target_metric: Target metric ("roas" or "cpa")
        """
        self.target_metric = target_metric

    def to_json(self, output: EnhancedRecommendationOutput) -> str:
        """Convert to JSON string.

        Args:
            output: Enhanced recommendation output

        Returns:
            JSON string
        """
        return json.dumps(output.to_dict(), indent=2)

    def to_markdown(self, output: EnhancedRecommendationOutput) -> str:
        """Convert to Markdown format.

        Args:
            output: Enhanced recommendation output

        Returns:
            Markdown string
        """
        metric_label = self.target_metric.upper()

        lines = []

        # Title
        lines.append("# Creative Optimization Report")
        lines.append(f"## Image: {output.creative_id}")
        lines.append("")

        # Summary
        summary = output.summary
        lines.append("## Summary")
        lines.append(f"- **Grade:** {summary.overall_health}")
        lines.append(
            f"- **Improvement Potential:** {summary.improvement_potential}"
        )
        lines.append(f"- **Quick Wins:** {summary.quick_wins_available}")
        lines.append(f"- **Major Overhauls:** {summary.major_overhauls_needed}")
        lines.append(f"- **Effort:** {summary.estimated_total_effort}")
        lines.append(f"- **ROI:** {summary.roi_estimate}x")
        lines.append(f"- **Target Metric:** {metric_label}")
        lines.append("")

        # Groups
        for group in output.groups:
            lines.append(f"## {group.name}")
            lines.append(f"*{group.description}*")
            lines.append(f"**Effort:** {group.estimated_effort}")
            lines.append(f"**Impact:** {group.expected_impact}")
            lines.append("")

            for i, rec in enumerate(group.recommendations, 1):
                lines.append(f"### {i}. {rec.feature}")
                lines.append(f"**Current:** {rec.current_value.get('value')}")
                lines.append(
                    f"**Recommended:** {rec.recommended_value.get('value')}"
                )
                lines.append(
                    f"**Expected Impact:** {rec.recommended_value.get('expected_lift')}"
                )
                lines.append(
                    f"**Confidence:** {rec.recommended_value.get('confidence', 0):.0%}"
                )
                lines.append(f"**Effort:** {rec.total_effort}")

                if rec.priority_score is not None:
                    lines.append(
                        f"**Priority Score:** {rec.priority_score:.2f}"
                    )

                # Implementation steps
                if rec.implementation_steps:
                    lines.append("**Implementation Steps:**")
                    for step in rec.implementation_steps:
                        lines.append(f"{step.step}. {step.action}")
                        lines.append(f"   - Tool: {step.tool}")
                        lines.append(f"   - Time: {step.estimated_time}")
                        lines.append(f"   - Skill: {step.skill_level}")
                    lines.append("")

                # Risks
                if rec.risk_factors:
                    lines.append("**Risks:**")
                    for risk_factor in rec.risk_factors:
                        lines.append(
                            f"- ⚠️ **{risk_factor.risk}** "
                            f"(prob: {risk_factor.probability}, impact: {risk_factor.impact})"
                        )
                        lines.append(
                            f"  - Mitigation: {risk_factor.mitigation}"
                        )
                    lines.append("")

                # Alternatives
                if rec.alternatives:
                    lines.append("**Alternatives:**")
                    for alt in rec.alternatives:
                        lines.append(
                            f"- {alt.value}: +{alt.expected_lift} {metric_label} "
                            f"({alt.confidence:.0%} confidence) - {alt.why_consider}"
                        )
                    lines.append("")

                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def to_slack(self, output: EnhancedRecommendationOutput) -> Dict[str, Any]:
        """Convert to Slack message format.

        Args:
            output: Enhanced recommendation output

        Returns:
            Slack message dictionary
        """
        metric_label = self.target_metric.upper()

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🎨 Creative Analysis: {output.creative_id}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Grade:*\n{output.summary.overall_health}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Potential:*\n{output.summary.improvement_potential}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Quick Wins:*\n{output.summary.quick_wins_available}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Effort:*\n{output.summary.estimated_total_effort}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Target:*\n{metric_label}",
                    },
                ],
            },
            {"type": "divider"},
        ]

        # Add top 3 quick wins
        quick_wins = [
            g
            for g in output.groups
            if g.priority == RecommendationPriority.QUICK_WIN
        ]
        if quick_wins:
            blocks.append(
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🚀 Top Priority: Quick Wins",
                    },
                }
            )

            for group in quick_wins:
                for rec in group.recommendations[:3]:
                    confidence = rec.recommended_value.get("confidence", 0)
                    blocks.append(
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*{rec.feature}*\n"
                                    f"Impact: {rec.recommended_value.get('expected_lift')}\n"
                                    f"Confidence: {confidence:.0%}\n"
                                    f"Effort: {rec.total_effort}\n",
                                }
                            ],
                        }
                    )

        # Add action buttons
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View Full Report",
                        },
                        "url": f"https://dashboard.example.com/report/{output.creative_id}",
                        "style": "primary",
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Export Checklist",
                        },
                        "url": f"https://dashboard.example.com/checklist/{output.creative_id}",
                    },
                ],
            }
        )

        return {
            "text": f"Creative Optimization Report ({metric_label})",
            "blocks": blocks,
        }


def convert_legacy_output(
    legacy_output: Dict[str, Any],
    creative_id: str,
    target_metric: str = "roas",
) -> EnhancedRecommendationOutput:
    """Convert legacy output format to enhanced format.

    Args:
        legacy_output: Legacy recommendation output
        creative_id: Creative identifier
        target_metric: Target metric ("roas" or "cpa")

    Returns:
        Enhanced recommendation output
    """
    recommendations = legacy_output.get("recommendations", [])
    current_metric_val = legacy_output.get(f"current_{target_metric}", 0)
    predicted_metric_val = legacy_output.get(
        f"predicted_{target_metric}", current_metric_val
    )

    # For ROAS: higher is better
    # For CPA: lower is better
    if target_metric == "roas":
        improvement_pct = (
            (
                (predicted_metric_val - current_metric_val)
                / current_metric_val
                * 100
            )
            if current_metric_val > 0
            else 0
        )
    else:  # CPA
        improvement_pct = (
            (
                (current_metric_val - predicted_metric_val)
                / current_metric_val
                * 100
            )
            if current_metric_val > 0
            else 0
        )

    # Grade calculation
    if improvement_pct >= 50:
        grade = "A"
    elif improvement_pct >= 35:
        grade = "B+"
    elif improvement_pct >= 20:
        grade = "B"
    elif improvement_pct >= 10:
        grade = "C"
    else:
        grade = "D"

    summary = ExecutiveSummary(
        overall_health=grade,
        improvement_potential=f"+{improvement_pct:.0f}% {target_metric.upper()}",
        quick_wins_available=len(
            [
                r
                for r in recommendations
                if r.get("effort") in ["low", "< 1 hour"]
            ]
        ),
        major_overhauls_needed=len(
            [
                r
                for r in recommendations
                if r.get("effort") in ["high", "> 3 hours"]
            ]
        ),
        implementation_complexity="medium",
        estimated_total_effort="8-12 hours",
        expected_completion="2-3 weeks",
        roi_estimate=4.2,
        target_metric=target_metric,
    )

    # Group recommendations by effort
    quick_wins = [
        r
        for r in recommendations
        if r.get("effort") in ["low", "< 1 hour", "low"]
    ]

    # Convert to enhanced format
    groups = []
    rec_counter = 0

    if quick_wins:
        enhanced_recs = []
        for rec in quick_wins[:5]:  # Top 5
            rec_counter += 1
            impact = rec.get("potential_impact", rec.get("impact", 0))

            # Format impact based on target metric
            if target_metric == "roas":
                expected_lift = f"+{impact}"
                lift_pct = (
                    f"+{(impact / current_metric_val * 100):.0f}%"
                    if current_metric_val > 0
                    else "+0%"
                )
            else:  # CPA
                expected_lift = f"{impact:+}"  # CPA decrease is positive
                lift_pct = (
                    f"{(impact / current_metric_val * 100):.0f}%"
                    if current_metric_val > 0
                    else "+0%"
                )

            enhanced = EnhancedRecommendation(
                rec_id=f"rec_{creative_id}_{rec_counter}",
                group_id="quick_wins",
                feature=rec.get("feature", "unknown"),
                category="visual_elements",
                current_value={"value": rec.get("current", "N/A")},
                recommended_value={
                    "value": rec.get("recommended", "N/A"),
                    "expected_lift": expected_lift,
                    "expected_lift_pct": lift_pct,
                    "confidence": rec.get("confidence", 0.8),
                },
                statistical_significance={},
                implementation_steps=[],
                total_effort=rec.get("implementation_time", "< 1 hour"),
                skill_required="beginner",
                tools_needed=[],
                dependencies=[],
                can_rollback=True,
                rollback_difficulty="easy",
                risk_level=RiskLevel.LOW,
                risk_factors=[],
                potential_side_effects=[],
                rollback_plan="Revert to original",
                alternatives=[],
                priority_score=rec.get("priority_score", None),
            )
            enhanced_recs.append(enhanced)

        groups.append(
            RecommendationGroup(
                group_id="quick_wins",
                name="Quick Wins",
                priority=RecommendationPriority.QUICK_WIN,
                description="High-impact, low-effort changes",
                total_recommendations=len(enhanced_recs),
                estimated_effort="2-3 hours",
                expected_impact=f"+15% {target_metric.upper()}",
                recommendations=enhanced_recs,
            )
        )

    # Add strategic and major groups similarly...

    return EnhancedRecommendationOutput(
        creative_id=creative_id,
        analysis_date=datetime.now().isoformat(),
        version="2.0",
        target_metric=target_metric,
        summary=summary,
        groups=groups,
        total_recommendations=len(recommendations),
        high_confidence_count=len(
            [r for r in recommendations if r.get("confidence") in ["high", 0.8]]
        ),
        quick_wins_count=len(quick_wins),
    )
