"""Enhanced recommendation output structures with multi-format support.

This module provides structured, actionable recommendations with:
1. Executive summaries (C-Suite view)
2. Prioritized groups (Manager view)
3. Detailed implementation (Designer view)
4. Multi-format output (JSON, HTML, Markdown, Slack, etc.)
"""

# pylint: disable=line-too-long

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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
    CANCELLED = "cancelled"
    VALIDATED = "validated"
    FAILED = "failed"


@dataclass
class ExecutiveSummary:
    """Executive summary for C-Suite stakeholders."""

    overall_health: str  # A-F grade
    improvement_potential: str  # "+35% ROAS"
    quick_wins_available: int
    major_overhauls_needed: int
    implementation_complexity: str  # "low", "medium", "high"
    estimated_total_effort: str  # "8-12 hours"
    expected_completion: str  # "2-3 weeks"
    roi_estimate: float  # 4.2x

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
        }


@dataclass
class ImplementationStep:
    """Single implementation step."""

    step: int
    action: str
    tool: str
    estimated_time: str
    skill_level: str  # "beginner", "intermediate", "advanced"


@dataclass
class RiskFactor:
    """A risk factor for a recommendation."""

    risk: str
    probability: str  # "low", "medium", "high"
    impact: str  # "low", "medium", "high"
    mitigation: str


@dataclass
class AlternativeOption:
    """Alternative recommendation option."""

    value: Any
    expected_lift: float
    confidence: float
    why_consider: str


@dataclass
class Evidence:
    """Statistical and cross-customer evidence."""

    statistical_significance: Dict[str, Any]
    cross_customer_validation: Optional[Dict[str, Any]] = None
    top_performer_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "statistical_significance": self.statistical_significance,
            "cross_customer_validation": self.cross_customer_validation,
            "top_performer_analysis": self.top_performer_analysis,
        }


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
    evidence: Evidence

    # Implementation
    implementation_steps: List[ImplementationStep]
    total_effort: str
    skill_required: str
    tools_needed: List[str]
    dependencies: List[str]
    can_rollback: bool
    rollback_difficulty: str

    # Risk assessment
    risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    potential_side_effects: List[str]
    rollback_plan: str

    # Alternatives
    alternatives: List[AlternativeOption]

    # Validation
    validation: Optional[ValidationInfo] = None

    # Tracking
    tracking: Optional[RecommendationTracking] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.rec_id,
            "group_id": self.group_id,
            "feature": self.feature,
            "category": self.category,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "evidence": self.evidence.to_dict(),
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
                lines.append("")

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
                        lines.append(f"- ⚠️ **{risk_factor.risk}**")
                        lines.append(
                            f"  - Probability: {risk_factor.probability}"
                        )
                        lines.append(f"  - Impact: {risk_factor.impact}")
                        lines.append(
                            f"  - Mitigation: {risk_factor.mitigation}"
                        )
                    lines.append("")

                # Alternatives
                if rec.alternatives:
                    lines.append("**Alternatives:**")
                    for alt in rec.alternatives:
                        lines.append(
                            f"- {alt.value}: +{alt.expected_lift} ROAS "
                            f"({alt.confidence:.0%} confidence) - {alt.why_consider}"
                        )
                    lines.append("")

                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def to_html(self, output: EnhancedRecommendationOutput) -> str:
        """Convert to HTML format.

        Args:
            output: Enhanced recommendation output

        Returns:
            HTML string
        """
        # This would typically use a template engine like Jinja2
        # For now, return a simple HTML structure
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Creative Optimization - {output.creative_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        .grade {{
            font-size: 4rem;
            font-weight: bold;
        }}
        .quick-win {{
            border-left: 4px solid #10B981;
            padding: 1rem;
            margin: 1rem 0;
            background: #F0FDF4;
            border-radius: 4px;
        }}
        .strategic {{
            border-left: 4px solid #F59E0B;
            padding: 1rem;
            margin: 1rem 0;
            background: #FFFBEB;
            border-radius: 4px;
        }}
        .major {{
            border-left: 4px solid #EF4444;
            padding: 1rem;
            margin: 1rem 0;
            background: #FEF2F2;
            border-radius: 4px;
        }}
        .confidence-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        .confidence-high {{
            background: #D1FAE5;
            color: #065F46;
        }}
        .confidence-medium {{
            background: #FEF3C7;
            color: #92400E;
        }}
    </style>
</head>
<body>
    <div class="summary-card">
        <h1>Creative Performance Analysis</h1>
        <div class="grade">{output.summary.overall_health}</div>
        <p>Improvement Potential: <strong>{output.summary.improvement_potential}</strong></p>
        <p>Estimated Effort: <strong>{output.summary.estimated_total_effort}</strong></p>
        <p>ROI: <strong>{output.summary.roi_estimate}x</strong></p>
    </div>
"""
        for group in output.groups:
            css_class = (
                "quick-win"
                if group.priority == RecommendationPriority.QUICK_WIN
                else (
                    "strategic"
                    if group.priority == RecommendationPriority.STRATEGIC
                    else "major"
                )
            )
            html += f"    <h2>{group.name}</h2>\n"
            html += f"    <p>{group.description}</p>\n"
            html += (
                f"    <p><strong>Effort:</strong> {group.estimated_effort} | "
            )
            html += f"<strong>Impact:</strong> {group.expected_impact}</p>\n"

            for rec in group.recommendations:
                confidence = rec.recommended_value.get("confidence", 0)
                conf_class = (
                    "confidence-high"
                    if confidence >= 0.8
                    else "confidence-medium"
                )

                html += f'    <div class="{css_class}">\n'
                html += f"        <h3>{rec.feature}</h3>\n"
                html += (
                    f'        <p><strong>Current:</strong> {rec.current_value.get("value")} → '
                    f"<strong>Recommended:</strong> "
                    f'{rec.recommended_value.get("value")}</p>\n'
                )
                html += (
                    f"        <p><strong>Expected Impact:</strong> "
                    f'{rec.recommended_value.get("expected_lift")} '
                    f'({rec.recommended_value.get("expected_lift_pct")})</p>\n'
                )
                html += (
                    f'        <span class="confidence-badge {conf_class}">'
                    f"{confidence:.0%} Confidence</span>\n"
                )
                html += f"        <p><strong>Effort:</strong> {rec.total_effort}</p>\n"
                html += "    </div>\n"

        html += "</body>\n</html>"
        return html

    def to_slack(self, output: EnhancedRecommendationOutput) -> Dict[str, Any]:
        """Convert to Slack message format.

        Args:
            output: Enhanced recommendation output

        Returns:
            Slack message dictionary
        """
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

        return {"text": "Creative Optimization Report", "blocks": blocks}


def convert_legacy_output(
    legacy_output: Dict[str, Any],
    creative_id: str,
) -> EnhancedRecommendationOutput:
    """Convert legacy output format to enhanced format.

    Args:
        legacy_output: Legacy recommendation output
        creative_id: Creative identifier

    Returns:
        Enhanced recommendation output
    """
    recommendations = legacy_output.get("recommendations", [])
    current_roas = legacy_output.get("current_roas", 0)
    predicted_roas = legacy_output.get("predicted_roas", 0)

    # Calculate executive summary
    improvement_pct = (
        ((predicted_roas - current_roas) / current_roas * 100)
        if current_roas > 0
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
        improvement_potential=f"+{improvement_pct:.0f}% ROAS",
        quick_wins_available=len(
            [r for r in recommendations if r.get("effort") == "low"]
        ),
        major_overhauls_needed=len(
            [r for r in recommendations if r.get("effort") == "high"]
        ),
        implementation_complexity="medium",
        estimated_total_effort="8-12 hours",
        expected_completion="2-3 weeks",
        roi_estimate=4.2,
    )

    # Group recommendations by effort
    quick_wins = [
        r for r in recommendations if r.get("effort") in ["low", "< 1 hour"]
    ]

    # Convert to enhanced format
    groups = []
    rec_counter = 0

    if quick_wins:
        enhanced_recs = []
        for rec in quick_wins[:5]:  # Top 5
            rec_counter += 1
            enhanced = EnhancedRecommendation(
                rec_id=f"rec_{creative_id}_{rec_counter}",
                group_id="quick_wins",
                feature=rec.get("feature", "unknown"),
                category="visual_elements",
                current_value={"value": rec.get("current", "N/A")},
                recommended_value={
                    "value": rec.get("recommended", "N/A"),
                    "expected_lift": rec.get("potential_impact", 0),
                    "confidence": 0.8,
                },
                evidence=Evidence(statistical_significance={}),
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
                expected_impact="+15% ROAS",
                recommendations=enhanced_recs,
            )
        )

    # Add strategic and major groups similarly...

    return EnhancedRecommendationOutput(
        creative_id=creative_id,
        analysis_date=datetime.now().isoformat(),
        version="2.0",
        summary=summary,
        groups=groups,
        total_recommendations=len(recommendations),
        high_confidence_count=len(
            [r for r in recommendations if r.get("confidence") == "high"]
        ),
        quick_wins_count=len(quick_wins),
    )
