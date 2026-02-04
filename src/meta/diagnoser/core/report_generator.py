"""
Report Generator - Formats diagnosis reports in multiple formats.

This module provides the ReportGenerator class that converts DiagnosisReport
objects into various output formats for different use cases.

Key Classes:
    ReportGenerator: Multi-format report generator

Supported Formats:
    - JSON: Machine-readable format for API responses and data storage
    - Markdown: Human-readable format for documentation and chat
    - HTML: Web dashboard format with styling

Key Features:
    - Multi-format output from single report object
    - Localized support (Chinese/English)
    - Severity-based color coding and formatting
    - Comprehensive issue and recommendation rendering

Usage:
    >>> from src.meta.diagnoser.core import ReportGenerator
    >>> generator = ReportGenerator()
    >>>
    >>> # Generate JSON for APIs
    >>> json_report = generator.generate_json(diagnosis_report)
    >>>
    >>> # Generate Markdown for documentation
    >>> md_report = generator.generate_markdown(diagnosis_report)
    >>>
    >>> # Generate HTML for dashboards
    >>> html_report = generator.generate_html(diagnosis_report)

See Also:
    - DiagnosisReport: Report data model
    - Issue: Individual issue representation
    - Recommendation: Action items
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from src.meta.diagnoser.schemas.models import (
    DiagnosisReport,
    Issue,
    IssueSeverity,
    IssueCategory,
    Recommendation,
)


logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate diagnosis reports in various formats.

    Supports:
    - JSON (machine-readable)
    - Markdown (human-readable)
    - HTML (web dashboard)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize report generator."""
        self.config = config or {}

    def generate_json(self, report: DiagnosisReport) -> str:
        """Generate JSON report."""
        return json.dumps(report.to_dict(), indent=2)

    def generate_markdown(self, report: DiagnosisReport) -> str:
        """Generate Markdown report."""
        lines = [
            "# 🩺 DevEase 诊断报告",
            "",
            f"**账户**: {report.account_id}",
            f"**实体类型**: {report.entity_type}",
            f"**实体 ID**: {report.entity_id}",
            f"**生成时间**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]

        # Leakage Score
        lines.extend([
            "## 📊 整体评估",
            "",
            f"**资金泄漏评分**: {report.overall_health_score:.1f}/100",
            "",
            self._get_score_description(report.overall_health_score),
            "",
            f"**总结**: {report.summary}",
            "",
            "---",
            "",
        ])

        # Issues by severity
        lines.extend([
            "## ⚠️ 发现的问题",
            "",
        ])

        for severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH, IssueSeverity.MEDIUM, IssueSeverity.LOW]:
            issues = [i for i in report.issues if i.severity == severity]
            if issues:
                lines.append(f"### {severity.value.title()} Priority ({len(issues)})")
                lines.append("")
                for issue in issues:
                    lines.extend([
                        f"#### {issue.title}",
                        f"",
                        f"**类别**: {issue.category.value}",
                        f"**描述**: {issue.description}",
                        f"**影响实体**: {', '.join(issue.affected_entities)}",
                        f"",
                    ])
                    if issue.metrics:
                        lines.append("**指标**:")
                        for key, value in issue.metrics.items():
                            lines.append(f"  - {key}: {value}")
                        lines.append("")

        # Recommendations
        if report.recommendations:
            lines.extend([
                "---",
                "",
                "## 💡 建议",
                "",
            ])
            for rec in report.recommendations:
                lines.extend([
                    f"### Priority {rec.priority}: {rec.action}",
                    f"",
                    f"**预期影响**: {rec.expected_impact}",
                    f"**工作量**: {rec.effort}",
                    f"",
                ])
                if rec.implementation:
                    lines.append("**实施步骤**:")
                    for step in rec.implementation:
                        lines.append(f"1. {step}")
                    lines.append("")

        return "\n".join(lines)

    def _get_score_description(self, score: float) -> str:
        """Get score description."""
        if score >= 90:
            return "✅ **健康** (Optimal): 运行高效，AI 建议微调。"
        elif score >= 70:
            return "⚠️ **警告** (Warning): 存在明显漏洞，建议部分接管。"
        elif score >= 50:
            return "🔴 **高危** (High Risk): 资金严重泄漏，建议立即开启 AI 自动驾驶。"
        else:
            return "🚨 **严重** (Critical): 紧急需要 AI 完全接管。"

    def generate_html(self, report: DiagnosisReport) -> str:
        """Generate HTML report for web dashboard."""
        # This would integrate with a frontend template
        # For now, return a simple HTML structure
        return f"""
        <div class="diagnosis-report">
            <h1>Devease 诊断报告</h1>
            <div class="score-circle">
                <div class="score-value">{report.overall_health_score:.1f}</div>
                <div class="score-label">资金泄漏评分</div>
            </div>
            <div class="summary">{report.summary}</div>
            <div class="issues">
                {self._generate_issues_html(report.issues)}
            </div>
            <div class="recommendations">
                {self._generate_recommendations_html(report.recommendations)}
            </div>
        </div>
        """

    def _generate_issues_html(self, issues: List[Issue]) -> str:
        """Generate HTML for issues list."""
        if not issues:
            return "<p>未发现问题</p>"

        html = ["<ul class='issues-list'>"]
        for issue in issues:
            html.append(f"""
                <li class='issue {issue.severity.value}'>
                    <h4>{issue.title}</h4>
                    <p>{issue.description}</p>
                    <div class='metrics'>
                        {self._format_metrics_html(issue.metrics)}
                    </div>
                </li>
            """)
        html.append("</ul>")
        return "\n".join(html)

    def _generate_recommendations_html(self, recommendations: List[Recommendation]) -> str:
        """Generate HTML for recommendations."""
        if not recommendations:
            return "<p>暂无建议</p>"

        html = ["<ul class='recommendations-list'>"]
        for rec in recommendations:
            html.append(f"""
                <li class='recommendation priority-{rec.priority}'>
                    <h4>{rec.action}</h4>
                    <p><strong>预期影响</strong>: {rec.expected_impact}</p>
                    <p><strong>工作量</strong>: {rec.effort}</p>
                </li>
            """)
        html.append("</ul>")
        return "\n".join(html)

    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML."""
        if not metrics:
            return ""
        items = [f"<span class='metric'>{k}: {v}</span>" for k, v in metrics.items()]
        return " | ".join(items)
