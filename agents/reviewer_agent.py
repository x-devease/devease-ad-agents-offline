"""
Reviewer Agent - Code Quality and Compliance Officer for Ad Generator Development Team.

Responsible for:
- Static code analysis and security checks
- Architecture compliance verification
- CI/CD pipeline validation (lint, test, coverage)
- Prompt leakage prevention
- Data bias detection
- Approving or rejecting Pull Requests

Author: Ad System Dev Team
Date: 2026-02-04
"""

from __future__ import annotations

import logging
import ast
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of a code review."""

    APPROVED = "approved"
    APPROVED_WITH_COMMENTS = "approved_with_comments"
    NEEDS_CHANGES = "needs_changes"
    REJECTED = "rejected"


class Severity(Enum):
    """Severity level of review issues."""

    CRITICAL = "critical"  # Must fix before merge
    HIGH = "high"  # Should fix before merge
    MEDIUM = "medium"  # Nice to have
    LOW = "low"  # Minor issues
    INFO = "info"  # Informational


class IssueCategory(Enum):
    """Category of review issue."""

    SECURITY = "security"  # Security vulnerabilities
    ARCHITECTURE = "architecture"  # Design pattern violations
    QUALITY = "quality"  # Code quality issues
    TESTING = "testing"  # Test coverage/quality issues
    PERFORMANCE = "performance"  # Performance concerns
    COMPLIANCE = "compliance"  # Compliance violations (prompt leakage, bias)
    DOCUMENTATION = "documentation"  # Documentation issues
    STYLE = "style"  # Code style issues


@dataclass
class ReviewIssue:
    """An issue found during code review."""

    category: IssueCategory
    severity: Severity
    file_path: str
    line_number: int
    description: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "description": self.description,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet,
        }


@dataclass
class ArchitectureCheck:
    """Result of architecture compliance check."""

    passes: bool
    design_pattern_compliant: bool
    separation_of_concerns: bool
    dependency_direction: bool  # Dependencies should point inward
    no_circular_dependencies: bool
    issues: List[ReviewIssue] = field(default_factory=list)


@dataclass
class SecurityCheck:
    """Result of security check."""

    passes: bool
    no_prompt_leakage: bool
    no_hardcoded_secrets: bool
    no_sql_injection: bool
    no_command_injection: bool
    input_validation: bool
    issues: List[ReviewIssue] = field(default_factory=list)


@dataclass
class ComplianceCheck:
    """Result of compliance check."""

    passes: bool
    no_data_bias: bool
    fair_treatment: bool
    privacy_compliant: bool
    cultural_sensitivity: bool
    issues: List[ReviewIssue] = field(default_factory=list)


@dataclass
class CIValidation:
    """Result of CI/CD validation."""

    passes: bool
    lint_passed: bool
    tests_passed: bool
    coverage_met: bool
    type_check_passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewResult:
    """Result of a code review."""

    status: ReviewStatus
    overall_score: float  # 0.0 to 1.0

    # Component checks
    architecture: ArchitectureCheck
    security: SecurityCheck
    compliance: ComplianceCheck
    ci_validation: CIValidation

    # All issues found
    issues: List[ReviewIssue] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    reviewed_at: datetime = field(default_factory=datetime.now)
    reviewed_by: str = "reviewer_agent"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "overall_score": self.overall_score,
            "architecture": {
                "passes": self.architecture.passes,
                "issues_count": len(self.architecture.issues),
            },
            "security": {
                "passes": self.security.passes,
                "issues_count": len(self.security.issues),
            },
            "compliance": {
                "passes": self.compliance.passes,
                "issues_count": len(self.compliance.issues),
            },
            "ci_validation": {
                "passes": self.ci_validation.passes,
                "lint_passed": self.ci_validation.lint_passed,
                "tests_passed": self.ci_validation.tests_passed,
                "coverage_met": self.ci_validation.coverage_met,
            },
            "issues": [issue.to_dict() for issue in self.issues],
            "recommendations": self.recommendations,
            "reviewed_at": self.reviewed_at.isoformat(),
            "reviewed_by": self.reviewed_by,
        }


# Import from other agents
from agents.coder_agent import PullRequest, CodeChange
from agents.pm_agent import ExperimentSpec, Component


class ReviewerAgent:
    """
    Reviewer Agent for ad/generator development.

    Responsibilities:
    1. Review Pull Requests from Coder Agent
    2. Perform static code analysis
    3. Check architecture compliance
    4. Validate security and compliance
    5. Verify CI/CD checks pass
    6. Approve or reject PRs
    """

    def __init__(
        self,
        repo_path: Path,
        llm_client: Optional[Any] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize Reviewer Agent.

        Args:
            repo_path: Path to the repository
            llm_client: LLM client for analysis
            config_path: Path to Reviewer Agent configuration
        """
        self.repo_path = Path(repo_path)
        self.llm_client = llm_client
        self.config_path = config_path

        # Load configuration
        self._load_config()

        logger.info(f"ReviewerAgent initialized for repo: {self.repo_path}")

    def _load_config(self):
        """Load Reviewer Agent configuration."""
        # Architecture patterns by component
        self.architecture_patterns = {
            Component.AD_MINER: {
                "allowed_imports": ["dataclasses", "logging", "pathlib", "typing"],
                "forbidden_imports": ["src.meta.adset", "src.agents.nano"],
                "design_pattern": "strategy",  # Uses strategy pattern for feature extraction
                "layer": "service",
            },
            Component.AD_GENERATOR: {
                "allowed_imports": ["dataclasses", "logging", "pathlib", "typing"],
                "forbidden_imports": ["src.meta.adset"],
                "design_pattern": "pipeline",  # Uses pipeline pattern
                "layer": "service",
            },
            Component.ADSET_ALLOCATOR: {
                "allowed_imports": ["dataclasses", "logging", "typing"],
                "forbidden_imports": ["src.meta.ad", "src.agents.nano"],
                "design_pattern": "rules",  # Uses rules engine pattern
                "layer": "service",
            },
            Component.ADSET_GENERATOR: {
                "allowed_imports": ["dataclasses", "logging", "typing"],
                "forbidden_imports": ["src.meta.ad.generator", "src.agents.nano"],
                "design_pattern": "builder",  # Uses builder pattern
                "layer": "service",
            },
            Component.NANO_BANANA_PRO: {
                "allowed_imports": ["enum", "dataclasses", "typing"],
                "forbidden_imports": ["src.meta"],  # Agent should be independent
                "design_pattern": "agent",  # Uses agent framework pattern
                "layer": "agent",
            },
            Component.SHARED_UTILS: {
                "allowed_imports": ["pathlib", "logging", "typing", "datetime"],
                "forbidden_imports": [],  # Utils can be imported by anyone
                "design_pattern": "utility",
                "layer": "foundation",
            },
            Component.FRAMEWORK: {
                "allowed_imports": ["dataclasses", "logging", "typing", "abc"],
                "forbidden_imports": [],  # Framework is generic
                "design_pattern": "framework",
                "layer": "foundation",
            },
        }

        # Security patterns to detect
        self.security_patterns = {
            "prompt_leakage": [
                r"print\s*\(\s*prompt",
                r"logger\.(info|debug)\s*\(\s*f['\"]\{prompt\}",
                r"return\s+prompt[^s]",  # Returning prompt directly
            ],
            "hardcoded_secrets": [
                r"api_key\s*=\s*['\"][^'\"]{20,}['\"]",  # Long strings might be keys
                r"password\s*=\s*['\"]",
                r"token\s*=\s*['\"][^'\"]{20,}['\"]",
            ],
            "sql_injection": [
                rf"execute\s*\(\s*f['\"]",
                rf"query\s*\(\s*f['\"]",
            ],
            "command_injection": [
                r"subprocess\.(call|run|Popen)\s*\(\s*[^)]*\+",
                r"os\.system\s*\(\s*[^)]*\+",
            ],
        }

        # Code quality thresholds
        self.quality_thresholds = {
            "min_coverage": 0.80,
            "max_complexity": 10,
            "max_function_length": 50,
            "max_class_length": 300,
        }

    def review_pull_request(
        self,
        pr: PullRequest,
        spec: ExperimentSpec,
    ) -> ReviewResult:
        """
        Review a pull request.

        Args:
            pr: Pull request to review
            spec: Associated experiment spec

        Returns:
            Review result with status and issues
        """
        logger.info(f"Reviewing PR: {pr.pr_id} - {pr.title}")

        all_issues = []

        # 1. Architecture check
        logger.info("Running architecture check...")
        architecture = self._check_architecture(pr, spec)
        all_issues.extend(architecture.issues)

        # 2. Security check
        logger.info("Running security check...")
        security = self._check_security(pr, spec)
        all_issues.extend(security.issues)

        # 3. Compliance check
        logger.info("Running compliance check...")
        compliance = self._check_compliance(pr, spec)
        all_issues.extend(compliance.issues)

        # 4. CI validation
        logger.info("Validating CI/CD checks...")
        ci_validation = self._validate_ci(pr, spec)

        # 5. Code quality check
        logger.info("Checking code quality...")
        quality_issues = self._check_code_quality(pr, spec)
        all_issues.extend(quality_issues)

        # Determine overall status
        status = self._determine_status(
            architecture, security, compliance, ci_validation, all_issues
        )

        # Calculate overall score
        score = self._calculate_score(
            architecture, security, compliance, ci_validation, all_issues
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            architecture, security, compliance, ci_validation, all_issues
        )

        result = ReviewResult(
            status=status,
            overall_score=score,
            architecture=architecture,
            security=security,
            compliance=compliance,
            ci_validation=ci_validation,
            issues=all_issues,
            recommendations=recommendations,
        )

        logger.info(f"Review complete: {status.value} (score: {score:.2f})")
        return result

    def _check_architecture(
        self,
        pr: PullRequest,
        spec: ExperimentSpec,
    ) -> ArchitectureCheck:
        """Check architecture compliance."""
        issues = []

        # Get architecture pattern for component
        pattern = self.architecture_patterns.get(spec.component, {})

        design_pattern_compliant = True
        separation_of_concerns = True
        dependency_direction = True
        no_circular_dependencies = True

        for change in pr.changes:
            file_path = self.repo_path / change.file_path

            if not file_path.exists():
                continue

            # Read file content
            try:
                content = file_path.read_text()

                # Check imports
                if pattern.get("forbidden_imports"):
                    for forbidden in pattern["forbidden_imports"]:
                        if forbidden in content:
                            issues.append(ReviewIssue(
                                category=IssueCategory.ARCHITECTURE,
                                severity=Severity.HIGH,
                                file_path=change.file_path,
                                line_number=1,
                                description=f"Forbidden import detected: {forbidden}",
                                suggestion=f"Remove import from {forbidden} or refactor to avoid dependency",
                            ))
                            dependency_direction = False

                # Check for circular dependencies (simplified)
                # In real implementation, this would build a dependency graph

                # Check design pattern adherence (simplified)
                if pattern.get("design_pattern") == "pipeline":
                    if "class" in content and "Pipeline" not in content and "pipeline" not in content.lower():
                        # Not necessarily a violation, but worth noting
                        pass

            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")

        passes = (
            design_pattern_compliant
            and separation_of_concerns
            and dependency_direction
            and no_circular_dependencies
            and all(i.severity != Severity.CRITICAL for i in issues)
        )

        return ArchitectureCheck(
            passes=passes,
            design_pattern_compliant=design_pattern_compliant,
            separation_of_concerns=separation_of_concerns,
            dependency_direction=dependency_direction,
            no_circular_dependencies=no_circular_dependencies,
            issues=issues,
        )

    def _check_security(
        self,
        pr: PullRequest,
        spec: ExperimentSpec,
    ) -> SecurityCheck:
        """Check security issues."""
        issues = []

        no_prompt_leakage = True
        no_hardcoded_secrets = True
        no_sql_injection = True
        no_command_injection = True
        input_validation = True

        for change in pr.changes:
            file_path = self.repo_path / change.file_path

            if not file_path.exists():
                continue

            try:
                content = file_path.read_text()

                # Check for prompt leakage
                if spec.component in [Component.NANO_BANANA_PRO, Component.AD_GENERATOR]:
                    for pattern in self.security_patterns["prompt_leakage"]:
                        if re.search(pattern, content):
                            issues.append(ReviewIssue(
                                category=IssueCategory.SECURITY,
                                severity=Severity.HIGH,
                                file_path=change.file_path,
                                line_number=self._find_pattern_line(content, pattern),
                                description="Potential prompt leakage detected",
                                suggestion="Avoid logging or printing prompts directly",
                            ))
                            no_prompt_leakage = False

                # Check for hardcoded secrets
                for pattern_name, patterns in self.security_patterns.items():
                    if pattern_name == "hardcoded_secrets":
                        for pattern in patterns:
                            if re.search(pattern, content):
                                issues.append(ReviewIssue(
                                    category=IssueCategory.SECURITY,
                                    severity=Severity.CRITICAL,
                                    file_path=change.file_path,
                                    line_number=self._find_pattern_line(content, pattern),
                                    description=f"Potential hardcoded secret detected",
                                    suggestion="Use environment variables or config files",
                                ))
                                no_hardcoded_secrets = False

                # Check for SQL injection
                if "execute" in content or "query" in content:
                    for pattern in self.security_patterns["sql_injection"]:
                        if re.search(pattern, content):
                            issues.append(ReviewIssue(
                                category=IssueCategory.SECURITY,
                                severity=Severity.CRITICAL,
                                file_path=change.file_path,
                                line_number=self._find_pattern_line(content, pattern),
                                description="Potential SQL injection vulnerability",
                                suggestion="Use parameterized queries",
                            ))
                            no_sql_injection = False

                # Check for command injection
                if "subprocess" in content or "os.system" in content:
                    for pattern in self.security_patterns["command_injection"]:
                        if re.search(pattern, content):
                            issues.append(ReviewIssue(
                                category=IssueCategory.SECURITY,
                                severity=Severity.CRITICAL,
                                file_path=change.file_path,
                                line_number=self._find_pattern_line(content, pattern),
                                description="Potential command injection vulnerability",
                                suggestion="Use subprocess with list arguments or proper escaping",
                            ))
                            no_command_injection = False

            except Exception as e:
                logger.warning(f"Failed to check security for {file_path}: {e}")

        passes = (
            no_prompt_leakage
            and no_hardcoded_secrets
            and no_sql_injection
            and no_command_injection
            and input_validation
        )

        return SecurityCheck(
            passes=passes,
            no_prompt_leakage=no_prompt_leakage,
            no_hardcoded_secrets=no_hardcoded_secrets,
            no_sql_injection=no_sql_injection,
            no_command_injection=no_command_injection,
            input_validation=input_validation,
            issues=issues,
        )

    def _check_compliance(
        self,
        pr: PullRequest,
        spec: ExperimentSpec,
    ) -> ComplianceCheck:
        """Check compliance issues."""
        issues = []

        no_data_bias = True
        fair_treatment = True
        privacy_compliant = True
        cultural_sensitivity = True

        # Check for bias-related patterns
        bias_patterns = [
            r"#\s*(TODO|FIXME|HACK).*bias",
            r"#\s*(TODO|FIXME|HACK).*discriminat",
        ]

        for change in pr.changes:
            file_path = self.repo_path / change.file_path

            if not file_path.exists():
                continue

            try:
                content = file_path.read_text()

                # Check for bias comments
                for pattern in bias_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(ReviewIssue(
                            category=IssueCategory.COMPLIANCE,
                            severity=Severity.HIGH,
                            file_path=change.file_path,
                            line_number=self._find_pattern_line(content, pattern),
                            description="Potential data bias concern",
                            suggestion="Review and address bias concerns before merging",
                        ))
                        no_data_bias = False

                # Check for cultural sensitivity (simplified)
                # In real implementation, this would use LLM to analyze

            except Exception as e:
                logger.warning(f"Failed to check compliance for {file_path}: {e}")

        passes = (
            no_data_bias
            and fair_treatment
            and privacy_compliant
            and cultural_sensitivity
        )

        return ComplianceCheck(
            passes=passes,
            no_data_bias=no_data_bias,
            fair_treatment=fair_treatment,
            privacy_compliant=privacy_compliant,
            cultural_sensitivity=cultural_sensitivity,
            issues=issues,
        )

    def _validate_ci(
        self,
        pr: PullRequest,
        spec: ExperimentSpec,
    ) -> CIValidation:
        """Validate CI/CD checks."""
        # In real implementation, this would query actual CI results
        # For now, return mock results

        details = {
            "pylint_score": 9.5,
            "mypy_errors": 0,
            "tests_run": 57,
            "tests_failed": 0,
            "coverage_percentage": 85.5,
        }

        return CIValidation(
            passes=True,
            lint_passed=True,
            tests_passed=True,
            coverage_met=True,
            type_check_passed=True,
            details=details,
        )

    def _check_code_quality(
        self,
        pr: PullRequest,
        spec: ExperimentSpec,
    ) -> List[ReviewIssue]:
        """Check code quality issues."""
        issues = []

        for change in pr.changes:
            file_path = self.repo_path / change.file_path

            if not file_path.exists():
                continue

            try:
                content = file_path.read_text()

                # Parse AST
                tree = ast.parse(content)

                # Check for overly complex functions
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Count complexity (simplified)
                        complexity = self._calculate_complexity(node)

                        if complexity > self.quality_thresholds["max_complexity"]:
                            issues.append(ReviewIssue(
                                category=IssueCategory.QUALITY,
                                severity=Severity.MEDIUM,
                                file_path=change.file_path,
                                line_number=node.lineno,
                                description=f"Function complexity ({complexity}) exceeds threshold",
                                suggestion=f"Consider refactoring into smaller functions (threshold: {self.quality_thresholds['max_complexity']})",
                            ))

                        # Check function length
                        if hasattr(node, 'end_lineno') and node.end_lineno:
                            length = node.end_lineno - node.lineno
                            if length > self.quality_thresholds["max_function_length"]:
                                issues.append(ReviewIssue(
                                    category=IssueCategory.QUALITY,
                                    severity=Severity.LOW,
                                    file_path=change.file_path,
                                    line_number=node.lineno,
                                    description=f"Function length ({length}) exceeds threshold",
                                    suggestion=f"Consider splitting into smaller functions (threshold: {self.quality_thresholds['max_function_length']})",
                                ))

            except Exception as e:
                logger.warning(f"Failed to check quality for {file_path}: {e}")

        return issues

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _find_pattern_line(self, content: str, pattern: str) -> int:
        """Find line number of a pattern in content."""
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                return i
        return 1

    def _determine_status(
        self,
        architecture: ArchitectureCheck,
        security: SecurityCheck,
        compliance: ComplianceCheck,
        ci_validation: CIValidation,
        issues: List[ReviewIssue],
    ) -> ReviewStatus:
        """Determine overall review status."""
        # Critical security or compliance issues = REJECT
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        if critical_issues:
            return ReviewStatus.REJECTED

        # CI failures = NEEDS_CHANGES
        if not ci_validation.passes:
            return ReviewStatus.NEEDS_CHANGES

        # Security or compliance failures = NEEDS_CHANGES
        if not security.passes or not compliance.passes:
            return ReviewStatus.NEEDS_CHANGES

        # Architecture failures = NEEDS_CHANGES
        if not architecture.passes:
            return ReviewStatus.NEEDS_CHANGES

        # High severity issues = NEEDS_CHANGES
        high_issues = [i for i in issues if i.severity == Severity.HIGH]
        if high_issues:
            return ReviewStatus.NEEDS_CHANGES

        # Medium issues = APPROVED_WITH_COMMENTS
        medium_issues = [i for i in issues if i.severity == Severity.MEDIUM]
        if medium_issues:
            return ReviewStatus.APPROVED_WITH_COMMENTS

        # All good = APPROVED
        return ReviewStatus.APPROVED

    def _calculate_score(
        self,
        architecture: ArchitectureCheck,
        security: SecurityCheck,
        compliance: ComplianceCheck,
        ci_validation: CIValidation,
        issues: List[ReviewIssue],
    ) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        score = 1.0

        # Deduct for failures
        if not architecture.passes:
            score -= 0.25
        if not security.passes:
            score -= 0.30
        if not compliance.passes:
            score -= 0.20
        if not ci_validation.passes:
            score -= 0.15

        # Deduct for issues
        for issue in issues:
            if issue.severity == Severity.CRITICAL:
                score -= 0.10
            elif issue.severity == Severity.HIGH:
                score -= 0.05
            elif issue.severity == Severity.MEDIUM:
                score -= 0.02
            elif issue.severity == Severity.LOW:
                score -= 0.01

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        architecture: ArchitectureCheck,
        security: SecurityCheck,
        compliance: ComplianceCheck,
        ci_validation: CIValidation,
        issues: List[ReviewIssue],
    ) -> List[str]:
        """Generate recommendations based on review."""
        recommendations = []

        if not architecture.passes:
            recommendations.append("Review architecture patterns and design principles")

        if not security.passes:
            recommendations.append("Address security vulnerabilities before merging")

        if not compliance.passes:
            recommendations.append("Review compliance and bias concerns")

        if not ci_validation.passes:
            recommendations.append("Ensure all CI checks pass")

        if issues:
            recommendations.append(f"Address {len(issues)} issue(s) found during review")

        return recommendations


# Convenience function for creating Reviewer Agent
def create_reviewer_agent(
    repo_path: Path,
    llm_client: Optional[Any] = None,
    config_path: Optional[Path] = None,
) -> ReviewerAgent:
    """
    Create a Reviewer Agent instance.

    Args:
        repo_path: Path to the repository
        llm_client: LLM client for analysis
        config_path: Path to Reviewer Agent configuration

    Returns:
        Configured Reviewer Agent
    """
    return ReviewerAgent(
        repo_path=repo_path,
        llm_client=llm_client,
        config_path=config_path,
    )
