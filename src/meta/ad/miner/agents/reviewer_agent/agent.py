"""
Reviewer Agent - Logic Police & Compliance Officer

Objective: Review code changes from Coder Agent for compliance, security,
architecture, and potential issues before Judge Agent evaluation.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
import re
import ast

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of code review."""
    pr_id: str
    decision: str  # "APPROVE", "REQUEST_CHANGES", "REJECT"
    critical_issues: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    architecture_compliance: bool
    security_score: float  # 0.0 - 1.0
    bias_score: float  # 0.0 - 1.0 (lower is better)
    timestamp: str


class ReviewerAgent:
    """
    Logic Police & Compliance Officer Agent.

    Reviews code changes for:
    - Architecture compliance
    - Security vulnerabilities
    - Bias and fairness issues
    - Code quality
    - Test coverage
    - Documentation completeness
    """

    def __init__(
        self,
        memory_agent=None,
        repo_root: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Reviewer Agent.

        Args:
            memory_agent: Memory Agent for learning from past failures
            repo_root: Root directory of the git repository
            config: Agent configuration
        """
        self.memory_agent = memory_agent
        self.config = config or {}

        # Set repository root
        if repo_root is None:
            repo_root = Path.cwd()
        self.repo_root = Path(repo_root)

        # Architecture rules
        self.architecture_rules = {
            "no_hardcoded_paths": True,
            "no_test_data_leakage": True,
            "min_modularity": True,
            "backward_compatibility": True,
            "no_undocumented_changes": True,
        }

        # Security patterns to check
        self.security_patterns = {
            "sql_injection": [
                r'execute\(.+%\s*.',
                r'query\(.+format\(',
                r'cursor\.execute.*%',
            ],
            "command_injection": [
                r'subprocess\.call\(.+shell=True',
                r'os\.system\(',
                r'os\.popen\(',
            ],
            "hardcoded_secrets": [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
            ],
            "insecure_deserialization": [
                r'pickle\.loads?\(',
                r'cPickle\.loads?\(',
            ],
        }

        # Bias detection patterns
        self.bias_patterns = {
            "demographic_bias": [
                r'gender.*=.*["\']male["\']',
                r'race.*=.*["\']white["\']',
            ],
            "geographic_bias": [
                r'country.*=.*["\']US["\']',
                r'location.*=.*["\']USA["\']',
            ],
        }

        # Track review history
        self.review_history: List[ReviewResult] = []

        logger.info("Reviewer Agent: Initialized")

    def review_pull_request(
        self,
        pr: Dict[str, Any],
        experiment_spec: Dict[str, Any],
    ) -> ReviewResult:
        """
        Review a pull request from Coder Agent.

        Args:
            pr: Pull request metadata from Coder Agent
            experiment_spec: Original experiment specification

        Returns:
            ReviewResult: Review decision and feedback
        """
        pr_id = pr.get("pr_id", "unknown")

        logger.info(f"Reviewer Agent: Reviewing PR {pr_id}")
        logger.info(f"  Files to review: {len(pr.get('files_modified', []))}")

        # Initialize review result
        critical_issues = []
        warnings = []
        suggestions = []

        # Step 1: Check for failure patterns from Memory
        if self.memory_agent:
            failure_warnings = self._check_failure_patterns(pr, experiment_spec)
            warnings.extend(failure_warnings)

        # Step 2: Review each modified file
        for file_path in pr.get("files_modified", []):
            file_issues = self._review_file(file_path, experiment_spec)
            critical_issues.extend(file_issues["critical"])
            warnings.extend(file_issues["warning"])
            suggestions.extend(file_issues["suggestion"])

        # Step 3: Check architecture compliance
        architecture_compliance = self._check_architecture_compliance(
            pr,
            experiment_spec,
        )

        # Step 4: Security analysis
        security_score = self._analyze_security(pr)

        # Step 5: Bias detection
        bias_score = self._detect_bias(pr)

        # Step 6: Check test coverage
        test_coverage_check = self._check_test_coverage(pr)
        if not test_coverage_check["adequate"]:
            critical_issues.append({
                "type": "test_coverage",
                "severity": "critical",
                "message": "Insufficient test coverage",
                "details": test_coverage_check,
            })

        # Step 7: Make decision
        decision = self._make_decision(
            critical_issues,
            architecture_compliance,
            security_score,
            bias_score,
        )

        result = ReviewResult(
            pr_id=pr_id,
            decision=decision,
            critical_issues=critical_issues,
            warnings=warnings,
            suggestions=suggestions,
            architecture_compliance=architecture_compliance,
            security_score=security_score,
            bias_score=bias_score,
            timestamp=datetime.now().isoformat(),
        )

        self.review_history.append(result)

        logger.info(f"✓ Review complete: {decision}")
        logger.info(f"  Critical issues: {len(critical_issues)}")
        logger.info(f"  Warnings: {len(warnings)}")
        logger.info(f"  Security score: {security_score:.2f}")
        logger.info(f"  Bias score: {bias_score:.2f}")

        return result

    def _check_failure_patterns(
        self,
        pr: Dict[str, Any],
        experiment_spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check if PR matches historical failure patterns."""

        warnings = []

        if not self.memory_agent:
            return warnings

        # Check for similar failed experiments
        similar_failures = self.memory_agent.search_similar(
            query=f"failed {experiment_spec.get('objective', '')}",
            top_k=3,
        )

        for failure in similar_failures:
            if failure.get("judge_decision") == "FAIL":
                failure_reason = failure.get("failure_reason", "Unknown")

                warnings.append({
                    "type": "historical_failure",
                    "severity": "warning",
                    "message": f"Similar approach failed in past: {failure_reason}",
                    "experiment_id": failure.get("experiment_id"),
                })

        return warnings

    def _review_file(
        self,
        file_path: str,
        experiment_spec: Dict[str, Any],
    ) -> Dict[str, List[Dict]]:
        """Review a single file for issues."""

        issues = {"critical": [], "warning": [], "suggestion": []}

        full_path = self.repo_root / file_path
        if not full_path.exists():
            issues["warning"].append({
                "type": "file_not_found",
                "severity": "warning",
                "message": f"File not found: {file_path}",
            })
            return issues

        try:
            with open(full_path, 'r') as f:
                content = f.read()

            # Syntax check
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues["critical"].append({
                    "type": "syntax_error",
                    "severity": "critical",
                    "message": f"Syntax error in {file_path}: {e}",
                })

            # Security checks
            security_issues = self._check_security_issues(content, file_path)
            issues["critical"].extend(security_issues)

            # Bias checks
            bias_issues = self._check_bias_issues(content, file_path)
            issues["warning"].extend(bias_issues)

            # Code quality checks
            quality_issues = self._check_code_quality(content, file_path)
            issues["suggestion"].extend(quality_issues)

        except Exception as e:
            issues["warning"].append({
                "type": "review_error",
                "severity": "warning",
                "message": f"Failed to review {file_path}: {e}",
            })

        return issues

    def _check_security_issues(self, content: str, file_path: str) -> List[Dict]:
        """Check for security vulnerabilities."""

        issues = []

        for vuln_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append({
                        "type": "security",
                        "severity": "critical",
                        "message": f"Potential {vuln_type} vulnerability in {file_path}",
                        "pattern": pattern,
                    })

        return issues

    def _check_bias_issues(self, content: str, file_path: str) -> List[Dict]:
        """Check for bias and fairness issues."""

        issues = []

        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append({
                        "type": "bias",
                        "severity": "warning",
                        "message": f"Potential {bias_type} bias in {file_path}",
                        "pattern": pattern,
                    })

        return issues

    def _check_code_quality(self, content: str, file_path: str) -> List[Dict]:
        """Check code quality issues."""

        issues = []

        # Check for long lines
        for i, line in enumerate(content.split('\n'), 1):
            if len(line) > 100:
                issues.append({
                    "type": "code_quality",
                    "severity": "suggestion",
                    "message": f"Line too long ({len(line)} chars) in {file_path}:{i}",
                })

        # Check for missing docstrings
        if "def " in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "def " in line and not line.strip().startswith("#"):
                    # Check if next non-empty line is a docstring
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and not next_line.startswith('"""') and not next_line.startswith("'''"):
                            issues.append({
                                "type": "documentation",
                                "severity": "suggestion",
                                "message": f"Missing docstring in {file_path}:{i + 1}",
                            })

        return issues

    def _check_architecture_compliance(
        self,
        pr: Dict[str, Any],
        experiment_spec: Dict[str, Any],
    ) -> bool:
        """Check if changes comply with architecture rules."""

        constraints = experiment_spec.get("constraints", {})

        # Check if modifying only allowed files
        if "allowed_files" in constraints:
            allowed_files = set(constraints["allowed_files"])
            modified_files = set(pr.get("files_modified", []))

            for file_path in modified_files:
                # Check if any allowed file matches
                if not any(allowed in file_path for allowed in allowed_files):
                    logger.warning(f"Architecture violation: {file_path} not in allowed files")
                    return False

        # Check backward compatibility constraint
        if constraints.get("backward_compatibility", True):
            # In production, would run compatibility tests
            pass

        # Check max files constraint
        if "max_files_to_modify" in constraints:
            max_files = constraints["max_files_to_modify"]
            if len(pr.get("files_modified", [])) > max_files:
                logger.warning(f"Architecture violation: Exceeded max files ({max_files})")
                return False

        return True

    def _analyze_security(self, pr: Dict[str, Any]) -> float:
        """
        Analyze security of code changes.

        Returns:
            Security score (0.0 - 1.0, higher is better)
        """

        # Base score
        score = 1.0

        # Deductions for various issues
        # (In production, would do more sophisticated analysis)

        return score

    def _detect_bias(self, pr: Dict[str, Any]) -> float:
        """
        Detect bias in code changes.

        Returns:
            Bias score (0.0 - 1.0, lower is better)
        """

        # In production, would analyze:
        # - Demographic bias
        # - Geographic bias
        # - Cultural bias
        # - Temporal bias

        base_score = 0.0

        return base_score

    def _check_test_coverage(self, pr: Dict[str, Any]) -> Dict[str, Any]:
        """Check if tests are adequate."""

        tests_run = pr.get("tests_run", [])
        tests_passed = pr.get("tests_passed", False)

        return {
            "adequate": len(tests_run) > 0 and tests_passed,
            "tests_run": tests_run,
            "tests_passed": tests_passed,
        }

    def _make_decision(
        self,
        critical_issues: List[Dict],
        architecture_compliance: bool,
        security_score: float,
        bias_score: float,
    ) -> str:
        """Make review decision."""

        # Reject if critical issues
        if critical_issues:
            return "REJECT"

        # Request changes if architecture not compliant
        if not architecture_compliance:
            return "REQUEST_CHANGES"

        # Request changes if security score too low
        if security_score < 0.5:
            return "REQUEST_CHANGES"

        # Approve otherwise
        return "APPROVE"

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "total_reviews": len(self.review_history),
            "approval_rate": sum(1 for r in self.review_history if r.decision == "APPROVE") / len(self.review_history) if self.review_history else 0.0,
            "config": self.config,
        }
