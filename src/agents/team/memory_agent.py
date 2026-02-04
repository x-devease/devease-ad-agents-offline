"""
Memory Agent - Organizational Memory and Knowledge Base for Ad Generator Development Team.

Responsible for:
- Recording all experiment inputs, processes, and results
- Retrieving relevant historical experiments for PM Agent
- Detecting repeated failure patterns
- Storing Judge Agent scores and Reviewer feedback
- Providing context for Coder Agent when stuck
- Warning when current evolution path overlaps with historical failures

Author: Ad System Dev Team
Date: 2026-02-04
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib

logger = logging.getLogger(__name__)


class ExperimentOutcome(Enum):
    """Outcome of an experiment."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Some improvements but with issues
    CANCELLED = "cancelled"


class FailureReason(Enum):
    """Reason for experiment failure."""

    TESTS_FAILED = "tests_failed"
    REGRESSION = "regression"
    SECURITY_ISSUE = "security_issue"
    COMPLIANCE_ISSUE = "compliance_issue"
    POOR_PERFORMANCE = "poor_performance"
    ARCHITECTURE_VIOLATION = "architecture_violation"
    SIDE_EFFECTS = "side_effects"
    OTHER = "other"


@dataclass
class ExperimentRecord:
    """Record of a complete experiment."""

    # Identification
    experiment_id: str
    spec_id: str  # Reference to ExperimentSpec
    pr_id: str  # Reference to PullRequest

    # Metadata
    component: str
    title: str
    description: str

    # Timeline
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Input
    problem_statement: str = ""
    success_criteria: List[str] = field(default_factory=list)
    scope: str = ""
    affected_modules: List[str] = field(default_factory=list)

    # Process
    changes_made: List[Dict[str, Any]] = field(default_factory=list)
    reviewer_comments: List[str] = field(default_factory=list)
    judge_issues: List[str] = field(default_factory=list)

    # Outcome
    outcome: Optional[ExperimentOutcome] = None
    failure_reasons: List[FailureReason] = field(default_factory=list)

    # Metrics
    lift_score: float = 0.0
    confidence: float = 0.0
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)

    # Files changed
    files_changed: List[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0

    # Learnings
    lessons_learned: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "spec_id": self.spec_id,
            "pr_id": self.pr_id,
            "component": self.component,
            "title": self.title,
            "description": self.description,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "problem_statement": self.problem_statement,
            "success_criteria": self.success_criteria,
            "scope": self.scope,
            "affected_modules": self.affected_modules,
            "changes_made": self.changes_made,
            "reviewer_comments": self.reviewer_comments,
            "judge_issues": self.judge_issues,
            "outcome": self.outcome.value if self.outcome else None,
            "failure_reasons": [r.value for r in self.failure_reasons],
            "lift_score": self.lift_score,
            "confidence": self.confidence,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "files_changed": self.files_changed,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "lessons_learned": self.lessons_learned,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        """Create from dictionary."""
        return cls(
            experiment_id=data["experiment_id"],
            spec_id=data["spec_id"],
            pr_id=data["pr_id"],
            component=data["component"],
            title=data["title"],
            description=data["description"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            problem_statement=data.get("problem_statement", ""),
            success_criteria=data.get("success_criteria", []),
            scope=data.get("scope", ""),
            affected_modules=data.get("affected_modules", []),
            changes_made=data.get("changes_made", []),
            reviewer_comments=data.get("reviewer_comments", []),
            judge_issues=data.get("judge_issues", []),
            outcome=ExperimentOutcome(data["outcome"]) if data.get("outcome") else None,
            failure_reasons=[FailureReason(r) for r in data.get("failure_reasons", [])],
            lift_score=data.get("lift_score", 0.0),
            confidence=data.get("confidence", 0.0),
            metrics_before=data.get("metrics_before", {}),
            metrics_after=data.get("metrics_after", {}),
            files_changed=data.get("files_changed", []),
            lines_added=data.get("lines_added", 0),
            lines_removed=data.get("lines_removed", 0),
            lessons_learned=data.get("lessons_learned", []),
            tags=data.get("tags", []),
        )


@dataclass
class MemoryQuery:
    """Query for memory retrieval."""

    component: Optional[str] = None
    outcome: Optional[ExperimentOutcome] = None
    tags: List[str] = field(default_factory=list)
    text_search: Optional[str] = None  # Semantic search in description
    file_patterns: List[str] = field(default_factory=list)  # Files affected
    min_lift_score: Optional[float] = None
    max_lift_score: Optional[float] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    limit: int = 10


@dataclass
class HistoricalContext:
    """Context retrieved from memory for decision making."""

    related_experiments: List[ExperimentRecord]
    similar_failures: List[ExperimentRecord]
    similar_successes: List[ExperimentRecord]

    # Patterns detected
    failure_patterns: List[str] = field(default_factory=list)
    success_patterns: List[str] = field(default_factory=list)

    # Recommendations
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# Import from other agents
from agents.team.pm_agent import ExperimentSpec, Component
from agents.team.coder_agent import PullRequest
from agents.team.reviewer_agent import ReviewResult
from agents.team.judge_agent import JudgeDecision, PerformanceReport


class MemoryAgent:
    """
    Memory Agent for ad/generator development.

    Responsibilities:
    1. Store all experiment records
    2. Retrieve relevant historical context
    3. Detect repeated failure patterns
    4. Warn about overlapping failure paths
    5. Provide learnings and suggestions
    """

    def __init__(
        self,
        memory_db_path: Optional[Path] = None,
        max_records: int = 10000,
    ):
        """
        Initialize Memory Agent.

        Args:
            memory_db_path: Path to memory database file
            max_records: Maximum number of records to keep
        """
        self.memory_db_path = Path(memory_db_path) if memory_db_path else Path("data/agents/memory.json")
        self.max_records = max_records

        # Create parent directory if needed
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing records
        self.records: List[ExperimentRecord] = self._load_records()

        logger.info(f"MemoryAgent initialized with {len(self.records)} records")

    def _load_records(self) -> List[ExperimentRecord]:
        """Load records from disk."""
        if not self.memory_db_path.exists():
            return []

        try:
            data = json.loads(self.memory_db_path.read_text())
            return [ExperimentRecord.from_dict(r) for r in data]
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return []

    def _save_records(self):
        """Save records to disk."""
        try:
            data = [r.to_dict() for r in self.records]
            self.memory_db_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def add_experiment(
        self,
        spec: ExperimentSpec,
        pr: PullRequest,
        review_result: ReviewResult,
        judge_decision: JudgeDecision,
    ) -> ExperimentRecord:
        """
        Add a complete experiment record to memory.

        Args:
            spec: Experiment specification
            pr: Pull request
            review_result: Review result
            judge_decision: Judge decision

        Returns:
            Created experiment record
        """
        # Generate experiment ID
        experiment_id = f"exp-{hashlib.md5(spec.spec_id.encode()).hexdigest()[:12]}"

        # Determine outcome
        if judge_decision.approve:
            if review_result.overall_score >= 0.9:
                outcome = ExperimentOutcome.SUCCESS
            else:
                outcome = ExperimentOutcome.PARTIAL
        else:
            outcome = ExperimentOutcome.FAILURE

        # Determine failure reasons
        failure_reasons = []
        if not judge_decision.approve:
            if judge_decision.backtest_result and judge_decision.backtest_result.failed_tests > 0:
                failure_reasons.append(FailureReason.TESTS_FAILED)
            if judge_decision.performance_report:
                if judge_decision.performance_report.regressions:
                    failure_reasons.append(FailureReason.REGRESSION)
            if not review_result.security.passes:
                failure_reasons.append(FailureReason.SECURITY_ISSUE)
            if not review_result.compliance.passes:
                failure_reasons.append(FailureReason.COMPLIANCE_ISSUE)
            if not review_result.architecture.passes:
                failure_reasons.append(FailureReason.ARCHITECTURE_VIOLATION)

        # Extract metrics
        metrics_before = {}
        metrics_after = {}
        lift_score = 0.0
        confidence = judge_decision.confidence

        if judge_decision.performance_report:
            for metric in judge_decision.performance_report.metrics:
                if metric.baseline_value is not None:
                    metrics_before[metric.metric_name] = metric.baseline_value
                if metric.value is not None:
                    metrics_after[metric.metric_name] = metric.value
            lift_score = judge_decision.performance_report.lift_score

        # Extract files changed
        files_changed = [c.file_path for c in pr.changes]

        # Extract reviewer comments
        reviewer_comments = [
            issue.description
            for issue in review_result.issues
            if issue.severity.value in ["critical", "high"]
        ]

        # Extract judge issues
        judge_issues = []
        if judge_decision.performance_report:
            judge_issues.extend(judge_decision.performance_report.regressions)
            judge_issues.extend(judge_decision.performance_report.side_effects)

        # Generate lessons learned
        lessons_learned = self._extract_lessons_learned(
            spec, review_result, judge_decision
        )

        # Generate tags
        tags = self._generate_tags(spec, review_result, judge_decision)

        # Create record
        record = ExperimentRecord(
            experiment_id=experiment_id,
            spec_id=spec.spec_id,
            pr_id=pr.pr_id,
            component=spec.component.value,
            title=spec.title,
            description=spec.description,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            problem_statement=spec.problem_statement,
            success_criteria=spec.success_criteria,
            scope=spec.scope.value,
            affected_modules=spec.affected_modules,
            changes_made=[c.to_dict() for c in pr.changes],
            reviewer_comments=reviewer_comments,
            judge_issues=judge_issues,
            outcome=outcome,
            failure_reasons=failure_reasons,
            lift_score=lift_score,
            confidence=confidence,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            files_changed=files_changed,
            lines_added=sum(c.lines_added for c in pr.changes),
            lines_removed=sum(c.lines_removed for c in pr.changes),
            lessons_learned=lessons_learned,
            tags=tags,
        )

        # Add to memory
        self.records.append(record)

        # Trim if too many records
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]

        # Save to disk
        self._save_records()

        logger.info(f"Added experiment record: {experiment_id} (outcome: {outcome.value})")
        return record

    def _extract_lessons_learned(
        self,
        spec: ExperimentSpec,
        review_result: ReviewResult,
        judge_decision: JudgeDecision,
    ) -> List[str]:
        """Extract lessons learned from an experiment."""
        lessons = []

        # If successful
        if judge_decision.approve:
            if judge_decision.performance_report:
                if judge_decision.performance_report.lift_score > 0.10:
                    lessons.append(f"Significant lift achieved ({judge_decision.performance_report.lift_score:.2%})")
                    lessons.append(f"Approach on {spec.component.value} was effective")

            # What went well
            if review_result.architecture.passes:
                lessons.append("Architecture patterns were followed correctly")

            if review_result.security.passes:
                lessons.append("No security concerns - good coding practices")

        # If failed
        else:
            # What went wrong
            if not review_result.architecture.passes:
                lessons.append("Architecture violations detected - need better design review")

            if not review_result.security.passes:
                lessons.append("Security issues found - must add security checks to process")

            if judge_decision.performance_report and judge_decision.performance_report.regressions:
                lessons.append("Regressions detected - need better testing coverage")

        return lessons

    def _generate_tags(
        self,
        spec: ExperimentSpec,
        review_result: ReviewResult,
        judge_decision: JudgeDecision,
    ) -> List[str]:
        """Generate tags for an experiment."""
        tags = []

        # Component tag
        tags.append(spec.component.value)

        # Scope tag
        tags.append(spec.scope.value)

        # Outcome tag
        if judge_decision.approve:
            tags.append("approved")
            if review_result.overall_score >= 0.9:
                tags.append("high_quality")
        else:
            tags.append("rejected")

        # Issue type tags
        if not review_result.security.passes:
            tags.append("security_issue")
        if not review_result.compliance.passes:
            tags.append("compliance_issue")
        if not review_result.architecture.passes:
            tags.append("architecture_issue")

        # Performance tag
        if judge_decision.performance_report:
            if judge_decision.performance_report.lift_score > 0.10:
                tags.append("high_lift")
            elif judge_decision.performance_report.lift_score > 0.05:
                tags.append("moderate_lift")
            elif judge_decision.performance_report.lift_score < 0:
                tags.append("negative_lift")

        return tags

    def query(self, query: MemoryQuery) -> List[ExperimentRecord]:
        """
        Query memory for relevant experiments.

        Args:
            query: Query parameters

        Returns:
            List of matching experiment records
        """
        results = self.records

        # Filter by component
        if query.component:
            results = [r for r in results if r.component == query.component]

        # Filter by outcome
        if query.outcome:
            results = [r for r in results if r.outcome == query.outcome]

        # Filter by tags
        if query.tags:
            results = [r for r in results if any(tag in r.tags for tag in query.tags)]

        # Filter by text search (simple keyword matching)
        if query.text_search:
            keywords = query.text_search.lower().split()
            results = [
                r for r in results
                if any(kw in r.title.lower() or kw in r.description.lower() for kw in keywords)
            ]

        # Filter by file patterns
        if query.file_patterns:
            results = [
                r for r in results
                if any(any(pattern in f for pattern in query.file_patterns) for f in r.files_changed)
            ]

        # Filter by lift score
        if query.min_lift_score is not None:
            results = [r for r in results if r.lift_score >= query.min_lift_score]
        if query.max_lift_score is not None:
            results = [r for r in results if r.lift_score <= query.max_lift_score]

        # Filter by time range
        if query.time_range:
            start, end = query.time_range
            results = [r for r in results if start <= r.started_at <= end]

        # Sort by recency
        results = sorted(results, key=lambda r: r.started_at, reverse=True)

        # Limit
        return results[:query.limit]

    def get_context_for_spec(self, spec: ExperimentSpec) -> HistoricalContext:
        """
        Get historical context for an experiment spec.

        Args:
            spec: Experiment specification

        Returns:
            Historical context with related experiments
        """
        # Find related experiments (same component, similar scope)
        related_query = MemoryQuery(
            component=spec.component.value,
            limit=5,
        )
        related_experiments = self.query(related_query)

        # Find similar failures (same component, failed, similar scope)
        failure_query = MemoryQuery(
            component=spec.component.value,
            outcome=ExperimentOutcome.FAILURE,
            limit=5,
        )
        similar_failures = self.query(failure_query)

        # Filter for similar scope
        similar_failures = [
            f for f in similar_failures
            if f.scope == spec.scope.value or f.scope == "full_stack"
        ]

        # Find similar successes
        success_query = MemoryQuery(
            component=spec.component.value,
            outcome=ExperimentOutcome.SUCCESS,
            limit=5,
        )
        similar_successes = self.query(success_query)

        # Detect patterns
        failure_patterns = self._detect_failure_patterns(similar_failures)
        success_patterns = self._detect_success_patterns(similar_successes)

        # Generate warnings
        warnings = self._generate_warnings(spec, similar_failures)

        # Generate suggestions
        suggestions = self._generate_suggestions(spec, similar_successes)

        return HistoricalContext(
            related_experiments=related_experiments,
            similar_failures=similar_failures,
            similar_successes=similar_successes,
            failure_patterns=failure_patterns,
            success_patterns=success_patterns,
            warnings=warnings,
            suggestions=suggestions,
        )

    def _detect_failure_patterns(self, failures: List[ExperimentRecord]) -> List[str]:
        """Detect common patterns in failures."""
        patterns = []

        if not failures:
            return patterns

        # Check for common failure reasons
        reason_counts = {}
        for failure in failures:
            for reason in failure.failure_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # If a reason appears in > 50% of failures, it's a pattern
        if reason_counts:
            most_common = max(reason_counts, key=reason_counts.get)
            if reason_counts[most_common] / len(failures) > 0.5:
                patterns.append(f"Common failure reason: {most_common.value}")

        # Check for common file patterns
        file_counts = {}
        for failure in failures:
            for file_path in failure.files_changed:
                # Get just the directory/module
                module = "/".join(file_path.split("/")[:3])
                file_counts[module] = file_counts.get(module, 0) + 1

        if file_counts:
            most_common = max(file_counts, key=file_counts.get)
            if file_counts[most_common] / len(failures) > 0.5:
                patterns.append(f"Risky module: {most_common}")

        return patterns

    def _detect_success_patterns(self, successes: List[ExperimentRecord]) -> List[str]:
        """Detect common patterns in successes."""
        patterns = []

        if not successes:
            return patterns

        # Check for common scopes
        scope_counts = {}
        for success in successes:
            scope_counts[success.scope] = scope_counts.get(success.scope, 0) + 1

        # If a scope appears in > 50% of successes, it's a pattern
        if scope_counts:
            most_common = max(scope_counts, key=scope_counts.get)
            if scope_counts[most_common] / len(successes) > 0.5:
                patterns.append(f"Successful scope: {most_common.value}")

        # Check average lift
        avg_lift = sum(s.lift_score for s in successes) / len(successes)
        if avg_lift > 0.10:
            patterns.append(f"High average lift: {avg_lift:.2%}")

        return patterns

    def _generate_warnings(self, spec: ExperimentSpec, failures: List[ExperimentRecord]) -> List[str]:
        """Generate warnings based on similar failures."""
        warnings = []

        if not failures:
            return warnings

        # Count similar failures
        similar_count = len([f for f in failures if f.scope == spec.scope.value])

        if similar_count >= 3:
            warnings.append(f"High failure rate for {spec.component.value} with {spec.scope.value} scope ({similar_count} similar failures)")

        # Check for common modules
        for failure in failures[:3]:
            for module in spec.affected_modules:
                if any(module in f for f in failure.files_changed):
                    warnings.append(f"Previous failures involved module: {module}")
                    break

        return warnings

    def _generate_suggestions(self, spec: ExperimentSpec, successes: List[ExperimentRecord]) -> List[str]:
        """Generate suggestions based on similar successes."""
        suggestions = []

        if not successes:
            return suggestions

        # Suggest successful scopes
        successful_scopes = set(s.scope for s in successes if s.lift_score > 0.05)
        if successful_scopes and spec.scope.value not in successful_scopes:
            suggestions.append(f"Consider trying scopes that worked before: {', '.join(successful_scopes)}")

        # Suggest approaches from high-lift successes
        high_lift = [s for s in successes if s.lift_score > 0.10]
        if high_lift:
            suggestions.append(f"{len(high_lift)} previous experiments achieved high lift (>10%)")

        return suggestions

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.records:
            return {"total_records": 0}

        total = len(self.records)
        by_outcome = {}
        for record in self.records:
            if record.outcome:
                outcome = record.outcome.value
                by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

        by_component = {}
        for record in self.records:
            by_component[record.component] = by_component.get(record.component, 0) + 1

        avg_lift = sum(r.lift_score for r in self.records) / total

        return {
            "total_records": total,
            "by_outcome": by_outcome,
            "by_component": by_component,
            "average_lift": avg_lift,
        }


# Convenience function for creating Memory Agent
def create_memory_agent(
    memory_db_path: Optional[Path] = None,
    max_records: int = 10000,
) -> MemoryAgent:
    """
    Create a Memory Agent instance.

    Args:
        memory_db_path: Path to memory database file
        max_records: Maximum number of records to keep

    Returns:
        Configured Memory Agent
    """
    return MemoryAgent(
        memory_db_path=memory_db_path,
        max_records=max_records,
    )
