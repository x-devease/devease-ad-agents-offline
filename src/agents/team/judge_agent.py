"""
Judge Agent - Quality Evaluation and Testing for Ad Generator Development Team.

Responsible for:
- Running automated backtests on changed code
- Evaluating algorithm performance metrics (CTR, ROAS, Lift Score)
- Testing against Golden Set and real Bad Cases
- Detecting regressions and side effects
- Publishing performance audit reports
- Deciding whether to merge code based on objective metrics

Author: Ad System Dev Team
Date: 2026-02-04
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Result of a test run."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class MetricDirection(Enum):
    """Direction of metric improvement."""

    HIGHER_IS_BETTER = "higher_is_better"  # e.g., CTR, ROAS
    LOWER_IS_BETTER = "lower_is_better"  # e.g., cost, error_rate
    TARGET_VALUE = "target_value"  # e.g., achieve specific target


@dataclass
class MetricDefinition:
    """Definition of a performance metric."""

    name: str
    direction: MetricDirection
    threshold: Optional[float] = None
    weight: float = 1.0
    description: str = ""


@dataclass
class MetricValue:
    """Value of a metric for a specific run."""

    metric_name: str
    value: float
    baseline_value: Optional[float] = None
    lift_percentage: Optional[float] = None
    threshold: Optional[float] = None
    passes_threshold: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "baseline_value": self.baseline_value,
            "lift_percentage": self.lift_percentage,
            "threshold": self.threshold,
            "passes_threshold": self.passes_threshold,
        }


@dataclass
class BacktestResult:
    """Result of running backtests."""

    branch_name: str
    commit_hash: str

    # Test results
    unit_tests: Dict[str, Any] = field(default_factory=dict)
    integration_tests: Dict[str, Any] = field(default_factory=dict)
    e2e_tests: Dict[str, Any] = field(default_factory=dict)

    # Overall
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0

    # Coverage
    coverage_percentage: float = 0.0
    coverage_delta: float = 0.0  # Change from baseline

    # Duration
    duration_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "branch_name": self.branch_name,
            "commit_hash": self.commit_hash,
            "unit_tests": self.unit_tests,
            "integration_tests": self.integration_tests,
            "e2e_tests": self.e2e_tests,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "pass_rate": self.pass_rate,
            "coverage_percentage": self.coverage_percentage,
            "coverage_delta": self.coverage_delta,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class PerformanceReport:
    """Performance evaluation report."""

    experiment_id: str
    branch_name: str
    component: str

    # Metrics
    metrics: List[MetricValue] = field(default_factory=list)

    # Business metrics (for ad-related components)
    ctr: Optional[float] = None
    roas: Optional[float] = None
    conversion_rate: Optional[float] = None
    cost_per_acquisition: Optional[float] = None

    # Algorithm metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # Overall lift score
    lift_score: float = 0.0

    # Regression detection
    regressions: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)

    # Overall assessment
    passes: bool = True
    confidence: float = 0.0

    # Timestamp
    evaluated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "branch_name": self.branch_name,
            "component": self.component,
            "metrics": [m.to_dict() for m in self.metrics],
            "business_metrics": {
                "ctr": self.ctr,
                "roas": self.roas,
                "conversion_rate": self.conversion_rate,
                "cost_per_acquisition": self.cost_per_acquisition,
            },
            "algorithm_metrics": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
            },
            "lift_score": self.lift_score,
            "regressions": self.regressions,
            "side_effects": self.side_effects,
            "passes": self.passes,
            "confidence": self.confidence,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass
class JudgeDecision:
    """Final decision on whether to merge code."""

    approve: bool
    confidence: float
    reason: str

    # Supporting evidence
    backtest_result: Optional[BacktestResult] = None
    performance_report: Optional[PerformanceReport] = None

    # Conditions
    conditions: List[str] = field(default_factory=list)

    # Metadata
    decided_at: datetime = field(default_factory=datetime.now)
    decided_by: str = "judge_agent"


# Import from other agents
from agents.team.coder_agent import PullRequest
from agents.team.pm_agent import ExperimentSpec, Component, JudgeFindings, ExperimentPriority


class JudgeAgent:
    """
    Judge Agent for ad/generator development.

    Responsibilities:
    1. Run backtests on experiment branches
    2. Evaluate performance metrics
    3. Detect regressions
    4. Generate performance reports
    5. Make merge/no-merge decisions
    6. Create findings for PM Agent
    """

    def __init__(
        self,
        repo_path: Path,
        config_path: Optional[Path] = None,
        baseline_data_path: Optional[Path] = None,
    ):
        """
        Initialize Judge Agent.

        Args:
            repo_path: Path to the repository
            config_path: Path to Judge Agent configuration
            baseline_data_path: Path to baseline performance data
        """
        self.repo_path = Path(repo_path)
        self.config_path = config_path
        self.baseline_data_path = Path(baseline_data_path) if baseline_data_path else None

        # Load configuration
        self._load_config()

        logger.info(f"JudgeAgent initialized for repo: {self.repo_path}")

    def _load_config(self):
        """Load Judge Agent configuration."""
        # Define metrics for each component
        self.component_metrics = {
            Component.AD_MINER: [
                MetricDefinition(
                    name="feature_extraction_accuracy",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.90,
                    weight=2.0,
                    description="Accuracy of feature extraction from images",
                ),
                MetricDefinition(
                    name="roas_prediction_error",
                    direction=MetricDirection.LOWER_IS_BETTER,
                    threshold=0.15,
                    weight=1.5,
                    description="Error in ROAS prediction",
                ),
                MetricDefinition(
                    name="recommendation_relevance",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.80,
                    weight=1.0,
                    description="Relevance of creative recommendations",
                ),
            ],
            Component.AD_GENERATOR: [
                MetricDefinition(
                    name="prompt_quality_score",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.85,
                    weight=2.0,
                    description="Quality of generated prompts",
                ),
                MetricDefinition(
                    name="image_generation_success_rate",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.95,
                    weight=1.5,
                    description="Success rate of image generation",
                ),
                MetricDefinition(
                    name="hallucination_rate",
                    direction=MetricDirection.LOWER_IS_BETTER,
                    threshold=0.05,
                    weight=2.0,
                    description="Rate of prompt hallucinations",
                ),
            ],
            Component.ADSET_ALLOCATOR: [
                MetricDefinition(
                    name="budget_utilization",
                    direction=MetricDirection.TARGET_VALUE,
                    threshold=0.95,
                    weight=2.0,
                    description="Percentage of budget utilized",
                ),
                MetricDefinition(
                    name="allocation_efficiency",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.80,
                    weight=1.5,
                    description="Efficiency of budget allocation",
                ),
                MetricDefinition(
                    name="safety_violation_rate",
                    direction=MetricDirection.LOWER_IS_BETTER,
                    threshold=0.0,
                    weight=3.0,
                    description="Rate of safety rule violations",
                ),
            ],
            Component.ADSET_GENERATOR: [
                MetricDefinition(
                    name="audience_match_rate",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.85,
                    weight=1.5,
                    description="Rate of audience matching",
                ),
                MetricDefinition(
                    name="configuration_success_rate",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.90,
                    weight=1.0,
                    description="Success rate of configuration generation",
                ),
            ],
            Component.NANO_BANANA_PRO: [
                MetricDefinition(
                    name="prompt_enhancement_quality",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.85,
                    weight=2.0,
                    description="Quality of prompt enhancement",
                ),
                MetricDefinition(
                    name="technique_application_accuracy",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=0.90,
                    weight=1.5,
                    description="Accuracy of technique application",
                ),
            ],
        }

        # Business metrics thresholds
        self.business_thresholds = {
            "min_ctr_improvement": 0.05,  # 5% improvement
            "min_roas_improvement": 0.10,  # 10% improvement
            "max_regression_rate": 0.02,  # Max 2% regression
        }

    def evaluate_experiment(
        self,
        pr: PullRequest,
        spec: ExperimentSpec,
        run_backtest: bool = True,
    ) -> JudgeDecision:
        """
        Evaluate an experiment and make merge decision.

        Args:
            pr: Pull request to evaluate
            spec: Associated experiment spec
            run_backtest: Whether to run backtests

        Returns:
            Judge decision with approval/rejection
        """
        logger.info(f"Evaluating experiment: {spec.spec_id}")

        # Run backtests
        if run_backtest:
            backtest_result = self._run_backtests(pr.branch_name)
        else:
            backtest_result = None

        # Evaluate performance
        performance_report = self._evaluate_performance(pr, spec, backtest_result)

        # Check for regressions
        regressions = self._detect_regressions(performance_report, spec)
        side_effects = self._detect_side_effects(performance_report, spec)

        performance_report.regressions = regressions
        performance_report.side_effects = side_effects

        # Make decision
        decision = self._make_decision(performance_report, backtest_result, spec)

        logger.info(f"Decision: {'APPROVE' if decision.approve else 'REJECT'} - {decision.reason}")
        return decision

    def _run_backtests(self, branch_name: str) -> BacktestResult:
        """Run backtests on the branch."""
        logger.info(f"Running backtests for branch: {branch_name}")

        # In real implementation, this would:
        # 1. Checkout the branch
        # 2. Run unit tests
        # 3. Run integration tests
        # 4. Run e2e tests
        # 5. Collect coverage

        # For now, return mock results
        result = BacktestResult(
            branch_name=branch_name,
            commit_hash="abc123",
            unit_tests={"passed": 45, "failed": 0, "skipped": 2},
            integration_tests={"passed": 12, "failed": 0, "skipped": 0},
            e2e_tests={"passed": 5, "failed": 0, "skipped": 0},
            total_tests=64,
            passed_tests=62,
            failed_tests=0,
            skipped_tests=2,
            coverage_percentage=85.5,
            coverage_delta=2.3,
            duration_seconds=120.5,
        )

        logger.info(f"Backtests complete: {result.passed_tests}/{result.total_tests} passed")
        return result

    def _evaluate_performance(
        self,
        pr: PullRequest,
        spec: ExperimentSpec,
        backtest_result: Optional[BacktestResult],
    ) -> PerformanceReport:
        """Evaluate performance metrics."""
        logger.info(f"Evaluating performance for: {spec.spec_id}")

        report = PerformanceReport(
            experiment_id=spec.spec_id,
            branch_name=pr.branch_name,
            component=spec.component.value,
        )

        # Get metrics for component
        metric_defs = self.component_metrics.get(spec.component, [])

        # Load baseline
        baseline = self._load_baseline_metrics(spec.component)

        # For each metric, calculate current value
        for metric_def in metric_defs:
            # In real implementation, this would run actual tests
            # For now, simulate values
            current_value = baseline.get(metric_def.name, 0.8) + 0.05  # Simulate 5% improvement
            baseline_value = baseline.get(metric_def.name, 0.8)

            # Calculate lift
            if baseline_value > 0:
                lift_percentage = ((current_value - baseline_value) / baseline_value) * 100
            else:
                lift_percentage = 0.0

            # Check threshold
            if metric_def.direction == MetricDirection.HIGHER_IS_BETTER:
                passes_threshold = current_value >= metric_def.threshold
            elif metric_def.direction == MetricDirection.LOWER_IS_BETTER:
                passes_threshold = current_value <= metric_def.threshold
            else:  # TARGET_VALUE
                passes_threshold = abs(current_value - metric_def.threshold) < 0.1

            metric_value = MetricValue(
                metric_name=metric_def.name,
                value=current_value,
                baseline_value=baseline_value,
                lift_percentage=lift_percentage,
                threshold=metric_def.threshold,
                passes_threshold=passes_threshold,
            )

            report.metrics.append(metric_value)

            # Update algorithm metrics
            if metric_def.name == "feature_extraction_accuracy":
                report.accuracy = current_value
            elif metric_def.name == "roas_prediction_error":
                report.precision = 1.0 - current_value

        # Calculate lift score (weighted average of metric lifts)
        report.lift_score = self._calculate_lift_score(report.metrics, metric_defs)

        # Calculate confidence
        report.confidence = self._calculate_confidence(report, backtest_result)

        # Determine if passes
        report.passes = (
            report.lift_score > 0
            and all(m.passes_threshold for m in report.metrics)
            and report.confidence > 0.7
        )

        logger.info(f"Performance evaluation complete: lift={report.lift_score:.2%}, passes={report.passes}")
        return report

    def _calculate_lift_score(
        self,
        metrics: List[MetricValue],
        metric_defs: List[MetricDefinition],
    ) -> float:
        """Calculate overall lift score."""
        total_weight = 0.0
        weighted_lift = 0.0

        for metric, metric_def in zip(metrics, metric_defs):
            if metric.lift_percentage is not None:
                weight = metric_def.weight
                lift = metric.lift_percentage / 100.0  # Convert to decimal

                # For lower-is-better metrics, invert the lift
                if metric_def.direction == MetricDirection.LOWER_IS_BETTER:
                    lift = -lift

                weighted_lift += lift * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_lift / total_weight

    def _calculate_confidence(
        self,
        report: PerformanceReport,
        backtest_result: Optional[BacktestResult],
    ) -> float:
        """Calculate confidence in the evaluation."""
        confidence = 0.5  # Base confidence

        # Increase confidence if metrics are clear
        if report.lift_score > 0.10:
            confidence += 0.2
        elif report.lift_score > 0.05:
            confidence += 0.1

        # Increase confidence if all metrics pass
        if all(m.passes_threshold for m in report.metrics):
            confidence += 0.2

        # Increase confidence if backtests pass
        if backtest_result and backtest_result.pass_rate >= 0.95:
            confidence += 0.1

        return min(1.0, confidence)

    def _detect_regressions(
        self,
        report: PerformanceReport,
        spec: ExperimentSpec,
    ) -> List[str]:
        """Detect regressions in performance."""
        regressions = []

        for metric in report.metrics:
            # Check if metric got worse
            if metric.lift_percentage is not None and metric.lift_percentage < -self.business_thresholds["max_regression_rate"] * 100:
                regressions.append(
                    f"Regression in {metric.metric_name}: {metric.lift_percentage:.2%}"
                )

        return regressions

    def _detect_side_effects(
        self,
        report: PerformanceReport,
        spec: ExperimentSpec,
    ) -> List[str]:
        """Detect side effects of changes."""
        side_effects = []

        # Check if improvement in one area caused degradation in another
        # This is component-specific

        if spec.component == Component.AD_MINER:
            # Check if feature extraction improvements hurt recommendation quality
            for metric in report.metrics:
                if metric.metric_name == "feature_extraction_accuracy":
                    if metric.lift_percentage and metric.lift_percentage > 20:
                        side_effects.append(
                            "Large improvement in feature extraction may indicate overfitting"
                        )

        elif spec.component == Component.AD_GENERATOR:
            # Check if prompt quality improvements hurt generation success rate
            for metric in report.metrics:
                if metric.metric_name == "prompt_quality_score":
                    if metric.lift_percentage and metric.lift_percentage > 15:
                        side_effects.append(
                            "Verify that prompt improvements don't reduce generation success rate"
                        )

        return side_effects

    def _make_decision(
        self,
        performance_report: PerformanceReport,
        backtest_result: Optional[BacktestResult],
        spec: ExperimentSpec,
    ) -> JudgeDecision:
        """Make merge/no-merge decision."""
        approve = True
        reason = []
        conditions = []

        # Check if performance passes
        if not performance_report.passes:
            approve = False
            reason.append("Performance metrics do not meet thresholds")

        # Check for regressions
        if performance_report.regressions:
            approve = False
            reason.append(f"Regressions detected: {', '.join(performance_report.regressions)}")

        # Check backtests
        if backtest_result and backtest_result.failed_tests > 0:
            approve = False
            reason.append(f"Backtests failed: {backtest_result.failed_tests} failures")

        # Check lift score
        if performance_report.lift_score <= 0:
            approve = False
            reason.append(f"No positive lift: {performance_report.lift_score:.2%}")

        # Add conditions for approval
        if approve:
            if performance_report.confidence < 0.9:
                conditions.append("Monitor production metrics closely after deployment")

            if performance_report.side_effects:
                conditions.append(f"Verify side effects: {', '.join(performance_report.side_effects)}")

        reason_str = "; ".join(reason) if reason else "All checks passed"

        return JudgeDecision(
            approve=approve,
            confidence=performance_report.confidence,
            reason=reason_str,
            backtest_result=backtest_result,
            performance_report=performance_report,
            conditions=conditions,
        )

    def _load_baseline_metrics(self, component: Component) -> Dict[str, float]:
        """Load baseline metrics for a component."""
        if self.baseline_data_path:
            baseline_file = self.baseline_data_path / f"{component.value}_baseline.json"
            if baseline_file.exists():
                try:
                    data = json.loads(baseline_file.read_text())
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load baseline: {e}")

        # Return default baselines
        return {
            "feature_extraction_accuracy": 0.85,
            "roas_prediction_error": 0.15,
            "recommendation_relevance": 0.80,
            "prompt_quality_score": 0.80,
            "image_generation_success_rate": 0.90,
            "hallucination_rate": 0.05,
            "budget_utilization": 0.95,
            "allocation_efficiency": 0.75,
            "safety_violation_rate": 0.0,
            "audience_match_rate": 0.80,
            "configuration_success_rate": 0.85,
            "prompt_enhancement_quality": 0.80,
            "technique_application_accuracy": 0.85,
        }

    def create_findings(self, decision: JudgeDecision, spec: ExperimentSpec) -> JudgeFindings:
        """
        Create findings for PM Agent based on decision.

        Args:
            decision: Judge decision
            spec: Associated experiment spec

        Returns:
            Judge findings for PM Agent
        """
        # Determine issue type
        if not decision.approve:
            if decision.backtest_result and decision.backtest_result.failed_tests > 0:
                issue_type = "bug"
            elif decision.performance_report and decision.performance_report.regressions:
                issue_type = "performance_drop"
            else:
                issue_type = "optimization"
        else:
            issue_type = "feature_request"

        # Determine severity
        if not decision.approve:
            if decision.confidence < 0.5:
                severity = "critical"
            elif decision.confidence < 0.7:
                severity = "high"
            else:
                severity = "medium"
        else:
            severity = "low"

        # Build description
        description = decision.reason

        # Gather evidence
        evidence = {}
        if decision.performance_report:
            evidence["metrics"] = [m.to_dict() for m in decision.performance_report.metrics]
            evidence["lift_score"] = decision.performance_report.lift_score

        # Determine priority
        if severity == "critical":
            priority = ExperimentPriority.CRITICAL
        elif severity == "high":
            priority = ExperimentPriority.HIGH
        else:
            priority = ExperimentPriority.MEDIUM

        return JudgeFindings(
            issue_type=issue_type,
            component=spec.component,
            severity=severity,
            description=description,
            evidence=evidence,
            suggested_priority=priority,
        )


# Convenience function for creating Judge Agent
def create_judge_agent(
    repo_path: Path,
    config_path: Optional[Path] = None,
    baseline_data_path: Optional[Path] = None,
) -> JudgeAgent:
    """
    Create a Judge Agent instance.

    Args:
        repo_path: Path to the repository
        config_path: Path to Judge Agent configuration
        baseline_data_path: Path to baseline performance data

    Returns:
        Configured Judge Agent
    """
    return JudgeAgent(
        repo_path=repo_path,
        config_path=config_path,
        baseline_data_path=baseline_data_path,
    )
