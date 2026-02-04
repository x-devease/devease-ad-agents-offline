"""
Ad Generator Development Team - AI Agents for Self-Evolving Code.

This package contains the development team agents for the ad/generator system:
- PM Agent: Product requirements and experiment planning
- Coder Agent: Code implementation engineer
- Reviewer Agent: Code quality and compliance officer
- Judge Agent: Quality evaluation and testing
- Memory Agent: Knowledge base and historical records
- Orchestrator: Team coordination and workflow

The agents work together in a closed-loop system to continuously improve
the ad/generator codebase through automated experiments.

Architecture:
    1. Judge Agent detects issues → creates findings
    2. PM Agent + Memory → creates experiment spec
    3. Coder Agent → implements changes → creates PR
    4. Reviewer Agent → approves/rejects PR
    5. Judge Agent → evaluates results → makes decision
    6. Memory Agent → records learnings
    7. Loop back to step 1

Usage:
    from agents import create_orchestrator

    orchestrator = create_orchestrator(
        repo_path=Path("/path/to/repo"),
        mode=OrchestratorMode.SUPERVISED,
    )

    # Run a single experiment
    findings = JudgeFindings(...)
    result = await orchestrator.run_experiment_from_findings(findings)

    # Or run continuously
    results = await orchestrator.run_continuous(max_experiments=10)

Author: Ad System Dev Team
Date: 2026-02-04
"""

# Core agents
from agents.pm_agent import (
    PMAgent,
    ExperimentSpec,
    ChangeScope,
    Component,
    ExperimentPriority,
    ExperimentConstraint,
    JudgeFindings,
    create_pm_agent,
)

from agents.coder_agent import (
    CoderAgent,
    PullRequest,
    CodeChange,
    ChangeType,
    CodeQuality,
    ImplementationResult,
    create_coder_agent,
)

from agents.reviewer_agent import (
    ReviewerAgent,
    ReviewStatus,
    ReviewResult,
    ReviewIssue,
    Severity,
    IssueCategory,
    create_reviewer_agent,
)

from agents.judge_agent import (
    JudgeAgent,
    JudgeDecision,
    BacktestResult,
    PerformanceReport,
    MetricValue,
    create_judge_agent,
)

from agents.memory_agent import (
    MemoryAgent,
    ExperimentRecord,
    ExperimentOutcome,
    MemoryQuery,
    HistoricalContext,
    create_memory_agent,
)

from agents.orchestrator import (
    Orchestrator,
    OrchestratorMode,
    WorkflowStatus,
    WorkflowState,
    ExperimentResult,
    create_orchestrator,
)

__all__ = [
    # PM Agent
    "PMAgent",
    "ExperimentSpec",
    "ChangeScope",
    "Component",
    "ExperimentPriority",
    "ExperimentConstraint",
    "JudgeFindings",
    "create_pm_agent",
    # Coder Agent
    "CoderAgent",
    "PullRequest",
    "CodeChange",
    "ChangeType",
    "CodeQuality",
    "ImplementationResult",
    "create_coder_agent",
    # Reviewer Agent
    "ReviewerAgent",
    "ReviewStatus",
    "ReviewResult",
    "ReviewIssue",
    "Severity",
    "IssueCategory",
    "create_reviewer_agent",
    # Judge Agent
    "JudgeAgent",
    "JudgeDecision",
    "BacktestResult",
    "PerformanceReport",
    "MetricValue",
    "create_judge_agent",
    # Memory Agent
    "MemoryAgent",
    "ExperimentRecord",
    "ExperimentOutcome",
    "MemoryQuery",
    "HistoricalContext",
    "create_memory_agent",
    # Orchestrator
    "Orchestrator",
    "OrchestratorMode",
    "WorkflowStatus",
    "WorkflowState",
    "ExperimentResult",
    "create_orchestrator",
]

__version__ = "1.0.0"
