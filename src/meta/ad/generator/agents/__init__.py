"""
Team Agents for Ad Generator Development.

This package contains the development team agents that work together
in a closed-loop system to continuously improve the ad/generator codebase
through automated experiments.
"""

# PM Agent
from .pm_agent import (
    PMAgent,
    ExperimentSpec,
    ChangeScope,
    Component,
    ExperimentPriority,
    ExperimentConstraint,
    JudgeFindings,
    create_pm_agent,
)

# Coder Agent
from .coder_agent import (
    CoderAgent,
    PullRequest,
    CodeChange,
    ChangeType,
    CodeQuality,
    ImplementationResult,
    create_coder_agent,
)

# Reviewer Agent
from .reviewer_agent import (
    ReviewerAgent,
    ReviewStatus,
    ReviewResult,
    ReviewIssue,
    Severity,
    IssueCategory,
    create_reviewer_agent,
)

# Judge Agent
from .judge_agent import (
    JudgeAgent,
    JudgeDecision,
    BacktestResult,
    PerformanceReport,
    MetricValue,
    create_judge_agent,
)

# Memory Agent
from .memory_agent import (
    MemoryAgent,
    ExperimentRecord,
    ExperimentOutcome,
    MemoryQuery,
    HistoricalContext,
    create_memory_agent,
)

# Orchestrator
from .orchestrator import (
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
