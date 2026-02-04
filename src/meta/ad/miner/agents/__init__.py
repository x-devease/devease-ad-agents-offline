"""
Ad Miner Agents Package

Self-evolving pattern mining system with autonomous agent team.
"""

from .orchestrator import AgentOrchestrator, EvolutionCycle
from .pm_agent.agent import PMAgent, ExperimentSpec
from .memory_agent.agent import MemoryAgent, ExperimentRecord
from .judge_agent.agent import JudgeAgent, EvaluationResult
from .coder_agent.agent import CoderAgent, PullRequest
from .reviewer_agent.agent import ReviewerAgent, ReviewResult
from .monitor_agent.agent import MonitorAgent, MetricSnapshot, AnomalyAlert, HealthCheck

__all__ = [
    "AgentOrchestrator",
    "EvolutionCycle",
    "PMAgent",
    "ExperimentSpec",
    "MemoryAgent",
    "ExperimentRecord",
    "JudgeAgent",
    "EvaluationResult",
    "CoderAgent",
    "PullRequest",
    "ReviewerAgent",
    "ReviewResult",
    "MonitorAgent",
    "MetricSnapshot",
    "AnomalyAlert",
    "HealthCheck",
]

__version__ = "1.0.0"
