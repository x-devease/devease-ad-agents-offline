"""
Orchestrator - Team Coordination and Workflow for Ad Generator Development Team.

Responsible for:
- Coordinating all agents in the development team
- Managing the complete experiment lifecycle
- Enforcing the closed-loop evolution process
- Handling agent communication and handoffs
- Implementing the adversarial relationship between Coder and Judge
- Leveraging Memory Agent for continuous learning

Workflow:
1. Judge Agent detects issues → creates findings
2. PM Agent receives findings → queries Memory → creates spec
3. Coder Agent receives spec → implements changes → creates PR
4. Reviewer Agent reviews PR → approves or rejects
5. Judge Agent evaluates experiment → makes merge decision
6. Memory Agent records results → feeds back to step 2

Author: Ad System Dev Team
Date: 2026-02-04
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import asyncio

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of the workflow."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_REVIEW = "waiting_for_review"
    WAITING_FOR_JUDGMENT = "waiting_for_judgment"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class OrchestratorMode(Enum):
    """Mode of the orchestrator."""

    CONTINUOUS = "continuous"  # Keep running experiments
    SINGLE_EXPERIMENT = "single_experiment"  # Run one experiment and stop
    SUPERVISED = "supervised"  # Require human approval at each step


@dataclass
class WorkflowState:
    """Current state of the workflow."""

    status: WorkflowStatus = WorkflowStatus.IDLE
    current_experiment_id: Optional[str] = None
    current_spec_id: Optional[str] = None
    current_pr_id: Optional[str] = None

    # Tracking
    experiments_completed: int = 0
    experiments_succeeded: int = 0
    experiments_failed: int = 0

    # Current step
    current_step: str = ""
    step_history: List[str] = field(default_factory=list)

    # Metrics
    total_lift_achieved: float = 0.0
    best_lift_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "current_experiment_id": self.current_experiment_id,
            "current_spec_id": self.current_spec_id,
            "current_pr_id": self.current_pr_id,
            "experiments_completed": self.experiments_completed,
            "experiments_succeeded": self.experiments_succeeded,
            "experiments_failed": self.experiments_failed,
            "current_step": self.current_step,
            "step_history": self.step_history,
            "total_lift_achieved": self.total_lift_achieved,
            "best_lift_score": self.best_lift_score,
        }


@dataclass
class ExperimentResult:
    """Result of a complete experiment."""

    experiment_id: str
    success: bool
    lift_score: float
    confidence: float

    # Artifacts
    spec_id: str
    pr_id: str

    # Outcomes
    approved: bool = False
    merged: bool = False

    # Learnings
    lessons_learned: List[str] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def duration_seconds(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# Import all agents
from agents.pm_agent import PMAgent, ExperimentSpec, JudgeFindings, create_pm_agent
from agents.coder_agent import CoderAgent, PullRequest, create_coder_agent
from agents.reviewer_agent import ReviewerAgent, ReviewResult, create_reviewer_agent
from agents.judge_agent import JudgeAgent, JudgeDecision, create_judge_agent
from agents.memory_agent import MemoryAgent, HistoricalContext, create_memory_agent


class Orchestrator:
    """
    Orchestrator for the ad/generator development team.

    Manages the complete workflow:
    1. Receive findings from Judge Agent
    2. PM Agent creates experiment spec (with Memory context)
    3. Coder Agent implements spec
    4. Reviewer Agent reviews implementation
    5. Judge Agent evaluates results
    6. Memory Agent records learnings
    7. Loop back to step 1

    The orchestrator implements the adversarial dynamic between Coder and Judge,
    and leverages Memory for continuous improvement.
    """

    def __init__(
        self,
        repo_path: Path,
        mode: OrchestratorMode = OrchestratorMode.SUPERVISED,
        llm_client: Optional[Any] = None,
        memory_db_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize Orchestrator.

        Args:
            repo_path: Path to the repository
            mode: Orchestrator mode
            llm_client: LLM client for agents
            memory_db_path: Path to memory database
            config_path: Path to configuration
        """
        self.repo_path = Path(repo_path)
        self.mode = mode
        self.llm_client = llm_client
        self.config_path = config_path

        # Initialize all agents
        self.pm_agent = create_pm_agent(memory_client=None, config_path=config_path)
        self.coder_agent = create_coder_agent(repo_path=repo_path, llm_client=llm_client, config_path=config_path)
        self.reviewer_agent = create_reviewer_agent(repo_path=repo_path, llm_client=llm_client, config_path=config_path)
        self.judge_agent = create_judge_agent(repo_path=repo_path, config_path=config_path)
        self.memory_agent = create_memory_agent(memory_db_path=memory_db_path)

        # Link Memory Agent to PM Agent
        self.pm_agent.memory_client = self.memory_agent

        # Workflow state
        self.state = WorkflowState()

        # Callbacks for supervised mode
        self.callbacks: Dict[str, Callable] = {}

        logger.info(f"Orchestrator initialized in {mode.value} mode")

    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for supervised mode events.

        Events:
        - "spec_created": Called when PM Agent creates a spec
        - "pr_created": Called when Coder Agent creates a PR
        - "review_complete": Called when Reviewer Agent completes review
        - "decision_made": Called when Judge Agent makes a decision
        - "experiment_complete": Called when experiment is complete

        Callback signature: callback(data: Dict[str, Any]) -> bool
        Return True to continue, False to pause/stop.
        """
        self.callbacks[event] = callback
        logger.info(f"Registered callback for event: {event}")

    async def run_experiment_from_findings(
        self,
        findings: JudgeFindings,
    ) -> ExperimentResult:
        """
        Run a complete experiment from Judge Agent findings.

        Args:
            findings: Issues identified by Judge Agent

        Returns:
            Experiment result
        """
        experiment_id = f"exp-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        logger.info("=" * 70)
        logger.info(f"STARTING EXPERIMENT: {experiment_id}")
        logger.info("=" * 70)

        self.state.status = WorkflowStatus.RUNNING
        self.state.current_experiment_id = experiment_id
        self.state.current_step = "create_spec"
        result = ExperimentResult(
            experiment_id=experiment_id,
            success=False,
            lift_score=0.0,
            confidence=0.0,
            spec_id="",
            pr_id="",
        )

        try:
            # Step 1: PM Agent creates spec (with Memory context)
            logger.info("Step 1: PM Agent creating experiment spec...")
            self.state.step_history.append("pm_create_spec")

            historical_context = self.memory_agent.get_context_for_spec_findings(findings)
            spec = self.pm_agent.create_experiment_spec(findings, historical_context)

            result.spec_id = spec.spec_id
            self.state.current_spec_id = spec.spec_id

            # Callback for supervised mode
            if not await self._check_callback("spec_created", {"spec": spec, "findings": findings}):
                logger.info("Experiment paused by user at spec creation")
                self.state.status = WorkflowStatus.PAUSED
                return result

            # Step 2: Coder Agent implements spec
            logger.info("Step 2: Coder Agent implementing spec...")
            self.state.current_step = "implement_spec"
            self.state.step_history.append("coder_implement")

            implementation_result = self.coder_agent.implement_spec(spec)

            if not implementation_result.success:
                logger.error(f"Coder Agent failed: {implementation_result.errors}")
                result.success = False
                result.completed_at = datetime.now()
                self.state.status = WorkflowStatus.FAILED
                self.state.experiments_failed += 1
                return result

            pr = implementation_result.pull_request
            if not pr:
                logger.error("No PR created by Coder Agent")
                result.success = False
                result.completed_at = datetime.now()
                self.state.status = WorkflowStatus.FAILED
                self.state.experiments_failed += 1
                return result

            result.pr_id = pr.pr_id
            self.state.current_pr_id = pr.pr_id

            # Callback for supervised mode
            if not await self._check_callback("pr_created", {"pr": pr, "spec": spec}):
                logger.info("Experiment paused by user at PR creation")
                self.state.status = WorkflowStatus.PAUSED
                return result

            # Step 3: Reviewer Agent reviews PR
            logger.info("Step 3: Reviewer Agent reviewing PR...")
            self.state.current_step = "review_pr"
            self.state.step_history.append("reviewer_review")

            review_result = self.reviewer_agent.review_pull_request(pr, spec)

            # Callback for supervised mode
            if not await self._check_callback("review_complete", {
                "review": review_result,
                "pr": pr,
                "spec": spec,
            }):
                logger.info("Experiment paused by user at review completion")
                self.state.status = WorkflowStatus.PAUSED
                return result

            # If review failed, stop here
            if review_result.status.value in ["needs_changes", "rejected"]:
                logger.warning(f"Review failed: {review_result.status.value}")
                result.success = False
                result.completed_at = datetime.now()
                self.state.status = WorkflowStatus.FAILED
                self.state.experiments_failed += 1

                # Still record in memory
                self._record_experiment(spec, pr, review_result, None)
                return result

            # Step 4: Judge Agent evaluates
            logger.info("Step 4: Judge Agent evaluating experiment...")
            self.state.current_step = "evaluate"
            self.state.step_history.append("judge_evaluate")

            judge_decision = self.judge_agent.evaluate_experiment(pr, spec)

            result.lift_score = judge_decision.performance_report.lift_score if judge_decision.performance_report else 0.0
            result.confidence = judge_decision.confidence
            result.approved = judge_decision.approve

            # Callback for supervised mode
            if not await self._check_callback("decision_made", {
                "decision": judge_decision,
                "pr": pr,
                "spec": spec,
            }):
                logger.info("Experiment paused by user at decision")
                self.state.status = WorkflowStatus.PAUSED
                return result

            # Step 5: Record in Memory
            logger.info("Step 5: Recording experiment in Memory...")
            self.state.current_step = "record"
            self.state.step_history.append("memory_record")

            record = self._record_experiment(spec, pr, review_result, judge_decision)

            result.lessons_learned = record.lessons_learned

            # Update state
            result.success = judge_decision.approve
            result.completed_at = datetime.now()

            if judge_decision.approve:
                self.state.experiments_completed += 1
                self.state.experiments_succeeded += 1
                self.state.total_lift_achieved += result.lift_score
                if result.lift_score > self.state.best_lift_score:
                    self.state.best_lift_score = result.lift_score
                self.state.status = WorkflowStatus.COMPLETED
            else:
                self.state.experiments_completed += 1
                self.state.experiments_failed += 1
                self.state.status = WorkflowStatus.FAILED

            # Final callback
            await self._check_callback("experiment_complete", {
                "result": result,
                "spec": spec,
                "pr": pr,
                "review": review_result,
                "decision": judge_decision,
            })

            logger.info("=" * 70)
            logger.info(f"EXPERIMENT COMPLETE: {experiment_id}")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Lift: {result.lift_score:.2%}")
            logger.info(f"  Approved: {result.approved}")
            logger.info("=" * 70)

            return result

        except Exception as e:
            logger.error(f"Experiment failed with error: {e}", exc_info=True)
            result.success = False
            result.completed_at = datetime.now()
            self.state.status = WorkflowStatus.FAILED
            self.state.experiments_failed += 1
            return result

    async def _check_callback(self, event: str, data: Dict[str, Any]) -> bool:
        """
        Check if callback allows continuation.

        Returns True to continue, False to pause.
        """
        if self.mode == OrchestratorMode.SUPERVISED and event in self.callbacks:
            try:
                callback = self.callbacks[event]
                should_continue = await self._async_wrap(callback, data)
                if not should_continue:
                    logger.warning(f"Callback for '{event}' requested pause")
                return should_continue
            except Exception as e:
                logger.error(f"Callback for '{event}' failed: {e}")
                return True  # Continue on callback error
        return True

    async def _async_wrap(self, callback: Callable, data: Dict[str, Any]) -> bool:
        """Wrap sync/async callback."""
        import asyncio

        if asyncio.iscoroutinefunction(callback):
            return await callback(data)
        else:
            # Run sync callback in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, callback, data)

    def _record_experiment(
        self,
        spec: ExperimentSpec,
        pr: PullRequest,
        review_result: ReviewResult,
        judge_decision: Optional[JudgeDecision],
    ) -> Any:
        """Record experiment in Memory Agent."""
        # Create a mock judge decision if not provided
        if judge_decision is None:
            from agents.judge_agent import PerformanceReport
            judge_decision = JudgeDecision(
                approve=False,
                confidence=0.0,
                reason="Review failed",
                performance_report=None,
            )

        return self.memory_agent.add_experiment(
            spec=spec,
            pr=pr,
            review_result=review_result,
            judge_decision=judge_decision,
        )

    async def run_continuous(
        self,
        max_experiments: int = 10,
        min_lift_threshold: float = 0.01,
    ) -> List[ExperimentResult]:
        """
        Run experiments continuously until stopping condition.

        Args:
            max_experiments: Maximum number of experiments to run
            min_lift_threshold: Minimum lift to consider successful

        Returns:
            List of experiment results
        """
        logger.info(f"Starting continuous mode (max: {max_experiments}, min_lift: {min_lift_threshold})")

        # Start with some initial findings
        # In real implementation, this would come from monitoring
        findings = self._create_initial_findings()

        results = []

        for i in range(max_experiments):
            if self.mode == OrchestratorMode.SUPERVISED:
                logger.info(f"Experiment {i + 1}/{max_experiments} (supervised mode)")
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    break

            result = await self.run_experiment_from_findings(findings)
            results.append(result)

            # Check if we should continue
            if not result.success:
                logger.warning("Experiment failed, stopping continuous mode")
                break

            if result.lift_score < min_lift_threshold:
                logger.info(f"Lift ({result.lift_score:.2%}) below threshold, stopping")
                break

            # Generate new findings for next iteration
            # In real implementation, this would detect new issues
            findings = self._create_next_findings(result)

        logger.info(f"Continuous mode complete: {len(results)} experiments run")
        return results

    def _create_initial_findings(self) -> JudgeFindings:
        """Create initial findings to start the process."""
        from agents.pm_agent import Component, ExperimentPriority

        # Find component with most failures in memory
        stats = self.memory_agent.get_stats()
        by_component = stats.get("by_component", {})

        if by_component:
            # Pick component with most records (most activity)
            component_name = max(by_component, key=by_component.get)
            component = Component(component_name)
        else:
            # Default to AD_GENERATOR
            component = Component.AD_GENERATOR

        return JudgeFindings(
            issue_type="optimization",
            component=component,
            severity="medium",
            description=f"Initial optimization opportunity for {component.value}",
            evidence={},
            suggested_priority=ExperimentPriority.MEDIUM,
        )

    def _create_next_findings(self, result: ExperimentResult) -> JudgeFindings:
        """Create findings for next iteration based on result."""
        from agents.pm_agent import Component, ExperimentPriority

        # In real implementation, this would analyze the results
        # and find the next best opportunity

        # For now, just create a generic finding
        return JudgeFindings(
            issue_type="optimization",
            component=Component.AD_GENERATOR,
            severity="low",
            description="Continue optimization process",
            evidence={"previous_lift": result.lift_score},
            suggested_priority=ExperimentPriority.MEDIUM,
        )

    def get_state(self) -> WorkflowState:
        """Get current workflow state."""
        return self.state

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory_agent.get_stats()


# Monkey patch to add method to PM Agent for context retrieval
def _get_context_for_spec_findings(pm_agent: PMAgent, findings: JudgeFindings) -> Dict[str, Any]:
    """Get historical context from Memory Agent for findings."""
    if pm_agent.memory_client:
        from agents.pm_agent import Component

        # Create a temporary spec to query memory
        temp_spec = ExperimentSpec(
            spec_id="temp",
            title="temp",
            description="temp",
            component=findings.component,
            affected_modules=[],
            problem_statement=findings.description,
            success_criteria=[],
            failure_conditions=[],
            scope=pm_agent._determine_scope(findings, None),
            constraints=pm_agent.scope_constraints[pm_agent._determine_scope(findings, None)],
            priority=findings.suggested_priority,
        )

        context = pm_agent.memory_client.get_context_for_spec(temp_spec)

        return {
            "related_experiments": [e.spec_id for e in context.related_experiments],
            "similar_failures": [e.spec_id for e in context.similar_failures],
            "warnings": context.warnings,
            "suggestions": context.suggestions,
        }

    return {}


# Add the method to PMAgent
PMAgent.get_context_for_spec_findings = _get_context_for_spec_findings


# Convenience function for creating Orchestrator
def create_orchestrator(
    repo_path: Path,
    mode: OrchestratorMode = OrchestratorMode.SUPERVISED,
    llm_client: Optional[Any] = None,
    memory_db_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Orchestrator:
    """
    Create an Orchestrator instance.

    Args:
        repo_path: Path to the repository
        mode: Orchestrator mode
        llm_client: LLM client for agents
        memory_db_path: Path to memory database
        config_path: Path to configuration

    Returns:
        Configured Orchestrator
    """
    return Orchestrator(
        repo_path=repo_path,
        mode=mode,
        llm_client=llm_client,
        memory_db_path=memory_db_path,
        config_path=config_path,
    )
