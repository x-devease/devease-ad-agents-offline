"""
Agent Orchestrator - System Coordinator

Coordinates all agents in the self-evving ad mining system.
Implements the core evolution loop: Observation → Cognition → Production → Validation → Landing
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

# Import agents
from .pm_agent.agent import PMAgent, ExperimentSpec
from .memory_agent.agent import MemoryAgent
from .judge_agent.agent import JudgeAgent, EvaluationResult
from .coder_agent.agent import CoderAgent, PullRequest
from .reviewer_agent.agent import ReviewerAgent, ReviewResult
from .monitor_agent.agent import MonitorAgent, AnomalyAlert

logger = logging.getLogger(__name__)


@dataclass
class EvolutionCycle:
    """Complete record of one evolution cycle."""
    cycle_id: str
    timestamp: str
    objective: str
    domain: Optional[str]
    phases: Dict[str, Any] = field(default_factory=dict)
    final_decision: str = "PENDING"
    duration_seconds: float = 0.0


class AgentOrchestrator:
    """
    System Coordinator for Self-Evolving Ad Mining System.

    Orchestrates the interaction between all agents:
    - PM Agent (Planning)
    - Coder Agent (Implementation)
    - Reviewer Agent (Validation)
    - Judge Agent (Evaluation)
    - Memory Agent (Knowledge)
    - Monitor Agent (Observation)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Orchestrator with all agents.

        Args:
            config_path: Path to agents configuration file
            config: Configuration dictionary
        """
        self.config = config or {}
        self.config_path = config_path

        # Initialize agents
        self.memory_agent = MemoryAgent()
        self.pm_agent = PMAgent(memory_agent=self.memory_agent)
        self.coder_agent = CoderAgent(memory_agent=self.memory_agent, repo_root=Path.cwd())
        self.reviewer_agent = ReviewerAgent(memory_agent=self.memory_agent, repo_root=Path.cwd())
        self.judge_agent = JudgeAgent()
        self.monitor_agent = MonitorAgent(alert_callback=self._handle_anomaly_alert)

        # Evolution history
        self.evolution_history: List[EvolutionCycle] = []

        # System state
        self.active_cycle: Optional[EvolutionCycle] = None

        logger.info("Agent Orchestrator initialized")
        logger.info("  Agents: PM, Memory, Coder, Reviewer, Judge, Monitor")

    def run_evolution_loop(
        self,
        objective: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 10,
    ) -> EvolutionCycle:
        """
        Run complete evolution loop.

        Phases:
        1. Observation (Monitor Agent detects issue)
        2. Cognition (PM + Memory plan experiment)
        3. Production (Coder implements, Reviewer validates)
        4. Validation (Judge evaluates)
        5. Landing (Memory archives, merge or retry)

        Args:
            objective: High-level optimization objective
            domain: Specific domain (e.g., "gaming_ads")
            context: Additional context
            max_iterations: Maximum iterations before giving up

        Returns:
            EvolutionCycle: Complete cycle record
        """
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        cycle = EvolutionCycle(
            cycle_id=cycle_id,
            timestamp=start_time.isoformat(),
            objective=objective,
            domain=domain,
        )

        self.active_cycle = cycle

        logger.info("="*80)
        logger.info(f"EVOLUTION CYCLE: {cycle_id}")
        logger.info(f"  Objective: {objective}")
        logger.info(f"  Domain: {domain}")
        logger.info("="*80)

        try:
            # PHASE 1: Observation (if context provided)
            cycle.phases["observation"] = self._phase_observation(context)

            # Run iterations
            for iteration in range(max_iterations):
                logger.info(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

                # PHASE 2: Cognition
                cycle.phases[f"iteration_{iteration}_cognition"] = self._phase_cognition(
                    objective, domain, context
                )

                # PHASE 3: Production
                cycle.phases[f"iteration_{iteration}_production"] = self._phase_production(
                    cycle.phases[f"iteration_{iteration}_cognition"]["spec"]
                )

                # Check if production was rejected
                if cycle.phases[f"iteration_{iteration}_production"]["status"] == "rejected":
                    logger.warning(f"✗ Production rejected on iteration {iteration + 1}")
                    cycle.final_decision = "FAIL"
                    break

                # PHASE 4: Validation
                cycle.phases[f"iteration_{iteration}_validation"] = self._phase_validation(
                    cycle.phases[f"iteration_{iteration}_production"]["pr_id"]
                )

                # Check if passed
                decision = cycle.phases[f"iteration_{iteration}_validation"]["decision"]
                if decision == "PASS":
                    logger.info(f"✓ Experiment passed on iteration {iteration + 1}")
                    cycle.final_decision = "PASS"
                    break
                else:
                    logger.warning(f"✗ Experiment failed on iteration {iteration + 1}")
                    # Update context with failure reason and retry
                    context = context or {}
                    context["failure_reason"] = cycle.phases[f"iteration_{iteration}_validation"]["failure_reason"]

            else:
                # Max iterations reached without success
                logger.error(f"Failed after {max_iterations} iterations")
                cycle.final_decision = "FAIL"

            # PHASE 5: Landing (archive or merge)
            cycle.phases["landing"] = self._phase_landing(cycle)

        except Exception as e:
            logger.exception(f"Evolution cycle failed with error: {e}")
            cycle.final_decision = "ERROR"
            cycle.phases["error"] = str(e)

        # Calculate duration
        end_time = datetime.now()
        cycle.duration_seconds = (end_time - start_time).total_seconds()

        # Archive cycle
        self.evolution_history.append(cycle)

        logger.info("="*80)
        logger.info(f"CYCLE COMPLETE: {cycle_id}")
        logger.info(f"  Final Decision: {cycle.final_decision}")
        logger.info(f"  Duration: {cycle.duration_seconds:.1f}s")
        logger.info("="*80)

        return cycle

    def _phase_observation(
        self,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Phase 1: Observation

        Monitor Agent detects anomaly or issue.
        """
        logger.info("\n[PHASE 1: OBSERVATION]")
        logger.info("Detecting performance issues or opportunities...")

        phase_result = {
            "status": "complete",
            "issues_detected": [],
            "anomalies": [],
        }

        # Check Monitor Agent for anomalies
        system_status = self.monitor_agent.get_system_status()

        if system_status["alerts"]["total_unresolved"] > 0:
            logger.info(f"  Monitor Agent detected {system_status['alerts']['total_unresolved']} unresolved alerts")

            # Get recent alerts
            unresolved_alerts = [a for a in self.monitor_agent.alerts if not a.resolved]
            for alert in unresolved_alerts[-3:]:  # Show last 3
                phase_result["anomalies"].append({
                    "metric": alert.metric_name,
                    "severity": alert.severity,
                    "description": alert.description,
                })
                logger.info(f"    - {alert.severity}: {alert.description}")

        if context:
            # Use provided context
            issue = context.get("issue", "")
            if issue:
                phase_result["issues_detected"].append(issue)
                logger.info(f"  Context Issue: {issue}")

        # If metrics provided in context, collect them
        if context and "current_metrics" in context:
            self.monitor_agent.collect_metrics(context["current_metrics"])
            logger.info(f"  Collected {len(context['current_metrics'])} metrics")

        logger.info(f"  Total issues detected: {len(phase_result['issues_detected'])}")
        logger.info(f"  Total anomalies: {len(phase_result['anomalies'])}")

        return phase_result

    def _phase_cognition(
        self,
        objective: str,
        domain: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Phase 2: Cognition

        PM Agent creates experiment spec with Memory Agent's historical context.
        """
        logger.info("\n[PHASE 2: COGNITION]")
        logger.info("Planning experiment based on historical data...")

        # PM Agent creates spec
        spec = self.pm_agent.create_experiment_spec(
            objective=objective,
            domain=domain,
            context=context,
        )

        # Check for failure patterns
        if self.memory_agent:
            warnings = self.memory_agent.check_failure_pattern(
                spec.__dict__,
            )
            if warnings:
                logger.warning(f"  ⚠ Found {len(warnings)} matching failure patterns:")
                for warning in warnings[:3]:  # Show top 3
                    logger.warning(f"    - {warning['type']}: {warning['failure_reason']}")

        logger.info(f"  ✓ Experiment spec created: {spec.id}")
        logger.info(f"    Approach: {spec.approach}")
        logger.info(f"    Priority: {spec.priority}")

        return {
            "status": "complete",
            "spec": spec,
            "failure_warnings": len(warnings) if warnings else 0,
        }

    def _phase_production(
        self,
        spec: ExperimentSpec,
    ) -> Dict[str, Any]:
        """
        Phase 3: Production

        Coder Agent implements spec, Reviewer Agent validates.
        """
        logger.info("\n[PHASE 3: PRODUCTION]")
        logger.info("Implementing experiment...")

        # Step 1: Coder Agent implements the spec
        logger.info("  [Coder Agent] Implementing experiment spec...")
        pr = self.coder_agent.implement_experiment(spec.__dict__)

        logger.info(f"    PR ID: {pr.pr_id}")
        logger.info(f"    Branch: {pr.branch_name}")
        logger.info(f"    Files modified: {len(pr.files_modified)}")
        logger.info(f"    Tests: {'PASSED' if pr.tests_passed else 'FAILED'}")

        # Step 2: Reviewer Agent reviews the PR
        logger.info("  [Reviewer Agent] Reviewing pull request...")
        review = self.reviewer_agent.review_pull_request(
            pr=pr.__dict__,
            experiment_spec=spec.__dict__,
        )

        logger.info(f"    Decision: {review.decision}")
        logger.info(f"    Critical issues: {len(review.critical_issues)}")
        logger.info(f"    Warnings: {len(review.warnings)}")
        logger.info(f"    Security score: {review.security_score:.2f}")
        logger.info(f"    Architecture compliance: {review.architecture_compliance}")

        # Check if reviewer rejected
        if review.decision == "REJECT":
            logger.warning("  Reviewer Agent REJECTED changes")
            return {
                "status": "rejected",
                "pr_id": pr.pr_id,
                "review_decision": review.decision,
                "critical_issues": review.critical_issues,
            }

        # Check if reviewer requested changes
        if review.decision == "REQUEST_CHANGES":
            logger.warning("  Reviewer Agent requested changes")
            # In production, would iterate with Coder Agent
            # For now, proceed with caution

        logger.info(f"  ✓ Production phase complete: {pr.pr_id}")

        return {
            "status": "complete",
            "pr_id": pr.pr_id,
            "review_decision": review.decision,
            "files_modified": pr.files_modified,
            "tests_passed": pr.tests_passed,
            "security_score": review.security_score,
            "architecture_compliance": review.architecture_compliance,
        }

    def _phase_validation(
        self,
        pr_id: str,
    ) -> Dict[str, Any]:
        """
        Phase 4: Validation

        Judge Agent evaluates experiment against golden set and baseline.
        """
        logger.info("\n[PHASE 4: VALIDATION]")
        logger.info("Evaluating experiment results...")

        # Extract experiment_id from pr_id
        experiment_id = pr_id.replace("pr_", "exp_")

        # Judge Agent evaluates
        result = self.judge_agent.evaluate_experiment(
            experiment_id=experiment_id,
            branch_name=f"experiment/{experiment_id}",
            baseline_results={
                "accuracy": 0.67,  # Example baseline
                "lift": 1.2,
            },
        )

        logger.info(f"  Decision: {result.decision}")
        logger.info(f"  Lift Score: {result.lift_score:.1f}%")
        logger.info(f"  Confidence: {result.confidence:.3f}")
        logger.info(f"  Regression: {result.regression_detected}")

        if result.decision == "FAIL":
            logger.warning(f"  Failure reason: {result.failure_reason}")

        return {
            "status": "complete",
            "decision": result.decision,
            "lift_score": result.lift_score,
            "confidence": result.confidence,
            "regression_detected": result.regression_detected,
            "failure_reason": result.failure_reason,
        }

    def _phase_landing(
        self,
        cycle: EvolutionCycle,
    ) -> Dict[str, Any]:
        """
        Phase 5: Landing

        Memory Agent archives experiment, system merges PR or logs failure.
        """
        logger.info("\n[PHASE 5: LANDING]")
        logger.info("Archiving results and finalizing...")

        if cycle.final_decision == "PASS":
            logger.info("  ✓ Merging PR to main branch")
            logger.info("  ✓ Archiving successful experiment to Memory")

            # In production, this would:
            # 1. Merge PR
            # 2. Deploy to production
            # 3. Memory Agent stores success pattern

            result = {
                "status": "merged",
                "action": "deployed_to_production",
            }
        else:
            logger.info("  ✗ Experiment failed, archiving lessons learned")

            # In production, Memory Agent would:
            # 1. Store failure pattern
            # 2. Extract lessons learned
            # 3. Prevent similar approaches

            result = {
                "status": "logged",
                "action": "archived_as_failure",
            }

        return result

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "total_cycles": len(self.evolution_history),
            "successful_cycles": sum(1 for c in self.evolution_history if c.final_decision == "PASS"),
            "failed_cycles": sum(1 for c in self.evolution_history if c.final_decision == "FAIL"),
            "memory_stats": self.memory_agent.get_statistics(),
            "pm_stats": self.pm_agent.to_dict(),
            "coder_stats": self.coder_agent.to_dict(),
            "reviewer_stats": self.reviewer_agent.to_dict(),
            "judge_stats": self.judge_agent.to_dict(),
            "monitor_stats": self.monitor_agent.to_dict(),
        }

    def _handle_anomaly_alert(self, alert: AnomalyAlert):
        """
        Callback for Monitor Agent when anomaly is detected.

        Args:
            alert: Anomaly alert from Monitor Agent
        """
        logger.warning(f"Anomaly alert received: {alert.description}")

        # In production, could trigger evolution loop automatically
        # For now, just log the alert

    def visualize_evolution(self) -> str:
        """Generate ASCII visualization of evolution history."""
        if not self.evolution_history:
            return "No evolution cycles yet"

        lines = []
        lines.append("\n" + "="*80)
        lines.append("EVOLUTION HISTORY")
        lines.append("="*80)

        for cycle in self.evolution_history[-10:]:  # Show last 10
            status_icon = "✓" if cycle.final_decision == "PASS" else "✗"
            lines.append(
                f"{status_icon} {cycle.cycle_id}: {cycle.objective} "
                f"({cycle.final_decision}, {cycle.duration_seconds:.1f}s)"
            )

        lines.append("="*80 + "\n")

        return "\n".join(lines)


def main():
    """Example usage of Agent Orchestrator."""
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()

    # Run evolution loop
    cycle = orchestrator.run_evolution_loop(
        objective="improve_psychology_classification",
        domain="gaming_ads",
        context={
            "issue": "Psychology classification accuracy dropped to 67%",
            "current_metrics": {
                "accuracy": 0.67,
            },
            "severity": "high",
        },
        max_iterations=3,
    )

    # Show system status
    print(orchestrator.visualize_evolution())
    print(json.dumps(orchestrator.get_system_status(), indent=2))


if __name__ == "__main__":
    main()
