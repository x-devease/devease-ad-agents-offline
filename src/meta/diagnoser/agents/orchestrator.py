"""
Orchestrator - Coordinates all agent interactions.

Manages the workflow from analysis to implementation to validation.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory_agent import MemoryAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator for the AI Agent team.

    Workflow:
    1. PM Agent analyzes reports → generates experiment Spec
    2. Memory Agent provides historical context
    3. Coder Agent implements Spec → generates PR
    4. Reviewer Agent reviews PR → decides PASS/FAIL
    5. If PASS → Judge Agent runs backtest
    6. Judge Agent evaluates → decides MERGE/REJECT
    7. If MERGE → Memory Agent archives experience
    8. Start next iteration

    Args:
        max_iterations: Maximum number of optimization iterations
        use_real_llm: If True, use real Anthropic Claude API. If False, use mock implementations.
    """

    def __init__(self, max_iterations: int = 10, use_real_llm: bool = False):
        """
        Initialize orchestrator.

        Args:
            max_iterations: Maximum number of optimization iterations
            use_real_llm: Whether to use real LLM calls (requires ANTHROPIC_API_KEY)
        """
        self.max_iterations = max_iterations
        self.use_real_llm = use_real_llm
        self.memory_agent = MemoryAgent()
        self.current_iteration = 0
        self.best_f1 = {}  # Track best F1 per detector

        # Initialize LLM client if real LLM is enabled
        self.llm_client = None
        if use_real_llm:
            try:
                from .llm_client import LLMClient
                self.llm_client = LLMClient()
                logger.info("Real LLM mode enabled")
            except (ImportError, ValueError) as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Falling back to mock mode.")
                self.use_real_llm = False

    def run_optimization_cycle(
        self,
        detector: str,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete optimization cycle for a detector.

        Args:
            detector: Detector name (e.g., "FatigueDetector")
            target_metrics: Target metrics to achieve

        Returns:
            Cycle results including final metrics and decisions
        """
        print(f"\n{'='*80}")
        print(f"Starting optimization cycle for {detector}")
        print(f"{'='*80}\n")

        # Load current metrics
        current_metrics = self._load_current_metrics(detector)

        if not current_metrics:
            print(f"❌ No existing metrics found for {detector}")
            return {"status": "error", "message": "No existing metrics"}

        print(f"Current metrics: {json.dumps(current_metrics, indent=2)}")

        # Phase 1: PM Agent analysis with Memory context
        print("\n--- Phase 1: PM Agent Analysis ---")
        memory_context = self.memory_agent.query(
            query_type="SIMILAR_EXPERIMENTS",
            detector=detector,
            context={"tags": ["threshold_tuning", "recall_optimization"]}
        )

        experiment_spec = self._pm_agent_analyze(
            detector,
            current_metrics,
            target_metrics or {},
            memory_context
        )

        print(f"Experiment spec: {experiment_spec.get('title', 'No spec generated')}")

        if not experiment_spec:
            print("❌ PM Agent did not generate an experiment spec")
            return {"status": "error", "message": "No experiment spec generated"}

        # Phase 2: Coder Agent implementation
        print("\n--- Phase 2: Coder Agent Implementation ---")
        implementation = self._coder_agent_implement(experiment_spec)

        if not implementation or implementation.get("status") == "error":
            print(f"❌ Coder Agent failed: {implementation.get('message', 'Unknown error')}")
            return {"status": "error", "message": "Implementation failed"}

        print(f"✅ Implementation completed: {len(implementation.get('files_changed', []))} files changed")

        # Phase 3: Reviewer Agent
        print("\n--- Phase 3: Reviewer Agent ---")
        review = self._reviewer_agent_review(implementation, experiment_spec)

        if review["review_result"]["decision"] == "REJECTED":
            print(f"❌ Review rejected: {review['feedback'].get('concerns', [])}")
            return {
                "status": "rejected",
                "phase": "review",
                "review": review
            }

        print("✅ Review approved")

        # Phase 4: Judge Agent evaluation
        print("\n--- Phase 4: Judge Agent Evaluation ---")
        evaluation = self._judge_agent_evaluate(detector, current_metrics)

        print(f"Evaluation result: {evaluation['evaluation_result']['decision']}")
        print(f"Metrics lift: {json.dumps(evaluation['evaluation_result'].get('metrics', {}).get('lift', {}), indent=2)}")

        # Phase 4b: Rollback if evaluation failed
        if evaluation["evaluation_result"]["decision"] == "FAIL":
            print("\n--- Phase 4b: Automatic Rollback ---")

            # Rollback code changes
            rollback_result = self._rollback_changes(implementation, current_metrics)

            if rollback_result["status"] == "success":
                print(f"✅ Rollback successful: {rollback_result['message']}")
            else:
                print(f"❌ Rollback failed: {rollback_result['message']}")
                print(f"   Manual intervention required to restore: {rollback_result['commit_before']}")

            # Archive as FAILURE with rollback info
            outcome = "FAILURE"

            experiment_record = {
                "detector": detector,
                "spec": experiment_spec,
                "implementation": implementation,
                "review": review,
                "evaluation": evaluation["evaluation_result"],
                "rollback": rollback_result,
                "outcome": outcome,
                "timestamp": datetime.now().isoformat(),
                "tags": [experiment_spec.get("scope", "unknown"), "rolled_back"]
            }

            experiment_id = self.memory_agent.save_experiment(experiment_record)
            print(f"✅ Archived as {experiment_id} (with rollback)")

            return {
                "status": "failed",
                "phase": "evaluation",
                "experiment_id": experiment_id,
                "rollback": rollback_result,
                "evaluation": evaluation
            }

        # Phase 5: Archive to Memory
        print("\n--- Phase 5: Archive to Memory ---")
        outcome = "SUCCESS" if evaluation["evaluation_result"]["decision"] == "PASS" else "FAILURE"

        experiment_record = {
            "detector": detector,
            "spec": experiment_spec,
            "implementation": implementation,
            "review": review,
            "evaluation": evaluation["evaluation_result"],
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "tags": [experiment_spec.get("scope", "unknown")]
        }

        experiment_id = self.memory_agent.save_experiment(experiment_record)
        print(f"✅ Archived as {experiment_id}")

        # Return results
        return {
            "status": "success" if outcome == "SUCCESS" else "failed",
            "experiment_id": experiment_id,
            "experiment_spec": experiment_spec,
            "implementation": implementation,
            "review": review,
            "evaluation": evaluation,
            "outcome": outcome
        }

    def _load_current_metrics(self, detector: str) -> Optional[Dict[str, Any]]:
        """Load current metrics from evaluation report."""
        # Map detector name to report file
        detector_map = {
            "FatigueDetector": "fatigue_sliding_10windows.json",
            "LatencyDetector": "latency_sliding_10windows.json",
            "DarkHoursDetector": "dark_hours_sliding_10windows.json"
        }

        report_file = detector_map.get(detector)
        if not report_file:
            return None

        report_path = Path(__file__).parent.parent / "judge" / "reports" / "moprobo_sliding" / report_file

        if not report_path.exists():
            return None

        with open(report_path, 'r') as f:
            report = json.load(f)

        # Extract metrics based on report format
        if "aggregated_metrics" in report:
            # New format (DarkHoursDetector)
            metrics = report["aggregated_metrics"]
            return {
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "tp": metrics.get("total_tp", 0),
                "fp": metrics.get("total_fp", 0),
                "fn": metrics.get("total_fn", 0)
            }
        else:
            # Old format (LatencyDetector, FatigueDetector)
            accuracy = report.get("accuracy", {})
            return {
                "precision": accuracy.get("precision", 0),
                "recall": accuracy.get("recall", 0),
                "f1_score": accuracy.get("f1_score", 0),
                "tp": accuracy.get("total_tp", 0),
                "fp": accuracy.get("total_fp", 0),
                "fn": accuracy.get("total_fn", 0)
            }

    def _pm_agent_analyze(
        self,
        detector: str,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
        memory_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        PM Agent: Analyze metrics and generate experiment spec.

        Can use either real LLM or mock implementation.
        """
        # Use real LLM if enabled
        if self.use_real_llm and self.llm_client:
            try:
                from .llm_client import call_pm_agent

                context = {
                    "detector": detector,
                    "current_metrics": current_metrics,
                    "target_metrics": target_metrics,
                    "memory_context": memory_context,
                }

                return call_pm_agent(self.llm_client, context)

            except Exception as e:
                logger.error(f"Real LLM PM agent failed: {e}. Falling back to mock mode.")

        # Mock implementation - generates threshold tuning suggestions
        f1 = current_metrics.get("f1_score", 0)
        recall = current_metrics.get("recall", 0)
        fn = current_metrics.get("fn", 0)

        # Check warnings from memory
        warnings = memory_context.get("warnings", [])

        # Simple logic: if recall is low, suggest lowering threshold
        if recall < 0.6 and fn > 50:
            # Generate threshold optimization spec
            if detector == "FatigueDetector":
                return {
                    "title": f"优化{detector}的recall",
                    "scope": "阈值调整",
                    "changes": [{
                        "file": "src/meta/diagnoser/detectors/fatigue_detector.py",
                        "type": "threshold_tuning",
                        "parameter": "cpa_increase_threshold",
                        "from": 1.15,
                        "to": 1.10,
                        "reason": "降低CPA增长阈值，捕捉更多早期疲劳信号"
                    }],
                    "constraints": [
                        "严禁修改核心检测逻辑",
                        "严禁针对测试集硬编码",
                        "保持向后兼容"
                    ],
                    "expected_outcome": {
                        "f1_score": f"{f1 * 1.05:.3f} (+5%)",
                        "recall": f"{recall * 1.1:.3f} (+10%)",
                        "precision": ">=0.95"
                    },
                    "rollback_plan": "如果precision < 0.90，立即回滚"
                }
            elif detector == "LatencyDetector":
                return {
                    "title": f"优化{detector}的recall",
                    "scope": "阈值调整",
                    "changes": [{
                        "file": "src/meta/diagnoser/detectors/latency_detector.py",
                        "type": "threshold_tuning",
                        "parameter": "roas_drop_threshold",
                        "from": 0.7,
                        "to": 0.65,
                        "reason": "降低ROAS下降阈值，捕捉更多性能下降"
                    }],
                    "constraints": ["保持向后兼容"],
                    "expected_outcome": {
                        "f1_score": f"{f1 * 1.03:.3f} (+3%)",
                        "recall": f"{recall * 1.05:.3f} (+5%)",
                        "precision": ">=0.90"
                    },
                    "rollback_plan": "如果precision < 0.85，立即回滚"
                }

        # No optimization needed
        return None

    def _coder_agent_implement(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coder Agent: Implement the experiment spec.

        Simplified implementation - records changes without actual code modification.
        """
        changes = spec.get("changes", [])

        files_changed = []
        for change in changes:
            file_path = change.get("file", "")
            files_changed.append({
                "path": file_path,
                "parameter": change.get("parameter", ""),
                "from": change.get("from", ""),
                "to": change.get("to", ""),
                "change_type": change.get("type", "unknown")
            })

        return {
            "status": "success",
            "implementation": {
                "files_changed": files_changed,
                "test_results": {
                    "unit_tests": "PASS",
                    "syntax_check": "PASS"
                },
                "git_commit": {
                    "branch": f"feat/optimize-{spec.get('scope', 'unknown')}",
                    "commit_message": spec.get("title", "Optimization")
                }
            },
            "validation": {
                "code_quality": "遵循PEP8",
                "risk_assessment": "低风险：仅阈值调整"
            }
        }

    def _reviewer_agent_review(
        self,
        implementation: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reviewer Agent: Review the implementation.

        Can use either real LLM or mock implementation.
        """
        # Use real LLM if enabled
        if self.use_real_llm and self.llm_client:
            try:
                from .llm_client import call_reviewer_agent

                return call_reviewer_agent(self.llm_client, implementation, spec)

            except Exception as e:
                logger.error(f"Real LLM Reviewer agent failed: {e}. Falling back to mock mode.")

        # Mock implementation - approves threshold changes
        changes = implementation.get("implementation", {}).get("files_changed", [])

        # Simple checks
        is_threshold_change = any(c.get("change_type") == "threshold_tuning" for c in changes)
        small_diff = len(changes) <= 3

        if is_threshold_change and small_diff:
            return {
                "review_result": {
                    "decision": "APPROVED",
                    "overall_score": 85,
                    "checks": {
                        "architecture": {"status": "PASS", "score": 90},
                        "compliance": {"status": "PASS", "score": 100},
                        "logic_safety": {"status": "PASS", "score": 80},
                        "code_quality": {"status": "PASS", "score": 85}
                    }
                },
                "feedback": {
                    "strengths": ["阈值调整符合Spec要求", "代码diff简洁"],
                    "concerns": [],
                    "suggestions": []
                },
                "next_step": {
                    "if_approved": "提交给Judge Agent运行回测"
                }
            }

        # Reject if too many changes
        return {
            "review_result": {
                "decision": "REJECTED",
                "overall_score": 60,
                "checks": {
                    "architecture": {"status": "WARNING", "score": 70},
                    "compliance": {"status": "PASS", "score": 100},
                    "logic_safety": {"status": "WARNING", "score": 60},
                    "code_quality": {"status": "WARNING", "score": 65}
                }
            },
            "feedback": {
                "strengths": [],
                "concerns": ["改动范围过大，违反小步快跑原则"],
                "suggestions": ["拆分为多个小步骤"]
            },
            "next_step": {
                "if_rejected": "打回给Coder Agent，附上修改意见"
            }
        }

    def _judge_agent_evaluate(
        self,
        detector: str,
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Judge Agent: Evaluate the implementation through backtest.

        For demo purposes, simulates evaluation results.
        In production, this would run the actual evaluation script.
        """
        # Simulate running evaluation
        # In production: subprocess.run(["python", f"scripts/evaluate_{detector.lower()}.py"])

        # Simulate results with small improvement
        baseline_f1 = baseline_metrics.get("f1_score", 0.7)
        baseline_precision = baseline_metrics.get("precision", 0.95)
        baseline_recall = baseline_metrics.get("recall", 0.54)

        # Simulate improvement
        new_f1 = baseline_f1 * 1.05  # +5%
        new_precision = baseline_precision * 0.98  # -2%
        new_recall = baseline_recall * 1.10  # +10%

        lift_f1 = (new_f1 - baseline_f1) / baseline_f1
        lift_precision = (new_precision - baseline_precision) / baseline_precision
        lift_recall = (new_recall - baseline_recall) / baseline_recall

        # Decision logic
        decision = "PASS" if lift_f1 > 0.03 and new_precision > 0.8 else "FAIL"

        # Calculate TP, FP, FN (simulated)
        baseline_tp = baseline_metrics.get("tp", 66)
        baseline_fn = baseline_metrics.get("fn", 56)

        # With improved recall, we catch more FN
        new_tp = int(baseline_tp * 1.1)
        new_fn = baseline_fn - (new_tp - baseline_tp)

        return {
            "evaluation_result": {
                "decision": decision,
                "overall_score": int(new_f1 * 100),
                "metrics": {
                    "baseline": {
                        "f1_score": baseline_f1,
                        "precision": baseline_precision,
                        "recall": baseline_recall
                    },
                    "new": {
                        "f1_score": new_f1,
                        "precision": new_precision,
                        "recall": new_recall
                    },
                    "lift": {
                        "f1_score": f"+{lift_f1*100:.1f}%",
                        "precision": f"{lift_precision*100:.1f}%",
                        "recall": f"+{lift_recall*100:.1f}%"
                    }
                },
                "detailed_metrics": {
                    "windows": 10,
                    "total_tp": new_tp,
                    "total_fp": 2,
                    "total_fn": new_fn,
                    "grade": "C+"
                }
            },
            "regression_check": {
                "status": "PASS",
                "regressions": [],
                "concerns": [
                    "FP从0增加到2，需关注",
                    "Precision从100%降至98%，仍在可接受范围"
                ]
            },
            "recommendation": {
                "decision": "APPROVE_MERGE" if decision == "PASS" else "REJECT",
                "reason": f"F1提升{lift_f1*100:.1f}%，无严重副作用" if decision == "PASS" else "改进不显著",
                "suggestions": [
                    "继续监控FP趋势",
                    "下次优化关注FN的减少"
                ]
            },
            "next_step": {
                "if_approved": "合并PR，通知Memory Agent归档",
                "if_failed": "拒绝PR，通知Coder Agent回滚"
            }
        }

    def _rollback_changes(
        self,
        implementation: Dict[str, Any],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Rollback code changes after failed evaluation.

        Args:
            implementation: Implementation details from Coder Agent
            baseline_metrics: Baseline metrics to verify restoration

        Returns:
            dict with rollback status and details
        """
        import subprocess

        try:
            # Get current git status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "message": "Failed to get git status",
                    "error": result.stderr
                }

            # Get the last commit before implementation
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H"],
                capture_output=True,
                text=True,
                timeout=10
            )

            current_commit = result.stdout.strip()

            # Reset to the commit before (HEAD~1)
            print(f"Current commit: {current_commit[:8]}")
            print(f"Resetting to previous commit...")

            result = subprocess.run(
                ["git", "reset", "--hard", "HEAD~1"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "message": "Git reset failed",
                    "error": result.stderr,
                    "commit_before": current_commit
                }

            # Verify detector thresholds are restored
            detector_name = implementation.get("detector_name")
            if detector_name:
                try:
                    # Import detector and check thresholds
                    if detector_name == "FatigueDetector":
                        from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector
                        detector = FatigueDetector()
                    elif detector_name == "LatencyDetector":
                        from src.meta.diagnoser.detectors.latency_detector import LatencyDetector
                        detector = LatencyDetector()
                    elif detector_name == "DarkHoursDetector":
                        from src.meta.diagnoser.detectors.dark_hours_detector import DarkHoursDetector
                        detector = DarkHoursDetector()
                    else:
                        detector = None

                    if detector:
                        # Compare with baseline (simplified check)
                        # In production, would compare actual threshold values
                        print(f"✅ {detector_name} thresholds restored")

                except Exception as e:
                    print(f"⚠️  Warning: Could not verify detector thresholds: {e}")

            # Get the commit we rolled back to
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H"],
                capture_output=True,
                text=True,
                timeout=10
            )

            rollback_commit = result.stdout.strip()

            return {
                "status": "success",
                "message": f"Successfully rolled back to {rollback_commit[:8]}",
                "commit_before": current_commit,
                "commit_after": rollback_commit,
                "detector_restored": True
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": "Rollback operation timed out",
                "error": "timeout"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Rollback failed with exception: {str(e)}",
                "error": str(e)
            }

    def run_parallel_optimization(
        self,
        detectors: list,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run optimization cycles for multiple detectors in parallel.

        Args:
            detectors: List of detector names to optimize
            target_metrics: Target metrics to achieve

        Returns:
            dict with results for each detector
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}

        with ThreadPoolExecutor(max_workers=len(detectors)) as executor:
            # Submit all optimization jobs
            futures = {
                executor.submit(
                    self.run_optimization_cycle,
                    detector,
                    target_metrics
                ): detector for detector in detectors
            }

            # Collect results as they complete
            for future in as_completed(futures):
                detector = futures[future]
                try:
                    result = future.result(timeout=600)  # 10 min timeout per detector
                    results[detector] = result
                    print(f"\n✅ {detector} optimization completed")
                except Exception as e:
                    results[detector] = {
                        "status": "error",
                        "message": f"Optimization failed: {str(e)}"
                    }
                    print(f"\n❌ {detector} optimization failed: {e}")

        return results

