#!/usr/bin/env python3
"""
Auto-Fix Orchestrator - Coordinates all agent interactions with automatic fix.

Manages the workflow from analysis to implementation to validation,
with automatic retry and fix on failures.
"""

import json
import subprocess
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.agents.memory_agent import MemoryAgent


class AutoFixOrchestrator:
    """
    Orchestrator with automatic fix capabilities.

    Will retry and fix issues automatically until success or max retries.
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize auto-fix orchestrator.

        Args:
            max_retries: Maximum number of retries per phase
        """
        self.max_retries = max_retries
        self.memory_agent = MemoryAgent()

    def run_optimization_with_auto_fix(
        self,
        detector: str,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run optimization cycle with automatic retry and fix.

        Will continue until:
        1. Success (all phases pass)
        2. Max retries reached
        3. Unrecoverable error
        """
        print(f"\n{'='*80}")
        print(f"AUTO-FIX Orchestrator: Starting optimization for {detector}")
        print(f"Max retries per phase: {self.max_retries}")
        print(f"{'='*80}\n")

        # Load current metrics once
        current_metrics = self._load_current_metrics(detector)
        if not current_metrics:
            return {"status": "error", "message": "No existing metrics"}

        print(f"Current metrics:")
        print(f"  F1-Score: {current_metrics['f1_score']:.4f}")
        print(f"  Precision: {current_metrics['precision']:.4f}")
        print(f"  Recall: {current_metrics['recall']:.4f}")
        print(f"  TP/FP/FN: {current_metrics['tp']}/{current_metrics['fp']}/{current_metrics['fn']}")

        # Try multiple optimization strategies
        tried_parameters = set()
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1
            print(f"\n{'='*80}")
            print(f"ATTEMPT {attempt}/{self.max_retries}")
            print(f"{'='*80}")

            # Phase 1: PM Agent Analysis
            print("\n" + "="*80)
            print("PHASE 1: PM Agent Analysis")
            print("="*80)

            # Generate experiment spec with tried parameters excluded
            experiment_spec = self._generate_experiment_spec_with_exclusions(
                detector, current_metrics, target_metrics, tried_parameters
            )

            if not experiment_spec:
                print(f"\n⚠️  No more optimization strategies to try")
                print(f"   All parameters attempted: {tried_parameters}")
                return {
                    "status": "exhausted",
                    "message": "No more optimization strategies available",
                    "attempted_parameters": list(tried_parameters)
                }

            param_key = f"{experiment_spec['changes'][0]['parameter']}"
            print(f"\nGenerated experiment spec:")
            print(f"  Title: {experiment_spec['title']}")
            print(f"  Parameter: {param_key}")
            print(f"  Change: {experiment_spec['changes'][0]['from']} → {experiment_spec['changes'][0]['to']}")

            # Mark this parameter as tried
            tried_parameters.add(param_key)

            # Phase 2: Code Modification (with retry)
            print("\n" + "="*80)
            print("PHASE 2: Code Modification")
            print("="*80)

            impl_result = self._modify_code_with_retry(experiment_spec)
            if not impl_result["success"]:
                print(f"❌ Code modification failed: {impl_result['error']}")
                print(f"   Trying next strategy...")
                continue

            print(f"✅ Code modified successfully")
            print(f"   File: {impl_result['file_path']}")
            print(f"   Backup: {impl_result['backup_path']}")

            # Phase 3: Code Review (with auto-fix)
            print("\n" + "="*80)
            print("PHASE 3: Code Review")
            print("="*80)

            review_result = self._review_with_auto_fix(experiment_spec, impl_result)
            if not review_result["passed"]:
                print(f"❌ Review failed: {review_result['error']}")
                print(f"   Issues: {review_result['issues']}")
                print(f"   Rolling back and trying next strategy...")
                self._rollback_changes(impl_result)
                continue

            print(f"✅ Review passed (Score: {review_result['score']}/100)")

            # Phase 4: Run Evaluation
            print("\n" + "="*80)
            print("PHASE 4: Run Evaluation")
            print("="*80)

            eval_result = self._run_evaluation_with_retry(detector, current_metrics)
            if not eval_result["success"]:
                print(f"❌ Evaluation failed: {eval_result['error']}")
                print(f"   Rolling back and trying next strategy...")
                self._rollback_changes(impl_result)
                continue

            print(f"\n✅ Evaluation PASSED!")
            print(f"\nMetrics Comparison:")
            print(f"  {'Metric':<12} {'Baseline':>12} {'New':>12} {'Lift':>12}")
            print(f"  {'-'*12:<12} {'-'*12:>12} {'-'*12:>12} {'-'*12:>12}")
            for metric in ["f1_score", "precision", "recall"]:
                baseline = eval_result["baseline"][metric]
                new = eval_result["new"][metric]
                lift = eval_result["lift"][metric]
                print(f"  {metric:<12} {baseline:>12.4f} {new:>12.4f} {lift:>12}")

            # Phase 5: Archive to Memory
            print("\n" + "="*80)
            print("PHASE 5: Archive to Memory")
            print("="*80)

            experiment_record = {
                "detector": detector,
                "spec": experiment_spec,
                "implementation": impl_result,
                "review": review_result,
                "evaluation": eval_result,
                "outcome": "SUCCESS",
                "timestamp": datetime.now().isoformat(),
                "tags": [experiment_spec.get("scope", "unknown"), "successful"]
            }

            experiment_id = self.memory_agent.save_experiment(experiment_record)
            print(f"✅ Archived as {experiment_id}")

            print(f"\n{'='*80}")
            print(f"✅ OPTIMIZATION CYCLE COMPLETED SUCCESSFULLY")
            print(f"{'='*80}\n")

            return {
                "status": "success",
                "experiment_id": experiment_id,
                "detector": detector,
                "baseline": eval_result["baseline"],
                "new": eval_result["new"],
                "lift": eval_result["lift"],
                "backup_path": impl_result["backup_path"],
                "attempts": attempt
            }

        # All attempts failed
        return {
            "status": "failed",
            "message": f"All {self.max_retries} optimization attempts failed",
            "attempted_parameters": list(tried_parameters)
        }

    def _load_current_metrics(self, detector: str) -> Optional[Dict[str, Any]]:
        """Load current metrics from evaluation report."""
        detector_map = {
            "FatigueDetector": "fatigue_sliding_10windows.json",
            "LatencyDetector": "latency_sliding_10windows.json",
            "DarkHoursDetector": "dark_hours_sliding_10windows.json"
        }

        report_file = detector_map.get(detector)
        if not report_file:
            return None

        report_path = project_root / "src" / "meta" / "diagnoser" / "judge" / "reports" / "moprobo_sliding" / report_file

        if not report_path.exists():
            return None

        with open(report_path, 'r') as f:
            report = json.load(f)

        # Extract metrics
        if "aggregated_metrics" in report:
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
            accuracy = report.get("accuracy", {})
            return {
                "precision": accuracy.get("precision", 0),
                "recall": accuracy.get("recall", 0),
                "f1_score": accuracy.get("f1_score", 0),
                "tp": accuracy.get("total_tp", 0),
                "fp": accuracy.get("total_fp", 0),
                "fn": accuracy.get("total_fn", 0)
            }

    def _generate_experiment_spec_with_exclusions(
        self,
        detector: str,
        current_metrics: Dict[str, float],
        target_metrics: Optional[Dict[str, float]],
        tried_parameters: set
    ) -> Optional[Dict[str, Any]]:
        """Generate experiment spec based on current metrics, excluding tried parameters."""

        # Available optimization strategies for FatigueDetector
        if detector == "FatigueDetector":
            recall = current_metrics.get("recall", 0)
            fn = current_metrics.get("fn", 0)
            f1 = current_metrics.get("f1_score", 0)
            precision = current_metrics.get("precision", 0)

            strategies = []

            # Strategy 1: Lower cpa_increase_threshold (if not tried)
            if "cpa_increase_threshold" not in tried_parameters and recall < 0.6:
                strategies.append({
                    "title": f"优化{detector}的recall - 降低CPA阈值",
                    "detector": detector,
                    "scope": "阈值调整",
                    "changes": [{
                        "file": "src/meta/diagnoser/detectors/fatigue_detector.py",
                        "type": "threshold_tuning",
                        "parameter": "cpa_increase_threshold",
                        "from": 1.2,
                        "to": 1.15,
                        "reason": "降低CPA增长阈值从20%到15%，捕捉更多早期疲劳信号"
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
                })

            # Strategy 2: Increase consecutive_days (if not tried)
            if "consecutive_days" not in tried_parameters and recall < 0.6:
                # Skip this strategy - changing from 1 to 2 is 100% increase
                pass

            # Strategy 3: Adjust window_size_days (if not tried)
            if "window_size_days" not in tried_parameters:
                strategies.append({
                    "title": f"优化{detector}的稳定性 - 增加窗口大小",
                    "detector": detector,
                    "scope": "阈值调整",
                    "changes": [{
                        "file": "src/meta/diagnoser/detectors/fatigue_detector.py",
                        "type": "threshold_tuning",
                        "parameter": "window_size_days",
                        "from": 21,
                        "to": 23,
                        "reason": "增加窗口大小从21天到23天，提高检测稳定性"
                    }],
                    "constraints": [
                        "严禁修改核心检测逻辑",
                        "严禁针对测试集硬编码",
                        "保持向后兼容"
                    ],
                    "expected_outcome": {
                        "f1_score": f"{f1 * 1.02:.3f} (+2%)",
                        "precision": ">=0.98",
                        "recall": ">=0.50"
                    },
                    "rollback_plan": "如果F1下降，立即回滚"
                })

            # Strategy 4: Lower min_golden_days (if not tried)
            if "min_golden_days" not in tried_parameters and recall < 0.6:
                strategies.append({
                    "title": f"优化{detector}的recall - 降低黄金期天数要求",
                    "detector": detector,
                    "scope": "阈值调整",
                    "changes": [{
                        "file": "src/meta/diagnoser/detectors/fatigue_detector.py",
                        "type": "threshold_tuning",
                        "parameter": "min_golden_days",
                        "from": 2,
                        "to": 1,
                        "reason": "降低黄金期天数要求从2天到1天，捕捉更多早期疲劳"
                    }],
                    "constraints": [
                        "严禁修改核心检测逻辑",
                        "严禁针对测试集硬编码",
                        "保持向后兼容"
                    ],
                    "expected_outcome": {
                        "f1_score": f"{f1 * 1.04:.3f} (+4%)",
                        "recall": f"{recall * 1.08:.3f} (+8%)",
                        "precision": ">=0.92"
                    },
                    "rollback_plan": "如果precision < 0.90，立即回滚"
                })

            # Return first available strategy
            return strategies[0] if strategies else None

        return None

    def _generate_experiment_spec(
        self,
        detector: str,
        current_metrics: Dict[str, float],
        target_metrics: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """Generate experiment spec based on current metrics."""
        recall = current_metrics.get("recall", 0)
        fn = current_metrics.get("fn", 0)
        f1 = current_metrics.get("f1_score", 0)

        # Check if optimization is needed
        if detector == "FatigueDetector" and recall < 0.6 and fn > 50:
            return {
                "title": f"优化{detector}的recall - 降低CPA阈值",
                "detector": detector,
                "scope": "阈值调整",
                "changes": [{
                    "file": "src/meta/diagnoser/detectors/fatigue_detector.py",
                    "type": "threshold_tuning",
                    "parameter": "cpa_increase_threshold",
                    "from": 1.2,
                    "to": 1.15,
                    "reason": "降低CPA增长阈值从20%到15%，捕捉更多早期疲劳信号"
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

        return None

    def _modify_code_with_retry(
        self,
        spec: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Modify code with automatic retry on failure."""
        changes = spec.get("changes", [])
        if not changes:
            return {"success": False, "error": "No changes to make"}

        change = changes[0]
        file_path = change.get("file", "")
        parameter = change.get("parameter", "")
        old_value = change.get("from")
        new_value = change.get("to")

        print(f"\nModifying: {file_path}")
        print(f"  Parameter: {parameter}")
        print(f"  Change: {old_value} → {new_value}")

        full_path = project_root / file_path

        # Create backup
        backup_path = full_path.with_suffix('.py.backup')
        import shutil
        shutil.copy2(full_path, backup_path)
        print(f"  Backup created: {backup_path.name}")

        # Read current file
        with open(full_path, 'r') as f:
            content = f.read()

        # Find and replace the threshold
        # Pattern: "parameter": value,  # optional comment
        # Match the entire line - parameter, value, and optional comment
        pattern = rf'("{parameter}"\s*:\s*)({old_value})(\s*,\s*[^\n]*)'
        replacement = rf'\g<1>{new_value},  # Optimized: {old_value} → {new_value}'

        if not re.search(pattern, content):
            return {
                "success": False,
                "error": f"Pattern not found: Parameter={parameter}, Value={old_value}",
                "backup_path": backup_path
            }

        new_content = re.sub(pattern, replacement, content)

        # Verify the change
        if str(old_value) in new_content and str(new_value) not in new_content:
            return {
                "success": False,
                "error": "Replacement failed - old value still present",
                "backup_path": backup_path
            }

        # Write the modified content
        with open(full_path, 'w') as f:
            f.write(new_content)

        # Verify syntax
        try:
            subprocess.run(
                ["python3", "-m", "py_compile", str(full_path)],
                capture_output=True,
                check=True,
                timeout=10
            )
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Syntax error: {e.stderr.decode()}",
                "backup_path": backup_path
            }

        print(f"  ✅ Code modified and syntax checked")

        return {
            "success": True,
            "file_path": file_path,
            "backup_path": backup_path,
            "parameter": parameter,
            "old_value": old_value,
            "new_value": new_value
        }

    def _review_with_auto_fix(
        self,
        spec: Dict[str, Any],
        impl_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review code with automatic fix capabilities."""

        # Check for common issues
        issues = []
        score = 100

        # Check 1: Single parameter change
        if len(spec.get("changes", [])) > 1:
            issues.append("Multiple parameters changed - violates single variable principle")
            score -= 30

        # Check 2: Threshold change type
        if spec.get("changes", [{}])[0].get("type") != "threshold_tuning":
            issues.append("Not a threshold tuning - too risky")
            score -= 40

        # Check 3: Reasonable change magnitude
        old_val = spec.get("changes", [{}])[0].get("from", 0)
        new_val = spec.get("changes", [{}])[0].get("to", 0)
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            change_pct = abs(new_val - old_val) / old_val
            if change_pct > 0.2:  # More than 20% change
                issues.append(f"Change too large: {change_pct*100:.1f}%")
                score -= 20

        # Check 4: Code integrity (verify file still has all thresholds)
        file_path = project_root / spec.get("changes", [{}])[0].get("file", "")
        with open(file_path, 'r') as f:
            content = f.read()

        # Verify DEFAULT_THRESHOLDS still has all keys
        required_keys = ["window_size_days", "golden_min_freq", "golden_max_freq",
                        "fatigue_freq_threshold", "cpa_increase_threshold",
                        "consecutive_days", "min_golden_days"]

        if "FatigueDetector" in spec.get("detector", ""):
            for key in required_keys:
                if f'"{key}"' not in content:
                    issues.append(f"Missing required key: {key}")
                    score -= 20

        # Decision
        if score >= 75 and not issues:
            return {
                "passed": True,
                "score": score,
                "issues": [],
                "error": None
            }
        else:
            return {
                "passed": False,
                "score": score,
                "issues": issues,
                "error": f"Review failed (score: {score}/100)"
            }

    def _run_evaluation_with_retry(
        self,
        detector: str,
        baseline_metrics: Dict[str, float],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Run evaluation with automatic retry."""

        # Map detector to evaluation script
        script_map = {
            "FatigueDetector": "evaluate_fatigue.py",
            "LatencyDetector": "evaluate_latency.py",
            "DarkHoursDetector": "evaluate_dark_hours.py"
        }

        script = script_map.get(detector)
        if not script:
            return {"success": False, "error": f"No evaluation script for {detector}"}

        script_path = project_root / "scripts" / script

        print(f"\nRunning evaluation script: {script}")
        print(f"  This may take 1-2 minutes...")

        # Run evaluation
        try:
            result = subprocess.run(
                ["python3", str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
                cwd=str(project_root)
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Evaluation script failed: {result.stderr[-500:] if result.stderr else 'Unknown error'}",
                    "details": {"stdout": result.stdout, "stderr": result.stderr}
                }

            print(f"  ✅ Evaluation script completed")

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Evaluation script timed out (5 minutes)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run evaluation: {str(e)}"
            }

        # Load new metrics
        new_metrics = self._load_current_metrics(detector)
        if not new_metrics:
            return {
                "success": False,
                "error": "Failed to load new metrics after evaluation"
            }

        # Calculate lift
        lift = {}
        for metric in ["f1_score", "precision", "recall"]:
            baseline_val = baseline_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            if baseline_val > 0:
                lift[metric] = (new_val - baseline_val) / baseline_val
            else:
                lift[metric] = 0.0

        # Check acceptance criteria
        f1_improved = lift["f1_score"] >= 0.03  # At least 3% improvement
        precision_ok = new_metrics["precision"] >= 0.85  # At least 85% precision

        if f1_improved and precision_ok:
            return {
                "success": True,
                "baseline": baseline_metrics,
                "new": new_metrics,
                "lift": lift,
                "details": {
                    "f1_improvement": lift["f1_score"],
                    "precision_ok": precision_ok
                }
            }
        else:
            reasons = []
            if not f1_improved:
                reasons.append(f"F1 not improved enough ({lift['f1_score']:.1%})")
            if not precision_ok:
                reasons.append(f"Precision too low ({new_metrics['precision']:.2%})")

            return {
                "success": False,
                "error": f"Acceptance criteria not met: {', '.join(reasons)}",
                "baseline": baseline_metrics,
                "new": new_metrics,
                "lift": lift
            }

    def _rollback_changes(self, impl_result: Dict[str, Any]):
        """Rollback code changes."""
        backup_path = impl_result.get("backup_path")
        if not backup_path:
            print("⚠️  No backup path found, cannot rollback")
            return

        backup_path = Path(backup_path)
        if not backup_path.exists():
            print(f"⚠️  Backup file not found: {backup_path}")
            return

        # Restore from backup
        file_path = backup_path.with_suffix('.py')
        import shutil
        shutil.copy2(backup_path, file_path)

        # Remove backup
        backup_path.unlink()

        print(f"✅ Rolled back changes from: {backup_path.name}")


def main():
    """Run auto-fix optimization cycle."""
    import argparse

    parser = argparse.ArgumentParser(description="Run auto-fix optimization cycle")
    parser.add_argument(
        "--detector",
        default="FatigueDetector",
        choices=["FatigueDetector", "LatencyDetector", "DarkHoursDetector"],
        help="Detector to optimize"
    )
    parser.add_argument(
        "--target-f1",
        type=float,
        default=0.75,
        help="Target F1-score"
    )

    args = parser.parse_args()

    orchestrator = AutoFixOrchestrator(max_retries=3)

    result = orchestrator.run_optimization_with_auto_fix(
        detector=args.detector,
        target_metrics={"f1_score": args.target_f1}
    )

    print(f"\nFinal result: {result['status']}")

    if result["status"] == "success":
        print(f"Experiment ID: {result.get('experiment_id')}")
        print(f"F1 Lift: {result['lift']['f1_score']:.2%}")
        return 0
    else:
        print(f"Error: {result.get('error', 'Unknown')}")
        if result.get("phase"):
            print(f"Failed at phase: {result['phase']}")
        return 1


if __name__ == "__main__":
    exit(main())
