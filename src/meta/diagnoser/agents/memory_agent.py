"""
Memory Agent - Organizational memory and knowledge base.

Records and retrieves experiment history to prevent repeated failures.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from .memory.storage import MemoryStorage


class MemoryAgent:
    """
    Memory Agent for managing experiment history.

    Provides:
    - Storage of experiment records
    - Query by detector, outcome, tags
    - Warning signals for repeated failures
    - Performance trend analysis
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize Memory Agent.

        Args:
            storage_dir: Directory for storing experiment records
        """
        self.storage = MemoryStorage(storage_dir)
        self._load_system_prompt()

    def _load_system_prompt(self):
        """Load the system prompt for this agent."""
        prompt_path = Path(__file__).parent / "prompts" / "memory_system_prompt.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = "Memory Agent - Organizational memory and knowledge base."

    def query(
        self,
        query_type: str,
        detector: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Query memory for relevant historical data.

        Args:
            query_type: Type of query (SIMILAR_EXPERIMENTS, FAILURE_PATTERNS, SUCCESS_PATTERNS)
            detector: Detector name
            context: Query context (tags, approach, etc.)

        Returns:
            Query results with relevant experiments and warnings
        """
        results = []
        warnings = []

        if query_type == "SIMILAR_EXPERIMENTS":
            # Find similar experiments by detector and tags
            tags = context.get("tags", [])
            results = self.storage.query_experiments(
                detector=detector,
                tags=tags,
                limit=5
            )

        elif query_type == "FAILURE_PATTERNS":
            # Find similar failure cases
            approach = context.get("approach", "")
            if self.storage.check_repeated_failures(detector, approach):
                failures_count = self._count_similar_failures(detector, approach)
                warnings.append({
                    "type": "REPEATED_FAILURE",
                    "message": f"过去有{failures_count}次类似的优化失败",
                    "action": "建议改变优化方向"
                })

            # Load failure records
            results = self._load_failure_records(detector, approach)

        elif query_type == "SUCCESS_PATTERNS":
            # Find successful experiments and patterns
            results = self.storage.query_experiments(
                detector=detector,
                outcome="SUCCESS",
                limit=5
            )

        # Add relevance scores
        for result in results:
            result["relevance_score"] = self._calculate_relevance(result, context)

        # Sort by relevance
        results.sort(key=lambda r: r.get("relevance_score", 0), reverse=True)

        # Check for additional warning signals
        warnings.extend(self._check_warning_signals(detector, context))

        return {
            "query_result": {
                "query_type": query_type,
                "results": results[:5],
            },
            "warnings": warnings,
            "context_provided": {
                "similar_experiments": len(results),
                "failure_cases": sum(1 for r in results if r.get("outcome") == "FAILURE"),
                "success_patterns": sum(1 for r in results if r.get("outcome") == "SUCCESS"),
            }
        }

    def save_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """
        Save a complete experiment record.

        Args:
            experiment_data: Full experiment data including spec, implementation, review, evaluation

        Returns:
            Experiment ID
        """
        experiment_id = self.storage.save_experiment(experiment_data)

        # If successful, extract patterns
        if experiment_data.get("outcome") == "SUCCESS":
            self._extract_and_save_pattern(experiment_data)

        # If failed, save to failure database
        elif experiment_data.get("outcome") == "FAILURE":
            self._save_failure_record(experiment_data)

        return experiment_id

    def get_recent_experiments(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent experiments across all detectors.

        Args:
            count: Number of experiments to return

        Returns:
            List of recent experiment data
        """
        return self.storage.get_recent_experiments(count)

    def get_performance_trend(self, detector: str) -> str:
        """
        Get performance trend for a detector.

        Args:
            detector: Detector name

        Returns:
            Trend: "IMPROVING", "DECLINING", or "STABLE"
        """
        return self.storage.get_performance_trend(detector)

    def _calculate_relevance(self, experiment: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate relevance score for an experiment."""
        score = 0.5  # Base score

        # Boost if detector matches
        if experiment.get("detector") == context.get("detector"):
            score += 0.3

        # Boost if tags match
        exp_tags = set(experiment.get("tags", []))
        ctx_tags = set(context.get("tags", []))
        if exp_tags & ctx_tags:  # Intersection
            score += 0.2 * len(exp_tags & ctx_tags) / max(len(exp_tags), 1)

        return min(score, 1.0)

    def _check_warning_signals(self, detector: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for warning signals in the context."""
        warnings = []

        # Check for overfitting risk
        recent_experiments = self.get_recent_experiments(count=10)
        if all(exp.get("outcome") == "SUCCESS" for exp in recent_experiments if exp.get("detector") == detector):
            warnings.append({
                "type": "OVERFITTING_RISK",
                "message": "连续10次实验成功，可能过拟合测试集",
                "action": "建议在真实数据上验证"
            })

        # Check for performance decline
        trend = self.get_performance_trend(detector)
        if trend == "DECLINING":
            warnings.append({
                "type": "PERFORMANCE_DECLINE",
                "message": "最近5次实验F1呈下降趋势",
                "action": "建议暂停优化，review代码"
            })

        return warnings

    def _count_similar_failures(self, detector: str, approach: str) -> int:
        """Count similar failures for an approach."""
        failures_dir = Path(self.storage.failures_dir)
        count = 0

        for file_path in failures_dir.glob(f"fail_{detector}_*.json"):
            with open(file_path, 'r') as f:
                fail_data = json.load(f)
                if approach.lower() in fail_data.get("approach", "").lower():
                    count += 1

        return count

    def _load_failure_records(self, detector: str, approach: str) -> List[Dict[str, Any]]:
        """Load failure records for a detector and approach."""
        failures_dir = Path(self.storage.failures_dir)
        failures = []

        for file_path in failures_dir.glob(f"fail_{detector}_*.json"):
            with open(file_path, 'r') as f:
                fail_data = json.load(f)
                if not approach or approach.lower() in fail_data.get("approach", "").lower():
                    failures.append(fail_data)

        return failures

    def _extract_and_save_pattern(self, experiment_data: Dict[str, Any]):
        """Extract success pattern from a successful experiment."""
        # Simple pattern extraction based on spec changes
        spec = experiment_data.get("spec", {})
        changes = spec.get("changes", [])

        if not changes:
            return

        # Determine pattern type
        change_types = set(c.get("type", "unknown") for c in changes)

        if "threshold_tuning" in change_types:
            pattern = {
                "pattern_id": f"pattern_threshold_{experiment_data.get('detector', 'unknown')}",
                "name": "渐进式阈值优化",
                "description": f"调整{experiment_data.get('detector', 'detector')}的阈值参数",
                "applicable_detectors": [experiment_data.get("detector", "")],
                "success_rate": 1.0,  # Will be averaged over time
                "steps": [
                    "1. 识别需要调整的阈值参数",
                    "2. 小幅调整（5-10%）",
                    "3. 验证precision不下降超过5%",
                    "4. 如果成功，固定该参数"
                ],
                "examples": [experiment_data.get("experiment_id", "")]
            }

            self.storage.save_pattern(pattern)

    def _save_failure_record(self, experiment_data: Dict[str, Any]):
        """Save failure record from a failed experiment."""
        eval_result = experiment_data.get("evaluation", {})

        failure = {
            "failure_id": experiment_data.get("experiment_id", "").replace("exp_", "fail_"),
            "timestamp": experiment_data.get("timestamp", ""),
            "detector": experiment_data.get("detector", ""),
            "approach": json.dumps(experiment_data.get("spec", {})),
            "result": eval_result,
            "root_cause": eval_result.get("reason", "Unknown"),
            "lessons": eval_result.get("suggestions", []),
            "tags": experiment_data.get("tags", []) + ["failure"]
        }

        self.storage.save_failure(failure)
