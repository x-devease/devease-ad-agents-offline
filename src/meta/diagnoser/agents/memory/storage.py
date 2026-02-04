"""
Memory storage for experiment records.

Simple JSON-based storage for agent experiments.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


class MemoryStorage:
    """Storage for experiment records and historical data."""

    def __init__(self, storage_dir: str = None):
        """
        Initialize memory storage.

        Args:
            storage_dir: Directory to store experiment records
        """
        if storage_dir is None:
            # Default to agents/memory directory
            storage_dir = Path(__file__).parent

        self.storage_dir = Path(storage_dir)
        self.experiments_dir = self.storage_dir / "experiments"
        self.failures_dir = self.storage_dir / "failures"
        self.patterns_dir = self.storage_dir / "patterns"

        # Create directories if they don't exist
        for dir_path in [self.experiments_dir, self.failures_dir, self.patterns_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """
        Save an experiment record.

        Args:
            experiment_data: Experiment data dictionary

        Returns:
            Experiment ID
        """
        # Generate experiment ID if not provided
        if "experiment_id" not in experiment_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detector = experiment_data.get("detector", "unknown")
            experiment_id = f"exp_{detector}_{timestamp}"
            experiment_data["experiment_id"] = experiment_id

        # Add timestamp if not provided
        if "timestamp" not in experiment_data:
            experiment_data["timestamp"] = datetime.now().isoformat()

        # Save to file
        file_path = self.experiments_dir / f"{experiment_data['experiment_id']}.json"
        with open(file_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)

        return experiment_data["experiment_id"]

    def load_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load an experiment record.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment data dictionary or None if not found
        """
        file_path = self.experiments_dir / f"{experiment_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return json.load(f)

    def get_recent_experiments(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent experiments.

        Args:
            count: Number of recent experiments to return

        Returns:
            List of experiment data dictionaries
        """
        # Get all experiment files
        experiment_files = list(self.experiments_dir.glob("*.json"))

        # Sort by modification time
        experiment_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Load and return the most recent ones
        experiments = []
        for file_path in experiment_files[:count]:
            with open(file_path, 'r') as f:
                experiments.append(json.load(f))

        return experiments

    def query_experiments(
        self,
        detector: Optional[str] = None,
        outcome: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query experiments by filters.

        Args:
            detector: Filter by detector name
            outcome: Filter by outcome (SUCCESS/FAILURE)
            tags: Filter by tags
            limit: Maximum number of results

        Returns:
            List of matching experiment data dictionaries
        """
        all_experiments = []

        for file_path in self.experiments_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                exp = json.load(f)

                # Apply filters
                if detector and exp.get("detector") != detector:
                    continue

                if outcome and exp.get("outcome") != outcome:
                    continue

                if tags:
                    exp_tags = exp.get("tags", [])
                    if not any(tag in exp_tags for tag in tags):
                        continue

                all_experiments.append(exp)

        # Sort by timestamp (most recent first)
        all_experiments.sort(
            key=lambda e: e.get("timestamp", ""),
            reverse=True
        )

        return all_experiments[:limit]

    def save_failure(self, failure_data: Dict[str, Any]) -> str:
        """
        Save a failure record.

        Args:
            failure_data: Failure data dictionary

        Returns:
            Failure ID
        """
        # Generate failure ID if not provided
        if "failure_id" not in failure_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detector = failure_data.get("detector", "unknown")
            failure_id = f"fail_{detector}_{timestamp}"
            failure_data["failure_id"] = failure_id

        # Add timestamp if not provided
        if "timestamp" not in failure_data:
            failure_data["timestamp"] = datetime.now().isoformat()

        # Save to file
        file_path = self.failures_dir / f"{failure_data['failure_id']}.json"
        with open(file_path, 'w') as f:
            json.dump(failure_data, f, indent=2)

        return failure_data["failure_id"]

    def save_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """
        Save a success pattern.

        Args:
            pattern_data: Pattern data dictionary

        Returns:
            Pattern ID
        """
        # Generate pattern ID if not provided
        if "pattern_id" not in pattern_data:
            pattern_id = f"pattern_{uuid.uuid4().hex[:8]}"
            pattern_data["pattern_id"] = pattern_id

        # Add timestamp if not provided
        if "timestamp" not in pattern_data:
            pattern_data["timestamp"] = datetime.now().isoformat()

        # Save to file
        file_path = self.patterns_dir / f"{pattern_data['pattern_id']}.json"
        with open(file_path, 'w') as f:
            json.dump(pattern_data, f, indent=2)

        return pattern_data["pattern_id"]

    def get_performance_trend(self, detector: str, window: int = 5) -> str:
        """
        Analyze performance trend for a detector.

        Args:
            detector: Detector name
            window: Number of recent experiments to consider

        Returns:
            Trend: "IMPROVING", "DECLINING", or "STABLE"
        """
        experiments = self.query_experiments(detector=detector, limit=window)

        if len(experiments) < 2:
            return "STABLE"

        # Extract F1 scores
        f1_scores = []
        for exp in experiments:
            eval_data = exp.get("evaluation", {})
            if "new_f1" in eval_data:
                f1_scores.append(eval_data["new_f1"])

        if len(f1_scores) < 2:
            return "STABLE"

        # Calculate trend
        recent_avg = sum(f1_scores[:len(f1_scores)//2]) / (len(f1_scores)//2 or 1)
        older_avg = sum(f1_scores[len(f1_scores)//2:]) / (len(f1_scores) - len(f1_scores)//2)

        if recent_avg > older_avg * 1.05:
            return "IMPROVING"
        elif recent_avg < older_avg * 0.95:
            return "DECLINING"
        else:
            return "STABLE"

    def check_repeated_failures(self, detector: str, approach: str, threshold: int = 2) -> bool:
        """
        Check if an approach has failed multiple times.

        Args:
            detector: Detector name
            approach: Approach description
            threshold: Number of failures to trigger warning

        Returns:
            True if repeated failures detected
        """
        failures = list(self.failures_dir.glob(f"fail_{detector}_*.json"))

        matching_failures = 0
        for file_path in failures:
            with open(file_path, 'r') as f:
                fail_data = json.load(f)
                if approach.lower() in fail_data.get("approach", "").lower():
                    matching_failures += 1

        return matching_failures >= threshold
