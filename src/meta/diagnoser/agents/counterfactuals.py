"""
Counterfactual Memory for What-If Scenarios.

Stores and retrieves hypothetical experiment outcomes.
Enable via agent_config.yaml → advanced_features.counterfactuals_enabled = true
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Counterfactual:
    """
    A counterfactual (what-if) scenario.

    Represents: "What if we had tried X instead of Y?"
    """
    timestamp: str
    actual_experiment: Dict[str, Any]
    what_if_changes: Dict[str, Any]
    predicted_outcome: Dict[str, Any]
    validation_status: str  # "pending", "validated", "rejected"
    actual_outcome: Optional[Dict[str, Any]] = None
    notes: str = ""


class CounterfactualMemory:
    """
    Store and retrieve "what-if" scenarios.

    This feature is experimental and disabled by default.
    Enable via: agent_config.yaml → advanced_features.counterfactuals_enabled = true

    Examples:
        # Store a counterfactual
        >>> memory = CounterfactualMemory()
        >>> memory.store(
        ...     actual_exp={"threshold": 1.10, "f1": 0.70},
        ...     what_if={"threshold": 1.05},
        ...     predicted={"f1": 0.73}
        ... )

        # Query for similar counterfactuals
        >>> similar = memory.query({"threshold": 1.08})
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        enabled: bool = False
    ):
        """
        Initialize counterfactual memory.

        Args:
            storage_path: Path to counterfactual storage file
            enabled: Whether counterfactual memory is enabled
        """
        self.enabled = enabled

        if storage_path is None:
            storage_path = (
                Path(__file__).parent / "memory" / "counterfactuals.json"
            )
        else:
            storage_path = Path(storage_path)

        self.storage_path = storage_path
        self.counterfactuals: List[Counterfactual] = []
        self._load()

    def _load(self):
        """Load counterfactuals from storage."""
        if not self.enabled:
            return

        if not self.storage_path.exists():
            logger.info(f"Counterfactual storage not found at {self.storage_path}, creating new")
            self.counterfactuals = []
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self.counterfactuals = [
                Counterfactual(**cf) for cf in data.get("counterfactuals", [])
            ]

            logger.info(f"Loaded {len(self.counterfactuals)} counterfactuals from storage")
        except Exception as e:
            logger.error(f"Failed to load counterfactuals: {e}")
            self.counterfactuals = []

    def _save(self):
        """Save counterfactuals to storage."""
        if not self.enabled:
            return

        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "last_updated": datetime.now().isoformat(),
                "counterfactuals": [asdict(cf) for cf in self.counterfactuals]
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.counterfactuals)} counterfactuals to storage")
        except Exception as e:
            logger.error(f"Failed to save counterfactuals: {e}")

    def store(
        self,
        actual_experiment: Dict[str, Any],
        what_if_changes: Dict[str, Any],
        predicted_outcome: Dict[str, Any],
        notes: str = ""
    ) -> str:
        """
        Store a counterfactual scenario.

        Args:
            actual_experiment: The actual experiment that was run
            what_if_changes: What changes would have been made
            predicted_outcome: Predicted results of those changes
            notes: Additional notes or context

        Returns:
            Counterfactual ID (timestamp-based)

        Example:
            >>> memory.store(
            ...     actual_exp={"title": "Lower threshold", "threshold": 1.10, "f1": 0.70},
            ...     what_if={"threshold": 1.05},
            ...     predicted={"f1": 0.73},
            ...     notes="Aggressive lowering might hurt precision"
            ... )
        """
        if not self.enabled:
            logger.warning("Counterfactual memory not enabled")
            return ""

        cf = Counterfactual(
            timestamp=datetime.now().isoformat(),
            actual_experiment=actual_experiment,
            what_if_changes=what_if_changes,
            predicted_outcome=predicted_outcome,
            validation_status="pending",
            notes=notes
        )

        self.counterfactuals.append(cf)
        self._save()

        cf_id = f"cf_{cf.timestamp.replace(':', '').replace('-', '')}"
        logger.info(f"Stored counterfactual {cf_id}")
        return cf_id

    def query(
        self,
        actual_experiment: Dict[str, Any],
        max_results: int = 5
    ) -> List[Counterfactual]:
        """
        Find counterfactuals for similar experiments.

        Args:
            actual_experiment: Query experiment
            max_results: Maximum number of results

        Returns:
            List of similar counterfactuals

        Example:
            >>> similar = memory.query(
            ...     {"detector": "FatigueDetector", "threshold": 1.10}
            ... )
        """
        if not self.enabled:
            return []

        # Extract key attributes for matching
        detector = actual_experiment.get("detector", "")
        scope = actual_experiment.get("scope", "")

        results = []
        for cf in self.counterfactuals:
            cf_detector = cf.actual_experiment.get("detector", "")
            cf_scope = cf.actual_experiment.get("scope", "")

            # Match if same detector and scope
            if detector and cf_detector:
                if detector.lower() != cf_detector.lower():
                    continue

            if scope and cf_scope:
                if scope != cf_scope:
                    continue

            results.append(cf)

            if len(results) >= max_results:
                break

        return results

    def validate_counterfactual(
        self,
        cf_id: str,
        actual_outcome: Dict[str, Any],
        validation_status: str
    ):
        """
        Update counterfactual with actual outcome.

        Call this when a counterfactual scenario is actually tried.

        Args:
            cf_id: Counterfactual ID
            actual_outcome: Actual results when the what-if was tried
            validation_status: "validated" if prediction was accurate, "rejected" if not
        """
        if not self.enabled:
            return

        for cf in self.counterfactuals:
            cf_timestamp = cf.timestamp.replace(':', '').replace('-', '')
            if cf_timestamp in cf_id:
                cf.actual_outcome = actual_outcome
                cf.validation_status = validation_status
                self._save()

                logger.info(f"Validated counterfactual {cf_id} as {validation_status}")
                return

        logger.warning(f"Counterfactual {cf_id} not found")

    def get_learning_insights(self) -> List[Dict[str, Any]]:
        """
        Get insights from validated counterfactuals.

        Returns:
            List of insights learned from counterfactuals

        Example:
            >>> insights = memory.get_learning_insights()
            >>> for insight in insights:
            ...     print(f"Change: {insight['change']}, Result: {insight['result']}")
        """
        if not self.enabled:
            return []

        insights = []

        for cf in self.counterfactuals:
            if cf.validation_status == "validated":
                # Compare predicted vs actual
                predicted = cf.predicted_outcome
                actual = cf.actual_outcome or {}

                insight = {
                    "what_if_changes": cf.what_if_changes,
                    "predicted_outcome": predicted,
                    "actual_outcome": actual,
                    "was_accurate": self._assess_accuracy(predicted, actual),
                    "notes": cf.notes
                }

                insights.append(insight)

        logger.info(f"Generated {len(insights)} insights from validated counterfactuals")
        return insights

    def _assess_accuracy(
        self,
        predicted: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> bool:
        """
        Assess if prediction was accurate.

        Args:
            predicted: Predicted outcome
            actual: Actual outcome

        Returns:
            True if prediction was reasonably accurate
        """
        # Check if key metrics were within 10% of prediction
        for key in ["f1_score", "precision", "recall"]:
            if key in predicted and key in actual:
                pred_val = predicted[key]
                act_val = actual[key]

                if isinstance(pred_val, str):
                    # Handle string values like "0.73 (+5%)"
                    try:
                        pred_val = float(pred_val.split()[0])
                    except:
                        continue

                if isinstance(act_val, str):
                    try:
                        act_val = float(act_val.split()[0])
                    except:
                        continue

                if isinstance(pred_val, (int, float)) and isinstance(act_val, (int, float)):
                    if abs(pred_val - act_val) > 0.10:
                        return False

        return True

    def get_pending_count(self) -> int:
        """Get count of pending (unvalidated) counterfactuals."""
        return sum(1 for cf in self.counterfactuals if cf.validation_status == "pending")

    def get_validated_count(self) -> int:
        """Get count of validated counterfactuals."""
        return sum(1 for cf in self.counterfactuals if cf.validation_status == "validated")

    def cleanup_old(self, days: int = 90):
        """
        Remove counterfactuals older than specified days.

        Args:
            days: Age threshold in days
        """
        if not self.enabled:
            return

        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        original_count = len(self.counterfactuals)

        self.counterfactuals = [
            cf for cf in self.counterfactuals
            if datetime.fromisoformat(cf.timestamp) > cutoff
        ]

        removed = original_count - len(self.counterfactuals)
        if removed > 0:
            self._save()
            logger.info(f"Cleaned up {removed} old counterfactuals (> {days} days)")
