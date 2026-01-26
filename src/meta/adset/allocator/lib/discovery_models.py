"""Data models for pattern discovery system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class DiscoveredRule:
    """A rule discovered from data mining.

    Attributes:
        rule_id: Unique identifier for this rule
        conditions: Feature conditions that trigger this rule
        outcome: Expected outcome ("increase" or "decrease" budget)
        adjustment_factor: Budget adjustment multiplier (e.g., 1.15 = +15%)
        support: Number of samples matching this rule in training data
        confidence: Accuracy of the rule (0-1)
        lift: Improvement over baseline ROAS
        feature_importance: Importance scores for each feature in the rule
        discovery_date: When this rule was discovered
        discovery_method: How the rule was found (decision_tree, association, shap, rl)
        validation_metric: Forward-looking ROAS from validation (if validated)
        tier: Priority tier for rule execution (similar to DecisionRules)
        metadata: Additional information about the rule
    """

    rule_id: str
    conditions: Dict[str, Any]
    outcome: str  # "increase" or "decrease"
    adjustment_factor: float
    support: int
    confidence: float
    lift: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    discovery_date: datetime = field(default_factory=datetime.now)
    discovery_method: str = "unknown"
    validation_metric: Optional[float] = None
    tier: Optional[int] = None  # DecisionRules tier (2, 3, 4, 5, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "conditions": self.conditions,
            "outcome": self.outcome,
            "adjustment_factor": self.adjustment_factor,
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "feature_importance": self.feature_importance,
            "discovery_date": self.discovery_date.isoformat(),
            "discovery_method": self.discovery_method,
            "validation_metric": self.validation_metric,
            "tier": self.tier,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveredRule":
        """Create rule from dictionary representation."""
        data = data.copy()
        if isinstance(data.get("discovery_date"), str):
            data["discovery_date"] = datetime.fromisoformat(data["discovery_date"])
        return cls(**data)


@dataclass
class ValidationResult:
    """Validation results for a discovered rule.

    Attributes:
        rule_id: ID of the rule being validated
        mean_roas: Mean ROAS achieved when applying this rule
        std_roas: Standard deviation of ROAS
        budget_utilization: Budget utilization rate (0-1)
        safety_violations: Number of safety constraint violations
        recommendation: Whether to deploy, test further, or reject
        validation_samples: Number of samples used for validation
        forward_roas: Forward-looking ROAS (next period performance)
        backtest_roas: Historical backtest ROAS
        confidence_interval: 95% confidence interval for ROAS
        metadata: Additional validation information
    """

    rule_id: str
    mean_roas: float
    std_roas: float
    budget_utilization: float
    safety_violations: int
    recommendation: str  # "deploy", "test", "reject"
    validation_samples: int = 0
    forward_roas: Optional[float] = None
    backtest_roas: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "rule_id": self.rule_id,
            "mean_roas": self.mean_roas,
            "std_roas": self.std_roas,
            "budget_utilization": self.budget_utilization,
            "safety_violations": self.safety_violations,
            "recommendation": self.recommendation,
            "validation_samples": self.validation_samples,
            "forward_roas": self.forward_roas,
            "backtest_roas": self.backtest_roas,
            "confidence_interval": self.confidence_interval,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Create validation result from dictionary."""
        return cls(**data)
