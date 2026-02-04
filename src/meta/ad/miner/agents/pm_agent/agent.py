"""
PM Agent - Pattern Mining Strategist & Experiment Planner

Objective: Maximize mining ROI by discovering high-impact ad creative patterns
while controlling evolution risk.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentSpec:
    """Structured experiment specification."""
    id: str
    timestamp: str
    objective: str
    domain: Optional[str] = None  # e.g., "gaming_ads", "ecommerce"
    approach: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    historical_context: List[Dict] = field(default_factory=list)
    priority: str = "medium"  # low, medium, high, critical


class PMAgent:
    """
    Pattern Mining Strategist & Experiment Planner Agent.

    Converts performance insights into specific mining experiment specs,
    sets mining parameters, and defines experiment boundaries.
    """

    def __init__(
        self,
        memory_agent=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PM Agent.

        Args:
            memory_agent: Memory Agent instance for historical context
            config: Agent configuration
        """
        self.memory_agent = memory_agent
        self.config = config or {}
        self.active_experiments: List[ExperimentSpec] = []

        # Ad Miner specific parameters
        self.parameter_ranges = {
            "winner_quantile": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            "loser_quantile": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "confidence_threshold": [0.50, 0.60, 0.70, 0.80, 0.90],
            "min_sample_size": [30, 50, 100, 200],
            "min_prevalence": [0.05, 0.10, 0.15, 0.20, 0.25],  # Min pattern prevalence
        }

        # Ad Miner specific optimization objectives
        self.objective_templates = {
            # === PATTERN DISCOVERY OBJECTIVES ===
            "discover_high_lift_patterns": {
                "description": "Find new feature combinations with high ROAS lift",
                "business_value": "Directly impacts ad performance and revenue",
                "typical_approaches": [
                    "Lower confidence threshold to capture more patterns",
                    "Increase winner quantile for stricter analysis",
                    "Add interaction features (e.g., lighting × angle)",
                    "Implement feature clustering for rare combinations",
                    "Analyze pattern prevalence across customer verticals",
                ],
                "success_metrics": [
                    "new_pattern_count",  # Number of new patterns discovered
                    "avg_lift_score",       # Average ROAS lift of patterns
                    "pattern_prevalence",   # How often patterns appear in winners
                ],
                "typical_files": [
                    "stages/miner.py",
                    "stages/miner_v2.py",
                    "stages/synthesizer.py",
                ],
                "target_improvement": "Discover 5+ new patterns with >1.5x lift",
            },

            "increase_winner_precision": {
                "description": "Improve accuracy of identifying true winner ads",
                "business_value": "Reduces wasted spend on false-positive patterns",
                "typical_approaches": [
                    "Tighten winner quantile threshold",
                    "Add temporal validation (patterns must persist over time)",
                    "Implement cross-validation across campaigns",
                    "Filter seasonal anomalies",
                ],
                "success_metrics": [
                    "winner_precision",      # True winners / predicted winners
                    "false_positive_rate",   # Predicted winners that actually lose
                    "pattern_stability",     # Consistency across time periods
                ],
                "typical_files": [
                    "stages/miner_v2.py",
                    "validation/input_validator.py",
                ],
                "target_improvement": "Increase precision from 70% to 85%+",
            },

            # === PSYCHOLOGY CLASSIFICATION OBJECTIVES ===
            "improve_psychology_accuracy": {
                "description": "Improve accuracy of psychology type classification",
                "business_value": "Better creative recommendations aligned with psychological triggers",
                "typical_approaches": [
                    "Add vertical-specific psychology keywords (gaming, ecommerce, etc.)",
                    "Fine-tune GPT-4 prompts for psychology extraction",
                    "Implement ensemble classifier (rule-based + VLM)",
                    "Use VLM for direct image psychology analysis",
                    "Add psychology interaction detection (e.g., Trust + Luxury)",
                ],
                "success_metrics": [
                    "psychology_accuracy",    # Classification accuracy
                    "f1_score",               # F1 score across psychology types
                    "coverage_rate",          # % of ads successfully classified
                ],
                "typical_files": [
                    "features/psychology_classifier.py",
                    "stages/psych_composer.py",
                    "stages/miner_v2.py",
                ],
                "target_improvement": "Improve accuracy from 67% to 80%+",
            },

            "discover_new_psychology_triggers": {
                "description": "Identify new psychological patterns in winner ads",
                "business_value": "Uncover untapped creative strategies",
                "typical_approaches": [
                    "Cluster winner ad features by psychology themes",
                    "Analyze emotional tone distribution in winners",
                    "Detect emergent psychology patterns (new combinations)",
                    "Use topic modeling on winner creative descriptions",
                ],
                "success_metrics": [
                    "new_psychology_patterns",  # New patterns discovered
                    "pattern_lift",              # Lift of new patterns
                    "pattern_coverage",          # % of winners covered
                ],
                "typical_files": [
                    "stages/miner_v2.py",
                    "features/psychology_classifier.py",
                ],
                "target_improvement": "Discover 3+ new psychology patterns with >2x lift",
            },

            # === FEATURE EXTRACTION OBJECTIVES ===
            "improve_visual_feature_extraction": {
                "description": "Enhance extraction of visual features (camera_angle, lighting, etc.)",
                "business_value": "More accurate pattern mining with better features",
                "typical_approaches": [
                    "Update GPT-4 prompts with few-shot examples",
                    "Add new feature categories (e.g., product_position, color_scheme)",
                    "Implement ensemble of multiple extractors",
                    "Use VLM to detect subtle visual attributes",
                    "Add feature confidence scoring",
                ],
                "success_metrics": [
                    "extraction_accuracy",     # Accuracy vs human labels
                    "feature_coverage",        # % of ads with all features extracted
                    "feature_confidence",      # Average confidence of extracted features
                ],
                "typical_files": [
                    "features/extractors/gpt4_feature_extractor.py",
                    "features/transformers/gpt4_feature_transformer.py",
                    "features/lib/parsers.py",
                ],
                "target_improvement": "Improve extraction accuracy from 75% to 90%+",
            },

            # === PERFORMANCE OBJECTIVES ===
            "reduce_processing_time": {
                "description": "Optimize mining pipeline for faster processing",
                "business_value": "Handle larger datasets and enable real-time mining",
                "typical_approaches": [
                    "Parallelize feature extraction across multiple ads",
                    "Cache GPT-4 API results for duplicate creatives",
                    "Optimize data structures (use Polars instead of Pandas)",
                    "Implement incremental mining (only process new ads)",
                    "Batch GPT-4 API calls",
                ],
                "success_metrics": [
                    "processing_time_reduction",  # % reduction in time
                    "throughput_improvement",      # Ads processed per second
                    "memory_efficiency",          # Memory usage reduction
                ],
                "typical_files": [
                    "pipeline.py",
                    "pipeline_v2.py",
                    "features/extract.py",
                ],
                "target_improvement": "Reduce processing time by 50%+",
            },

            "reduce_false_positive_patterns": {
                "description": "Reduce patterns that predict well but don't perform in reality",
                "business_value": "Prevent wasted spend on ineffective creative strategies",
                "typical_approaches": [
                    "Implement hold-out validation set",
                    "Require temporal persistence (pattern must work across time)",
                    "Add statistical significance testing",
                    "Filter low-prevalence patterns",
                    "Implement ensemble validation across multiple metrics",
                ],
                "success_metrics": [
                    "false_positive_rate",     # % of patterns that fail in validation
                    "pattern_reliability",     # % of patterns that persist
                    "validation_coverage",     # % of patterns tested on hold-out set
                ],
                "typical_files": [
                    "stages/miner.py",
                    "validation/output_validator.py",
                ],
                "target_improvement": "Reduce false positive rate from 30% to 10% or less",
            },

            # === DOMAIN-SPECIFIC OBJECTIVES ===
            "optimize_vertical_performance": {
                "description": "Improve pattern mining for specific vertical (e.g., gaming, ecommerce)",
                "business_value": "Better recommendations for high-value customer segments",
                "typical_approaches": [
                    "Add vertical-specific visual features",
                    "Tune psychology keywords for vertical",
                    "Adjust confidence thresholds by vertical",
                    "Implement vertical-specific pattern filtering",
                ],
                "success_metrics": [
                    "vertical_lift_score",     # ROAS lift for target vertical
                    "vertical_pattern_count",  # Number of patterns discovered
                    "vertical_coverage",        # % of vertical ads covered
                ],
                "typical_files": [
                    "features/psychology_classifier.py",
                    "stages/miner_v2.py",
                ],
                "target_improvement": "Achieve >2x lift for target vertical",
            },
        }

    def create_experiment_spec(
        self,
        objective: str,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExperimentSpec:
        """
        Create a structured experiment specification.

        Args:
            objective: High-level objective (e.g., "improve_psychology_classification")
            domain: Specific domain (e.g., "gaming_ads", "ecommerce")
            context: Additional context (performance issues, metrics, etc.)

        Returns:
            ExperimentSpec: Structured experiment specification
        """
        timestamp = datetime.now().isoformat()
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"PM Agent: Creating experiment {exp_id}")
        logger.info(f"  Objective: {objective}")
        logger.info(f"  Domain: {domain}")

        # Retrieve historical context from Memory Agent
        historical_context = []
        if self.memory_agent:
            historical_context = self.memory_agent.search_similar(
                query=f"{objective} {domain}" if domain else objective,
                top_k=3,
            )
            logger.info(f"  Retrieved {len(historical_context)} historical experiments")

        # Get objective template
        template = self.objective_templates.get(objective, {
            "description": objective,
            "typical_approaches": ["Analyze and optimize"],
            "success_metrics": ["lift_score"],
            "typical_files": [],
        })

        # Determine approach based on historical context
        approach = self._select_approach(
            objective,
            template,
            historical_context,
            context,
        )

        # Set constraints based on domain and historical lessons
        constraints = self._set_constraints(
            objective,
            domain,
            historical_context,
        )

        # Define success criteria
        success_criteria = self._define_success_criteria(
            objective,
            template,
            context,
        )

        # Set mining parameters
        parameters = self._set_parameters(
            objective,
            context,
            historical_context,
        )

        # Generate rationale
        rationale = self._generate_rationale(
            objective,
            approach,
            historical_context,
            context,
        )

        spec = ExperimentSpec(
            id=exp_id,
            timestamp=timestamp,
            objective=objective,
            domain=domain,
            approach=approach,
            constraints=constraints,
            success_criteria=success_criteria,
            parameters=parameters,
            rationale=rationale,
            historical_context=historical_context,
            priority=self._assess_priority(context, historical_context),
        )

        self.active_experiments.append(spec)

        logger.info(f"✓ Experiment spec created:")
        logger.info(f"  Approach: {approach}")
        logger.info(f"  Priority: {spec.priority}")
        logger.info(f"  Constraints: {len(constraints)} items")

        return spec

    def _select_approach(
        self,
        objective: str,
        template: Dict[str, Any],
        historical_context: List[Dict],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Select the best approach based on history and context."""

        # Check if similar experiments succeeded in the past
        successful_approaches = []
        failed_approaches = []

        for exp in historical_context:
            if exp.get("judge_decision") == "PASS":
                successful_approaches.append(exp.get("approach", ""))
            elif exp.get("judge_decision") == "FAIL":
                failed_approaches.append(exp.get("approach", ""))

        # Prioritize successful approaches
        if successful_approaches:
            # Return most recent successful approach
            return successful_approaches[0]

        # Avoid failed approaches
        available = template.get("typical_approaches", [])
        for approach in failed_approaches:
            if approach in available:
                available.remove(approach)

        # Return first available approach
        return available[0] if available else "Analyze and optimize"

    def _set_constraints(
        self,
        objective: str,
        domain: Optional[str],
        historical_context: List[Dict],
    ) -> Dict[str, Any]:
        """Set experiment constraints based on domain and history."""

        constraints = {
            "max_files_to_modify": 3,
            "require_tests": True,
            "backward_compatible": True,
            "allow_config_changes": False,
        }

        # Learn from historical failures
        for exp in historical_context:
            failure_reason = exp.get("failure_reason", "")
            if "overfitting" in failure_reason.lower():
                constraints["max_complexity_increase"] = 0.2  # 20%
            if "regression" in failure_reason.lower():
                constraints["require_regression_test"] = True
            if "test_leakage" in failure_reason.lower():
                constraints["forbidden_test_access"] = True

        # Domain-specific constraints
        if domain:
            constraints["scope"] = f"domain_specific:{domain}"

        # Objective-specific constraints
        if objective == "improve_psychology_classification":
            constraints["allowed_files"] = [
                "psychology_classifier.py",
                "psych_composer.py",
                "miner_v2.py",
            ]
        elif objective == "discover_new_patterns":
            constraints["allowed_files"] = [
                "miner.py",
                "miner_v2.py",
                "synthesizer.py",
                "patterns_io.py",
            ]
        elif objective == "improve_feature_extraction":
            constraints["allowed_files"] = [
                "gpt4_feature_extractor.py",
                "gpt4_feature_transformer.py",
                "parsers.py",
            ]

        return constraints

    def _define_success_criteria(
        self,
        objective: str,
        template: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Define success criteria for the experiment."""

        criteria = {}

        # Default criteria
        criteria["min_lift_score"] = 5.0  # 5% minimum improvement
        criteria["max_regression_rate"] = 2.0  # Max 2% regression
        criteria["statistical_significance"] = 0.05  # p < 0.05

        # Objective-specific criteria
        if objective == "improve_psychology_classification":
            criteria["accuracy_improvement"] = ">10%"
            criteria["min_f1_score"] = 0.75
        elif objective == "discover_new_patterns":
            criteria["min_new_patterns"] = 3
            criteria["avg_pattern_lift"] = ">1.5x"
        elif objective == "reduce_processing_time":
            criteria["time_reduction"] = ">20%"
            criteria["accuracy_loss"] = "<2%"

        # Context-aware criteria
        if context:
            current_metrics = context.get("current_metrics", {})
            if "accuracy" in current_metrics:
                baseline = current_metrics["accuracy"]
                criteria["target_accuracy"] = f">{baseline + 10}%"

        return criteria

    def _set_parameters(
        self,
        objective: str,
        context: Optional[Dict[str, Any]],
        historical_context: List[Dict],
    ) -> Dict[str, Any]:
        """Set mining parameters for the experiment."""

        # Start with defaults
        parameters = {
            "winner_quantile": 0.80,
            "loser_quantile": 0.20,
            "confidence_threshold": 0.70,
            "min_sample_size": 50,
        }

        # Adjust based on context
        if context:
            current_params = context.get("current_parameters", {})
            parameters.update(current_params)

        # Learn from successful experiments
        for exp in historical_context:
            if exp.get("judge_decision") == "PASS":
                successful_params = exp.get("parameters", {})
                # Use parameters that led to success
                for key, value in successful_params.items():
                    if key in parameters:
                        parameters[key] = value

        return parameters

    def _generate_rationale(
        self,
        objective: str,
        approach: str,
        historical_context: List[Dict],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Generate rationale for the experiment."""

        rationale_parts = []

        # Base rationale
        rationale_parts.append(f"Objective: {objective}")

        # Approach rationale
        rationale_parts.append(f"Selected approach: {approach}")

        # Historical rationale
        if historical_context:
            similar_count = len(historical_context)
            successful = sum(1 for e in historical_context if e.get("judge_decision") == "PASS")
            rationale_parts.append(
                f"Based on {similar_count} similar experiments ({successful} successful)"
            )

        # Context rationale
        if context:
            issue = context.get("issue", "")
            if issue:
                rationale_parts.append(f"Addressing: {issue}")

        return ". ".join(rationale_parts) + "."

    def _assess_priority(
        self,
        context: Optional[Dict[str, Any]],
        historical_context: List[Dict],
    ) -> str:
        """Assess experiment priority."""

        # Critical if there's a significant performance issue
        if context:
            severity = context.get("severity", "low")
            if severity == "critical":
                return "critical"

        # High if recent similar experiments succeeded
        recent_success = False
        for exp in historical_context:
            if exp.get("judge_decision") == "PASS":
                days_ago = (datetime.now() - datetime.fromisoformat(exp["timestamp"])).days
                if days_ago < 7:
                    recent_success = True

        if recent_success:
            return "high"

        return "medium"

    def query_mining_performance(self) -> Dict[str, Any]:
        """
        Query current mining performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        # This would integrate with Monitor Agent or data sources
        # For now, return placeholder
        return {
            "avg_pattern_lift": 2.1,
            "psychology_accuracy": 0.78,
            "processing_time": 45.2,  # seconds
            "new_patterns_discovered": 12,
        }

    def set_mining_parameters(
        self,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update mining parameters.

        Args:
            parameters: New parameter values

        Returns:
            Updated parameters
        """
        # Validate parameters
        for key, value in parameters.items():
            if key in self.parameter_ranges:
                if value not in self.parameter_ranges[key]:
                    raise ValueError(
                        f"Invalid value for {key}: {value}. "
                        f"Must be one of {self.parameter_ranges[key]}"
                    )

        return parameters

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "active_experiments": len(self.active_experiments),
            "config": self.config,
            "parameter_ranges": self.parameter_ranges,
        }
