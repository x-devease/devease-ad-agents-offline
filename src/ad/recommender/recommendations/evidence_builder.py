"""Evidence builder for recommendation backing.

This module provides comprehensive evidence structures and builders:
1. Statistical evidence (p-values, confidence intervals, effect sizes)
2. Cross-customer validation evidence
3. Top performer analysis evidence
4. Distribution evidence (prevalence, edge cases)
5. Headroom evidence (predicted ROAS boost/CPA reduction)

All evidence is aggregated into RecommendationEvidence for complete backing.
"""

# pylint: disable=line-too-long

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StatisticalEvidence:
    """Statistical validation evidence from hypothesis tests.

    Attributes:
        test_name: Name of statistical test (e.g., "fisher_exact", "chi_square")
        p_value: Raw p-value from test
        is_significant: Whether result is statistically significant (p < 0.05)
        confidence_interval: 95% confidence interval (lower, upper)
        effect_size: Effect size magnitude (Cramér's V, Cohen's d, etc.)
        sample_size: Total sample size used in test
        correction_method: Multiple comparison correction applied (if any)
    """

    test_name: str
    p_value: float
    is_significant: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    sample_size: int = 0
    correction_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test": self.test_name,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_interval": self.confidence_interval,
            "effect_size": self.effect_size,
            "sample_size": self.sample_size,
            "correction": self.correction_method,
        }


@dataclass
class CrossCustomerEvidence:
    """Validation across multiple customers.

    Attributes:
        validation_type: Type of validation ("pattern_holds", "counterfactual_works")
        customer_count: Number of customers showing this pattern
        success_rate: Proportion of customers where it worked (0-1)
        global_avg_lift: Average metric improvement across customers
        confidence_interval: Confidence interval for global lift
        date_range: Date range of validation data
    """

    validation_type: str
    customer_count: int
    success_rate: float
    global_avg_lift: float
    confidence_interval: Optional[Tuple[float, float]] = None
    date_range: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.validation_type,
            "customers": self.customer_count,
            "success_rate": self.success_rate,
            "avg_lift": self.global_avg_lift,
            "confidence_interval": self.confidence_interval,
            "date_range": self.date_range,
        }


@dataclass
class TopPerformerEvidence:
    """Analysis of top-performing creatives.

    Attributes:
        percentile_rank: Rank of this value (0-1, where 1.0 = best)
        is_top_quartile: Whether value is in top 25% of performers
        avg_metric_for_value: Average ROAS/CPA for this value
        median_metric_overall: Median metric across all values
        lift_over_median: Percentage improvement over median
        support_count: Number of creatives with this value
    """

    percentile_rank: float
    is_top_quartile: bool
    avg_metric_for_value: float
    median_metric_overall: float
    lift_over_median: float
    support_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "percentile": self.percentile_rank,
            "is_top_quartile": self.is_top_quartile,
            "avg_metric": self.avg_metric_for_value,
            "median_overall": self.median_metric_overall,
            "lift_pct": self.lift_over_median * 100,
            "support_count": self.support_count,
        }


@dataclass
class DistributionEvidence:
    """Data distribution and prevalence information.

    Attributes:
        prevalence: How common this value is (0-1)
        is_edge_case: Whether this is a rare value
        in_training_distribution: Whether value exists in training data
        support_level: Categorized support ("high", "medium", "low")
        distance_from_current: How different from current value
    """

    prevalence: float
    is_edge_case: bool
    in_training_distribution: bool
    support_level: str
    distance_from_current: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prevalence": self.prevalence,
            "is_edge_case": self.is_edge_case,
            "in_distribution": self.in_training_distribution,
            "support_level": self.support_level,
            "distance": self.distance_from_current,
        }


@dataclass
class OpportunitySizeEvidence:
    """Predicted additional benefit (ROAS boost or CPA reduction).

    Opportunity size represents the improvement we can achieve by implementing
    this recommendation, based on model predictions and historical data.

    Attributes:
        predicted_absolute_improvement: Absolute change in metric (+1.2 ROAS or -$0.45 CPA)
        predicted_relative_improvement: Relative change (0.57 = 57%, -0.32 = 32% reduction)
        current_metric_value: Current metric value
        predicted_metric_value: Predicted metric value after implementation
        confidence: Confidence in prediction (0-1)
        metric_name: Target metric ("roas" or "cpa")
        is_benefit: True if higher is better (ROAS), False if lower is better (CPA)
    """

    predicted_absolute_improvement: float
    predicted_relative_improvement: float
    current_metric_value: float
    predicted_metric_value: float
    confidence: float
    metric_name: str
    is_benefit: bool

    @property
    def opportunity_tier(self) -> str:
        """Categorize opportunity_size as high/medium/low impact."""
        pct = abs(self.predicted_relative_improvement)
        if pct >= 0.5:
            return "high"
        if pct >= 0.2:
            return "medium"
        return "low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "absolute_improvement": self.predicted_absolute_improvement,
            "relative_improvement_pct": self.predicted_relative_improvement
            * 100,
            "from": self.current_metric_value,
            "to": self.predicted_metric_value,
            "tier": self.opportunity_tier,
            "metric": self.metric_name,
            "confidence": self.confidence,
        }


@dataclass
class RecommendationEvidence:
    """Complete evidence package for a recommendation.

    Aggregates all evidence types into a single structure for complete
    backing of every recommendation.

    Attributes:
        statistical: Statistical validation evidence
        cross_customer: Cross-customer validation evidence
        top_performer: Top performer analysis evidence
        distribution: Distribution evidence
        opportunity_size: Opportunity size (predicted improvement) evidence
    """

    statistical: Optional[StatisticalEvidence] = None
    cross_customer: Optional[CrossCustomerEvidence] = None
    top_performer: Optional[TopPerformerEvidence] = None
    distribution: Optional[DistributionEvidence] = None
    opportunity_size: Optional[OpportunitySizeEvidence] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "statistical": (
                self.statistical.to_dict() if self.statistical else None
            ),
            "cross_customer": (
                self.cross_customer.to_dict() if self.cross_customer else None
            ),
            "top_performer": (
                self.top_performer.to_dict() if self.top_performer else None
            ),
            "distribution": (
                self.distribution.to_dict() if self.distribution else None
            ),
            "opportunity_size": (
                self.opportunity_size.to_dict()
                if self.opportunity_size
                else None
            ),
        }

    @property
    def has_strong_evidence(self) -> bool:
        """Check if recommendation has strong supporting evidence.

        Returns True if at least 2 of these conditions are met:
        - Statistical significance
        - High cross-customer success rate (>70%)
        - In top performer quartile
        - High or medium opportunity_size
        """
        checks = [
            self.statistical.is_significant if self.statistical else False,
            (
                self.cross_customer.success_rate > 0.7
                if self.cross_customer
                else False
            ),
            self.top_performer.is_top_quartile if self.top_performer else False,
            (
                self.opportunity_size.opportunity_tier in ["high", "medium"]
                if self.opportunity_size
                else False
            ),
        ]
        return sum(checks) >= 2


class EvidenceBuilder:
    """Build evidence from pattern validation, counterfactuals, and training data.

    This class aggregates evidence from multiple sources:
    - ValidatedPattern objects from stability validation
    - CounterfactualRecommendation objects from counterfactual generation
    - FeatureDistribution objects from training data analysis
    - Recommendation dictionaries with predicted impacts
    """

    def __init__(
        self,
        training_data: Optional[Any] = None,
        target_col: str = "roas",
    ):
        """Initialize evidence builder.

        Args:
            training_data: Optional training DataFrame for distribution analysis
            target_col: Target metric column name ("roas" or "cpa")
        """
        self.training_data = training_data
        self.target_col = target_col
        # NOTE: analyzer is disabled until counterfactuals_stability module is implemented
        self.analyzer = None

    def build_evidence(
        self,
        recommendation: Dict[str, Any],
        validated_pattern: Optional[Any] = None,
        counterfactual: Optional[Any] = None,
        current_metric_value: float = 0.0,
    ) -> RecommendationEvidence:
        """Aggregate all evidence sources into complete package.

        Args:
            recommendation: Recommendation dict with feature, current, recommended values
            validated_pattern: ValidatedPattern object (if available)
            counterfactual: CounterfactualRecommendation object (if available)
            current_metric_value: Current ROAS/CPA value

        Returns:
            Complete evidence package
        """
        feature = recommendation.get("feature", "")
        current_value = recommendation.get("current")
        recommended_value = recommendation.get("recommended")

        # Build each evidence type
        statistical = self._build_statistical_evidence(validated_pattern)
        cross_customer = self._build_cross_customer_evidence(validated_pattern)
        top_performer = self._build_top_performer_evidence(
            feature, recommended_value, counterfactual
        )
        distribution = self._build_distribution_evidence(
            feature, current_value, counterfactual
        )
        opportunity_size = self._build_opportunity_size_evidence(
            recommendation, current_metric_value
        )

        return RecommendationEvidence(
            statistical=statistical,
            cross_customer=cross_customer,
            top_performer=top_performer,
            distribution=distribution,
            opportunity_size=opportunity_size,
        )

    def _build_statistical_evidence(
        self, pattern: Optional[Any]
    ) -> Optional[StatisticalEvidence]:
        """Extract statistical evidence from ValidatedPattern.

        Args:
            pattern: ValidatedPattern object with p_value, effect_size, etc.

        Returns:
            StatisticalEvidence or None if pattern is invalid
        """
        if pattern is None:
            return None

        # Extract attributes with fallbacks
        p_value = getattr(pattern, "p_value", None)
        if p_value is None:
            return None

        is_significant = p_value < 0.05

        confidence_interval = getattr(pattern, "bootstrap_ci", None)
        effect_size = getattr(pattern, "effect_size", None)
        sample_size = getattr(pattern, "sample_size", 0)

        # Check for Bonferroni correction
        bonferroni_p = getattr(pattern, "bonferroni_adjusted_p", None)
        correction_method = "bonferroni" if bonferroni_p else None

        return StatisticalEvidence(
            test_name="fisher_exact",
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            sample_size=sample_size,
            correction_method=correction_method,
        )

    def _build_cross_customer_evidence(
        self, pattern: Optional[Any]
    ) -> Optional[CrossCustomerEvidence]:
        """Extract cross-customer validation evidence.

        Args:
            pattern: ValidatedPattern with global validation flags

        Returns:
            CrossCustomerEvidence or None if not globally validated
        """
        if pattern is None:
            return None

        validated_globally = getattr(pattern, "validated_globally", False)
        if not validated_globally:
            return None

        customer_count = getattr(pattern, "global_customer_count", 0) or 0
        success_rate = getattr(pattern, "global_success_rate", 0.0) or 0.0
        avg_lift = getattr(pattern, "global_avg_lift", 0.0) or 0.0

        return CrossCustomerEvidence(
            validation_type="pattern_holds",
            customer_count=customer_count,
            success_rate=success_rate,
            global_avg_lift=avg_lift,
            confidence_interval=None,
            date_range="2024-01 to 2024-12",
        )

    def _build_top_performer_evidence(
        self,
        feature: str,
        value: Any,
        counterfactual: Optional[Any],
    ) -> Optional[TopPerformerEvidence]:
        """Extract top performer analysis evidence.

        Args:
            feature: Feature name
            value: Feature value to analyze
            counterfactual: CounterfactualRecommendation with performance data

        Returns:
            TopPerformerEvidence or None if no data available
        """
        if counterfactual is not None:
            # Extract from counterfactual
            percentile = getattr(counterfactual, "percentile_rank", 0.0)
            in_top = getattr(counterfactual, "in_top_performers", False)
            avg_roas = getattr(counterfactual, "avg_roas", 0.0)
            lift = getattr(counterfactual, "top_performer_lift", 0.0)
            count = getattr(counterfactual, "support_count", 0)

            return TopPerformerEvidence(
                percentile_rank=percentile,
                is_top_quartile=in_top,
                avg_metric_for_value=avg_roas,
                median_metric_overall=0.0,
                lift_over_median=lift,
                support_count=count,
            )

        # Try to get from training data analyzer
        if (
            self.analyzer is None
            or feature not in self.analyzer.feature_distributions
        ):
            return None

        dist = self.analyzer.feature_distributions[feature]
        roas = dist.roas_by_value.get(value, 0)
        count = dist.count_by_value.get(value, 0)

        # Calculate percentile
        all_roas = list(dist.roas_by_value.values())
        percentile = self.calculate_percentile(roas, all_roas)
        median = np.median(all_roas)

        lift = (roas - median) / median if median > 0 else 0.0

        return TopPerformerEvidence(
            percentile_rank=percentile,
            is_top_quartile=percentile >= 0.75,
            avg_metric_for_value=roas,
            median_metric_overall=float(median),
            lift_over_median=lift,
            support_count=count,
        )

    def _build_distribution_evidence(
        self,
        feature: str,
        current_value: Any,
        counterfactual: Optional[Any],
    ) -> Optional[DistributionEvidence]:
        """Extract distribution evidence.

        Args:
            feature: Feature name
            current_value: Current feature value
            counterfactual: CounterfactualRecommendation with distribution info

        Returns:
            DistributionEvidence or None if no data available
        """
        if counterfactual is not None:
            # Extract from counterfactual
            prevalence = getattr(counterfactual, "percentile_rank", 0.0)
            is_edge = getattr(counterfactual, "is_edge_case", False)
            distance = getattr(counterfactual, "distance_from_current", 0.0)
            count = getattr(counterfactual, "support_count", 0)

            return DistributionEvidence(
                prevalence=prevalence,
                is_edge_case=is_edge,
                in_training_distribution=True,
                support_level=self.get_support_level(count),
                distance_from_current=distance,
            )

        # Try to get from training data analyzer
        if (
            self.analyzer is None
            or feature not in self.analyzer.feature_distributions
        ):
            return None

        dist = self.analyzer.feature_distributions[feature]
        # For distribution evidence without counterfactual, use current value
        recommended_value = current_value

        count = dist.count_by_value.get(recommended_value, 0)
        total = sum(dist.count_by_value.values())
        prevalence = count / total if total > 0 else 0

        return DistributionEvidence(
            prevalence=prevalence,
            is_edge_case=count < 10,
            in_training_distribution=recommended_value in dist.unique_values,
            support_level=self.get_support_level(count),
            distance_from_current=0.0,
        )

    def _build_opportunity_size_evidence(
        self,
        recommendation: Dict[str, Any],
        current_metric_value: float,
    ) -> OpportunitySizeEvidence:
        """Extract opportunity_size from calculated predictions.

        Args:
            recommendation: Recommendation dict with potential_impact
            current_metric_value: Current ROAS/CPA value

        Returns:
            OpportunitySizeEvidence with predicted improvement
        """
        absolute_change = recommendation.get("potential_impact", 0.0)

        # Try multiple fields for relative change
        relative_change = recommendation.get("expected_lift_pct", 0.0)
        if relative_change == 0.0:
            # Try to calculate from absolute
            if current_metric_value > 0:
                relative_change = absolute_change / current_metric_value

        # Normalize to 0-1 range
        if abs(relative_change) > 1.0:
            relative_change = relative_change / 100.0

        predicted_value = current_metric_value + absolute_change

        # Get confidence
        confidence = recommendation.get("confidence", 0.7)
        if isinstance(confidence, str):
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            confidence = confidence_map.get(confidence, 0.7)

        is_benefit = self.target_col == "roas"

        return OpportunitySizeEvidence(
            predicted_absolute_improvement=absolute_change,
            predicted_relative_improvement=relative_change,
            current_metric_value=current_metric_value,
            predicted_metric_value=predicted_value,
            confidence=confidence,
            metric_name=self.target_col,
            is_benefit=is_benefit,
        )

    @staticmethod
    def calculate_percentile(value: float, distribution: List[float]) -> float:
        """Calculate percentile rank of a value in a distribution.

        Args:
            value: Value to find percentile for
            distribution: List of values to compare against

        Returns:
            Percentile rank (0-1)
        """
        if not distribution:
            return 0.0

        sorted_vals = sorted(distribution)
        num_vals = len(sorted_vals)

        # Count values <= target
        rank = sum(1 for v in sorted_vals if v <= value)

        return rank / num_vals if num_vals > 0 else 0.0

    @staticmethod
    def get_support_level(count: int) -> str:
        """Categorize sample size into support level.

        Args:
            count: Number of samples

        Returns:
            Support level: "high", "medium", or "low"
        """
        if count >= 100:
            return "high"
        if count >= 30:
            return "medium"
        return "low"


class EvidenceFormatter:
    """Format evidence for human-readable output.

    Provides methods to format each evidence type for different audiences:
    - Detailed: Full scientific notation with all details
    - Compact: Brief summaries for Slack/mobile
    - Summary: Bullet lists for reports
    """

    def format_statistical(self, evidence: StatisticalEvidence) -> str:
        """Format statistical evidence for display.

        Example: "Fisher's Exact: p=0.002 (significant), 95% CI [0.15, 0.32], effect size=0.28"

        Args:
            evidence: StatisticalEvidence to format

        Returns:
            Formatted string
        """
        sig_text = (
            "significant" if evidence.is_significant else "not significant"
        )
        parts = [f"{evidence.test_name}: p={evidence.p_value:.3f} ({sig_text})"]

        if evidence.confidence_interval:
            ci_low, ci_high = evidence.confidence_interval
            parts.append(f"95% CI [{ci_low:.2f}, {ci_high:.2f}]")

        if evidence.effect_size is not None:
            parts.append(f"effect size={evidence.effect_size:.2f}")

        return ", ".join(parts)

    def format_cross_customer(self, evidence: CrossCustomerEvidence) -> str:
        """Format cross-customer evidence for display.

        Example: "Validated across 47 customers: 82% success rate, +0.45 avg lift"

        Args:
            evidence: CrossCustomerEvidence to format

        Returns:
            Formatted string
        """
        return (
            f"Validated across {evidence.customer_count} customers: "
            f"{evidence.success_rate*100:.0f}% success rate, "
            f"+{evidence.global_avg_lift:.2f} avg lift"
        )

    def format_top_performer(self, evidence: TopPerformerEvidence) -> str:
        """Format top performer evidence for display.

        Example: "Top 15% of values: 3.2 vs 2.1 median (+52% lift, 156 samples)"

        Args:
            evidence: TopPerformerEvidence to format

        Returns:
            Formatted string
        """
        percentile = int((1.0 - evidence.percentile_rank) * 100)
        return (
            f"Top {percentile}% of values: "
            f"{evidence.avg_metric_for_value:.2f} vs {evidence.median_metric_overall:.2f} median "
            f"(+{evidence.lift_over_median*100:.0f}% lift, {evidence.support_count} samples)"
        )

    def format_distribution(self, evidence: DistributionEvidence) -> str:
        """Format distribution evidence for display.

        Example: "Appears in 23% of creatives (high support, common pattern)"

        Args:
            evidence: DistributionEvidence to format

        Returns:
            Formatted string
        """
        edge_text = "edge case" if evidence.is_edge_case else "common pattern"
        return (
            f"Appears in {evidence.prevalence*100:.0f}% of creatives "
            f"({evidence.support_level} support, {edge_text})"
        )

    def format_opportunity_size(self, evidence: OpportunitySizeEvidence) -> str:
        """Format opportunity_size evidence for display.

        Examples:
        - ROAS: "Opportunity Size: +1.2 ROAS (+57%) from 2.1 → 3.3 (high tier, 78% confidence)"
        - CPA: "Opportunity Size: -$0.45 CPA (-32%) from $1.40 → $0.95 (medium tier, 65% confidence)"

        Args:
            evidence: OpportunitySizeEvidence to format

        Returns:
            Formatted string
        """
        metric_label = evidence.metric_name.upper()

        if evidence.is_benefit:
            # ROAS: higher is better
            direction = (
                "+" if evidence.predicted_absolute_improvement > 0 else ""
            )
            absolute_str = f"{direction}{evidence.predicted_absolute_improvement:.2f} {metric_label}"
            relative_str = (
                f"+{evidence.predicted_relative_improvement*100:.0f}%"
            )
        else:
            # CPA: lower is better (reduction)
            absolute_str = f"${evidence.predicted_absolute_improvement:+.2f} {metric_label}"
            relative_str = f"{evidence.predicted_relative_improvement*100:.0f}%"

        progression = f"{evidence.current_metric_value:.2f} → {evidence.predicted_metric_value:.2f}"

        return (
            f"Opportunity Size: {absolute_str} ({relative_str}) from {progression} "
            f"({evidence.opportunity_tier} tier, {evidence.confidence*100:.0f}% confidence)"
        )

    def format_compact_opportunity_size(
        self, evidence: OpportunitySizeEvidence
    ) -> str:
        """Format opportunity_size in compact form for Slack/mobile.

        Examples:
        - "📈 +57% ROAS (high opportunity)"
        - "📉 -32% CPA (medium opportunity)"

        Args:
            evidence: OpportunitySizeEvidence to format

        Returns:
            Compact formatted string
        """
        if evidence.is_benefit:
            icon = "📈" if evidence.predicted_relative_improvement > 0 else "📉"
        else:
            icon = "📉" if evidence.predicted_relative_improvement > 0 else "📈"

        return (
            f"{icon} {evidence.predicted_relative_improvement*100:+.0f}% {evidence.metric_name.upper()} "
            f"({evidence.opportunity_tier} opportunity)"
        )

    def format_evidence_summary(self, evidence: RecommendationEvidence) -> str:
        """Format complete evidence as bullet list.

        Example:
        - Opportunity Size: +1.2 ROAS (+57%) from 2.1 → 3.3 (high tier)
        - Statistical: Fisher's Exact p=0.002 (significant)
        - Top Performer: Top 15% (+52% lift over median)
        - Distribution: Appears in 23% of creatives (high support)

        Args:
            evidence: Complete evidence package

        Returns:
            Formatted bullet list
        """
        bullets = []

        if evidence.opportunity_size:
            bullets.append(
                f"**Opportunity Size:** {self.format_opportunity_size(evidence.opportunity_size)}"
            )

        if evidence.statistical:
            bullets.append(
                f"**Statistical:** {self.format_statistical(evidence.statistical)}"
            )

        if evidence.cross_customer:
            bullets.append(
                f"**Cross-Customer:** {self.format_cross_customer(evidence.cross_customer)}"
            )

        if evidence.top_performer:
            bullets.append(
                f"**Top Performer:** {self.format_top_performer(evidence.top_performer)}"
            )

        if evidence.distribution:
            bullets.append(
                f"**Distribution:** {self.format_distribution(evidence.distribution)}"
            )

        return "\n".join(bullets)
