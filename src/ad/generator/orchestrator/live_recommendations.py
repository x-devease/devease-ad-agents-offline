"""
Live Recommendations Module.

Integrates live recommendations from creative scorer into prompt generation
with real-time feature prioritization and dynamic optimization.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class LiveRecommendation:
    """Single live recommendation from creative scorer."""

    feature_name: str
    current_value: str
    recommended_value: str
    confidence: float
    potential_impact: float
    evidence: Dict
    priority: str  # HIGH, MEDIUM, LOW


@dataclass
class RecommendationSet:
    """Set of live recommendations for a product."""

    product_name: str
    recommendations: List[LiveRecommendation]
    timestamp: str
    source: str

    def get_high_priority(self) -> List[LiveRecommendation]:
        """Get high-priority recommendations."""
        return [r for r in self.recommendations if r.priority == "HIGH"]

    def get_by_feature(self, feature_name: str) -> Optional[LiveRecommendation]:
        """Get recommendation for specific feature."""
        for rec in self.recommendations:
            if rec.feature_name == feature_name:
                return rec
        return None


class LiveRecommendationIntegrator:
    """
    Integrates live recommendations into prompt generation.

    Ensures recommendations are applied based on priority, confidence,
    and potential impact.
    """

    def __init__(self):
        self.current_recommendations: Optional[RecommendationSet] = None
        self.applied_recommendations: List[str] = []

    def load_recommendations(
        self,
        recommendation_set: RecommendationSet,
    ) -> None:
        """
        Load live recommendations.

        Args:
            recommendation_set: RecommendationSet from creative scorer
        """
        self.current_recommendations = recommendation_set
        self.applied_recommendations = []

        logger.info(
            "Loaded %d recommendations for %s (timestamp: %s)",
            len(recommendation_set.recommendations),
            recommendation_set.product_name,
            recommendation_set.timestamp,
        )

    def should_apply_recommendation(
        self,
        recommendation: LiveRecommendation,
        min_confidence: float = 0.7,
        min_impact: float = 0.1,
    ) -> bool:
        """
        Determine if recommendation should be applied.

        Args:
            recommendation: Recommendation to evaluate
            min_confidence: Minimum confidence threshold
            min_impact: Minimum impact threshold

        Returns:
            True if recommendation should be applied
        """
        # Check if already applied
        if recommendation.feature_name in self.applied_recommendations:
            return False

        # Check confidence threshold
        if recommendation.confidence < min_confidence:
            logger.debug(
                "Skipping '%s': confidence %.2f below threshold %.2f",
                recommendation.feature_name,
                recommendation.confidence,
                min_confidence,
            )
            return False

        # Check impact threshold
        if recommendation.potential_impact < min_impact:
            logger.debug(
                "Skipping '%s': impact %.2f below threshold %.2f",
                recommendation.feature_name,
                recommendation.potential_impact,
                min_impact,
            )
            return False

        return True

    def prioritize_recommendations(
        self,
        recommendations: List[LiveRecommendation],
    ) -> List[LiveRecommendation]:
        """
        Prioritize recommendations by impact and confidence.

        Args:
            recommendations: List of recommendations

        Returns:
            Prioritized list (highest priority first)
        """
        # Priority scoring
        priority_score = {
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1,
        }

        # Sort by priority score, then confidence, then impact
        prioritized = sorted(
            recommendations,
            key=lambda r: (
                priority_score.get(r.priority, 0),
                r.confidence,
                r.potential_impact,
            ),
            reverse=True,
        )

        logger.info(
            "Prioritized %d recommendations: %d HIGH, %d MEDIUM, %d LOW",
            len(prioritized),
            sum(1 for r in prioritized if r.priority == "HIGH"),
            sum(1 for r in prioritized if r.priority == "MEDIUM"),
            sum(1 for r in prioritized if r.priority == "LOW"),
        )

        return prioritized

    def apply_to_prompt(
        self,
        base_prompt: str,
        max_recommendations: int = 5,
        min_confidence: float = 0.7,
        min_impact: float = 0.1,
    ) -> str:
        """
        Apply live recommendations to prompt.

        Args:
            base_prompt: Original prompt
            max_recommendations: Maximum number of recommendations to apply
            min_confidence: Minimum confidence threshold
            min_impact: Minimum impact threshold

        Returns:
            Enhanced prompt with recommendations applied
        """
        if not self.current_recommendations:
            logger.warning("No recommendations loaded, returning base prompt")
            return base_prompt

        # Get applicable recommendations
        applicable = [
            r
            for r in self.current_recommendations.recommendations
            if self.should_apply_recommendation(r, min_confidence, min_impact)
        ]

        if not applicable:
            logger.info("No applicable recommendations meet thresholds")
            return base_prompt

        # Prioritize
        prioritized = self.prioritize_recommendations(applicable)

        # Limit to max_recommendations
        to_apply = prioritized[:max_recommendations]

        # Build enhancement section
        enhancement_parts = []

        enhancement_parts.append("[Live Recommendations Applied]")

        for rec in to_apply:
            enhancement_parts.append(f"- {rec.feature_name}: {rec.recommended_value}")
            enhancement_parts.append(f"  Confidence: {rec.confidence:.2%}, Impact: {rec.potential_impact:.2%}")
            self.applied_recommendations.append(rec.feature_name)

        # Append to base prompt
        enhanced_prompt = base_prompt + "\n\n" + "\n".join(enhancement_parts)

        logger.info(
            "Applied %d recommendations to prompt (total length: %d chars)",
            len(to_apply),
            len(enhanced_prompt),
        )

        return enhanced_prompt

    def get_recommendation_summary(self) -> Dict:
        """
        Get summary of loaded recommendations.

        Returns:
            Dictionary with recommendation summary
        """
        if not self.current_recommendations:
            return {
                "loaded": False,
                "count": 0,
                "applied": 0,
            }

        recommendations = self.current_recommendations.recommendations

        return {
            "loaded": True,
            "product": self.current_recommendations.product_name,
            "timestamp": self.current_recommendations.timestamp,
            "total_count": len(recommendations),
            "applied_count": len(self.applied_recommendations),
            "high_priority": sum(1 for r in recommendations if r.priority == "HIGH"),
            "medium_priority": sum(1 for r in recommendations if r.priority == "MEDIUM"),
            "low_priority": sum(1 for r in recommendations if r.priority == "LOW"),
            "avg_confidence": sum(r.confidence for r in recommendations) / len(recommendations),
            "avg_impact": sum(r.potential_impact for r in recommendations) / len(recommendations),
        }

    def validate_prompt_coverage(
        self,
        prompt: str,
        feature_names: List[str],
    ) -> Dict:
        """
        Validate that prompt covers recommended features.

        Args:
            prompt: Generated prompt
            feature_names: List of feature names to check

        Returns:
            Dictionary with coverage results
        """
        if not self.current_recommendations:
            return {
                "checked": False,
                "reason": "No recommendations loaded",
            }

        prompt_lower = prompt.lower()
        checked_features = []

        for feature_name in feature_names:
            rec = self.current_recommendations.get_by_feature(feature_name)
            if rec and rec.feature_name in self.applied_recommendations:
                # Check if recommended value is in prompt
                if rec.recommended_value.lower() in prompt_lower:
                    checked_features.append({
                        "feature": feature_name,
                        "covered": True,
                        "recommended_value": rec.recommended_value,
                    })
                else:
                    checked_features.append({
                        "feature": feature_name,
                        "covered": False,
                        "recommended_value": rec.recommended_value,
                    })

        return {
            "checked": True,
            "total_features": len(feature_names),
            "covered_count": sum(1 for f in checked_features if f["covered"]),
            "coverage_ratio": sum(1 for f in checked_features if f["covered"]) / len(feature_names) if feature_names else 0,
            "features": checked_features,
        }


def create_live_recommendation_prompt(
    recommendations: List[LiveRecommendation],
    context: Dict,
) -> str:
    """
    Create prompt section from live recommendations.

    Args:
        recommendations: List of recommendations
        context: Product context

    Returns:
        Formatted prompt section
    """
    if not recommendations:
        return ""

    parts = []

    parts.append("[Live Creative Scorer Recommendations]")
    parts.append(f"Product: {context.get('product_name', 'Unknown')}")

    # Group by priority
    high_priority = [r for r in recommendations if r.priority == "HIGH"]
    medium_priority = [r for r in recommendations if r.priority == "MEDIUM"]
    low_priority = [r for r in recommendations if r.priority == "LOW"]

    if high_priority:
        parts.append("\n[High Priority - Apply First]")
        for rec in high_priority:
            parts.append(f"- {rec.feature_name}: {rec.recommended_value}")
            parts.append(f"  (Confidence: {rec.confidence:.1%}, Impact: {rec.potential_impact:.1%})")

    if medium_priority:
        parts.append("\n[Medium Priority]")
        for rec in medium_priority:
            parts.append(f"- {rec.feature_name}: {rec.recommended_value}")

    if low_priority:
        parts.append("\n[Low Priority - If Space Permits]")
        for rec in low_priority:
            parts.append(f"- {rec.feature_name}: {rec.recommended_value}")

    return "\n".join(parts)


# Recommendation impact estimation
IMPACT_ESTIMATION = {
    "high": {
        "range": "0.2 - 0.5 (20-50% ROAS improvement)",
        "description": "Significant impact on performance",
        "examples": [
            "Adding high-performing creative element",
            "Fixing critical anti-pattern",
            "Optimizing key visual feature",
        ],
    },
    "medium": {
        "range": "0.05 - 0.2 (5-20% ROAS improvement)",
        "description": "Moderate impact on performance",
        "examples": [
            "Adding secondary creative element",
            "Minor visual optimization",
            "Enhancing existing feature",
        ],
    },
    "low": {
        "range": "0.01 - 0.05 (1-5% ROAS improvement)",
        "description": "Minor impact on performance",
        "examples": [
            "Fine-tuning details",
            "Optional enhancements",
            "Nice-to-have features",
        ],
    },
}


def get_impact_estimation(
    impact_score: float,
) -> Dict:
    """
    Get impact estimation details.

    Args:
        impact_score: Impact score (0.0 to 1.0)

    Returns:
        Dictionary with impact estimation
    """
    if impact_score >= 0.2:
        category = "high"
    elif impact_score >= 0.05:
        category = "medium"
    else:
        category = "low"

    return {
        "category": category,
        "score": impact_score,
        "description": IMPACT_ESTIMATION[category]["description"],
        "range": IMPACT_ESTIMATION[category]["range"],
        "examples": IMPACT_ESTIMATION[category]["examples"],
    }


# Recommendation filtering
def filter_recommendations(
    recommendations: List[LiveRecommendation],
    min_confidence: float = 0.7,
    min_impact: float = 0.05,
    priorities: Optional[List[str]] = None,
) -> List[LiveRecommendation]:
    """
    Filter recommendations by criteria.

    Args:
        recommendations: List of recommendations
        min_confidence: Minimum confidence threshold
        min_impact: Minimum impact threshold
        priorities: List of priorities to include (None = all)

    Returns:
        Filtered list of recommendations
    """
    filtered = []

    for rec in recommendations:
        # Confidence filter
        if rec.confidence < min_confidence:
            continue

        # Impact filter
        if rec.potential_impact < min_impact:
            continue

        # Priority filter
        if priorities and rec.priority not in priorities:
            continue

        filtered.append(rec)

    logger.info(
        "Filtered %d recommendations from %d total",
        len(filtered),
        len(recommendations),
    )

    return filtered


def create_recommendation_report(
    recommendation_set: RecommendationSet,
) -> str:
    """
    Create human-readable report of recommendations.

    Args:
        recommendation_set: Set of recommendations

    Returns:
        Formatted report string
    """
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("LIVE RECOMMENDATIONS REPORT")
    report_lines.append("=" * 60)

    report_lines.append(f"\nProduct: {recommendation_set.product_name}")
    report_lines.append(f"Timestamp: {recommendation_set.timestamp}")
    report_lines.append(f"Source: {recommendation_set.source}")
    report_lines.append(f"Total Recommendations: {len(recommendation_set.recommendations)}")

    # Group by priority
    high = [r for r in recommendation_set.recommendations if r.priority == "HIGH"]
    medium = [r for r in recommendation_set.recommendations if r.priority == "MEDIUM"]
    low = [r for r in recommendation_set.recommendations if r.priority == "LOW"]

    report_lines.append(f"\n  High Priority: {len(high)}")
    report_lines.append(f"  Medium Priority: {len(medium)}")
    report_lines.append(f"  Low Priority: {len(low)}")

    # High priority details
    if high:
        report_lines.append("\n--- HIGH PRIORITY ---")
        for i, rec in enumerate(high, 1):
            impact = get_impact_estimation(rec.potential_impact)
            report_lines.append(f"\n{i}. {rec.feature_name}")
            report_lines.append(f"   Current: {rec.current_value}")
            report_lines.append(f"   Recommended: {rec.recommended_value}")
            report_lines.append(f"   Confidence: {rec.confidence:.1%}")
            report_lines.append(f"   Impact: {impact['category']} ({impact['range']})")
            report_lines.append(f"   Evidence: {rec.evidence.get('pattern', 'N/A')}")

    report_lines.append("\n" + "=" * 60)

    return "\n".join(report_lines)
