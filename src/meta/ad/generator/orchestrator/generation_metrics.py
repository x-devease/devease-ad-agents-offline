"""
Generation Metrics Module.

Tracks and reports metrics for prompt generation to enable
continuous improvement and quality monitoring.

Metrics categories:
- Quality metrics: Prompt completeness, feature coverage, technical accuracy
- Performance metrics: Generation time, token usage, prompt length
- Fidelity metrics: Adherence to template, feature reproduction accuracy
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics for a single generation run."""

    # Quality metrics
    feature_coverage: float = 0.0  # 0-1, percentage of features covered
    brand_integrity: float = 0.0  # 0-1, brand guideline adherence
    technical_accuracy: float = 0.0  # 0-1, technical specification accuracy
    prompt_quality_score: float = 0.0  # 0-100, overall quality

    # Performance metrics
    generation_time_ms: float = 0.0
    prompt_length: int = 0
    token_count: int = 0

    # Feature metrics
    total_features: int = 0
    high_confidence_features: int = 0
    medium_confidence_features: int = 0
    low_confidence_features: int = 0

    # Template metrics
    template_used: str = ""
    template_adherence: float = 0.0  # 0-1

    # Metadata
    customer: str = ""
    platform: str = ""
    product_name: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "quality": {
                "feature_coverage": self.feature_coverage,
                "brand_integrity": self.brand_integrity,
                "technical_accuracy": self.technical_accuracy,
                "prompt_quality_score": self.prompt_quality_score,
            },
            "performance": {
                "generation_time_ms": self.generation_time_ms,
                "prompt_length": self.prompt_length,
                "token_count": self.token_count,
            },
            "features": {
                "total": self.total_features,
                "high_confidence": self.high_confidence_features,
                "medium_confidence": self.medium_confidence_features,
                "low_confidence": self.low_confidence_features,
            },
            "template": {
                "used": self.template_used,
                "adherence": self.template_adherence,
            },
            "metadata": {
                "customer": self.customer,
                "platform": self.platform,
                "product_name": self.product_name,
                "timestamp": self.timestamp,
            },
        }


@dataclass
class MetricsSummary:
    """Summary statistics across multiple generations."""

    total_generations: int = 0
    avg_quality_score: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_feature_coverage: float = 0.0
    avg_template_adherence: float = 0.0

    quality_distribution: Dict[str, int] = field(default_factory=dict)
    template_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert summary to dictionary."""
        return {
            "total_generations": self.total_generations,
            "averages": {
                "quality_score": self.avg_quality_score,
                "generation_time_ms": self.avg_generation_time_ms,
                "feature_coverage": self.avg_feature_coverage,
                "template_adherence": self.avg_template_adherence,
            },
            "quality_distribution": self.quality_distribution,
            "template_distribution": self.template_distribution,
        }


class MetricsTracker:
    """
    Tracks generation metrics over time.

    Provides methods to record, aggregate, and analyze metrics.
    """

    def __init__(self):
        self.metrics_history: List[GenerationMetrics] = []
        self.current_generation_start: Optional[float] = None

    def start_generation(self) -> None:
        """Mark the start of a generation run."""
        self.current_generation_start = time.time() * 1000  # Convert to ms

    def record_generation(
        self,
        metrics: GenerationMetrics,
    ) -> None:
        """
        Record metrics for a generation run.

        Args:
            metrics: GenerationMetrics object to record
        """
        # Record generation time if not set
        if metrics.generation_time_ms == 0.0 and self.current_generation_start:
            end_time = time.time() * 1000
            metrics.generation_time_ms = end_time - self.current_generation_start

        self.metrics_history.append(metrics)
        logger.info(
            "Recorded generation metrics: Quality=%.2f, Coverage=%.2f, Time=%.2fms",
            metrics.prompt_quality_score,
            metrics.feature_coverage,
            metrics.generation_time_ms,
        )

        # Reset timer
        self.current_generation_start = None

    def get_summary(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        template: Optional[str] = None,
    ) -> MetricsSummary:
        """
        Get summary statistics for generations.

        Args:
            customer: Filter by customer
            platform: Filter by platform
            template: Filter by template

        Returns:
            MetricsSummary object with aggregated statistics
        """
        # Filter metrics
        filtered = self.metrics_history

        if customer:
            filtered = [m for m in filtered if m.customer == customer]
        if platform:
            filtered = [m for m in filtered if m.platform == platform]
        if template:
            filtered = [m for m in filtered if m.template_used == template]

        if not filtered:
            return MetricsSummary()

        # Calculate averages
        summary = MetricsSummary()
        summary.total_generations = len(filtered)
        summary.avg_quality_score = sum(m.prompt_quality_score for m in filtered) / len(filtered)
        summary.avg_generation_time_ms = sum(m.generation_time_ms for m in filtered) / len(filtered)
        summary.avg_feature_coverage = sum(m.feature_coverage for m in filtered) / len(filtered)
        summary.avg_template_adherence = sum(m.template_adherence for m in filtered) / len(filtered)

        # Quality distribution
        quality_buckets = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for m in filtered:
            if m.prompt_quality_score >= 80:
                quality_buckets["excellent"] += 1
            elif m.prompt_quality_score >= 60:
                quality_buckets["good"] += 1
            elif m.prompt_quality_score >= 40:
                quality_buckets["fair"] += 1
            else:
                quality_buckets["poor"] += 1

        summary.quality_distribution = quality_buckets

        # Template distribution
        template_counts = {}
        for m in filtered:
            template_counts[m.template_used] = template_counts.get(m.template_used, 0) + 1

        summary.template_distribution = template_counts

        return summary

    def get_recent_metrics(
        self,
        count: int = 10,
    ) -> List[GenerationMetrics]:
        """
        Get the most recent metrics.

        Args:
            count: Number of recent metrics to return

        Returns:
            List of recent GenerationMetrics objects
        """
        return self.metrics_history[-count:]

    def get_quality_trend(
        self,
        window_size: int = 10,
    ) -> List[float]:
        """
        Get quality score trend over time.

        Args:
            window_size: Size of rolling average window

        Returns:
            List of rolling average quality scores
        """
        if len(self.metrics_history) < window_size:
            return []

        trend = []
        for i in range(window_size, len(self.metrics_history) + 1):
            window = self.metrics_history[i - window_size:i]
            avg = sum(m.prompt_quality_score for m in window) / len(window)
            trend.append(avg)

        return trend


def calculate_feature_coverage(
    prompt: str,
    features: List[Dict],
) -> float:
    """
    Calculate feature coverage in prompt.

    Args:
        prompt: Generated prompt string
        features: List of feature dictionaries

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if not features:
        return 1.0

    prompt_lower = prompt.lower()
    covered = 0

    for feature in features:
        feature_name = feature.get("feature_name", "").lower()
        feature_value = feature.get("feature_value", "").lower()

        # Check if feature name or value appears in prompt
        if feature_name and feature_name in prompt_lower:
            covered += 1
        elif feature_value and feature_value in prompt_lower:
            covered += 1

    return covered / len(features)


def calculate_template_adherence(
    prompt: str,
    template_requirements: Dict[str, any],
) -> float:
    """
    Calculate adherence to template requirements.

    Args:
        prompt: Generated prompt string
        template_requirements: Template requirements dict

    Returns:
        Adherence ratio (0.0 to 1.0)
    """
    if not template_requirements:
        return 1.0

    prompt_lower = prompt.lower()
    requirements_met = 0
    total_requirements = 0

    # Check required sections
    required_sections = template_requirements.get("required_sections", [])
    for section in required_sections:
        total_requirements += 1
        if section.lower() in prompt_lower:
            requirements_met += 1

    # Check required keywords
    required_keywords = template_requirements.get("required_keywords", [])
    for keyword in required_keywords:
        total_requirements += 1
        if keyword.lower() in prompt_lower:
            requirements_met += 1

    # Check prohibited keywords
    prohibited_keywords = template_requirements.get("prohibited_keywords", [])
    for keyword in prohibited_keywords:
        total_requirements += 1
        if keyword.lower() not in prompt_lower:
            requirements_met += 1

    if total_requirements == 0:
        return 1.0

    return requirements_met / total_requirements


def estimate_token_count(
    prompt: str,
) -> int:
    """
    Estimate token count for prompt.

    Args:
        prompt: Prompt string

    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 characters per token
    return len(prompt) // 4


def get_performance_metrics(
    metrics_list: List[GenerationMetrics],
) -> Dict:
    """
    Get performance metrics from a list of metrics.

    Args:
        metrics_list: List of GenerationMetrics objects

    Returns:
        Dictionary with performance statistics
    """
    if not metrics_list:
        return {
            "min_time_ms": 0,
            "max_time_ms": 0,
            "avg_time_ms": 0,
            "min_length": 0,
            "max_length": 0,
            "avg_length": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "avg_tokens": 0,
        }

    times = [m.generation_time_ms for m in metrics_list]
    lengths = [m.prompt_length for m in metrics_list]
    tokens = [m.token_count for m in metrics_list]

    return {
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "avg_time_ms": sum(times) / len(times),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "avg_tokens": sum(tokens) / len(tokens),
    }


def get_quality_metrics(
    metrics_list: List[GenerationMetrics],
) -> Dict:
    """
    Get quality metrics from a list of metrics.

    Args:
        metrics_list: List of GenerationMetrics objects

    Returns:
        Dictionary with quality statistics
    """
    if not metrics_list:
        return {
            "min_quality": 0,
            "max_quality": 0,
            "avg_quality": 0,
            "min_coverage": 0,
            "max_coverage": 0,
            "avg_coverage": 0,
            "min_brand_integrity": 0,
            "max_brand_integrity": 0,
            "avg_brand_integrity": 0,
        }

    quality_scores = [m.prompt_quality_score for m in metrics_list]
    coverages = [m.feature_coverage for m in metrics_list]
    brand_integrities = [m.brand_integrity for m in metrics_list]

    return {
        "min_quality": min(quality_scores),
        "max_quality": max(quality_scores),
        "avg_quality": sum(quality_scores) / len(quality_scores),
        "min_coverage": min(coverages),
        "max_coverage": max(coverages),
        "avg_coverage": sum(coverages) / len(coverages),
        "min_brand_integrity": min(brand_integrities),
        "max_brand_integrity": max(brand_integrities),
        "avg_brand_integrity": sum(brand_integrities) / len(brand_integrities),
    }


def format_metrics_report(
    summary: MetricsSummary,
    performance_metrics: Dict,
    quality_metrics: Dict,
) -> str:
    """
    Format metrics summary as a readable report.

    Args:
        summary: MetricsSummary object
        performance_metrics: Performance metrics dict
        quality_metrics: Quality metrics dict

    Returns:
        Formatted report string
    """
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("GENERATION METRICS REPORT")
    report_lines.append("=" * 60)

    # Overview
    report_lines.append(f"\nTotal Generations: {summary.total_generations}")

    # Quality metrics
    report_lines.append("\n--- Quality Metrics ---")
    report_lines.append(f"Average Quality Score: {summary.avg_quality_score:.2f}/100")
    report_lines.append(f"Average Feature Coverage: {summary.avg_feature_coverage:.2%}")
    report_lines.append(f"Average Template Adherence: {summary.avg_template_adherence:.2%}")

    report_lines.append("\nQuality Distribution:")
    for bucket, count in summary.quality_distribution.items():
        pct = (count / summary.total_generations * 100) if summary.total_generations > 0 else 0
        report_lines.append(f"  {bucket.title()}: {count} ({pct:.1f}%)")

    # Performance metrics
    report_lines.append("\n--- Performance Metrics ---")
    report_lines.append(f"Average Generation Time: {summary.avg_generation_time_ms:.2f}ms")
    report_lines.append(f"Min Time: {performance_metrics['min_time_ms']:.2f}ms")
    report_lines.append(f"Max Time: {performance_metrics['max_time_ms']:.2f}ms")

    report_lines.append(f"\nAverage Prompt Length: {performance_metrics['avg_length']:.0f} chars")
    report_lines.append(f"Min Length: {performance_metrics['min_length']:.0f}")
    report_lines.append(f"Max Length: {performance_metrics['max_length']:.0f}")

    # Template distribution
    if summary.template_distribution:
        report_lines.append("\n--- Template Distribution ---")
        for template, count in summary.template_distribution.items():
            pct = (count / summary.total_generations * 100) if summary.total_generations > 0 else 0
            report_lines.append(f"  {template}: {count} ({pct:.1f}%)")

    report_lines.append("\n" + "=" * 60)

    return "\n".join(report_lines)


# Metric thresholds for quality assessment
QUALITY_THRESHOLDS = {
    "excellent": 80.0,
    "good": 60.0,
    "fair": 40.0,
    "poor": 0.0,
}


def assess_quality(
    quality_score: float,
) -> str:
    """
    Assess quality category from score.

    Args:
        quality_score: Quality score (0-100)

    Returns:
        Quality category (excellent, good, fair, poor)
    """
    if quality_score >= QUALITY_THRESHOLDS["excellent"]:
        return "excellent"
    elif quality_score >= QUALITY_THRESHOLDS["good"]:
        return "good"
    elif quality_score >= QUALITY_THRESHOLDS["fair"]:
        return "fair"
    else:
        return "poor"


def get_quality_recommendations(
    metrics: GenerationMetrics,
) -> List[str]:
    """
    Get recommendations to improve quality based on metrics.

    Args:
        metrics: GenerationMetrics object

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Feature coverage
    if metrics.feature_coverage < 0.8:
        recommendations.append(
            f"Improve feature coverage (current: {metrics.feature_coverage:.1%}). "
            "Ensure all high-confidence features are included in prompt."
        )

    # Brand integrity
    if metrics.brand_integrity < 0.7:
        recommendations.append(
            f"Strengthen brand integrity (current: {metrics.brand_integrity:.1%}). "
            "Review brand guidelines for logo, colors, and typography."
        )

    # Technical accuracy
    if metrics.technical_accuracy < 0.7:
        recommendations.append(
            f"Enhance technical accuracy (current: {metrics.technical_accuracy:.1%}). "
            "Add more specific technical details and specifications."
        )

    # Prompt length
    if metrics.prompt_length < 800:
        recommendations.append(
            f"Increase prompt length (current: {metrics.prompt_length} chars). "
            "Add more detail to meet minimum threshold of 800 chars."
        )
    elif metrics.prompt_length > 2000:
        recommendations.append(
            f"Reduce prompt length (current: {metrics.prompt_length} chars). "
            "Focus on high-confidence features to stay under 2000 chars."
        )

    # Template adherence
    if metrics.template_adherence < 0.8:
        recommendations.append(
            f"Improve template adherence (current: {metrics.template_adherence:.1%}). "
            "Ensure all required template sections are present."
        )

    if not recommendations:
        recommendations.append("Quality metrics are within acceptable ranges. Maintain current standards.")

    return recommendations
