"""
Feature Validation Module

Provides validation functionality for comparing expected features (injected into prompts)
against extracted features (from generated images) to detect mismatches and negative hits.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of feature validation for a single sample."""

    sample_id: str
    expected_features: Dict[str, Any]
    extracted_features: Dict[str, Any]
    negative_features: Dict[str, Any]
    matches: Dict[str, bool]
    mismatches: Dict[str, bool]
    negative_hits: Dict[str, bool]
    match_count: int
    mismatch_count: int
    negative_hit_count: int
    total_features: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_id": self.sample_id,
            "match_count": self.match_count,
            "mismatch_count": self.mismatch_count,
            "negative_hit_count": self.negative_hit_count,
            "total_features": self.total_features,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "negative_hits": self.negative_hits,
        }


def _normalize_value(value: Any) -> Optional[str]:
    """
    Normalize feature value for comparison.

    Handles None, NaN, case variations, and common separators.

    Args:
        value: Value to normalize

    Returns:
        Normalized lowercase string, or None if value is None/empty/NaN
    """
    if value is None:
        return None
    if isinstance(value, float):
        # Check for NaN
        if value != value:  # NaN != NaN is True
            return None
    s = str(value).strip().lower()
    if not s or s in {"none", "no", "n/a", "na", "null", "nan", "-"}:
        return None
    # Standardize separators
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())  # Normalize whitespace
    return s if s else None


def compare_features(
    *,
    expected_feature_values: Dict[str, Any],
    negative_feature_values: Dict[str, Any],
    extracted_features: Dict[str, Any],
) -> ValidationResult:
    """
    Compare expected features against extracted features.

    Args:
        expected_feature_values: Dict of feature_name -> expected_value
        negative_feature_values: Dict of feature_name -> negative_value (to avoid)
        extracted_features: Dict of feature_name -> extracted_value

    Returns:
        ValidationResult with match/mismatch/negative_hit details
    """
    matches = {}
    mismatches = {}
    negative_hits = {}

    for feature_name, expected_val in expected_feature_values.items():
        extracted_val = extracted_features.get(feature_name)
        negative_val = negative_feature_values.get(feature_name)

        expected_norm = _normalize_value(expected_val)
        extracted_norm = _normalize_value(extracted_val)
        negative_norm = _normalize_value(negative_val)
        # Check for match
        match = expected_norm is not None and extracted_norm == expected_norm
        matches[feature_name] = match
        # Check for mismatch (no match)
        mismatches[feature_name] = not match
        # Check for negative hit (extracted matches negative value)
        negative_hit = (
            negative_norm is not None
            and extracted_norm is not None
            and extracted_norm == negative_norm
        )
        negative_hits[feature_name] = negative_hit
        # Log problematic cases
        if negative_hit:
            logger.warning(
                "Negative hit detected: feature '%s' extracted as '%s' "
                "(negative value: '%s')",
                feature_name,
                extracted_val,
                negative_val,
            )

    match_count = sum(1 for v in matches.values() if v)
    mismatch_count = sum(1 for v in mismatches.values() if v)
    negative_hit_count = sum(1 for v in negative_hits.values() if v)

    return ValidationResult(
        sample_id="unknown",
        expected_features=expected_feature_values,
        extracted_features=extracted_features,
        negative_features=negative_feature_values,
        matches=matches,
        mismatches=mismatches,
        negative_hits=negative_hits,
        match_count=match_count,
        mismatch_count=mismatch_count,
        negative_hit_count=negative_hit_count,
        total_features=len(expected_feature_values),
    )


def validate_generation_batch(
    *,
    samples: List[Dict[str, Any]],
    max_mismatches: int = 2,
    max_negative_hits: int = 0,
) -> Dict[str, Any]:
    """
    Validate a batch of generated images against expected features.

    Args:
        samples: List of sample dicts, each containing:
            - sample_id: str
            - expected_feature_values: Dict[str, Any]
            - negative_feature_values: Dict[str, Any]
            - extracted_features: Dict[str, Any]
        max_mismatches: Maximum allowable mismatches per sample
        max_negative_hits: Maximum allowable negative hits per sample

    Returns:
        Dict with validation results:
            - total_samples: int
            - passed_samples: int
            - failed_samples: int
            - samples: List of ValidationResult dicts
            - summary: Aggregated statistics by feature
    """
    results = []
    passed = 0
    failed = 0

    feature_stats: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        sample_id = sample.get("sample_id", "unknown")
        expected = sample.get("expected_feature_values", {})
        negative = sample.get("negative_feature_values", {})
        extracted = sample.get("extracted_features", {})

        result = compare_features(
            expected_feature_values=expected,
            negative_feature_values=negative,
            extracted_features=extracted,
        )
        result.sample_id = sample_id
        results.append(result.to_dict())
        # Aggregate feature statistics
        for feature_name in expected.keys():
            if feature_name not in feature_stats:
                feature_stats[feature_name] = {
                    "total": 0,
                    "match_count": 0,
                    "mismatch_count": 0,
                    "negative_hit_count": 0,
                }
            feature_stats[feature_name]["total"] += 1
            if result.matches.get(feature_name, False):
                feature_stats[feature_name]["match_count"] += 1
            if result.mismatches.get(feature_name, False):
                feature_stats[feature_name]["mismatch_count"] += 1
            if result.negative_hits.get(feature_name, False):
                feature_stats[feature_name]["negative_hit_count"] += 1
        # Check gates
        sample_pass = (
            result.mismatch_count <= max_mismatches
            and result.negative_hit_count <= max_negative_hits
        )

        if sample_pass:
            passed += 1
        else:
            failed += 1

        logger.info(
            "Sample %s: %d matches, %d mismatches, %d negative hits - %s",
            sample_id,
            result.match_count,
            result.mismatch_count,
            result.negative_hit_count,
            "PASS" if sample_pass else "FAIL",
        )

    return {
        "total_samples": len(samples),
        "passed_samples": passed,
        "failed_samples": failed,
        "pass_rate": passed / len(samples) if samples else 0.0,
        "samples": results,
        "feature_summary": feature_stats,
        "gates": {
            "max_mismatches": max_mismatches,
            "max_negative_hits": max_negative_hits,
        },
    }


def get_failing_features(
    validation_result: Dict[str, Any], min_failure_rate: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Identify features with high failure rates across the batch.

    Args:
        validation_result: Result from validate_generation_batch
        min_failure_rate: Minimum failure rate to include (0.0-1.0)

    Returns:
        List of feature stats sorted by failure rate (descending)
    """
    feature_summary = validation_result.get("feature_summary", {})
    failing_features = []

    for feature_name, stats in feature_summary.items():
        total = stats.get("total", 0)
        if total == 0:
            continue

        mismatch_count = stats.get("mismatch_count", 0)
        negative_hit_count = stats.get("negative_hit_count", 0)
        failure_rate = (mismatch_count + negative_hit_count) / total

        if failure_rate >= min_failure_rate:
            failing_features.append(
                {
                    "feature": feature_name,
                    "total": total,
                    "mismatch_count": mismatch_count,
                    "negative_hit_count": negative_hit_count,
                    "failure_rate": failure_rate,
                    "mismatch_rate": mismatch_count / total,
                    "negative_hit_rate": negative_hit_count / total,
                }
            )
    # Sort by failure rate (descending), then negative hits
    failing_features.sort(
        key=lambda x: (x["failure_rate"], x["negative_hit_count"]), reverse=True
    )

    return failing_features
