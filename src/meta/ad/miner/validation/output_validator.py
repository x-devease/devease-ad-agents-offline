"""Output schema validator for mined patterns JSON."""

from pathlib import Path
from typing import Any, Dict
import json
import logging

logger = logging.getLogger(__name__)


class OutputSchemaValidator:
    """Validate mined patterns JSON against schema."""

    SCHEMA_VERSION = "2.0"

    REQUIRED_TOP_LEVEL_KEYS = [
        "metadata",
        "patterns",
        "anti_patterns",
        "low_priority_insights",
    ]

    REQUIRED_METADATA_KEYS = [
        "schema_version",
        "customer",
        "product",
        "branch",
        "campaign_goal",
        "granularity_level",
        "sample_size",
        "analysis_date",
    ]

    # Valid enum values
    VALID_CONFIDENCE = ["high", "medium", "low"]
    VALID_PATTERN_TYPES = ["DO", "DO_CONVERSION", "DO_AWARENESS", "DO_TRAFFIC", "DON'T", "ANTI_PATTERN"]

    def __init__(self, json_path: str | Path):
        """
        Initialize validator.

        Args:
            json_path: Path to JSON file to validate
        """
        self.json_path = Path(json_path)
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> bool:
        """
        Validate JSON against schema.

        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []

        # Load JSON
        try:
            with open(self.json_path) as f:
                data = json.load(f)
        except Exception as e:
            self.errors.append(f"Failed to read JSON: {e}")
            return False

        # Run validation checks
        self._validate_top_level(data)
        self._validate_metadata(data.get("metadata", {}))
        self._validate_patterns(data.get("patterns", []))
        self._validate_anti_patterns(data.get("anti_patterns", []))
        self._validate_low_priority_insights(data.get("low_priority_insights", []))
        self._validate_ranges(data)
        self._validate_consistency(data)

        is_valid = len(self.errors) == 0

        if not is_valid:
            logger.error(f"Output validation failed with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  ✗ {error}")

        if self.warnings:
            logger.warning(f"Output validation produced {len(self.warnings)} warnings")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")

        return is_valid

    def _validate_top_level(self, data: Dict) -> None:
        """Validate top-level structure."""
        if not isinstance(data, dict):
            self.errors.append(f"Root must be object, got {type(data)}")
            return

        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            if key not in data:
                self.errors.append(f"Missing required top-level key: {key}")

    def _validate_metadata(self, metadata: Dict) -> None:
        """Validate metadata section."""
        if not isinstance(metadata, dict):
            self.errors.append("metadata must be object")
            return

        for key in self.REQUIRED_METADATA_KEYS:
            if key not in metadata:
                self.errors.append(f"Missing required metadata key: {key}")

        # Validate schema version
        if metadata.get("schema_version") != self.SCHEMA_VERSION:
            self.warnings.append(
                f"Schema version mismatch: expected {self.SCHEMA_VERSION}, "
                f"got {metadata.get('schema_version')}"
            )

        # Validate granularity level
        granularity = metadata.get("granularity_level")
        if granularity is not None:
            if not isinstance(granularity, int) or not (1 <= granularity <= 4):
                self.errors.append(f"granularity_level must be integer 1-4, got {granularity}")

        # Validate sample size
        sample_size = metadata.get("sample_size")
        if sample_size is not None:
            if not isinstance(sample_size, int) or sample_size <= 0:
                self.errors.append(f"sample_size must be positive integer, got {sample_size}")

        # Validate data quality if present
        data_quality = metadata.get("data_quality", {})
        if data_quality:
            completeness = data_quality.get("completeness_score")
            if completeness is not None:
                if not (0.0 <= completeness <= 1.0):
                    self.errors.append(f"completeness_score must be 0-1, got {completeness}")

    def _validate_patterns(self, patterns: list) -> None:
        """Validate patterns array."""
        if not isinstance(patterns, list):
            self.errors.append("patterns must be array")
            return

        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, dict):
                self.errors.append(f"Pattern {i}: must be object")
                continue

            # Required fields
            required_fields = [
                "feature", "value", "pattern_type", "confidence",
                "roas_lift_multiple", "roas_lift_pct",
                "top_quartile_prevalence", "priority_score"
            ]
            for field in required_fields:
                if field not in pattern:
                    self.errors.append(f"Pattern {i}: missing required field '{field}'")

            # Validate confidence
            confidence = pattern.get("confidence")
            if confidence not in self.VALID_CONFIDENCE:
                self.errors.append(f"Pattern {i}: invalid confidence '{confidence}'")

            # Validate pattern_type
            pattern_type = pattern.get("pattern_type")
            if pattern_type not in self.VALID_PATTERN_TYPES:
                self.errors.append(f"Pattern {i}: invalid pattern_type '{pattern_type}'")

            # Validate ranges
            roas_lift = pattern.get("roas_lift_multiple")
            if roas_lift is not None and roas_lift < 1.0:
                self.errors.append(f"Pattern {i}: roas_lift_multiple must be >= 1.0, got {roas_lift}")

            top_prev = pattern.get("top_quartile_prevalence")
            if top_prev is not None and not (0.0 <= top_prev <= 1.0):
                self.errors.append(f"Pattern {i}: top_quartile_prevalence must be 0-1, got {top_prev}")

            priority_score = pattern.get("priority_score")
            if priority_score is not None and not (0.0 <= priority_score <= 10.0):
                self.errors.append(f"Pattern {i}: priority_score must be 0-10, got {priority_score}")

    def _validate_anti_patterns(self, anti_patterns: list) -> None:
        """Validate anti_patterns array."""
        if not isinstance(anti_patterns, list):
            self.errors.append("anti_patterns must be array")
            return

        for i, pattern in enumerate(anti_patterns):
            if not isinstance(pattern, dict):
                self.errors.append(f"Anti-pattern {i}: must be object")
                continue

            required_fields = [
                "feature", "avoid_value", "pattern_type", "confidence",
                "roas_penalty_multiple", "roas_penalty_pct", "bottom_quartile_prevalence"
            ]
            for field in required_fields:
                if field not in pattern:
                    self.errors.append(f"Anti-pattern {i}: missing required field '{field}'")

            # Validate confidence
            confidence = pattern.get("confidence")
            if confidence not in self.VALID_CONFIDENCE:
                self.errors.append(f"Anti-pattern {i}: invalid confidence '{confidence}'")

            # Validate penalty
            penalty = pattern.get("roas_penalty_multiple")
            if penalty is not None and not (0.0 <= penalty <= 1.0):
                self.errors.append(f"Anti-pattern {i}: roas_penalty_multiple must be 0-1, got {penalty}")

    def _validate_low_priority_insights(self, insights: list) -> None:
        """Validate low_priority_insights array."""
        if not isinstance(insights, list):
            self.errors.append("low_priority_insights must be array")
            return

        for i, insight in enumerate(insights):
            if not isinstance(insight, dict):
                self.errors.append(f"Insight {i}: must be object")
                continue

            required_fields = ["feature", "value", "confidence", "roas_lift_multiple", "reason"]
            for field in required_fields:
                if field not in insight:
                    self.errors.append(f"Insight {i}: missing required field '{field}'")

            # Low priority insights should always have "low" confidence
            confidence = insight.get("confidence")
            if confidence != "low":
                self.warnings.append(f"Insight {i}: has '{confidence}' confidence, expected 'low'")

    def _validate_ranges(self, data: Dict) -> None:
        """Validate value ranges across document."""
        metadata = data.get("metadata", {})

        # Validate data quality ranges
        data_quality = metadata.get("data_quality", {})
        for key in ["avg_roas", "top_quartile_roas", "bottom_quartile_roas"]:
            value = data_quality.get(key)
            if value is not None and value < 0:
                self.errors.append(f"data_quality.{key} must be >= 0, got {value}")

    def _validate_consistency(self, data: Dict) -> None:
        """Validate internal consistency."""
        metadata = data.get("metadata", {})
        patterns = data.get("patterns", [])
        anti_patterns = data.get("anti_patterns", [])

        # Check that pattern count doesn't exceed sample size
        sample_size = metadata.get("sample_size", 0)
        total_patterns = len(patterns) + len(anti_patterns)

        if total_patterns > sample_size:
            self.warnings.append(
                f"Total patterns ({total_patterns}) exceeds sample size ({sample_size})"
            )

        # Check that priority scores are sorted (descending)
        priorities = [p.get("priority_score", 0) for p in patterns]
        if priorities != sorted(priorities, reverse=True):
            self.warnings.append("Patterns are not sorted by priority_score (descending)")

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get validation report.

        Returns:
            Dict with validation results
        """
        return {
            "valid": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "schema_version": self.SCHEMA_VERSION,
        }
