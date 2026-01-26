"""
Prompt Quality Validator.

Validates prompt quality before image generation.
Checks feature coverage, length, brand identifiers, and provides actionable improvements.

Rules-based validation - no ML.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score for a generated prompt."""

    total_score: float  # 0-100
    feature_coverage: float  # 0-100
    length_score: float  # 0-100
    brand_integrity_score: float  # 0-100
    technical_quality_score: float  # 0-100

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_score": self.total_score,
            "feature_coverage": self.feature_coverage,
            "length_score": self.length_score,
            "brand_integrity": self.brand_integrity_score,
            "technical_quality": self.technical_quality_score,
        }


@dataclass
class ValidationIssue:
    """A validation issue found in prompt."""

    severity: str  # "critical", "warning", "info"
    category: str  # "coverage", "length", "brand", "technical"
    message: str
    suggestion: Optional[str] = None


class PromptQualityValidator:
    """
    Validates prompt quality before image generation.

    Checks:
    - Feature coverage (% of features from formula)
    - Prompt length (optimal 800-1800 chars for Nano Banana)
    - Brand identifiers preserved
    - Color specifications present
    - Technical quality indicators
    """

    # Quality thresholds
    MIN_FEATURE_COVERAGE = 0.95  # 95%
    OPTIMAL_LENGTH_MIN = 800
    OPTIMAL_LENGTH_MAX = 1800
    MAX_LENGTH = 2500  # Absolute maximum

    # Critical keywords that must be preserved
    CRITICAL_KEYWORDS = [
        "strictly maintain",
        "exact geometric",
        "product integrity",
        "no blur",
        "no distortion",
        "no missing",
        "delta e",
        "color accuracy",
    ]

    def __init__(self):
        """Initialize validator."""
        pass

    def validate(
        self,
        prompt: str,
        visual_formula: Dict[str, Any],
        min_coverage: float = MIN_FEATURE_COVERAGE,
    ) -> QualityScore:
        """
        Validate prompt quality.

        Args:
            prompt: Generated prompt text
            visual_formula: Visual formula with entrance_features, headroom_features
            min_coverage: Minimum feature coverage threshold (default 95%)

        Returns:
            QualityScore object with detailed scoring
        """
        logger.info("Validating prompt quality...")

        # Calculate component scores
        feature_coverage = self._validate_feature_coverage(prompt, visual_formula)
        length_score = self._validate_length(prompt)
        brand_score = self._validate_brand_integrity(prompt)
        technical_score = self._validate_technical_quality(prompt)

        # Calculate total score (weighted average)
        total_score = (
            feature_coverage * 0.40 +  # 40% weight - most important
            length_score * 0.20 +  # 20% weight
            brand_score * 0.25 +  # 25% weight
            technical_score * 0.15  # 15% weight
        )

        quality_score = QualityScore(
            total_score=total_score,
            feature_coverage=feature_coverage,
            length_score=length_score,
            brand_integrity_score=brand_score,
            technical_quality_score=technical_score,
        )

        logger.info(
            "Quality score: %.0f/100 (coverage: %.0f, length: %.0f, brand: %.0f, technical: %.0f)",
            total_score,
            feature_coverage,
            length_score,
            brand_score,
            technical_score,
        )

        return quality_score

    def suggest_improvements(
        self,
        prompt: str,
        quality_score: QualityScore,
        visual_formula: Dict[str, Any],
    ) -> List[str]:
        """
        Suggest improvements to boost quality score.

        Args:
            prompt: Generated prompt text
            quality_score: Quality score from validate()
            visual_formula: Visual formula

        Returns:
            List of actionable improvement suggestions
        """
        suggestions = []

        # Feature coverage suggestions
        if quality_score.feature_coverage < 80:
            suggestions.append(
                "CRITICAL: Feature coverage below 80%. "
                "Add missing entrance features from formula."
            )
        elif quality_score.feature_coverage < 95:
            suggestions.append(
                f"Feature coverage {quality_score.feature_coverage:.0f}% - "
                "add missing features to reach 95% threshold"
            )

        # Length suggestions
        prompt_len = len(prompt)
        if prompt_len < self.OPTIMAL_LENGTH_MIN:
            suggestions.append(
                f"Prompt too short ({prompt_len} chars). "
                f"Add detail to reach {self.OPTIMAL_LENGTH_MIN}-{self.OPTIMAL_LENGTH_MAX} chars."
            )
        elif prompt_len > self.MAX_LENGTH:
            suggestions.append(
                f"Prompt too long ({prompt_len} chars). "
                f"Trim to under {self.MAX_LENGTH} chars for optimal generation."
            )

        # Brand integrity suggestions
        if quality_score.brand_integrity_score < 70:
            suggestions.append(
                "Brand identifiers may be compromised. "
                "Ensure text/label preservation rules are explicit."
            )

        # Technical quality suggestions
        if quality_score.technical_quality_score < 70:
            suggestions.append(
                "Add technical specifications: camera, lighting, color accuracy, "
                "material textures"
            )

        return suggestions

    def _validate_feature_coverage(
        self,
        prompt: str,
        visual_formula: Dict[str, Any],
    ) -> float:
        """
        Validate feature coverage percentage.

        Returns:
            Score 0-100 based on coverage percentage
        """
        # Extract all features from formula
        total_features = 0
        covered_features = 0

        # Count entrance features
        for feat in visual_formula.get("entrance_features", []):
            total_features += 1
            feature_name = feat.get("feature_name", "")
            feature_value = feat.get("feature_value", "")
            if self._check_feature_in_prompt(prompt, feature_name, feature_value):
                covered_features += 1

        # Count headroom features
        for feat in visual_formula.get("headroom_features", []):
            total_features += 1
            feature_name = feat.get("feature_name", "")
            feature_value = feat.get("feature_value", "")
            if self._check_feature_in_prompt(prompt, feature_name, feature_value):
                covered_features += 1

        # Count synergy features
        for pair in visual_formula.get("synergy_pairs", []):
            total_features += 2  # Two features per pair
            feat1 = pair.get("feature1_name", "")
            feat1_val = pair.get("feature1_value", "")
            feat2 = pair.get("feature2_name", "")
            feat2_val = pair.get("feature2_value", "")

            if self._check_feature_in_prompt(prompt, feat1, feat1_val):
                covered_features += 1
            if self._check_feature_in_prompt(prompt, feat2, feat2_val):
                covered_features += 1

        if total_features == 0:
            return 100.0  # No features to validate

        coverage = covered_features / total_features
        # Convert to score (100 * coverage)
        return coverage * 100

    def _validate_length(self, prompt: str) -> float:
        """
        Validate prompt length.

        Returns:
            Score 0-100 based on optimal length range
        """
        prompt_len = len(prompt)

        # Perfect score if in optimal range
        if self.OPTIMAL_LENGTH_MIN <= prompt_len <= self.OPTIMAL_LENGTH_MAX:
            return 100.0

        # Penalize if too short
        if prompt_len < self.OPTIMAL_LENGTH_MIN:
            # Linear penalty: 50% at 0 chars, 100% at MIN
            ratio = prompt_len / self.OPTIMAL_LENGTH_MIN
            return 50.0 + (50.0 * ratio)

        # Penalize if too long but under max
        if prompt_len <= self.MAX_LENGTH:
            # Linear penalty from 100% at MAX to 50% at MAX_LENGTH
            excess = prompt_len - self.OPTIMAL_LENGTH_MAX
            excess_limit = self.MAX_LENGTH - self.OPTIMAL_LENGTH_MAX
            ratio = 1.0 - (excess / excess_limit)
            return 50.0 + (50.0 * ratio)

        # Zero score if over absolute maximum
        return 0.0

    def _validate_brand_integrity(self, prompt: str) -> float:
        """
        Validate brand integrity preservation.

        Returns:
            Score 0-100 based on brand keyword presence
        """
        prompt_lower = prompt.lower()

        # Check for critical keywords
        critical_count = 0
        for keyword in self.CRITICAL_KEYWORDS:
            if keyword in prompt_lower:
                critical_count += 1

        # Score based on critical keyword presence
        return (critical_count / len(self.CRITICAL_KEYWORDS)) * 100

    def _validate_technical_quality(self, prompt: str) -> float:
        """
        Validate technical quality indicators.

        Returns:
            Score 0-100 based on technical elements present
        """
        prompt_lower = prompt.lower()

        # Technical quality indicators
        indicators = {
            "camera": ["canon", "eos", "f/", "iso", "aperture"],
            "lighting": ["three-point", "key light", "fill light", "rim light"],
            "color": ["delta e", "color accuracy", "metamerism", "bit depth"],
            "material": ["surface grain", "texture", "anisotropic", "subsurface"],
            "composition": ["depth of field", "bokeh", "composition", "focal point"],
        }

        categories_covered = 0
        for category, keywords in indicators.items():
            if any(kw in prompt_lower for kw in keywords):
                categories_covered += 1

        # Score based on categories covered (5 categories max)
        return (categories_covered / len(indicators)) * 100

    def _check_feature_in_prompt(
        self,
        prompt: str,
        feature_name: str,
        feature_value: str,
    ) -> bool:
        """
        Check if a feature appears in the prompt.

        Args:
            prompt: Prompt text
            feature_name: Feature name
            feature_value: Feature value

        Returns:
            True if feature detected in prompt
        """
        prompt_lower = prompt.lower()
        feature_lower = feature_name.lower().replace("_", " ")
        value_lower = feature_value.lower()

        # Direct match
        if feature_lower in prompt_lower or value_lower in prompt_lower:
            return True

        # Semantic equivalents (basic set)
        equivalents = {
            "warm-dominant": ["warm", "warm tones"],
            "partial": ["partially visible", "partial view"],
            "dominant": ["prominently", "prominent"],
            "balanced": ["balanced", "harmonious"],
        }

        for equiv_key, equiv_values in equivalents.items():
            if equiv_key in value_lower:
                if any(equiv in prompt_lower for equiv in equiv_values):
                    return True

        return False


def validate_prompt_before_generation(
    prompt: str,
    visual_formula: Dict[str, Any],
    min_score: float = 70.0,
) -> tuple[QualityScore, List[str]]:
    """
    Convenience function to validate prompt and get suggestions.

    Args:
        prompt: Generated prompt text
        visual_formula: Visual formula
        min_score: Minimum acceptable quality score (default 70)

    Returns:
        Tuple of (QualityScore, suggestions_list)

    Example:
        >>> score, suggestions = validate_prompt_before_generation(prompt, formula)
        >>> if score.total_score < 70:
        ...     print("Suggestions:", suggestions)
    """
    validator = PromptQualityValidator()
    quality_score = validator.validate(prompt, visual_formula)
    suggestions = validator.suggest_improvements(prompt, quality_score, visual_formula)

    if quality_score.total_score < min_score:
        logger.warning(
            "Prompt quality score %.0f below minimum %.0f. Suggestions: %s",
            quality_score.total_score,
            min_score,
            suggestions,
        )

    return quality_score, suggestions
