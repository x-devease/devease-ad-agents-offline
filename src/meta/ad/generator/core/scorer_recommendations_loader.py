"""
Creative Scorer Recommendations Loader

Loads recommendations from the creative scorer offline repository
and converts them to the format expected by the prompt generation pipeline.

The creative scorer outputs an array of per-creative recommendations:
[
  {
    "creative_id": "...",
    "current_roas": 0.23,
    "predicted_roas": 0.23,
    "recommendations": [
      {
        "source": "rule",
        "feature": "direction",
        "current": "side",
        "recommended": "Overhead",
        "lift": 7.0,
        "high_performer_pct": 0.58,
        "potential_impact": 0.69,
        "confidence": "high",
        "reason": "Present in 58.3% of top performers",
        "type": "improvement",
        "priority_score": 3.0
      }
    ],
    "confidence_scores": {...},
    "timestamp": "..."
  }
]

This loader aggregates across all creatives and converts to the internal format.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)
# Default thresholds for filtering recommendations
DEFAULT_MIN_CONFIDENCE_LEVEL = "medium"
DEFAULT_MIN_HIGH_PERFORMER_PCT = 0.1  # 10%
DEFAULT_MIN_PRIORITY_SCORE = 1.0


class CreativeScorerLoader:
    """
    Load recommendations from creative scorer output.

    Aggregates per-creative recommendations and converts them to
    a format suitable for prompt generation.
    """

    def __init__(
        self,
        recommendations_path: Path,
        min_confidence: Optional[str] = None,
        min_high_performer_pct: Optional[float] = None,
        min_priority_score: Optional[float] = None,
    ):
        """
        Initialize loader.

        Args:
            recommendations_path: Path to recommendations.json from creative scorer repository
                         (Note: ad/miner outputs .md format, use ad_miner_adapter for that)
            min_confidence: Minimum confidence level (high, medium, low)
            min_high_performer_pct: Minimum high performer percentage (0.0-1.0)
            min_priority_score: Minimum priority score
        """
        self.recommendations_path = Path(recommendations_path)
        self.min_confidence = min_confidence or DEFAULT_MIN_CONFIDENCE_LEVEL
        self.min_high_performer_pct = (
            min_high_performer_pct
            if min_high_performer_pct is not None
            else DEFAULT_MIN_HIGH_PERFORMER_PCT
        )
        self.min_priority_score = (
            min_priority_score
            if min_priority_score is not None
            else DEFAULT_MIN_PRIORITY_SCORE
        )
        # Confidence level ordering
        self._confidence_levels = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "very_high": 4,
        }

    def load(self) -> Dict[str, Any]:
        """
        Load and aggregate recommendations.

        Returns:
            Dict with aggregated recommendations

        Raises:
            FileNotFoundError: If recommendations file doesn't exist
            ValueError: If recommendations format is invalid
        """
        if not self.recommendations_path.exists():
            raise FileNotFoundError(
                f"Recommendations file not found: {self.recommendations_path}"
            )

        with open(self.recommendations_path, "r", encoding="utf-8") as f:
            creative_recommendations = json.load(f)

        if not isinstance(creative_recommendations, list):
            raise ValueError(
                f"Expected list of recommendations, got {type(creative_recommendations)}"
            )

        logger.info(
            "Loaded %d creative recommendations from %s",
            len(creative_recommendations),
            self.recommendations_path,
        )
        # Aggregate recommendations across all creatives
        aggregated = self._aggregate_recommendations(creative_recommendations)

        logger.info(
            "Aggregated to %d unique features across %d creatives",
            len(aggregated["feature_recommendations"]),
            len(creative_recommendations),
        )

        return aggregated

    def _aggregate_recommendations(
        self, creative_recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate recommendations across all creatives.

        Args:
            creative_recommendations: List of per-creative recommendation dicts

        Returns:
            Aggregated recommendations dict
        """
        # Track feature-level aggregations
        feature_data: defaultdict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "recommendation_count": 0,
                "confidence_sum": 0,
                "high_performer_pct_sum": 0.0,
                "priority_score_sum": 0.0,
                "lift_sum": 0.0,
                "potential_impact_sum": 0.0,
                "sources": Counter(),
                "current_values": Counter(),
                "recommended_values": Counter(),
                "reasons": [],
            }
        )

        total_creatives = len(creative_recommendations)
        total_recommendations = 0
        filtered_recommendations = 0

        for creative_rec in creative_recommendations:
            recommendations = creative_rec.get("recommendations", [])
            total_recommendations += len(recommendations)

            for rec in recommendations:
                # Filter by confidence
                confidence = rec.get("confidence", "low")
                if self._confidence_levels.get(
                    confidence, 0
                ) < self._confidence_levels.get(self.min_confidence, 0):
                    filtered_recommendations += 1
                    continue
                # Filter by high performer percentage
                high_performer_pct = rec.get("high_performer_pct", 0.0)
                if high_performer_pct < self.min_high_performer_pct:
                    filtered_recommendations += 1
                    continue
                # Filter by priority score
                priority_score = rec.get("priority_score", 0.0)
                if priority_score < self.min_priority_score:
                    filtered_recommendations += 1
                    continue

                feature = rec.get("feature")
                if not feature:
                    continue
                # Aggregate data
                data = feature_data[feature]
                data["recommendation_count"] += 1
                data["confidence_sum"] += self._confidence_levels.get(
                    confidence, 0
                )
                data["high_performer_pct_sum"] += high_performer_pct
                data["priority_score_sum"] += priority_score
                data["lift_sum"] += rec.get("lift", 0.0)
                data["potential_impact_sum"] += rec.get("potential_impact", 0.0)
                data["sources"][rec.get("source", "unknown")] += 1
                data["current_values"][rec.get("current", "unknown")] += 1
                data["recommended_values"][
                    rec.get("recommended", "unknown")
                ] += 1
                # Store a few example reasons
                if len(data["reasons"]) < 3:
                    data["reasons"].append(rec.get("reason", ""))

        logger.info(
            "Filtered %d/%d recommendations below thresholds",
            filtered_recommendations,
            total_recommendations,
        )
        # Build aggregated feature recommendations
        feature_recommendations = []
        for feature, data in feature_data.items():
            count = data["recommendation_count"]
            # Get most common recommended value
            most_recommended = data["recommended_values"].most_common(1)[0][0]

            feature_rec = {
                "feature": feature,
                "recommended_value": most_recommended,
                "current_value": (
                    data["current_values"].most_common(1)[0][0]
                    if data["current_values"]
                    else "unknown"
                ),
                "importance_score": data["priority_score_sum"] / count,
                "confidence": self._level_from_score(
                    data["confidence_sum"] / count
                ),
                "high_performer_pct": data["high_performer_pct_sum"] / count,
                "lift": data["lift_sum"] / count,
                "potential_impact": data["potential_impact_sum"] / count,
                "recommendation_count": count,
                "penetration_pct": (count / total_creatives) * 100,
                "source": (
                    data["sources"].most_common(1)[0][0]
                    if data["sources"]
                    else "unknown"
                ),
                "reason": data["reasons"][0] if data["reasons"] else "",
            }
            feature_recommendations.append(feature_rec)
        # Sort by importance score (priority)
        feature_recommendations.sort(
            key=lambda x: x["importance_score"], reverse=True
        )
        # Build the result dict compatible with prompt generation
        result = {
            "source": str(self.recommendations_path),
            "total_creatives": total_creatives,
            "total_features": len(feature_recommendations),
            "feature_recommendations": feature_recommendations,
            # Legacy format fields for compatibility
            "recommended_features": [
                f["feature"] for f in feature_recommendations
            ],
            "feature_importance": {
                f["feature"]: f["importance_score"]
                for f in feature_recommendations
            },
            "feature_values": {
                f["feature"]: f["recommended_value"]
                for f in feature_recommendations
            },
            "negative_feature_values": {
                f["feature"]: f["current_value"]
                for f in feature_recommendations
                if f["current_value"] != "unknown"
            },
        }

        return result

    def _level_from_score(self, score: float) -> str:
        """
        Convert confidence score to level string.

        Args:
            score: Average confidence score (1-4 scale based on _confidence_levels)

        Returns:
            Confidence level string (low, medium, high, very_high)

        Thresholds align with _confidence_levels mapping:
            very_high: [3.5, 4.0] -> (3+4)/2 = 3.5
            high:      [2.5, 3.5) -> (2+3)/2 = 2.5
            medium:    [1.5, 2.5) -> (1+2)/2 = 1.5
            low:       [0.0, 1.5) -> below (1+2)/2
        """
        if score >= 3.5:
            return "very_high"
        if score >= 2.5:
            return "high"
        if score >= 1.5:
            return "medium"
        return "low"

    def get_top_features(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N features by importance score.

        Args:
            n: Number of top features to return

        Returns:
            List of top feature recommendations
        """
        recommendations = self.load()
        return recommendations["feature_recommendations"][:n]

    def get_features_by_confidence(
        self, confidence_level: str = "high"
    ) -> List[Dict[str, Any]]:
        """
        Get features filtered by confidence level.

        Args:
            confidence_level: Minimum confidence level (high, medium, low)

        Returns:
            List of feature recommendations
        """
        recommendations = self.load()
        min_level = self._confidence_levels.get(confidence_level, 0)

        return [
            f
            for f in recommendations["feature_recommendations"]
            if self._confidence_levels.get(f["confidence"], 0) >= min_level
        ]


def load_scorer_recommendations(
    recommendations_path: Path,
    min_confidence: str = "medium",
    min_high_performer_pct: float = 0.1,
) -> Dict[str, Any]:
    """
    Convenience function to load creative scorer recommendations.

    Args:
        recommendations_path: Path to recommendations.json from creative scorer repository
        min_confidence: Minimum confidence level
        min_high_performer_pct: Minimum high performer percentage

    Returns:
        Aggregated recommendations dict

    Note:
        This function loads JSON format from the creative scorer repository.
        For markdown format from ad/miner, use load_recommendations_as_visual_formula()
        from ad_miner_adapter instead.

    Example:
        recs = load_scorer_recommendations(
            "devease-creative-scorer-offline/data/recommendations/moprobo/taboola/2026-01-21/recommendations.json"
        )
        print(f"Loaded {recs['total_features']} features")
    """
    loader = CreativeScorerLoader(
        recommendations_path=recommendations_path,
        min_confidence=min_confidence,
        min_high_performer_pct=min_high_performer_pct,
    )
    return loader.load()
