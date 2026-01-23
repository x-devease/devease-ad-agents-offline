"""Rule-based recommendation engine.

Fast, transparent recommendations based on statistical patterns
in high-performing vs low-performing creatives.
"""

import logging
from typing import Any, Dict, List

import numpy as np


logger = logging.getLogger(__name__)


class RuleEngine:
    """Rule-based recommendation engine.

    Generates recommendations based on pre-defined patterns discovered
    from statistical analysis of top vs bottom performers.

    Attributes:
        patterns: List of high-confidence patterns
        anti_patterns: List of patterns to avoid
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the rule engine.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.patterns: List[Dict[str, Any]] = []
        self.anti_patterns: List[Dict[str, Any]] = []

    def load_patterns(
        self,
        recommendations: List[Dict[str, Any]],
        anti_patterns: List[Dict[str, Any]] = None,
    ) -> None:
        """Load patterns from analysis results.

        Args:
            recommendations: List of recommendation dictionaries from analysis.
            anti_patterns: Optional list of anti-pattern recommendations.
        """
        # Filter to high-confidence patterns
        self.patterns = [
            rec
            for rec in recommendations
            if rec.get("lift", 0) >= 1.5 and rec.get("high_pct", 0) >= 0.25
        ]

        # Load anti-patterns if provided
        if anti_patterns:
            self.anti_patterns = anti_patterns

        logger.info(
            "Loaded %d high-confidence patterns and %d anti-patterns",
            len(self.patterns),
            len(self.anti_patterns),
        )

    def analyze(
        self, creative: Dict[str, Any], target_col: str = "roas"
    ) -> List[Dict[str, Any]]:
        """Analyze a creative and generate rule-based recommendations.

        Args:
            creative: Creative dictionary with features and current ROAS.
            target_col: Name of the target column.

        Returns:
            List of recommendation dictionaries.
        """
        current_roas = creative.get(target_col, 0)
        recommendations = []

        # Check against high-confidence patterns
        for pattern in self.patterns:
            feature = pattern["feature"]
            recommended_val = pattern["value"]

            # Skip if feature not in creative data
            if feature not in creative:
                continue

            current_val = creative.get(feature)
            has_feature = current_val == recommended_val

            if not has_feature:
                # Calculate potential impact (conservative: 50% of lift)
                lift = pattern["lift"]
                # Avoid division by zero: ensure lift > 1 and current_roas > 0
                if lift > 1 and current_roas > 0:
                    potential_impact = current_roas * (lift - 1) * 0.5
                else:
                    # If lift <= 1 or current_roas is 0, set minimal impact
                    potential_impact = 0.0

                # Determine confidence level
                if lift >= 2.5 and pattern["high_pct"] >= 0.5:
                    confidence = "high"
                elif lift >= 2.0:
                    confidence = "medium"
                else:
                    confidence = "low"

                recommendations.append(
                    {
                        "source": "rule",
                        "feature": feature,
                        "current": (
                            str(current_val)
                            if current_val is not None
                            else "None"
                        ),
                        "recommended": str(recommended_val),
                        "high_performer_pct": pattern["high_pct"],
                        "potential_impact": potential_impact,
                        "confidence": confidence,
                        "reason": (
                            f"Present in {pattern['high_pct']:.1%} of top "
                            f"performers"
                        ),
                        "type": "improvement",
                        "recommendation_type": pattern.get(
                            "recommendation_type", "DO"
                        ),
                    }
                )

        # Check anti-patterns (what to avoid)
        for anti_pattern in self.anti_patterns:
            feature = anti_pattern["feature"]
            bad_value = anti_pattern["value"]

            if feature not in creative:
                continue

            current_val = creative.get(feature)
            has_bad_feature = current_val == bad_value

            if has_bad_feature:
                # Calculate negative impact
                lift = anti_pattern.get("reverse_lift", 1.0)
                # Avoid division by zero: ensure lift > 1 and current_roas > 0
                if lift > 1 and current_roas > 0:
                    potential_gain = current_roas * (lift - 1) * 0.5
                else:
                    # If lift <= 1 or current_roas is 0, set minimal gain
                    potential_gain = 0.0

                recommendations.append(
                    {
                        "source": "rule",
                        "feature": feature,
                        "current": str(current_val),
                        "recommended": f"NOT {bad_value}",
                        "low_performer_pct": anti_pattern.get("low_pct", 0),
                        "potential_impact": potential_gain,
                        "confidence": "medium",
                        "reason": (
                            f"Present in {anti_pattern['low_pct']:.1%} of "
                            f"bottom performers"
                        ),
                        "type": "anti_pattern",
                    }
                )

        # Sort by potential impact
        recommendations.sort(key=lambda x: x["potential_impact"], reverse=True)

        return recommendations[:10]  # Top 10 rule-based recommendations

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded patterns.

        Returns:
            Dictionary with pattern statistics.
        """
        if not self.patterns:
            return {"total": 0}

        lifts = [p["lift"] for p in self.patterns]
        prevelances = [p["high_pct"] for p in self.patterns]

        return {
            "total": len(self.patterns),
            "avg_lift": np.mean(lifts),
            "max_lift": np.max(lifts),
            "avg_prevalence": np.mean(prevelances),
            "features": list(set(p["feature"] for p in self.patterns)),
        }
