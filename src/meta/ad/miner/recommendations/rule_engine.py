"""Rule-based recommendation engine.

Fast, transparent recommendations based on statistical patterns
in high-performing vs low-performing creatives.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Hard-coded thresholds (SELF_REFLECTION: no tuning on test data)
TOP_PCT = 0.25
BOTTOM_PCT = 0.25
LIFT_MIN = 1.5
PREVALENCE_MIN = 0.10
MAX_RECS_PER_CREATIVE = 10


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

        return recommendations[:MAX_RECS_PER_CREATIVE]

    def _discover_patterns_from_df(
        self,
        df: pd.DataFrame,
        target_col: str = "roas_parsed",
        top_pct: float = TOP_PCT,
        bottom_pct: float = BOTTOM_PCT,
        lift_min: float = LIFT_MIN,
        prevalence_min: float = PREVALENCE_MIN,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Discover DO/anti-patterns from top vs bottom performers.

        Uses top_pct / bottom_pct of rows by target; lift and prevalence
        thresholds are hard-coded to avoid data leakage.
        """
        candidates = [
            c
            for c in ["roas_parsed", "roas", "mean_roas", "total_roas"]
            if c in df.columns
        ]
        tc = target_col if target_col in df.columns else (candidates[0] if candidates else None)
        if tc is None:
            logger.warning("No ROAS-like column found; using no patterns")
            return [], []

        df = df.dropna(subset=[tc]).copy()
        if len(df) < 20:
            logger.warning("Too few rows with ROAS; skipping pattern discovery")
            return [], []

        exclude = {tc, "id", "creative_id", "filename", "image_path"}
        feature_cols = [
            c
            for c in df.select_dtypes(include=["object", "bool", "category"]).columns
            if c not in exclude and df[c].nunique() >= 1 and df[c].nunique() < 100
        ]
        if not feature_cols:
            logger.warning("No suitable feature columns for pattern discovery")
            return [], []

        q_top = 1.0 - top_pct
        q_bot = bottom_pct
        t = df[tc].quantile([q_bot, q_top])
        bot_thresh, top_thresh = t.iloc[0], t.iloc[1]
        top_df = df[df[tc] >= top_thresh]
        bot_df = df[df[tc] <= bot_thresh]
        n_top = max(1, len(top_df))
        n_bot = max(1, len(bot_df))

        patterns: List[Dict[str, Any]] = []
        anti_patterns: List[Dict[str, Any]] = []

        for col in feature_cols:
            for val in df[col].dropna().unique():
                val = str(val).strip()
                if not val or val.lower() in ("nan", "none", ""):
                    continue
                top_has = (top_df[col].astype(str).str.strip() == val).sum()
                bot_has = (bot_df[col].astype(str).str.strip() == val).sum()
                p_top = top_has / n_top
                p_bot = bot_has / n_bot
                if p_bot <= 0:
                    lift = float("inf") if p_top > 0 else 1.0
                else:
                    lift = p_top / p_bot
                if lift >= lift_min and p_top >= prevalence_min:
                    patterns.append({
                        "feature": col,
                        "value": val,
                        "lift": lift,
                        "high_pct": p_top,
                        "recommendation_type": "DO",
                    })
                rev_lift = p_bot / p_top if p_top > 0 else (float("inf") if p_bot > 0 else 1.0)
                if rev_lift >= lift_min and p_bot >= prevalence_min:
                    anti_patterns.append({
                        "feature": col,
                        "value": val,
                        "low_pct": p_bot,
                        "reverse_lift": rev_lift,
                    })

        logger.info(
            "Discovered %d patterns and %d anti-patterns from %d top / %d bottom rows",
            len(patterns),
            len(anti_patterns),
            n_top,
            n_bot,
        )
        return patterns, anti_patterns

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        target_col: str = "roas_parsed",
        discover_patterns: bool = True,
    ) -> Dict[str, Any]:
        """Run pattern discovery (if enabled), then aggregate recommendations.

        Returns a single dict compatible with format_recs_as_prompts:
        {"recommendations": [...], "creative_id": "aggregated", "confidence_scores": {...}}
        """
        if discover_patterns:
            pats, anti = self._discover_patterns_from_df(df, target_col=target_col)
            self.load_patterns(pats, anti)

        all_recs: List[Dict[str, Any]] = []
        candidates = [
            c
            for c in ["roas_parsed", "roas", "mean_roas", "total_roas"]
            if c in df.columns
        ]
        tc = target_col if target_col in df.columns else (candidates[0] if candidates else "roas_parsed")

        for _, row in df.iterrows():
            creative = row.to_dict()
            recs = self.analyze(creative, target_col=tc)
            for r in recs:
                all_recs.append(dict(r))

        # Deduplicate by (feature, recommended) keeping highest impact
        seen: Dict[tuple, Dict[str, Any]] = {}
        for r in all_recs:
            key = (r["feature"], r.get("recommended", ""))
            prev = seen.get(key)
            if prev is None or (r.get("potential_impact") or 0) > (prev.get("potential_impact") or 0):
                seen[key] = r
        unique = list(seen.values())
        unique.sort(key=lambda x: -(x.get("potential_impact") or 0))

        conf = {"combined_confidence": 0.7}
        if unique:
            conf["combined_confidence"] = min(
                0.95,
                0.5 + 0.1 * sum(1 for r in unique if r.get("confidence") == "high"),
            )

        return {
            "recommendations": unique,
            "creative_id": "aggregated",
            "current_roas": float(df[tc].mean()) if tc in df.columns else 0.0,
            "predicted_roas": float(df[tc].mean()) if tc in df.columns else 0.0,
            "confidence_scores": conf,
        }

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
