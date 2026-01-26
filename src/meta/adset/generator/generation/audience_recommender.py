"""
Audience-level recommendation generation.

Creates recommendations at the adset (audience) level with comprehensive
evidence and confidence scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.utils import Config
from src.meta.adset.generator.generation.audience_aggregator import (
    AudienceAggregator,
    AudienceSegment,
)


class AudienceRecommender:
    """
    Generate audience-level recommendations with evidence.

    Features:
    - Aggregates ad-level predictions to audience level
    - Generates recommendations based on opportunity and confidence
    - Provides comprehensive evidence for each recommendation
    - Supports different recommendation strategies
    """

    def __init__(
        self,
        metric_config: dict = None,
        scale_threshold: float = None,
        pause_threshold: float = None,
        use_percentiles: bool = False,
    ):
        """
        Initialize audience recommender.

        Args:
            metric_config: Metric configuration
            scale_threshold: Threshold for scale_up recommendations
            pause_threshold: Threshold for pause recommendations
            use_percentiles: Use percentile-based thresholds
        """
        self.metric_config = metric_config or Config.get_metric_config()
        self.direction = self.metric_config.get("direction", "maximize")

        # Set thresholds
        if scale_threshold is None:
            scale_threshold = self.metric_config.get("scale_threshold", 100)
        if pause_threshold is None:
            pause_threshold = self.metric_config.get("pause_threshold", -50)

        self.scale_threshold = scale_threshold
        self.pause_threshold = pause_threshold
        self.use_percentiles = use_percentiles

        self.aggregator = AudienceAggregator(metric_config)

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        top_k: int = 50,
        calibrate_predictions: bool = True,
        uncertainty: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate audience-level recommendations.

        Args:
            df: Ad-level dataframe
            predictions: Model predictions
            top_k: Number of top recommendations to return
            calibrate_predictions: If True, cap predictions at historical maximum
            uncertainty: Optional prediction uncertainty (higher = less confident)

        Returns:
            Tuple of (recommendations DataFrame, summary dict)
        """
        # Aggregate to audience level
        audience_df = self.aggregator.aggregate_to_audience(
            df, predictions, uncertainty=uncertainty
        )

        # Calibrate predictions to historical maximum
        if calibrate_predictions:
            audience_df = self._calibrate_predictions(audience_df)

        # Calculate thresholds if using percentiles
        if self.use_percentiles:
            scale_thresh = np.percentile(audience_df["realistic_opportunity"], 75)
            pause_thresh = np.percentile(audience_df["realistic_opportunity"], 25)
        else:
            scale_thresh = self.scale_threshold
            pause_thresh = self.pause_threshold

        # Classify recommendations based on realistic opportunity
        audience_df = self._classify_recommendations(
            audience_df, scale_thresh, pause_thresh
        )

        # Generate evidence for each audience
        audience_df["evidence"] = audience_df.apply(
            lambda row: self._generate_evidence(row, scale_thresh, pause_thresh), axis=1
        )

        # Generate action descriptions
        audience_df["action"] = audience_df.apply(
            lambda row: self._generate_action(row), axis=1
        )

        # Rank and filter to top_k by realistic opportunity
        top_recs = audience_df.nlargest(top_k, "realistic_opportunity")

        # Add rank
        top_recs = top_recs.copy()
        top_recs["rank"] = range(1, len(top_recs) + 1)

        # Create summary
        summary = self._create_summary(top_recs, scale_thresh, pause_thresh)

        return top_recs, summary

    def _calibrate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate predictions to historical maximum performance.

        Caps predicted values at historical max (mean + 2*std ≈ 95th percentile)
        to provide realistic opportunity estimates.

        Args:
            df: Audience dataframe with predictions

        Returns:
            Dataframe with calibrated predictions
        """
        df = df.copy()

        # Calculate historical maximum (95th percentile)
        df["historical_max_95th"] = df["avg_actual"] + 2 * df["actual_std"]

        # Cap prediction at historical maximum
        df["calibrated_predicted"] = df[["avg_predicted", "historical_max_95th"]].min(
            axis=1
        )

        # Calculate realistic opportunity based on calibrated prediction
        if self.direction == "maximize":
            df["realistic_opportunity"] = (
                df["calibrated_predicted"] - df["avg_actual"]
            ) * df["total_spend"]
        else:
            df["realistic_opportunity"] = (
                df["avg_actual"] - df["calibrated_predicted"]
            ) * df["total_spend"]

        # Calculate prediction excess
        df["prediction_excess"] = df["avg_predicted"] - df["historical_max_95th"]
        df["prediction_excess_pct"] = (
            df["prediction_excess"] / df["historical_max_95th"] * 100
        ).clip(lower=0)

        return df

    def _classify_recommendations(
        self, df: pd.DataFrame, scale_thresh: float, pause_thresh: float
    ) -> pd.DataFrame:
        """Classify recommendations based on realistic opportunity scores."""
        df = df.copy()

        # Use realistic_opportunity if available, otherwise fall back to opportunity_score
        opp_col = (
            "realistic_opportunity"
            if "realistic_opportunity" in df.columns
            else "opportunity_score"
        )

        # Initialize defaults
        df["recommendation"] = "hold"
        df["priority"] = "medium"

        # Classify
        high_opp_mask = df[opp_col] > scale_thresh
        low_opp_mask = df[opp_col] < pause_thresh

        df.loc[high_opp_mask, "recommendation"] = "scale_up"
        df.loc[high_opp_mask, "priority"] = "high"

        df.loc[low_opp_mask, "recommendation"] = "optimize_or_pause"
        df.loc[low_opp_mask, "priority"] = "high"

        # Check for new opportunities (high predicted, low actual spend)
        pred_col = (
            "calibrated_predicted"
            if "calibrated_predicted" in df.columns
            else "avg_predicted"
        )
        new_opp_mask = (
            (df[pred_col] > df["avg_actual"] * 1.5)
            & (df["total_spend"] < 1000)
            & (df["confidence"] > 0.5)
        )
        df.loc[new_opp_mask, "recommendation"] = "launch_new"
        df.loc[new_opp_mask, "priority"] = "high"

        return df

    def _generate_evidence(
        self, row: pd.Series, scale_thresh: float, pause_thresh: float
    ) -> str:
        """Generate evidence string for recommendation."""
        evidence_parts = []

        # Use calibrated prediction if available
        pred = row.get(
            "calibrated_predicted",
            row.get("avg_predicted", row.get("avg_predicted", 0)),
        )
        actual = row["avg_actual"]

        # Performance gap
        gap = pred - actual
        gap_pct = (gap / actual * 100) if actual > 0 else 0

        if self.direction == "maximize":
            if gap > 0:
                evidence_parts.append(
                    f"Predicted {gap_pct:.1f}% higher than actual ({pred:.4f} vs {actual:.4f})"
                )
            else:
                evidence_parts.append(
                    f"Predicted {abs(gap_pct):.1f}% lower than actual ({pred:.4f} vs {actual:.4f})"
                )
        else:  # minimize
            if gap < 0:
                evidence_parts.append(
                    f"Predicted {abs(gap_pct):.1f}% better than actual ({pred:.4f} vs {actual:.4f})"
                )
            else:
                evidence_parts.append(
                    f"Predicted {gap_pct:.1f}% worse than actual ({pred:.4f} vs {actual:.4f})"
                )

        # Add historical context if available
        if "historical_max_95th" in row and pd.notna(row["historical_max_95th"]):
            evidence_parts.append(f"Historical max: {row['historical_max_95th']:.4f}")

        # Add prediction excess warning if significant
        if "prediction_excess_pct" in row and row["prediction_excess_pct"] > 10:
            evidence_parts.append(
                f"(Original prediction was {row['prediction_excess_pct']:.0f}% above historical max)"
            )

        # Spend context
        spend = row["total_spend"]
        evidence_parts.append(f"Total spend: ${spend:,.2f}")

        # Opportunity/risk context - use realistic opportunity if available
        opp = row.get("realistic_opportunity", row.get("opportunity_score", 0))
        if opp > scale_thresh:
            evidence_parts.append(f"High opportunity: +${opp:,.2f}")
        elif opp < pause_thresh:
            evidence_parts.append(f"High risk: -${abs(opp):,.2f}")
        elif opp > 0:
            evidence_parts.append(f"Positive opportunity: +${opp:,.2f}")
        else:
            evidence_parts.append(f"Negative risk: -${abs(opp):,.2f}")

        # Confidence
        conf = row["confidence"]
        evidence_parts.append(f"Confidence: {conf:.1%}")

        # Ad count
        num_ads = row["num_ads"]
        evidence_parts.append(f"Based on {num_ads} ad(s)")

        # Targeting context
        if "targeting_summary" in row and pd.notna(row["targeting_summary"]):
            evidence_parts.append(f"Targeting: {row['targeting_summary']}")

        return "; ".join(evidence_parts)

    def _generate_action(self, row: pd.Series) -> str:
        """Generate actionable recommendation."""
        rec = row["recommendation"]
        opp = row.get("realistic_opportunity", row.get("opportunity_score", 0))
        conf = row["confidence"]

        if rec == "scale_up":
            if conf > 0.7:
                return f"Increase budget by 20-50% (high confidence: ${opp:,.2f} opportunity)"
            else:
                return f"Consider increasing budget (moderate confidence: ${opp:,.2f} opportunity)"

        elif rec == "optimize_or_pause":
            if conf > 0.7:
                return f"Review targeting and creative, or pause if underperforming continues (risk: -${abs(opp):,.2f})"
            else:
                return (
                    f"Monitor closely and optimize settings (risk: -${abs(opp):,.2f})"
                )

        elif rec == "launch_new":
            return f"Launch new similar audience - high potential with low current investment (${opp:,.2f} opportunity)"

        else:  # hold
            return f"Maintain current settings - performance is acceptable"

    def _create_summary(
        self, recs: pd.DataFrame, scale_thresh: float, pause_thresh: float
    ) -> Dict:
        """Create summary statistics."""
        opp_col = (
            "realistic_opportunity"
            if "realistic_opportunity" in recs.columns
            else "opportunity_score"
        )

        return {
            "metric": self.metric_config["display_name"],
            "total_audiences": len(recs),
            "scale_up": int((recs["recommendation"] == "scale_up").sum()),
            "optimize_or_pause": int(
                (recs["recommendation"] == "optimize_or_pause").sum()
            ),
            "launch_new": int((recs["recommendation"] == "launch_new").sum()),
            "hold": int((recs["recommendation"] == "hold").sum()),
            "total_opportunity": (
                float(recs[recs[opp_col] > 0][opp_col].sum()) if len(recs) > 0 else 0.0
            ),
            "total_risk": (
                float(abs(recs[recs[opp_col] < 0][opp_col]).sum())
                if len(recs) > 0
                else 0.0
            ),
            "avg_confidence": (
                float(recs["confidence"].mean()) if len(recs) > 0 else 0.0
            ),
            "thresholds": {"scale": scale_thresh, "pause": pause_thresh},
        }

    def save_recommendations(
        self, recs: pd.DataFrame, summary: Dict, path: str
    ) -> None:
        """Save recommendations and summary to CSV."""
        # Select and order columns for output
        output_cols = [
            "rank",
            "adset_id",
            "adset_name",
            "recommendation",
            "priority",
            "opportunity_score",
            "realistic_opportunity",
            "confidence",
            "avg_predicted",
            "avg_actual",
            "total_spend",
            "num_ads",
            "targeting_summary",
            "evidence",
            "action",
        ]

        # Add headroom columns if available
        if "historical_max_95th" in recs.columns:
            output_cols.insert(
                output_cols.index("avg_actual") + 1, "historical_max_95th"
            )
        if "calibrated_predicted" in recs.columns:
            output_cols.insert(
                (
                    output_cols.index("historical_max_95th") + 1
                    if "historical_max_95th" in output_cols
                    else output_cols.index("avg_actual") + 1
                ),
                "calibrated_predicted",
            )

        # Filter to available columns
        output_cols = [c for c in output_cols if c is not None and c in recs.columns]

        # Add metric name to filename
        metric_name = (
            self.metric_config.get("display_name", "").lower().replace(" ", "_")
        )
        if path.endswith(".csv"):
            output_path = path.replace(".csv", f"_{metric_name}.csv")
        else:
            output_path = f"{path}_{metric_name}.csv"

        recs[output_cols].to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")

    def print_summary(self, summary: Dict) -> None:
        """Print formatted summary."""
        print(f"\n{'=' * 60}")
        print(f"AUDIENCE RECOMMENDATIONS SUMMARY - {summary['metric'].upper()}")
        print(f"{'=' * 60}")
        print(f"  Total audiences: {summary['total_audiences']}")
        print(
            f"  Scale up: {summary['scale_up']} ({100*summary['scale_up']/summary['total_audiences']:.1f}%)"
        )
        print(
            f"  Launch new: {summary['launch_new']} ({100*summary['launch_new']/summary['total_audiences']:.1f}%)"
        )
        print(
            f"  Optimize/pause: {summary['optimize_or_pause']} ({100*summary['optimize_or_pause']/summary['total_audiences']:.1f}%)"
        )
        print(
            f"  Hold: {summary['hold']} ({100*summary['hold']/summary['total_audiences']:.1f}%)"
        )
        print(f"\n  Total opportunity: ${summary['total_opportunity']:,.2f}")
        print(f"  Total risk: ${summary['total_risk']:,.2f}")
        print(f"  Average confidence: {summary['avg_confidence']:.1%}")
        print(
            f"  Thresholds: scale={summary['thresholds']['scale']:.2f}, pause={summary['thresholds']['pause']:.2f}"
        )
        print(f"{'=' * 60}\n")
