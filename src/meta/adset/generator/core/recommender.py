"""
Generic recommender system for ad optimization.

Fully configurable through:
- YAML config files
- Percentile-based thresholds
- Custom recommendation strategies
- Configurable opportunity scoring
- Integration with Config system

Usage:
    # Simple usage (default ROAS)
    from src.meta.adset.generator.core.recommender import create_recommender
    recommender = create_recommender()

    # With custom thresholds
    recommender = create_roas_recommender(scale_threshold=200)

    # Percentile-based (auto-adjusts to data)
    recommender = create_percentile_recommender()

    # Fully custom
    recommender = create_custom_recommender(
        name='my_metric',
        target_column='my_column',
        direction='maximize',
        strategies=[...]
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field

# =============================================================================
# CORE CLASSES
# =============================================================================


@dataclass
class RecommendationStrategy:
    """Configuration for a recommendation category."""

    name: str  # e.g., 'scale_up', 'optimize', 'hold'
    condition: str  # 'greater_than', 'less_than', 'between', 'outside'
    threshold: Union[float, Tuple[float, float]]  # Single value or (min, max) range
    priority: str = "medium"  # 'high', 'medium', 'low'
    label: Optional[str] = None  # Custom display label
    action: Optional[str] = None  # Action description

    def __post_init__(self):
        if self.label is None:
            self.label = self.name.replace("_", " ").title()
        if self.action is None:
            self.action = f"Consider {self.label.lower()}"


@dataclass
class MetricConfig:
    """
    Configuration for a performance metric.

    Supports:
    - Fixed thresholds
    - Percentile-based thresholds
    - Custom opportunity score functions
    - Multiple recommendation strategies
    """

    name: str
    target_column: str
    direction: Literal["maximize", "minimize"]

    # Threshold configuration
    scale_threshold: Optional[float] = None
    pause_threshold: Optional[float] = None
    use_percentiles: bool = False
    percentile_high: float = 0.75  # For scale_up
    percentile_low: float = 0.25  # For optimize_or_pause

    # Display
    unit: str = ""
    display_name: Optional[str] = None

    # Custom opportunity score function (optional)
    # Signature: (predicted, actual, spend) -> opportunity_score
    opportunity_fn: Optional[Callable] = None

    # Custom recommendation strategies (optional)
    # If None, uses default strategies based on thresholds
    strategies: List[RecommendationStrategy] = field(default_factory=list)

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.name.replace("_", " ").title()

        # Create default strategies if not provided
        if not self.strategies and self.scale_threshold is not None:
            self.strategies = [
                RecommendationStrategy(
                    name="scale_up",
                    condition="greater_than",
                    threshold=self.scale_threshold,
                    priority="high",
                    action="Increase budget/investment",
                ),
                RecommendationStrategy(
                    name="optimize_or_pause",
                    condition="less_than",
                    threshold=self.pause_threshold,
                    priority="high",
                    action="Optimize settings or pause",
                ),
                RecommendationStrategy(
                    name="hold",
                    condition="between",
                    threshold=(self.pause_threshold, self.scale_threshold),
                    priority="medium",
                    action="Maintain current settings",
                ),
            ]


# Default opportunity score functions
def default_opportunity_maximize(predicted, actual, spend):
    """Default opportunity score for maximize metrics (e.g., ROAS)."""
    return (predicted - actual) * spend


def default_opportunity_minimize(predicted, actual, spend):
    """Default opportunity score for minimize metrics (e.g., CPA)."""
    return (actual - predicted) * spend


def roi_opportunity(predicted, actual, spend):
    """ROI-based opportunity score (percentage improvement)."""
    if actual == 0:
        return predicted * spend if predicted > 0 else 0
    return ((predicted - actual) / actual) * spend


class ConfigurableRecommender:
    """
    Fully configurable recommender for ad optimization.

    Features:
    - Load configuration from YAML
    - Percentile-based automatic thresholds
    - Custom recommendation strategies
    - Configurable opportunity scoring
    - Support for any metric direction
    """

    def __init__(
        self,
        config: Optional[MetricConfig] = None,
        load_from_config: Optional[str] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize recommender.

        Args:
            config: MetricConfig object
            load_from_config: Load from 'recommenders' section in Config
            config_path: Load from YAML file
        """
        if config:
            self.config = config
        elif load_from_config:
            self.config = self._load_from_system_config(load_from_config)
        elif config_path:
            self.config = self._load_from_yaml(config_path)
        else:
            # Default ROAS config
            self.config = MetricConfig(
                name="roas",
                target_column="purchase_roas",
                direction="maximize",
                scale_threshold=100,
                pause_threshold=-50,
                opportunity_fn=default_opportunity_maximize,
            )

        self.direction = self.config.direction

    def _load_from_system_config(self, metric_name: str) -> MetricConfig:
        """Load configuration from Config.recommenders section."""
        from src.utils import Config

        try:
            rec_config = Config.get("recommenders", {}).get(metric_name, {})
            return MetricConfig(
                name=metric_name,
                target_column=rec_config.get("target_column", "purchase_roas"),
                direction=rec_config.get("direction", "maximize"),
                scale_threshold=rec_config.get("scale_threshold"),
                pause_threshold=rec_config.get("pause_threshold"),
                use_percentiles=rec_config.get("use_percentiles", False),
                percentile_high=rec_config.get("percentile_high", 0.75),
                percentile_low=rec_config.get("percentile_low", 0.25),
                unit=rec_config.get("unit", ""),
                display_name=rec_config.get("display_name"),
            )
        except Exception:
            # Fallback to default
            return MetricConfig(
                name=metric_name,
                target_column="purchase_roas",
                direction="maximize",
                scale_threshold=100,
                pause_threshold=-50,
            )

    def _load_from_yaml(self, path: Path) -> MetricConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return MetricConfig(**data)

    def _calculate_thresholds_from_percentiles(
        self, opportunity_scores: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate thresholds based on percentiles."""
        scale_threshold = np.percentile(
            opportunity_scores, self.config.percentile_high * 100
        )
        pause_threshold = np.percentile(
            opportunity_scores, self.config.percentile_low * 100
        )
        return scale_threshold, pause_threshold

    def _calculate_opportunity_score(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        spend: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate opportunity score using configured function.

        Args:
            predicted: Predicted metric values
            actual: Actual metric values
            spend: Spend values
            uncertainty: Optional prediction uncertainty (higher = less confident)

        Returns:
            Opportunity scores, discounted by uncertainty if provided
        """
        if self.config.opportunity_fn:
            # Use custom function
            scores = np.array(
                [
                    self.config.opportunity_fn(p, a, s)
                    for p, a, s in zip(predicted, actual, spend)
                ]
            )
        else:
            # Use default based on direction
            if self.direction == "maximize":
                scores = (predicted - actual) * spend
            else:  # minimize
                scores = (actual - predicted) * spend

        # Apply uncertainty discount if provided
        # Higher uncertainty reduces the opportunity score
        # Formula: score * (1 - uncertainty) where uncertainty is normalized to [0, 1]
        # This means high uncertainty (1) → score becomes 0
        # Low uncertainty (0) → score remains unchanged
        if uncertainty is not None:
            # Normalize uncertainty to [0, 1] range using robust scaling
            # Use 75th percentile as reference to avoid outliers
            if len(uncertainty) > 0 and uncertainty.max() > 0:
                norm_uncertainty = np.clip(
                    uncertainty / np.percentile(uncertainty, 75), 0, 1
                )
                scores = scores * (1 - 0.5 * norm_uncertainty)  # Max 50% discount

        return scores

    def _classify_recommendations(
        self, recs: pd.DataFrame, scale_threshold: float, pause_threshold: float
    ) -> pd.DataFrame:
        """Classify recommendations using configured strategies."""
        # Use custom strategies if provided
        if self.config.strategies:
            for strategy in self.config.strategies:
                mask = None

                if strategy.condition == "greater_than":
                    mask = recs["opportunity_score"] > strategy.threshold
                elif strategy.condition == "less_than":
                    mask = recs["opportunity_score"] < strategy.threshold
                elif strategy.condition == "between":
                    min_val, max_val = strategy.threshold
                    mask = (recs["opportunity_score"] >= min_val) & (
                        recs["opportunity_score"] <= max_val
                    )
                elif strategy.condition == "outside":
                    min_val, max_val = strategy.threshold
                    mask = (recs["opportunity_score"] < min_val) | (
                        recs["opportunity_score"] > max_val
                    )

                if mask is not None and mask.any():
                    recs.loc[mask, "recommendation"] = strategy.name
                    recs.loc[mask, "priority"] = strategy.priority
                    recs.loc[mask, "action"] = strategy.action
        else:
            # Default classification
            recs["recommendation"] = "hold"
            recs["priority"] = "medium"
            recs["action"] = "Maintain current settings"

            recs.loc[recs["opportunity_score"] > scale_threshold, "recommendation"] = (
                "scale_up"
            )
            recs.loc[recs["opportunity_score"] > scale_threshold, "priority"] = "high"
            recs.loc[recs["opportunity_score"] > scale_threshold, "action"] = (
                "Increase budget"
            )

            recs.loc[recs["opportunity_score"] < pause_threshold, "recommendation"] = (
                "optimize_or_pause"
            )
            recs.loc[recs["opportunity_score"] < pause_threshold, "priority"] = "high"
            recs.loc[recs["opportunity_score"] < pause_threshold, "action"] = (
                "Optimize or pause"
            )

        return recs

    def _generate_evidence(
        self,
        recs: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
        spend: np.ndarray,
        scale_threshold: float,
        pause_threshold: float,
    ) -> list:
        """
        Generate evidence for recommendations.

        Args:
            recs: Recommendations dataframe
            predictions: Predicted values
            actuals: Actual values
            spend: Spend values
            scale_threshold: Scale up threshold
            pause_threshold: Pause threshold

        Returns:
            List of evidence strings
        """
        evidence_list = []

        for i in range(len(recs)):
            evidence_parts = []
            pred = predictions[i]
            actual = actuals[i]
            opp_score = recs["opportunity_score"].iloc[i]
            rec = recs["recommendation"].iloc[i]

            # Performance gap evidence
            gap = pred - actual
            gap_pct = (gap / actual * 100) if actual != 0 else 0

            if self.direction == "maximize":
                if gap > 0:
                    evidence_parts.append(
                        f"Predicted {gap_pct:.1f}% higher than actual"
                    )
                elif gap < 0:
                    evidence_parts.append(
                        f"Predicted {abs(gap_pct):.1f}% lower than actual"
                    )
            else:  # minimize
                if gap < 0:
                    evidence_parts.append(
                        f"Predicted {abs(gap_pct):.1f}% better than actual"
                    )
                elif gap > 0:
                    evidence_parts.append(f"Predicted {gap_pct:.1f}% worse than actual")

            # Spend context
            spend_val = spend[i]
            if spend_val > 0:
                evidence_parts.append(f"Spend: ${spend_val:,.2f}")

            # Opportunity/risk context
            if opp_score > scale_threshold:
                evidence_parts.append(f"High opportunity: +${opp_score:,.2f}")
            elif opp_score < pause_threshold:
                evidence_parts.append(f"High risk: -${abs(opp_score):,.2f}")
            elif opp_score > 0:
                evidence_parts.append(f"Positive opportunity: +${opp_score:,.2f}")
            else:
                evidence_parts.append(f"Negative risk: -${abs(opp_score):,.2f}")

            # Performance benchmark
            pred_percentile = (predictions < pred).sum() / len(predictions) * 100
            actual_percentile = (actuals < actual).sum() / len(actuals) * 100
            evidence_parts.append(f"Predicted in {pred_percentile:.0f}th percentile")

            evidence_list.append("; ".join(evidence_parts))

        return evidence_list

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        spend_col: str = "spend",
        impressions_col: Optional[str] = None,
        use_percentiles: Optional[bool] = None,
        include_evidence: bool = True,
        uncertainty: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Generate recommendations based on model predictions.

        Args:
            df: Input dataframe with ad data
            predictions: Model predicted metric values
            spend_col: Column name for spend data
            impressions_col: Column name for impressions (optional)
            use_percentiles: Override config percentile setting
            include_evidence: Whether to include evidence field
            uncertainty: Optional prediction uncertainty (higher = less confident)

        Returns:
            DataFrame with recommendations and scores
        """
        # Setup
        use_pct = (
            use_percentiles
            if use_percentiles is not None
            else self.config.use_percentiles
        )

        # Determine ID column (prefer adset_id for audience-level predictions)
        if "adset_id" in df.columns:
            id_col = "adset_id"
            id_values = df["adset_id"].values
        elif "ad_id" in df.columns:
            id_col = "ad_id"
            id_values = df["ad_id"].values
        else:
            id_col = "index"
            id_values = df.index

        # Create base dataframe
        recs = pd.DataFrame(
            {
                id_col: id_values,
                f"predicted_{self.config.name}": predictions,
                f"actual_{self.config.name}": df[self.config.target_column].values,
                "spend": df[spend_col].values,
            }
        )

        # Add uncertainty if provided
        if uncertainty is not None:
            recs["prediction_uncertainty"] = uncertainty

        # Calculate opportunity score (with uncertainty discount if provided)
        recs["opportunity_score"] = self._calculate_opportunity_score(
            predictions,
            df[self.config.target_column].values,
            df[spend_col].values,
            uncertainty=uncertainty,
        )

        # Determine thresholds
        if use_pct:
            scale_thresh, pause_thresh = self._calculate_thresholds_from_percentiles(
                recs["opportunity_score"].values
            )
            print(
                f"  Percentile-based thresholds: scale={scale_thresh:.2f}, pause={pause_thresh:.2f}"
            )
        else:
            scale_thresh = self.config.scale_threshold or 100
            pause_thresh = self.config.pause_threshold or -50

        # Classify recommendations
        recs = self._classify_recommendations(recs, scale_thresh, pause_thresh)

        # Add evidence if requested
        if include_evidence:
            recs["evidence"] = self._generate_evidence(
                recs,
                predictions,
                df[self.config.target_column].values,
                df[spend_col].values,
                scale_thresh,
                pause_thresh,
            )

        # Add context if available
        if impressions_col and impressions_col in df.columns:
            recs["impressions"] = df[impressions_col].values
            recs["cpc"] = recs["spend"] / recs["impressions"]

        return recs

    def get_summary(self, recs: pd.DataFrame) -> Dict:
        """Get summary statistics of recommendations."""
        return {
            "metric": self.config.name,
            "display_name": self.config.display_name,
            "total": len(recs),
            "scale_up": int((recs["recommendation"] == "scale_up").sum()),
            "optimize_or_pause": int(
                (recs["recommendation"] == "optimize_or_pause").sum()
            ),
            "hold": int((recs["recommendation"] == "hold").sum()),
            "total_opportunity": float(
                recs[recs["opportunity_score"] > 0]["opportunity_score"].sum()
            ),
            "total_risk": float(
                abs(recs[recs["opportunity_score"] < 0]["opportunity_score"]).sum()
            ),
            "avg_predicted": float(recs[f"predicted_{self.config.name}"].mean()),
            "avg_actual": float(recs[f"actual_{self.config.name}"].mean()),
        }

    def save_recommendations(self, recs: pd.DataFrame, path: Path) -> None:
        """Save recommendations to CSV."""
        recs.to_csv(path, index=False)

    def print_summary(self, recs: pd.DataFrame) -> None:
        """Print formatted summary of recommendations."""
        summary = self.get_summary(recs)
        print(f"\n{summary['display_name'].upper()} Recommendations Summary:")
        print(f"  Total: {summary['total']:,}")
        print(
            f"  Scale up: {summary['scale_up']:,} ({100*summary['scale_up']/summary['total']:.1f}%)"
        )
        print(
            f"  Optimize/pause: {summary['optimize_or_pause']:,} ({100*summary['optimize_or_pause']/summary['total']:.1f}%)"
        )
        print(
            f"  Hold: {summary['hold']:,} ({100*summary['hold']/summary['total']:.1f}%)"
        )
        print(f"  Total opportunity: ${summary['total_opportunity']:,.2f}")
        print(f"  Total risk: ${summary['total_risk']:,.2f}")
        print(f"  Avg predicted: {summary['avg_predicted']:.4f}{self.config.unit}")
        print(f"  Avg actual: {summary['avg_actual']:.4f}{self.config.unit}")


# =============================================================================
# FACTORY FUNCTIONS (User-Facing API)
# =============================================================================


def create_recommender(
    metric_name: str = "roas",
    scale_threshold: Optional[float] = None,
    pause_threshold: Optional[float] = None,
    use_percentiles: bool = False,
) -> ConfigurableRecommender:
    """
    Create a recommender with custom thresholds.

    Args:
        metric_name: Name of the metric ('roas', 'cpa', etc.)
        scale_threshold: Override scale threshold
        pause_threshold: Override pause threshold
        use_percentiles: Use percentile-based thresholds

    Returns:
        Configured recommender

    Example:
        recommender = create_recommender(metric_name='roas', scale_threshold=200)
        recs = recommender.generate_recommendations(df, predictions)
    """
    config = MetricConfig(
        name=metric_name,
        target_column=f"{metric_name}" if metric_name != "roas" else "purchase_roas",
        direction=(
            "maximize"
            if metric_name in ["roas", "ctr", "conversion_rate"]
            else "minimize"
        ),
        scale_threshold=scale_threshold or (100 if metric_name == "roas" else None),
        pause_threshold=pause_threshold or (-50 if metric_name == "roas" else None),
        use_percentiles=use_percentiles,
        opportunity_fn=(
            default_opportunity_maximize
            if metric_name in ["roas", "ctr", "conversion_rate"]
            else default_opportunity_minimize
        ),
    )

    return ConfigurableRecommender(config=config)


def create_roas_recommender(
    scale_threshold: float = 100,
    pause_threshold: float = -50,
    use_percentiles: bool = False,
) -> ConfigurableRecommender:
    """
    Create a ROAS recommender with custom thresholds.

    Args:
        scale_threshold: Opportunity score above which to recommend scaling
        pause_threshold: Opportunity score below which to recommend pausing
        use_percentiles: Use percentile-based thresholds instead of fixed

    Returns:
        Configured ROAS recommender

    Example:
        recommender = create_roas_recommender(scale_threshold=200, pause_threshold=-100)
        recs = recommender.generate_recommendations(df, predictions)
    """
    config = MetricConfig(
        name="roas",
        target_column="purchase_roas",
        direction="maximize",
        scale_threshold=scale_threshold,
        pause_threshold=pause_threshold,
        use_percentiles=use_percentiles,
        unit="",
        display_name="ROAS",
        opportunity_fn=default_opportunity_maximize,
    )
    return ConfigurableRecommender(config=config)


def create_cpa_recommender(
    scale_threshold: float = -50,
    pause_threshold: float = 50,
    use_percentiles: bool = False,
) -> ConfigurableRecommender:
    """Create a CPA (cost per action) recommender."""
    config = MetricConfig(
        name="cpa",
        target_column="cost_per_action",
        direction="minimize",
        scale_threshold=scale_threshold,  # Negative for minimize
        pause_threshold=pause_threshold,
        use_percentiles=use_percentiles,
        unit="$",
        display_name="Cost Per Action",
        opportunity_fn=default_opportunity_minimize,
    )
    return ConfigurableRecommender(config=config)


def create_cpc_recommender(
    pause_threshold: float = 0.1, use_percentiles: bool = False
) -> ConfigurableRecommender:
    """Create a CPC (cost per click) recommender."""
    config = MetricConfig(
        name="cpc",
        target_column="cost_per_click",
        direction="minimize",
        scale_threshold=-pause_threshold / 2,
        pause_threshold=pause_threshold,
        use_percentiles=use_percentiles,
        unit="$",
        display_name="Cost Per Click",
        opportunity_fn=default_opportunity_minimize,
    )
    return ConfigurableRecommender(config=config)


def create_ctr_recommender(
    scale_threshold: float = 0.01, use_percentiles: bool = False
) -> ConfigurableRecommender:
    """Create a CTR (click-through rate) recommender."""
    config = MetricConfig(
        name="ctr",
        target_column="ctr",
        direction="maximize",
        scale_threshold=scale_threshold,
        pause_threshold=-scale_threshold / 2,
        use_percentiles=use_percentiles,
        unit="%",
        display_name="CTR",
        opportunity_fn=default_opportunity_maximize,
    )
    return ConfigurableRecommender(config=config)


def create_percentile_recommender(
    metric_name: str = "roas",
    percentile_high: float = 0.75,
    percentile_low: float = 0.25,
    **kwargs,
) -> ConfigurableRecommender:
    """
    Create a recommender with percentile-based automatic thresholds.

    Automatically adjusts thresholds based on data distribution.
    Top 25% get 'optimize_or_pause', bottom 25% get 'scale_up' (for maximize).

    Args:
        metric_name: Name of the metric
        percentile_high: Percentile for scale_up threshold (0-1)
        percentile_low: Percentile for pause threshold (0-1)
        **kwargs: Additional arguments for MetricConfig

    Returns:
        Configured percentile-based recommender

    Example:
        # Automatically adjusts thresholds based on 75th/25th percentiles
        recommender = create_percentile_recommender(
            metric_name='roas',
            percentile_high=0.8,  # Top 20% get scale_up
            percentile_low=0.2     # Bottom 20% get pause
        )
        recs = recommender.generate_recommendations(df, predictions, use_percentiles=True)
    """
    config = MetricConfig(
        name=metric_name,
        target_column=f"{metric_name}" if metric_name != "roas" else "purchase_roas",
        direction="maximize",
        use_percentiles=True,
        percentile_high=percentile_high,
        percentile_low=percentile_low,
        **kwargs,
    )
    return ConfigurableRecommender(config=config)


def create_custom_recommender(
    name: str,
    target_column: str,
    direction: str,
    scale_threshold: float,
    pause_threshold: float,
    strategies: Optional[List[RecommendationStrategy]] = None,
    opportunity_fn=None,
    unit: str = "",
    display_name: Optional[str] = None,
) -> ConfigurableRecommender:
    """
    Create a fully custom recommender.

    Args:
        name: Metric name (used in column names)
        target_column: Data column containing actual values
        direction: 'maximize' or 'minimize'
        scale_threshold: Threshold for scale_up recommendations
        pause_threshold: Threshold for pause recommendations
        strategies: Custom recommendation strategies (optional)
        opportunity_fn: Custom opportunity score function (optional)
        unit: Display unit ($, %, etc.)
        display_name: Pretty name for display

    Returns:
        Configured custom recommender

    Example:
        strategies = [
            RecommendationStrategy('aggressive', 'greater_than', 1000),
            RecommendationStrategy('moderate', 'between', (100, 1000)),
            RecommendationStrategy('conservative', 'less_than', 100),
        ]
        recommender = create_custom_recommender(
            name='my_metric',
            target_column='my_column',
            direction='maximize',
            scale_threshold=1000,
            pause_threshold=-100,
            strategies=strategies
        )
    """
    config = MetricConfig(
        name=name,
        target_column=target_column,
        direction=direction,
        scale_threshold=scale_threshold,
        pause_threshold=pause_threshold,
        strategies=strategies or [],
        opportunity_fn=opportunity_fn,
        unit=unit,
        display_name=display_name,
    )
    return ConfigurableRecommender(config=config)


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

# Old class names (deprecated, but kept for compatibility)
ROASRecommender = ConfigurableRecommender
MetricRecommender = ConfigurableRecommender

__all__ = [
    # Classes
    "ConfigurableRecommender",
    "ROASRecommender",
    "MetricRecommender",
    "MetricConfig",
    "RecommendationStrategy",
    # Factory functions
    "create_recommender",
    "create_roas_recommender",
    "create_cpa_recommender",
    "create_cpc_recommender",
    "create_ctr_recommender",
    "create_percentile_recommender",
    "create_custom_recommender",
    # Utility functions
    "default_opportunity_maximize",
    "default_opportunity_minimize",
    "roi_opportunity",
]
