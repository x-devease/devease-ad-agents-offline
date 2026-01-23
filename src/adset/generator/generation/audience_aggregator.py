"""
Audience-level aggregation and strategy detection.

Aggregates ad-level predictions to adset (audience) level and detects
best performing targeting configurations.
"""

import pandas as pd
import numpy as np
import ast
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.utils import Config


@dataclass
class AudienceSegment:
    """Represents a segment of audiences with similar targeting."""

    segment_name: str
    audience_count: int
    avg_predicted_metric: float
    avg_actual_metric: float
    avg_spend: float
    opportunity_score: float
    confidence: float
    characteristics: Dict


class AudienceAggregator:
    """
    Aggregates ad-level recommendations to audience (adset) level.

    Features:
    - Aggregates predictions by adset_id
    - Ranks audiences by predicted performance
    - Segments audiences by targeting characteristics
    - Identifies best performing targeting setups
    """

    def __init__(self, metric_config: dict = None):
        """
        Initialize aggregator.

        Args:
            metric_config: Metric configuration dict
        """
        self.metric_config = metric_config or Config.get_metric_config()
        self.direction = self.metric_config.get("direction", "maximize")

    def aggregate_to_audience(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Aggregate ad-level predictions to audience level.

        Args:
            df: Ad-level dataframe
            predictions: Model predictions for each ad
            uncertainty: Optional prediction uncertainty for each ad

        Returns:
            Audience-level dataframe with aggregated metrics
        """
        # Create working dataframe
        ad_df = df.copy()
        ad_df["predicted_metric"] = predictions
        target_col = self.metric_config.get("target_column", "purchase_roas")
        ad_df["actual_metric"] = ad_df[target_col]

        # Add uncertainty if provided
        if uncertainty is not None:
            ad_df["prediction_uncertainty"] = uncertainty

        # Identify audience-level columns
        # Use only essential columns for groupby (avoid NaN issues)
        essential_cols = ["adset_id", "adset_name"]
        targeting_cols = [
            "adset_targeting_age_min",
            "adset_targeting_age_max",
            "adset_targeting_genders",
            "adset_targeting_countries",
            "adset_targeting_location_types",
            "adset_targeting_custom_audiences_count",
            "adset_targeting_advantage_audience",
            "adset_daily_budget",
            "adset_status",
            "campaign_id",
            "campaign_name",
        ]

        # Select available essential columns
        available_essential = [c for c in essential_cols if c in ad_df.columns]
        available_targeting = [c for c in targeting_cols if c in ad_df.columns]

        if "adset_id" not in ad_df.columns:
            raise ValueError(
                "DataFrame must contain 'adset_id' column for audience aggregation"
            )

        # Aggregate metrics - group only by essential columns
        agg_dict = {
            "predicted_metric": ["mean", "std", "min", "max", "count"],
            "actual_metric": [
                "mean",
                "std",
                lambda x: x.quantile(0.95),
            ],  # Empirical 95th percentile
            "spend": ["sum", "mean"],
        }

        # Add uncertainty aggregation if available
        if "prediction_uncertainty" in ad_df.columns:
            agg_dict["prediction_uncertainty"] = ["mean", "max"]

        # Add first() aggregation for targeting columns to capture their values
        for col in available_targeting:
            agg_dict[col] = "first"

        # Perform aggregation
        groupby_cols = ["adset_id"] if "adset_id" in ad_df.columns else []
        if "adset_name" in ad_df.columns:
            groupby_cols.append("adset_name")

        audience_df = ad_df.groupby(groupby_cols, as_index=False).agg(agg_dict)

        # Flatten column names
        audience_df.columns = [
            "_".join(col).strip("_") for col in audience_df.columns.values
        ]

        # Remove '_first' suffix from targeting column names
        audience_df.columns = [
            col.replace("_first", "") if col.endswith("_first") else col
            for col in audience_df.columns
        ]

        # Calculate opportunity score with uncertainty discount if available
        predicted = audience_df["predicted_metric_mean"].values
        actual = audience_df["actual_metric_mean"].values
        spend = audience_df["spend_sum"].values

        # Apply uncertainty discount if available
        if "prediction_uncertainty_mean" in audience_df.columns:
            uncertainty = audience_df["prediction_uncertainty_mean"].values
            # Normalize uncertainty to [0, 1] range using 75th percentile
            if uncertainty.max() > 0:
                norm_uncertainty = np.clip(
                    uncertainty / np.percentile(uncertainty, 75), 0, 1
                )
                # Reduce effective opportunity by uncertainty (max 50% reduction)
                if self.direction == "maximize":
                    opportunity = (
                        (predicted - actual) * (1 - 0.5 * norm_uncertainty) * spend
                    )
                else:
                    opportunity = (
                        (actual - predicted) * (1 - 0.5 * norm_uncertainty) * spend
                    )
            else:
                opportunity = self._calculate_audience_opportunity(
                    predicted, actual, spend
                )
        else:
            opportunity = self._calculate_audience_opportunity(predicted, actual, spend)

        audience_df["opportunity_score"] = opportunity

        # Calculate confidence based on sample size and uncertainty
        base_confidence = self._calculate_confidence(
            audience_df["predicted_metric_count"].values
        )

        # Adjust confidence downward based on uncertainty if available
        if "prediction_uncertainty_mean" in audience_df.columns:
            uncertainty = audience_df["prediction_uncertainty_mean"].values
            if uncertainty.max() > 0:
                norm_uncertainty = np.clip(
                    uncertainty / np.percentile(uncertainty, 75), 0, 1
                )
                # Reduce confidence by uncertainty (max 40% reduction)
                audience_df["confidence"] = base_confidence * (
                    1 - 0.4 * norm_uncertainty
                )
            else:
                audience_df["confidence"] = base_confidence
        else:
            audience_df["confidence"] = base_confidence

        # Add targeting characteristics summary
        audience_df["targeting_summary"] = audience_df.apply(
            self._create_targeting_summary, axis=1
        )

        # Rename for clarity
        rename_map = {
            "predicted_metric_mean": "avg_predicted",
            "predicted_metric_std": "predicted_std",
            "actual_metric_mean": "avg_actual",
            "actual_metric_std": "actual_std",
            "actual_metric_<lambda>": "actual_metric_95th",  # Rename lambda column
            "spend_sum": "total_spend",
            "spend_mean": "avg_spend",
            "predicted_metric_count": "num_ads",
        }
        audience_df = audience_df.rename(columns=rename_map)

        # Handle the lambda column name - pandas might name it differently
        # Find the column with 'lambda' in its name and rename it
        for col in audience_df.columns:
            if "lambda" in col:
                audience_df = audience_df.rename(columns={col: "actual_metric_95th"})
                break

        return audience_df

    def _calculate_audience_opportunity(
        self, predicted: np.ndarray, actual: np.ndarray, spend: np.ndarray
    ) -> np.ndarray:
        """Calculate opportunity score at audience level."""
        if self.direction == "maximize":
            return (predicted - actual) * spend
        else:  # minimize
            return (actual - predicted) * spend

    def _calculate_confidence(self, sample_sizes: np.ndarray) -> np.ndarray:
        """Calculate confidence score based on sample size."""
        # Confidence: 0-1 scale based on sample size
        # Using log scale: more samples = higher confidence up to a point
        return np.clip(np.log1p(sample_sizes) / np.log1p(100), 0.3, 1.0)

    def _create_targeting_summary(self, row: pd.Series) -> str:
        """Create human-readable targeting summary."""
        parts = []

        # Age
        age_min = row.get("adset_targeting_age_min", None)
        age_max = row.get("adset_targeting_age_max", None)
        if pd.notna(age_min) and pd.notna(age_max):
            parts.append(f"Age {int(age_min)}-{int(age_max)}")

        # Gender
        genders = row.get("adset_targeting_genders", None)
        if pd.notna(genders) and genders:
            if isinstance(genders, str):
                try:
                    gender_list = (
                        ast.literal_eval(genders)
                        if genders.startswith("[")
                        else [genders]
                    )
                    parts.append(f"Genders: {', '.join(str(g) for g in gender_list)}")
                except:
                    parts.append(f"Genders: {genders}")

        # Location
        countries = row.get("adset_targeting_countries", None)
        if pd.notna(countries) and countries:
            if isinstance(countries, str):
                try:
                    country_list = (
                        ast.literal_eval(countries)
                        if countries.startswith("[")
                        else [countries]
                    )
                    if len(country_list) <= 3:
                        parts.append(f"Locations: {', '.join(country_list)}")
                    else:
                        parts.append(
                            f"Locations: {country_list[0]}, +{len(country_list)-1} more"
                        )
                except:
                    pass

        # Custom audiences
        custom_count = row.get("adset_targeting_custom_audiences_count", 0)
        if pd.notna(custom_count) and custom_count and custom_count > 0:
            parts.append(f"{int(custom_count)} custom audience(s)")

        return "; ".join(parts) if parts else "Unknown targeting"

    def rank_audiences(
        self, audience_df: pd.DataFrame, top_k: int = 20
    ) -> pd.DataFrame:
        """
        Rank audiences by opportunity score.

        Args:
            audience_df: Audience-level dataframe
            top_k: Number of top audiences to return

        Returns:
            Ranked audience dataframe
        """
        ranked = (
            audience_df.sort_values("opportunity_score", ascending=False)
            .head(top_k)
            .copy()
        )

        ranked["rank"] = range(1, len(ranked) + 1)
        return ranked

    def detect_best_segments(
        self, audience_df: pd.DataFrame, min_audiences_per_segment: int = 3
    ) -> List[AudienceSegment]:
        """
        Detect best performing audience segments based on targeting.

        Args:
            audience_df: Audience-level dataframe
            min_audiences_per_segment: Minimum audiences to form a segment

        Returns:
            List of AudienceSegment objects sorted by opportunity
        """
        segments = []

        # Segment by age range
        segments.extend(self._segment_by_age(audience_df, min_audiences_per_segment))

        # Segment by gender targeting
        segments.extend(self._segment_by_gender(audience_df, min_audiences_per_segment))

        # Segment by location
        segments.extend(
            self._segment_by_location(audience_df, min_audiences_per_segment)
        )

        # Segment by custom audience usage
        segments.extend(
            self._segment_by_custom_audiences(audience_df, min_audiences_per_segment)
        )

        # Sort by opportunity score
        segments.sort(key=lambda x: x.opportunity_score, reverse=True)

        return segments[:10]  # Return top 10 segments

    def _segment_by_age(
        self, df: pd.DataFrame, min_count: int
    ) -> List[AudienceSegment]:
        """Segment audiences by age range."""
        segments = []

        # Create age buckets
        df_copy = df.copy()
        df_copy["age_bucket"] = df_copy.apply(
            lambda x: f"{x.get('adset_targeting_age_min', 0)}-{x.get('adset_targeting_age_max', 100)}",
            axis=1,
        )

        for age_bucket, group in df_copy.groupby("age_bucket"):
            if len(group) >= min_count:
                segments.append(
                    AudienceSegment(
                        segment_name=f"Age {age_bucket}",
                        audience_count=len(group),
                        avg_predicted_metric=float(group["avg_predicted"].mean()),
                        avg_actual_metric=float(group["avg_actual"].mean()),
                        avg_spend=float(group["total_spend"].mean()),
                        opportunity_score=float(group["opportunity_score"].sum()),
                        confidence=float(group["confidence"].mean()),
                        characteristics={"age_range": age_bucket},
                    )
                )

        return segments

    def _segment_by_gender(
        self, df: pd.DataFrame, min_count: int
    ) -> List[AudienceSegment]:
        """Segment by gender targeting."""
        segments = []

        def normalize_gender(genders_val):
            """Normalize gender field for grouping."""
            if pd.isna(genders_val):
                return "Unknown"
            if isinstance(genders_val, str):
                try:
                    gender_list = (
                        ast.literal_eval(genders_val)
                        if genders_val.startswith("[")
                        else [genders_val]
                    )
                    return ",".join(sorted(str(g) for g in gender_list))
                except:
                    return str(genders_val)
            return "Unknown"

        df_copy = df.copy()
        df_copy["gender_segment"] = df_copy["adset_targeting_genders"].apply(
            normalize_gender
        )

        for gender_seg, group in df_copy.groupby("gender_segment"):
            if len(group) >= min_count:
                segments.append(
                    AudienceSegment(
                        segment_name=f"Gender: {gender_seg}",
                        audience_count=len(group),
                        avg_predicted_metric=float(group["avg_predicted"].mean()),
                        avg_actual_metric=float(group["avg_actual"].mean()),
                        avg_spend=float(group["total_spend"].mean()),
                        opportunity_score=float(group["opportunity_score"].sum()),
                        confidence=float(group["confidence"].mean()),
                        characteristics={"genders": gender_seg},
                    )
                )

        return segments

    def _segment_by_location(
        self, df: pd.DataFrame, min_count: int
    ) -> List[AudienceSegment]:
        """Segment by location."""
        segments = []

        def get_primary_country(countries_val):
            """Get primary country from location list."""
            if pd.isna(countries_val):
                return "Unknown"
            if isinstance(countries_val, str):
                try:
                    country_list = (
                        ast.literal_eval(countries_val)
                        if countries_val.startswith("[")
                        else [countries_val]
                    )
                    return country_list[0] if country_list else "Unknown"
                except:
                    return "Unknown"
            return "Unknown"

        df_copy = df.copy()
        df_copy["primary_country"] = df_copy["adset_targeting_countries"].apply(
            get_primary_country
        )

        for country, group in df_copy.groupby("primary_country"):
            if len(group) >= min_count:
                segments.append(
                    AudienceSegment(
                        segment_name=f"Location: {country}",
                        audience_count=len(group),
                        avg_predicted_metric=float(group["avg_predicted"].mean()),
                        avg_actual_metric=float(group["avg_actual"].mean()),
                        avg_spend=float(group["total_spend"].mean()),
                        opportunity_score=float(group["opportunity_score"].sum()),
                        confidence=float(group["confidence"].mean()),
                        characteristics={"primary_country": country},
                    )
                )

        return segments

    def _segment_by_custom_audiences(
        self, df: pd.DataFrame, min_count: int
    ) -> List[AudienceSegment]:
        """Segment by custom audience usage."""
        segments = []

        df_copy = df.copy()
        df_copy["has_custom_audiences"] = (
            df_copy["adset_targeting_custom_audiences_count"] > 0
        )

        for has_custom, group in df_copy.groupby("has_custom_audiences"):
            if len(group) >= min_count:
                segment_name = (
                    "With Custom Audiences"
                    if has_custom
                    else "Without Custom Audiences"
                )
                segments.append(
                    AudienceSegment(
                        segment_name=segment_name,
                        audience_count=len(group),
                        avg_predicted_metric=float(group["avg_predicted"].mean()),
                        avg_actual_metric=float(group["avg_actual"].mean()),
                        avg_spend=float(group["total_spend"].mean()),
                        opportunity_score=float(group["opportunity_score"].sum()),
                        confidence=float(group["confidence"].mean()),
                        characteristics={"custom_audiences": bool(has_custom)},
                    )
                )

        return segments

    def analyze_headroom(self, audience_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze headroom for each audience based on historical performance.

        Calculates whether predicted improvements are achievable based on
        historical maximum performance using empirical 95th percentile from ad-level data.

        Args:
            audience_df: Audience-level dataframe with avg_predicted, avg_actual, actual_metric_95th

        Returns:
            Audience dataframe with headroom analysis columns added
        """
        df = audience_df.copy()

        # Use empirical 95th percentile if available, otherwise fall back to mean + 2*std
        if "actual_metric_95th" in df.columns:
            # Use empirical percentile calculated from ad-level data (more accurate for skewed distributions)
            df["historical_max_95th"] = df["actual_metric_95th"]
        else:
            # Fallback: Use parametric estimate (assumes normal distribution, less accurate for skewed data)
            df["historical_max_95th"] = df["avg_actual"] + 2 * df["actual_std"]

        # Check if prediction is within historical range
        df["prediction_achievable"] = df["avg_predicted"] <= df["historical_max_95th"]

        # Calculate realistic vs predicted improvement
        df["realistic_headroom"] = df["historical_max_95th"] - df["avg_actual"]
        df["predicted_improvement"] = df["avg_predicted"] - df["avg_actual"]

        # Calculate realistic opportunity (based on historical max)
        df["realistic_opportunity"] = df.apply(
            lambda row: (
                row["realistic_headroom"] * row["total_spend"]
                if row["realistic_headroom"] > 0
                else 0
            ),
            axis=1,
        )

        # Calculate how much prediction exceeds historical max
        df["prediction_excess"] = df.apply(
            lambda row: max(0, row["avg_predicted"] - row["historical_max_95th"]),
            axis=1,
        )

        # Calculate excess as percentage of historical max
        df["excess_pct"] = df.apply(
            lambda row: (
                (row["prediction_excess"] / row["historical_max_95th"] * 100)
                if row["historical_max_95th"] > 0
                else 0
            ),
            axis=1,
        )

        return df

    def generate_strategy_report(
        self, top_audiences: pd.DataFrame, best_segments: List[AudienceSegment]
    ) -> dict:
        """
        Generate comprehensive strategy report.

        Args:
            top_audiences: Top ranked audiences
            best_segments: Best performing segments

        Returns:
            Dictionary with strategy insights
        """
        # Analyze headroom before generating report
        audiences_with_headroom = self.analyze_headroom(top_audiences)

        report = {
            "metric": self.metric_config["display_name"],
            "top_audiences": {
                "count": len(top_audiences),
                "total_opportunity": (
                    float(top_audiences["opportunity_score"].sum())
                    if len(top_audiences) > 0
                    else 0.0
                ),
                "realistic_opportunity": (
                    float(audiences_with_headroom["realistic_opportunity"].sum())
                    if len(audiences_with_headroom) > 0
                    else 0.0
                ),
                "avg_confidence": (
                    float(top_audiences["confidence"].mean())
                    if len(top_audiences) > 0
                    else 0.0
                ),
                "audiences": (
                    audiences_with_headroom.head(10).to_dict("records")
                    if len(audiences_with_headroom) > 0
                    else []
                ),
            },
            "best_segments": [
                {
                    "name": seg.segment_name,
                    "audience_count": seg.audience_count,
                    "opportunity_score": float(seg.opportunity_score),
                    "avg_predicted": float(seg.avg_predicted_metric),
                    "avg_actual": float(seg.avg_actual_metric),
                    "confidence": float(seg.confidence),
                    "characteristics": seg.characteristics,
                }
                for seg in best_segments[:5]
            ],
            "headroom_summary": self._generate_headroom_summary(
                audiences_with_headroom
            ),
            "recommendations": self._generate_strategy_recommendations(
                audiences_with_headroom, best_segments
            ),
        }

        return report

    def _generate_headroom_summary(self, df: pd.DataFrame) -> dict:
        """Generate headroom analysis summary."""
        if len(df) == 0:
            return {}

        achievable_count = df["prediction_achievable"].sum()
        total_count = len(df)

        return {
            "total_audiences": int(total_count),
            "achievable_predictions": int(achievable_count),
            "over_optimistic_predictions": int(total_count - achievable_count),
            "predicted_opportunity": float(df["opportunity_score"].sum()),
            "realistic_opportunity": float(df["realistic_opportunity"].sum()),
            "opportunity_inflation_pct": float(
                (
                    (df["opportunity_score"].sum() - df["realistic_opportunity"].sum())
                    / df["realistic_opportunity"].sum()
                    * 100
                )
                if df["realistic_opportunity"].sum() > 0
                else 0
            ),
            "avg_excess_over_historical": (
                float(df["excess_pct"].mean()) if len(df) > 0 else 0.0
            ),
        }

    def _generate_strategy_recommendations(
        self, top_audiences: pd.DataFrame, best_segments: List[AudienceSegment]
    ) -> List[str]:
        """Generate actionable strategy recommendations."""
        recommendations = []

        # Analyze top audiences - use realistic opportunity if available
        if len(top_audiences) > 0:
            top = top_audiences.iloc[0]
            adset_name = top.get("adset_name", "Unknown audience")

            # Use realistic_opportunity if available, otherwise fall back to opportunity_score
            opportunity = top.get(
                "realistic_opportunity", top.get("opportunity_score", 0)
            )
            recommendations.append(
                f"Highest opportunity audience: {adset_name} "
                f"(${opportunity:,.2f} realistic potential)"
            )

        # Analyze best segments
        if best_segments:
            best_seg = best_segments[0]
            gap = best_seg.avg_predicted_metric - best_seg.avg_actual_metric
            direction = "improvement" if gap > 0 else "underperformance"
            recommendations.append(
                f"Best segment: {best_seg.segment_name} shows {direction} "
                f"of {abs(gap):.4f} {self.metric_config.get('unit', '')}"
            )

        # Check custom audiences impact - provide nuanced recommendation
        custom_segments = [
            s for s in best_segments if "custom" in s.segment_name.lower()
        ]
        if custom_segments:
            custom_with = [
                s for s in custom_segments if "with" in s.segment_name.lower()
            ]
            custom_without = [
                s for s in custom_segments if "without" in s.segment_name.lower()
            ]

            if custom_with and custom_without:
                with_opp = custom_with[0].opportunity_score
                without_opp = custom_without[0].opportunity_score
                with_conf = custom_with[0].confidence
                without_conf = custom_without[0].confidence

                # More nuanced recommendation based on multiple factors
                if with_opp > without_opp * 1.2:
                    recommendations.append(
                        f"Custom audiences show strong performance (confidence: {with_conf:.0%})"
                    )
                elif without_opp > with_opp * 1.2:
                    recommendations.append(
                        f"Broad targeting outperforms custom audiences currently - "
                        f"test custom audiences for niche segments"
                    )
                else:
                    # Similar performance - recommend testing both
                    recommendations.append(
                        f"Custom and broad targeting show similar results - "
                        f"test both for your use case"
                    )
            elif custom_with:
                recommendations.append(
                    f"Custom audiences available but limited data - "
                    f"test against broad targeting"
                )
            elif custom_without:
                recommendations.append(
                    f"Broad targeting performing well - consider testing "
                    f"custom audiences for niche segments"
                )

        # Add headroom warning if predictions are optimistic
        if "headroom_summary" in dir(self) and len(top_audiences) > 0:
            achievable = (
                top_audiences["prediction_achievable"].sum()
                if "prediction_achievable" in top_audiences.columns
                else 0
            )
            total = len(top_audiences)
            if achievable < total / 2:
                recommendations.append(
                    f"Many predictions exceed historical maximums - focus on "
                    f"audiences with achievable improvements"
                )

        return recommendations
