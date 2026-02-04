"""
Stage 2: The Synthesizer - V1.8 Combinatorial Synthesizer

Discovers "golden combinations" using co-occurrence analysis instead of random feature concatenation.
Uses co-occurrence matrix to find feature pairs with high conditional probability.
"""
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class CombinatorialSynthesizer:
    """
    V1.8 Combinatorial Synthesizer.

    Discovers "golden combinations" instead of random feature concatenation.
    Uses co-occurrence matrix to find feature pairs with high conditional probability.
    """

    def __init__(self, min_confidence: float = 0.8):
        """
        Initialize synthesizer.

        Args:
            min_confidence: Minimum co-occurrence confidence for locked combinations
        """
        self.min_confidence = min_confidence

    def build_co_occurrence_matrix(
        self,
        winners_df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> Dict:
        """
        Build co-occurrence matrix for feature pairs.

        Args:
            winners_df: Winner creatives DataFrame
            feature_pairs: List of (feature1, feature2) tuples to analyze

        Returns:
            Dict mapping (feature1, feature2, value1, value2) → confidence_score
        """
        co_occurrence = {}

        for feature1, feature2 in feature_pairs:
            if feature1 not in winners_df.columns or feature2 not in winners_df.columns:
                logger.debug(f"Skipping {feature1}, {feature2} - not in DataFrame")
                continue

            # Get unique values for each feature
            values1 = winners_df[feature1].dropna().unique()
            values2 = winners_df[feature2].dropna().unique()

            # Calculate co-occurrence for each pair of values
            for v1 in values1:
                for v2 in values2:
                    # P(feature2=v2 | feature1=v1)
                    with_v1 = winners_df[winners_df[feature1] == v1]
                    with_both = with_v1[with_v1[feature2] == v2]

                    if len(with_v1) > 0:
                        confidence = len(with_both) / len(with_v1)
                        co_occurrence[(feature1, feature2, v1, v2)] = {
                            "confidence": confidence,
                            "count": len(with_both),
                            "total": len(with_v1)
                        }

        logger.info(f"Built co-occurrence matrix with {len(co_occurrence)} entries")
        return co_occurrence

    def find_locked_combinations(
        self,
        co_occurrence: Dict,
        primary_feature: str = "surface_material"
    ) -> Dict:
        """
        Find locked combinations with highest confidence.

        Args:
            co_occurrence: Co-occurrence matrix from build_co_occurrence_matrix
            primary_feature: Primary feature to anchor combinations

        Returns:
            Dict with locked combination and confidence
        """
        locked = {
            "locked_combination": {},
            "excluded_features": [],
            "confidence_score": 0.0,
            "reasoning": ""
        }

        # Find highest confidence pair involving primary feature
        best_pair = None
        best_confidence = 0.0

        for key, stats in co_occurrence.items():
            f1, f2, v1, v2 = key

            if f1 == primary_feature or f2 == primary_feature:
                if stats["confidence"] > best_confidence:
                    best_confidence = stats["confidence"]
                    best_pair = (f1, f2, v1, v2)

        if best_pair and best_confidence >= self.min_confidence:
            f1, f2, v1, v2 = best_pair

            # Determine primary and secondary
            if f1 == primary_feature:
                locked["locked_combination"] = {
                    "primary_feature": f1,
                    "primary_value": v1,
                    "secondary_feature": f2,
                    "secondary_value": v2
                }
            else:
                locked["locked_combination"] = {
                    "primary_feature": f2,
                    "primary_value": v2,
                    "secondary_feature": f1,
                    "secondary_value": v1
                }

            locked["confidence_score"] = best_confidence

            # Find conflicting features (low co-occurrence with primary)
            conflicting = []
            primary_value = locked["locked_combination"]["primary_value"]

            for key, stats in co_occurrence.items():
                cf1, cf2, cv1, cv2 = key

                # Check if conflicts with primary value
                if (cf1 == primary_feature and cv1 == primary_value and
                    stats["confidence"] < 0.2):
                    conflicting.append({
                        "feature": cf2,
                        "value": cv2,
                        "confidence": stats["confidence"]
                    })

            locked["excluded_features"] = conflicting

            # Generate reasoning
            primary_feature_name = locked["locked_combination"]["primary_feature"]
            primary_value_name = locked["locked_combination"]["primary_value"]
            secondary_feature_name = locked["locked_combination"]["secondary_feature"]
            secondary_value_name = locked["locked_combination"]["secondary_value"]

            locked["reasoning"] = (
                f"{best_confidence*100:.0f}% of {primary_value_name} winners use "
                f"{secondary_value_name}. "
                f"{len(conflicting)} conflicting features excluded."
            )

            logger.info(
                f"Locked combination: {primary_feature_name}={primary_value_name} + "
                f"{secondary_feature_name}={secondary_value_name} "
                f"(confidence: {best_confidence:.2f})"
            )

        return locked

    def extract_individual_features(
        self,
        winners_df: pd.DataFrame,
        losers_df: pd.DataFrame,
        roas_col: str = "roas",
        min_confidence: float = 0.70,
        min_roas_lift: float = 1.3
    ) -> List[Dict]:
        """
        Extract individual feature performance for ALL features.

        Analyzes each feature value to calculate standalone ROAS lift, prevalence,
        and confidence. This replaces hardcoded individual features with
        data-driven extraction.

        Args:
            winners_df: Winner creatives DataFrame
            losers_df: Loser creatives DataFrame
            roas_col: Column name for ROAS metric
            min_confidence: Minimum confidence threshold (0-1)
            min_roas_lift: Minimum ROAS lift threshold

        Returns:
            List of individual feature patterns sorted by ROAS lift (descending)
        """
        individual_features = []

        # Get all feature columns (exclude metadata columns)
        exclude_cols = {
            'ad_id', 'ad_name', 'adset_id', 'campaign_id', 'account_id',
            'date_start', 'date_stop', 'export_date', 'filename', 'roas',
            'purchase_roas', 'website_purchase_roas', 'mobile_app_purchase_roas',
            'spend', 'impressions', 'clicks', 'reach', 'frequency'
        }

        feature_cols = [col for col in winners_df.columns
                       if col not in exclude_cols
                       and winners_df[col].dtype == 'object']

        logger.info(f"Analyzing {len(feature_cols)} feature columns for individual performance")

        total_winners = len(winners_df)
        total_losers = len(losers_df)
        avg_winner_roas = winners_df[roas_col].mean()
        avg_loser_roas = losers_df[roas_col].mean()

        for feature in feature_cols:
            # Get unique values (categorical)
            unique_values = winners_df[feature].dropna().unique()

            if len(unique_values) == 0:
                continue

            for value in unique_values:
                # Calculate prevalence in winners
                winners_with_value = winners_df[winners_df[feature] == value]
                winner_prevalence = len(winners_with_value) / total_winners if total_winners > 0 else 0

                # Calculate prevalence in losers
                losers_with_value = losers_df[losers_df[feature] == value]
                loser_prevalence = len(losers_with_value) / total_losers if total_losers > 0 else 0

                # Skip if not enough samples
                if len(winners_with_value) < 10:
                    continue

                # Calculate ROAS lift for this feature value
                value_avg_roas = winners_with_value[roas_col].mean()
                roas_lift = value_avg_roas / avg_winner_roas if avg_winner_roas > 0 else 1.0

                # Calculate confidence based on prevalence difference
                prevalence_lift = winner_prevalence - loser_prevalence
                confidence = min(0.95, 0.5 + prevalence_lift)  # Scale to 0-1

                # Skip if below thresholds
                if confidence < min_confidence or roas_lift < min_roas_lift:
                    continue

                # Calculate sample count
                sample_count = len(winners_with_value)

                # Build individual feature pattern
                individual_feature = {
                    "feature": feature,
                    "value": value,
                    "pattern_type": "DO",
                    "individual_roas_lift": round(roas_lift, 2),
                    "individual_roas_pct": round((roas_lift - 1) * 100, 1),
                    "winner_prevalence": round(winner_prevalence, 3),
                    "loser_prevalence": round(loser_prevalence, 3),
                    "prevalence_lift": round(prevalence_lift, 3),
                    "confidence": round(confidence, 2),
                    "reason": (
                        f"{value} appears in {winner_prevalence*100:.1f}% of winners vs "
                        f"{loser_prevalence*100:.1f}% of losers. "
                        f"{roas_lift:.1f}x ROAS lift when used standalone."
                    ),
                    "priority_score": round(roas_lift * 5, 1),
                    "sample_count": sample_count
                }

                individual_features.append(individual_feature)

        # Sort by ROAS lift (descending)
        individual_features.sort(key=lambda x: x["individual_roas_lift"], reverse=True)

        logger.info(
            f"Extracted {len(individual_features)} individual feature patterns "
            f"(confidence >= {min_confidence}, ROAS lift >= {min_roas_lift})"
        )

        return individual_features

    def generate_locked_combinations(
        self,
        winners_df: pd.DataFrame,
        raw_tags: Dict[str, List]
    ) -> Dict:
        """
        Generate locked combinations from winners.

        Args:
            winners_df: Winner creatives DataFrame
            raw_tags: Raw tags from miner

        Returns:
            Dict with locked combinations for material, lighting, scene
        """
        # Define feature pairs to analyze
        feature_pairs = [
            # Material + Lighting pairs
            ("surface_material", "lighting_style"),
            ("surface_material", "lighting_type"),

            # Scene + Lighting pairs
            ("product_context", "lighting_style"),

            # Material + Scene pairs
            ("surface_material", "product_context"),

            # Position + Lighting pairs
            ("product_position", "lighting_type"),
        ]

        # Build co-occurrence matrix
        logger.info("Building co-occurrence matrix...")
        co_occurrence = self.build_co_occurrence_matrix(winners_df, feature_pairs)

        # Find locked combinations
        locked_combinations = {}

        # Try to find material-based combination (highest priority)
        if "surface_material" in winners_df.columns:
            logger.info("Finding material-lighting combination...")
            material_combo = self.find_locked_combinations(
                co_occurrence, primary_feature="surface_material"
            )

            if material_combo["confidence_score"] >= self.min_confidence:
                locked_combinations["material_lighting"] = material_combo

        # Try to find scene-based combination
        if "product_context" in winners_df.columns:
            logger.info("Finding scene-lighting combination...")
            scene_combo = self.find_locked_combinations(
                co_occurrence, primary_feature="product_context"
            )

            if scene_combo["confidence_score"] >= self.min_confidence:
                locked_combinations["scene_lighting"] = scene_combo

        logger.info(f"Generated {len(locked_combinations)} locked combinations")
        return locked_combinations

    def run(
        self,
        winners_df: pd.DataFrame,
        raw_tags: Dict[str, List]
    ) -> Dict:
        """
        Run complete synthesizer pipeline.

        Args:
            winners_df: Winner creatives DataFrame
            raw_tags: Raw tags from miner

        Returns:
            Dict with locked combinations
        """
        logger.info("Starting Stage 2: The Synthesizer")

        locked_combinations = self.generate_locked_combinations(winners_df, raw_tags)

        logger.info("Stage 2: The Synthesizer completed")

        return locked_combinations
