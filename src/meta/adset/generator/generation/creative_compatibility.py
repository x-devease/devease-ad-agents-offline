"""
Creative compatibility recommendations.

Generates creative recommendations based on audience characteristics.
Output metadata only - NOT used as model input features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from pathlib import Path


class CreativeCompatibility:
    """
    Generate creative recommendations based on audience features.

    These are OUTPUT metadata, not model input features.
    Maintains clean separation: audience scorer ≠ creative scorer.
    """

    def __init__(self):
        """Initialize creative compatibility engine."""
        self.audience_creative_performance = self._load_baseline_performance()

    def _load_baseline_performance(self) -> Dict:
        """
        Load baseline creative performance by audience type.

        Returns:
            Dictionary mapping audience types to creative performance
        """
        return {
            "prospecting_cold": {
                "storytelling": {"roas": 1.2, "ctr": 0.8},
                "product_benefits": {"roas": 1.5, "ctr": 1.0},
                "social_proof": {"roas": 1.3, "ctr": 0.9},
                "urgency": {"roas": 1.1, "ctr": 0.7},
            },
            "retargeting_warm": {
                "product_reminder": {"roas": 2.5, "ctr": 1.5},
                "offer_promotion": {"roas": 2.8, "ctr": 1.8},
                "social_proof": {"roas": 2.2, "ctr": 1.3},
                "cross_sell": {"roas": 2.0, "ctr": 1.2},
            },
            "lookalike": {
                "brand_story": {"roas": 1.8, "ctr": 1.1},
                "social_proof": {"roas": 2.0, "ctr": 1.2},
                "product_demo": {"roas": 1.7, "ctr": 1.0},
                "user_generated": {"roas": 1.9, "ctr": 1.1},
            },
        }

    def detect_audience_type(self, audience_features: Dict) -> str:
        """
        Detect audience type from features.

        Args:
            audience_features: Dictionary of audience features

        Returns:
            Audience type label
        """
        # Check for lookalike
        if audience_features.get("is_lookalike", False):
            return "lookalike"

        # Check for retargeting
        if audience_features.get("is_retargeting", False):
            return "retargeting_warm"

        # Default to prospecting
        return "prospecting_cold"

    def recommend_creatives(
        self, audience_features: Dict, top_k: int = 3
    ) -> List[Tuple[str, Dict]]:
        """
        Recommend creative themes for audience.

        Args:
            audience_features: Dictionary of audience features
            top_k: Number of top recommendations to return

        Returns:
            List of (creative_theme, performance_metrics) tuples, sorted by ROAS
        """
        audience_type = self.detect_audience_type(audience_features)
        creative_perf = self.audience_creative_performance.get(audience_type, {})

        # Sort by ROAS
        recommendations = sorted(
            creative_perf.items(), key=lambda x: x[1]["roas"], reverse=True
        )

        return recommendations[:top_k]

    def generate_creative_metadata(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Generate creative recommendations as metadata columns.

        Args:
            df: Feature DataFrame
            feature_columns: List of feature column names

        Returns:
            DataFrame with added creative recommendation columns
        """
        df = df.copy()

        # Initialize recommendation columns
        df["creative_rec_1"] = None
        df["creative_rec_2"] = None
        df["creative_rec_3"] = None
        df["audience_type"] = None

        for idx, row in df.iterrows():
            audience_features = row[feature_columns].to_dict()
            audience_type = self.detect_audience_type(audience_features)
            recommendations = self.recommend_creatives(audience_features)

            df.at[idx, "audience_type"] = audience_type

            if len(recommendations) > 0:
                df.at[idx, "creative_rec_1"] = recommendations[0][0]
            if len(recommendations) > 1:
                df.at[idx, "creative_rec_2"] = recommendations[1][0]
            if len(recommendations) > 2:
                df.at[idx, "creative_rec_3"] = recommendations[2][0]

        return df

    def save_baseline_performance(self, path: Path) -> None:
        """Save baseline performance data."""
        with open(path, "w") as f:
            json.dump(self.audience_creative_performance, f, indent=2)

    def load_baseline_performance(self, path: Path) -> None:
        """Load baseline performance data."""
        with open(path, "r") as f:
            self.audience_creative_performance = json.load(f)
