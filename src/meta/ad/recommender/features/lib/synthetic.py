"""Synthetic data generation utilities.

This module provides functions for generating synthetic ROAS values
for testing purposes when actual ad data is not available.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.constants import FeatureLinkConstants

logger = logging.getLogger(__name__)

# Type alias for synthetic methods
# pylint: disable=invalid-name
SYNTHETIC_METHODS = Literal["random", "feature_based"]


# pylint: disable=too-many-locals
def create_synthetic_roas(
    features_df: pd.DataFrame, method: SYNTHETIC_METHODS = "random"
) -> pd.DataFrame:
    """
    Create synthetic ROAS values for testing when no ad data is available.

    This is useful for testing the pipeline when actual ROAS data cannot
    be linked.

    Args:
        features_df: DataFrame containing image features
        method: Method for generating synthetic ROAS
            ('random' or 'feature_based')

    Returns:
        DataFrame with added 'roas_parsed' column containing synthetic values

    Raises:
        ValueError: If method is not one of the supported methods
    """
    if method not in FeatureLinkConstants.SYNTHETIC_METHODS:
        methods = FeatureLinkConstants.SYNTHETIC_METHODS
        raise ValueError(
            f"Method must be one of {methods}, got '{method}'"
        )

    logger.warning("Creating SYNTHETIC ROAS values for testing purposes!")

    df = features_df.copy()
    n_samples = len(df)

    if method == "random":
        # Random ROAS between 0 and max value
        max_roas = FeatureLinkConstants.RANDOM_ROAS_MAX
        df["roas_parsed"] = np.random.uniform(0, max_roas, n_samples)
    elif method == "feature_based":
        # Generate ROAS based on some features (for testing feature importance)
        # This creates a relationship between features and ROAS
        roas_values = np.zeros(n_samples)

        # Brightness effect
        if "brightness" in df.columns:
            brightness_map = FeatureLinkConstants.BRIGHTNESS_MAP
            brightness_range = FeatureLinkConstants.BRIGHTNESS_ROAS_RANGE
            roas_values += (
                df["brightness"].map(brightness_map).fillna(1.0)
                * np.random.uniform(
                    brightness_range[0],
                    brightness_range[1],
                    n_samples
                )
            )

        # Color vibrancy effect
        if "color_vibrancy" in df.columns:
            vibrancy_map = FeatureLinkConstants.COLOR_VIBRANCY_MAP
            vibrancy_range = FeatureLinkConstants.COLOR_VIBRANCY_ROAS_RANGE
            roas_values += (
                df["color_vibrancy"].map(vibrancy_map).fillna(1.0)
                * np.random.uniform(
                    vibrancy_range[0],
                    vibrancy_range[1],
                    n_samples
                )
            )

        # Human presence effect
        if "human_presence" in df.columns:
            human_map = FeatureLinkConstants.HUMAN_PRESENCE_MAP
            human_range = FeatureLinkConstants.HUMAN_PRESENCE_ROAS_RANGE
            roas_values += (
                df["human_presence"].map(human_map).fillna(0.5)
                * np.random.uniform(
                    human_range[0],
                    human_range[1],
                    n_samples
                )
            )

        # Add noise
        roas_values += np.random.normal(
            FeatureLinkConstants.NOISE_MEAN,
            FeatureLinkConstants.NOISE_STD,
            n_samples
        )

        # Ensure positive values
        min_roas = FeatureLinkConstants.MIN_ROAS_VALUE
        roas_values = np.maximum(roas_values, min_roas)

        df["roas_parsed"] = roas_values

    roas_min = df["roas_parsed"].min()
    roas_max = df["roas_parsed"].max()
    logger.info("Synthetic ROAS range: %.4f - %.4f", roas_min, roas_max)

    return df
