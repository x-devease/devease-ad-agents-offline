"""Data merging utilities.

This module provides functions for merging feature data with ROAS data.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def merge_features_with_roas(
    features_df: pd.DataFrame, ad_df: pd.DataFrame, info_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge feature data with ROAS data.

    Args:
        features_df: DataFrame containing image features
        ad_df: DataFrame containing ad performance data with ROAS
        info_df: DataFrame containing extracted creative info from filenames

    Returns:
        Merged DataFrame with features and ROAS data

    Raises:
        ValueError: If dataframes have incompatible shapes or missing
            required columns
    """
    if len(features_df) != len(info_df):
        raise ValueError(
            f"features_df and info_df must have same length. "
            f"Got {len(features_df)} and {len(info_df)}"
        )

    # Validate required columns in info_df
    required_info_cols = ["creative_id", "image_hash", "filename_type"]
    missing_cols = [
        col for col in required_info_cols if col not in info_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"info_df missing required columns: {missing_cols}"
        )

    logger.info("Merging features with ROAS data...")

    # Add extracted info to features
    features_with_info = features_df.copy()
    features_with_info["creative_id"] = info_df["creative_id"].values
    features_with_info["image_hash"] = info_df["image_hash"].values
    features_with_info["filename_type"] = info_df["filename_type"].values

    # Aggregate ad data by creative_id (take mean of metrics)
    if "creative_id" in ad_df.columns and "roas_parsed" in ad_df.columns:
        # Group by creative_id and aggregate ROAS
        roas_by_creative = (
            ad_df.groupby("creative_id")
            .agg(
                {
                    "roas_parsed": ["mean", "median", "std", "count"],
                    "spend": "sum",
                    "impressions": "sum",
                    "clicks": "sum",
                }
            )
            .reset_index()
        )

        # Flatten column names
        roas_by_creative.columns = [
            "creative_id",
            "roas_mean",
            "roas_median",
            "roas_std",
            "roas_count",
            "total_spend",
            "total_impressions",
            "total_clicks",
        ]

        logger.info(
            "Aggregated ROAS for %d unique creatives",
            len(roas_by_creative),
        )

        # Merge with features
        merged_df = features_with_info.merge(
            roas_by_creative, on="creative_id", how="left"
        )

        # Use mean ROAS as primary target
        merged_df["roas_parsed"] = merged_df["roas_mean"]

        matched_count = merged_df["roas_parsed"].notna().sum()
        total_count = len(merged_df)
        logger.info(
            "Matched %d / %d features with ROAS data",
            matched_count,
            total_count,
        )
    else:
        logger.warning(
            "Cannot merge - missing creative_id or roas_parsed columns"
        )
        merged_df = features_with_info
        merged_df["roas_parsed"] = np.nan

    return merged_df
