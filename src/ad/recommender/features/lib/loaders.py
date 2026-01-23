"""Data loading utilities for ad performance and feature data.

This module provides functions for loading and preprocessing CSV data files.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_ad_data(ad_data_path: str) -> pd.DataFrame:
    """
    Load and preprocess ad performance data.

    Args:
        ad_data_path: Path to the ad performance data CSV file

    Returns:
        DataFrame with parsed ROAS and standardized column types

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is empty or cannot be read
    """
    # Import here to avoid circular dependency
    # pylint: disable=import-outside-toplevel
    from .parsers import parse_roas_value

    path = Path(ad_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Ad data file not found: {ad_data_path}")

    logger.info("Loading ad data from: %s", ad_data_path)

    try:
        df = pd.read_csv(ad_data_path)
    except Exception as error:
        raise ValueError(
            f"Failed to read CSV file {ad_data_path}: {error}"
        ) from error

    if df.empty:
        raise ValueError(f"Ad data file is empty: {ad_data_path}")
    logger.info("Ad data shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))

    # Parse ROAS
    if "purchase_roas" in df.columns:
        df["roas_parsed"] = df["purchase_roas"].apply(parse_roas_value)
        roas_count = df["roas_parsed"].notna().sum()
        logger.info(
            "ROAS parsed: %d / %d rows have ROAS", roas_count, len(df)
        )
        roas_min = df["roas_parsed"].min()
        roas_max = df["roas_parsed"].max()
        logger.info("ROAS range: %.4f - %.4f", roas_min, roas_max)

    # Ensure creative_id is string
    if "creative_id" in df.columns:
        df["creative_id"] = df["creative_id"].astype(str)

    # Ensure creative_image_hash is string
    if "creative_image_hash" in df.columns:
        df["creative_image_hash"] = (
            df["creative_image_hash"].fillna("").astype(str)
        )

    return df


def load_feature_data(features_csv: str) -> pd.DataFrame:
    """
    Load image features data.

    Args:
        features_csv: Path to the features CSV file

    Returns:
        DataFrame containing image features

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is empty or cannot be read
    """
    path = Path(features_csv)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {features_csv}")

    logger.info("Loading features from: %s", features_csv)

    try:
        df = pd.read_csv(features_csv)
    except Exception as error:
        raise ValueError(
            f"Failed to read CSV file {features_csv}: {error}"
        ) from error

    if df.empty:
        raise ValueError(f"Features file is empty: {features_csv}")
    logger.info("Features shape: %s", df.shape)
    logger.info("Features columns: %s...", list(df.columns[:20]))

    return df
