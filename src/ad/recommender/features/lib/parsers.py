"""Parsing utilities for filenames and ROAS values.

This module provides functions for parsing creative information from
filenames and ROAS values from various formats, as well as batch
extraction from DataFrames.
"""

import json
import logging
import re
from typing import Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# pylint: disable=invalid-name
def parse_creative_info_from_filename(
    filename: str,
) -> Dict[str, Optional[str]]:
    """
    Parse creative_id and image_hash from ad image filename.

    Expected formats:
    - ad_<ad_id>_creative_<creative_id>_<image_hash>.png
    - ad_<ad_id>_creative_<creative_id>_<image_hash>_<n>.jpg
      (duplicate with suffix)
    - <random_id>_<fb_id>_<timestamp>_n.png (social media style)

    Args:
        filename: The image filename to parse

    Returns:
        Dict with 'ad_id', 'creative_id', 'image_hash', 'filename_type'
        (all may be None)
    """
    result = {
        "ad_id": None,
        "creative_id": None,
        "image_hash": None,
        "filename_type": "unknown",
    }

    # Pattern 1: ad_<ad_id>_creative_<creative_id>_<image_hash>.ext
    # Allow any alphanumeric characters for image_hash (not just hex)
    ad_pattern = (
        r"ad_(\d+)_creative_(\d+)_([a-zA-Z0-9]+)(?:_\d+)?"
        r"\.(?:jpg|jpeg|png|gif|webp)"
    )
    match = re.match(ad_pattern, filename, re.IGNORECASE)
    if match:
        result["ad_id"] = match.group(1)
        result["creative_id"] = match.group(2)
        result["image_hash"] = match.group(3)
        result["filename_type"] = "ad_creative"
        return result

    # Pattern 2: Social media style (e.g., 565355582_793603996633801_...)
    social_pattern = r"(\d+)_(\d+)_\d+_n\.(?:jpg|jpeg|png|gif|webp)"
    match = re.match(social_pattern, filename, re.IGNORECASE)
    if match:
        result["filename_type"] = "social_media"
        # These may not directly map to ad data
        return result

    return result


# pylint: disable=too-many-branches,too-many-nested-blocks
def parse_roas_value(roas_str: Union[str, int, float, None]) -> Optional[float]:
    """
    Parse ROAS value from various formats.

    ROAS can be:
    - A direct float/int
    - A JSON array like [{"action_type": "omni_purchase", "value": "5.52"}]
    - Empty/null

    Args:
        roas_str: The ROAS value in various formats

    Returns:
        Parsed ROAS as float, or None if cannot be parsed
    """
    if pd.isna(roas_str) or roas_str == "" or roas_str is None:
        return None

    # If already a number
    if isinstance(roas_str, (int, float)):
        return float(roas_str)

    # If string that looks like a number
    try:
        return float(roas_str)
    except (ValueError, TypeError):
        pass

    # If JSON array (may be string with escaped quotes from CSV)
    if isinstance(roas_str, str):
        # Remove surrounding quotes if present
        roas_str_clean = roas_str.strip()
        if roas_str_clean.startswith('"') and roas_str_clean.endswith('"'):
            roas_str_clean = roas_str_clean[1:-1]
        if roas_str_clean.startswith("'") and roas_str_clean.endswith("'"):
            roas_str_clean = roas_str_clean[1:-1]

        if roas_str_clean.startswith("["):
            try:
                data = json.loads(roas_str_clean)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            # Look for purchase ROAS
                            action_type = item.get("action_type", "")
                            if "purchase" in action_type.lower():
                                value = item.get("value")
                                if value:
                                    return float(value)
            except (json.JSONDecodeError, ValueError):
                pass

    return None


# pylint: disable=invalid-name
def parse_creative_info_from_filenames(
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Parse creative info from filenames in features dataframe.

    This function processes each filename in the features dataframe and
    parses creative metadata such as ad_id, creative_id, and image_hash.

    Args:
        features_df: DataFrame containing image features with 'filename'
            column

    Returns:
        DataFrame with parsed creative information
        (ad_id, creative_id, image_hash, etc.)

    Raises:
        ValueError: If 'filename' column is missing from the dataframe
    """
    if "filename" not in features_df.columns:
        raise ValueError("DataFrame must contain 'filename' column")

    logger.info("Parsing creative info from filenames...")

    # Use vectorized approach for better performance
    filenames = features_df["filename"].astype(str).fillna("")

    # Apply parsing function to all filenames
    info_list = [
        parse_creative_info_from_filename(fname) for fname in filenames
    ]

    # Convert to DataFrame
    info_df = pd.DataFrame(info_list)

    # Handle empty DataFrame
    if len(info_df) == 0:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "ad_id", "creative_id", "image_hash", "filename_type",
            "original_filename", "feature_index"
        ])

    # Add original filename and index
    info_df["original_filename"] = filenames.values
    info_df["feature_index"] = features_df.index.values

    # Log statistics
    ad_creative_count = (info_df["filename_type"] == "ad_creative").sum()
    social_count = (info_df["filename_type"] == "social_media").sum()
    unknown_count = (info_df["filename_type"] == "unknown").sum()

    logger.info(
        "Filename types: ad_creative=%d, social_media=%d, unknown=%d",
        ad_creative_count,
        social_count,
        unknown_count,
    )

    return info_df
