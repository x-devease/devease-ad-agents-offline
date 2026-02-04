"""
Data loading utilities for diagnoser evaluation scripts.

This module provides unified data loading and preprocessing functions
to eliminate code duplication across evaluation scripts.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_moprobo_data(
    customer: str = "moprobo",
    platform: str = "meta",
    data_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load and preprocess moprobo daily data.

    This function loads daily ad insights data and applies standard preprocessing:
    - Converts numeric columns to proper types
    - Extracts purchase_roas from JSON format
    - Converts date_start to datetime
    - Sorts by date

    Args:
        customer: Customer name (default: "moprobo")
        platform: Platform name (default: "meta")
        data_root: Root directory for data (default: datasets/{customer}/{platform}/raw/)

    Returns:
        Preprocessed daily data DataFrame

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if data_root is None:
        data_root = Path(f"datasets/{customer}/{platform}/raw")

    daily_path = data_root / "ad_daily_insights_2024-12-17_2025-12-17.csv"

    if not daily_path.exists():
        raise FileNotFoundError(f"Data file not found: {daily_path}")

    logger.info(f"Loading daily data from: {daily_path}")
    ad_daily = pd.read_csv(daily_path)

    # Preprocess
    ad_daily = preprocess_daily_data(ad_daily)

    logger.info(f"Loaded {len(ad_daily)} rows")
    return ad_daily


def preprocess_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess daily ad insights data.

    Performs the following transformations:
    - Converts numeric columns (spend, impressions, reach, clicks) to numeric types
    - Extracts purchase_roas from JSON format
    - Converts date_start to datetime
    - Sorts by date and removes invalid dates

    Args:
        df: Raw daily data DataFrame

    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()

    # Convert numeric columns
    numeric_cols = ['spend', 'impressions', 'reach', 'clicks']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Extract purchase_roas from JSON
    if 'purchase_roas' in df.columns:
        df['purchase_roas'] = df['purchase_roas'].apply(_extract_roas_value)

    # Convert and sort by date
    if 'date_start' in df.columns:
        df['date'] = pd.to_datetime(df['date_start'], errors='coerce')
        df = df.sort_values('date').dropna(subset=['date'])

    return df


def preprocess_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess hourly ad insights data.

    Performs the same transformations as preprocess_daily_data but for hourly data.

    Args:
        df: Raw hourly data DataFrame

    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()

    # Convert numeric columns
    numeric_cols = ['spend', 'impressions', 'reach', 'clicks']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Extract purchase_roas from JSON
    if 'purchase_roas' in df.columns:
        df['purchase_roas'] = df['purchase_roas'].apply(_extract_roas_value)

    # Convert and sort by date
    if 'date_start' in df.columns:
        df['date'] = pd.to_datetime(df['date_start'], errors='coerce')
        df = df.sort_values('date').dropna(subset=['date'])

    return df


def _extract_roas_value(roas_str: str) -> float:
    """
    Extract ROAS value from JSON string.

    Args:
        roas_str: JSON string containing ROAS data

    Returns:
        Extracted ROAS value (0.0 if parsing fails)
    """
    import json

    if pd.isna(roas_str) or roas_str == '':
        return 0.0

    try:
        data = json.loads(roas_str)
        if isinstance(data, list) and len(data) > 0:
            return float(data[0].get('value', 0))
        return 0.0
    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
        logger.debug(f"Failed to parse ROAS JSON: {e}")
        return 0.0
