#!/usr/bin/env python
"""
Parse Moprobo Data - Preprocess raw Meta insights for diagnoser.

This script:
1. Parses JSON fields (actions, action_values)
2. Extracts conversions and revenue
3. Calculates cumulative frequency for fatigue detection
4. Infers status changes for latency detection
5. Outputs processed data ready for diagnoser
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.meta.diagnoser.detectors.latency_detector import infer_status_changes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_json_actions(actions_str: str) -> dict:
    """
    Parse actions JSON string and extract conversion types.

    Args:
        actions_str: JSON string like "[{'action_type': 'purchase', 'value': '12'}]"

    Returns:
        Dict with action_type -> value mappings
    """
    if pd.isna(actions_str) or actions_str == "":
        return {}

    try:
        # Clean JSON: replace single quotes with double quotes
        cleaned = actions_str.replace("'", '"')
        actions = json.loads(cleaned)

        result = {}
        for action in actions:
            action_type = action.get("action_type", "")
            value = action.get("value", 0)

            # Convert to float if numeric
            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0

            result[action_type] = value

        return result
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse actions: {e}")
        return {}


def parse_json_action_values(action_values_str: str) -> dict:
    """
    Parse action_values JSON string and extract revenue.

    Args:
        action_values_str: JSON string with revenue values

    Returns:
        Dict with action_type -> value mappings
    """
    if pd.isna(action_values_str) or action_values_str == "":
        return {}

    try:
        cleaned = action_values_str.replace("'", '"')
        values = json.loads(cleaned)

        result = {}
        for value_obj in values:
            action_type = value_obj.get("action_type", "")
            value = value_obj.get("value", "0")

            try:
                value = float(value)
            except (ValueError, TypeError):
                value = 0

            result[action_type] = value

        return result
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse action_values: {e}")
        return {}


def extract_purchases(actions: dict) -> float:
    """Extract purchase conversions from actions dict."""
    if not actions:
        return 0.0

    # Check common purchase action types
    purchase_keys = [
        "offsite_conversion.fb_pixel_purchase",
        "omni_purchase",
        "purchase",
        "onsite_web_purchase",
        "web_app_in_store_purchase",
    ]

    total = 0.0
    for key in purchase_keys:
        total += actions.get(key, 0)

    return total


def extract_revenue(action_values: dict) -> float:
    """Extract revenue from action_values dict."""
    if not action_values:
        return 0.0

    # Check common revenue action types
    revenue_keys = [
        "offsite_conversion.fb_pixel_purchase",
        "omni_purchase",
        "purchase",
        "onsite_web_purchase",
        "web_app_in_store_purchase",
    ]

    total = 0.0
    for key in revenue_keys:
        total += action_values.get(key, 0)

    return total


def process_ad_daily_data(
    input_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Process ad_daily insights for fatigue detection.

    Adds columns:
    - conversions: purchase conversions
    - revenue: purchase revenue
    - cum_impressions: cumulative impressions
    - cum_reach: cumulative reach
    - cum_freq: cumulative frequency
    """
    logger.info(f"Processing ad_daily data from {input_path}")

    df = pd.read_csv(input_path)
    original_rows = len(df)

    # Parse JSON fields
    logger.info("Parsing actions JSON...")
    df["actions_parsed"] = df["actions"].apply(parse_json_actions)
    df["conversions"] = df["actions_parsed"].apply(extract_purchases)

    logger.info("Parsing action_values JSON...")
    df["action_values_parsed"] = df["action_values"].apply(parse_json_action_values)
    df["revenue"] = df["action_values_parsed"].apply(extract_revenue)

    # Handle missing data
    df["conversions"] = df["conversions"].fillna(0)
    df["revenue"] = df["revenue"].fillna(0)
    df["impressions"] = df["impressions"].fillna(0)
    df["reach"] = df["reach"].fillna(0)

    # Calculate cumulative frequency per ad
    logger.info("Calculating cumulative frequency...")
    df = df.sort_values(["ad_id", "date_start"])

    df["cum_impressions"] = df.groupby("ad_id")["impressions"].cumsum()
    df["cum_reach"] = df.groupby("ad_id")["reach"].cumsum()

    # Handle division by zero
    df["cum_freq"] = df["cum_impressions"] / df["cum_reach"].replace(0, np.nan)

    # Clean up
    df = df.drop(columns=["actions_parsed", "action_values_parsed"])

    # Parse roas if it's a string
    if df["purchase_roas"].dtype == "object":
        df["purchase_roas"] = pd.to_numeric(df["purchase_roas"], errors="coerce").fillna(0)

    logger.info(f"Processed {len(df)} rows (original: {original_rows})")
    logger.info(f"Conversions extracted: {df['conversions'].sum():.0f}")
    logger.info(f"Revenue extracted: ${df['revenue'].sum():.2f}")

    return df


def process_adset_hourly_data(
    input_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Process adset_hourly insights for latency and dark hours detection.

    Adds columns:
    - conversions: purchase conversions
    - revenue: purchase revenue
    - datetime: combined date_start + hour
    """
    logger.info(f"Processing adset_hourly data from {input_path}")

    df = pd.read_csv(input_path)
    original_rows = len(df)

    # Parse JSON fields
    logger.info("Parsing actions JSON...")
    df["actions_parsed"] = df["actions"].apply(parse_json_actions)
    df["conversions"] = df["actions_parsed"].apply(extract_purchases)

    logger.info("Parsing action_values JSON...")
    if "action_values" in df.columns:
        df["action_values_parsed"] = df["action_values"].apply(parse_json_action_values)
        df["revenue"] = df["action_values_parsed"].apply(extract_revenue)
    else:
        df["revenue"] = 0.0

    # Handle missing data
    df["conversions"] = df["conversions"].fillna(0)
    df["revenue"] = df["revenue"].fillna(0)
    df["spend"] = df["spend"].fillna(0)

    # Create datetime column
    df["datetime"] = pd.to_datetime(df["date_start"]) + pd.to_timedelta(
        df["hour"].astype(str) + ":00:00"
    )

    # Parse roas
    if df["purchase_roas"].dtype == "object":
        df["purchase_roas"] = pd.to_numeric(df["purchase_roas"], errors="coerce").fillna(0)

    # Clean up
    df = df.drop(columns=["actions_parsed", "action_values_parsed"], errors="ignore")

    logger.info(f"Processed {len(df)} rows (original: {original_rows})")

    return df


def infer_status_changes_from_daily(
    input_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Infer status change events from adset_daily data.

    Outputs:
    - adset_id
    - change_date
    - old_status
    - new_status
    """
    logger.info(f"Inferring status changes from {input_path}")

    df = pd.read_csv(input_path)

    # Use the infer_status_changes function
    changes = infer_status_changes(df)

    logger.info(f"Found {len(changes)} status change events")

    return changes


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Moprobo Meta insights data for diagnoser"
    )
    parser.add_argument(
        "--customer",
        type=str,
        default="moprobo",
        help="Customer name (default: moprobo)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="meta",
        help="Platform name (default: meta)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory (default: datasets/{customer}/{platform}/raw/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: datasets/{customer}/{platform}/processed/)"
    )

    args = parser.parse_args()

    # Set paths
    if args.input_dir is None:
        input_dir = Path(f"datasets/{args.customer}/{args.platform}/raw")
    else:
        input_dir = Path(args.input_dir)

    if args.output_dir is None:
        output_dir = Path(f"datasets/{args.customer}/{args.platform}/processed")
    else:
        output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Process ad_daily
    ad_daily_input = input_dir / "ad_daily_insights_2024-12-17_2025-12-17.csv"
    if ad_daily_input.exists():
        logger.info("=" * 60)
        ad_daily_df = process_ad_daily_data(
            str(ad_daily_input),
            str(output_dir / "ad_daily_processed.csv"),
        )
        ad_daily_df.to_csv(output_dir / "ad_daily_processed.csv", index=False)
        logger.info(f"✅ Saved: {output_dir / 'ad_daily_processed.csv'}")
    else:
        logger.warning(f"⚠️  File not found: {ad_daily_input}")

    # Process adset_hourly
    adset_hourly_input = input_dir / "adset_hourly_insights_2025-09-01_2025-12-11.csv"
    if adset_hourly_input.exists():
        logger.info("=" * 60)
        adset_hourly_df = process_adset_hourly_data(
            str(adset_hourly_input),
            str(output_dir / "adset_hourly_processed.csv"),
        )
        adset_hourly_df.to_csv(output_dir / "adset_hourly_processed.csv", index=False)
        logger.info(f"✅ Saved: {output_dir / 'adset_hourly_processed.csv'}")
    else:
        logger.warning(f"⚠️  File not found: {adset_hourly_input}")

    # Infer status changes
    adset_daily_input = input_dir / "adset_daily_insights_2024-12-17_2025-12-17.csv"
    if adset_daily_input.exists():
        logger.info("=" * 60)
        status_changes_df = infer_status_changes_from_daily(
            str(adset_daily_input),
            str(output_dir / "status_changes.csv"),
        )
        status_changes_df.to_csv(output_dir / "status_changes.csv", index=False)
        logger.info(f"✅ Saved: {output_dir / 'status_changes.csv'}")
    else:
        logger.warning(f"⚠️  File not found: {adset_daily_input}")

    logger.info("=" * 60)
    logger.info("✅ Data preprocessing complete!")
    logger.info(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
