#!/usr/bin/env python3
"""
Feature extraction script.
Joins account, campaign, and adset data into ad-level data.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path (script is at src/meta/adset/allocator/cli/commands/extract.py)
project_root = Path(__file__).parent.parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.meta.adset.allocator.features import Loader, Joiner, Aggregator
from src.meta.adset.allocator.features.utils.constants import BASE_SUM_COLUMNS
from src.meta.adset.allocator.features.utils.revenue_utils import (
    calculate_revenue_from_purchase_actions,
)
from src.utils.customer_paths import (
    get_customer_data_dir,
    get_customer_ad_features_path,
    get_customer_adset_features_path,
    ensure_customer_dirs,
)
from src.utils.script_helpers import (
    add_customer_argument,
    add_config_argument,
    add_platform_argument,
    process_all_customers,
)

# Configure logging
from src.utils.logger_config import setup_logging

logger = setup_logging()


def _build_agg_dict(sum_cols, mean_cols, first_cols, other_cols, groupby_cols):
    """Build aggregation dictionary from column lists."""
    agg_dict = {}
    for col in sum_cols:
        agg_dict[col] = "sum"
    for col in mean_cols:
        agg_dict[col] = "mean"
    for col in first_cols:
        agg_dict[col] = "first"
    for col in other_cols:
        if col not in agg_dict:
            agg_dict[col] = "first"
    for col in groupby_cols:
        if col in agg_dict:
            del agg_dict[col]
    return agg_dict


def _handle_revenue_calculation(ad_features_df, agg_dict, sum_cols):
    """Handle revenue calculation for purchase_roas aggregation.

    Revenue should ALWAYS come from purchase action_value columns if they
    exist. Never recalculate revenue from purchase_roas * spend if purchase
    action_value columns exist (even if revenue is 0).
    """
    # Check if purchase action_value columns exist
    purchase_action_value_cols = [
        col
        for col in ad_features_df.columns
        if "action_value" in col.lower() and "purchase" in col.lower()
    ]

    # If purchase action_value columns exist, revenue MUST come from them
    # (even if the sum is 0). Never use purchase_roas * spend as fallback.
    if purchase_action_value_cols:
        # Always recalculate revenue from purchase action_value columns
        # This ensures revenue is always the sum, even if it was previously
        # calculated from purchase_roas * spend
        calculate_revenue_from_purchase_actions(
            ad_features_df, update_inplace=True, custom_logger=logger
        )
        logger.info(
            "Revenue calculated from %d purchase action_value columns "
            "(overriding any previous revenue calculation)",
            len(purchase_action_value_cols),
        )
    # Only fallback to purchase_roas * spend if no purchase action_value columns
    # AND only when purchase_roas > 0 to avoid zero ROAS
    elif "purchase_roas" in ad_features_df.columns:
        if "revenue" not in ad_features_df.columns:
            ad_features_df = ad_features_df.copy()
            purchase_roas = pd.to_numeric(
                ad_features_df["purchase_roas"], errors="coerce"
            )
            spend = ad_features_df["spend"]

            # Only calculate revenue when purchase_roas > 0 to avoid zero ROAS
            mask = (purchase_roas > 0) & spend.notna() & (spend > 0)
            ad_features_df["revenue"] = np.nan
            ad_features_df.loc[mask, "revenue"] = (
                spend[mask] * purchase_roas[mask]
            ).replace([np.inf, -np.inf], np.nan)
            logger.info(
                "Revenue calculated from purchase_roas * spend "
                "(no purchase action_value columns found). "
                "Only calculated for %d rows where purchase_roas > 0 "
                "(avoiding zero ROAS)",
                mask.sum(),
            )

    # Add revenue to sum_cols for aggregation
    if "revenue" in ad_features_df.columns and "revenue" not in sum_cols:
        agg_dict["revenue"] = "sum"
    return ad_features_df, agg_dict


def _recalculate_roas_metrics(adset_df):
    """Recalculate ROAS-related metrics."""
    _recalc_revenue_from_actions(adset_df)
    _recalculate_purchase_roas(adset_df)

    _calculate_revenue_metrics(adset_df)
    _calculate_cost_metrics(adset_df)
    _calculate_roas_comparisons(adset_df)
    return adset_df


def _recalc_revenue_from_actions(adset_df):
    """Recalculate revenue from purchase action_value columns."""
    # Recalculate revenue from purchase action_value columns at adset-level
    # This ensures revenue is always from authoritative source, even if ad-level
    # revenue was calculated incorrectly (e.g., from purchase_roas * spend).
    # Purchase action_value columns were already summed during aggregation,
    # so we just need to sum them across columns to get total revenue per row.
    purchase_action_value_cols = [
        col
        for col in adset_df.columns
        if "action_value" in col.lower() and "purchase" in col.lower()
    ]

    if not purchase_action_value_cols:
        return

    # Recalculate revenue from summed purchase action_value columns
    # This overrides any revenue that was summed from ad-level
    # (which might be wrong if ad-level revenue was from purchase_roas * spend)
    calculate_revenue_from_purchase_actions(
        adset_df, update_inplace=True, custom_logger=logger
    )
    numeric_cols = [
        col
        for col in adset_df.columns
        if "action_value" in col.lower()
        and "purchase" in col.lower()
        and pd.api.types.is_numeric_dtype(adset_df[col])
    ]
    logger.info(
        "Recalculated revenue from %d purchase action_value columns "
        "at adset-level (authoritative source, overriding summed "
        "ad-level revenue)",
        len(numeric_cols),
    )


def _recalculate_purchase_roas(adset_df):
    """Recalculate purchase_roas from revenue/spend, avoiding zero ROAS."""
    if "revenue" in adset_df.columns and "spend" in adset_df.columns:
        revenue = pd.to_numeric(adset_df["revenue"], errors="coerce")
        spend = pd.to_numeric(adset_df["spend"], errors="coerce").replace(0, np.nan)

        # Calculate ROAS only when revenue > 0 and spend > 0
        calculated_roas = (revenue / spend).replace([np.inf, -np.inf], np.nan)
        # Replace zero ROAS with NaN to avoid zero values
        calculated_roas = calculated_roas.replace(0, np.nan)

        # Only update where calculated_roas > 0 or is NaN
        # (preserve existing valid ROAS)
        if "purchase_roas" in adset_df.columns:
            # Preserve existing non-zero ROAS, update where calculated is better
            existing_roas = pd.to_numeric(adset_df["purchase_roas"], errors="coerce")
            # Use calculated_roas where it's valid (> 0),
            # otherwise keep existing
            mask = calculated_roas.notna() & (calculated_roas > 0)
            adset_df.loc[mask, "purchase_roas"] = calculated_roas[mask]
            # If existing is zero or NaN, use calculated (which may be NaN)
            zero_or_na_mask = (
                existing_roas.isna() | (existing_roas == 0)
            ) & calculated_roas.notna()
            adset_df.loc[zero_or_na_mask, "purchase_roas"] = calculated_roas[
                zero_or_na_mask
            ]
        else:
            adset_df["purchase_roas"] = calculated_roas

        logger.info(
            "Recalculated purchase_roas from aggregated revenue/spend "
            "(zero values set to NaN to avoid zero ROAS)"
        )


def _calculate_revenue_metrics(adset_df):
    """Calculate revenue-related metrics."""
    if "revenue" in adset_df.columns and "impressions" in adset_df.columns:
        adset_df["revenue_per_impression"] = (
            adset_df["revenue"] / adset_df["impressions"]
        ).replace([np.inf, -np.inf], np.nan)

    if "revenue" in adset_df.columns and "clicks" in adset_df.columns:
        adset_df["revenue_per_click"] = (
            adset_df["revenue"] / adset_df["clicks"]
        ).replace([np.inf, -np.inf], np.nan)


def _calculate_cost_metrics(adset_df):
    """Calculate cost-related metrics."""
    if "spend" in adset_df.columns and "impressions" in adset_df.columns:
        adset_df["cost_per_impression"] = (
            adset_df["spend"] / adset_df["impressions"]
        ).replace([np.inf, -np.inf], np.nan)


def _calculate_roas_comparisons(adset_df):
    """Calculate ROAS comparison metrics."""
    if "purchase_roas" in adset_df.columns and "adset_roas" in adset_df.columns:
        adset_df["roas_vs_adset"] = (
            adset_df["purchase_roas"]
            / pd.to_numeric(adset_df["adset_roas"], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan)

    if "purchase_roas" in adset_df.columns and "campaign_roas" in adset_df.columns:
        adset_df["roas_vs_campaign"] = (
            adset_df["purchase_roas"]
            / pd.to_numeric(adset_df["campaign_roas"], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan)

    if "purchase_roas" in adset_df.columns and "account_roas" in adset_df.columns:
        adset_df["roas_vs_account"] = (
            adset_df["purchase_roas"]
            / pd.to_numeric(adset_df["account_roas"], errors="coerce")
        ).replace([np.inf, -np.inf], np.nan)


def _recalculate_budget_metrics(adset_df):
    """Recalculate budget-related metrics."""

    # Recalculate budget features using adset-level aggregated values
    if "adset_daily_budget" in adset_df.columns and "adset_spend" in adset_df.columns:
        # Budget utilization rate = (adset_spend / adset_daily_budget) * 100
        adset_df["budget_utilization_rate"] = (
            adset_df["adset_spend"] / adset_df["adset_daily_budget"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan) * 100
        logger.info("Recalculated budget_utilization_rate from adset_spend/budget")

        # Budget headroom = adset_daily_budget - adset_spend (clipped to 0)
        adset_df["budget_headroom"] = (
            adset_df["adset_daily_budget"] - adset_df["adset_spend"]
        ).clip(lower=0)
        logger.info(
            "Recalculated budget_headroom from " "adset_daily_budget - adset_spend"
        )

    # Recalculate budget ROAS efficiency using adset-level values
    if "adset_daily_budget" in adset_df.columns and "purchase_roas" in adset_df.columns:
        adset_df["budget_roas_efficiency"] = (
            adset_df["purchase_roas"]
            / adset_df["adset_daily_budget"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        logger.info("Recalculated budget_roas_efficiency from purchase_roas/budget")

    # Recalculate budget shares at adset-level
    # Ad share of adset budget = adset_spend / adset_daily_budget
    if "ad_share_of_adset_budget" in adset_df.columns:
        if (
            "adset_daily_budget" in adset_df.columns
            and "adset_spend" in adset_df.columns
        ):
            adset_df["ad_share_of_adset_budget"] = (
                adset_df["adset_spend"]
                / adset_df["adset_daily_budget"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)
            logger.info(
                "Recalculated ad_share_of_adset_budget from " "adset_spend/budget"
            )
    return adset_df


def _recalculate_cost_metrics(adset_df):
    """Recalculate cost metrics (CPC, CPM, CTR) from aggregated sums.

    P1-6: All ratio metrics must be calculated from aggregated sums,
    not from averaging individual ratios. This ensures mathematical correctness:
    - cpc = sum(spend) / sum(clicks)
    - cpm = sum(spend) / sum(impressions) * 1000
    - ctr = sum(clicks) / sum(impressions) * 100
    - unique_ctr = sum(unique_clicks) / sum(impressions) * 100
    - cost_per_unique_click = sum(spend) / sum(unique_clicks)
    - cost_per_unique_outbound_click = sum(spend) / sum(outbound_clicks)
    - frequency = sum(impressions) / sum(reach)
    """

    if "spend" in adset_df.columns and "clicks" in adset_df.columns:
        adset_df["cpc"] = (
            adset_df["spend"] / adset_df["clicks"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        logger.info("Recalculated CPC from aggregated spend/clicks")

    if "spend" in adset_df.columns and "impressions" in adset_df.columns:
        adset_df["cpm"] = (
            adset_df["spend"] / adset_df["impressions"].replace(0, np.nan) * 1000
        ).replace([np.inf, -np.inf], np.nan)
        logger.info("Recalculated CPM from aggregated spend/impressions")

    if "clicks" in adset_df.columns and "impressions" in adset_df.columns:
        adset_df["ctr"] = (
            adset_df["clicks"] / adset_df["impressions"].replace(0, np.nan) * 100
        ).replace([np.inf, -np.inf], np.nan)
        logger.info("Recalculated CTR from aggregated clicks/impressions")

    # P1-6: Recalculate unique_* metrics from aggregated sums
    if "unique_clicks" in adset_df.columns and "impressions" in adset_df.columns:
        adset_df["unique_ctr"] = (
            adset_df["unique_clicks"] / adset_df["impressions"].replace(0, np.nan) * 100
        ).replace([np.inf, -np.inf], np.nan)
        logger.info("Recalculated unique_ctr from aggregated unique_clicks/impressions")

    if "spend" in adset_df.columns and "unique_clicks" in adset_df.columns:
        adset_df["cost_per_unique_click"] = (
            adset_df["spend"] / adset_df["unique_clicks"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        logger.info(
            "Recalculated cost_per_unique_click from aggregated spend/unique_clicks"
        )

    if "spend" in adset_df.columns and "outbound_clicks" in adset_df.columns:
        adset_df["cost_per_unique_outbound_click"] = (
            adset_df["spend"] / adset_df["outbound_clicks"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        logger.info(
            "Recalculated cost_per_unique_outbound_click from aggregated spend/outbound_clicks"
        )

    # P1-6: Recalculate frequency from aggregated sums
    if "impressions" in adset_df.columns and "reach" in adset_df.columns:
        adset_df["frequency"] = (
            adset_df["impressions"] / adset_df["reach"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        logger.info("Recalculated frequency from aggregated impressions/reach")

    return adset_df


def _recalc_interaction_metrics(adset_df):
    """Recalculate interaction features."""

    # ROAS * Spend interaction = revenue (since purchase_roas = revenue/spend)
    if "revenue" in adset_df.columns:
        adset_df["roas_spend_interaction"] = adset_df["revenue"]
        logger.info("Recalculated roas_spend_interaction from revenue")

    # Expected revenue = spend * purchase_roas = revenue
    if "revenue" in adset_df.columns:
        adset_df["expected_revenue"] = adset_df["revenue"]
        logger.info("Recalculated expected_revenue from revenue")

    # Expected clicks = impressions * CTR / 100
    if "impressions" in adset_df.columns and "ctr" in adset_df.columns:
        adset_df["expected_clicks"] = (
            adset_df["impressions"] * adset_df["ctr"].fillna(0) / 100
        )
        logger.info("Recalculated expected_clicks from impressions * ctr/100")

    # CTR * CPC interaction
    if "ctr" in adset_df.columns and "cpc" in adset_df.columns:
        adset_df["ctr_cpc_interaction"] = adset_df["ctr"].fillna(0) * adset_df[
            "cpc"
        ].fillna(0)
        logger.info("Recalculated ctr_cpc_interaction from ctr * cpc")
    return adset_df


def _recalculate_engagement_metrics(adset_df):
    """Recalculate engagement and reach metrics."""

    # Recalculate engagement rate = (clicks / impressions) * 100
    if "clicks" in adset_df.columns and "impressions" in adset_df.columns:
        adset_df["engagement_rate"] = (
            adset_df["clicks"] / adset_df["impressions"].replace(0, np.nan) * 100
        ).replace([np.inf, -np.inf], np.nan)
        logger.info("Recalculated engagement_rate from aggregated clicks/impressions")

    # Recalculate reach efficiency = impressions / reach
    if "impressions" in adset_df.columns and "reach" in adset_df.columns:
        adset_df["reach_efficiency"] = (
            adset_df["impressions"] / adset_df["reach"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        logger.info("Recalculated reach_efficiency from aggregated impressions/reach")
    return adset_df


def _recalculate_derived_metrics(adset_df):
    """Recalculate derived efficiency and relative performance metrics."""
    adset_df = _recalculate_roas_metrics(adset_df)
    adset_df = _recalculate_budget_metrics(adset_df)
    adset_df = _recalculate_cost_metrics(adset_df)
    adset_df = _recalc_interaction_metrics(adset_df)
    adset_df = _recalculate_engagement_metrics(adset_df)
    return adset_df


def _calc_roas_rolling_metrics(adset_df):
    """Calculate rolling and EMA metrics for ROAS."""
    adset_df["purchase_roas_rolling_7d"] = adset_df.groupby("adset_id")[
        "purchase_roas"
    ].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    logger.info("Calculated purchase_roas_rolling_7d")

    adset_df["purchase_roas_rolling_14d"] = adset_df.groupby("adset_id")[
        "purchase_roas"
    ].transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    logger.info("Calculated purchase_roas_rolling_14d")

    adset_df["purchase_roas_ema_7d"] = adset_df.groupby("adset_id")[
        "purchase_roas"
    ].transform(lambda x: x.ewm(span=7, min_periods=1, adjust=False).mean())
    logger.info("Calculated purchase_roas_ema_7d")

    adset_df["purchase_roas_ema_14d"] = adset_df.groupby("adset_id")[
        "purchase_roas"
    ].transform(lambda x: x.ewm(span=14, min_periods=1, adjust=False).mean())
    logger.info("Calculated purchase_roas_ema_14d")
    return adset_df


def _calc_spend_rolling_metrics(adset_df):
    """Calculate rolling and EMA metrics for spend."""
    adset_df["spend_rolling_7d"] = adset_df.groupby("adset_id")["spend"].transform(
        lambda x: x.rolling(window=7, min_periods=1).sum()
    )
    logger.info("Calculated spend_rolling_7d")

    adset_df["spend_rolling_14d"] = adset_df.groupby("adset_id")["spend"].transform(
        lambda x: x.rolling(window=14, min_periods=1).sum()
    )
    logger.info("Calculated spend_rolling_14d")

    adset_df["spend_ema_7d"] = adset_df.groupby("adset_id")["spend"].transform(
        lambda x: x.ewm(span=7, min_periods=1, adjust=False).mean()
    )
    logger.info("Calculated spend_ema_7d")

    adset_df["spend_rolling_7d_std"] = adset_df.groupby("adset_id")["spend"].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )
    logger.info("Calculated spend_rolling_7d_std")

    adset_df["spend_rolling_14d_std"] = adset_df.groupby("adset_id")["spend"].transform(
        lambda x: x.rolling(window=14, min_periods=1).std()
    )
    logger.info("Calculated spend_rolling_14d_std")
    return adset_df


def _calc_roas_std_metrics(adset_df):
    """Calculate rolling std metrics for purchase_roas."""
    adset_df["purchase_roas_rolling_7d_std"] = adset_df.groupby("adset_id")[
        "purchase_roas"
    ].transform(lambda x: x.rolling(window=7, min_periods=1).std())
    logger.info("Calculated purchase_roas_rolling_7d_std")

    adset_df["purchase_roas_rolling_14d_std"] = adset_df.groupby("adset_id")[
        "purchase_roas"
    ].transform(lambda x: x.rolling(window=14, min_periods=1).std())
    logger.info("Calculated purchase_roas_rolling_14d_std")
    return adset_df


def _calc_roas_trend(adset_df):
    """Calculate ROAS trend metric."""
    adset_df["roas_trend"] = adset_df.groupby("adset_id")[
        "purchase_roas_rolling_7d"
    ].transform(
        lambda x: (
            (x - x.shift(7)) / x.shift(7).replace(0, np.nan)
            if x.shift(7).notna().any()
            else pd.Series(0, index=x.index)
        )
    )
    adset_df["roas_trend"] = adset_df["roas_trend"].fillna(0)
    logger.info("Calculated roas_trend")
    return adset_df


def _calc_rolling_window_coverage(adset_df):
    """Calculate data quality indicators for rolling window coverage.

    For each adset, calculates what fraction of the rolling window
    actually contains data. A value < 1.0 indicates partial data.

    Examples:
        - Day 3 adset: rolling_7d_coverage = 3/7 = 0.43
        - Day 10 adset: rolling_7d_coverage = 7/7 = 1.0
        - Day 3 adset: rolling_14d_coverage = 3/14 = 0.21

    This helps rules understand when rolling metrics are based on
    insufficient data and should be down-weighted.
    """
    # Calculate row number (days of data) for each adset
    adset_df["days_of_data"] = adset_df.groupby("adset_id").cumcount() + 1

    # Calculate coverage ratios for different window sizes
    adset_df["rolling_7d_coverage"] = (adset_df["days_of_data"] / 7.0).clip(upper=1.0)
    logger.info("Calculated rolling_7d_coverage")

    adset_df["rolling_14d_coverage"] = (adset_df["days_of_data"] / 14.0).clip(upper=1.0)
    logger.info("Calculated rolling_14d_coverage")

    # Flag for low-quality rolling windows (< 50% coverage)
    adset_df["rolling_low_quality"] = (
        (adset_df["rolling_7d_coverage"] < 0.5)
        | (adset_df["rolling_14d_coverage"] < 0.5)
    ).astype(int)
    logger.info("Calculated rolling_low_quality flag")

    return adset_df


def _calculate_rolling_metrics(adset_df):
    """Calculate rolling and EMA metrics for ROAS and spend."""
    if "purchase_roas" not in adset_df.columns:
        return adset_df
    if "date_start" not in adset_df.columns:
        return adset_df

    # Ensure date_start is datetime
    adset_df["date_start"] = pd.to_datetime(adset_df["date_start"])

    # Sort by adset_id and date_start for rolling calculations
    adset_df = adset_df.sort_values(["adset_id", "date_start"]).reset_index(drop=True)

    # Calculate rolling window coverage (data quality indicator)
    adset_df = _calc_rolling_window_coverage(adset_df)

    # Calculate purchase_roas rolling metrics
    adset_df = _calc_roas_rolling_metrics(adset_df)

    # Calculate spend rolling metrics
    if "spend" in adset_df.columns:
        adset_df = _calc_spend_rolling_metrics(adset_df)
        adset_df = _calc_roas_std_metrics(adset_df)

    # Calculate roas_trend
    adset_df = _calc_roas_trend(adset_df)

    return adset_df


def _create_lagged_features(adset_df):
    """Create lagged versions of rolling metrics to prevent lookahead bias.

    Lagged features represent data through yesterday (excluding today).
    These are used as features for predicting today's ROAS.

    For example:
    - purchase_roas_rolling_7d_lagged = rolling avg of past 7 days ending yesterday
    - This can be used to predict purchase_roas (today's actual value)
    """
    if "date_start" not in adset_df.columns:
        return adset_df

    # Sort by adset_id and date_start for proper shifting
    adset_df = adset_df.sort_values(["adset_id", "date_start"]).reset_index(drop=True)

    # Lagged rolling ROAS metrics (shift by 1 day)
    rolling_cols = [
        "purchase_roas_rolling_7d",
        "purchase_roas_rolling_14d",
        "purchase_roas_ema_7d",
        "purchase_roas_ema_14d",
    ]

    for col in rolling_cols:
        if col in adset_df.columns:
            adset_df[f"{col}_lagged"] = adset_df.groupby("adset_id")[col].shift(1)

    # Lagged rolling spend metrics
    if "spend" in adset_df.columns:
        spend_rolling_cols = [
            "spend_rolling_7d",
            "spend_rolling_14d",
            "spend_ema_7d",
        ]

        for col in spend_rolling_cols:
            if col in adset_df.columns:
                adset_df[f"{col}_lagged"] = adset_df.groupby("adset_id")[col].shift(1)

    # Lagged CTR/CPC metrics
    ctr_cpc_cols = [
        "ctr_rolling_7d",
        "ctr_rolling_14d",
        "cpc_rolling_7d",
        "cpc_rolling_14d",
    ]

    for col in ctr_cpc_cols:
        if col in adset_df.columns:
            adset_df[f"{col}_lagged"] = adset_df.groupby("adset_id")[col].shift(1)

    # Lagged trend metrics
    trend_cols = [
        "roas_trend",
        "cpc_trend_7d",
        "ctr_trend_7d",
    ]

    for col in trend_cols:
        if col in adset_df.columns:
            adset_df[f"{col}_lagged"] = adset_df.groupby("adset_id")[col].shift(1)

    logger.info("Created lagged features to prevent lookahead bias")

    return adset_df


def _calculate_ad_level_statistics(
    adset_df: pd.DataFrame, ad_features_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate comprehensive ad-level statistical features from ad-level data.

    These features capture the distribution and diversity of ad performance within
    each adset using statistics like mean, median, min, max, and std.

    IMPORTANT: All ad-level statistics use LAGGED data (yesterday's ad performance)
    to predict today's adset-level ROAS. This prevents lookahead bias.

    Features include:
    - Ad diversity: num_ads, num_active_ads, ad_diversity
    - ROAS distribution: mean, median, min, max, std, range
    - CTR distribution: mean, median, min, max, std
    - CPC distribution: mean, median, min, max, std
    - Spend distribution: mean, median, min, max, std
    - Impressions distribution: mean, median, min, max, std
    - Format diversity: video_ads_ratio, format_diversity_score
    - Spend concentration: ad_spend_gini, top_ad_spend_pct

    Args:
        adset_df: Adset-level aggregated DataFrame
        ad_features_df: Original ad-level DataFrame

    Returns:
        adset_df with ad-level statistical features added
    """
    logger.info("Calculating comprehensive ad-level statistical features...")

    # Ensure we have the required columns
    required_cols = ["adset_id", "date_start", "spend", "purchase_roas"]
    missing = [c for c in required_cols if c not in ad_features_df.columns]
    if missing:
        logger.warning(
            f"Missing required columns for ad-level statistics: {missing}. "
            f"Skipping ad-level feature calculation."
        )
        return adset_df

    # Drop existing ad-level features to avoid merge conflicts
    # These may exist from previous extractions, including _x and _y suffixes

    # Define all ad-level feature names that will be created
    ad_level_feature_names = [
        "num_ads",
        "num_active_ads",
        "ad_diversity",
        "video_ads_ratio",
        "format_diversity_score",
        "ad_roas_mean",
        "ad_roas_median",
        "ad_roas_min",
        "ad_roas_max",
        "ad_roas_std",
        "ad_roas_range",
        "ad_ctr_mean",
        "ad_ctr_median",
        "ad_ctr_min",
        "ad_ctr_max",
        "ad_ctr_std",
        "ad_cpc_mean",
        "ad_cpc_median",
        "ad_cpc_min",
        "ad_cpc_max",
        "ad_cpc_std",
        "ad_spend_mean",
        "ad_spend_median",
        "ad_spend_min",
        "ad_spend_max",
        "ad_spend_std",
        "ad_impressions_mean",
        "ad_impressions_median",
        "ad_impressions_min",
        "ad_impressions_max",
        "ad_impressions_std",
        "ad_pct_high_roas",
        "ad_pct_low_roas",
        "ad_pct_zero_roas",
        "ad_spend_gini",
        "top_ad_spend_pct",
    ]

    existing_ad_cols = []
    for feat in ad_level_feature_names:
        if feat in adset_df.columns:
            existing_ad_cols.append(feat)
        if f"{feat}_x" in adset_df.columns:
            existing_ad_cols.append(f"{feat}_x")
        if f"{feat}_y" in adset_df.columns:
            existing_ad_cols.append(f"{feat}_y")

    # Also drop any other columns starting with ad_ (except adset_id)
    existing_ad_cols.extend(
        [
            col
            for col in adset_df.columns
            if col.startswith("ad_")
            and col != "adset_id"
            and col not in existing_ad_cols
        ]
    )

    # Remove duplicates
    existing_ad_cols = list(set(existing_ad_cols))

    if existing_ad_cols:
        logger.info(
            f"Dropping {len(existing_ad_cols)} existing ad-level features before recalculation"
        )
        adset_df = adset_df.drop(columns=existing_ad_cols)

    # To prevent lookahead bias, we'll shift ad-level data by 1 day
    # For each adset-date, we'll use statistics from the PREVIOUS day's ad performance
    ad_features_sorted = ad_features_df.sort_values(["adset_id", "date_start"])

    # Create lagged version of ad-level data (shift by 1 day within each adset)
    ad_features_lagged = ad_features_sorted.copy()
    lagged_cols = [
        "purchase_roas",
        "ctr",
        "cpc",
        "cpm",
        "spend",
        "impressions",
        "clicks",
        "actions",
        "frequency",
    ]
    for col in lagged_cols:
        if col in ad_features_lagged.columns:
            ad_features_lagged[f"{col}_lagged"] = ad_features_lagged.groupby(
                "adset_id"
            )[col].shift(1)

    logger.info("Created lagged ad-level features to prevent lookahead bias")

    # Group by adset and date to calculate statistical metrics
    stats_metrics = []

    for (adset_id, date_start), group in ad_features_lagged.groupby(
        ["adset_id", "date_start"]
    ):
        # Filter to ads with spend (active ads) - using lagged spend
        if "spend_lagged" in group.columns:
            active_ads = group[group["spend_lagged"] > 0]
        else:
            active_ads = group[group["spend"] > 0]

        metrics = {
            "adset_id": adset_id,
            "date_start": date_start,
        }

        # === 1. AD DIVERSITY FEATURES ===
        metrics["num_ads"] = len(group)
        metrics["num_active_ads"] = len(active_ads)

        if "ad_name" in group.columns:
            metrics["ad_diversity"] = group["ad_name"].nunique()
        else:
            metrics["ad_diversity"] = len(group)

        # === 2. FORMAT DIVERSITY ===
        if "ad_format" in group.columns:
            video_count = (group["ad_format"] == "video").sum()
            metrics["video_ads_ratio"] = (
                video_count / len(group) if len(group) > 0 else 0
            )
            metrics["format_diversity_score"] = group["ad_format"].nunique()
        else:
            # Try to detect from ad_name if available
            if "ad_name" in group.columns:
                video_count = (
                    group["ad_name"].str.contains("video", case=False, na=False).sum()
                )
                metrics["video_ads_ratio"] = (
                    video_count / len(group) if len(group) > 0 else 0
                )
            else:
                metrics["video_ads_ratio"] = 0
            metrics["format_diversity_score"] = 1

        # === 3. ROAS DISTRIBUTION (using lagged ROAS to prevent lookahead) ===
        roas_col = (
            "purchase_roas_lagged"
            if "purchase_roas_lagged" in active_ads.columns
            else "purchase_roas"
        )
        if roas_col in active_ads.columns and len(active_ads) > 0:
            roas_values = active_ads[roas_col].replace([np.inf, -np.inf], 0).fillna(0)
            metrics["ad_roas_mean"] = roas_values.mean()
            metrics["ad_roas_median"] = roas_values.median()
            metrics["ad_roas_min"] = roas_values.min()
            metrics["ad_roas_max"] = roas_values.max()
            metrics["ad_roas_std"] = roas_values.std() if len(roas_values) > 1 else 0
            metrics["ad_roas_range"] = metrics["ad_roas_max"] - metrics["ad_roas_min"]
        else:
            metrics["ad_roas_mean"] = 0
            metrics["ad_roas_median"] = 0
            metrics["ad_roas_min"] = 0
            metrics["ad_roas_max"] = 0
            metrics["ad_roas_std"] = 0
            metrics["ad_roas_range"] = 0

        # === 4. CTR DISTRIBUTION ===
        ctr_col = "ctr_lagged" if "ctr_lagged" in active_ads.columns else "ctr"
        if ctr_col in active_ads.columns and len(active_ads) > 0:
            ctr_values = active_ads[ctr_col].replace([np.inf, -np.inf], 0).fillna(0)
            metrics["ad_ctr_mean"] = ctr_values.mean()
            metrics["ad_ctr_median"] = ctr_values.median()
            metrics["ad_ctr_min"] = ctr_values.min()
            metrics["ad_ctr_max"] = ctr_values.max()
            metrics["ad_ctr_std"] = ctr_values.std() if len(ctr_values) > 1 else 0
        else:
            metrics["ad_ctr_mean"] = 0
            metrics["ad_ctr_median"] = 0
            metrics["ad_ctr_min"] = 0
            metrics["ad_ctr_max"] = 0
            metrics["ad_ctr_std"] = 0

        # === 5. CPC DISTRIBUTION ===
        cpc_col = "cpc_lagged" if "cpc_lagged" in active_ads.columns else "cpc"
        if cpc_col in active_ads.columns and len(active_ads) > 0:
            cpc_values = active_ads[cpc_col].replace([np.inf, -np.inf], 0).fillna(0)
            metrics["ad_cpc_mean"] = cpc_values.mean()
            metrics["ad_cpc_median"] = cpc_values.median()
            metrics["ad_cpc_min"] = cpc_values.min()
            metrics["ad_cpc_max"] = cpc_values.max()
            metrics["ad_cpc_std"] = cpc_values.std() if len(cpc_values) > 1 else 0
        else:
            metrics["ad_cpc_mean"] = 0
            metrics["ad_cpc_median"] = 0
            metrics["ad_cpc_min"] = 0
            metrics["ad_cpc_max"] = 0
            metrics["ad_cpc_std"] = 0

        # === 6. SPEND DISTRIBUTION ===
        spend_col = "spend_lagged" if "spend_lagged" in active_ads.columns else "spend"
        if spend_col in active_ads.columns and len(active_ads) > 0:
            spend_values = active_ads[spend_col].replace([np.inf, -np.inf], 0).fillna(0)
            metrics["ad_spend_mean"] = spend_values.mean()
            metrics["ad_spend_median"] = spend_values.median()
            metrics["ad_spend_min"] = spend_values.min()
            metrics["ad_spend_max"] = spend_values.max()
            metrics["ad_spend_std"] = spend_values.std() if len(spend_values) > 1 else 0
        else:
            metrics["ad_spend_mean"] = 0
            metrics["ad_spend_median"] = 0
            metrics["ad_spend_min"] = 0
            metrics["ad_spend_max"] = 0
            metrics["ad_spend_std"] = 0

        # === 7. IMPRESSIONS DISTRIBUTION ===
        imp_col = (
            "impressions_lagged"
            if "impressions_lagged" in active_ads.columns
            else "impressions"
        )
        if imp_col in active_ads.columns and len(active_ads) > 0:
            imp_values = active_ads[imp_col].replace([np.inf, -np.inf], 0).fillna(0)
            metrics["ad_impressions_mean"] = imp_values.mean()
            metrics["ad_impressions_median"] = imp_values.median()
            metrics["ad_impressions_min"] = imp_values.min()
            metrics["ad_impressions_max"] = imp_values.max()
            metrics["ad_impressions_std"] = (
                imp_values.std() if len(imp_values) > 1 else 0
            )
        else:
            metrics["ad_impressions_mean"] = 0
            metrics["ad_impressions_median"] = 0
            metrics["ad_impressions_min"] = 0
            metrics["ad_impressions_max"] = 0
            metrics["ad_impressions_std"] = 0

        # === 8. SPEND CONCENTRATION ===
        if len(active_ads) > 1 and spend_col in active_ads.columns:
            spend_values = active_ads[spend_col].values
            # Calculate Gini coefficient
            sorted_spend = np.sort(spend_values)
            n = len(sorted_spend)
            cumsum = np.cumsum(sorted_spend)
            gini = (
                (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
            )
            metrics["ad_spend_gini"] = gini
            # Top ad spend percentage
            top_ad_spend = spend_values.max()
            total_spend = spend_values.sum()
            metrics["top_ad_spend_pct"] = (
                top_ad_spend / total_spend if total_spend > 0 else 1.0
            )
        else:
            metrics["ad_spend_gini"] = 0
            metrics["top_ad_spend_pct"] = 1.0

        # === 9. PERFORMANCE RATIOS (based on lagged data) ===
        # Percentage of high/low/zero ROAS ads
        if roas_col in active_ads.columns and len(active_ads) > 0:
            roas_values = active_ads[roas_col].fillna(0)
            metrics["ad_pct_high_roas"] = (roas_values > 3.0).mean()
            metrics["ad_pct_low_roas"] = (roas_values < 1.0).mean()
            metrics["ad_pct_zero_roas"] = (roas_values == 0).mean()
        else:
            metrics["ad_pct_high_roas"] = 0
            metrics["ad_pct_low_roas"] = 0
            metrics["ad_pct_zero_roas"] = 0

        stats_metrics.append(metrics)

    # Create DataFrame and merge
    stats_df = pd.DataFrame(stats_metrics)

    # Ensure consistent datetime types for date columns before merging
    # This prevents merge errors when one DataFrame has datetime and another has object
    if "date_start" in adset_df.columns:
        adset_df["date_start"] = pd.to_datetime(adset_df["date_start"])
    if "date_start" in stats_df.columns:
        stats_df["date_start"] = pd.to_datetime(stats_df["date_start"])

    # Merge with adset_df
    merged_df = adset_df.merge(
        stats_df,
        on=["adset_id", "date_start"],
        how="left",
        indicator=True,
    )

    # Check merge result
    matched_count = (merged_df["_merge"] == "both").sum()
    logger.info(
        f"Merged ad-level statistics for {matched_count}/{len(adset_df)} adset-date combinations"
    )

    # Debug: Check if key features exist in merged_df
    key_features = [
        "num_ads",
        "num_active_ads",
        "video_ads_ratio",
        "format_diversity_score",
        "top_ad_spend_pct",
    ]
    for feat in key_features:
        if feat in merged_df.columns:
            logger.info(
                f"  {feat}: EXISTS (non-null count: {merged_df[feat].notna().sum()})"
            )
        else:
            logger.info(f"  {feat}: MISSING!")

    # Drop indicator column
    adset_df = merged_df.drop(columns=["_merge"])

    # Fill NaN values for adsets with no ad-level data
    # Most features default to 0, except ratios which default to sensible values
    numeric_defaults = 0
    ratio_defaults = {
        "video_ads_ratio": 0,
        "format_diversity_score": 1,
        "top_ad_spend_pct": 1.0,
        "ad_spend_gini": 0,
        "num_ads": 1,
        "num_active_ads": 0,
        "ad_diversity": 1,
    }

    # Get all ad-level feature columns
    ad_feature_cols = [
        col for col in stats_df.columns if col not in ["adset_id", "date_start"]
    ]

    logger.info(
        f"Processing {len(ad_feature_cols)} ad-level feature columns from stats_df"
    )

    for col in ad_feature_cols:
        if col in adset_df.columns:
            default_val = ratio_defaults.get(col, numeric_defaults)
            adset_df[col] = adset_df[col].fillna(default_val)
        else:
            logger.warning(
                f"  Column {col} from stats_df not found in adset_df after merge!"
            )

    feature_count = len(ad_feature_cols)
    logger.info(f"Calculated {feature_count} ad-level statistical features")
    logger.info(
        f"Features include: ROAS (6), CTR (5), CPC (5), Spend (5), Impressions (5), Diversity (5), Concentration (3)"
    )

    return adset_df


def _aggregate_ad_to_adset(ad_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ad-level features to adset-level.

    Args:
        ad_features_df: Ad-level DataFrame with all features

    Returns:
        Adset-level aggregated DataFrame
    """
    logger.info(
        "Aggregating %d ad-level records to adset-level...", len(ad_features_df)
    )

    # Identify grouping columns
    # (should be same for all ads in an adset on same date)
    groupby_cols = ["adset_id", "date_start"]

    # Identify columns that should be summed (ad-level metrics)
    # Start with base columns, then add additional ones
    sum_cols = list(BASE_SUM_COLUMNS) + [
        "actions",
        "action_values",
    ]
    # Note: 'reach' is in BASE_SUM_COLUMNS for hourly-to-daily aggregation,
    # but for ad-to-adset aggregation, it should be averaged
    # (handled in mean_cols)

    # Identify columns that should be averaged (ratios/rates)
    mean_cols = [
        "frequency",
        "cpc",
        "cpm",
        "ctr",
        "unique_ctr",
        "cost_per_unique_click",
        "cost_per_unique_outbound_click",
        "reach",  # Average reach across ads in adset
        "days_since_start",  # Should be same, but use mean for safety
    ]

    # Identify columns that should take first (same for all ads in adset/date)
    first_cols = [
        # Account/Campaign/Adset identifiers
        "account_id",
        "account_name",
        "campaign_id",
        "campaign_name",
        "adset_id",
        "adset_name",
        "date_start",
        "date_stop",
        "export_date",
        # Ranking metrics (same for all ads in adset/date)
        "conversion_rate_ranking",
        "engagement_rate_ranking",
        "quality_ranking",
        # Adset-level features (already aggregated, same for all ads)
        "adset_name_adset",
        "campaign_id_adset",
        "adset_spend",
        "adset_impressions",
        "adset_clicks",
        "adset_roas",
        "adset_cpc",
        "adset_cpm",
        "adset_ctr",
        "adset_reach",
        "adset_frequency",
        "adset_targeting",
        "adset_targeting_age_min",
        "adset_targeting_age_max",
        "adset_targeting_countries",
        "adset_targeting_location_types",
        "adset_targeting_advantage_audience",
        "adset_targeting_genders",
        "adset_targeting_age_range",
        "adset_targeting_auto_age",
        "adset_targeting_auto_gender",
        "adset_targeting_excluded_custom_audiences_count",
        "adset_targeting_custom_audiences_count",
        "adset_targeting_custom_audiences",
        # Campaign-level features (same for all adsets in campaign)
        "campaign_name_campaign",
        "campaign_spend",
        "campaign_impressions",
        "campaign_clicks",
        "campaign_roas",
        "campaign_cpc",
        "campaign_cpm",
        "campaign_ctr",
        "campaign_reach",
        "campaign_frequency",
        # Account-level features (same for all campaigns in account)
        "account_spend",
        "account_impressions",
        "account_clicks",
        "account_roas",
        "account_cpc",
        "account_cpm",
        "account_ctr",
        "account_reach",
        "account_frequency",
        # Time features (same for all ads on same date)
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "is_weekend",
        # Adset time/budget features (same for all ads in adset)
        "adset_end_time",
        "adset_start_time",
        "adset_lifetime_budget",
        # Normalized and bucketed features
        # (take first since they're based on same source data)
        # We'll recalculate some of these after aggregation
    ]

    # Filter to existing columns only
    sum_cols = [col for col in sum_cols if col in ad_features_df.columns]
    mean_cols = [col for col in mean_cols if col in ad_features_df.columns]
    first_cols = [col for col in first_cols if col in ad_features_df.columns]

    # Move object-type columns from mean_cols to first_cols
    # (can't aggregate with mean)
    object_mean_cols = [
        col
        for col in mean_cols
        if col in ad_features_df.columns
        and not pd.api.types.is_numeric_dtype(ad_features_df[col])
    ]
    mean_cols = [col for col in mean_cols if col not in object_mean_cols]
    first_cols.extend(object_mean_cols)

    # Add all action_value columns to sum_cols (they should be summed)
    # Only include numeric columns to avoid type errors during aggregation
    action_value_cols = [
        col
        for col in ad_features_df.columns
        if (
            "action_value" in col.lower()
            and col not in sum_cols + mean_cols + first_cols + groupby_cols
            and pd.api.types.is_numeric_dtype(ad_features_df[col])
        )
    ]
    sum_cols.extend(action_value_cols)
    if action_value_cols:
        logger.info("Added %d action_value columns to sum_cols", len(action_value_cols))

    # Get all other columns (normalized, bucketed, calculated features)
    other_cols = [
        col
        for col in ad_features_df.columns
        if col not in groupby_cols + sum_cols + mean_cols + first_cols
    ]

    # Build aggregation dictionary
    agg_dict = _build_agg_dict(
        sum_cols,
        mean_cols,
        first_cols,
        other_cols,
        groupby_cols,
    )

    # Special handling for purchase_roas:
    # calculate from aggregated revenue/spend
    ad_features_df, agg_dict = _handle_revenue_calculation(
        ad_features_df,
        agg_dict,
        sum_cols,
    )

    # Perform aggregation
    logger.info("Grouping by: %s", groupby_cols)
    logger.info(
        "Summing: %d columns", len([c for c in agg_dict.values() if c == "sum"])
    )
    logger.info(
        "Averaging: %d columns",
        len([c for c in agg_dict.values() if c == "mean"]),
    )
    logger.info(
        "Taking first: %d columns",
        len([c for c in agg_dict.values() if c == "first"]),
    )

    adset_df = ad_features_df.groupby(groupby_cols).agg(agg_dict).reset_index()

    # Recalculate derived metrics
    adset_df = _recalculate_derived_metrics(adset_df)

    # Calculate rolling metrics with proper date ordering
    adset_df = _calculate_rolling_metrics(adset_df)

    # Create lagged features to prevent lookahead bias in ML predictions
    adset_df = _create_lagged_features(adset_df)

    # Calculate ad-level statistical features from ad-level data
    adset_df = _calculate_ad_level_statistics(adset_df, ad_features_df)

    # Calculate simple health_score using LAGGED ROAS to prevent lookahead bias
    # P0-4: Always use lagged features, never fall back to non-lagged
    roas_col_for_health = "purchase_roas_rolling_7d_lagged"

    if roas_col_for_health in adset_df.columns:
        # Simple health score: based on ROAS (normalized to 0-1)
        # ROAS >= 3.0 -> health = 1.0, ROAS = 0 -> health = 0.0
        adset_df["health_score"] = (
            adset_df[roas_col_for_health]
            .fillna(0)  # Fill NaN with 0 (no ROAS data)
            .apply(lambda x: min(max(x / 3.0, 0.0), 1.0) if x >= 0 else 0.5)
        )
        # If ROAS is 0, use default 0.5 (neutral) instead of 0
        adset_df["health_score"] = adset_df["health_score"].replace(0, 0.5)
        logger.info(
            f"Calculated health_score from {roas_col_for_health} (simplified, lagged)"
        )
    else:
        # P0-4: Lagged feature not available - this is an error
        # Use default health_score but log a warning
        adset_df["health_score"] = 0.5
        logger.warning(
            f"Lagged ROAS feature '{roas_col_for_health}' not found. "
            f"Using default health_score=0.5. This may indicate look-ahead bias!"
        )

    logger.info("Aggregation complete!")
    logger.info("Original ad-level records: %d", len(ad_features_df))
    logger.info("Aggregated adset-level records: %d", len(adset_df))
    logger.info("Unique adsets: %d", adset_df["adset_id"].nunique())
    if "date_start" in adset_df.columns:
        logger.info(
            "Date range: %s to %s",
            adset_df["date_start"].min(),
            adset_df["date_start"].max(),
        )

    return adset_df


def _integrate_shopify_data(
    adset_df: pd.DataFrame, customer: str, platform: str
) -> pd.DataFrame:
    """Integrate Shopify data into adset features.

    Args:
        adset_df: Adset-level features DataFrame.
        customer: Customer name.
        platform: Platform name.

    Returns:
        Adset DataFrame with Shopify metrics added.
    """
    try:
        from src.meta.adset.allocator.features.integrations.shopify import (
            ShopifyFeatureExtractor,
            get_shopify_data_path,
        )

        # Get path to Shopify CSV
        shopify_path = get_shopify_data_path(customer, platform)

        # Check if Shopify data exists
        if not shopify_path.exists():
            logger.info(f"Shopify data not found: {shopify_path}")
            return adset_df

        logger.info("=" * 70)
        logger.info("Integrating Shopify Data")
        logger.info("=" * 70)

        # Create extractor and load data
        extractor = ShopifyFeatureExtractor(str(shopify_path))

        if not extractor.load_and_process():
            logger.warning("Failed to load Shopify data")
            return adset_df

        # Enrich adset features with Shopify ROAS
        enriched_df = extractor.enrich_adset_features(adset_df)

        # Log new features
        shopify_cols = [col for col in enriched_df.columns if "shopify" in col.lower()]
        if shopify_cols:
            logger.info(f"Added {len(shopify_cols)} Shopify features:")
            for col in shopify_cols:
                logger.info(f"  - {col}")

        return enriched_df

    except ImportError as e:
        logger.warning(f"Shopify integration not available: {e}")
        return adset_df
    except Exception as e:
        logger.error(f"Error integrating Shopify data: {e}")
        return adset_df


def main():
    """Main function for feature extraction."""
    args = _parse_arguments()

    # If explicit files are provided, don't use "all" mode
    has_explicit_files = (
        args.ad_file or args.adset_file or args.campaign_file or args.account_file
    )

    if args.customer == "all" and not has_explicit_files:
        return _run_all_customers(args)

    # If customer is "all" but explicit files are provided, treat as None
    if args.customer == "all" and has_explicit_files:
        args.customer = None

    logger.info("=" * 70)
    logger.info("Feature Extraction: Multi-Level Data Joining & Preprocessing")
    if args.customer:
        logger.info("Customer: %s", args.customer)
    logger.info("=" * 70)

    enriched_df = _extract_features(args)
    _print_summary(enriched_df)
    return 0


def _parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract and preprocess features from multi-level data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_customer_argument(parser)
    add_platform_argument(parser)
    add_config_argument(parser)
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip feature preprocessing (normalization, bucketing)",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip feature normalization"
    )
    parser.add_argument(
        "--no-bucket", action="store_true", help="Skip feature bucketing"
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip adset-level aggregation (only extract ad-level features)",
    )
    parser.add_argument(
        "--ad-file",
        type=str,
        default=None,
        help="Explicit path to ad-level CSV file",
    )
    parser.add_argument(
        "--adset-file",
        type=str,
        default=None,
        help="Explicit path to adset-level CSV file",
    )
    parser.add_argument(
        "--campaign-file",
        type=str,
        default=None,
        help="Explicit path to campaign-level CSV file",
    )
    parser.add_argument(
        "--account-file",
        type=str,
        default=None,
        help="Explicit path to account-level CSV file",
    )
    return parser.parse_args()


def _process_customer_extract(customer, args):
    """Process extraction for a single customer"""
    ensure_customer_dirs(customer, args.platform)
    customer_args = argparse.Namespace(**vars(args))
    customer_args.customer = customer

    try:
        enriched_df = _extract_features(customer_args)
        _print_summary(enriched_df)
        # Aggregation is handled inside _extract_features if enabled
        return True
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as err:
        logger.error("Failed to extract features for %s: %s", customer, err)
        return False


def _run_all_customers(args):
    """Run extraction for all customers"""
    return process_all_customers(
        args.config, "extraction", _process_customer_extract, args
    )


def _extract_features(args):
    """Extract and process features from data.

    Automatically processes raw Meta Ads CSV files by flattening JSON columns
    (actions, action_values, cost_per_action_type, adset_targeting) if detected.
    The Loader class handles this transparently - no additional steps needed.

    Directory structure:
    - Raw data: datasets/{customer}/raw/ (or datasets/{customer}/ if no raw/)
    - Output features: datasets/{customer}/features/
      (or datasets/{customer}/ if no features/)
    """
    # Get customer-specific data directory
    # If customer is None, use base datasets directory
    # get_customer_data_dir() automatically checks for raw/ subdirectory
    if args.customer:
        data_dir = str(get_customer_data_dir(args.customer, args.platform))
        ensure_customer_dirs(args.customer, args.platform)
        logger.info("Processing customer: %s", args.customer)
        if args.platform:
            logger.info("Platform: %s", args.platform)
        logger.info("Data directory: %s", data_dir)
    else:
        data_dir = "datasets"

    # Load data - automatically processes raw Meta Ads format if detected
    # The Loader detects JSON columns and flattens them before loading
    if args.ad_file or args.adset_file or args.campaign_file or args.account_file:
        data = Loader.load_all_data(
            data_dir=data_dir,
            account_file=args.account_file,
            campaign_file=args.campaign_file,
            adset_file=args.adset_file,
            ad_file=args.ad_file,
        )
    else:
        data = Loader.load_all_data(data_dir=data_dir)

    if "ad" not in data or data["ad"] is None:
        raise ValueError("Ad-level data is required for feature extraction")

    enriched_df = Joiner.join_all_levels(
        ad_df=data["ad"],
        account_df=data.get("account"),
        campaign_df=data.get("campaign"),
        adset_df=data.get("adset"),
    )

    enriched_df = Aggregator.create_aggregated_features(
        enriched_df,
        preprocess=not args.no_preprocess,
        normalize=not args.no_normalize,
        bucket=not args.no_bucket,
    )

    # Determine output file path
    if args.customer:
        output_file = get_customer_ad_features_path(args.customer, args.platform)
    else:
        # Use default location when customer is None (explicit files provided)
        output_file = Path("datasets/ad_features.csv")
        logger.info("No customer specified, using default output path: %s", output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(output_file, index=False)
    logger.info("Enriched data saved to: %s", output_file)

    # Aggregate to adset level if not disabled
    if not args.no_aggregate:
        logger.info("=" * 70)
        logger.info("Aggregating to Adset-Level")
        logger.info("=" * 70)
        adset_df = _aggregate_ad_to_adset(enriched_df)
        # Determine output file path
        if args.customer:
            adset_output_file = get_customer_adset_features_path(
                args.customer, args.platform
            )
        else:
            # Use default location when customer is None
            adset_output_file = Path("datasets/adset_features.csv")
        adset_output_file.parent.mkdir(parents=True, exist_ok=True)
        adset_df.to_csv(adset_output_file, index=False)
        logger.info("Adset-level features saved to: %s", adset_output_file)

        # Integrate Shopify data if available
        if args.customer:
            adset_df = _integrate_shopify_data(
                adset_df, args.customer, args.platform
            )
            # Save enriched features
            adset_df.to_csv(adset_output_file, index=False)
            logger.info(
                "Shopify-enriched adset features saved to: %s", adset_output_file
            )

    return enriched_df


def _print_summary(enriched_df):
    """Print summary of enriched data"""
    logger.info("=" * 70)
    logger.info("ENRICHED DATA SUMMARY")
    logger.info("=" * 70)

    logger.info("Total rows: %d", len(enriched_df))
    logger.info("Total columns: %d", len(enriched_df.columns))

    prefixes_to_exclude = ["adset_", "campaign_", "account_"]
    ad_cols = [
        c
        for c in enriched_df.columns
        if not any(c.startswith(prefix) for prefix in prefixes_to_exclude)
    ]
    adset_cols = [c for c in enriched_df.columns if c.startswith("adset_")]
    campaign_cols = [c for c in enriched_df.columns if c.startswith("campaign_")]
    account_cols = [c for c in enriched_df.columns if c.startswith("account_")]

    engineered_cols = [
        c
        for c in enriched_df.columns
        if any(
            c.startswith(prefix)
            for prefix in [
                "day_of_",
                "week_of_",
                "is_weekend",
                "days_since_",
                "revenue_per_",
                "cost_per_impression",
                "engagement_rate",
                "reach_efficiency",
                "frequency_efficiency",
                "_interaction",
                "expected_",
                "_rolling_",
                "budget_",
            ]
        )
    ]
    normalized_cols = [c for c in enriched_df.columns if c.endswith("_norm")]
    bucketed_cols = [c for c in enriched_df.columns if c.endswith("_bucket")]

    logger.info("Column breakdown:")
    logger.info("  Ad-level columns: %d", len(ad_cols))
    logger.info("  Adset-level columns: %d", len(adset_cols))
    logger.info("  Campaign-level columns: %d", len(campaign_cols))
    logger.info("  Account-level columns: %d", len(account_cols))
    logger.info("  Engineered features: %d", len(engineered_cols))
    logger.info("  Normalized features: %d", len(normalized_cols))
    logger.info("  Bucketed features: %d", len(bucketed_cols))

    logger.info("Sample enriched data (first 5 rows, key columns):")
    key_cols = [
        "ad_id",
        "ad_name",
        "adset_id",
        "adset_name",
        "campaign_id",
        "campaign_name",
        "spend",
        "adset_spend",
        "campaign_spend",
        "account_spend",
        "purchase_roas",
        "adset_roas",
        "campaign_roas",
    ]
    available_cols = [c for c in key_cols if c in enriched_df.columns]
    logger.info("\n%s", enriched_df[available_cols].head().to_string())

    logger.info("=" * 70)
    logger.info("Feature extraction complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
