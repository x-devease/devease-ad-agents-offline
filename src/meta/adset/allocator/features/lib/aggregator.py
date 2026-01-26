"""
Feature aggregation utilities.
Creates aggregated features from multi-level data.
"""

import numpy as np
import pandas as pd
from ..utils.constants import (
    LOW_ROAS_THRESHOLD,
    HIGH_ROAS_THRESHOLD,
    PERCENTAGE_MULTIPLIER,
)
from .preprocessor import Preprocessor


class Aggregator:
    """Creates aggregated features from joined multi-level data."""

    @staticmethod
    def create_aggregated_features(
        enriched_df: pd.DataFrame,
        preprocess: bool = True,
        normalize: bool = True,
        bucket: bool = True,
    ) -> pd.DataFrame:
        """
        Create aggregated features from multi-level data.

        Args:
            enriched_df: Enriched ad-level data with all levels joined
            preprocess: Whether to apply feature preprocessing
            normalize: Whether to normalize features (if preprocess=True)
            bucket: Whether to bucket features (if preprocess=True)

        Returns:
            DataFrame with additional aggregated features
        """
        df = enriched_df.copy()

        # Create different types of aggregated features
        df = Aggregator.create_budget_shares(df)
        df = Aggregator.create_performance_ratios(df)
        df = Aggregator.create_budget_utilization(df)
        df = Aggregator.create_adset_internal_signals(df)

        # Apply feature preprocessing
        if preprocess:
            df = Preprocessor.preprocess_features(
                df, normalize=normalize, bucket=bucket, engineer=True
            )

        return df

    @staticmethod
    def create_adset_internal_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create adset-level signals from internal ad performance.

        Args:
            df: Enriched DataFrame

        Returns:
            DataFrame with internal adset signals
        """
        if "adset_id" not in df.columns or "purchase_roas" not in df.columns:
            return df

        # Group by adset and date to get internal distribution
        # Note: purchase_roas here is the ad-level ROAS

        # 1. Count ads per adset-day
        ad_counts = df.groupby(["adset_id", "date_start"])["ad_id"].transform("count")
        df["adset_num_active_ads"] = ad_counts

        # 2. Percentage of ads with low/high ROAS
        # We need a temporary series to calculate this
        df["is_low_roas"] = (df["purchase_roas"] < LOW_ROAS_THRESHOLD).astype(int)
        df["is_high_roas"] = (df["purchase_roas"] > HIGH_ROAS_THRESHOLD).astype(int)

        df["adset_pct_ads_low_roas"] = (
            df.groupby(["adset_id", "date_start"])["is_low_roas"].transform("mean")
            * PERCENTAGE_MULTIPLIER
        )
        df["adset_pct_ads_high_roas"] = (
            df.groupby(["adset_id", "date_start"])["is_high_roas"].transform("mean")
            * PERCENTAGE_MULTIPLIER
        )

        # Cleanup temp columns
        df = df.drop(columns=["is_low_roas", "is_high_roas"])

        return df

    @staticmethod
    def create_budget_shares(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create budget share features.

        Args:
            df: Enriched DataFrame

        Returns:
            DataFrame with budget share features
        """
        # Ad share of adset budget
        if "adset_daily_budget" in df.columns:
            df["ad_share_of_adset_budget"] = Aggregator.safe_divide(
                df["spend"], df["adset_daily_budget"]
            )

        # Ad share of campaign budget
        if "campaign_daily_budget" in df.columns:
            df["ad_share_of_campaign_budget"] = Aggregator.safe_divide(
                df["spend"], df["campaign_daily_budget"]
            )

        return df

    @staticmethod
    def create_performance_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create performance ratio features.

        Args:
            df: Enriched DataFrame

        Returns:
            DataFrame with performance ratio features
        """
        # Performance ratios (ad vs adset)
        if "adset_roas" in df.columns:
            df["roas_vs_adset"] = Aggregator.safe_divide(
                df["purchase_roas"], df["adset_roas"]
            )

        if "adset_cpc" in df.columns:
            df["cpc_vs_adset"] = Aggregator.safe_divide(df["cpc"], df["adset_cpc"])

        # Performance ratios (ad vs campaign)
        if "campaign_roas" in df.columns:
            df["roas_vs_campaign"] = Aggregator.safe_divide(
                df["purchase_roas"], df["campaign_roas"]
            )

        # Performance ratios (ad vs account)
        if "account_roas" in df.columns:
            df["roas_vs_account"] = Aggregator.safe_divide(
                df["purchase_roas"], df["account_roas"]
            )

        return df

    @staticmethod
    def create_budget_utilization(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create budget utilization features.

        Args:
            df: Enriched DataFrame

        Returns:
            DataFrame with budget utilization features
        """
        # Adset budget utilization = (adset_spend / adset_daily_budget) * 100
        if "adset_daily_budget" in df.columns:
            df["adset_budget_utilization"] = (
                Aggregator.safe_divide(df["adset_spend"], df["adset_daily_budget"])
                * PERCENTAGE_MULTIPLIER
            )

        # Campaign budget utilization
        if "campaign_daily_budget" in df.columns:
            df["campaign_budget_utilization"] = (
                Aggregator.safe_divide(
                    df["campaign_spend"], df["campaign_daily_budget"]
                )
                * PERCENTAGE_MULTIPLIER
            )

        return df

    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """
        Safely divide two series, handling inf and NaN.

        Args:
            numerator: Numerator series
            denominator: Denominator series

        Returns:
            Division result with inf/NaN handled
        """
        result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan)
