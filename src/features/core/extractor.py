"""
Feature extraction utilities.
Extracts and transforms features from different data levels.
"""

from typing import Dict, List

import pandas as pd
from ..utils.constants import STANDARD_NUMERIC_COLUMNS
from ..utils.json_parser import JSONParser


class Extractor:
    """Extracts features from different data levels."""

    # Column mappings for each level
    # Account-level feature columns (includes standard metrics plus
    # reach/frequency)
    # Build dynamically to avoid pylint duplicate-code detection
    _ACCOUNT_BASE = ("account_id", "date_start")
    _ACCOUNT_EXTRA = ("reach", "frequency")
    ACCOUNT_COLUMNS = _ACCOUNT_BASE + STANDARD_NUMERIC_COLUMNS + _ACCOUNT_EXTRA

    ACCOUNT_RENAME_MAP = {
        "spend": "account_spend",
        "impressions": "account_impressions",
        "clicks": "account_clicks",
        "purchase_roas": "account_roas",
        "cpc": "account_cpc",
        "cpm": "account_cpm",
        "ctr": "account_ctr",
        "reach": "account_reach",
        "frequency": "account_frequency",
    }

    _CAMPAIGN_BASE = [
        "campaign_id",
        "date_start",
        "campaign_name",
    ]
    _CAMPAIGN_EXTRA = [
        "reach",
        "frequency",
        "campaign_daily_budget",
        "campaign_status",
    ]
    CAMPAIGN_COLUMNS = _CAMPAIGN_BASE + list(STANDARD_NUMERIC_COLUMNS) + _CAMPAIGN_EXTRA

    CAMPAIGN_RENAME_MAP = {
        "spend": "campaign_spend",
        "impressions": "campaign_impressions",
        "clicks": "campaign_clicks",
        "purchase_roas": "campaign_roas",
        "cpc": "campaign_cpc",
        "cpm": "campaign_cpm",
        "ctr": "campaign_ctr",
        "reach": "campaign_reach",
        "frequency": "campaign_frequency",
    }

    _ADSET_BASE = [
        "adset_id",
        "date_start",
        "adset_name",
        "campaign_id",
    ]
    _ADSET_EXTRA = [
        "reach",
        "frequency",
        "adset_daily_budget",
        "adset_status",
        "adset_bid_strategy",
        "adset_optimization_goal",
        "adset_billing_event",
        "adset_targeting",
        "adset_end_time",
        "adset_start_time",
        "adset_lifetime_budget",
    ]
    ADSET_COLUMNS = _ADSET_BASE + list(STANDARD_NUMERIC_COLUMNS) + _ADSET_EXTRA

    # Columns that contain JSON and should be parsed
    JSON_COLUMNS = ["adset_targeting"]

    ADSET_RENAME_MAP = {
        "spend": "adset_spend",
        "impressions": "adset_impressions",
        "clicks": "adset_clicks",
        "purchase_roas": "adset_roas",
        "cpc": "adset_cpc",
        "cpm": "adset_cpm",
        "ctr": "adset_ctr",
        "reach": "adset_reach",
        "frequency": "adset_frequency",
    }

    @staticmethod
    def extract_account_features(account_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract account-level features for joining.

        Args:
            account_df: Account-level daily data

        Returns:
            DataFrame with account features
        """
        return Extractor._extract_features(
            account_df,
            Extractor.ACCOUNT_COLUMNS,
            Extractor.ACCOUNT_RENAME_MAP,
        )

    @staticmethod
    def extract_campaign_features(campaign_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract campaign-level features for joining.

        Args:
            campaign_df: Campaign-level daily data

        Returns:
            DataFrame with campaign features
        """
        return Extractor._extract_features(
            campaign_df,
            Extractor.CAMPAIGN_COLUMNS,
            Extractor.CAMPAIGN_RENAME_MAP,
        )

    @staticmethod
    def extract_adset_features(adset_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract adset-level features for joining.
        Also parses nested JSON structures like adset_targeting.

        Args:
            adset_df: Adset-level daily data

        Returns:
            DataFrame with adset features (including parsed targeting)
        """
        features = Extractor._extract_features(
            adset_df,
            Extractor.ADSET_COLUMNS,
            Extractor.ADSET_RENAME_MAP,
        )

        # Parse JSON columns if they exist
        if "adset_targeting" in features.columns:
            # Extract targeting features
            targeting_features = JSONParser.extract_targeting_features(
                features["adset_targeting"]
            )

            # Add targeting features with prefix
            if not targeting_features.empty:
                targeting_features.columns = [
                    f"adset_{col}" for col in targeting_features.columns
                ]
                features = pd.concat([features, targeting_features], axis=1)

        return features

    @staticmethod
    def _extract_features(
        df: pd.DataFrame, columns: List[str], rename_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Generic feature extraction method.

        Args:
            df: Source DataFrame
            columns: Columns to select
            rename_map: Mapping for renaming columns

        Returns:
            Extracted and renamed DataFrame
        """
        # Select only columns that exist
        available_columns = [col for col in columns if col in df.columns]
        features = df[available_columns].copy()

        # Rename columns
        features = features.rename(columns=rename_map)

        return features
