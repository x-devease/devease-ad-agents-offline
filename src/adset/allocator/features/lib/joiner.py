"""
Feature joining utilities.
Handles joining account, campaign, and adset data into ad-level data.
"""

import logging
from typing import Optional

import pandas as pd
from ..core.extractor import Extractor
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class Joiner:
    """Handles joining features from multiple data levels."""

    # Join keys for each level
    ADSET_JOIN_KEYS = ["adset_id", "date_start"]
    CAMPAIGN_JOIN_KEYS = ["campaign_id", "date_start"]
    ACCOUNT_JOIN_KEYS = ["account_id", "date_start"]

    @staticmethod
    def join_all_levels(
        ad_df: pd.DataFrame,
        account_df: Optional[pd.DataFrame] = None,
        campaign_df: Optional[pd.DataFrame] = None,
        adset_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Join account, campaign, and adset data into ad-level data.

        Args:
            ad_df: Ad-level daily data (base)
            account_df: Account-level daily data (optional)
            campaign_df: Campaign-level daily data (optional)
            adset_df: Adset-level daily data (optional)

        Returns:
            Enriched ad-level DataFrame with features from all levels
        """
        logger.info("=" * 70)
        logger.info("FEATURE EXTRACTION: Joining Multi-Level Data")
        logger.info("=" * 70)

        # Validate base data
        Preprocessor.validate_join_keys(ad_df, ["adset_id", "date_start"], "ad")

        # Keep ALL ad-level columns
        enriched_df = ad_df.copy()
        original_ad_columns = list(enriched_df.columns)
        logger.info(
            "Starting with ad-level data: %d rows, %d columns",
            len(enriched_df),
            len(original_ad_columns),
        )

        # Join adset data (most granular after ad)
        if adset_df is not None:
            enriched_df = Joiner.join_level(
                enriched_df,
                adset_df,
                "adset",
                Joiner.ADSET_JOIN_KEYS,
                Extractor.extract_adset_features,
            )

        # Join campaign data
        if campaign_df is not None:
            enriched_df = Joiner.join_level(
                enriched_df,
                campaign_df,
                "campaign",
                Joiner.CAMPAIGN_JOIN_KEYS,
                Extractor.extract_campaign_features,
            )

        # Join account data
        if account_df is not None:
            enriched_df = Joiner.join_level(
                enriched_df,
                account_df,
                "account",
                Joiner.ACCOUNT_JOIN_KEYS,
                Extractor.extract_account_features,
            )

        logger.info("Feature extraction complete")
        logger.info(
            "Final enriched data: %d rows, %d columns",
            len(enriched_df),
            len(enriched_df.columns),
        )
        preserved_count = len(
            [c for c in original_ad_columns if c in enriched_df.columns]
        )
        logger.info("Preserved %d ad-level columns", preserved_count)

        return enriched_df

    @staticmethod
    def join_level(
        base_df: pd.DataFrame,
        level_df: pd.DataFrame,
        level_name: str,
        join_keys: list,
        extractor_func,
    ) -> pd.DataFrame:
        """
        Join a single level of data.

        Args:
            base_df: Base DataFrame to join into
            level_df: Level DataFrame to join from
            level_name: Name of the level (for logging)
            join_keys: Keys to join on
            extractor_func: Function to extract features from level_df

        Returns:
            Joined DataFrame
        """
        logger.info("Joining %s-level data...", level_name)

        # Validate join keys
        Preprocessor.validate_join_keys(level_df, join_keys, level_name)

        # Extract features
        level_features = extractor_func(level_df)

        # Ensure consistent datetime types for date columns in join keys
        # This prevents merge errors when one DataFrame has datetime and another has object
        for key in join_keys:
            if "date" in key.lower():
                if key in base_df.columns:
                    base_df[key] = pd.to_datetime(base_df[key])
                if key in level_features.columns:
                    level_features[key] = pd.to_datetime(level_features[key])

        # Perform join
        enriched_df = base_df.merge(
            level_features,
            on=join_keys,
            how="left",
            suffixes=("", f"_{level_name}"),
        )

        # Log join results
        feature_col = f"{level_name}_spend"
        if feature_col in enriched_df.columns:
            matched = enriched_df[feature_col].notna().sum()
            logger.info("Joined %s data: %d rows matched", level_name, matched)

        return enriched_df
