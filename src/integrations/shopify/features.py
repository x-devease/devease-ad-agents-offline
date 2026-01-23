"""Shopify feature extraction module.

Extracts Shopify-specific features for integration with budget allocation.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .loader import ShopifyDataLoader

logger = logging.getLogger(__name__)


class ShopifyFeatureExtractor:
    """Extract Shopify features for budget allocation integration."""

    def __init__(self, shopify_csv_path: str):
        """Initialize Shopify feature extractor.

        Args:
            shopify_csv_path: Path to Shopify orders CSV file.
        """
        self.loader = ShopifyDataLoader(shopify_csv_path)
        self.df_orders = None
        self.df_daily = None
        self.df_roas = None

    def load_and_process(self) -> bool:
        """Load and process Shopify data.

        Returns:
            True if data loaded successfully, False otherwise.
        """
        try:
            # Load raw orders
            self.df_orders = self.loader.load_orders()
            if self.df_orders.empty:
                return False

            # Aggregate daily metrics
            self.df_daily = self.loader.aggregate_daily_metrics(
                self.df_orders
            )

            return True
        except Exception as e:
            logger.error(f"Failed to load Shopify data: {e}")
            return False

    def get_daily_revenue_features(self, days_back: int = 30) -> pd.DataFrame:
        """Get daily revenue features for recent days.

        Args:
            days_back: Number of recent days to return.

        Returns:
            DataFrame with daily revenue features.
        """
        if self.df_daily is None or self.df_daily.empty:
            return pd.DataFrame()

        # Get recent data
        cutoff_date = pd.Timestamp(datetime.now().date()) - timedelta(days=days_back)

        # Ensure date column is Timestamp type for comparison
        df_copy = self.df_daily.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])

        recent = df_copy[df_copy["date"] >= cutoff_date].copy()

        # Sort by date descending
        recent = recent.sort_values("date", ascending=False)

        return recent

    def calculate_shopify_roas(
        self, meta_spend_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate ROAS using Shopify revenue.

        Args:
            meta_spend_df: Meta ads spend DataFrame with date_start and spend.

        Returns:
            DataFrame with date and shopify_roas columns.
        """
        if self.df_daily is None or self.df_daily.empty:
            return pd.DataFrame()

        if meta_spend_df.empty:
            return pd.DataFrame()

        self.df_roas = self.loader.calculate_roas_from_shopify(
            self.df_daily, meta_spend_df
        )

        return self.df_roas

    def get_recent_shopify_roas(
        self, meta_spend_df: pd.DataFrame, days_back: int = 7
    ) -> float:
        """Get average Shopify ROAS for recent days.

        Args:
            meta_spend_df: Meta ads spend DataFrame.
            days_back: Number of recent days to average.

        Returns:
            Average ROAS for recent period. Returns 0 if no data available.
        """
        if self.df_roas is None or self.df_roas.empty:
            # Calculate ROAS first
            self.calculate_shopify_roas(meta_spend_df)

        if self.df_roas is None or self.df_roas.empty:
            return 0.0

        # Get recent data
        cutoff_date = pd.Timestamp(datetime.now().date()) - timedelta(days=days_back)
        recent = self.df_roas[self.df_roas["date"] >= cutoff_date]

        if recent.empty:
            return 0.0

        avg_roas = recent["shopify_roas"].mean()
        return float(avg_roas) if not pd.isna(avg_roas) else 0.0

    def enrich_adset_features(
        self, adset_features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Enrich adset features with Shopify metrics.

        Adds Shopify-based ROAS and revenue metrics to adset features.

        Args:
            adset_features_df: Adset features DataFrame with date_start.

        Returns:
            Enriched adset features DataFrame.
        """
        if adset_features_df.empty:
            return adset_features_df

        # Load ROAS data if not already loaded
        if self.df_roas is None or self.df_roas.empty:
            # We need Meta spend data to calculate ROAS
            # For now, return unchanged features
            logger.warning(
                "Shopify ROAS not calculated (need Meta spend data). "
                "Use calculate_shopify_roas() first."
            )
            return adset_features_df

        # Merge Shopify ROAS by date
        enriched = adset_features_df.copy()

        # Ensure date columns match
        if "date_start" in enriched.columns:
            enriched["merge_date"] = pd.to_datetime(
                enriched["date_start"]
            ).dt.date
        else:
            enriched["merge_date"] = enriched.index

        # Merge with Shopify ROAS data
        self.df_roas["merge_date"] = pd.to_datetime(
            self.df_roas["date"]
        ).dt.date

        enriched = enriched.merge(
            self.df_roas[["merge_date", "shopify_roas", "total_revenue"]],
            on="merge_date",
            how="left",
        )

        # Fill NaN values
        enriched["shopify_roas"] = enriched["shopify_roas"].fillna(0)
        enriched["shopify_revenue"] = enriched["total_revenue"].fillna(0)

        # Drop merge date column
        if "merge_date" in enriched.columns:
            enriched = enriched.drop(columns=["merge_date"])

        logger.info(
            f"Enriched {len(enriched)} adset features with Shopify metrics"
        )

        return enriched


def extract_shopify_features_for_allocation(
    shopify_csv_path: str,
    meta_spend_df: pd.DataFrame,
    adset_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract Shopify features for budget allocation.

    Convenience function to load Shopify data and enrich adset features.

    Args:
        shopify_csv_path: Path to Shopify CSV file.
        meta_spend_df: Meta ads spend data.
        adset_features_df: Adset features to enrich.

    Returns:
        Enriched adset features DataFrame with Shopify metrics.
    """
    extractor = ShopifyFeatureExtractor(shopify_csv_path)

    if not extractor.load_and_process():
        logger.warning("Failed to load Shopify data, returning original features")
        return adset_features_df

    return extractor.enrich_adset_features(adset_features_df)
