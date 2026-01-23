"""Shopify data integration module.

This module provides functionality to load, process, and extract features
from Shopify order data CSV files for integration into budget allocation decisions.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ShopifyDataLoader:
    """Load and process Shopify order data from CSV exports."""

    def __init__(self, shopify_csv_path: str):
        """Initialize Shopify data loader.

        Args:
            shopify_csv_path: Path to Shopify orders export CSV file.
        """
        self.shopify_csv_path = Path(shopify_csv_path)

    def load_orders(self) -> pd.DataFrame:
        """Load Shopify orders from CSV file.

        Returns:
            DataFrame with processed order data.
        """
        if not self.shopify_csv_path.exists():
            logger.warning(f"Shopify CSV not found: {self.shopify_csv_path}")
            return pd.DataFrame()

        logger.info(f"Loading Shopify orders from: {self.shopify_csv_path}")

        # Load raw data
        df = pd.read_csv(self.shopify_csv_path)

        # Parse date columns
        date_columns = ["Paid at", "Created at", "Fulfilled at"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Parse numeric columns
        numeric_columns = ["Subtotal", "Shipping", "Taxes", "Total", "Discount Amount"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter to paid orders only
        if "Financial Status" in df.columns:
            df = df[df["Financial Status"] == "paid"]

        # Filter out cancelled orders
        if "Cancelled at" in df.columns:
            df = df[df["Cancelled at"].isna()]

        logger.info(f"Loaded {len(df)} paid Shopify orders")

        return df

    def aggregate_daily_metrics(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate Shopify metrics by date.

        Args:
            df: Shopify orders DataFrame.

        Returns:
            DataFrame with daily aggregated metrics.
        """
        if df.empty or "Paid at" not in df.columns:
            return pd.DataFrame()

        # Extract date from paid_at
        df = df.copy()
        df["date"] = df["Paid at"].dt.date

        # Aggregate by date
        daily_metrics = (
            df.groupby("date")
            .agg(
                {
                    "Id": "count",  # Order count
                    "Total": "sum",  # Total revenue
                    "Subtotal": "sum",  # Product revenue
                    "Discount Amount": "sum",  # Total discounts
                }
            )
            .reset_index()
            .rename(
                columns={
                    "Id": "order_count",
                    "Total": "total_revenue",
                    "Subtotal": "product_revenue",
                    "Discount Amount": "total_discount",
                }
            )
        )

        # Calculate additional metrics
        daily_metrics["avg_order_value"] = (
            daily_metrics["total_revenue"] / daily_metrics["order_count"]
        )

        logger.info(
            f"Aggregated {len(daily_metrics)} days of Shopify metrics "
            f"(from {daily_metrics['date'].min()} to {daily_metrics['date'].max()})"
        )

        return daily_metrics

    def extract_customer_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract customer-level features from Shopify data.

        Args:
            df: Shopify orders DataFrame.

        Returns:
            DataFrame with customer-level metrics.
        """
        if df.empty or "Email" not in df.columns:
            return pd.DataFrame()

        # Clean email for customer matching
        df = df.copy()
        df["customer_email"] = df["Email"].str.lower().str.strip()

        # Aggregate by customer
        customer_metrics = (
            df.groupby("customer_email")
            .agg(
                {
                    "Id": "count",  # Order count
                    "Total": ["sum", "mean"],  # Total and average revenue
                }
            )
            .reset_index()
        )
        customer_metrics.columns = ["customer_email", "order_count", "total_revenue", "avg_order_value"]

        # Calculate customer lifetime value (CLV) proxy
        customer_metrics["clv"] = customer_metrics["total_revenue"]

        logger.info(f"Extracted features for {len(customer_metrics)} unique customers")

        return customer_metrics

    def calculate_roas_from_shopify(
        self,
        shopify_df: pd.DataFrame,
        meta_spend_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate ROAS using Shopify revenue and Meta spend data.

        Args:
            shopify_df: Shopify orders DataFrame (daily aggregated).
            meta_spend_df: Meta ads spend DataFrame with date and spend columns.

        Returns:
            DataFrame with calculated ROAS metrics.
        """
        if shopify_df.empty or meta_spend_df.empty:
            return pd.DataFrame()

        # Ensure date columns are datetime
        if "date" not in shopify_df.columns:
            return pd.DataFrame()

        shopify_daily = shopify_df.copy()
        if not isinstance(shopify_daily["date"], pd.DatetimeIndex):
            shopify_daily["date"] = pd.to_datetime(shopify_daily["date"])

        # Prepare Meta spend data
        meta_daily = meta_spend_df.copy()
        if "date_start" in meta_daily.columns:
            meta_daily["date"] = pd.to_datetime(meta_daily["date_start"]).dt.date
            meta_daily = meta_daily.groupby("date")["spend"].sum().reset_index()
            meta_daily["date"] = pd.to_datetime(meta_daily["date"])

        # Merge Shopify revenue with Meta spend
        merged = pd.merge(
            shopify_daily[["date", "total_revenue"]],
            meta_daily[["date", "spend"]],
            on="date",
            how="outer",
        )

        # Fill NaN values
        merged["total_revenue"] = merged["total_revenue"].fillna(0)
        merged["spend"] = merged["spend"].fillna(0)

        # Calculate ROAS
        merged["shopify_roas"] = (
            merged["total_revenue"] / merged["spend"]
        ).replace([float("inf"), -float("inf")], 0)

        # Only include days where spend > 0
        merged = merged[merged["spend"] > 0]

        logger.info(
            f"Calculated Shopify ROAS for {len(merged)} days "
            f"(avg: {merged['shopify_roas'].mean():.2f})"
        )

        return merged


def get_shopify_data_path(customer: str, platform: str = "meta") -> Path:
    """Get path to Shopify CSV file for a customer.

    Args:
        customer: Customer name.
        platform: Platform name (default: "meta").

    Returns:
        Path to Shopify CSV file.
    """
    from src.config.path_manager import get_path_manager

    path_manager = get_path_manager(customer, platform)
    return path_manager.raw_data_dir() / "shopify.csv"
