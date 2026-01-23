"""
Data loading utilities for feature extraction.
Handles file loading and initial normalization.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from ..utils.constants import (
    DEFAULT_DATA_DIR,
    STANDARD_NUMERIC_COLUMNS,
    BASE_SUM_COLUMNS,
)
from ..utils.file_discovery import FileDiscovery
from ..utils.meta_data_processor import MetaDataProcessor
from ..utils.revenue_utils import calculate_revenue_from_purchase_actions
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class Loader:
    """Handles loading and initial processing of data files."""

    # Numeric columns to normalize when loading raw data
    NUMERIC_COLUMNS = STANDARD_NUMERIC_COLUMNS

    @staticmethod
    def load_all_data(
        data_dir: str = DEFAULT_DATA_DIR,
        account_file: Optional[str] = None,
        campaign_file: Optional[str] = None,
        adset_file: Optional[str] = None,
        ad_file: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all data files from datasets directory.

        Args:
            data_dir: Directory containing data files
            account_file: Optional explicit account file path
            campaign_file: Optional explicit campaign file path
            adset_file: Optional explicit adset file path
            ad_file: Optional explicit ad file path

        Returns:
            Dictionary with keys: 'account', 'campaign', 'adset', 'ad'
        """
        # Discover files
        files = FileDiscovery.discover_data_files(
            data_dir, account_file, campaign_file, adset_file, ad_file
        )

        data = {}

        # Load each data type
        for data_type, file_path in files.items():
            if file_path and Path(file_path).exists():
                data[data_type] = Loader.load_and_normalize(file_path, data_type)
            else:
                logger.warning("%s data file not found", data_type.capitalize())

        return data

    @staticmethod
    def load_and_normalize(file_path: str, data_type: str) -> pd.DataFrame:
        """
        Load a CSV file and normalize numeric columns.

        Automatically processes raw Meta Ads CSV files by flattening
        JSON columns if needed (actions, action_values,
        cost_per_action_type, adset_targeting).

        Args:
            file_path: Path to CSV file
            data_type: Type of data ('account', 'campaign', 'adset', 'ad')

        Returns:
            Normalized DataFrame
        """
        # Check if file needs Meta Ads processing (has JSON columns)
        df_sample = pd.read_csv(file_path, nrows=1)
        json_cols = [
            "actions",
            "action_values",
            "cost_per_action_type",
            "adset_targeting",
        ]
        needs_processing = any(col in df_sample.columns for col in json_cols)

        if needs_processing:
            logger.info("Detected raw Meta Ads format, processing JSON columns...")
            df = MetaDataProcessor.process_meta_csv(file_path, data_type)
        else:
            df = pd.read_csv(file_path)

        # Validate basic structure
        if "date_start" not in df.columns:
            raise ValueError(f"{data_type} data missing 'date_start' column")

        # Parse date
        df["date_start"] = pd.to_datetime(df["date_start"])

        # Aggregate hourly data to daily if needed
        df = Loader._aggregate_hourly_to_daily(df, data_type)

        # Validate data quality
        Preprocessor.check_data_quality(df, data_type)

        # Normalize numeric columns
        df = Preprocessor.normalize_numeric_columns(df, Loader.NUMERIC_COLUMNS)

        logger.info("Loaded %s data: %d rows", data_type, len(df))
        return df

    @staticmethod
    def _aggregate_hourly_to_daily(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Aggregate hourly data to daily if multiple rows exist per day.

        Args:
            df: Input DataFrame
            data_type: Type of data ('account', 'campaign', 'adset', 'ad')

        Returns:
            Aggregated DataFrame (daily level)
        """
        # Normalize date_start to just the date (remove hour component)
        df["date_start"] = df["date_start"].dt.normalize()

        # Check if aggregation is needed
        id_col, groupby_cols, max_group_size = Loader._check_aggregation_needed(
            df, data_type
        )
        if max_group_size <= 1:
            return df

        logger.info(
            "Detected hourly data (max %d rows per %s/date). "
            "Aggregating to daily...",
            max_group_size,
            id_col,
        )

        # Build aggregation configuration and perform aggregation
        agg_dict = Loader._build_aggregation_dict(df, groupby_cols, id_col, data_type)
        df_agg = df.groupby(groupby_cols).agg(agg_dict).reset_index()

        # Recalculate derived metrics after aggregation
        df_agg = Loader._recalculate_derived_metrics(df_agg)

        logger.info(
            "Aggregated from %d to %d rows (daily level)",
            len(df),
            len(df_agg),
        )

        return df_agg

    @staticmethod
    def _check_aggregation_needed(
        df: pd.DataFrame, data_type: str
    ) -> tuple[Optional[str], list, int]:
        """
        Check if aggregation is needed and return aggregation parameters.

        Returns:
            Tuple of (id_col, groupby_cols, max_group_size)
        """
        # Handle empty DataFrame
        if len(df) == 0:
            return None, [], 1

        # Determine ID column based on data type
        id_col = Loader._get_id_column(data_type)
        if not id_col or id_col not in df.columns:
            Loader._handle_missing_id_column(df)
            return None, [], 1

        # Check if we have multiple rows per (id, date_start)
        groupby_cols = [id_col, "date_start"]
        group_sizes = df.groupby(groupby_cols).size()

        # Handle empty groupby result
        if len(group_sizes) == 0:
            return id_col, groupby_cols, 1

        # Get max group size - convert to list to avoid pandas version issues
        max_group_size = max(group_sizes.tolist()) if len(group_sizes) > 0 else 1

        return id_col, groupby_cols, max_group_size

    @staticmethod
    def _build_aggregation_dict(
        df: pd.DataFrame,
        groupby_cols: list,
        id_col: str,
        data_type: str,
    ) -> dict:
        """Build aggregation dictionary for groupby operation."""
        sum_cols = Loader._get_sum_columns(df)
        first_cols = Loader._get_first_columns(df, id_col, data_type)
        other_cols = Loader._get_other_columns(df, groupby_cols, sum_cols, first_cols)
        return Loader._build_agg_dict(sum_cols, first_cols, other_cols, groupby_cols)

    @staticmethod
    def _get_id_column(data_type: str) -> Optional[str]:
        """Get the ID column name for a given data type."""
        id_mapping = {
            "account": "account_id",
            "campaign": "campaign_id",
            "adset": "adset_id",
            "ad": "ad_id",
        }
        return id_mapping.get(data_type)

    @staticmethod
    def _handle_missing_id_column(df: pd.DataFrame) -> pd.DataFrame:
        """Handle case when ID column is missing."""
        if len(df) > 0:
            group_sizes = df.groupby("date_start").size()
            if len(group_sizes) > 0:
                max_size = max(group_sizes.tolist()) if len(group_sizes) > 0 else 1
                if max_size > 1:
                    logger.warning(
                        "Multiple rows per date_start but no ID column found. "
                        "Skipping aggregation."
                    )
        return df

    @staticmethod
    def _get_sum_columns(df: pd.DataFrame) -> list:
        """Get list of columns to sum during aggregation."""
        sum_cols = list(BASE_SUM_COLUMNS)

        # Add all action-related columns (flattened from JSON)
        action_cols = [
            col
            for col in df.columns
            if col.startswith("action_") or col.startswith("action_value_")
        ]
        sum_cols.extend(action_cols)

        # Add all cost_per_action columns
        cost_cols = [col for col in df.columns if col.startswith("cost_per_action_")]
        sum_cols.extend(cost_cols)

        # Filter to existing columns only and ensure they're numeric
        return [
            col
            for col in sum_cols
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

    @staticmethod
    def _get_first_columns(df: pd.DataFrame, id_col: str, data_type: str) -> list:
        """Get list of columns to take first value during aggregation."""
        first_cols = [id_col, "date_start", "date_stop", "export_date"]

        # Add name columns if they exist
        name_col = f"{data_type}_name"
        if name_col in df.columns:
            first_cols.append(name_col)

        # Add account/campaign/adset identifiers if they exist
        for parent_id in ["account_id", "campaign_id", "adset_id"]:
            if parent_id in df.columns and parent_id not in first_cols:
                first_cols.append(parent_id)

        # Filter to existing columns only
        return [col for col in first_cols if col in df.columns]

    @staticmethod
    def _get_other_columns(
        df: pd.DataFrame, groupby_cols: list, sum_cols: list, first_cols: list
    ) -> list:
        """Get all other columns not in groupby, sum, or first lists."""
        return [
            col for col in df.columns if col not in groupby_cols + sum_cols + first_cols
        ]

    @staticmethod
    def _build_agg_dict(
        sum_cols: list, first_cols: list, other_cols: list, groupby_cols: list
    ) -> dict:
        """Build aggregation dictionary for groupby operation."""
        agg_dict = {}
        for col in sum_cols:
            agg_dict[col] = "sum"
        for col in first_cols + other_cols:
            if col not in groupby_cols:
                agg_dict[col] = "first"
        return agg_dict

    @staticmethod
    def _recalculate_derived_metrics(df_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Recalculate derived metrics after aggregation.

        This ensures all derived metrics (CPC, CPM, CTR, purchase_roas) are
        calculated from aggregated values, and revenue is recalculated from
        purchase action_value columns (authoritative source).
        """
        # Recalculate revenue from purchase action_value columns if they exist
        # This ensures revenue is always from authoritative source after
        # aggregation, even if hourly-level revenue was calculated incorrectly
        purchase_action_value_cols = [
            col
            for col in df_agg.columns
            if "action_value" in col.lower() and "purchase" in col.lower()
        ]

        if purchase_action_value_cols:
            # Recalculate revenue from summed purchase action_value columns
            # This overrides any revenue that was summed (which might be
            # wrong if hourly-level revenue was from purchase_roas * spend)
            calculate_revenue_from_purchase_actions(
                df_agg, update_inplace=True, custom_logger=logger
            )
            numeric_cols = [
                col
                for col in df_agg.columns
                if "action_value" in col.lower()
                and "purchase" in col.lower()
                and pd.api.types.is_numeric_dtype(df_agg[col])
            ]
            logger.info(
                "Recalculated revenue from %d purchase action_value "
                "columns after hourly-to-daily aggregation "
                "(authoritative source)",
                len(numeric_cols),
            )

        # Recalculate derived metrics from aggregated values
        if "spend" in df_agg.columns and "impressions" in df_agg.columns:
            df_agg["cpm"] = (
                df_agg["spend"] / df_agg["impressions"].replace(0, pd.NA) * 1000
            ).replace([float("inf"), -float("inf")], pd.NA)

        if "spend" in df_agg.columns and "clicks" in df_agg.columns:
            df_agg["cpc"] = (
                df_agg["spend"] / df_agg["clicks"].replace(0, pd.NA)
            ).replace([float("inf"), -float("inf")], pd.NA)

        if "clicks" in df_agg.columns and "impressions" in df_agg.columns:
            df_agg["ctr"] = (
                df_agg["clicks"] / df_agg["impressions"].replace(0, pd.NA) * 100
            ).replace([float("inf"), -float("inf")], pd.NA)

        if "revenue" in df_agg.columns and "spend" in df_agg.columns:
            revenue = pd.to_numeric(df_agg["revenue"], errors="coerce")
            spend = pd.to_numeric(df_agg["spend"], errors="coerce").replace(0, pd.NA)
            calculated_roas = (revenue / spend).replace(
                [float("inf"), -float("inf")], pd.NA
            )
            # Replace zero ROAS with NaN to avoid zero values
            calculated_roas = calculated_roas.replace(0, pd.NA)

            # Only update purchase_roas where calculated_roas > 0
            if "purchase_roas" in df_agg.columns:
                existing_roas = pd.to_numeric(df_agg["purchase_roas"], errors="coerce")
                # Use calculated_roas where it's valid (> 0)
                mask = calculated_roas.notna() & (calculated_roas > 0)
                df_agg.loc[mask, "purchase_roas"] = calculated_roas[mask]
                # If existing is zero or NaN, use calculated
                zero_or_na_mask = (
                    existing_roas.isna() | (existing_roas == 0)
                ) & calculated_roas.notna()
                df_agg.loc[zero_or_na_mask, "purchase_roas"] = calculated_roas[
                    zero_or_na_mask
                ]
            else:
                df_agg["purchase_roas"] = calculated_roas

        return df_agg
