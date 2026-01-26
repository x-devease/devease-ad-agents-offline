"""
Meta Ads data preprocessing utilities.

This module handles flattening of JSON columns and formatting of raw Meta Ads
CSV exports to match the format expected by extract.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from .file_discovery import FileDiscovery
from .json_parser import JSONParser

logger = logging.getLogger(__name__)


class MetaDataProcessor:
    """Processes raw Meta Ads CSV exports for feature extraction."""

    @staticmethod
    def process_meta_csv(
        file_path: str,
        data_type: str,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Process a raw Meta Ads CSV file by flattening JSON columns.

        Args:
            file_path: Path to raw Meta CSV file
            data_type: Type of data ('account', 'campaign', 'adset', 'ad')
            output_path: Optional path to save processed CSV

        Returns:
            Processed DataFrame with flattened columns
        """
        logger.info("Processing %s data from: %s", data_type, file_path)
        df = pd.read_csv(file_path)

        # Flatten JSON columns
        df = MetaDataProcessor._flatten_json_columns(df, data_type)

        # Map column names if needed
        df = MetaDataProcessor._map_column_names(df, data_type)

        # Ensure required columns exist
        df = MetaDataProcessor._ensure_required_columns(df, data_type)

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info("Processed data saved to: %s", output_path)

        return df

    @staticmethod
    def _normalize_roas_flattened(
        flattened: pd.DataFrame, col_name: str
    ) -> pd.DataFrame:
        """Normalize flattened ROAS columns to a single column."""
        num_cols = len(flattened.columns)
        if col_name == "purchase_roas":
            if num_cols == 1:
                old_col = flattened.columns[0]
                flattened = flattened.rename(columns={old_col: "purchase_roas"})
                logger.info("  Renamed %s -> purchase_roas", old_col)
            elif num_cols > 1:
                purchase_roas_sum = flattened.sum(axis=1)
                flattened = pd.DataFrame({"purchase_roas": purchase_roas_sum})
                logger.info("  Combined %d ROAS columns into purchase_roas", num_cols)
        elif col_name in ["website_purchase_roas", "mobile_app_purchase_roas"]:
            if num_cols == 1:
                old_col = flattened.columns[0]
                flattened = flattened.rename(columns={old_col: col_name})
                logger.info("  Renamed %s -> %s", old_col, col_name)
            elif num_cols > 1:
                roas_sum = flattened.sum(axis=1)
                flattened = pd.DataFrame({col_name: roas_sum})
                logger.info("  Combined %d ROAS columns into %s", num_cols, col_name)
        return flattened

    @staticmethod
    def _add_flattened_columns(
        df: pd.DataFrame, flattened: pd.DataFrame, col_name: str
    ) -> pd.DataFrame:
        """Add flattened columns to DataFrame, handling special cases."""
        if col_name == "adset_targeting":
            # Keep original column, add flattened columns with prefix
            flattened.columns = [f"targeting_{col}" for col in flattened.columns]
            df = pd.concat([df, flattened], axis=1)
        else:
            # Drop original column and add flattened columns
            if col_name in df.columns:
                df = df.drop(columns=[col_name])
            df = pd.concat([df, flattened], axis=1)
        logger.info(
            "  Added %d flattened columns from %s",
            len(flattened.columns),
            col_name,
        )
        return df

    @staticmethod
    def _ensure_purchase_roas_exists(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure purchase_roas column exists.

        Creates from alternatives if needed.
        """
        if "purchase_roas" in df.columns:
            return df

        # Look for flattened ROAS columns that might be purchase_roas
        roas_cols = [
            c
            for c in df.columns
            if "roas" in c.lower() and ("omni" in c.lower() or "purchase" in c.lower())
        ]
        if roas_cols:
            df = df.rename(columns={roas_cols[0]: "purchase_roas"})
            logger.info("  Created purchase_roas from %s", roas_cols[0])
            return df

        # Look for any ROAS column
        all_roas = [c for c in df.columns if "roas" in c.lower()]
        if all_roas:
            df = df.rename(columns={all_roas[0]: "purchase_roas"})
            logger.info("  Created purchase_roas from %s", all_roas[0])
        return df

    @staticmethod
    def _flatten_json_columns(df: pd.DataFrame, _data_type: str) -> pd.DataFrame:
        """Flatten JSON columns in the DataFrame."""
        df = df.copy()

        # Columns that may contain JSON and need flattening
        json_columns = {
            "actions": MetaDataProcessor._flatten_actions,
            "action_values": MetaDataProcessor._flatten_action_values,
            "cost_per_action_type": MetaDataProcessor._flatten_cost_per_action,
            "adset_targeting": MetaDataProcessor._flatten_targeting,
            "purchase_roas": MetaDataProcessor._flatten_roas,
            "website_purchase_roas": MetaDataProcessor._flatten_roas,
            "mobile_app_purchase_roas": MetaDataProcessor._flatten_roas,
        }

        roas_cols = [
            "purchase_roas",
            "website_purchase_roas",
            "mobile_app_purchase_roas",
        ]

        for col_name, flatten_func in json_columns.items():
            if col_name not in df.columns:
                continue

            logger.info("Flattening column: %s", col_name)
            flattened = flatten_func(df[col_name])
            if flattened.empty:
                continue

            # Handle ROAS columns specially
            if col_name in roas_cols:
                flattened = MetaDataProcessor._normalize_roas_flattened(
                    flattened, col_name
                )

            df = MetaDataProcessor._add_flattened_columns(df, flattened, col_name)

        # Ensure purchase_roas exists if it was in the original JSON columns
        # but wasn't successfully created during flattening
        original_json_cols = set(json_columns.keys())
        has_roas_in_json = "purchase_roas" in original_json_cols
        if has_roas_in_json:
            df = MetaDataProcessor._ensure_purchase_roas_exists(df)

        return df

    @staticmethod
    def _parse_json_value(val):
        """
        Parse JSON value, handling string, list, dict, or other types.

        Args:
            val: Value to parse (string, list, dict, None, etc.)

        Returns:
            Parsed value (list or dict), or empty list if parsing fails
        """
        if isinstance(val, str):
            # Handle empty strings
            if val.strip() == "":
                return []
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                # Return empty list for invalid JSON strings
                return []
        if isinstance(val, (list, dict)):
            return val
        # Return empty list for None, NaN, or other types
        return []

    @staticmethod
    def _flatten_action_list(actions, prefix, default_value=0):
        """
        Flatten a list of action dictionaries into a result dict.

        If multiple actions have the same action_type, their values are summed.
        This handles cases where Meta Ads exports multiple entries for the same
        action type (e.g., multiple purchase events in the same row).
        """
        result = {}
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict):
                    action_type = action.get("action_type", "")
                    value = action.get("value", default_value)
                    col_name = f"{prefix}_{action_type}"
                    # Sum values if action_type already exists
                    # (handles duplicates)
                    if col_name in result:
                        result[col_name] = result[col_name] + value
                    else:
                        result[col_name] = value
        return result

    @staticmethod
    def _flatten_actions(series: pd.Series) -> pd.DataFrame:
        """
        Flatten actions column from JSON format.

        Meta Ads exports actions as JSON array of objects like:
        [{'action_type': 'link_click', 'value': 123}, ...]

        Returns DataFrame with columns like:
        action_link_click, action_add_to_cart, etc.
        """
        result_dicts = []

        for idx, val in series.items():
            result = {}
            if pd.notna(val) and val != "":
                try:
                    actions = MetaDataProcessor._parse_json_value(val)
                    result = MetaDataProcessor._flatten_action_list(
                        actions, "action", default_value=0
                    )
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    logger.debug("Error parsing actions at row %d: %s", idx, error)
                    result = {}

            result_dicts.append(result)

        if result_dicts:
            return pd.DataFrame(result_dicts, index=series.index).fillna(0)
        return pd.DataFrame(index=series.index)

    @staticmethod
    def _flatten_action_values(series: pd.Series) -> pd.DataFrame:
        """
        Flatten action_values column from JSON format.

        Meta Ads exports action_values as JSON array of objects like:
        [{'action_type': 'purchase', 'value': 456.78}, ...]

        Returns DataFrame with columns like: action_value_purchase, etc.
        """
        result_dicts = []

        for idx, val in series.items():
            result = {}
            if pd.notna(val) and val != "":
                try:
                    action_values = MetaDataProcessor._parse_json_value(val)
                    result = MetaDataProcessor._flatten_action_list(
                        action_values, "action_value", default_value=0.0
                    )
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    logger.debug(
                        "Error parsing action_values at row %d: %s", idx, error
                    )
                    result = {}

            result_dicts.append(result)

        if result_dicts:
            return pd.DataFrame(result_dicts, index=series.index).fillna(0.0)
        return pd.DataFrame(index=series.index)

    @staticmethod
    def _flatten_cost_per_action(series: pd.Series) -> pd.DataFrame:
        """
        Flatten cost_per_action_type column from JSON format.

        Meta Ads exports cost_per_action_type as JSON array of objects like:
        [{'action_type': 'link_click', 'value': 0.45}, ...]

        Returns DataFrame with columns like: cost_per_action_link_click, etc.
        """
        result_dicts = []

        for idx, val in series.items():
            result = {}
            if pd.notna(val) and val != "":
                try:
                    cost_per_actions = MetaDataProcessor._parse_json_value(val)
                    result = MetaDataProcessor._flatten_action_list(
                        cost_per_actions, "cost_per_action", default_value=0.0
                    )
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    logger.debug(
                        "Error parsing cost_per_action_type at row %d: %s",
                        idx,
                        error,
                    )
                    result = {}

            result_dicts.append(result)

        if result_dicts:
            return pd.DataFrame(result_dicts, index=series.index).fillna(0.0)
        return pd.DataFrame(index=series.index)

    @staticmethod
    def _flatten_targeting(series: pd.Series) -> pd.DataFrame:
        """
        Flatten adset_targeting column from JSON format.

        Uses JSONParser to extract targeting features.
        """
        return JSONParser.extract_targeting_features(series)

    @staticmethod
    def _flatten_roas(series: pd.Series) -> pd.DataFrame:
        """
        Flatten ROAS columns from JSON format.

        Flattens purchase_roas, website_purchase_roas,
        mobile_app_purchase_roas from JSON format.

        Meta Ads exports ROAS columns as JSON array of objects like:
        [{'action_type': 'omni_purchase', 'value': 2.5}, ...]
        or
        [{'action_type': 'offsite_conversion.fb_pixel_purchase',
          'value': 1.8}, ...]

        Returns DataFrame with columns like:
        roas_omni_purchase, roas_offsite_conversion.fb_pixel_purchase, etc.
        """
        prefix = "roas"
        default_value = 0.0
        result_dicts = []

        for idx, val in series.items():
            result = {}
            if pd.notna(val) and val != "":
                try:
                    roas_data = MetaDataProcessor._parse_json_value(val)
                    result = MetaDataProcessor._flatten_action_list(
                        roas_data, prefix, default_value
                    )
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    logger.debug("Error parsing ROAS at row %d: %s", idx, error)
                    result = {}

            result_dicts.append(result)

        if result_dicts:
            return pd.DataFrame(result_dicts, index=series.index).fillna(default_value)
        return pd.DataFrame(index=series.index)

    @staticmethod
    def _extract_value_from_list(parsed_list):
        """Extract and sum values from a list of dicts."""
        total_value = 0.0
        for item in parsed_list:
            if isinstance(item, dict):
                value = item.get("value", 0)
                try:
                    total_value += float(value)
                except (ValueError, TypeError):
                    pass
        return total_value if total_value > 0 else None

    @staticmethod
    def _extract_value_from_dict(parsed_dict):
        """Extract value from a single dict."""
        value = parsed_dict.get("value", 0)
        try:
            parsed_value = float(value)
            return parsed_value if parsed_value > 0 else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_single_roas_value(val):
        """Parse a single ROAS value from various formats."""
        if not (pd.notna(val) and val != ""):
            return None

        # Try parsing as JSON first
        try:
            if isinstance(val, str):
                parsed = json.loads(val)
            else:
                parsed = val

            if isinstance(parsed, list):
                return MetaDataProcessor._extract_value_from_list(parsed)
            if isinstance(parsed, dict):
                return MetaDataProcessor._extract_value_from_dict(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try direct numeric conversion
        try:
            parsed_value = float(val)
            return parsed_value if parsed_value > 0 else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_roas_json(series: pd.Series) -> pd.Series:
        """
        Parse JSON string in ROAS columns and extract numeric values.

        Handles both JSON strings and already-parsed JSON objects.
        Only extracts non-zero values to avoid zero ROAS issues.
        """
        result = pd.Series(index=series.index, dtype=float)

        for idx, val in series.items():
            parsed_value = MetaDataProcessor._parse_single_roas_value(val)
            if parsed_value is not None:
                result[idx] = parsed_value

        return result

    @staticmethod
    def _process_roas_column(df: pd.DataFrame, col_name: str) -> tuple:
        """Process a single ROAS column, parsing JSON if needed."""
        if col_name not in df.columns:
            return df, False

        col = df[col_name]
        # Check for string dtype (object or StringDtype in newer pandas)
        if col.dtype == "object" or pd.api.types.is_string_dtype(col):
            parsed = MetaDataProcessor._parse_roas_json(col)
            df[col_name] = parsed
            logger.info(
                "Parsed %s JSON, non-zero count: %d", col_name, (parsed > 0).sum()
            )
            return df, True

        if pd.api.types.is_numeric_dtype(col):
            numeric_col = pd.to_numeric(col, errors="coerce")
            df[col_name] = numeric_col
            logger.info(
                "%s already numeric, non-zero count: %d",
                col_name,
                (numeric_col > 0).sum(),
            )
            return df, True

        return df, False

    @staticmethod
    def _combine_roas_columns(df: pd.DataFrame, roas_columns: list) -> pd.DataFrame:
        """Combine multiple ROAS columns into purchase_roas."""
        if not roas_columns:
            return df

        # Sum all ROAS columns, replacing 0 with NaN to avoid zero ROAS
        combined_roas = pd.Series(0.0, index=df.index)
        for col_name in roas_columns:
            col_values = df[col_name].fillna(0)
            combined_roas = combined_roas + col_values
        combined_roas = combined_roas.replace(0, np.nan)

        if "purchase_roas" not in df.columns:
            df["purchase_roas"] = combined_roas
        else:
            existing_roas = pd.to_numeric(df["purchase_roas"], errors="coerce")
            df["purchase_roas"] = combined_roas.fillna(existing_roas)

        logger.info(
            "Combined %d ROAS columns into purchase_roas, non-zero count: %d",
            len(roas_columns),
            (df["purchase_roas"] > 0).sum(),
        )
        return df

    @staticmethod
    def _apply_fallback_column_mappings(df: pd.DataFrame) -> pd.DataFrame:
        """Apply fallback column mappings when no JSON ROAS columns exist."""
        column_mappings = {
            "website_purchase_roas": "purchase_roas",
            "website_purchase_roas_value": "purchase_roas",
            "mobile_app_purchase_roas": "purchase_roas",
        }

        for old_name, new_name in column_mappings.items():
            if old_name not in df.columns or not new_name:
                continue

            if new_name not in df.columns:
                df[new_name] = df[old_name]
            else:
                # Prefer the one with more non-NaN values
                old_numeric = pd.to_numeric(df[old_name], errors="coerce")
                new_numeric = pd.to_numeric(df[new_name], errors="coerce")
                if old_numeric.notna().sum() > new_numeric.notna().sum():
                    df[new_name] = df[old_name].fillna(df[new_name])

        return df

    @staticmethod
    def _apply_column_mappings(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column name mappings to standardize column names.

        Parses JSON ROAS columns (mobile_app_purchase_roas,
        website_purchase_roas) and combines them into purchase_roas,
        avoiding zero values.
        """
        df = df.copy()

        # Parse and combine ROAS columns from different sources
        roas_columns = []

        # Process mobile_app_purchase_roas
        df, processed = MetaDataProcessor._process_roas_column(
            df, "mobile_app_purchase_roas"
        )
        if processed:
            roas_columns.append("mobile_app_purchase_roas")

        # Process website_purchase_roas
        df, processed = MetaDataProcessor._process_roas_column(
            df, "website_purchase_roas"
        )
        if processed:
            roas_columns.append("website_purchase_roas")

        # Combine ROAS columns into purchase_roas
        if roas_columns:
            df = MetaDataProcessor._combine_roas_columns(df, roas_columns)
        else:
            df = MetaDataProcessor._apply_fallback_column_mappings(df)

        return df

    @staticmethod
    def _calc_revenue_from_actions(df: pd.DataFrame) -> tuple:
        """
        Calculate revenue from purchase action_value columns.

        Returns:
            Tuple of (df, revenue_from_action_values bool)
        """
        purchase_action_value_cols = [
            col
            for col in df.columns
            if "action_value" in col.lower() and "purchase" in col.lower()
        ]

        if not purchase_action_value_cols:
            return df, False

        # Sum all purchase action_value columns to get total revenue
        # pylint: disable=duplicate-code
        purchase_values = pd.DataFrame()
        for col in purchase_action_value_cols:
            purchase_values[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        numeric_cols = [
            col
            for col in purchase_values.columns
            if pd.api.types.is_numeric_dtype(purchase_values[col])
        ]
        # pylint: enable=duplicate-code
        if numeric_cols:
            df["revenue"] = purchase_values[numeric_cols].sum(axis=1)
            logger.info(
                "Calculated revenue from %d purchase action_value columns",
                len(numeric_cols),
            )
            return df, True
        return df, False

    @staticmethod
    def _calculate_revenue_fallback(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate revenue from purchase_roas * spend as fallback.

        Only uses purchase_roas * spend when purchase_roas > 0 to avoid
        creating zero revenue when we have valid ROAS data.
        """
        has_required = (
            "revenue" not in df.columns
            and "purchase_roas" in df.columns
            and "spend" in df.columns
        )
        if not has_required:
            return df

        purchase_roas = pd.to_numeric(df["purchase_roas"], errors="coerce")
        spend = pd.to_numeric(df["spend"], errors="coerce")

        # Only calculate revenue when purchase_roas > 0 to avoid zero ROAS
        # This ensures we don't create zero revenue when ROAS is valid
        mask = (purchase_roas > 0) & spend.notna() & (spend > 0)
        df["revenue"] = np.nan
        df.loc[mask, "revenue"] = (purchase_roas[mask] * spend[mask]).replace(
            [np.inf, -np.inf], np.nan
        )

        logger.info(
            "Calculated revenue from purchase_roas * spend "
            "(no purchase action_value columns found). "
            "Only calculated for %d rows where purchase_roas > 0 "
            "(avoiding zero ROAS).",
            mask.sum(),
        )
        return df

    @staticmethod
    def _calculate_purchase_roas(
        df: pd.DataFrame, revenue_from_action_values: bool
    ) -> pd.DataFrame:
        """Calculate purchase_roas from revenue/spend."""
        if "revenue" not in df.columns or "spend" not in df.columns:
            return df

        revenue = pd.to_numeric(df["revenue"], errors="coerce")
        spend = pd.to_numeric(df["spend"], errors="coerce").replace(0, np.nan)
        calculated_roas = (revenue / spend).replace([np.inf, -np.inf], np.nan)

        if revenue_from_action_values:
            df = MetaDataProcessor._update_roas_authoritative(
                df, calculated_roas, revenue, spend
            )
        else:
            df = MetaDataProcessor._update_roas_fallback(
                df, calculated_roas, revenue, spend
            )

        df["purchase_roas"] = pd.to_numeric(df["purchase_roas"], errors="coerce")
        return df

    @staticmethod
    def _update_roas_authoritative(
        df: pd.DataFrame,
        calculated_roas: pd.Series,
        revenue: pd.Series,
        spend: pd.Series,
    ) -> pd.DataFrame:
        """Update purchase_roas when revenue is from action values."""
        if "purchase_roas" not in df.columns:
            df["purchase_roas"] = calculated_roas
            logger.info(
                "Calculated purchase_roas from revenue/spend "
                "(revenue from purchase action_value columns)"
            )
        else:
            mask = revenue.notna() & (revenue >= 0) & spend.notna() & (spend > 0)
            if mask.any():
                df.loc[mask, "purchase_roas"] = calculated_roas[mask]
                logger.info(
                    "Recalculated purchase_roas from revenue/spend "
                    "for %d rows (revenue from purchase action_value "
                    "columns, authoritative)",
                    mask.sum(),
                )
        return df

    @staticmethod
    def _update_roas_fallback(
        df: pd.DataFrame,
        calculated_roas: pd.Series,
        revenue: pd.Series,
        spend: pd.Series,
    ) -> pd.DataFrame:
        """
        Update purchase_roas when revenue is from fallback.

        Only updates when calculated_roas > 0 to avoid zero ROAS.
        """
        if "purchase_roas" in df.columns:
            roas_is_missing = df["purchase_roas"].isna() | (df["purchase_roas"] == 0)
            # Only update when calculated_roas > 0 to avoid zero ROAS
            mask = (
                roas_is_missing
                & revenue.notna()
                & (revenue > 0)
                & spend.notna()
                & (spend > 0)
                & calculated_roas.notna()
                & (calculated_roas > 0)  # Avoid zero ROAS
            )
            if mask.any():
                df.loc[mask, "purchase_roas"] = calculated_roas[mask]
                logger.info(
                    "Recalculated purchase_roas from revenue/spend "
                    "for %d rows (fallback: purchase_roas was missing "
                    "or zero, only updated where calculated_roas > 0)",
                    mask.sum(),
                )
        else:
            # Only set purchase_roas where calculated_roas > 0
            df["purchase_roas"] = calculated_roas
            # Replace zero with NaN to avoid zero ROAS
            df["purchase_roas"] = df["purchase_roas"].replace(0, np.nan)
            logger.info(
                "Calculated purchase_roas from revenue/spend "
                "(purchase_roas column missing, zero values set to NaN)"
            )
        return df

    @staticmethod
    def _calculate_standard_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard metrics: cpc, cpm, ctr."""
        if "spend" in df.columns and "clicks" in df.columns:
            if "cpc" not in df.columns:
                df["cpc"] = (
                    pd.to_numeric(df["spend"], errors="coerce")
                    / pd.to_numeric(df["clicks"], errors="coerce").replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)

        if "spend" in df.columns and "impressions" in df.columns:
            if "cpm" not in df.columns:
                df["cpm"] = (
                    pd.to_numeric(df["spend"], errors="coerce")
                    / pd.to_numeric(df["impressions"], errors="coerce").replace(
                        0, np.nan
                    )
                    * 1000
                ).replace([np.inf, -np.inf], np.nan)

        if "clicks" in df.columns and "impressions" in df.columns:
            if "ctr" not in df.columns:
                df["ctr"] = (
                    pd.to_numeric(df["clicks"], errors="coerce")
                    / pd.to_numeric(df["impressions"], errors="coerce").replace(
                        0, np.nan
                    )
                    * 100
                ).replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def _map_column_names(df: pd.DataFrame, _data_type: str) -> pd.DataFrame:
        """
        Map column names to expected format.

        Handles common column name variations in Meta Ads exports.
        Ensures all required columns for extract.py exist.
        """
        df = df.copy()

        # Apply column mappings
        df = MetaDataProcessor._apply_column_mappings(df)

        # Calculate revenue from purchase action_value columns
        df, revenue_from_action_values = MetaDataProcessor._calc_revenue_from_actions(
            df
        )

        # Fallback: Calculate revenue from purchase_roas * spend
        if not revenue_from_action_values:
            df = MetaDataProcessor._calculate_revenue_fallback(df)

        # Calculate purchase_roas from revenue/spend
        df = MetaDataProcessor._calculate_purchase_roas(df, revenue_from_action_values)

        # Calculate standard metrics
        df = MetaDataProcessor._calculate_standard_metrics(df)

        return df

    @staticmethod
    def _ensure_required_columns(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Ensure required columns exist for the data type.

        Adds missing columns with default values if needed.
        """
        df = df.copy()

        # Required columns by data type
        required_columns = {
            "account": ["account_id", "date_start"],
            "campaign": ["campaign_id", "date_start"],
            "adset": ["adset_id", "date_start", "campaign_id"],
            "ad": ["ad_id", "date_start", "adset_id"],
        }

        req_cols = required_columns.get(data_type, [])
        for col in req_cols:
            if col not in df.columns:
                logger.warning("Missing required column: %s (filling with NaN)", col)
                df[col] = np.nan

        # Ensure date_start is datetime
        if "date_start" in df.columns:
            df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")

        return df

    @staticmethod
    def process_all_meta_files(
        data_dir: str,
        output_dir: Optional[str] = None,
        _file_patterns: Optional[Dict[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all Meta Ads CSV files in a directory.

        Automatically discovers and processes all raw Meta CSV files.

        Args:
            data_dir: Directory containing raw Meta CSV files
            output_dir: Optional directory to save processed files
            _file_patterns: Optional dict mapping data types to
                filename patterns

        Returns:
            Dictionary with processed DataFrames by data type
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Discover files
        files = FileDiscovery.discover_data_files(data_dir)

        processed = {}

        for data_type, file_path in files.items():
            if file_path and Path(file_path).exists():
                df = MetaDataProcessor.process_meta_csv(file_path, data_type)

                if output_dir:
                    output_file = output_path / Path(file_path).name.replace(
                        ".csv", "_processed.csv"
                    )
                    df.to_csv(output_file, index=False)
                    logger.info(
                        "Saved processed %s data to: %s",
                        data_type,
                        output_file,
                    )

                processed[data_type] = df
            else:
                logger.warning("%s data file not found", data_type.capitalize())

        return processed
