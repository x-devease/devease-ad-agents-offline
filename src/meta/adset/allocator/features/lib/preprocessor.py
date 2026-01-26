"""
Feature preprocessing utilities.
Includes normalization, bucketing, additional feature engineering,
data parsing, and validation.
"""

import numpy as np
import pandas as pd
from ..utils.constants import (
    WEEKEND_DAY_THRESHOLD,
    LIFECYCLE_COLD_START_DAYS,
    LIFECYCLE_EARLY_LEARNING_DAYS,
    LIFECYCLE_LEARNING_DAYS,
    ROLLING_WINDOW_DAYS,
    ROLLING_MIN_PERIODS,
    PERCENTAGE_MULTIPLIER,
    CTR_TO_DECIMAL_DIVISOR,
)


class Preprocessor:
    """Preprocesses features with normalization, bucketing, and engineering."""

    @staticmethod
    def preprocess_features(
        df: pd.DataFrame,
        normalize: bool = True,
        bucket: bool = True,
        engineer: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all preprocessing steps.

        Args:
            df: Input DataFrame
            normalize: Whether to normalize features
            bucket: Whether to create bucketed features
            engineer: Whether to engineer additional features

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        if engineer:
            df = Preprocessor.engineer_features(df)

        if normalize:
            df = Preprocessor.normalize_features(df)

        if bucket:
            df = Preprocessor.bucket_features(df)

        return df

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        # Time-based features
        if "date_start" in df.columns:
            df = Preprocessor._add_time_features(df)

        # Lifecycle features (per ad)
        df = Preprocessor._add_lifecycle_features(df)

        # Performance efficiency features
        df = Preprocessor._add_efficiency_features(df)

        # Interaction features
        df = Preprocessor._add_interaction_features(df)

        # Statistical features (rolling windows)
        df = Preprocessor._add_statistical_features(df)

        # Budget efficiency features
        df = Preprocessor._add_budget_efficiency_features(df)

        return df

    @staticmethod
    def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if "date_start" not in df.columns:
            return df

        df["date_start"] = pd.to_datetime(df["date_start"])

        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = df["date_start"].dt.dayofweek

        # Day of month
        df["day_of_month"] = df["date_start"].dt.day

        # Week of year
        df["week_of_year"] = df["date_start"].dt.isocalendar().week

        # Is weekend
        df["is_weekend"] = (df["day_of_week"] >= WEEKEND_DAY_THRESHOLD).astype(int)

        # Days since start (relative to min date in dataset)
        if len(df) > 0:
            min_date = df["date_start"].min()
            df["days_since_dataset_start"] = (df["date_start"] - min_date).dt.days
            # Alias for compatibility (some code uses days_since_start)
            df["days_since_start"] = df["days_since_dataset_start"]

        return df

    @staticmethod
    def _add_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add lifecycle features per ad and per adset.

        P0-3: Fixed to calculate days_since_start correctly for adsets.
        """
        # Per-ad lifecycle features
        if "ad_id" in df.columns and "date_start" in df.columns:
            # Ensure sorted
            df = df.sort_values(["ad_id", "date_start"])

            # Days since ad start
            df["ad_start_date"] = df.groupby("ad_id")["date_start"].transform("min")
            df["days_since_ad_start"] = (df["date_start"] - df["ad_start_date"]).dt.days

            # Lifecycle stage
            def get_lifecycle_stage(days):
                if days <= LIFECYCLE_COLD_START_DAYS:
                    return "cold_start"
                if days <= LIFECYCLE_EARLY_LEARNING_DAYS:
                    return "early_learning"
                if days <= LIFECYCLE_LEARNING_DAYS:
                    return "learning"
                return "established"

            df["lifecycle_stage"] = df["days_since_ad_start"].apply(get_lifecycle_stage)

        # P0-3: Per-adset lifecycle features
        # If adset_start_time is available, calculate days since adset start
        if "adset_id" in df.columns and "date_start" in df.columns:
            # Ensure sorted
            df = df.sort_values(["adset_id", "date_start"])

            # Check if adset_start_time is available
            if "adset_start_time" in df.columns:
                # Use actual adset start time
                # Ensure both are timezone-naive for consistent subtraction
                df["adset_start_time_dt"] = pd.to_datetime(
                    df["adset_start_time"]
                ).dt.tz_localize(None)
                df["days_since_adset_start"] = (
                    df["date_start"] - df["adset_start_time_dt"]
                ).dt.days
                # For backward compatibility, also update days_since_start
                # but only if it was previously set to days_since_dataset_start
                if "days_since_start" in df.columns:
                    # Check if it's the dataset-level calculation (all same value)
                    if df["days_since_start"].nunique() == 1:
                        df["days_since_start"] = df["days_since_adset_start"]
            else:
                # Fallback: use per-adset minimum date as start time
                df["adset_start_date"] = df.groupby("adset_id")["date_start"].transform(
                    "min"
                )
                df["days_since_adset_start"] = (
                    df["date_start"] - df["adset_start_date"]
                ).dt.days
                # Update days_since_start for adsets
                if "days_since_start" in df.columns:
                    # Check if it's the dataset-level calculation
                    if df["days_since_start"].nunique() == 1:
                        df["days_since_start"] = df["days_since_adset_start"]

        return df

    @staticmethod
    def _add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add efficiency and performance features."""
        # Revenue per impression = revenue / impressions
        # Use revenue directly (from purchase action_value columns),
        # not spend * roas.
        if "revenue" in df.columns and "impressions" in df.columns:
            df["revenue_per_impression"] = Preprocessor._safe_divide(
                df["revenue"], df["impressions"]
            )
        # Fallback: calculate from spend * purchase_roas
        # if revenue not available
        elif "impressions" in df.columns and "spend" in df.columns:
            purchase_roas = (
                df["purchase_roas"].fillna(0)
                if "purchase_roas" in df.columns
                else pd.Series(0, index=df.index)
            )
            df["revenue_per_impression"] = Preprocessor._safe_divide(
                df["spend"] * purchase_roas, df["impressions"]
            )

        # Revenue per click = revenue / clicks
        # Use revenue directly (from purchase action_value columns),
        # not spend * roas.
        if "revenue" in df.columns and "clicks" in df.columns:
            df["revenue_per_click"] = Preprocessor._safe_divide(
                df["revenue"], df["clicks"]
            )
        # Fallback: calculate from spend * purchase_roas if revenue
        # not available
        elif "clicks" in df.columns and "spend" in df.columns:
            purchase_roas = (
                df["purchase_roas"].fillna(0)
                if "purchase_roas" in df.columns
                else pd.Series(0, index=df.index)
            )
            df["revenue_per_click"] = Preprocessor._safe_divide(
                df["spend"] * purchase_roas, df["clicks"]
            )

        # Cost efficiency (spend per impression)
        if "impressions" in df.columns:
            df["cost_per_impression"] = Preprocessor._safe_divide(
                df["spend"], df["impressions"]
            )

        # Engagement rate (clicks per impression)
        if "clicks" in df.columns and "impressions" in df.columns:
            df["engagement_rate"] = (
                Preprocessor._safe_divide(df["clicks"], df["impressions"])
                * PERCENTAGE_MULTIPLIER
            )

        # Reach efficiency (impressions per reach)
        if "reach" in df.columns and "impressions" in df.columns:
            df["reach_efficiency"] = Preprocessor._safe_divide(
                df["impressions"], df["reach"]
            )

        # Frequency efficiency (impressions per reach / frequency)
        if "frequency" in df.columns and "reach" in df.columns:
            df["frequency_efficiency"] = Preprocessor._safe_divide(
                df["reach"] * df["frequency"], df.get("impressions", 1)
            )

        return df

    @staticmethod
    def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key metrics."""
        # ROAS * Spend (efficiency score)
        if "purchase_roas" in df.columns and "spend" in df.columns:
            df["roas_spend_interaction"] = df["purchase_roas"].fillna(0) * df["spend"]

        # CTR * CPC (engagement cost efficiency)
        if "ctr" in df.columns and "cpc" in df.columns:
            df["ctr_cpc_interaction"] = df["ctr"].fillna(0) * df["cpc"].fillna(0)

        # Impressions * CTR (expected clicks)
        if "impressions" in df.columns and "ctr" in df.columns:
            df["expected_clicks"] = (
                df["impressions"] * df["ctr"].fillna(0) / CTR_TO_DECIMAL_DIVISOR
            )

        # Spend * ROAS (expected revenue)
        if "spend" in df.columns and "purchase_roas" in df.columns:
            df["expected_revenue"] = df["spend"] * df["purchase_roas"].fillna(0)

        return df

    @staticmethod
    def _add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features (rolling windows, trends, EMA)."""
        # Group by ad_id for rolling calculations
        if "ad_id" in df.columns and "date_start" in df.columns:
            df = df.sort_values(["ad_id", "date_start"])

            # Rolling averages (7-day and 14-day windows)
            for col in ["spend", "purchase_roas", "cpc", "ctr"]:
                if col in df.columns:
                    # 7-day rolling average
                    df[f"{col}_rolling_7d"] = df.groupby("ad_id")[col].transform(
                        lambda x: x.rolling(
                            window=ROLLING_WINDOW_DAYS,
                            min_periods=ROLLING_MIN_PERIODS,
                        ).mean()
                    )

                    # 14-day rolling average
                    df[f"{col}_rolling_14d"] = df.groupby("ad_id")[col].transform(
                        lambda x: x.rolling(
                            window=14,
                            min_periods=ROLLING_MIN_PERIODS,
                        ).mean()
                    )

                    # Trend (difference from rolling average)
                    df[f"{col}_trend_7d"] = df.groupby("ad_id")[
                        f"{col}_rolling_7d"
                    ].diff()

                    # EMA (Exponential Moving Average) - 7-day and 14-day
                    df[f"{col}_ema_7d"] = df.groupby("ad_id")[col].transform(
                        lambda x: x.ewm(
                            span=7,
                            min_periods=ROLLING_MIN_PERIODS,
                            adjust=False,
                        ).mean()
                    )

                    df[f"{col}_ema_14d"] = df.groupby("ad_id")[col].transform(
                        lambda x: x.ewm(
                            span=14,
                            min_periods=ROLLING_MIN_PERIODS,
                            adjust=False,
                        ).mean()
                    )

            # Rolling standard deviation (7-day and 14-day windows)
            for col in ["spend", "purchase_roas"]:
                if col in df.columns:
                    # 7-day rolling std
                    df[f"{col}_rolling_7d_std"] = df.groupby("ad_id")[col].transform(
                        lambda x: x.rolling(
                            window=ROLLING_WINDOW_DAYS,
                            min_periods=ROLLING_MIN_PERIODS,
                        ).std()
                    )

                    # 14-day rolling std
                    df[f"{col}_rolling_14d_std"] = df.groupby("ad_id")[col].transform(
                        lambda x: x.rolling(
                            window=14,
                            min_periods=ROLLING_MIN_PERIODS,
                        ).std()
                    )

                    # Rolling median (7-day) for robustness
                    df[f"{col}_rolling_7d_median"] = df.groupby("ad_id")[col].transform(
                        lambda x: x.rolling(
                            window=ROLLING_WINDOW_DAYS,
                            min_periods=ROLLING_MIN_PERIODS,
                        ).median()
                    )

        return df

    @staticmethod
    def _add_budget_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add budget-related efficiency features."""
        # Budget utilization rate
        if "adset_daily_budget" in df.columns and "spend" in df.columns:
            df["budget_utilization_rate"] = (
                Preprocessor._safe_divide(df["spend"], df["adset_daily_budget"])
                * PERCENTAGE_MULTIPLIER
            )

        # Budget headroom (remaining budget)
        if "adset_daily_budget" in df.columns and "spend" in df.columns:
            df["budget_headroom"] = (df["adset_daily_budget"] - df["spend"]).clip(
                lower=0
            )

        # Budget efficiency (ROAS per budget dollar)
        if "adset_daily_budget" in df.columns and "purchase_roas" in df.columns:
            df["budget_roas_efficiency"] = df["purchase_roas"].fillna(0) / df[
                "adset_daily_budget"
            ].replace(0, np.nan)

        return df

    @staticmethod
    def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply z-score normalization to numeric features.

        Standardizes features by removing mean and scaling to unit variance.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with z-score normalized numeric features
        """
        df = df.copy()

        # Columns to normalize (exclude IDs, dates, and already normalized)
        exclude_cols = [
            "ad_id",
            "adset_id",
            "campaign_id",
            "account_id",
            "date_start",
            "date_stop",
            "export_date",
            "creative_id",
            "creative_image_hash",
            "creative_link_image_hash",
        ]

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        normalize_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Remove columns that are already normalized
        # (contain '_norm' or '_normalized')
        normalize_cols = [
            col
            for col in normalize_cols
            if "_norm" not in col and "_normalized" not in col
        ]

        # Z-score normalization
        for col in normalize_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()

                if std > 0 and pd.notna(mean) and pd.notna(std):
                    df[f"{col}_norm"] = (df[col] - mean) / std
                else:
                    df[f"{col}_norm"] = 0

        return df

    @staticmethod
    def bucket_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create bucketed (binned) versions of continuous features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with bucketed features
        """
        df = df.copy()

        # Features to bucket
        bucket_configs = {
            "spend": [0, 10, 50, 100, 500, 1000, float("inf")],
            "purchase_roas": [0, 1, 2, 3, 5, 10, float("inf")],
            "cpc": [0, 0.5, 1, 2, 5, 10, float("inf")],
            "cpm": [0, 5, 10, 20, 50, 100, float("inf")],
            "ctr": [0, 0.5, 1, 2, 5, 10, float("inf")],
            "impressions": [0, 1000, 5000, 10000, 50000, 100000, float("inf")],
            "clicks": [0, 10, 50, 100, 500, 1000, float("inf")],
            "adset_daily_budget": [0, 50, 100, 500, 1000, 5000, float("inf")],
        }

        # Create buckets
        for col, bins in bucket_configs.items():
            if col in df.columns:
                # Create labels
                labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
                labels[-1] = labels[-1].replace("-inf", "+")

                # Bucket the feature
                df[f"{col}_bucket"] = pd.cut(
                    df[col].fillna(0),
                    bins=bins,
                    labels=labels,
                    include_lowest=True,
                    duplicates="drop",
                )

                # Also create numeric bucket (0-indexed)
                df[f"{col}_bucket_num"] = pd.cut(
                    df[col].fillna(0),
                    bins=bins,
                    labels=False,
                    include_lowest=True,
                    duplicates="drop",
                )

        # Bucket ratios and percentages
        ratio_bucket_configs = {
            "roas_vs_adset": [0, 0.5, 0.8, 1.0, 1.2, 1.5, float("inf")],
            "roas_vs_campaign": [0, 0.5, 0.8, 1.0, 1.2, 1.5, float("inf")],
            "ad_share_of_adset_budget": [
                0,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                float("inf"),
            ],
            "adset_budget_utilization": [0, 50, 75, 90, 100, 110, float("inf")],
        }

        for col, bins in ratio_bucket_configs.items():
            if col in df.columns:
                labels = [
                    f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)
                ]
                labels[-1] = labels[-1].replace("-inf", "+")

                df[f"{col}_bucket"] = pd.cut(
                    df[col].fillna(0),
                    bins=bins,
                    labels=labels,
                    include_lowest=True,
                    duplicates="drop",
                )

        return df

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safely divide two series, handling inf and NaN."""
        result = numerator / denominator.replace(0, np.nan)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def check_data_quality(df: pd.DataFrame, data_type: str) -> None:
        """
        Check data quality for a given data type.

        Validates that the DataFrame has required columns and basic data
        quality.

        Args:
            df: DataFrame to validate
            data_type: Type of data
                ('account', 'campaign', 'adset', 'ad')

        Raises:
            ValueError: If data quality checks fail
        """
        if df.empty:
            raise ValueError(f"{data_type} data is empty")

        # Check for required ID column based on data type
        required_id_column = f"{data_type}_id" if data_type != "ad" else "ad_id"
        if required_id_column not in df.columns:
            raise ValueError(f"{data_type} data missing '{required_id_column}' column")

        # Check that ID column has no null values
        if df[required_id_column].isna().any():
            raise ValueError(
                f"{data_type} data has null values in " f"'{required_id_column}' column"
            )

    @staticmethod
    def normalize_numeric_columns(df: pd.DataFrame, columns: tuple) -> pd.DataFrame:
        """
        Normalize numeric columns by converting to numeric type and handling
        errors.

        This is a basic normalization step for loading raw data. For full
        normalization (z-score scaling), use normalize_features().

        Args:
            df: DataFrame to normalize
            columns: Tuple of column names to normalize

        Returns:
            DataFrame with normalized numeric columns
        """
        df = df.copy()

        for col in columns:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def validate_join_keys(df: pd.DataFrame, join_keys: list, level_name: str) -> None:
        """
        Validate that required join keys exist in the DataFrame.

        Args:
            df: DataFrame to validate
            join_keys: List of required column names for joining
            level_name: Name of the data level (for error messages)

        Raises:
            ValueError: If any join keys are missing
        """
        missing_keys = [key for key in join_keys if key not in df.columns]
        if missing_keys:
            keys_str = ", ".join(missing_keys)
            raise ValueError(
                f"{level_name} data missing required join keys: " f"{keys_str}"
            )
