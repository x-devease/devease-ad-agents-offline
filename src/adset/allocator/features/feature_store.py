"""
Feature store for audience quality scoring.
Handles feature loading, generation, and versioning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.utils import Config


class FeatureStore:
    """Manages feature loading, generation, and preprocessing."""

    def __init__(self, customer: str = "moprobo", platform: str = "meta"):
        """
        Initialize FeatureStore.

        Args:
            customer: Customer name (default: moprobo)
            platform: Platform name (default: meta)
        """
        self.customer = customer
        self.platform = platform
        self.scalers = {}
        self.encoders = {}
        self.impute_values = {}  # Store median/mode values for imputation
        self.feature_version = "1.0"
        self.feature_names = []

    def load_ad_features(self) -> pd.DataFrame:
        """Load ad-level features."""
        config = Config.get_customer_config(self.customer, self.platform)
        features_path = (
            Config.BASE_DIR / config["data_sources"]["features"]["base_path"]
        )
        ad_features_file = (
            features_path / config["data_sources"]["features"]["files"]["ad_features"]
        )
        return pd.read_csv(ad_features_file)

    def load_adset_features(self) -> pd.DataFrame:
        """Load adset-level features."""
        config = Config.get_customer_config(self.customer, self.platform)
        features_path = (
            Config.BASE_DIR / config["data_sources"]["features"]["base_path"]
        )
        adset_features_file = (
            features_path
            / config["data_sources"]["features"]["files"]["adset_features"]
        )
        return pd.read_csv(adset_features_file)

    def load_lookalike_data(self) -> pd.DataFrame:
        """Load lookalike audience data."""
        config = Config.get_customer_config(self.customer, self.platform)
        raw_path = Config.BASE_DIR / config["data_sources"]["raw"]["base_path"]
        lookalike_file = raw_path / config["data_sources"]["raw"]["files"]["lookalike"]
        if lookalike_file.exists():
            return pd.read_csv(lookalike_file)
        return pd.DataFrame()

    def load_shopify_data(self) -> pd.DataFrame:
        """Load Shopify customer data."""
        config = Config.get_customer_config(self.customer, self.platform)
        raw_path = Config.BASE_DIR / config["data_sources"]["raw"]["base_path"]
        shopify_file = raw_path / config["data_sources"]["raw"]["files"]["shopify"]
        if shopify_file.exists():
            return pd.read_csv(shopify_file)
        return pd.DataFrame()

    def load_adset_data(self) -> pd.DataFrame:
        """
        Load adset-level data for audience scoring.

        Returns:
            DataFrame with one row per adset including targeting and performance
        """
        return self.load_adset_features()

    def merge_features(
        self, ad_df: pd.DataFrame, adset_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge ad and adset features."""
        # Merge on adset_id
        merged = ad_df.merge(
            adset_df, on="adset_id", suffixes=("_ad", "_adset"), how="left"
        )

        # Drop redundant columns
        columns_to_drop = [
            col
            for col in merged.columns
            if "_ad_" in col and col.replace("_ad_", "_adset_") in merged.columns
        ]
        merged = merged.drop(columns=columns_to_drop)

        return merged

    def preprocess_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess features: handle missing values, encode categoricals, scale numericals.

        Args:
            df: Input dataframe
            fit: Whether to fit scalers/encoders (True for training, False for inference)

        Returns:
            Preprocessed feature matrix and feature names
        """
        df = df.copy()

        # Filter to valid data
        df = self._filter_valid_data(df)

        # Separate numerical and categorical features (auto-detect if not in config)
        numerical_feats = Config.NUMERICAL_FEATURES()
        categorical_feats = Config.CATEGORICAL_FEATURES()

        # Check if config features exist in dataframe
        numerical_in_df = [col for col in numerical_feats if col in df.columns]
        categorical_in_df = [col for col in categorical_feats if col in df.columns]

        # Auto-detect if config features don't match data
        if not numerical_in_df and not categorical_in_df:
            # Auto-detect numerical and categorical features
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Remove non-feature columns
            exclude_cols = [
                Config.TARGET_COLUMN(),
                "ad_id",
                "adset_id",
                "campaign_id",
                "audience_id",
                "customer_id",
                "rfm_segment",
                "date_start",
                "date_stop",
                "export_date",
                "ad_start_date",
                "adset_end_time",
                "adset_start_time",
                "ad_name",
                "adset_name",
                "campaign_name",
                "account_name",
                "account_id",
            ]

            # Also remove date columns
            date_cols = [col for col in df.columns if "date" in col.lower()]
            exclude_cols.extend(date_cols)

            numerical_features = [
                col for col in numerical_features if col not in exclude_cols
            ]
            categorical_features = [
                col for col in categorical_features if col not in exclude_cols
            ]

            print(
                f"   Auto-detected {len(numerical_features)} numerical, {len(categorical_features)} categorical features"
            )
        else:
            numerical_features = numerical_in_df
            categorical_features = categorical_in_df
            print(
                f"   Using config: {len(numerical_features)} numerical, {len(categorical_features)} categorical features"
            )

        # Ensure we have some features to work with
        if not numerical_features and not categorical_features:
            raise ValueError(
                "No features available for preprocessing. Check input data."
            )

        # Handle missing values - prevent data leakage
        if fit:
            # Calculate imputation values from this dataset only
            for col in numerical_features:
                median_val = df[col].median()
                self.impute_values[f"{col}_median"] = median_val
                df[col] = df[col].fillna(median_val)

            for col in categorical_features:
                # Use 'Unknown' as the imputation value for categoricals
                self.impute_values[f"{col}_categorical"] = "Unknown"
                df[col] = df[col].fillna("Unknown")
        else:
            # Use stored imputation values from training data
            for col in numerical_features:
                impute_key = f"{col}_median"
                if impute_key in self.impute_values:
                    df[col] = df[col].fillna(self.impute_values[impute_key])
                else:
                    # Fallback if not stored (shouldn't happen in proper usage)
                    df[col] = df[col].fillna(df[col].median())

            for col in categorical_features:
                df[col] = df[col].fillna("Unknown")

        # Encode categorical features
        for col in categorical_features:
            if fit:
                encoder = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder
            else:
                if col in self.encoders:
                    # Handle unseen categories by mapping to a fixed unknown value
                    # We use the last known class + 1, or handle via masking
                    encoder = self.encoders[col]
                    known_classes = set(encoder.classes_)

                    # Map unseen values to a sentinel (-1) that will be handled
                    def encode_value(val):
                        val_str = str(val)
                        if val_str in known_classes:
                            # Find index in classes_
                            return np.where(encoder.classes_ == val_str)[0][0]
                        else:
                            # Return 0 (first class) as fallback for unseen
                            # In production, you might want a dedicated unknown class
                            return 0

                    df[col] = df[col].apply(encode_value)

        # Scale numerical features (if any)
        if numerical_features:
            if fit:
                scaler = StandardScaler()
                df[numerical_features] = scaler.fit_transform(df[numerical_features])
                self.scalers["numerical"] = scaler
            else:
                if "numerical" in self.scalers:
                    df[numerical_features] = self.scalers["numerical"].transform(
                        df[numerical_features]
                    )

        # Combine features
        feature_columns = numerical_features + categorical_features
        X = df[feature_columns].values
        self.feature_names = feature_columns

        return X, feature_columns

    def _filter_valid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to valid training data."""
        if Config.TARGET_COLUMN() in df.columns:
            df = df[
                (df[f"spend"] > Config.MIN_SPEND())
                & (df[f"impressions"] > Config.MIN_IMPRESSIONS())
            ]
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for improved model performance."""
        df = df.copy()

        # Budget x Lifetime metrics
        if "budget" in df.columns and "campaign_lifetime_spend" in df.columns:
            df["budget_spend_ratio"] = df["budget"] / (
                df["campaign_lifetime_spend"] + 1
            )

        # Age x Gender interaction
        if "age_min" in df.columns and "age_max" in df.columns:
            df["age_range"] = df["age_max"] - df["age_min"]

        # Campaign objective x Platform
        if "campaign_objective" in df.columns and "platform" in df.columns:
            df["objective_platform"] = (
                df["campaign_objective"].astype(str) + "_" + df["platform"].astype(str)
            )

        # Frequency x Reach
        if "avg_frequency" in df.columns and "avg_reach" in df.columns:
            df["frequency_reach_ratio"] = df["avg_frequency"] / (df["avg_reach"] + 1)

        return df

    def save_preprocessors(self, path: Optional[Path] = None) -> None:
        """Save fitted scalers and encoders."""
        if path is None:
            path = Config.MODELS_DIR() / "preprocessors.joblib"

        Config.ensure_directories()
        joblib.dump(
            {
                "scalers": self.scalers,
                "encoders": self.encoders,
                "feature_names": self.feature_names,
                "feature_version": self.feature_version,
            },
            path,
        )

    def load_preprocessors(self, path: Optional[Path] = None) -> None:
        """Load fitted scalers and encoders."""
        if path is None:
            path = Config.MODELS_DIR() / "preprocessors.joblib"

        if path.exists():
            data = joblib.load(path)
            self.scalers = data["scalers"]
            self.encoders = data["encoders"]
            self.feature_names = data["feature_names"]
            self.feature_version = data["feature_version"]

    def get_feature_importance(
        self, model: object, feature_names: List[str]
    ) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            raise ValueError("Model does not have feature importance")

        return pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
