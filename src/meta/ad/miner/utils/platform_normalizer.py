"""Platform normalization utilities for ROAS data.

This module provides utilities for normalizing ROAS values across different
ad platforms to account for platform-specific baselines and biases.
"""

import logging
from typing import Dict, Optional

import pandas as pd


class PlatformNormalizer:
    """Normalize ROAS values by platform baselines.

    Different ad platforms have different average ROAS due to:
    - Audience demographics
    - Ad formats and placements
    - Bidding mechanics
    - Industry verticals

    This class provides normalization to make ROAS values comparable
    across platforms.

    Attributes:
        platform_multipliers: Dict mapping platform names to multipliers
        platform_baselines: Dict mapping platform names to baseline ROAS
    """

    # Default platform multipliers based on industry benchmarks
    # Values > 1.0: platform tends to have higher ROAS
    # Values < 1.0: platform tends to have lower ROAS
    DEFAULT_MULTIPLIERS: Dict[str, float] = {
        "meta": 1.2,  # Facebook/Instagram: strong targeting
        "facebook": 1.2,
        "instagram": 1.2,
        "tiktok": 0.9,  # Younger audience, lower purchase intent
        "google": 1.0,  # Search intent: balanced
        "youtube": 1.0,
        "linkedin": 1.3,  # B2B: higher purchase values
        "twitter": 0.8,  # Lower engagement for commerce
        "pinterest": 1.1,  # Shopping intent
        "snapchat": 0.9,
    }

    def __init__(
        self,
        platform_multipliers: Optional[Dict[str, float]] = None,
        fit_baselines: bool = False,
    ):
        """Initialize PlatformNormalizer.

        Args:
            platform_multipliers: Custom platform multipliers. If None,
                uses DEFAULT_MULTIPLIERS.
            fit_baselines: Whether to learn baselines from data.
        """
        self.platform_multipliers = (
            platform_multipliers or self.DEFAULT_MULTIPLIERS.copy()
        )
        self.fit_baselines = fit_baselines
        self.platform_baselines: Dict[str, float] = {}
        self.is_fitted = False

        self.logger = logging.getLogger(__name__)

    def fit(
        self, df: pd.DataFrame, platform_col: str, roas_col: str
    ) -> "PlatformNormalizer":
        """Fit platform baselines from data.

        Calculates median ROAS for each platform to use as normalization
        baseline.

        Args:
            df: Input dataframe.
            platform_col: Name of platform column.
            roas_col: Name of ROAS column.

        Returns:
            self (fitted normalizer).
        """
        if platform_col not in df.columns:
            self.logger.warning(
                "Platform column '%s' not found, skipping normalization",
                platform_col,
            )
            return self

        # Calculate median ROAS per platform
        platform_stats = df.groupby(platform_col)[roas_col].median()

        # Calculate global median
        global_median = df[roas_col].median()

        # Store baselines
        for platform, median_roas in platform_stats.items():
            self.platform_baselines[platform] = float(median_roas)

        # Calculate multipliers based on baselines
        for platform, baseline in self.platform_baselines.items():
            if baseline > 0:
                # Platform with higher baseline gets multiplier < 1
                self.platform_multipliers[platform] = global_median / baseline
            else:
                self.platform_multipliers[platform] = 1.0

        self.is_fitted = True

        self.logger.info("Fitted platform baselines:")
        for platform, baseline in self.platform_baselines.items():
            self.logger.info("  %15s: %.4f", platform, baseline)

        self.logger.info("Platform multipliers:")
        for platform, multiplier in self.platform_multipliers.items():
            self.logger.info("  %15s: %.4f", platform, multiplier)

        return self

    def transform(
        self,
        df: pd.DataFrame,
        platform_col: str,
        roas_col: str,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Normalize ROAS by platform.

        Args:
            df: Input dataframe.
            platform_col: Name of platform column.
            roas_col: Name of ROAS column to normalize.
            inplace: Whether to modify df in place.

        Returns:
            DataFrame with normalized ROAS column.
        """
        if not inplace:
            df = df.copy()

        if platform_col not in df.columns:
            self.logger.warning(
                "Platform column '%s' not found, skipping normalization",
                platform_col,
            )
            return df

        # Apply normalization
        for platform, multiplier in self.platform_multipliers.items():
            mask = df[platform_col] == platform
            if mask.any():
                df.loc[mask, roas_col] = df.loc[mask, roas_col] / multiplier

        # Add normalized column
        normalized_col = f"{roas_col}_normalized"
        df[normalized_col] = df[roas_col]

        self.logger.info(
            "Normalized ROAS by platform (saved to '%s')", normalized_col
        )

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        platform_col: str,
        roas_col: str,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Fit normalizer and transform data.

        Args:
            df: Input dataframe.
            platform_col: Name of platform column.
            roas_col: Name of ROAS column.
            inplace: Whether to modify df in place.

        Returns:
            DataFrame with normalized ROAS.
        """
        self.fit(df, platform_col, roas_col)
        return self.transform(df, platform_col, roas_col, inplace)

    def get_platform_stats(
        self, df: pd.DataFrame, platform_col: str, roas_col: str
    ) -> pd.DataFrame:
        """Get statistics per platform.

        Args:
            df: Input dataframe.
            platform_col: Name of platform column.
            roas_col: Name of ROAS column.

        Returns:
            DataFrame with platform statistics.
        """
        if platform_col not in df.columns:
            self.logger.warning("Platform column '%s' not found", platform_col)
            return pd.DataFrame()

        stats = (
            df.groupby(platform_col)
            .agg({roas_col: ["count", "mean", "median", "std", "min", "max"]})
            .round(4)
        )

        stats.columns = ["count", "mean", "median", "std", "min", "max"]

        return stats


def add_platform_normalization(
    df: pd.DataFrame,
    platform_col: str = "platform",
    roas_col: str = "mean_roas",
    custom_multipliers: Optional[Dict[str, float]] = None,
    fit_from_data: bool = False,
) -> pd.DataFrame:
    """Convenience function to add platform-normalized ROAS column.

    WARNING: To avoid data leakage, fit_from_data should only be set to True
    if called on training data BEFORE train/test split. For test data or
    production use, fit_from_data should be False to use default multipliers.

    Args:
        df: Input dataframe.
        platform_col: Name of platform column.
        roas_col: Name of ROAS column.
        custom_multipliers: Optional custom platform multipliers.
        fit_from_data: Whether to learn platform baselines from the data.
            If False, uses DEFAULT_MULTIPLIERS (recommended to avoid
            data leakage).

    Returns:
        DataFrame with normalized ROAS column.
    """
    normalizer = PlatformNormalizer(platform_multipliers=custom_multipliers)

    if fit_from_data:
        # Only fit on training data to avoid data leakage
        return normalizer.fit_transform(df, platform_col, roas_col)
    # Use default multipliers to avoid data leakage
    return normalizer.transform(df, platform_col, roas_col)
