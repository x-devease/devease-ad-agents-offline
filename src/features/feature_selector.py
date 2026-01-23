"""
Smart feature selection pipeline.

Reduces features from 170 to 100 using multi-stage selection:
1. Variance threshold
2. Correlation filter
3. Mutual information
4. Model-based selection
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_regression,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Optional
import joblib
from pathlib import Path


class FeatureSelector:
    """
    Intelligent feature selection pipeline.

    Combines multiple selection strategies for optimal feature set.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        max_features: int = 100,
        k_mi: int = 150,
        random_state: int = 42,
    ):
        """
        Initialize feature selector.

        Args:
            variance_threshold: Remove features with variance below threshold
            correlation_threshold: Remove one of any features with correlation above threshold
            max_features: Maximum number of features to keep
            k_mi: Number of features to keep after mutual information selection
            random_state: Random seed
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.k_mi = k_mi
        self.random_state = random_state

        self.selected_features = None
        self.selection_history = {}

    def fit(
        self, X: np.ndarray, feature_names: List[str], y: np.ndarray
    ) -> "FeatureSelector":
        """
        Fit feature selector.

        Args:
            X: Feature matrix
            feature_names: List of feature names
            y: Target vector

        Returns:
            self
        """
        df = pd.DataFrame(X, columns=feature_names)
        original_count = len(feature_names)

        # Stage 1: Variance threshold
        df = self._apply_variance_threshold(df)
        self.selection_history["variance_threshold"] = {
            "kept": len(df.columns),
            "removed": original_count - len(df.columns),
        }

        # Stage 2: Correlation filter
        df = self._apply_correlation_filter(df)
        self.selection_history["correlation_filter"] = {
            "kept": len(df.columns),
            "removed": original_count
            - len(df.columns)
            - self.selection_history["variance_threshold"]["removed"],
        }

        # Stage 3: Mutual information
        if len(df.columns) > self.max_features:
            df = self._apply_mutual_information(df, y)
            self.selection_history["mutual_information"] = {
                "kept": len(df.columns),
                "removed": (
                    len(df.columns) - self.k_mi if len(df.columns) > self.k_mi else 0
                ),
            }

        # Stage 4: Model-based selection
        if len(df.columns) > self.max_features:
            df = self._apply_model_selection(df, y)
            self.selection_history["model_based"] = {
                "kept": len(df.columns),
                "removed": len(df.columns) - self.max_features,
            }

        self.selected_features = df.columns.tolist()
        self.selection_history["final"] = {
            "original": original_count,
            "selected": len(self.selected_features),
            "removed": original_count - len(self.selected_features),
        }

        return self

    def transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """
        Transform feature matrix using selected features.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Transformed feature matrix with selected features only
        """
        if self.selected_features is None:
            raise ValueError("Feature selector must be fitted before transform")

        df = pd.DataFrame(X, columns=feature_names)
        return df[self.selected_features].values

    def fit_transform(
        self, X: np.ndarray, feature_names: List[str], y: np.ndarray
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, feature_names, y)
        return self.transform(X, feature_names)

    def _apply_variance_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove low-variance features."""
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(df)

        selected_mask = vt.get_support()
        return df.loc[:, selected_mask]

    def _apply_correlation_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Calculate correlation with proper NaN handling
        # min_periods ensures we only calculate correlation when we have enough valid observations
        corr_matrix = df.corr(min_periods=max(5, len(df) // 2)).abs()

        # Check for NaN values in correlation matrix and handle them
        # If correlation is NaN, it means either:
        # 1. Not enough valid observations
        # 2. Feature is constant (zero variance)
        corr_matrix = corr_matrix.fillna(
            0
        )  # Treat NaN/uncorrelatable features as uncorrelated

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation above threshold
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

        return df.drop(columns=to_drop)

    def _apply_mutual_information(
        self, df: pd.DataFrame, y: np.ndarray
    ) -> pd.DataFrame:
        """
        Select top K features by mutual information.

        NOTE: Mutual information assumes independent observations.
        For time-series or grouped data (e.g., multiple ads per adset),
        MI estimates may be overconfident. This is a known limitation
        but acceptable for feature ranking purposes.

        For production use with temporal data, consider:
        - Time-series aware feature importance
        - Grouped cross-validation for feature selection
        - Lag-based features to capture temporal dependencies
        """
        k = min(self.k_mi, len(df.columns))

        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(df, y)

        selected_mask = selector.get_support()
        return df.loc[:, selected_mask]

    def _apply_model_selection(self, df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Select features using random forest importance."""
        rf = RandomForestRegressor(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )

        selector = SelectFromModel(rf, max_features=self.max_features)
        selector.fit(df, y)

        selected_mask = selector.get_support()
        return df.loc[:, selected_mask]

    def get_selection_history(self) -> dict:
        """Get feature selection history."""
        return self.selection_history

    def save(self, path: Path) -> None:
        """Save feature selector."""
        joblib.dump(
            {
                "selected_features": self.selected_features,
                "selection_history": self.selection_history,
                "params": {
                    "variance_threshold": self.variance_threshold,
                    "correlation_threshold": self.correlation_threshold,
                    "max_features": self.max_features,
                    "k_mi": self.k_mi,
                    "random_state": self.random_state,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureSelector":
        """Load feature selector."""
        data = joblib.load(path)
        selector = cls(**data["params"])
        selector.selected_features = data["selected_features"]
        selector.selection_history = data["selection_history"]
        return selector
