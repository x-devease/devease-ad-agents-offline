"""
Time-series cross-validation for proper model evaluation.

Uses expanding window approach to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from sklearn.model_selection import BaseCrossValidator


class ExpandingWindowCV(BaseCrossValidator):
    """
    Expanding window time-series cross-validation.

    Each fold expands the training window while maintaining temporal ordering.
    Prevents data leakage from future into past.
    """

    def __init__(self, min_train_size: int = 100, test_size: int = 20, gap: int = 0):
        """
        Initialize expanding window CV.

        Args:
            min_train_size: Minimum size of training window
            test_size: Size of test window
            gap: Gap between train and test (to account for deployment delay)
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Args:
            X: Feature matrix
            y: Target (unused, for API compatibility)
            groups: Group labels (unused, for API compatibility)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        # Start with minimum training size
        train_end = self.min_train_size

        while train_end + self.gap + self.test_size <= n_samples:
            train_start = 0
            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

            # Expand training window for next fold
            train_end += self.test_size

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Get number of splits."""
        n_samples = len(X) if X is not None else 0
        available = n_samples - self.min_train_size - self.gap
        return max(0, available // self.test_size)


class RollingWindowCV(BaseCrossValidator):
    """
    Rolling window time-series cross-validation.

    Maintains fixed training window size, rolling forward in time.
    """

    def __init__(self, train_size: int = 100, test_size: int = 20, gap: int = 0):
        """
        Initialize rolling window CV.

        Args:
            train_size: Fixed size of training window
            test_size: Size of test window
            gap: Gap between train and test
        """
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Args:
            X: Feature matrix
            y: Target (unused)
            groups: Group labels (unused)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        # Start at beginning
        train_start = 0

        while train_start + self.train_size + self.gap + self.test_size <= n_samples:
            train_end = train_start + self.train_size
            test_start = train_end + self.gap
            test_end = test_start + self.test_size

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

            # Roll forward
            train_start += self.test_size

    def get_n_splits(
        self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None
    ) -> int:
        """Get number of splits."""
        n_samples = len(X) if X is not None else 0
        required = self.train_size + self.gap + self.test_size
        if n_samples < required:
            return 0
        available = n_samples - required
        return available // self.test_size + 1


def time_based_split(df: pd.DataFrame, date_column: str, n_splits: int = 5) -> list:
    """
    Create time-based splits sorted by date.

    Args:
        df: DataFrame with date column
        date_column: Name of date column
        n_splits: Number of splits

    Returns:
        List of (train_df, test_df) tuples
    """
    # Sort by date
    df_sorted = df.sort_values(date_column).reset_index(drop=True)

    n_samples = len(df_sorted)
    fold_size = n_samples // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        train_df = df_sorted.iloc[:train_end]
        test_df = df_sorted.iloc[test_start:test_end]

        splits.append((train_df, test_df))

    return splits
