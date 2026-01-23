"""Test time-series cross-validation classes."""

import pytest
import numpy as np
from src.utils.time_series_cv import (
    ExpandingWindowCV,
    RollingWindowCV,
    time_based_split,
)


class TestExpandingWindowCV:
    """Test ExpandingWindowCV class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time-series data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        return X, y

    def test_initialization(self):
        """Test CV initialization."""
        cv = ExpandingWindowCV(min_train_size=50, test_size=10, gap=2)

        assert cv.min_train_size == 50
        assert cv.test_size == 10
        assert cv.gap == 2

    def test_split_generates_correct_indices(self, sample_data):
        """Test that split generates correct train/test indices."""
        X, y = sample_data
        cv = ExpandingWindowCV(min_train_size=50, test_size=10)

        splits = list(cv.split(X, y))

        assert len(splits) > 0

        # Check first split
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 50
        assert len(test_idx) == 10
        assert train_idx[0] == 0
        assert train_idx[-1] == 49
        assert test_idx[0] == 50
        assert test_idx[-1] == 59

    def test_expanding_window_behavior(self, sample_data):
        """Test that training window expands with each fold."""
        X, y = sample_data
        cv = ExpandingWindowCV(min_train_size=50, test_size=10)

        splits = list(cv.split(X, y))

        # Training size should increase with each fold
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_gap_between_train_and_test(self, sample_data):
        """Test that gap is respected between train and test."""
        X, y = sample_data
        cv = ExpandingWindowCV(min_train_size=40, test_size=10, gap=5)

        splits = list(cv.split(X, y))

        for train_idx, test_idx in splits:
            # Last train index should be at least 'gap' away from first test index
            assert train_idx[-1] + 1 < test_idx[0]
            assert test_idx[0] - train_idx[-1] - 1 >= cv.gap

    def test_no_overlap_in_splits(self, sample_data):
        """Test that train and test indices don't overlap."""
        X, y = sample_data
        cv = ExpandingWindowCV(min_train_size=50, test_size=10)

        splits = list(cv.split(X, y))

        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set.intersection(test_set)) == 0

    def test_temporal_ordering_preserved(self, sample_data):
        """Test that temporal ordering is preserved."""
        X, y = sample_data
        cv = ExpandingWindowCV(min_train_size=50, test_size=10)

        splits = list(cv.split(X, y))

        for train_idx, test_idx in splits:
            # All train indices should be less than test indices
            assert train_idx[-1] < test_idx[0]

    def test_get_n_splits(self, sample_data):
        """Test get_n_splits returns correct count."""
        X, y = sample_data
        cv = ExpandingWindowCV(min_train_size=50, test_size=10)

        n_splits = cv.get_n_splits(X, y)
        splits = list(cv.split(X, y))

        assert n_splits == len(splits)

    def test_handles_small_datasets(self):
        """Test with small dataset that may not produce splits."""
        X = np.random.randn(30, 5)
        y = np.random.randn(30)

        cv = ExpandingWindowCV(min_train_size=50, test_size=10)

        n_splits = cv.get_n_splits(X, y)
        assert n_splits == 0

    def test_exact_boundary_conditions(self):
        """Test edge case where data exactly fits parameters."""
        # Create data that exactly fits: 50 train + 10 test = 60 samples
        X = np.random.randn(60, 5)
        y = np.random.randn(60)

        cv = ExpandingWindowCV(min_train_size=50, test_size=10)

        n_splits = cv.get_n_splits(X, y)
        assert n_splits >= 1


class TestRollingWindowCV:
    """Test RollingWindowCV class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time-series data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        return X, y

    def test_initialization(self):
        """Test CV initialization."""
        cv = RollingWindowCV(train_size=50, test_size=10, gap=2)

        assert cv.train_size == 50
        assert cv.test_size == 10
        assert cv.gap == 2

    def test_split_generates_correct_indices(self, sample_data):
        """Test that split generates correct train/test indices."""
        X, y = sample_data
        cv = RollingWindowCV(train_size=50, test_size=10)

        splits = list(cv.split(X, y))

        assert len(splits) > 0

        # Check first split
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 50
        assert len(test_idx) == 10
        assert train_idx[0] == 0
        assert train_idx[-1] == 49

    def test_fixed_training_window_size(self, sample_data):
        """Test that training window size remains constant."""
        X, y = sample_data
        cv = RollingWindowCV(train_size=50, test_size=10)

        splits = list(cv.split(X, y))

        # All training folds should have same size
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        assert all(size == 50 for size in train_sizes)

    def test_window_moves_forward(self, sample_data):
        """Test that window rolls forward in time."""
        X, y = sample_data
        cv = RollingWindowCV(train_size=50, test_size=10)

        splits = list(cv.split(X, y))

        # Training start should increase with each fold
        train_starts = [train_idx[0] for train_idx, _ in splits]
        for i in range(1, len(train_starts)):
            assert train_starts[i] > train_starts[i - 1]

    def test_gap_between_train_and_test(self, sample_data):
        """Test that gap is respected between train and test."""
        X, y = sample_data
        cv = RollingWindowCV(train_size=40, test_size=10, gap=5)

        splits = list(cv.split(X, y))

        for train_idx, test_idx in splits:
            # Last train index should be at least 'gap' away from first test index
            assert test_idx[0] - train_idx[-1] - 1 >= cv.gap

    def test_get_n_splits(self, sample_data):
        """Test get_n_splits returns correct count."""
        X, y = sample_data
        cv = RollingWindowCV(train_size=50, test_size=10)

        n_splits = cv.get_n_splits(X, y)
        splits = list(cv.split(X, y))

        assert n_splits == len(splits)

    def test_handles_insufficient_data(self):
        """Test with insufficient data for any splits."""
        X = np.random.randn(30, 5)
        y = np.random.randn(30)

        cv = RollingWindowCV(train_size=50, test_size=10)

        n_splits = cv.get_n_splits(X, y)
        assert n_splits == 0


class TestTimeBasedSplit:
    """Test time_based_split function."""

    def test_time_based_split_creates_correct_splits(self):
        """Test that time_based_split creates correct splits."""
        import pandas as pd

        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n_samples),
                "value": np.random.randn(n_samples),
            }
        )

        splits = time_based_split(df, "date", n_splits=5)

        assert len(splits) == 5

        # Check first split
        train_df, test_df = splits[0]
        assert len(train_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(test_df) <= len(df)

    def test_time_based_split_maintains_temporal_order(self):
        """Test that temporal order is maintained."""
        import pandas as pd

        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n_samples),
                "value": np.random.randn(n_samples),
            }
        )

        splits = time_based_split(df, "date", n_splits=3)

        for train_df, test_df in splits:
            # Train dates should all be before test dates
            max_train_date = train_df["date"].max()
            min_test_date = test_df["date"].min()
            assert max_train_date <= min_test_date

    def test_time_based_split_no_leakage(self):
        """Test that there's no data leakage between train and test."""
        import pandas as pd

        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=n_samples),
                "value": np.random.randn(n_samples),
            }
        )

        splits = time_based_split(df, "date", n_splits=3)

        for train_df, test_df in splits:
            # No overlap in indices
            train_indices = set(train_df.index)
            test_indices = set(test_df.index)
            assert len(train_indices.intersection(test_indices)) == 0

    def test_time_based_split_sorts_by_date(self):
        """Test that data is sorted by date before splitting."""
        import pandas as pd

        np.random.seed(42)
        n_samples = 100

        # Create unsorted data
        dates = pd.date_range("2024-01-01", periods=n_samples)
        np.random.shuffle(dates.to_numpy().copy())

        df = pd.DataFrame({"date": dates, "value": np.random.randn(n_samples)})

        splits = time_based_split(df, "date", n_splits=3)

        # Each split should be properly ordered
        for train_df, test_df in splits:
            assert train_df["date"].is_monotonic_increasing
            assert test_df["date"].is_monotonic_increasing
