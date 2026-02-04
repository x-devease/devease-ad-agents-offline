"""
Data loading abstraction for diagnoser system.

This module provides an abstract interface for loading diagnostic data,
with concrete implementations for different data sources.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """
    Abstract interface for loading diagnosis data.

    Defines the contract for data loading, allowing different
    implementations (Meta, Google, mock, etc.) to be used
    interchangeably.

    Implementations must handle:
    - Data validation
    - Type conversion
    - Date parsing
    - Error handling
    """

    @abstractmethod
    def load_daily_data(
        self,
        customer: str,
        platform: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load daily performance data.

        Args:
            customer: Customer name (e.g., "moprobo")
            platform: Platform name (e.g., "meta", "google")
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with columns:
                - date_start (datetime)
                - spend (float)
                - impressions (int)
                - reach (int)
                - clicks (int)
                - conversions (float)
                - purchase_roas (float)

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        pass

    @abstractmethod
    def load_hourly_data(
        self,
        customer: str,
        platform: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load hourly performance data.

        Args:
            customer: Customer name
            platform: Platform name
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with hourly data, or None if not available
            Columns: date_start, hour, spend, impressions, clicks, purchase_roas

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        pass


class MetaDataLoader(DataLoader):
    """
    Meta/Facebook data loader implementation.

    Loads data from CSV files in the standard Meta format.
    """

    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize Meta data loader.

        Args:
            data_root: Root directory for data files
                       Default: datasets/{customer}/{platform}/raw/
        """
        self.data_root = data_root

    def _get_data_path(self, customer: str, platform: str) -> Path:
        """Get path to data directory."""
        if self.data_root is None:
            return Path(f"datasets/{customer}/{platform}/raw")
        return self.data_root

    def load_daily_data(
        self,
        customer: str,
        platform: str = "meta",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load daily Meta ad insights data.

        Args:
            customer: Customer name
            platform: Platform name (default: "meta")
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Preprocessed daily data DataFrame

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        data_path = self._get_data_path(customer, platform)
        daily_path = data_path / "ad_daily_insights_2024-12-17_2025-12-17.csv"

        if not daily_path.exists():
            raise FileNotFoundError(f"Data file not found: {daily_path}")

        logger.info(f"Loading daily data from: {daily_path}")
        df = pd.read_csv(daily_path)

        # Preprocess
        df = self._preprocess_daily_data(df)

        # Filter by date range if specified
        if start_date or end_date:
            df = self._filter_by_date(df, start_date, end_date)

        logger.info(f"Loaded {len(df)} rows")
        return df

    def load_hourly_data(
        self,
        customer: str,
        platform: str = "meta",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load hourly Meta ad insights data.

        Args:
            customer: Customer name
            platform: Platform name
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Hourly data DataFrame, or None if not available
        """
        data_path = self._get_data_path(customer, platform)
        hourly_path = data_path / "ad_hourly_insights_2024-12-17_2025-12-17.csv"

        if not hourly_path.exists():
            logger.warning(f"Hourly data file not found: {hourly_path}")
            return None

        logger.info(f"Loading hourly data from: {hourly_path}")
        df = pd.read_csv(hourly_path)

        # Preprocess
        df = self._preprocess_hourly_data(df)

        # Filter by date range if specified
        if start_date or end_date:
            df = self._filter_by_date(df, start_date, end_date)

        logger.info(f"Loaded {len(df)} hourly rows")
        return df

    def _preprocess_daily_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess daily data."""
        import json

        df = df.copy()

        # Convert numeric columns
        numeric_cols = ['spend', 'impressions', 'reach', 'clicks']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Extract purchase_roas from JSON
        if 'purchase_roas' in df.columns:
            df['purchase_roas'] = df['purchase_roas'].apply(self._extract_roas_value)

        # Convert date
        if 'date_start' in df.columns:
            df['date'] = pd.to_datetime(df['date_start'], errors='coerce')
            df = df.sort_values('date').dropna(subset=['date'])

        return df

    def _preprocess_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess hourly data."""
        import json

        df = df.copy()

        # Convert numeric columns
        numeric_cols = ['spend', 'impressions', 'reach', 'clicks']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Extract purchase_roas from JSON
        if 'purchase_roas' in df.columns:
            df['purchase_roas'] = df['purchase_roas'].apply(self._extract_roas_value)

        # Convert date
        if 'date_start' in df.columns:
            df['date'] = pd.to_datetime(df['date_start'], errors='coerce')
            df = df.sort_values('date').dropna(subset=['date'])

        return df

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if 'date' not in df.columns:
            return df

        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df['date'] >= start]

        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df['date'] <= end]

        return df

    def _extract_roas_value(self, roas_str: str) -> float:
        """Extract ROAS value from JSON string."""
        import json

        if pd.isna(roas_str) or roas_str == '':
            return 0.0

        try:
            data = json.loads(roas_str)
            if isinstance(data, list) and len(data) > 0:
                return float(data[0].get('value', 0))
            return 0.0
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            return 0.0


class MockDataLoader(DataLoader):
    """
    Mock data loader for testing.

    Generates synthetic data for unit tests and development.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize mock data loader.

        Args:
            seed: Random seed for reproducible data
        """
        import numpy as np
        np.random.seed(seed)

    def load_daily_data(
        self,
        customer: str,
        platform: str = "meta",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate synthetic daily data."""
        import numpy as np
        from datetime import datetime, timedelta

        # Generate 30 days of data
        start = datetime(2024, 12, 1)
        dates = [start + timedelta(days=i) for i in range(30)]

        data = {
            'date_start': [d.strftime('%Y-%m-%d') for d in dates],
            'date': dates,
            'spend': np.random.uniform(100, 500, 30),
            'impressions': np.random.randint(10000, 100000, 30),
            'reach': np.random.randint(5000, 50000, 30),
            'clicks': np.random.randint(100, 1000, 30),
            'conversions': np.random.randint(1, 50, 30),
            'purchase_roas': np.random.uniform(0.5, 5.0, 30),
        }

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} rows of mock daily data")
        return df

    def load_hourly_data(
        self,
        customer: str,
        platform: str = "meta",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Generate synthetic hourly data (last 24 hours)."""
        import numpy as np
        from datetime import datetime, timedelta

        # Generate 24 hours of data
        now = datetime.now()
        dates = [now - timedelta(hours=i) for i in range(24)][::-1]

        data = {
            'date_start': [d.strftime('%Y-%m-%d %H:00:00') for d in dates],
            'date': dates,
            'hour': [d.hour for d in dates],
            'spend': np.random.uniform(5, 50, 24),
            'impressions': np.random.randint(500, 5000, 24),
            'clicks': np.random.randint(5, 50, 24),
            'purchase_roas': np.random.uniform(0.5, 5.0, 24),
        }

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} rows of mock hourly data")
        return df
