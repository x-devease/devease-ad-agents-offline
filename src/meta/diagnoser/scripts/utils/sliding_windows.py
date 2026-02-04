"""
Sliding window generation utilities for diagnoser evaluation scripts.

This module provides functions to generate time-bounded sliding windows
from daily or hourly data for backtesting purposes.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def generate_sliding_windows_daily(
    daily_data: pd.DataFrame,
    window_size_days: int = 30,
    step_days: int = 7,
    max_windows: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate sliding windows from daily data.

    Creates time-bounded windows for backtesting. Windows are generated
    by sliding a window of size window_size_days across the data by
    step_days increments, limited to max_windows total.

    Args:
        daily_data: Daily data with 'date' column
        window_size_days: Size of each window in days (default: 30)
        step_days: Step size between windows in days (default: 7)
        max_windows: Maximum number of windows to generate (default: 10)

    Returns:
        List of window dictionaries, each containing:
            - window_num: Window index (0-based)
            - start_date: Window start date (datetime)
            - end_date: Window end date (datetime)
            - data: DataFrame with data in window
    """
    windows = []

    if len(daily_data) < window_size_days:
        logger.warning(
            f"Insufficient data: {len(daily_data)} < {window_size_days}"
        )
        return windows

    # Ensure data is sorted
    daily_data = daily_data.sort_values('date')

    # Get date range
    min_date = daily_data['date'].min()
    max_date = daily_data['date'].max()

    # Generate windows
    current_start = min_date
    window_num = 0

    while (
        current_start + timedelta(days=window_size_days) <= max_date + timedelta(days=1)
        and window_num < max_windows
    ):
        current_end = current_start + timedelta(days=window_size_days - 1)

        # Extract window data
        window_data = daily_data[
            (daily_data['date'] >= current_start) &
            (daily_data['date'] <= current_end)
        ].copy()

        if len(window_data) > 0:
            windows.append({
                'window_num': window_num,
                'start_date': current_start,
                'end_date': current_end,
                'data': window_data
            })

            window_num += 1

        # Move to next window
        current_start = current_start + timedelta(days=step_days)

    logger.info(f"Generated {len(windows)} daily windows (limited to {max_windows})")
    return windows


def generate_sliding_windows_hourly(
    hourly_data: pd.DataFrame,
    window_size_hours: int = 24,
    step_hours: int = 6,
    max_windows: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate sliding windows from hourly data.

    Creates time-bounded windows for backtesting. Windows are generated
    by sliding a window of size window_size_hours across the data by
    step_hours increments, limited to max_windows total.

    Args:
        hourly_data: Hourly data with 'date' column
        window_size_hours: Size of each window in hours (default: 24)
        step_hours: Step size between windows in hours (default: 6)
        max_windows: Maximum number of windows to generate (default: 10)

    Returns:
        List of window dictionaries, each containing:
            - window_num: Window index (0-based)
            - start_date: Window start date (datetime)
            - end_date: Window end date (datetime)
            - data: DataFrame with data in window
    """
    windows = []

    # Check if we have enough hours of data
    total_hours = len(hourly_data)
    if total_hours < window_size_hours:
        logger.warning(
            f"Insufficient data: {total_hours} < {window_size_hours} hours"
        )
        return windows

    # Ensure data is sorted
    hourly_data = hourly_data.sort_values('date')

    # Get date range
    min_date = hourly_data['date'].min()
    max_date = hourly_data['date'].max()

    # Generate windows
    current_start = min_date
    window_num = 0

    while (
        current_start + timedelta(hours=window_size_hours) <= max_date + timedelta(hours=1)
        and window_num < max_windows
    ):
        current_end = current_start + timedelta(hours=window_size_hours - 1)

        # Extract window data
        window_data = hourly_data[
            (hourly_data['date'] >= current_start) &
            (hourly_data['date'] <= current_end)
        ].copy()

        if len(window_data) > 0:
            windows.append({
                'window_num': window_num,
                'start_date': current_start,
                'end_date': current_end,
                'data': window_data
            })

            window_num += 1

        # Move to next window
        current_start = current_start + timedelta(hours=step_hours)

    logger.info(f"Generated {len(windows)} hourly windows (limited to {max_windows})")
    return windows
