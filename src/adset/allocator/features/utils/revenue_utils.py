"""
Utility functions for revenue calculation from purchase action values.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_revenue_from_purchase_actions(  # pylint: disable=invalid-name
    df: pd.DataFrame,
    update_inplace: bool = False,
    custom_logger: Optional[logging.Logger] = None,  # pylint: disable=unused-argument
) -> pd.Series:
    """
    Calculate revenue from purchase action_value columns.

    Finds all columns with "action_value" and "purchase" in the name,
    converts them to numeric, and sums them to get total revenue per row.

    Args:
        df: DataFrame to process
        update_inplace: If True, updates df["revenue"] in place.
                        If False, returns revenue Series without modifying df.
        custom_logger: Optional logger instance to use for logging.
                      If None, uses module logger.

    Returns:
        Series with calculated revenue values (index matches df index)
    """
    # Find purchase action_value columns
    purchase_action_value_cols = [
        col
        for col in df.columns
        if "action_value" in col.lower() and "purchase" in col.lower()
    ]

    if not purchase_action_value_cols:
        # Return zero revenue if no purchase action_value columns found
        return pd.Series(0.0, index=df.index)

    # Convert to numeric and fillna(0.0)
    purchase_values = pd.DataFrame()
    for col in purchase_action_value_cols:
        purchase_values[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Sum only numeric columns
    numeric_cols = [
        col
        for col in purchase_values.columns
        if pd.api.types.is_numeric_dtype(purchase_values[col])
    ]

    if numeric_cols:
        revenue = purchase_values[numeric_cols].sum(axis=1)
        if update_inplace:
            df["revenue"] = revenue
        return revenue

    # Return zero revenue if no numeric columns found
    return pd.Series(0.0, index=df.index)
