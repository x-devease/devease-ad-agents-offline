"""
Statistics Utilities

Statistical functions for feature value analysis:
- Chi-square test
- Cramér's V
- Other statistical measures
- Regression metrics
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chi_square_test(
    contingency_table: pd.DataFrame, significance_level: float = 0.05
) -> Dict[str, float]:
    """
    Perform Chi-square test on contingency table.

    Args:
        contingency_table: Contingency table (rows: groups, columns: values)
        significance_level: P-value threshold for significance (default: 0.05)

    Returns:
        Dictionary with:
        - chi2: Chi-square statistic
        - p_value: p-value
        - dof: Degrees of freedom
        - is_significant: Whether p < significance_level
    """
    # Check if table is valid
    if contingency_table.empty:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "dof": 0,
            "is_significant": False,
        }

    # Perform chi-square test
    try:
        chi2, p_value, dof, _ = chi2_contingency(contingency_table.values)
    except ValueError:
        # Handle case where expected frequencies are zero
        # This can happen with small sample sizes or sparse data
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "dof": 0,
            "is_significant": False,
        }

    # Check significance
    is_significant = p_value < significance_level

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "is_significant": is_significant,
    }


def cramers_v(contingency_table: pd.DataFrame, chi2: float = None) -> float:
    """
    Calculate Cramér's V effect size.

    Args:
        contingency_table: Contingency table
        chi2: Chi-square statistic (if None, will be calculated)

    Returns:
        Cramér's V value (0-1)

    Formula: V = sqrt(chi2 / (n * (min(rows, cols) - 1)))
    """
    if contingency_table.empty:
        return 0.0

    # Calculate chi2 if not provided
    if chi2 is None:
        chi2_result = chi_square_test(contingency_table, 0.05)
        chi2 = chi2_result["chi2"]

    total_count = contingency_table.values.sum()
    if total_count == 0:
        return 0.0

    min_dim = min(contingency_table.shape)
    if min_dim <= 1:
        return 0.0

    # Calculate Cramér's V
    cramers_v_value = np.sqrt(chi2 / (total_count * (min_dim - 1)))

    # Ensure value is between 0 and 1
    return float(min(cramers_v_value, 1.0))


def calculate_effect_size(cramers_v_value: float) -> str:
    """
    Interpret Cramér's V effect size.

    Args:
        cramers_v: Cramér's V value

    Returns:
        Effect size interpretation: "small", "medium", "large"

    Interpretation:
    - < 0.1: small
    - 0.1-0.3: medium
    - > 0.3: large
    """
    if cramers_v_value < 0.1:
        return "small"
    if cramers_v_value < 0.3:
        return "medium"
    return "large"


def chi_square_and_cramers_v(
    contingency_table: pd.DataFrame,
    min_samples_per_cell: int = 5,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform Chi-square test and calculate Cramér's V in one call.

    Args:
        contingency_table: Contingency table
        min_samples_per_cell: Minimum samples per cell for sparsity check
        significance_level: P-value threshold for significance

    Returns:
        Dictionary with:
        - chi2: Chi-square statistic
        - p_value: p-value
        - dof: Degrees of freedom
        - is_significant: Whether p < significance_level
        - cramers_v: Cramér's V value
        - effect_size: Effect size interpretation
        - sparsity_warning: Whether data is sparse
        - sparsity_ratio: Ratio of cells with sufficient expected frequency
    """
    chi2_result = chi_square_test(contingency_table, significance_level)
    cramers_v_value = cramers_v(contingency_table, chi2_result["chi2"])
    effect_size = calculate_effect_size(cramers_v_value)

    # Sparsity check
    n_samples = contingency_table.sum().sum()
    expected = (
        contingency_table.sum(axis=1).values[:, None]
        * contingency_table.sum(axis=0).values[None, :]
        / n_samples
    )
    sparsity_warning = (expected < min_samples_per_cell).any()
    sparsity_ratio = (
        (expected >= min_samples_per_cell).sum() / expected.size
        if expected.size > 0
        else 0.0
    )

    return {
        **chi2_result,
        "cramers_v": cramers_v_value,
        "effect_size": effect_size,
        "sparsity_warning": sparsity_warning,
        "sparsity_ratio": float(sparsity_ratio),
    }


def calculate_regression_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate regression metrics for train and test sets.

    Args:
        y_train: Training set true values
        y_test: Test set true values
        y_pred_train: Training set predictions
        y_pred_test: Test set predictions

    Returns:
        Dictionary with regression metrics:
        - train_rmse: Root mean squared error (train)
        - test_rmse: Root mean squared error (test)
        - train_r2: R-squared score (train)
        - test_r2: R-squared score (test)
        - train_mae: Mean absolute error (train)
        - test_mae: Mean absolute error (test)
    """
    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
    }
