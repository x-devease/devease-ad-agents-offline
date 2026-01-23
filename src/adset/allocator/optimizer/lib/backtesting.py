"""
Forward-looking evaluation and backtesting for budget allocation.

This module provides utilities for evaluating budget allocation decisions
using forward-looking ROAS (future performance) instead of historical ROAS.
This prevents data leakage and provides more realistic estimates of
production performance.

Key features:
- Temporal train/test splitting for time-series validation
- Forward-looking evaluation using future period ROAS
- Rolling window cross-validation support
- Proper handling of missing dates and adsets
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from src.adset.allocator.optimizer.tuning import (
    TuningResult,
    evaluate_allocation,
    simulate_allocation,
)


def temporal_train_test_split(
    df_features: pd.DataFrame,
    date_col: str = "date_start",
    train_ratio: float = 0.7,
    val_ratio: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Split DataFrame chronologically for time-series validation.

    Args:
        df_features: DataFrame with time-series features.
        date_col: Name of date column to split on.
        train_ratio: Proportion of data for training (earliest periods).
        val_ratio: Optional proportion for validation. If None, only train/test split.

    Returns:
        Tuple of (train_df, test_df) or (train_df, val_df, test_df) if val_ratio provided.
        All splits maintain chronological order.

    Examples:
        >>> train, test = temporal_train_test_split(df, train_ratio=0.7)
        >>> train, val, test = temporal_train_test_split(df, train_ratio=0.6, val_ratio=0.2)
    """
    # Ensure date column exists and is datetime
    if date_col not in df_features.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")

    # Convert to datetime if needed
    df = df_features.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Sort chronologically
    df = df.sort_values(date_col).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * train_ratio)

    if val_ratio is not None:
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train : n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val :].copy()

        return train_df, val_df, test_df
    else:
        train_df = df.iloc[:n_train].copy()
        test_df = df.iloc[n_train:].copy()

        return train_df, test_df


def forward_looking_evaluation(
    allocation_results: pd.DataFrame,
    future_features: pd.DataFrame,
    total_budget: float,
    target_roas: Optional[float] = None,
    target_ctr: Optional[float] = None,
) -> TuningResult:
    """Evaluate allocation using forward-looking ROAS from future period.

    This is the key function to prevent data leakage. Instead of evaluating
    allocations using the same period's ROAS (which would be unknown at
    decision time), we use the NEXT period's ROAS.

    Args:
        allocation_results: DataFrame from simulate_allocation() with new_budget column.
            Contains decisions made based on current period features.
        future_features: Features from the NEXT time period (e.g., day T+1).
            Used to get forward-looking ROAS for evaluation.
        total_budget: Total budget allocated.
        target_roas: Optional target ROAS for high/low ROAS budget percentage calculation.
        target_ctr: Optional target CTR for high CTR budget percentage calculation.

    Returns:
        TuningResult with forward-looking performance metrics.

    Example:
        >>> train_df, test_df = temporal_train_test_split(features, train_ratio=0.7)
        >>> allocator = Allocator(safety_rules, decision_rules, parser)
        >>> allocation = simulate_allocation(train_df, allocator, total_budget)
        >>> result = forward_looking_evaluation(allocation, test_df, total_budget)
        >>> # result.weighted_avg_roas uses future ROAS, not historical
    """
    if len(allocation_results) == 0 or len(future_features) == 0:
        # Return empty result if no data
        return TuningResult(
            param_config={},
            total_adsets=0,
            adsets_with_changes=0,
            change_rate=0.0,
            total_budget_allocated=0.0,
            budget_utilization=0.0,
            avg_roas=0.0,
            weighted_avg_roas=0.0,
            total_revenue=0.0,
            revenue_efficiency=0.0,
            max_single_adset_pct=0.0,
            roas_std=0.0,
            budget_gini=0.0,
            budget_entropy=0.0,
            avg_budget_change_pct=0.0,
            budget_change_volatility=0.0,
            high_roas_budget_pct=0.0,
            low_roas_budget_pct=0.0,
            baseline_comparison=None,
            avg_ctr=0.0,
            weighted_avg_ctr=0.0,
            ctr_std=0.0,
            high_ctr_budget_pct=0.0,
        )

    # Merge allocation results with future features to get forward-looking ROAS and CTR
    # We want to match adsets from allocation to their ROAS in the future period
    merged = allocation_results.merge(
        future_features[["adset_id"]].drop_duplicates(),
        on="adset_id",
        how="inner",
        indicator=True,
    )

    # Filter to only adsets that exist in both periods
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge"])

    # Now get their ROAS from future features
    # Try multiple possible ROAS column names
    roas_col = None
    for col in ["roas_7d", "purchase_roas_rolling_7d", "purchase_roas", "adset_roas"]:
        if col in future_features.columns:
            roas_col = col
            break

    if roas_col is not None:
        # Merge forward-looking ROAS
        merged = merged.merge(
            future_features[["adset_id", roas_col]].rename(
                columns={roas_col: "forward_roas"}
            ),
            on="adset_id",
            how="left",
        )
        merged["roas_7d"] = merged["forward_roas"].fillna(0.0)
    else:
        # No ROAS column found, use zeros
        merged["roas_7d"] = 0.0

    # Evaluate using the merged forward-looking data
    # Create a minimal DataFrame compatible with evaluate_allocation()
    eval_df = merged[["adset_id", "current_budget", "new_budget", "roas_7d"]].copy()

    # Use the existing evaluate_allocation function with forward-looking ROAS
    # We need to create a dummy features DataFrame for compatibility
    dummy_features = pd.DataFrame(
        {
            "adset_id": eval_df["adset_id"],
            "roas_7d": eval_df["roas_7d"],  # Forward-looking ROAS
        }
    )

    result = evaluate_allocation(
        eval_df,
        dummy_features,
        total_budget,
        target_roas=target_roas,
        target_ctr=target_ctr,
    )

    return result


def create_rolling_window_splits(
    df_features: pd.DataFrame,
    n_folds: int = 3,
    train_ratio: float = 0.7,
    date_col: str = "date_start",
    min_train_days: int = 14,
    min_test_days: int = 7,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create rolling window time-series cross-validation splits.

    Each fold uses a training window followed by a test window, with the window
    rolling forward in time. This prevents lookahead bias and simulates
    production deployment.

    Args:
        df_features: DataFrame with time-series features.
        n_folds: Number of folds to create.
        train_ratio: Ratio of training data in each fold.
        date_col: Name of date column.
        min_train_days: Minimum days in training set.
        min_test_days: Minimum days in test set.

    Returns:
        List of (train_df, test_df) tuples, one per fold.

    Example:
        >>> splits = create_rolling_window_splits(df, n_folds=3, train_ratio=0.7)
        >>> for train, test in splits:
        ...     print(f"Train: {train[date_col].min()} to {train[date_col].max()}")
        ...     print(f"Test:  {test[date_col].min()} to {test[date_col].max()}")
    """
    # Ensure date column is datetime
    df = df_features.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Sort chronologically
    df = df.sort_values(date_col).reset_index(drop=True)

    # Get unique dates
    unique_dates = df[date_col].unique()
    n_dates = len(unique_dates)

    # Calculate window sizes
    train_size = max(int(n_dates * train_ratio), min_train_days)
    test_size = max(n_dates - train_size, min_test_days)

    splits = []

    # Create rolling windows
    for fold in range(n_folds):
        # Calculate start index for this fold
        # Each fold shifts forward by test_size / (n_folds - 1) dates
        if n_folds > 1:
            shift = fold * (n_dates - train_size) // (n_folds - 1)
        else:
            shift = 0

        train_start_idx = shift
        train_end_idx = min(train_start_idx + train_size, n_dates - min_test_days)

        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + test_size, n_dates)

        # Get dates for this fold
        train_dates = unique_dates[train_start_idx:train_end_idx]
        test_dates = unique_dates[test_start_idx:test_end_idx]

        # Filter DataFrame by dates
        train_df = df[df[date_col].isin(train_dates)].copy()
        test_df = df[df[date_col].isin(test_dates)].copy()

        # Skip if not enough data
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        splits.append((train_df, test_df))

    return splits


def evaluate_with_cross_validation(
    df_features: pd.DataFrame,
    allocator_creator,
    total_budget: float,
    n_folds: int = 3,
    train_ratio: float = 0.7,
    date_col: str = "date_start",
    target_roas: Optional[float] = None,
) -> Tuple[List[TuningResult], TuningResult]:
    """Evaluate allocation strategy using rolling window cross-validation.

    This provides a robust estimate of production performance by averaging
    results across multiple time periods, reducing overfitting to any
    single period.

    Args:
        df_features: DataFrame with time-series features.
        allocator_creator: Callable that creates an Allocator instance.
            Called as allocator_creator() for each fold.
        total_budget: Total budget for allocation.
        n_folds: Number of CV folds.
        train_ratio: Ratio of training data in each fold.
        date_col: Name of date column.
        target_roas: Optional target ROAS.

    Returns:
        Tuple of (fold_results, aggregated_result):
        - fold_results: List of TuningResult, one per fold
        - aggregated_result: TuningResult with mean metrics across folds

    Example:
        >>> from src.adset import Allocator, SafetyRules, DecisionRules
        >>> def make_allocator():
        ...     return Allocator(safety_rules, decision_rules, parser)
        >>> fold_results, agg_result = evaluate_with_cross_validation(
        ...     df, make_allocator, total_budget, n_folds=3
        ... )
        >>> print(f"Mean forward-looking ROAS: {agg_result.weighted_avg_roas:.4f}")
    """
    # Create rolling window splits
    splits = create_rolling_window_splits(
        df_features,
        n_folds=n_folds,
        train_ratio=train_ratio,
        date_col=date_col,
    )

    if len(splits) == 0:
        raise ValueError("No valid cross-validation splits could be created")

    fold_results = []

    # Evaluate each fold
    for fold_idx, (train_df, test_df) in enumerate(splits):
        print(f"  Evaluating fold {fold_idx + 1}/{len(splits)}...")

        # Create allocator for this fold
        allocator = allocator_creator()

        # Simulate allocation on training data features
        allocation = simulate_allocation(train_df, allocator, total_budget)

        # Evaluate on TEST data (forward-looking ROAS)
        result = forward_looking_evaluation(
            allocation, test_df, total_budget, target_roas=target_roas
        )

        fold_results.append(result)

        print(
            f"    Fold {fold_idx + 1} - ROAS: {result.weighted_avg_roas:.4f}, "
            f"Utilization: {result.budget_utilization:.2%}"
        )

    # Aggregate results across folds
    aggregated = aggregate_cv_results(fold_results)

    return fold_results, aggregated


def aggregate_cv_results(fold_results: List[TuningResult]) -> TuningResult:
    """Aggregate cross-validation results by computing mean metrics.

    Args:
        fold_results: List of TuningResult from each fold.

    Returns:
        TuningResult with mean metrics across all folds.
    """
    if not fold_results:
        raise ValueError("No fold results to aggregate")

    n_folds = len(fold_results)

    # Compute mean of all numeric fields
    aggregated = TuningResult(
        param_config={},  # No single config for CV
        total_adsets=int(np.mean([r.total_adsets for r in fold_results])),
        adsets_with_changes=int(np.mean([r.adsets_with_changes for r in fold_results])),
        change_rate=np.mean([r.change_rate for r in fold_results]),
        total_budget_allocated=np.mean(
            [r.total_budget_allocated for r in fold_results]
        ),
        budget_utilization=np.mean([r.budget_utilization for r in fold_results]),
        avg_roas=np.mean([r.avg_roas for r in fold_results]),
        weighted_avg_roas=np.mean([r.weighted_avg_roas for r in fold_results]),
        total_revenue=np.mean([r.total_revenue for r in fold_results]),
        revenue_efficiency=np.mean([r.revenue_efficiency for r in fold_results]),
        max_single_adset_pct=np.mean([r.max_single_adset_pct for r in fold_results]),
        roas_std=np.mean([r.roas_std for r in fold_results]),
        budget_gini=np.mean([r.budget_gini for r in fold_results]),
        budget_entropy=np.mean([r.budget_entropy for r in fold_results]),
        avg_budget_change_pct=np.mean([r.avg_budget_change_pct for r in fold_results]),
        budget_change_volatility=np.mean(
            [r.budget_change_volatility for r in fold_results]
        ),
        high_roas_budget_pct=np.mean([r.high_roas_budget_pct for r in fold_results]),
        low_roas_budget_pct=np.mean([r.low_roas_budget_pct for r in fold_results]),
        baseline_comparison=np.mean(
            [
                r.baseline_comparison
                for r in fold_results
                if r.baseline_comparison is not None
            ]
            or [0]
        ),
    )

    return aggregated
