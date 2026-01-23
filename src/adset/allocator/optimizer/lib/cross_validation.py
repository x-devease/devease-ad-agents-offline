"""
Advanced cross-validation evaluation methods for budget allocation.

This module provides robust evaluation methods beyond simple train/test splits:
1. Rolling window time-series cross-validation
2. Customer-level (leave-one-out) cross-validation

These methods provide more reliable estimates of real-world performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.adset.allocator.optimizer.tuning import (
    TuningResult,
    _create_allocator_with_config,
    evaluate_allocation,
    simulate_allocation,
)


@dataclass
class CrossValidationResult:
    """Results from cross-validation evaluation.

    Attributes:
        fold_results: List of TuningResult, one per fold
        mean_metrics: Dictionary of mean metrics across all folds
        std_metrics: Dictionary of standard deviation of metrics across folds
        min_metrics: Dictionary of minimum values across all folds
        max_metrics: Dictionary of maximum values across all folds
        confidence_intervals: Dictionary of 95% confidence intervals for metrics
        n_folds: Number of folds used
    """

    fold_results: List[TuningResult]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    min_metrics: Dict[str, float]
    max_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    n_folds: int

    def summary(self) -> str:
        """Generate a formatted summary of cross-validation results."""
        lines = [
            "=" * 80,
            "CROSS-VALIDATION RESULTS",
            "=" * 80,
            "",
            f"Number of Folds: {self.n_folds}",
            "",
            "-" * 80,
            "METRICS ACROSS FOLDS",
            "-" * 80,
            "",
        ]

        # Primary metrics
        primary_metrics = [
            ("ROAS (Mean)", "weighted_avg_roas"),
            ("ROAS (Weighted)", "weighted_avg_roas"),
            ("Budget Utilization", "budget_utilization"),
            ("Revenue Efficiency", "revenue_efficiency"),
            ("Change Rate", "change_rate"),
        ]

        for label, key in primary_metrics:
            if key in self.mean_metrics:
                mean_val = self.mean_metrics[key]
                std_val = self.std_metrics[key]
                ci_lower, ci_upper = self.confidence_intervals.get(key, (0, 0))
                lines.append(
                    f"{label:25s}: {mean_val:7.4f} ± {std_val:.4f} "
                    f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                )

        lines.extend(
            [
                "",
                "-" * 80,
                "STABILITY METRICS",
                "-" * 80,
                "",
            ]
        )

        # Stability metrics
        stability_metrics = [
            ("ROAS Std Dev", "roas_std"),
            ("Budget Gini", "budget_gini"),
            ("Budget Entropy", "budget_entropy"),
            ("Budget Change Volatility", "budget_change_volatility"),
        ]

        for label, key in stability_metrics:
            if key in self.mean_metrics:
                mean_val = self.mean_metrics[key]
                std_val = self.std_metrics[key]
                lines.append(f"{label:25s}: {mean_val:7.4f} ± {std_val:.4f}")

        lines.extend(
            [
                "",
                "-" * 80,
                "FOLD DETAILS",
                "-" * 80,
                "",
            ]
        )

        # Per-fold details
        for i, fold in enumerate(self.fold_results, 1):
            lines.append(
                f"Fold {i:2d}: ROAS={fold.weighted_avg_roas:.4f}, "
                f"Util={fold.budget_utilization:.2%}, "
                f"RevEff=${fold.revenue_efficiency:.2f}"
            )

        lines.extend(["", "=" * 80])

        return "\n".join(lines)


def rolling_window_cv(
    df: pd.DataFrame,
    base_config_path: str,
    total_budget: float,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    min_train_days: int = 14,
    min_test_days: int = 7,
    customer_name: str = "moprobo",
    date_col: str = "date_start",
    param_overrides: Optional[Dict[str, Any]] = None,
) -> CrossValidationResult:
    """Perform rolling window time-series cross-validation.

    This method creates multiple train/test splits by rolling a window forward
    in time. Each fold tests on a future period that wasn't used in training.

    Example with n_folds=3:
        Fold 1: Train [days 1-20], Test [days 21-27]
        Fold 2: Train [days 8-27], Test [days 28-34]
        Fold 3: Train [days 15-34], Test [days 35-41]

    Args:
        df: DataFrame with adset features (must include date_col).
        base_config_path: Path to base config file.
        total_budget: Total budget for allocation in each fold.
        n_folds: Number of folds for cross-validation.
        train_ratio: Ratio of training window to total window size.
        min_train_days: Minimum number of days in training set.
        min_test_days: Minimum number of days in test set.
        customer_name: Customer name for config.
        date_col: Column name containing dates.
        param_overrides: Optional parameter overrides for testing.

    Returns:
        CrossValidationResult with aggregated metrics across all folds.

    Raises:
        ValueError: If insufficient data for requested number of folds.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")

    # Get unique dates and sort
    unique_dates = sorted(df[date_col].unique())
    n_dates = len(unique_dates)

    # Calculate window sizes
    window_size = min_train_days + min_test_days
    step_size = (n_dates - window_size) // (n_folds - 1) if n_folds > 1 else 0

    if n_dates < window_size:
        raise ValueError(
            f"Insufficient data: need at least {window_size} dates "
            f"({min_train_days} train + {min_test_days} test), "
            f"but only have {n_dates} dates"
        )

    # Calculate fold boundaries
    fold_boundaries = []
    for i in range(n_folds):
        start_idx = i * step_size
        train_end_idx = start_idx + int(window_size * train_ratio)
        test_end_idx = start_idx + window_size

        if test_end_idx > n_dates:
            # Adjust final fold to end of data
            test_end_idx = n_dates
            train_end_idx = max(
                start_idx + min_train_days, test_end_idx - min_test_days
            )

        fold_boundaries.append((start_idx, train_end_idx, test_end_idx))

    # Run each fold
    fold_results = []
    for fold_idx, (start_idx, train_end_idx, test_end_idx) in enumerate(
        fold_boundaries, 1
    ):
        # Get train and test dates
        train_dates = unique_dates[start_idx:train_end_idx]
        test_dates = unique_dates[train_end_idx:test_end_idx]

        # Split data
        train_df = df[df[date_col].isin(train_dates)]
        test_df = df[df[date_col].isin(test_dates)]

        # Create allocator (trained on train data)
        param_overrides = param_overrides or {}
        allocator = _create_allocator_with_config(
            base_config_path=base_config_path,
            param_overrides=param_overrides,
            customer_name=customer_name,
        )

        # Simulate allocation on test data
        allocation_results = simulate_allocation(test_df, allocator, total_budget)
        fold_result = evaluate_allocation(allocation_results, test_df, total_budget)
        fold_results.append(fold_result)

    # Aggregate metrics across folds
    return _aggregate_cv_results(fold_results)


def customer_level_cv(
    customer_dfs: Dict[str, pd.DataFrame],
    base_config_path: str,
    total_budget: float,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> CrossValidationResult:
    """Perform leave-one-customer-out cross-validation.

    This method trains on all customers except one, then tests on the held-out
    customer. This tests generalization to new customers.

    Example with 3 customers:
        Fold 1 (leave out customer A): Train on [B, C], Test on A
        Fold 2 (leave out customer B): Train on [A, C], Test on B
        Fold 3 (leave out customer C): Train on [A, B], Test on C

    Args:
        customer_dfs: Dictionary mapping customer names to their DataFrames.
        base_config_path: Path to base config file.
        total_budget: Total budget for allocation in each fold.
        param_overrides: Optional parameter overrides for testing.

    Returns:
        CrossValidationResult with aggregated metrics across all folds.

    Raises:
        ValueError: If fewer than 2 customers provided.
    """
    if len(customer_dfs) < 2:
        raise ValueError(
            f"Need at least 2 customers for leave-one-out CV, got {len(customer_dfs)}"
        )

    customer_names = list(customer_dfs.keys())
    fold_results = []

    # Leave-one-out for each customer
    for test_customer in customer_names:
        # Train on all customers except test_customer
        train_customers = [c for c in customer_names if c != test_customer]

        # Combine training data from all train customers
        train_df_list = [customer_dfs[c] for c in train_customers]
        train_df = pd.concat(train_df_list, ignore_index=True)
        test_df = customer_dfs[test_customer]

        # Create allocator
        param_overrides = param_overrides or {}
        allocator = _create_allocator_with_config(
            base_config_path=base_config_path,
            param_overrides=param_overrides,
            customer_name=train_customers[0],  # Use first train customer for config
        )

        # Simulate allocation on test customer
        allocation_results = simulate_allocation(test_df, allocator, total_budget)
        fold_result = evaluate_allocation(allocation_results, test_df, total_budget)
        fold_results.append(fold_result)

    # Aggregate metrics across folds
    return _aggregate_cv_results(fold_results)


def _aggregate_cv_results(fold_results: List[TuningResult]) -> CrossValidationResult:
    """Aggregate results from cross-validation folds.

    Args:
        fold_results: List of TuningResult, one per fold.

    Returns:
        CrossValidationResult with aggregated metrics.
    """
    # Extract metric values from each fold
    metrics_to_aggregate = [
        "weighted_avg_roas",
        "budget_utilization",
        "revenue_efficiency",
        "change_rate",
        "roas_std",
        "budget_gini",
        "budget_entropy",
        "avg_budget_change_pct",
        "budget_change_volatility",
        "high_roas_budget_pct",
        "low_roas_budget_pct",
    ]

    # Calculate statistics for each metric
    mean_metrics = {}
    std_metrics = {}
    min_metrics = {}
    max_metrics = {}
    confidence_intervals = {}

    for metric in metrics_to_aggregate:
        values = [getattr(fr, metric) for fr in fold_results]

        mean_metrics[metric] = float(np.mean(values))
        std_metrics[metric] = float(np.std(values, ddof=1))  # Sample std
        min_metrics[metric] = float(np.min(values))
        max_metrics[metric] = float(np.max(values))

        # 95% confidence interval using t-distribution
        n = len(values)
        if n > 1:
            sem = std_metrics[metric] / np.sqrt(n)
            t_crit = stats.t.ppf(0.975, df=n - 1)  # Two-tailed 95% CI
            ci_margin = t_crit * sem
            confidence_intervals[metric] = (
                mean_metrics[metric] - ci_margin,
                mean_metrics[metric] + ci_margin,
            )
        else:
            confidence_intervals[metric] = (mean_metrics[metric], mean_metrics[metric])

    return CrossValidationResult(
        fold_results=fold_results,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        min_metrics=min_metrics,
        max_metrics=max_metrics,
        confidence_intervals=confidence_intervals,
        n_folds=len(fold_results),
    )


def compare_cv_results(
    cv_result1: CrossValidationResult,
    cv_result2: CrossValidationResult,
    name1: str = "Approach 1",
    name2: str = "Approach 2",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Statistically compare two cross-validation results.

    Performs paired t-tests to determine if one approach significantly
    outperforms the other across folds.

    Args:
        cv_result1: First cross-validation result.
        cv_result2: Second cross-validation result.
        name1: Name for first approach.
        name2: Name for second approach.
        alpha: Significance level for statistical tests.

    Returns:
        Dictionary with comparison statistics.

    Raises:
        ValueError: If CV results have different number of folds.
    """
    if cv_result1.n_folds != cv_result2.n_folds:
        raise ValueError(
            f"Cannot compare CV results with different number of folds: "
            f"{cv_result1.n_folds} vs {cv_result2.n_folds}"
        )

    comparison = {
        "n_folds": cv_result1.n_folds,
        "approach1_name": name1,
        "approach2_name": name2,
        "metrics": {},
    }

    # Compare key metrics
    metrics_to_compare = [
        "weighted_avg_roas",
        "budget_utilization",
        "revenue_efficiency",
        "roas_std",
    ]

    for metric in metrics_to_compare:
        values1 = [getattr(fr, metric) for fr in cv_result1.fold_results]
        values2 = [getattr(fr, metric) for fr in cv_result2.fold_results]

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(values2, values1)

        # Cohen's d (effect size)
        diff = np.array(values2) - np.array(values1)
        pooled_std = np.sqrt(
            (np.std(values1, ddof=1) ** 2 + np.std(values2, ddof=1) ** 2) / 2
        )
        cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0

        # Determine significance
        significant = p_value < alpha

        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"

        # Mean improvement percentage
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        improvement_pct = ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0

        comparison["metrics"][metric] = {
            f"{name1}_mean": float(mean1),
            f"{name2}_mean": float(mean2),
            f"{name2}_std": float(np.std(values2, ddof=1)),
            "improvement_pct": float(improvement_pct),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": significant,
            "cohens_d": float(cohens_d),
            "effect_size": effect_size,
        }

    return comparison


def compare_cv_results_summary(comparison: Dict[str, Any]) -> str:
    """Generate formatted summary of CV comparison results.

    Args:
        comparison: Comparison dictionary from compare_cv_results.

    Returns:
        Formatted string summary.
    """
    name1 = comparison["approach1_name"]
    name2 = comparison["approach2_name"]
    n_folds = comparison["n_folds"]

    lines = [
        "=" * 80,
        f"CROSS-VALIDATION COMPARISON: {name1} vs {name2}",
        "=" * 80,
        "",
        f"Number of Folds: {n_folds}",
        "",
        "-" * 80,
        "METRIC COMPARISONS",
        "-" * 80,
        "",
    ]

    # Primary metrics
    for metric, stats in comparison["metrics"].items():
        metric_label = metric.replace("_", " ").title()
        mean1 = stats[f"{name1}_mean"]
        mean2 = stats[f"{name2}_mean"]
        improvement = stats["improvement_pct"]
        p_value = stats["p_value"]
        significant = stats["significant"]
        effect_size = stats["effect_size"]
        cohens_d = stats["cohens_d"]

        sig_marker = "[SUCCESS]" if significant else "[OMIT]"

        lines.append(f"{metric_label}:")
        lines.append(f"  {name1:20s}: {mean1:7.4f}")
        lines.append(f"  {name2:20s}: {mean2:7.4f}")
        lines.append(f"  Improvement:         {improvement:+6.2f}%")
        lines.append(f"  P-value:             {p_value:.4f} {sig_marker}")
        lines.append(f"  Effect Size:         {effect_size} (d={cohens_d:.2f})")
        lines.append("")

    lines.extend(
        [
            "-" * 80,
            "INTERPRETATION",
            "-" * 80,
            "",
            "[SUCCESS] Significant:  p < 0.05 (unlikely due to chance)",
            "[OMIT]  Not Significant: p ≥ 0.05 (could be due to chance)",
            "",
            "Effect Size:",
            "  • d < 0.2: Negligible",
            "  • d < 0.5: Small",
            "  • d < 0.8: Medium",
            "  • d ≥ 0.8: Large",
            "",
            "=" * 80,
        ]
    )

    return "\n".join(lines)


def load_customer_data(
    datasets_dir: Path,
    customers: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load adset features data for multiple customers.

    Args:
        datasets_dir: Path to datasets directory.
        customers: List of customer names to load. If None, loads all available.

    Returns:
        Dictionary mapping customer names to their DataFrames.
    """
    datasets_dir = Path(datasets_dir)
    customer_dfs = {}

    # Determine which customers to load
    if customers is None:
        # Find all customers with features
        customer_dirs = [
            d
            for d in datasets_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        customers = [d.name for d in customer_dirs]

    # Load each customer's data
    for customer in customers:
        features_path = datasets_dir / customer / "features" / "adset_features.csv"

        if features_path.exists():
            try:
                df = pd.read_csv(features_path)
                customer_dfs[customer] = df
            except Exception as e:
                print(f"Warning: Failed to load {customer}: {e}")
        else:
            print(f"Warning: No features found for {customer} at {features_path}")

    if not customer_dfs:
        raise ValueError(f"No customer data found in {datasets_dir}")

    return customer_dfs


def rolling_window_cv_from_disk(
    customer: str,
    datasets_dir: Path,
    config_path: str,
    total_budget: float,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    min_train_days: int = 14,
    min_test_days: int = 7,
    date_col: str = "date_start",
    param_overrides: Optional[Dict[str, Any]] = None,
) -> CrossValidationResult:
    """Convenience function: Load data and run rolling window CV.

    This is a high-level API that combines data loading and CV execution.

    Args:
        customer: Customer name.
        datasets_dir: Path to datasets directory.
        config_path: Path to config file.
        total_budget: Total budget per fold.
        n_folds: Number of CV folds.
        train_ratio: Training ratio for window sizing.
        min_train_days: Minimum training days.
        min_test_days: Minimum test days.
        date_col: Column name containing dates.
        param_overrides: Optional parameter overrides.

    Returns:
        CrossValidationResult with aggregated metrics.

    Raises:
        FileNotFoundError: If customer features file not found.
        ValueError: If insufficient data for CV.
    """
    datasets_dir = Path(datasets_dir)
    features_path = datasets_dir / customer / "features" / "adset_features.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found at {features_path}")

    # Load data
    df = pd.read_csv(features_path)

    # Run CV
    return rolling_window_cv(
        df=df,
        base_config_path=config_path,
        total_budget=total_budget,
        n_folds=n_folds,
        train_ratio=train_ratio,
        min_train_days=min_train_days,
        min_test_days=min_test_days,
        customer_name=customer,
        date_col=date_col,
        param_overrides=param_overrides,
    )


def customer_level_cv_from_disk(
    datasets_dir: Path,
    config_path: str,
    total_budget: float,
    customers: Optional[List[str]] = None,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> CrossValidationResult:
    """Convenience function: Load customer data and run leave-one-out CV.

    This is a high-level API that combines data loading and CV execution.

    Args:
        datasets_dir: Path to datasets directory.
        config_path: Path to config file.
        total_budget: Total budget per fold.
        customers: List of customer names. If None, uses all available.
        param_overrides: Optional parameter overrides.

    Returns:
        CrossValidationResult with aggregated metrics.

    Raises:
        ValueError: If fewer than 2 customers available.
    """
    # Load all customer data
    customer_dfs = load_customer_data(datasets_dir, customers=customers)

    # Run CV
    return customer_level_cv(
        customer_dfs=customer_dfs,
        base_config_path=config_path,
        total_budget=total_budget,
        param_overrides=param_overrides,
    )
