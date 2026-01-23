"""Optimizer module for budget allocation tuning and evaluation.

Note: ML predictor models have been removed in favor of rules-based approach.
The module now focuses on tuning utilities and backtesting for rule-based allocation.
"""

# Library modules for tuning and evaluation
from .lib import (
    aggregate_cv_results,
    compare_cv_results,
    create_rolling_window_splits,
    customer_level_cv,
    evaluate_with_cross_validation,
    forward_looking_evaluation,
    rolling_window_cv,
    temporal_train_test_split,
)

# Base classes and utilities
from .tuning import (
    TuningConstraints,
    TuningResult,
    evaluate_allocation,
    simulate_allocation,
)

__all__ = [
    # Library
    "aggregate_cv_results",
    "compare_cv_results",
    "create_rolling_window_splits",
    "customer_level_cv",
    "evaluate_with_cross_validation",
    "forward_looking_evaluation",
    "rolling_window_cv",
    "temporal_train_test_split",
    # Base
    "TuningConstraints",
    "TuningResult",
    "simulate_allocation",
    "evaluate_allocation",
]
