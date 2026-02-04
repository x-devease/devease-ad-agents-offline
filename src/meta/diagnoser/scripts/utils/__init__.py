"""
Shared utility modules for diagnoser evaluation scripts.

This package provides common functionality for data loading, window generation,
evaluation execution, and results aggregation to eliminate code duplication.
"""

from .data_loader import load_moprobo_data, preprocess_daily_data, preprocess_hourly_data
from .sliding_windows import generate_sliding_windows_daily, generate_sliding_windows_hourly
from .evaluation_runner import evaluate_detector_on_windows, evaluate_single_window
from .results_aggregator import aggregate_results, calculate_metrics_summary
from .metrics_utils import load_metrics, check_satisfaction, format_metrics_report

__all__ = [
    # Data loading
    "load_moprobo_data",
    "preprocess_daily_data",
    "preprocess_hourly_data",
    # Window generation
    "generate_sliding_windows_daily",
    "generate_sliding_windows_hourly",
    # Evaluation
    "evaluate_detector_on_windows",
    "evaluate_single_window",
    # Aggregation
    "aggregate_results",
    "calculate_metrics_summary",
    # Metrics
    "load_metrics",
    "check_satisfaction",
    "format_metrics_report",
]
