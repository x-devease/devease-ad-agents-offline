"""
Results aggregation utilities for diagnoser evaluation scripts.

This module provides functions to aggregate evaluation results across
multiple windows and calculate summary statistics.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def aggregate_results(results: List[Any], detector_name: str) -> Optional[Dict[str, Any]]:
    """
    Aggregate evaluation results across multiple windows.

    Sums TP, FP, FN across all windows and calculates overall
    precision, recall, and F1 score. Also calculates score statistics.

    Args:
        results: List of evaluation result objects from evaluator
        detector_name: Name of the detector

    Returns:
        Aggregation dictionary with:
            - detector_name: Name of detector
            - total_windows: Number of windows
            - accuracy: Dict with total_tp, total_fp, total_fn, precision, recall, f1_score
            - scores: Dict with avg, min, max, std
            - window_results: List of individual result objects
        Returns None if results list is empty
    """
    if not results:
        logger.warning("No results to aggregate")
        return None

    # Sum accuracy metrics across windows
    total_tp = sum(r.accuracy.true_positives for r in results)
    total_fp = sum(r.accuracy.false_positives for r in results)
    total_fn = sum(r.accuracy.false_negatives for r in results)

    # Calculate aggregated metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate score statistics
    scores = [r.overall_score for r in results]

    aggregation = {
        'detector_name': detector_name,
        'total_windows': len(results),
        'accuracy': {
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        },
        'scores': {
            'avg': float(np.mean(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'std': float(np.std(scores)),
        },
        'window_results': results,
    }

    return aggregation


def calculate_metrics_summary(aggregation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary metrics from aggregated results.

    Extracts key metrics into a flat dictionary for easy reporting.

    Args:
        aggregation: Aggregation dictionary from aggregate_results()

    Returns:
        Flat dictionary with summary metrics
    """
    if aggregation is None:
        return {}

    accuracy = aggregation.get('accuracy', {})
    scores = aggregation.get('scores', {})

    summary = {
        'detector_name': aggregation.get('detector_name'),
        'total_windows': aggregation.get('total_windows', 0),
        'total_tp': accuracy.get('total_tp', 0),
        'total_fp': accuracy.get('total_fp', 0),
        'total_fn': accuracy.get('total_fn', 0),
        'precision': accuracy.get('precision', 0),
        'recall': accuracy.get('recall', 0),
        'f1_score': accuracy.get('f1_score', 0),
        'avg_score': scores.get('avg', 0),
        'min_score': scores.get('min', 0),
        'max_score': scores.get('max', 0),
        'std_score': scores.get('std', 0),
    }

    return summary


def format_metrics_for_display(summary: Dict[str, Any]) -> str:
    """
    Format metrics summary for display/logging.

    Args:
        summary: Summary dictionary from calculate_metrics_summary()

    Returns:
        Formatted string with metrics
    """
    lines = [
        f"Detector: {summary.get('detector_name', 'Unknown')}",
        f"Windows: {summary.get('total_windows', 0)}",
        "",
        "Accuracy:",
        f"  Precision: {summary.get('precision', 0):.2%}",
        f"  Recall: {summary.get('recall', 0):.2%}",
        f"  F1-Score: {summary.get('f1_score', 0):.2%}",
        f"  TP: {summary.get('total_tp', 0)}, FP: {summary.get('total_fp', 0)}, FN: {summary.get('total_fn', 0)}",
        "",
        "Scores:",
        f"  Average: {summary.get('avg_score', 0):.1f}",
        f"  Range: {summary.get('min_score', 0):.1f} - {summary.get('max_score', 0):.1f}",
        f"  Std Dev: {summary.get('std_score', 0):.1f}",
    ]

    return "\n".join(lines)


def compare_aggregations(
    aggregations: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare multiple detector aggregations.

    Creates a DataFrame comparing metrics across multiple detectors.

    Args:
        aggregations: Dictionary mapping detector names to aggregation dicts

    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []

    for detector_name, agg in aggregations.items():
        if agg is None:
            continue

        summary = calculate_metrics_summary(agg)
        summary['detector_name'] = detector_name
        comparison_data.append(summary)

    if not comparison_data:
        return pd.DataFrame()

    df = pd.DataFrame(comparison_data)

    # Reorder columns for better readability
    column_order = [
        'detector_name', 'total_windows', 'f1_score', 'precision', 'recall',
        'avg_score', 'min_score', 'max_score', 'std_score',
        'total_tp', 'total_fp', 'total_fn'
    ]

    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]

    return df[column_order]
