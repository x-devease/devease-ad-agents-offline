"""
Metrics utilities for diagnoser evaluation scripts.

This module provides functions to load metrics from evaluation reports,
check if targets are satisfied, and format metrics for reporting.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Default detector to report file mapping
DETECTOR_REPORT_MAP = {
    "FatigueDetector": "fatigue_sliding_10windows.json",
    "LatencyDetector": "latency_sliding_10windows.json",
    "DarkHoursDetector": "dark_hours_sliding_10windows.json",
}

# Default report path
DEFAULT_REPORT_PATH = Path("src/meta/diagnoser/evaluator/reports/moprobo_sliding")


def load_metrics(
    detector_name: str,
    report_path: Optional[Path] = None,
    detector_map: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Load metrics for a detector from evaluation report.

    Handles two report formats:
    - New format: Has "aggregated_metrics" key (DarkHoursDetector)
    - Old format: Has "accuracy" key (LatencyDetector, FatigueDetector)

    Args:
        detector_name: Name of the detector (e.g., "FatigueDetector")
        report_path: Path to reports directory (default: DEFAULT_REPORT_PATH)
        detector_map: Optional custom mapping from detector names to report files

    Returns:
        Dictionary with keys: precision, recall, f1_score, tp, fp, fn
        Returns None if report file not found
    """
    if detector_map is None:
        detector_map = DETECTOR_REPORT_MAP

    if report_path is None:
        report_path = DEFAULT_REPORT_PATH

    report_file = detector_map.get(detector_name)
    if not report_file:
        logger.warning(f"No report file mapping for detector: {detector_name}")
        return None

    full_path = report_path / report_file

    if not full_path.exists():
        logger.warning(f"Report file not found: {full_path}")
        return None

    try:
        with open(full_path, 'r') as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load report from {full_path}: {e}")
        return None

    # Extract metrics based on report format
    if "aggregated_metrics" in report:
        # New format (DarkHoursDetector)
        metrics = report["aggregated_metrics"]
        return {
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1_score": metrics.get("f1_score", 0),
            "tp": metrics.get("total_tp", 0),
            "fp": metrics.get("total_fp", 0),
            "fn": metrics.get("total_fn", 0)
        }
    else:
        # Old format (LatencyDetector, FatigueDetector)
        accuracy = report.get("accuracy", {})
        return {
            "precision": accuracy.get("precision", 0),
            "recall": accuracy.get("recall", 0),
            "f1_score": accuracy.get("f1_score", 0),
            "tp": accuracy.get("total_tp", 0),
            "fp": accuracy.get("total_fp", 0),
            "fn": accuracy.get("total_fn", 0)
        }


def check_satisfaction(
    metrics: Dict[str, Any],
    targets: Dict[str, float]
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Check if metrics meet target thresholds.

    Compares each metric in targets against the corresponding value in metrics.
    A metric is satisfied if current >= target.

    Args:
        metrics: Dictionary with current metric values (from load_metrics)
        targets: Dictionary mapping metric names to target values

    Returns:
        Tuple of:
            - all_satisfied: True if all metrics meet targets
            - satisfaction: Dict with details for each metric:
                - current: Current value
                - target: Target value
                - satisfied: Boolean indicating if target met
    """
    satisfaction = {}
    all_satisfied = True

    for metric, target_value in targets.items():
        current_value = metrics.get(metric, 0)
        is_satisfied = current_value >= target_value

        satisfaction[metric] = {
            "current": current_value,
            "target": target_value,
            "satisfied": is_satisfied
        }

        if not is_satisfied:
            all_satisfied = False

    return all_satisfied, satisfaction


def format_metrics_report(
    detector_name: str,
    metrics: Dict[str, Any],
    targets: Optional[Dict[str, float]] = None
) -> str:
    """
    Format metrics for display/reporting.

    Args:
        detector_name: Name of the detector
        metrics: Dictionary with metric values
        targets: Optional target values for comparison

    Returns:
        Formatted string with metrics
    """
    lines = [
        f"Detector: {detector_name}",
        "",
        "Metrics:",
        f"  Precision: {metrics.get('precision', 0):.4f}",
        f"  Recall: {metrics.get('recall', 0):.4f}",
        f"  F1-Score: {metrics.get('f1_score', 0):.4f}",
        f"  TP: {metrics.get('tp', 0)}, FP: {metrics.get('fp', 0)}, FN: {metrics.get('fn', 0)}",
    ]

    if targets:
        lines.append("")
        lines.append("Targets:")

        all_satisfied, satisfaction = check_satisfaction(metrics, targets)

        for metric, details in satisfaction.items():
            status = "✅" if details["satisfied"] else "❌"
            lines.append(
                f"  {status} {metric}: {details['current']:.4f} (target: {details['target']:.2f})"
            )

        lines.append("")
        if all_satisfied:
            lines.append("Status: ✅ ALL TARGETS SATISFIED")
        else:
            lines.append("Status: ❌ SOME TARGETS NOT MET")

    return "\n".join(lines)


def load_all_metrics(
    detectors: Optional[list[str]] = None,
    report_path: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load metrics for multiple detectors.

    Args:
        detectors: List of detector names (default: all detectors in map)
        report_path: Path to reports directory (default: DEFAULT_REPORT_PATH)

    Returns:
        Dictionary mapping detector names to metrics dictionaries
    """
    if detectors is None:
        detectors = list(DETECTOR_REPORT_MAP.keys())

    all_metrics = {}

    for detector_name in detectors:
        metrics = load_metrics(detector_name, report_path)
        if metrics:
            all_metrics[detector_name] = metrics

    return all_metrics
