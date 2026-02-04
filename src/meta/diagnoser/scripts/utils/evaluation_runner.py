"""
Evaluation runner utilities for diagnoser evaluation scripts.

This module provides functions to execute detector evaluation on
sliding windows and handle evaluation lifecycle.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_detector_on_windows(
    detector,
    windows: List[Dict[str, Any]],
    detector_name: str,
    label_method: str = "rule_based",
    evaluator_class=None
) -> List[Any]:
    """
    Evaluate detector on multiple sliding windows.

    Iterates through windows and runs evaluation for each window.
    Handles exceptions gracefully and logs progress.

    Args:
        detector: Detector instance to evaluate
        windows: List of window dictionaries from sliding_windows module
        detector_name: Name of the detector for logging
        label_method: Method for auto-generating labels (default: "rule_based")
        evaluator_class: Evaluator class (defaults to DiagnoserEvaluator)

    Returns:
        List of evaluation result objects
    """
    if evaluator_class is None:
        from src.meta.diagnoser.evaluator import DiagnoserEvaluator
        evaluator_class = DiagnoserEvaluator

    evaluator = evaluator_class()
    results = []

    logger.info(f"Evaluating {detector_name} on {len(windows)} windows...")

    for window in windows:
        window_data = window['data']
        window_num = window['window_num']

        try:
            result = evaluator.evaluate(
                detector=detector,
                test_data=window_data,
                detector_name=f"{detector_name}_W{window_num}",
                auto_label=True,
                label_method=label_method,
            )

            # Add window number to result details
            result.details['window_num'] = window_num
            results.append(result)

            logger.info(
                f"  Window {window_num}: Score={result.overall_score:.1f}, "
                f"TP={result.accuracy.true_positives}, "
                f"FP={result.accuracy.false_positives}, "
                f"FN={result.accuracy.false_negatives}"
            )

        except Exception as e:
            logger.error(f"  Error evaluating window {window_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def evaluate_single_window(
    detector,
    data: pd.DataFrame,
    detector_name: str,
    window_num: int = 0,
    label_method: str = "rule_based",
    evaluator_class=None
) -> Optional[Any]:
    """
    Evaluate detector on a single window of data.

    Args:
        detector: Detector instance to evaluate
        data: Test data for evaluation
        detector_name: Name of the detector
        window_num: Window index for logging (default: 0)
        label_method: Method for auto-generating labels (default: "rule_based")
        evaluator_class: Evaluator class (defaults to DiagnoserEvaluator)

    Returns:
        Evaluation result object, or None if evaluation fails
    """
    if evaluator_class is None:
        from src.meta.diagnoser.evaluator import DiagnoserEvaluator
        evaluator_class = DiagnoserEvaluator

    evaluator = evaluator_class()

    try:
        result = evaluator.evaluate(
            detector=detector,
            test_data=data,
            detector_name=f"{detector_name}_W{window_num}",
            auto_label=True,
            label_method=label_method,
        )

        result.details['window_num'] = window_num

        logger.info(
            f"Window {window_num}: Score={result.overall_score:.1f}, "
            f"TP={result.accuracy.true_positives}, "
            f"FP={result.accuracy.false_positives}, "
            f"FN={result.accuracy.false_negatives}"
        )

        return result

    except Exception as e:
        logger.error(f"Error evaluating window {window_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_multiple_detectors(
    detectors: Dict[str, Any],
    windows: List[Dict[str, Any]],
    label_method: str = "rule_based",
    evaluator_class=None
) -> Dict[str, List[Any]]:
    """
    Evaluate multiple detectors on the same windows.

    Args:
        detectors: Dictionary mapping detector names to detector instances
        windows: List of window dictionaries from sliding_windows module
        label_method: Method for auto-generating labels (default: "rule_based")
        evaluator_class: Evaluator class (defaults to DiagnoserEvaluator)

    Returns:
        Dictionary mapping detector names to lists of evaluation results
    """
    all_results = {}

    for detector_name, detector in detectors.items():
        logger.info(f"\nEvaluating {detector_name}...")
        results = evaluate_detector_on_windows(
            detector=detector,
            windows=windows,
            detector_name=detector_name,
            label_method=label_method,
            evaluator_class=evaluator_class
        )
        all_results[detector_name] = results

    return all_results
