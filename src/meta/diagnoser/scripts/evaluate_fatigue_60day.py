#!/usr/bin/env python3
"""
Evaluate FatigueDetector using 60-day sliding window.
FatigueDetector needs 33+ days of data (30 + 3 consecutive).
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.detectors import FatigueDetector
from src.meta.diagnoser.judge import DiagnoserEvaluator, EvaluationReporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_moprobo_data():
    """Load and preprocess moprobo data."""
    import json

    daily_path = Path("datasets/moprobo/meta/raw/ad_daily_insights_2024-12-17_2025-12-17.csv")
    ad_daily = pd.read_csv(daily_path)

    numeric_cols = ['spend', 'impressions', 'reach', 'clicks', 'actions']
    for col in numeric_cols:
        if col in ad_daily.columns:
            ad_daily[col] = pd.to_numeric(ad_daily[col], errors='coerce').fillna(0)

    if 'purchase_roas' in ad_daily.columns:
        def extract_roas_value(roas_str):
            if pd.isna(roas_str) or roas_str == '':
                return 0.0
            try:
                data = json.loads(roas_str)
                if isinstance(data, list) and len(data) > 0:
                    return float(data[0].get('value', 0))
                return 0.0
            except:
                return 0.0

        ad_daily['purchase_roas'] = ad_daily['purchase_roas'].apply(extract_roas_value)

    if 'date_start' in ad_daily.columns:
        ad_daily['date'] = pd.to_datetime(ad_daily['date_start'], errors='coerce')
        ad_daily = ad_daily.sort_values('date').dropna(subset=['date'])

    return ad_daily


def generate_sliding_windows_daily(daily_data, window_days=60, step_days=14, max_windows=5):
    """Generate 60-day sliding windows for FatigueDetector."""
    windows = []

    if len(daily_data) < window_days:
        return windows

    daily_data = daily_data.sort_values('date')
    min_date = daily_data['date'].min()
    max_date = daily_data['date'].max()

    current_start = min_date
    window_num = 0

    while current_start + timedelta(days=window_days) <= max_date + timedelta(days=1) and window_num < max_windows:
        current_end = current_start + timedelta(days=window_days - 1)

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

        current_start = current_start + timedelta(days=step_days)

    logger.info(f"Generated {len(windows)} daily windows (window_size={window_days} days)")
    return windows


def evaluate_detector_on_windows(detector, windows, detector_name, label_method="rule_based"):
    """Evaluate detector on multiple sliding windows."""
    evaluator = DiagnoserEvaluator()
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

            result.details['window_num'] = window_num
            results.append(result)

            logger.info(f"  Window {window_num}: Score={result.overall_score:.1f}, TP={result.accuracy.true_positives}, FP={result.accuracy.false_positives}, FN={result.accuracy.false_negatives}")

        except Exception as e:
            logger.error(f"  Error evaluating window {window_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def aggregate_results(results, detector_name):
    """Aggregate evaluation results across multiple windows."""
    if not results:
        return None

    total_tp = sum(r.accuracy.true_positives for r in results)
    total_fp = sum(r.accuracy.false_positives for r in results)
    total_fn = sum(r.accuracy.false_negatives for r in results)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

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


def main():
    """Main function - Iteration 6: FatigueDetector Evaluation with 60-day windows."""
    logger.info("=" * 80)
    logger.info("FatigueDetector Sliding Window Evaluation - Iteration 6 (60-day windows)")
    logger.info("=" * 80)

    # 1. Load data
    daily_data = load_moprobo_data()
    logger.info(f"Loaded {len(daily_data)} rows")

    # 2. Generate 60-day sliding windows (5 windows)
    windows = generate_sliding_windows_daily(daily_data, window_days=60, step_days=14, max_windows=5)

    if not windows:
        logger.error("No windows generated")
        return 1

    # 3. Evaluate FatigueDetector
    fatigue_detector = FatigueDetector(config={
        "thresholds": {
            "cpa_increase_threshold": 0.3,  # 30% CPA increase
            "golden_period_hours": 24,  # Golden period: first 24 hours
            "min_conversions_golden": 5,  # Minimum conversions in golden period
            "min_daily_conversions": 3,  # Minimum daily conversions for fatigue detection
        }
    })

    fatigue_results = evaluate_detector_on_windows(
        fatigue_detector, windows, "FatigueDetector", "rule_based"
    )

    if fatigue_results:
        fatigue_agg = aggregate_results(fatigue_results, "FatigueDetector")

        logger.info("\n" + "=" * 80)
        logger.info("FATIGUE DETECTOR - AGGREGATED RESULTS")
        logger.info("=" * 80)
        logger.info(f"Windows: {fatigue_agg['total_windows']}")
        logger.info(f"Precision: {fatigue_agg['accuracy']['precision']:.2%}")
        logger.info(f"Recall: {fatigue_agg['accuracy']['recall']:.2%}")
        logger.info(f"F1-Score: {fatigue_agg['accuracy']['f1_score']:.2%}")
        logger.info(f"Avg Score: {fatigue_agg['scores']['avg']:.1f}/100")
        logger.info(f"Score Range: {fatigue_agg['scores']['min']:.1f} - {fatigue_agg['scores']['max']:.1f}")
        logger.info(f"Total TP: {fatigue_agg['accuracy']['total_tp']}, FP: {fatigue_agg['accuracy']['total_fp']}, FN: {fatigue_agg['accuracy']['total_fn']}")

        # Save report
        reporter = EvaluationReporter(customer="moprobo_sliding")
        import json
        summary_json = json.dumps(fatigue_agg, indent=2, default=str)
        reporter.save_report(summary_json, "fatigue_sliding_5windows_60day.json")

        logger.info(f"\nReport saved to: src/meta/diagnoser/judge/reports/moprobo_sliding/fatigue_sliding_5windows_60day.json")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Evaluation Complete!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
