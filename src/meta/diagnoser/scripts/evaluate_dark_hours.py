#!/usr/bin/env python3
"""
Evaluate DarkHoursDetector using sliding window approach.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.detectors import DarkHoursDetector
from src.meta.diagnoser.judge import DiagnoserEvaluator, EvaluationReporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_moprobo_data():
    """Load and preprocess moprobo data."""
    import json

    # Load daily data
    daily_path = Path("datasets/moprobo/meta/raw/ad_daily_insights_2024-12-17_2025-12-17.csv")
    ad_daily = pd.read_csv(daily_path)

    # Don't convert 'actions' to numeric - it contains JSON!
    numeric_cols = ['spend', 'impressions', 'reach', 'clicks']
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

    # Load hourly data
    hourly_path = Path("datasets/moprobo/meta/raw/ad_hourly_insights_2024-12-17_2025-12-17.csv")
    try:
        ad_hourly = pd.read_csv(hourly_path)

        numeric_cols = ['spend', 'impressions', 'reach', 'clicks']
        for col in numeric_cols:
            if col in ad_hourly.columns:
                ad_hourly[col] = pd.to_numeric(ad_hourly[col], errors='coerce').fillna(0)

        if 'purchase_roas' in ad_hourly.columns:
            ad_hourly['purchase_roas'] = ad_hourly['purchase_roas'].apply(extract_roas_value)

        if 'date_start' in ad_hourly.columns:
            ad_hourly['date'] = pd.to_datetime(ad_hourly['date_start'], errors='coerce')
            ad_hourly = ad_hourly.sort_values('date').dropna(subset=['date'])

        logger.info(f"Loaded {len(ad_hourly)} hourly rows")
    except Exception as e:
        logger.warning(f"Could not load hourly data: {e}")
        ad_hourly = None

    return ad_daily, ad_hourly


def generate_sliding_windows_daily(daily_data, window_days=30, step_days=7, max_windows=10):
    """Generate sliding windows for daily data (limited count)."""
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

    logger.info(f"Generated {len(windows)} daily windows (limited to {max_windows})")
    return windows


def main():
    """Main function - Iteration 7: DarkHoursDetector Evaluation."""
    logger.info("=" * 80)
    logger.info("DarkHoursDetector Sliding Window Evaluation - Iteration 7")
    logger.info("=" * 80)

    # 1. Load data
    daily_data, hourly_data = load_moprobo_data()
    logger.info(f"Loaded {len(daily_data)} daily rows")

    # 2. Generate sliding windows (10 windows)
    windows = generate_sliding_windows_daily(daily_data, window_days=30, step_days=7, max_windows=10)

    if not windows:
        logger.error("No windows generated")
        return 1

    # 3. Evaluate DarkHoursDetector
    # DarkHoursDetector uses daily data for day-of-week analysis
    detector = DarkHoursDetector()

    evaluator = DiagnoserEvaluator()
    results = []

    logger.info(f"Evaluating DarkHoursDetector on {len(windows)} windows...")

    for window in windows:
        window_data = window['data']
        window_num = window['window_num']

        try:
            # DarkHoursDetector doesn't use auto_labeling - it analyzes time patterns
            # We'll use rule_based method but need to adapt for DarkHours
            result = evaluator.evaluate(
                detector=detector,
                test_data=window_data,
                detector_name=f"DarkHoursDetector_W{window_num}",
                auto_label=True,
                label_method="statistical_anomaly",  # Use statistical method for time-based patterns
            )

            result.details['window_num'] = window_num
            results.append(result)

            logger.info(f"  Window {window_num}: Score={result.overall_score:.1f}, TP={result.accuracy.true_positives}, FP={result.accuracy.false_positives}, FN={result.accuracy.false_negatives}")

        except Exception as e:
            logger.error(f"  Error evaluating window {window_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 4. Aggregate results
    if results:
        total_tp = sum(r.accuracy.true_positives for r in results)
        total_fp = sum(r.accuracy.false_positives for r in results)
        total_fn = sum(r.accuracy.false_negatives for r in results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        scores = [r.overall_score for r in results]

        logger.info("\n" + "=" * 80)
        logger.info("DARK HOURS DETECTOR - AGGREGATED RESULTS")
        logger.info("=" * 80)
        logger.info(f"Windows: {len(results)}")
        logger.info(f"Precision: {precision:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"F1-Score: {f1_score:.2%}")
        logger.info(f"Avg Score: {np.mean(scores):.1f}/100")
        logger.info(f"Score Range: {np.min(scores):.1f} - {np.max(scores):.1f}")
        logger.info(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

        # Save report
        reporter = EvaluationReporter(customer="moprobo_sliding")
        import json
        summary = {
            'iteration': 7,
            'detector': 'DarkHoursDetector',
            'total_windows': len(results),
            'aggregated_metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'avg_score': float(np.mean(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
            },
        }

        summary_json = json.dumps(summary, indent=2, default=str)
        reporter.save_report(summary_json, "dark_hours_sliding_10windows.json")

        logger.info(f"\nReport saved to: src/meta/diagnoser/judge/reports/moprobo_sliding/dark_hours_sliding_10windows.json")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Evaluation Complete!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
