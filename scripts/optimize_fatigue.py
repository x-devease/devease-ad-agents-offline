#!/usr/bin/env python3
"""
Optimize FatigueDetector thresholds - Iteration 9.
Test different threshold combinations to improve recall while maintaining good precision.
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

    return ad_daily


def generate_sliding_windows_daily(daily_data, window_days=30, step_days=7, max_windows=5):
    """Generate sliding windows for daily data (smaller subset for faster testing)."""
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

    logger.info(f"Generated {len(windows)} daily windows")
    return windows


def test_threshold_config(detector_class, config, windows, label_method="rule_based"):
    """Test a specific threshold configuration."""
    detector = detector_class(config=config)
    evaluator = DiagnoserEvaluator()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_score = 0
    valid_windows = 0

    for window in windows:
        try:
            result = evaluator.evaluate(
                detector=detector,
                test_data=window['data'],
                detector_name=f"Test_W{window['window_num']}",
                auto_label=True,
                label_method=label_method,
            )

            total_tp += result.accuracy.true_positives
            total_fp += result.accuracy.false_positives
            total_fn += result.accuracy.false_negatives
            total_score += result.overall_score
            valid_windows += 1

        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_score = total_score / valid_windows if valid_windows > 0 else 0

    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_score': avg_score,
        'windows': valid_windows,
    }


def main():
    """Main function - Iteration 9: Optimize FatigueDetector thresholds."""
    logger.info("=" * 80)
    logger.info("FatigueDetector Threshold Optimization - Iteration 9")
    logger.info("=" * 80)

    # 1. Load data
    daily_data = load_moprobo_data()
    logger.info(f"Loaded {len(daily_data)} rows")

    # 2. Generate sliding windows (5 windows for faster testing)
    windows = generate_sliding_windows_daily(daily_data, window_days=30, step_days=7, max_windows=5)

    if not windows:
        logger.error("No windows generated")
        return 1

    # 3. Test different threshold configurations
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT THRESHOLD CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Config':<50} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 80)

    # Baseline (current config)
    baseline_config = {}  # Use defaults
    baseline = test_threshold_config(FatigueDetector, baseline_config, windows)
    print(f"{'Baseline (defaults)':<50} {baseline['precision']*100:>10.1f}%   {baseline['recall']*100:>10.1f}%   {baseline['f1_score']*100:>10.1f}%   {baseline['tp']:<6} {baseline['fp']:<6} {baseline['fn']:<6}")

    # Config 1: Reduce consecutive_days to 1
    config1 = {"thresholds": {"consecutive_days": 1}}
    result1 = test_threshold_config(FatigueDetector, config1, windows)
    print(f"{'Config 1: consecutive_days=1':<50} {result1['precision']*100:>10.1f}%   {result1['recall']*100:>10.1f}%   {result1['f1_score']*100:>10.1f}%   {result1['tp']:<6} {result1['fp']:<6} {result1['fn']:<6}")

    # Config 2: Reduce cpa_increase_threshold to 1.2 (20%)
    config2 = {"thresholds": {"cpa_increase_threshold": 1.2}}
    result2 = test_threshold_config(FatigueDetector, config2, windows)
    print(f"{'Config 2: cpa_increase_threshold=1.2':<50} {result2['precision']*100:>10.1f}%   {result2['recall']*100:>10.1f}%   {result2['f1_score']*100:>10.1f}%   {result2['tp']:<6} {result2['fp']:<6} {result2['fn']:<6}")

    # Config 3: Reduce min_golden_days to 2
    config3 = {"thresholds": {"min_golden_days": 2}}
    result3 = test_threshold_config(FatigueDetector, config3, windows)
    print(f"{'Config 3: min_golden_days=2':<50} {result3['precision']*100:>10.1f}%   {result3['recall']*100:>10.1f}%   {result3['f1_score']*100:>10.1f}%   {result3['tp']:<6} {result3['fp']:<6} {result3['fn']:<6}")

    # Config 4: Combine all improvements
    config4 = {"thresholds": {"consecutive_days": 1, "cpa_increase_threshold": 1.2, "min_golden_days": 2}}
    result4 = test_threshold_config(FatigueDetector, config4, windows)
    print(f"{'Config 4: ALL improvements':<50} {result4['precision']*100:>10.1f}%   {result4['recall']*100:>10.1f}%   {result4['f1_score']*100:>10.1f}%   {result4['tp']:<6} {result4['fp']:<6} {result4['fn']:<6}")

    # Config 5: Aggressive - consecutive_days=1, cpa_increase_threshold=1.15 (15%)
    config5 = {"thresholds": {"consecutive_days": 1, "cpa_increase_threshold": 1.15, "min_golden_days": 2}}
    result5 = test_threshold_config(FatigueDetector, config5, windows)
    print(f"{'Config 5: Aggressive (15% threshold)':<50} {result5['precision']*100:>10.1f}%   {result5['recall']*100:>10.1f}%   {result5['f1_score']*100:>10.1f}%   {result5['tp']:<6} {result5['fp']:<6} {result5['fn']:<6}")

    # 4. Select best configuration
    results = [
        ('Baseline', baseline),
        ('Config 1', result1),
        ('Config 2', result2),
        ('Config 3', result3),
        ('Config 4', result4),
        ('Config 5', result5),
    ]

    best_config = max(results, key=lambda x: x[1]['f1_score'])
    print("\n" + "=" * 80)
    print(f"BEST CONFIGURATION: {best_config[0]}")
    print("=" * 80)
    print(f"Precision: {best_config[1]['precision']*100:.2f}%")
    print(f"Recall: {best_config[1]['recall']*100:.2f}%")
    print(f"F1-Score: {best_config[1]['f1_score']*100:.2f}%")
    print(f"TP: {best_config[1]['tp']}, FP: {best_config[1]['fp']}, FN: {best_config[1]['fn']}")

    # 5. Save optimization report
    import json
    optimization_report = {
        'iteration': 9,
        'detector': 'FatigueDetector',
        'baseline': {
            'precision': baseline['precision'],
            'recall': baseline['recall'],
            'f1_score': baseline['f1_score'],
            'tp': baseline['tp'],
            'fp': baseline['fp'],
            'fn': baseline['fn'],
        },
        'best_config': best_config[0],
        'best_results': {
            'precision': best_config[1]['precision'],
            'recall': best_config[1]['recall'],
            'f1_score': best_config[1]['f1_score'],
            'tp': best_config[1]['tp'],
            'fp': best_config[1]['fp'],
            'fn': best_config[1]['fn'],
        },
        'all_configs': {
            name: {
                'precision': r['precision'],
                'recall': r['recall'],
                'f1_score': r['f1_score'],
                'tp': r['tp'],
                'fp': r['fp'],
                'fn': r['fn'],
            }
            for name, r in results
        }
    }

    reporter = EvaluationReporter(customer="moprobo_sliding")
    summary_json = json.dumps(optimization_report, indent=2, default=str)
    reporter.save_report(summary_json, "fatigue_optimization_5windows.json")

    logger.info(f"\nOptimization report saved to: src/meta/diagnoser/judge/reports/moprobo_sliding/fatigue_optimization_5windows.json")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Threshold Optimization Complete!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
