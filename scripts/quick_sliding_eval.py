#!/usr/bin/env python3
"""
Streamlined Sliding Window Evaluation for Quick Iteration.

Evaluates on 10 daily windows (instead of 49) for faster feedback.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.detectors import LatencyDetector, FatigueDetector
from src.meta.diagnoser.judge import DiagnoserEvaluator, ZeroCostLabelGenerator, EvaluationReporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_moprobo_data():
    """加载并预处理moprobo数据"""
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


def generate_sliding_windows_daily(daily_data, window_days=30, step_days=7, max_windows=10):
    """生成daily数据的滑动窗口（限制数量）"""
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


def evaluate_detector_on_windows(detector, windows, detector_name, label_method="rule_based"):
    """在多个滑动窗口上评估detector"""
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
            continue

    return results


def aggregate_results(results, detector_name):
    """聚合多个窗口的评估结果"""
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
    """主函数 - 第5次迭代"""
    logger.info("=" * 80)
    logger.info("Quick Sliding Window Evaluation - Iteration 5")
    logger.info("=" * 80)

    # 1. 加载数据
    daily_data = load_moprobo_data()
    logger.info(f"Loaded {len(daily_data)} rows")

    # 2. 生成滑动窗口（限制10个）
    windows = generate_sliding_windows_daily(daily_data, window_days=30, step_days=7, max_windows=10)

    if not windows:
        logger.error("No windows generated")
        return 1

    # 3. 评估LatencyDetector
    latency_detector = LatencyDetector(config={
        "thresholds": {
            "roas_threshold": 1.0,
            "rolling_window_days": 3,
            "min_daily_spend": 50,
            "min_drop_ratio": 0.2,
        }
    })

    latency_results = evaluate_detector_on_windows(
        latency_detector, windows, "LatencyDetector", "rule_based"
    )

    if latency_results:
        latency_agg = aggregate_results(latency_results, "LatencyDetector")

        logger.info("\n" + "=" * 80)
        logger.info("LATENCY DETECTOR - AGGREGATED RESULTS")
        logger.info("=" * 80)
        logger.info(f"Windows: {latency_agg['total_windows']}")
        logger.info(f"Precision: {latency_agg['accuracy']['precision']:.2%}")
        logger.info(f"Recall: {latency_agg['accuracy']['recall']:.2%}")
        logger.info(f"F1-Score: {latency_agg['accuracy']['f1_score']:.2%}")
        logger.info(f"Avg Score: {latency_agg['scores']['avg']:.1f}/100")
        logger.info(f"Score Range: {latency_agg['scores']['min']:.1f} - {latency_agg['scores']['max']:.1f}")
        logger.info(f"Total TP: {latency_agg['accuracy']['total_tp']}, FP: {latency_agg['accuracy']['total_fp']}, FN: {latency_agg['accuracy']['total_fn']}")

        # 保存报告
        reporter = EvaluationReporter(customer="moprobo_sliding")
        import json
        summary_json = json.dumps(latency_agg, indent=2, default=str)
        reporter.save_report(summary_json, "latency_sliding_10windows.json")

        logger.info(f"\nReport saved to: src/meta/diagnoser/judge/reports/moprobo_sliding/latency_sliding_10windows.json")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Evaluation Complete!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
