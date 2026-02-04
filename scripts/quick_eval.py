#!/usr/bin/env python3
"""
Quick Iterative Evaluation Script - Smaller subset for faster feedback.

Uses 30-day sliding window for daily data (smaller subset) for quick iteration.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.detectors import LatencyDetector
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


def quick_evaluation():
    """快速评估 - 只测试LatencyDetector"""
    logger.info("=" * 60)
    logger.info("Quick Iterative Evaluation - Iteration 1")
    logger.info("=" * 60)

    # 1. 加载数据
    daily_data = load_moprobo_data()
    logger.info(f"Loaded {len(daily_data)} rows")

    # 2. 生成少数几个窗口用于快速测试
    min_date = daily_data['date'].min()
    max_date = daily_data['date'].max()

    # 只生成3个窗口用于快速测试
    windows = []
    for i in range(3):
        window_start = min_date + timedelta(days=i*20)
        window_end = window_start + timedelta(days=29)

        window_data = daily_data[
            (daily_data['date'] >= window_start) &
            (daily_data['date'] <= window_end)
        ].copy()

        if len(window_data) > 0:
            windows.append({
                'window_num': i,
                'start_date': window_start,
                'end_date': window_end,
                'data': window_data
            })

    logger.info(f"Generated {len(windows)} test windows")

    # 3. 评估LatencyDetector
    detector = LatencyDetector(config={
        "thresholds": {
            "roas_threshold": 1.0,
            "rolling_window_days": 3,
            "min_daily_spend": 50,
            "min_drop_ratio": 0.2,
        }
    })

    evaluator = DiagnoserEvaluator()
    results = []

    for window in windows:
        logger.info(f"\nEvaluating window {window['window_num']}: {window['start_date'].date()} to {window['end_date'].date()}")

        try:
            result = evaluator.evaluate(
                detector=detector,
                test_data=window['data'],
                detector_name=f"LatencyDetector_W{window['window_num']}",
                auto_label=True,
                label_method="rule_based",  # 使用rule_based方法，它会调用_apply_latency_rules
            )

            result.details['window_num'] = window['window_num']
            results.append(result)

            logger.info(f"  Score: {result.overall_score:.1f}/100")
            logger.info(f"  TP: {result.accuracy.true_positives}, FP: {result.accuracy.false_positives}, FN: {result.accuracy.false_negatives}")
            logger.info(f"  Labels: {result.details.get('label_count', 0)}")

        except Exception as e:
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # 4. 聚合结果
    if results:
        total_tp = sum(r.accuracy.true_positives for r in results)
        total_fp = sum(r.accuracy.false_positives for r in results)
        total_fn = sum(r.accuracy.false_negatives for r in results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        scores = [r.overall_score for r in results]

        logger.info("\n" + "=" * 60)
        logger.info("AGGREGATED RESULTS")
        logger.info("=" * 60)
        logger.info(f"Windows: {len(results)}")
        logger.info(f"Precision: {precision:.2%}")
        logger.info(f"Recall: {recall:.2%}")
        logger.info(f"F1-Score: {f1_score:.2%}")
        logger.info(f"Avg Score: {np.mean(scores):.1f}/100")
        logger.info(f"Score Range: {np.min(scores):.1f} - {np.max(scores):.1f}")
        logger.info(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

        # 5. 保存报告
        reporter = EvaluationReporter(customer="moprobo_iter")
        summary = {
            'iteration': 1,
            'detector': 'LatencyDetector',
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
            'window_results': [
                {
                    'window_num': r.details.get('window_num'),
                    'score': r.overall_score,
                    'tp': r.accuracy.true_positives,
                    'fp': r.accuracy.false_positives,
                    'fn': r.accuracy.false_negatives,
                    'labels': r.details.get('label_count', 0),
                }
                for r in results
            ]
        }

        import json
        summary_json = json.dumps(summary, indent=2, default=str)
        reporter.save_report(summary_json, "iteration_1.json")

        logger.info(f"\nReport saved to: src/meta/diagnoser/judge/reports/moprobo_iter/iteration_1.json")

    return results


if __name__ == "__main__":
    exit(0 if quick_evaluation() else 1)
