#!/usr/bin/env python3
"""
Iterative Evaluation Script with Sliding Window Backtest.

Uses 30-day sliding window for daily data and 24-hour sliding window for hourly data
to generate multiple evaluation data points for robust backtesting.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.detectors import (
    FatigueDetector,
    LatencyDetector,
    DarkHoursDetector,
)
from src.meta.diagnoser.judge import (
    DiagnoserEvaluator,
    ZeroCostLabelGenerator,
    EvaluationReporter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_moprobo_data():
    """加载并预处理moprobo数据"""
    logger.info("Loading moprobo data...")

    import json

    # 加载daily数据
    daily_path = Path("datasets/moprobo/meta/raw/ad_daily_insights_2024-12-17_2025-12-17.csv")
    if not daily_path.exists():
        logger.error(f"Daily data not found: {daily_path}")
        return None, None

    ad_daily = pd.read_csv(daily_path)

    # 预处理numeric列
    numeric_cols = ['spend', 'impressions', 'reach', 'clicks', 'actions']
    for col in numeric_cols:
        if col in ad_daily.columns:
            ad_daily[col] = pd.to_numeric(ad_daily[col], errors='coerce').fillna(0)

    # 特殊处理purchase_roas列（JSON格式）
    if 'purchase_roas' in ad_daily.columns:
        def extract_roas_value(roas_str):
            """从JSON字符串中提取ROAS数值"""
            if pd.isna(roas_str) or roas_str == '':
                return 0.0
            try:
                # 尝试解析JSON
                data = json.loads(roas_str)
                if isinstance(data, list) and len(data) > 0:
                    # 取第一个action的value
                    return float(data[0].get('value', 0))
                return 0.0
            except (json.JSONDecodeError, ValueError, TypeError):
                # 如果不是JSON，尝试直接转换为float
                try:
                    return float(roas_str)
                except (ValueError, TypeError):
                    return 0.0

        ad_daily['purchase_roas'] = ad_daily['purchase_roas'].apply(extract_roas_value)
        logger.info(f"Extracted purchase_roas from JSON: {ad_daily['purchase_roas'].notna().sum()} non-null values")

    # 确保日期列
    if 'date_start' in ad_daily.columns:
        ad_daily['date'] = pd.to_datetime(ad_daily['date_start'], errors='coerce')
        ad_daily = ad_daily.sort_values('date')
        ad_daily = ad_daily.dropna(subset=['date'])

    # 加载hourly数据
    hourly_path = Path("datasets/moprobo/meta/raw/ad_hourly_insights_2025-09-01_2025-12-11.csv")
    if hourly_path.exists():
        ad_hourly = pd.read_csv(hourly_path)

        for col in numeric_cols:
            if col in ad_hourly.columns:
                ad_hourly[col] = pd.to_numeric(ad_hourly[col], errors='coerce').fillna(0)

        # 同样处理hourly的purchase_roas
        if 'purchase_roas' in ad_hourly.columns:
            ad_hourly['purchase_roas'] = ad_hourly['purchase_roas'].apply(extract_roas_value)

        if 'date_start' in ad_hourly.columns and 'hour' in ad_hourly.columns:
            ad_hourly['datetime'] = pd.to_datetime(ad_hourly['date_start']) + pd.to_timedelta(ad_hourly['hour'], unit='h')
            ad_hourly = ad_hourly.sort_values('datetime')
    else:
        ad_hourly = None
        logger.warning("Hourly data not found")

    logger.info(f"Loaded {len(ad_daily)} daily rows")
    if ad_hourly is not None:
        logger.info(f"Loaded {len(ad_hourly)} hourly rows")

    return ad_daily, ad_hourly


def generate_sliding_windows_daily(daily_data, window_days=30, step_days=7):
    """
    生成daily数据的滑动窗口

    Args:
        daily_data: 完整的daily数据
        window_days: 窗口大小（天数）
        step_days: 滑动步长（天数）

    Returns:
        窗口列表 [(window_start, window_end, window_data), ...]
    """
    windows = []

    if len(daily_data) < window_days:
        logger.warning(f"Insufficient daily data: {len(daily_data)} < {window_days}")
        return windows

    # 按日期排序
    daily_data = daily_data.sort_values('date')

    # 获取日期范围
    min_date = daily_data['date'].min()
    max_date = daily_data['date'].max()

    # 生成滑动窗口
    current_start = min_date
    window_num = 0

    while current_start + timedelta(days=window_days) <= max_date + timedelta(days=1):
        current_end = current_start + timedelta(days=window_days - 1)

        # 提取窗口数据
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

        # 滑动到下一个窗口
        current_start = current_start + timedelta(days=step_days)

    logger.info(f"Generated {len(windows)} daily sliding windows ({window_days}-day window, {step_days}-day step)")
    return windows


def generate_sliding_windows_hourly(hourly_data, window_hours=24, step_hours=6):
    """
    生成hourly数据的滑动窗口

    Args:
        hourly_data: 完整的hourly数据
        window_hours: 窗口大小（小时数）
        step_hours: 滑动步长（小时数）

    Returns:
        窗口列表 [(window_start, window_end, window_data), ...]
    """
    windows = []

    if hourly_data is None or len(hourly_data) < window_hours:
        logger.warning(f"Insufficient hourly data")
        return windows

    # 按datetime排序
    hourly_data = hourly_data.sort_values('datetime')

    # 获取时间范围
    min_datetime = hourly_data['datetime'].min()
    max_datetime = hourly_data['datetime'].max()

    # 生成滑动窗口
    current_start = min_datetime
    window_num = 0

    while current_start + timedelta(hours=window_hours) <= max_datetime + timedelta(hours=1):
        current_end = current_start + timedelta(hours=window_hours - 1)

        # 提取窗口数据
        window_data = hourly_data[
            (hourly_data['datetime'] >= current_start) &
            (hourly_data['datetime'] <= current_end)
        ].copy()

        if len(window_data) > 0:
            windows.append({
                'window_num': window_num,
                'start_time': current_start,
                'end_time': current_end,
                'data': window_data
            })

            window_num += 1

        # 滑动到下一个窗口
        current_start = current_start + timedelta(hours=step_hours)

    logger.info(f"Generated {len(windows)} hourly sliding windows ({window_hours}-hour window, {step_hours}-hour step)")
    return windows


def evaluate_detector_on_windows(detector, windows, detector_name, label_method="rule_based"):
    """
    在多个滑动窗口上评估detector

    Args:
        detector: 检测器实例
        windows: 滑动窗口列表
        detector_name: 检测器名称
        label_method: 标注方法

    Returns:
        评估结果列表
    """
    evaluator = DiagnoserEvaluator()
    results = []

    logger.info(f"\nEvaluating {detector_name} on {len(windows)} sliding windows...")

    for i, window in enumerate(windows):
        window_data = window['data']
        window_num = window['window_num']
        window_start = window.get('start_date', window.get('start_time'))
        window_end = window.get('end_date', window.get('end_time'))

        try:
            # 评估
            result = evaluator.evaluate(
                detector=detector,
                test_data=window_data,
                detector_name=f"{detector_name}_window{i}",
                auto_label=True,
                label_method=label_method,
            )

            # 添加窗口信息
            result.details['window_num'] = window_num
            result.details['window_start'] = str(window_start)
            result.details['window_end'] = str(window_end)

            results.append(result)

            if (i + 1) % 5 == 0:
                logger.info(f"  Processed {i + 1}/{len(windows)} windows...")

        except Exception as e:
            logger.error(f"Error evaluating window {window_num}: {e}")
            continue

    logger.info(f"Completed evaluation: {len(results)} successful windows")
    return results


def aggregate_window_results(results, detector_name):
    """
    聚合多个窗口的评估结果

    Args:
        results: 窗口评估结果列表
        detector_name: 检测器名称

    Returns:
        聚合后的统计信息
    """
    if not results:
        return None

    from collections import Counter

    # 聚合准确率指标
    total_tp = sum(r.accuracy.true_positives for r in results)
    total_fp = sum(r.accuracy.false_positives for r in results)
    total_fn = sum(r.accuracy.false_negatives for r in results)
    total_tn = sum(r.accuracy.true_negatives for r in results)

    # 计算总体precision/recall/F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 聚合分数
    scores = [r.overall_score for r in results]
    grades = Counter([r.grade for r in results])

    # 聚合时间指标
    avg_processing_time = np.mean([r.timeliness.processing_time_ms for r in results])
    avg_detection_delay = np.mean([r.timeliness.detection_delay_days for r in results])

    # 总标签数
    total_labels = sum(r.details.get('label_count', 0) for r in results)

    aggregation = {
        'detector_name': detector_name,
        'total_windows': len(results),
        'accuracy': {
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'total_true_negatives': total_tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        },
        'scores': {
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'grade_distribution': dict(grades),
        },
        'timeliness': {
            'avg_processing_time_ms': avg_processing_time,
            'avg_detection_delay_days': avg_detection_delay,
        },
        'labels': {
            'total_labels': total_labels,
            'avg_labels_per_window': total_labels / len(results) if results else 0,
        },
        'window_results': results,
    }

    return aggregation


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Iterative Evaluation with Sliding Windows")
    logger.info("=" * 80)

    # 1. 加载数据
    daily_data, hourly_data = load_moprobo_data()
    if daily_data is None:
        logger.error("Failed to load data")
        return 1

    # 2. 生成滑动窗口
    logger.info("\n" + "=" * 80)
    logger.info("Generating Sliding Windows")
    logger.info("=" * 80)

    daily_windows = generate_sliding_windows_daily(daily_data, window_days=30, step_days=7)
    hourly_windows = generate_sliding_windows_hourly(hourly_data, window_hours=24, step_hours=6)

    if not daily_windows and not hourly_windows:
        logger.error("No windows generated")
        return 1

    # 3. 评估各个检测器
    reporter = EvaluationReporter(customer="moprobo_iterative")
    all_aggregations = {}

    # FatigueDetector
    if daily_windows:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating FatigueDetector")
        logger.info("=" * 80)

        fatigue_detector = FatigueDetector(config={
            "thresholds": {
                "window_size_days": 30,
                "golden_min_freq": 1.0,
                "golden_max_freq": 2.5,
                "fatigue_freq_threshold": 3.0,
                "cpa_increase_threshold": 1.3,
                "consecutive_days": 3,
                "min_golden_days": 5,
            }
        })

        fatigue_results = evaluate_detector_on_windows(
            fatigue_detector,
            daily_windows,
            "FatigueDetector",
            label_method="combined"
        )

        if fatigue_results:
            fatigue_agg = aggregate_window_results(fatigue_results, "FatigueDetector")
            all_aggregations['fatigue'] = fatigue_agg

            # 打印聚合结果
            logger.info(f"\nFatigueDetector Aggregated Results:")
            logger.info(f"  Windows: {fatigue_agg['total_windows']}")
            logger.info(f"  Precision: {fatigue_agg['accuracy']['precision']:.2%}")
            logger.info(f"  Recall: {fatigue_agg['accuracy']['recall']:.2%}")
            logger.info(f"  F1-Score: {fatigue_agg['accuracy']['f1_score']:.2%}")
            logger.info(f"  Avg Score: {fatigue_agg['scores']['avg_score']:.1f}/100")
            logger.info(f"  Score Range: {fatigue_agg['scores']['min_score']:.1f} - {fatigue_agg['scores']['max_score']:.1f}")
            logger.info(f"  Total Labels: {fatigue_agg['labels']['total_labels']}")

    # LatencyDetector
    if daily_windows:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating LatencyDetector")
        logger.info("=" * 80)

        latency_detector = LatencyDetector(config={
            "thresholds": {
                "roas_threshold": 1.0,
                "rolling_window_days": 3,
                "min_daily_spend": 50,
                "min_drop_ratio": 0.2,
            }
        })

        latency_results = evaluate_detector_on_windows(
            latency_detector,
            daily_windows,
            "LatencyDetector",
            label_method="performance_drop"
        )

        if latency_results:
            latency_agg = aggregate_window_results(latency_results, "LatencyDetector")
            all_aggregations['latency'] = latency_agg

            logger.info(f"\nLatencyDetector Aggregated Results:")
            logger.info(f"  Windows: {latency_agg['total_windows']}")
            logger.info(f"  Precision: {latency_agg['accuracy']['precision']:.2%}")
            logger.info(f"  Recall: {latency_agg['accuracy']['recall']:.2%}")
            logger.info(f"  F1-Score: {latency_agg['accuracy']['f1_score']:.2%}")
            logger.info(f"  Avg Score: {latency_agg['scores']['avg_score']:.1f}/100")

    # DarkHoursDetector
    if hourly_windows:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating DarkHoursDetector")
        logger.info("=" * 80)

        dark_hours_detector = DarkHoursDetector(config={
            "thresholds": {
                "min_daily_spend": 50,
                "underperform_threshold": 0.7,
                "min_days_per_dow": 3,
            }
        })

        dark_hours_results = evaluate_detector_on_windows(
            dark_hours_detector,
            hourly_windows,
            "DarkHoursDetector",
            label_method="performance_drop"
        )

        if dark_hours_results:
            dark_hours_agg = aggregate_window_results(dark_hours_results, "DarkHoursDetector")
            all_aggregations['dark_hours'] = dark_hours_agg

            logger.info(f"\nDarkHoursDetector Aggregated Results:")
            logger.info(f"  Windows: {dark_hours_agg['total_windows']}")
            logger.info(f"  Precision: {dark_hours_agg['accuracy']['precision']:.2%}")
            logger.info(f"  Recall: {dark_hours_agg['accuracy']['recall']:.2%}")
            logger.info(f"  F1-Score: {dark_hours_agg['accuracy']['f1_score']:.2%}")
            logger.info(f"  Avg Score: {dark_hours_agg['scores']['avg_score']:.1f}/100")

    # 4. 生成总结报告
    logger.info("\n" + "=" * 80)
    logger.info("Generating Summary Report")
    logger.info("=" * 80)

    import json
    summary_json = json.dumps(all_aggregations, indent=2, default=str)
    summary_path = reporter.save_report(summary_json, "iterative_evaluation_summary.json")
    logger.info(f"Summary saved to: {summary_path}")

    # 生成markdown报告
    markdown_content = generate_iterative_summary_markdown(all_aggregations)
    markdown_path = reporter.save_summary_markdown(markdown_content, "ITERATIVE_SUMMARY.md")
    logger.info(f"Markdown summary saved to: {markdown_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Iterative Evaluation Complete!")
    logger.info("=" * 80)

    return 0


def generate_iterative_summary_markdown(aggregations):
    """生成迭代评估的markdown总结"""
    lines = []
    lines.append("# Iterative Evaluation Summary (Sliding Window)")
    lines.append("")
    lines.append("## Evaluation Methodology")
    lines.append("")
    lines.append("- **Daily Data**: 30-day sliding window, 7-day step")
    lines.append("- **Hourly Data**: 24-hour sliding window, 6-hour step")
    lines.append("- **Multiple Evaluations**: Each detector evaluated on multiple windows")
    lines.append("")
    lines.append("---")
    lines.append("")

    for detector_name, agg in aggregations.items():
        lines.append(f"## {detector_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"**Total Windows Evaluated**: {agg['total_windows']}")
        lines.append("")

        lines.append("### Aggregated Accuracy")
        lines.append("")
        lines.append(f"- **Precision**: {agg['accuracy']['precision']:.2%}")
        lines.append(f"- **Recall**: {agg['accuracy']['recall']:.2%}")
        lines.append(f"- **F1-Score**: {agg['accuracy']['f1_score']:.2%}")
        lines.append("")
        lines.append(f"- **Total True Positives**: {agg['accuracy']['total_true_positives']}")
        lines.append(f"- **Total False Positives**: {agg['accuracy']['total_false_positives']}")
        lines.append(f"- **Total False Negatives**: {agg['accuracy']['total_false_negatives']}")
        lines.append("")

        lines.append("### Score Distribution")
        lines.append("")
        lines.append(f"- **Average Score**: {agg['scores']['avg_score']:.1f}/100")
        lines.append(f"- **Min Score**: {agg['scores']['min_score']:.1f}/100")
        lines.append(f"- **Max Score**: {agg['scores']['max_score']:.1f}/100")
        lines.append(f"- **Std Dev**: {agg['scores']['std_score']:.2f}")
        lines.append("")
        lines.append("**Grade Distribution**:")
        for grade, count in agg['scores']['grade_distribution'].items():
            lines.append(f"- {grade}: {count} windows")
        lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    exit(main())
