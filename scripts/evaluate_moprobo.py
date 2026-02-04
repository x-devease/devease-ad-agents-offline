#!/usr/bin/env python3
"""
Test Diagnoser Judge with real Moprobo data.

Simulates production scenario:
- 30 days of daily data
- 24 hours of hourly data
- Real-time diagnosis evaluation
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


def simulate_production_scenario(daily_data, hourly_data=None):
    """
    模拟线上实时场景

    线上数据：
    - 30天的daily数据
    - 24小时的hourly数据
    """
    logger.info("=" * 60)
    logger.info("Simulating Production Scenario")
    logger.info("=" * 60)

    # 获取最新日期
    if 'date' in daily_data.columns:
        latest_date = daily_data['date'].max()
    else:
        latest_date = daily_data['date_start'].max()

    # 获取最近30天的daily数据
    thirty_days_ago = latest_date - timedelta(days=30)
    daily_sample = daily_data[daily_data['date'] >= thirty_days_ago].copy()

    logger.info(f"Date range: {thirty_days_ago.date()} to {latest_date.date()} ({len(daily_sample)} days)")

    # 获取最新24小时的hourly数据
    hourly_sample = None
    if hourly_data is not None and 'datetime' in hourly_data.columns:
        latest_datetime = hourly_data['datetime'].max()
        one_day_ago = latest_datetime - timedelta(days=1)
        hourly_sample = hourly_data[hourly_data['datetime'] >= one_day_ago].copy()
        logger.info(f"Hourly range: {one_day_ago} to {latest_datetime} ({len(hourly_sample)} hours)")

    logger.info(f"Simulating real-time diagnosis with:")
    logger.info(f"  - {len(daily_sample)} daily records")
    if hourly_sample is not None:
        logger.info(f"  - {len(hourly_sample)} hourly records")

    return daily_sample, hourly_sample


def test_zero_cost_labeling(daily_data):
    """测试零成本标注生成"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Zero-Cost Label Generation")
    logger.info("=" * 60)

    generator = ZeroCostLabelGenerator()

    # 测试不同的标注方法
    methods = ["performance_drop", "rule_based", "statistical_anomaly", "combined"]
    all_labels = {}

    for method in methods:
        logger.info(f"\nGenerating labels using method: {method}")
        labels = generator.generate(daily_data, method=method)
        all_labels[method] = labels
        logger.info(f"  Generated {len(labels)} labels")

        # 显示前3个标注
        if labels:
            logger.info("  Sample labels:")
            for i, label in enumerate(labels[:3], 1):
                entity = label['affected_entities'][0] if label['affected_entities'] else 'unknown'
                logger.info(f"    {i}. {entity} - {label['issue_type']} - {label['date']}")
        else:
            logger.info("  No labels generated")

    return all_labels


def evaluate_detectors(daily_data, hourly_data, customer_name="moprobo"):
    """评估所有检测器"""
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating Detectors with Zero-Cost Labels")
    logger.info("=" * 60)

    generator = ZeroCostLabelGenerator()
    evaluator = DiagnoserEvaluator()
    reporter = EvaluationReporter(customer=customer_name)

    results = {}

    # 生成所有类型的labels
    logger.info("\nGenerating zero-cost labels for all issue types...")
    all_labels = generator.generate(daily_data, method="rule_based")

    # 按issue_type分组
    labels_by_type = {}
    for label in all_labels:
        issue_type = label["issue_type"]
        if issue_type not in labels_by_type:
            labels_by_type[issue_type] = []
        labels_by_type[issue_type].append(label)

    logger.info(f"Label distribution:")
    for issue_type, labels in labels_by_type.items():
        logger.info(f"  {issue_type}: {len(labels)} labels")

    # 1. FatigueDetector
    logger.info("\n" + "-" * 60)
    logger.info("1. Evaluating FatigueDetector")
    logger.info("-" * 60)

    try:
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

        # 只使用fatigue类型的labels
        fatigue_labels = labels_by_type.get("fatigue", [])
        logger.info(f"Using {len(fatigue_labels)} fatigue labels for evaluation")

        result = evaluator.evaluate(
            detector=fatigue_detector,
            test_data=daily_data,
            detector_name="FatigueDetector",
            ground_truth=fatigue_labels if fatigue_labels else None,
            auto_label=(len(fatigue_labels) == 0),
        )

        results["fatigue"] = result

        # 保存报告
        report_json = reporter.generate_evaluation_report(result)
        report_path = reporter.save_report(report_json, "fatigue_detector_eval.json")
        logger.info(f"Report saved to: {report_path}")

        # 打印摘要
        logger.info(f"\nFatigueDetector Results:")
        logger.info(f"  Overall Score: {result.overall_score:.1f}/100 ({result.grade})")
        logger.info(f"  Precision: {result.accuracy.precision:.2%}")
        logger.info(f"  Recall: {result.accuracy.recall:.2%}")
        logger.info(f"  F1-Score: {result.accuracy.f1_score:.2%}")
        logger.info(f"  Labels used: {len(fatigue_labels)}")

    except Exception as e:
        logger.error(f"Error evaluating FatigueDetector: {e}")
        import traceback
        traceback.print_exc()

    # 2. LatencyDetector
    logger.info("\n" + "-" * 60)
    logger.info("2. Evaluating LatencyDetector")
    logger.info("-" * 60)

    try:
        latency_detector = LatencyDetector(config={
            "thresholds": {
                "roas_threshold": 1.0,
                "rolling_window_days": 3,
                "min_daily_spend": 50,
                "min_drop_ratio": 0.2,
            }
        })

        # 只使用latency类型的labels
        latency_labels = labels_by_type.get("latency", [])
        logger.info(f"Using {len(latency_labels)} latency labels for evaluation")

        result = evaluator.evaluate(
            detector=latency_detector,
            test_data=daily_data,
            detector_name="LatencyDetector",
            ground_truth=latency_labels if latency_labels else None,
            auto_label=(len(latency_labels) == 0),
        )

        results["latency"] = result

        report_json = reporter.generate_evaluation_report(result)
        report_path = reporter.save_report(report_json, "latency_detector_eval.json")
        logger.info(f"Report saved to: {report_path}")

        logger.info(f"\nLatencyDetector Results:")
        logger.info(f"  Overall Score: {result.overall_score:.1f}/100 ({result.grade})")
        logger.info(f"  F1-Score: {result.accuracy.f1_score:.2%}")
        logger.info(f"  Labels used: {len(latency_labels)}")

    except Exception as e:
        logger.error(f"Error evaluating LatencyDetector: {e}")
        import traceback
        traceback.print_exc()

    # 3. DarkHoursDetector (需要hourly数据)
    if hourly_data is not None and len(hourly_data) > 0:
        logger.info("\n" + "-" * 60)
        logger.info("3. Evaluating DarkHoursDetector")
        logger.info("-" * 60)

        try:
            dark_hours_detector = DarkHoursDetector(config={
                "thresholds": {
                    "min_daily_spend": 50,
                    "underperform_threshold": 0.7,
                    "min_days_per_dow": 3,
                }
            })

            # 只使用dark_hours类型的labels
            dark_hours_labels = labels_by_type.get("dark_hours", [])
            logger.info(f"Using {len(dark_hours_labels)} dark_hours labels for evaluation")

            result = evaluator.evaluate(
                detector=dark_hours_detector,
                test_data=daily_data,
                detector_name="DarkHoursDetector",
                ground_truth=dark_hours_labels if dark_hours_labels else None,
                auto_label=(len(dark_hours_labels) == 0),
            )

            results["dark_hours"] = result

            report_json = reporter.generate_evaluation_report(result)
            report_path = reporter.save_report(report_json, "dark_hours_detector_eval.json")
            logger.info(f"Report saved to: {report_path}")

            logger.info(f"\nDarkHoursDetector Results:")
            logger.info(f"  Overall Score: {result.overall_score:.1f}/100 ({result.grade})")
            logger.info(f"  F1-Score: {result.accuracy.f1_score:.2%}")
            logger.info(f"  Labels used: {len(dark_hours_labels)}")

        except Exception as e:
            logger.error(f"Error evaluating DarkHoursDetector: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("Skipping DarkHoursDetector: no hourly data available")

    return results


def generate_summary_report(results, customer_name="moprobo"):
    """生成总体报告"""
    logger.info("\n" + "=" * 60)
    logger.info("Generating Summary Report")
    logger.info("=" * 60)

    reporter = EvaluationReporter(customer=customer_name)

    summary = {
        "customer": customer_name,
        "evaluation_date": datetime.now().isoformat(),
        "scenario": "production_realtime",
        "data_description": "30 days daily + 24 hours hourly",
        "detectors": {}
    }

    for detector_name, result in results.items():
        summary["detectors"][detector_name] = {
            "overall_score": result.overall_score,
            "grade": result.grade,
            "accuracy": {
                "precision": result.accuracy.precision,
                "recall": result.accuracy.recall,
                "f1_score": result.accuracy.f1_score,
                "true_positives": result.accuracy.true_positives,
                "false_positives": result.accuracy.false_positives,
                "false_negatives": result.accuracy.false_negatives,
            },
            "reliability": {
                "stability_score": result.reliability.stability_score,
                "reproducibility_score": result.reliability.reproducibility_score,
                "robustness_score": result.reliability.robustness_score,
            },
            "timeliness": {
                "detection_delay_days": result.timeliness.detection_delay_days,
                "processing_time_ms": result.timeliness.processing_time_ms,
                "min_data_days": result.timeliness.min_data_days,
            },
            "label_method": result.details.get("label_method", "N/A"),
            "label_count": result.details.get("label_count", 0),
        }

    # 保存JSON summary
    import json
    summary_json = json.dumps(summary, indent=2, default=str)
    summary_path = reporter.save_report(summary_json, "evaluation_summary.json")
    logger.info(f"Summary JSON saved to: {summary_path}")

    # 生成Markdown summary
    markdown_content = reporter.generate_summary_markdown(
        results=results,
        customer_name=customer_name,
        scenario="Production Real-Time (30d daily + 24h hourly)",
        data_description="30 days daily + 24 hours hourly",
    )
    markdown_path = reporter.save_summary_markdown(markdown_content, "SUMMARY.md")
    logger.info(f"Summary Markdown saved to: {markdown_path}")

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Customer: {customer_name}")
    logger.info(f"Scenario: Production Real-Time (30d daily + 24h hourly)")
    logger.info(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("\nDetector Performance:")
    for detector_name, result in results.items():
        logger.info(f"\n{detector_name.upper()}:")
        logger.info(f"  Score: {result.overall_score:.1f}/100 ({result.grade})")
        logger.info(f"  F1-Score: {result.accuracy.f1_score:.2%}")
        logger.info(f"  Precision: {result.accuracy.precision:.2%}")
        logger.info(f"  Recall: {result.accuracy.recall:.2%}")

    logger.info("\n" + "=" * 60)
    logger.info(f"All reports saved to: src/meta/diagnoser/judge/reports/{customer_name}/")
    logger.info("=" * 60)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Moprobo Evaluation with Zero-Cost Labels")
    logger.info("=" * 60)

    # 1. 加载数据
    daily_data, hourly_data = load_moprobo_data()
    if daily_data is None:
        logger.error("Failed to load data")
        return 1

    # 2. 模拟生产场景
    daily_sample, hourly_sample = simulate_production_scenario(daily_data, hourly_data)

    # 3. 测试零成本标注
    all_labels = test_zero_cost_labeling(daily_sample)

    # 4. 评估检测器
    results = evaluate_detectors(daily_sample, hourly_sample, customer_name="moprobo")

    # 5. 生成总结报告
    generate_summary_report(results, customer_name="moprobo")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation Complete!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
