#!/usr/bin/env python3
"""
Test script for Diagnoser Judge evaluation system.

Tests the Judge system with the three detectors:
1. FatigueDetector - Creative fatigue detection
2. LatencyDetector - Human latency detection
3. DarkHoursDetector - Dark hours analysis
"""

import sys
import logging
from pathlib import Path

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
    BacktestEngine,
    EvaluationReporter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_moprobo_data():
    """Load and preprocess moprobo test data."""
    logger.info("Loading moprobo data...")

    # Load ad_daily data
    ad_daily_path = Path("datasets/moprobo/meta/raw/ad_daily_insights_2024-12-17_2025-12-17.csv")
    if not ad_daily_path.exists():
        logger.error(f"Data file not found: {ad_daily_path}")
        return None

    ad_daily = pd.read_csv(ad_daily_path)

    # Preprocess numeric columns
    numeric_cols = ['spend', 'impressions', 'reach', 'purchase_roas', 'clicks']
    for col in numeric_cols:
        if col in ad_daily.columns:
            ad_daily[col] = pd.to_numeric(ad_daily[col], errors='coerce').fillna(0)

    # Ensure date column
    if 'date' in ad_daily.columns:
        ad_daily['date'] = pd.to_datetime(ad_daily['date'], errors='coerce')
        ad_daily = ad_daily.sort_values('date')

    logger.info(f"Loaded {len(ad_daily)} rows from {ad_daily_path}")

    return ad_daily


def prepare_ground_truth(data):
    """
    Prepare ground truth for evaluation.

    For testing purposes, we'll create synthetic ground truth
    based on obvious issues in the data.
    """
    ground_truth = []

    # Find entities with clear issues
    if 'ad_id' in data.columns:
        # Group by ad_id
        for ad_id, ad_data in data.groupby('ad_id'):
            ad_data = ad_data.sort_values('date')

            if len(ad_data) < 30:
                continue

            # Check for fatigue: high frequency at the end
            if 'impressions' in ad_data.columns and 'reach' in ad_data.columns:
                ad_data = ad_data.copy()
                ad_data['cum_impressions'] = ad_data['impressions'].cumsum()
                ad_data['cum_reach'] = ad_data['reach'].cumsum()

                # Avoid division by zero
                ad_data['cum_reach'] = ad_data['cum_reach'].replace(0, 1)

                # Calculate cumulative frequency
                ad_data['cum_frequency'] = ad_data['cum_impressions'] / ad_data['cum_reach']

                # Check if last 7 days have high frequency
                last_7_days = ad_data.tail(7)
                avg_freq_last_7 = last_7_days['cum_frequency'].mean()

                if avg_freq_last_7 > 3.0:
                    ground_truth.append({
                        "affected_entities": [ad_id],
                        "has_issue": True,
                        "issue_type": "fatigue",
                        "severity": "high",
                    })

    logger.info(f"Created {len(ground_truth)} ground truth annotations")
    return ground_truth


def test_fatigue_detector_with_judge(data, evaluator, reporter):
    """Test FatigueDetector with Judge system."""
    logger.info("=" * 80)
    logger.info("Testing FatigueDetector with Judge")
    logger.info("=" * 80)

    # Prepare test data for a single ad
    if 'ad_id' not in data.columns:
        logger.warning("No ad_id column, skipping FatigueDetector test")
        return None

    # Get the first ad with enough data
    ad_id_counts = data['ad_id'].value_counts()
    valid_ads = ad_id_counts[ad_id_counts >= 33].index

    if len(valid_ads) == 0:
        logger.warning("No ads with enough data (>= 33 days)")
        return None

    test_ad_id = valid_ads[0]
    ad_data = data[data['ad_id'] == test_ad_id].copy()

    logger.info(f"Testing ad_id: {test_ad_id}, {len(ad_data)} days")

    # Create detector
    detector = FatigueDetector(config={
        "thresholds": {
            "window_size_days": 30,  # 30-day rolling window
            "golden_min_freq": 1.0,
            "golden_max_freq": 2.5,
            "fatigue_freq_threshold": 3.0,
            "cpa_increase_threshold": 1.3,
            "consecutive_days": 3,
            "min_golden_days": 5,
        }
    })

    # Run evaluation
    result = evaluator.evaluate(
        detector=detector,
        test_data=ad_data,
        detector_name=f"FatigueDetector_{test_ad_id}",
    )

    # Generate report
    report_json = reporter.generate_evaluation_report(result)
    report_path = reporter.save_report(report_json, "fatigue_detector_eval.json")
    logger.info(f"Report saved to: {report_path}")

    # Print summary
    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  Overall Score: {result.overall_score:.1f}/100")
    logger.info(f"  Grade: {result.grade}")
    logger.info(f"  F1-Score: {result.accuracy.f1_score:.2%}")
    logger.info(f"  Processing Time: {result.timeliness.processing_time_ms:.1f}ms")

    return result


def test_latency_detector_with_judge(data, evaluator, reporter):
    """Test LatencyDetector with Judge system."""
    logger.info("=" * 80)
    logger.info("Testing LatencyDetector with Judge")
    logger.info("=" * 80)

    # Prepare test data
    if 'adset_id' not in data.columns:
        logger.warning("No adset_id column, skipping LatencyDetector test")
        return None

    # Get the first adset with enough data
    adset_id_counts = data['adset_id'].value_counts()
    valid_adsets = adset_id_counts[adset_id_counts >= 5].index

    if len(valid_adsets) == 0:
        logger.warning("No adsets with enough data (>= 5 days)")
        return None

    test_adset_id = valid_adsets[0]
    adset_data = data[data['adset_id'] == test_adset_id].copy()

    logger.info(f"Testing adset_id: {test_adset_id}, {len(adset_data)} days")

    # Create detector
    detector = LatencyDetector(config={
        "thresholds": {
            "roas_threshold": 1.0,
            "rolling_window_days": 3,
            "min_daily_spend": 50,
            "min_drop_ratio": 0.2,
        }
    })

    # Run evaluation
    result = evaluator.evaluate(
        detector=detector,
        test_data=adset_data,
        detector_name=f"LatencyDetector_{test_adset_id}",
    )

    # Generate report
    report_json = reporter.generate_evaluation_report(result)
    report_path = reporter.save_report(report_json, "latency_detector_eval.json")
    logger.info(f"Report saved to: {report_path}")

    # Print summary
    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  Overall Score: {result.overall_score:.1f}/100")
    logger.info(f"  Grade: {result.grade}")
    logger.info(f"  F1-Score: {result.accuracy.f1_score:.2%}")
    logger.info(f"  Processing Time: {result.timeliness.processing_time_ms:.1f}ms")

    return result


def test_dark_hours_detector_with_judge(data, evaluator, reporter):
    """Test DarkHoursDetector with Judge system."""
    logger.info("=" * 80)
    logger.info("Testing DarkHoursDetector with Judge")
    logger.info("=" * 80)

    # Prepare test data
    if 'ad_id' not in data.columns:
        logger.warning("No ad_id column, skipping DarkHoursDetector test")
        return None

    # Get the first ad with enough data
    ad_id_counts = data['ad_id'].value_counts()
    valid_ads = ad_id_counts[ad_id_counts >= 30].index

    if len(valid_ads) == 0:
        logger.warning("No ads with enough data (>= 30 days)")
        return None

    test_ad_id = valid_ads[0]
    ad_data = data[data['ad_id'] == test_ad_id].copy()

    logger.info(f"Testing ad_id: {test_ad_id}, {len(ad_data)} days")

    # Create detector
    detector = DarkHoursDetector(config={
        "thresholds": {
            "min_daily_spend": 50,
            "underperform_threshold": 0.7,  # 70% below average
            "min_days_per_dow": 3,
        }
    })

    # Run evaluation (no hourly data, daily only)
    result = evaluator.evaluate(
        detector=detector,
        test_data=ad_data,
        detector_name=f"DarkHoursDetector_{test_ad_id}",
    )

    # Generate report
    report_json = reporter.generate_evaluation_report(result)
    report_path = reporter.save_report(report_json, "dark_hours_detector_eval.json")
    logger.info(f"Report saved to: {report_path}")

    # Print summary
    logger.info(f"\nEvaluation Summary:")
    logger.info(f"  Overall Score: {result.overall_score:.1f}/100")
    logger.info(f"  Grade: {result.grade}")
    logger.info(f"  F1-Score: {result.accuracy.f1_score:.2%}")
    logger.info(f"  Processing Time: {result.timeliness.processing_time_ms:.1f}ms")

    return result


def test_backtest_engine(data):
    """Test the backtest engine with 30-day rolling window."""
    logger.info("=" * 80)
    logger.info("Testing Backtest Engine (30-day rolling window)")
    logger.info("=" * 80)

    # Get a single ad with enough data
    if 'ad_id' not in data.columns:
        logger.warning("No ad_id column, skipping backtest test")
        return

    ad_id_counts = data['ad_id'].value_counts()
    valid_ads = ad_id_counts[ad_id_counts >= 60].index  # Need at least 60 days for backtest

    if len(valid_ads) == 0:
        logger.warning("No ads with enough data for backtest (>= 60 days)")
        return

    test_ad_id = valid_ads[0]
    ad_data = data[data['ad_id'] == test_ad_id].copy().sort_values('date_start')

    logger.info(f"Backtesting ad_id: {test_ad_id}, {len(ad_data)} days")

    # Create detector
    detector = FatigueDetector(config={
        "thresholds": {
            "window_size_days": 30,  # 30-day rolling window
            "golden_min_freq": 1.0,
            "golden_max_freq": 2.5,
            "fatigue_freq_threshold": 3.0,
            "cpa_increase_threshold": 1.3,
            "consecutive_days": 3,
            "min_golden_days": 5,
        }
    })

    # Create backtest engine with 30-day window
    backtest_engine = BacktestEngine(
        prediction_interval_days=7,
        min_history_days=30,
        max_history_days=30,  # 30-day rolling window
    )

    # Run backtest
    logger.info("Running backtest...")
    backtest_result = backtest_engine.run_backtest(
        detector=detector,
        historical_data=ad_data,
    )

    # Generate backtest report
    reporter = EvaluationReporter(output_dir="src/meta/diagnoser/judge/reports")
    report_json = reporter.generate_backtest_report(backtest_result)
    report_path = reporter.save_report(report_json, "backtest_results.json")
    logger.info(f"Backtest report saved to: {report_path}")

    # Print summary
    logger.info(f"\nBacktest Summary:")
    logger.info(f"  Total Days: {backtest_result.total_days}")
    logger.info(f"  Prediction Points: {len(backtest_result.prediction_points)}")
    logger.info(f"  Predictions Made: {len(backtest_result.predictions)}")
    logger.info(f"  F1-Score: {backtest_result.accuracy_metrics.f1_score:.2%}")
    logger.info(f"  Precision: {backtest_result.accuracy_metrics.precision:.2%}")
    logger.info(f"  Recall: {backtest_result.accuracy_metrics.recall:.2%}")

    return backtest_result


def main():
    """Main test function."""
    logger.info("Starting Diagnoser Judge System Tests")
    logger.info("=" * 80)

    # Load data
    data = load_moprobo_data()
    if data is None:
        logger.error("Failed to load data, exiting")
        return

    # Create judge components
    evaluator = DiagnoserEvaluator()
    reporter = EvaluationReporter(output_dir="src/meta/diagnoser/judge/reports")

    # Test each detector with Judge
    fatigue_result = test_fatigue_detector_with_judge(data, evaluator, reporter)
    latency_result = test_latency_detector_with_judge(data, evaluator, reporter)
    dark_hours_result = test_dark_hours_detector_with_judge(data, evaluator, reporter)

    # Test backtest engine
    backtest_result = test_backtest_engine(data)

    # Overall summary
    logger.info("=" * 80)
    logger.info("Overall Test Summary")
    logger.info("=" * 80)
    logger.info(f"FatigueDetector: Score={fatigue_result.overall_score:.1f}/100, Grade={fatigue_result.grade}" if fatigue_result else "FatigueDetector: SKIPPED")
    logger.info(f"LatencyDetector: Score={latency_result.overall_score:.1f}/100, Grade={latency_result.grade}" if latency_result else "LatencyDetector: SKIPPED")
    logger.info(f"DarkHoursDetector: Score={dark_hours_result.overall_score:.1f}/100, Grade={dark_hours_result.grade}" if dark_hours_result else "DarkHoursDetector: SKIPPED")

    if backtest_result:
        logger.info(f"Backtest: {len(backtest_result.predictions)} predictions, F1={backtest_result.accuracy_metrics.f1_score:.2%}")

    logger.info("=" * 80)
    logger.info("All tests completed!")
    logger.info("Reports saved to: src/meta/diagnoser/judge/reports/")


if __name__ == "__main__":
    main()
