#!/usr/bin/env python3
"""
Universal detector evaluation script.

Evaluates any detector using sliding window backtesting approach.
Supports all detectors (FatigueDetector, LatencyDetector, DarkHoursDetector)
with configurable window parameters.

Usage:
    # Evaluate FatigueDetector with defaults
    python eval/eval_detector.py --detector FatigueDetector

    # Evaluate LatencyDetector with custom windows
    python eval/eval_detector.py --detector LatencyDetector --window-size 60 --step-size 14 --max-windows 5

    # Evaluate DarkHoursDetector with different customer
    python eval/eval_detector.py --detector DarkHoursDetector --customer other_customer

Examples:
    # Quick test (3 windows)
    python eval/eval_detector.py --detector FatigueDetector --max-windows 3

    # Comprehensive evaluation (20 windows)
    python eval/eval_detector.py --detector FatigueDetector --max-windows 20

    # Production run with defaults
    python eval/eval_detector.py --detector FatigueDetector --window-size 30 --step-size 7 --max-windows 10
"""

import sys
import logging
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.meta.diagnoser.detectors import (
    FatigueDetector,
    LatencyDetector,
    DarkHoursDetector,
)
from src.meta.diagnoser.evaluator import EvaluationReporter
from src.meta.diagnoser.scripts.utils import (
    data_loader,
    sliding_windows,
    evaluation_runner,
    results_aggregator,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Detector registry
DETECTOR_REGISTRY = {
    "FatigueDetector": FatigueDetector,
    "LatencyDetector": LatencyDetector,
    "DarkHoursDetector": DarkHoursDetector,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate detector using sliding window backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        choices=list(DETECTOR_REGISTRY.keys()),
        help="Detector class name to evaluate"
    )

    parser.add_argument(
        "--customer",
        type=str,
        default="moprobo",
        help="Customer name (default: moprobo)"
    )

    parser.add_argument(
        "--platform",
        type=str,
        default="meta",
        help="Platform name (default: meta)"
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Window size in days (default: 30)"
    )

    parser.add_argument(
        "--step-size",
        type=int,
        default=7,
        help="Step size between windows in days (default: 7)"
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=10,
        help="Maximum number of windows to generate (default: 10)"
    )

    parser.add_argument(
        "--label-method",
        type=str,
        default="rule_based",
        choices=["rule_based", "heuristic"],
        help="Method for auto-generating labels (default: rule_based)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output report name (default: auto-generated from detector name)"
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    detector_name = args.detector
    detector_class = DETECTOR_REGISTRY[detector_name]

    logger.info("=" * 80)
    logger.info(f"{detector_name} Sliding Window Evaluation")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Detector: {detector_name}")
    logger.info(f"  Customer: {args.customer}")
    logger.info(f"  Platform: {args.platform}")
    logger.info(f"  Window size: {args.window_size} days")
    logger.info(f"  Step size: {args.step_size} days")
    logger.info(f"  Max windows: {args.max_windows}")
    logger.info(f"  Label method: {args.label_method}")
    logger.info("")

    # 1. Load data
    logger.info("Step 1: Loading data...")
    daily_data = data_loader.load_moprobo_data(args.customer, args.platform)
    logger.info(f"  Loaded {len(daily_data)} rows")

    # 2. Generate sliding windows
    logger.info("\nStep 2: Generating sliding windows...")
    windows = sliding_windows.generate_sliding_windows_daily(
        daily_data,
        window_size_days=args.window_size,
        step_days=args.step_size,
        max_windows=args.max_windows
    )

    if not windows:
        logger.error("No windows generated - exiting")
        return 1

    logger.info(f"  Generated {len(windows)} windows")

    # 3. Initialize detector
    logger.info("\nStep 3: Initializing detector...")
    detector = detector_class()
    logger.info(f"  Created {detector_name} instance")

    # 4. Evaluate on windows
    logger.info("\nStep 4: Evaluating detector on windows...")
    results = evaluation_runner.evaluate_detector_on_windows(
        detector=detector,
        windows=windows,
        detector_name=detector_name,
        label_method=args.label_method
    )

    if not results:
        logger.error("No evaluation results - exiting")
        return 1

    logger.info(f"  Completed {len(results)} window evaluations")

    # 5. Aggregate results
    logger.info("\nStep 5: Aggregating results...")
    aggregation = results_aggregator.aggregate_results(results, detector_name)

    # 6. Display results
    logger.info("\n" + "=" * 80)
    logger.info(f"{detector_name.upper()} - AGGREGATED RESULTS")
    logger.info("=" * 80)
    logger.info(f"Windows: {aggregation['total_windows']}")
    logger.info(f"Precision: {aggregation['accuracy']['precision']:.2%}")
    logger.info(f"Recall: {aggregation['accuracy']['recall']:.2%}")
    logger.info(f"F1-Score: {aggregation['accuracy']['f1_score']:.2%}")
    logger.info(f"Avg Score: {aggregation['scores']['avg']:.1f}/100")
    logger.info(f"Score Range: {aggregation['scores']['min']:.1f} - {aggregation['scores']['max']:.1f}")
    logger.info(f"Total TP: {aggregation['accuracy']['total_tp']}, "
                f"FP: {aggregation['accuracy']['total_fp']}, "
                f"FN: {aggregation['accuracy']['total_fn']}")

    # 7. Save report
    logger.info("\nStep 6: Saving report...")

    # Generate output filename
    if args.output_name:
        output_filename = args.output_name
    else:
        detector_short = detector_name.replace("Detector", "").lower()
        output_filename = f"{detector_short}_sliding_{args.max_windows}windows.json"

    reporter = EvaluationReporter(customer=f"{args.customer}_sliding")
    summary_json = json.dumps(aggregation, indent=2, default=str)
    reporter.save_report(summary_json, output_filename)

    report_path = Path(f"src/meta/diagnoser/evaluator/reports/{args.customer}_sliding/{output_filename}")
    logger.info(f"  Report saved to: {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Evaluation Complete!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
