#!/usr/bin/env python3
"""
Quick evaluation with optimized thresholds.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.meta.diagnoser.detectors import FatigueDetector, DarkHoursDetector
from src.meta.diagnoser.scripts.evaluate_fatigue import load_and_preprocess_data, generate_sliding_windows, evaluate_detector, aggregate_results
from src.meta.diagnoser.judge.evaluator import Evaluator
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("\n🔧 Running FatigueDetector with OPTIMIZED thresholds...")

    # Load optimized config
    config_path = Path("src/meta/diagnoser/detectors/config/fatigue_detector_config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"   Loaded config: {config_path}")
    print(f"   Thresholds: {config['thresholds']}")

    # Create detector with optimized config
    detector = FatigueDetector(config=config)

    # Load data
    data = load_and_preprocess_data()

    # Generate windows
    windows = generate_sliding_windows(data, window_size=30, step=7)
    print(f"   Generated {len(windows)} windows")

    # Evaluate
    results = evaluate_detector(detector, windows, "FatigueDetector", "rule_based")

    # Aggregate
    agg = aggregate_results(results, "FatigueDetector")

    # Print results
    acc = agg['accuracy']
    print(f"\n   RESULTS:")
    print(f"   Precision: {acc['precision']:.4f}")
    print(f"   Recall: {acc['recall']:.4f}")
    print(f"   F1-Score: {acc['f1_score']:.4f}")
    print(f"   TP: {acc['total_tp']}, FP: {acc['total_fp']}, FN: {acc['total_fn']}")

    # Check satisfaction
    precision_satisfied = acc['precision'] >= 0.70
    recall_satisfied = acc['recall'] >= 0.80
    f1_satisfied = acc['f1_score'] >= 0.75

    if recall_satisfied and f1_satisfied:
        print(f"\n   ✅ SATISFIED!")
        return 0
    else:
        print(f"\n   ❌ NOT SATISFIED")
        if not recall_satisfied:
            print(f"      Recall {acc['recall']:.4f} < 0.80")
        if not f1_satisfied:
            print(f"      F1 {acc['f1_score']:.4f} < 0.75")
        return 1

if __name__ == "__main__":
    sys.exit(main())
