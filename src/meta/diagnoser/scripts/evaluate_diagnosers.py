#!/usr/bin/env python3
"""
Evaluate all diagnoser agents and check satisfaction.

This script loads and evaluates all detectors, checking if they meet
their target metrics.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def load_metrics(detector_name: str) -> dict:
    """Load metrics for a detector."""
    detector_map = {
        "FatigueDetector": "fatigue_sliding_10windows.json",
        "LatencyDetector": "latency_sliding_10windows.json",
        "DarkHoursDetector": "dark_hours_sliding_10windows.json"
    }

    report_file = detector_map.get(detector_name)
    if not report_file:
        return None

    report_path = Path("src/meta/diagnoser/judge/reports/moprobo_sliding") / report_file

    if not report_path.exists():
        return None

    with open(report_path, 'r') as f:
        report = json.load(f)

    # Extract metrics based on report format
    if "aggregated_metrics" in report:
        # New format (DarkHoursDetector)
        metrics = report["aggregated_metrics"]
        return {
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1_score": metrics.get("f1_score", 0),
            "tp": metrics.get("total_tp", 0),
            "fp": metrics.get("total_fp", 0),
            "fn": metrics.get("total_fn", 0)
        }
    else:
        # Old format (LatencyDetector, FatigueDetector)
        accuracy = report.get("accuracy", {})
        return {
            "precision": accuracy.get("precision", 0),
            "recall": accuracy.get("recall", 0),
            "f1_score": accuracy.get("f1_score", 0),
            "tp": accuracy.get("total_tp", 0),
            "fp": accuracy.get("total_fp", 0),
            "fn": accuracy.get("total_fn", 0)
        }


def check_satisfaction(metrics: dict, targets: dict) -> tuple[bool, dict]:
    """Check if metrics meet target thresholds."""
    satisfaction = {}
    all_satisfied = True

    for metric, target_value in targets.items():
        current_value = metrics.get(metric, 0)
        is_satisfied = current_value >= target_value
        satisfaction[metric] = {
            "current": current_value,
            "target": target_value,
            "satisfied": is_satisfied
        }
        if not is_satisfied:
            all_satisfied = False

    return all_satisfied, satisfaction


def main():
    """Main evaluation function."""
    print("\n" + "="*80)
    print("DIAGNOSER AGENT EVALUATION")
    print("="*80 + "\n")

    # Define detectors and their target metrics
    detectors = {
        "FatigueDetector": {
            "f1_score": 0.75,
            "precision": 0.70,
            "recall": 0.80
        },
        "LatencyDetector": {
            "f1_score": 0.75,
            "precision": 0.70,
            "recall": 0.80
        },
        "DarkHoursDetector": {
            "f1_score": 0.75,
            "precision": 0.70,
            "recall": 0.80
        }
    }

    print(f"📋 Evaluating {len(detectors)} detectors:\n")

    results = {}
    for detector, targets in detectors.items():
        print(f"🔍 {detector}")

        metrics = load_metrics(detector)

        if metrics is None:
            print(f"   ⚠️  No metrics found - skipping")
            results[detector] = {"status": "no_metrics"}
            print()
            continue

        all_satisfied, satisfaction = check_satisfaction(metrics, targets)

        print(f"   Current Metrics:")
        for metric, values in satisfaction.items():
            status_icon = "✅" if values["satisfied"] else "❌"
            print(f"      {status_icon} {metric}: {values['current']:.4f} (target: {values['target']:.2f})")

        if all_satisfied:
            print(f"   ✅ SATISFIED")
            results[detector] = {"status": "satisfied", "metrics": metrics}
        else:
            print(f"   ❌ NOT SATISFIED")
            results[detector] = {"status": "not_satisfied", "metrics": metrics}

        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    satisfied_count = sum(
        1 for r in results.values()
        if r.get("status") == "satisfied"
    )
    not_satisfied_count = sum(
        1 for r in results.values()
        if r.get("status") == "not_satisfied"
    )
    no_metrics_count = sum(
        1 for r in results.values()
        if r.get("status") == "no_metrics"
    )

    print(f"✅ Satisfied: {satisfied_count}/{len(detectors)}")
    print(f"❌ Not satisfied: {not_satisfied_count}/{len(detectors)}")
    print(f"⚠️  No metrics: {no_metrics_count}/{len(detectors)}")

    print()

    if satisfied_count == len(detectors):
        print("🎉 ALL DETECTORS SATISFIED!")
        return 0
    else:
        print("⚠️  Some detectors need optimization:")
        for detector, result in results.items():
            if result.get("status") == "not_satisfied":
                print(f"   - {detector}")
            elif result.get("status") == "no_metrics":
                print(f"   - {detector} (no metrics available)")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
