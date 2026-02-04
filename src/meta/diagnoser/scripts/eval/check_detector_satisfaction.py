#!/usr/bin/env python3
"""
Check detector satisfaction against target metrics.

This script loads evaluation reports for all detectors and checks if they meet
their target metrics (F1, precision, recall).

Usage:
    python eval/check_detector_satisfaction.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.scripts.utils.metrics_utils import (
    load_metrics,
    check_satisfaction,
    format_metrics_report,
)


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
