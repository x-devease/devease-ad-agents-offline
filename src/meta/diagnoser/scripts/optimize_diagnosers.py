#!/usr/bin/env python3
"""
Run diagnoser agent optimization cycles until satisfaction.

This script will:
1. Evaluate current detector performance
2. Optimize thresholds for unsatisfied detectors
3. Re-run evaluation
4. Repeat until all satisfied or max cycles reached
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def backup_evaluation_reports():
    """Backup current evaluation reports."""
    reports_dir = Path("src/meta/diagnoser/judge/reports/moprobo_sliding")
    backup_dir = Path(f"src/meta/diagnoser/judge/reports/moprobo_sliding_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if reports_dir.exists():
        shutil.copytree(reports_dir, backup_dir)
        print(f"📦 Backed up reports to: {backup_dir}")
        return backup_dir
    return None


def optimize_fatigue_detector():
    """Optimize FatigueDetector thresholds to improve recall."""
    print("\n🔧 Optimizing FatigueDetector...")

    # Current recall is low (0.59), need to improve to 0.80
    # Lower thresholds to catch more cases (will reduce precision but increase recall)
    new_thresholds = {
        "fatigue_freq_threshold": 2.0,      # Lower from 3.0
        "cpa_increase_threshold": 1.05,    # Lower from 1.10
        "min_golden_days": 3,              # Lower from 5
        "golden_min_freq": 5,              # Lower from 7
        "golden_max_freq": 55,             # Widen range
        "score_threshold": 50,             # Lower from 60
        "window_size_days": 10,
        "consecutive_days": 3,
        "cv_threshold": 0.15,
        "target_roas": 1.0
    }

    config_file = Path("src/meta/diagnoser/detectors/config/fatigue_detector_config.json")

    # Create config directory if doesn't exist
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Write optimized config
    with open(config_file, 'w') as f:
        json.dump({
            "detector_name": "FatigueDetector",
            "description": "Optimized for high recall (80%)",
            "thresholds": new_thresholds
        }, f, indent=2)

    print(f"   ✅ Updated FatigueDetector thresholds in {config_file}")
    print(f"   - fatigue_freq_threshold: 3.0 → 2.0")
    print(f"   - cpa_increase_threshold: 1.10 → 1.05")
    print(f"   - min_golden_days: 5 → 3")
    print(f"   - score_threshold: 60 → 50")

    return True


def optimize_dark_hours_detector():
    """Optimize DarkHoursDetector thresholds to improve recall."""
    print("\n🔧 Optimizing DarkHoursDetector...")

    # Current recall is low (0.63), need to improve to 0.80
    # Lower cvr threshold to catch more dark hours
    new_thresholds = {
        "min_days": 7,
        "min_spend_ratio_hourly": 0.005,   # Lower from 0.01
        "min_spend_ratio_daily": 0.03,     # Lower from 0.05
        "cvr_threshold_ratio": 0.15,       # Lower from 0.20
        "target_roas": 0.8,                # Lower from 1.0
        "hour_spend_min": 10,
        "hour_spend_max": 10000
    }

    config_file = Path("src/meta/diagnoser/detectors/config/dark_hours_detector_config.json")

    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        json.dump({
            "detector_name": "DarkHoursDetector",
            "description": "Optimized for high recall (80%)",
            "thresholds": new_thresholds
        }, f, indent=2)

    print(f"   ✅ Updated DarkHoursDetector thresholds in {config_file}")
    print(f"   - min_spend_ratio_hourly: 0.01 → 0.005")
    print(f"   - min_spend_ratio_daily: 0.05 → 0.03")
    print(f"   - cvr_threshold_ratio: 0.20 → 0.15")
    print(f"   - target_roas: 1.0 → 0.8")

    return True


def run_evaluation_scripts():
    """Run the evaluation scripts to generate new metrics."""
    print("\n🔄 Running evaluation scripts...")

    scripts = [
        "src/meta/diagnoser/scripts/evaluate_fatigue.py",
        "src/meta/diagnoser/scripts/evaluate_dark_hours.py"
    ]

    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"   ⚠️  Script not found: {script}")
            continue

        print(f"   📜 Running: {script_path.name}")
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=Path.cwd()
            )
            if result.returncode == 0:
                print(f"      ✅ Completed")
            else:
                print(f"      ⚠️  Exit code: {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"      ⚠️  Timeout after 120s")
        except Exception as e:
            print(f"      ❌ Error: {e}")


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


def run_optimization_cycle(cycle_num: int):
    """Run one optimization cycle."""
    print("\n" + "="*80)
    print(f"OPTIMIZATION CYCLE #{cycle_num}")
    print("="*80)

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

    print(f"\n📋 Evaluating {len(detectors)} detectors...")

    # Load current metrics
    results = {}
    needs_optimization = []

    for detector, targets in detectors.items():
        print(f"\n🔍 {detector}")

        metrics = load_metrics(detector)

        if metrics is None:
            print(f"   ⚠️  No metrics found")
            results[detector] = {"status": "no_metrics", "needs_opt": False}
            continue

        all_satisfied, satisfaction = check_satisfaction(metrics, targets)

        print(f"   Current Metrics:")
        for metric, values in satisfaction.items():
            status_icon = "✅" if values["satisfied"] else "❌"
            print(f"      {status_icon} {metric}: {values['current']:.4f} (target: {values['target']:.2f})")

        if all_satisfied:
            print(f"   ✅ SATISFIED")
            results[detector] = {"status": "satisfied", "metrics": metrics, "needs_opt": False}
        else:
            print(f"   ❌ NOT SATISFIED - will optimize")
            results[detector] = {"status": "not_satisfied", "metrics": metrics, "needs_opt": True}
            needs_optimization.append(detector)

    # If all satisfied, return early
    if not needs_optimization:
        print("\n" + "="*80)
        print("🎉 ALL DETECTORS SATISFIED!")
        print("="*80)
        return True, results

    # Optimize unsatisfied detectors
    print(f"\n🔧 Optimizing {len(needs_optimization)} detectors...")

    if "FatigueDetector" in needs_optimization:
        optimize_fatigue_detector()

    if "DarkHoursDetector" in needs_optimization:
        optimize_dark_hours_detector()

    # Re-run evaluations with new thresholds
    run_evaluation_scripts()

    # Check if optimization should continue
    return False, results


def main():
    """Main optimization function."""
    print("\n" + "="*80)
    print("DIAGNOSER AGENT OPTIMIZATION - 5 CYCLES")
    print("="*80)

    # Backup original reports
    backup_dir = backup_evaluation_reports()

    max_cycles = 5
    all_satisfied = False

    for cycle in range(1, max_cycles + 1):
        all_satisfied, results = run_optimization_cycle(cycle)

        if all_satisfied:
            print(f"\n✅ All detectors satisfied after {cycle} cycle(s)!")
            break

        if cycle < max_cycles:
            print(f"\n⏳ Continuing to cycle {cycle + 1}...")
        else:
            print(f"\n⚠️  Reached maximum {max_cycles} cycles")

    # Final summary
    print("\n" + "="*80)
    print("FINAL OPTIMIZATION SUMMARY")
    print("="*80)

    if all_satisfied:
        print("\n🎉 SUCCESS: All detectors are now satisfied!")
        return 0
    else:
        print("\n⚠️  Optimization complete - some detectors may still need tuning:")
        for detector, result in results.items():
            if result.get("needs_opt"):
                print(f"   - {detector}: {result.get('status', 'unknown')}")
        print(f"\n💡 Original reports backed up to: {backup_dir}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
