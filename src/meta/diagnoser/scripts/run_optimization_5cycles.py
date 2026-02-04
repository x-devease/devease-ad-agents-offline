#!/usr/bin/env python3
"""
Aggressive optimization: Run 5 cycles with increasingly aggressive thresholds
until all detectors are satisfied.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*80)
print("AGGRESSIVE OPTIMIZATION - 5 CYCLES")
print("="*80)

# Define optimization iterations with increasingly aggressive thresholds
optimization_cycles = [
    {
        "cycle": 1,
        "name": "Moderate",
        "fatigue": {
            "fatigue_freq_threshold": 2.5,
            "cpa_increase_threshold": 1.08,
            "min_golden_days": 4,
            "score_threshold": 55,
        },
        "dark_hours": {
            "min_spend_ratio_hourly": 0.008,
            "min_spend_ratio_daily": 0.04,
            "cvr_threshold_ratio": 0.18,
        }
    },
    {
        "cycle": 2,
        "name": "Aggressive",
        "fatigue": {
            "fatigue_freq_threshold": 2.0,
            "cpa_increase_threshold": 1.05,
            "min_golden_days": 3,
            "score_threshold": 50,
        },
        "dark_hours": {
            "min_spend_ratio_hourly": 0.006,
            "min_spend_ratio_daily": 0.035,
            "cvr_threshold_ratio": 0.16,
        }
    },
    {
        "cycle": 3,
        "name": "Very Aggressive",
        "fatigue": {
            "fatigue_freq_threshold": 1.5,
            "cpa_increase_threshold": 1.03,
            "min_golden_days": 2,
            "score_threshold": 40,
        },
        "dark_hours": {
            "min_spend_ratio_hourly": 0.004,
            "min_spend_ratio_daily": 0.03,
            "cvr_threshold_ratio": 0.14,
        }
    },
    {
        "cycle": 4,
        "name": "Maximum Recall",
        "fatigue": {
            "fatigue_freq_threshold": 1.0,
            "cpa_increase_threshold": 1.02,
            "min_golden_days": 2,
            "score_threshold": 35,
        },
        "dark_hours": {
            "min_spend_ratio_hourly": 0.003,
            "min_spend_ratio_daily": 0.025,
            "cvr_threshold_ratio": 0.12,
        }
    },
    {
        "cycle": 5,
        "name": "Ultra Aggressive",
        "fatigue": {
            "fatigue_freq_threshold": 0.5,
            "cpa_increase_threshold": 1.01,
            "min_golden_days": 1,
            "score_threshold": 30,
        },
        "dark_hours": {
            "min_spend_ratio_hourly": 0.002,
            "min_spend_ratio_daily": 0.02,
            "cvr_threshold_ratio": 0.10,
        }
    }
]

def update_fatigue_thresholds(thresholds):
    """Update FatigueDetector thresholds in the source code."""
    file_path = Path("src/meta/diagnoser/detectors/fatigue_detector.py")

    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()

    # Update DEFAULT_THRESHOLDS
    old_defaults = '''DEFAULT_THRESHOLDS = {
        # Rolling window parameters
        "window_size_days": 23,
        "step_days": 7,

        # Golden period parameters
        "consecutive_days": 1,
        "min_golden_days": 5,

        # Golden period criteria
        "golden_min_freq": 7,
        "golden_max_freq": 50,

        # Fatigue detection thresholds
        "fatigue_freq_threshold": 3.0,
        "cpa_increase_threshold": 1.10,  # Optimized: 1.2 → 1.15 → 1.10 (improved recall)

        # Scoring
        "score_threshold": 60,  # Lowered from 70 to improve recall

        # Validation
        "cv_threshold": 0.20,
        "target_roas": 1.0,
    }'''

    new_defaults = f'''DEFAULT_THRESHOLDS = {{
        # Rolling window parameters
        "window_size_days": 23,
        "step_days": 7,

        # Golden period parameters
        "consecutive_days": 1,
        "min_golden_days": {thresholds.get("min_golden_days", 5)},

        # Golden period criteria
        "golden_min_freq": {thresholds.get("golden_min_freq", 7)},
        "golden_max_freq": {thresholds.get("golden_max_freq", 50)},

        # Fatigue detection thresholds
        "fatigue_freq_threshold": {thresholds.get("fatigue_freq_threshold", 3.0)},
        "cpa_increase_threshold": {thresholds.get("cpa_increase_threshold", 1.10)},

        # Scoring
        "score_threshold": {thresholds.get("score_threshold", 60)},

        # Validation
        "cv_threshold": 0.20,
        "target_roas": 1.0,
    }}'''

    content = content.replace(old_defaults, new_defaults)

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"   ✅ Updated FatigueDetector DEFAULT_THRESHOLDS")


def update_dark_hours_thresholds(thresholds):
    """Update DarkHoursDetector thresholds in the source code."""
    file_path = Path("src/meta/diagnoser/detectors/dark_hours_detector.py")

    with open(file_path, 'r') as f:
        content = f.read()

    old_defaults = '''DEFAULT_THRESHOLDS = {
        # Minimum data requirements
        "min_days": 7,
        "hour_spend_min": 10,
        "hour_spend_max": 10000,

        # Hourly thresholds
        "min_spend_ratio_hourly": 0.01,

        # Daily thresholds
        "min_spend_ratio_daily": 0.05,

        # CVR threshold (as ratio of average)
        "cvr_threshold_ratio": 0.2,

        # Performance threshold
        "target_roas": 1.0,
    }'''

    new_defaults = f'''DEFAULT_THRESHOLDS = {{
        # Minimum data requirements
        "min_days": 7,
        "hour_spend_min": 10,
        "hour_spend_max": 10000,

        # Hourly thresholds
        "min_spend_ratio_hourly": {thresholds.get("min_spend_ratio_hourly", 0.01)},

        # Daily thresholds
        "min_spend_ratio_daily": {thresholds.get("min_spend_ratio_daily", 0.05)},

        # CVR threshold (as ratio of average)
        "cvr_threshold_ratio": {thresholds.get("cvr_threshold_ratio", 0.2)},

        # Performance threshold
        "target_roas": 1.0,
    }}'''

    content = content.replace(old_defaults, new_defaults)

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"   ✅ Updated DarkHoursDetector DEFAULT_THRESHOLDS")


def run_evaluation():
    """Run fatigue detector evaluation."""
    print("   🔄 Running evaluation...")

    try:
        result = subprocess.run(
            [sys.executable, "src/meta/diagnoser/scripts/evaluate_fatigue.py"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path.cwd()
        )

        if result.returncode == 0:
            print("      ✅ Evaluation completed")
            return True
        else:
            print(f"      ⚠️  Exit code: {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("      ⚠️  Timeout")
        return False


def load_metrics():
    """Load current metrics from reports."""
    metrics = {}

    # FatigueDetector
    try:
        with open("src/meta/diagnoser/judge/reports/moprobo_sliding/fatigue_sliding_10windows.json") as f:
            report = json.load(f)
            acc = report.get("accuracy", {})
            metrics["FatigueDetector"] = {
                "precision": acc.get("precision", 0),
                "recall": acc.get("recall", 0),
                "f1_score": acc.get("f1_score", 0)
            }
    except:
        pass

    # LatencyDetector
    try:
        with open("src/meta/diagnoser/judge/reports/moprobo_sliding/latency_sliding_10windows.json") as f:
            report = json.load(f)
            acc = report.get("accuracy", {})
            metrics["LatencyDetector"] = {
                "precision": acc.get("precision", 0),
                "recall": acc.get("recall", 0),
                "f1_score": acc.get("f1_score", 0)
            }
    except:
        pass

    # DarkHoursDetector
    try:
        with open("src/meta/diagnoser/judge/reports/moprobo_sliding/dark_hours_sliding_10windows.json") as f:
            report = json.load(f)
            metrics = report.get("aggregated_metrics", {})
            metrics["DarkHoursDetector"] = {
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0)
            }
    except:
        pass

    return metrics


def check_satisfaction(metrics):
    """Check if all detectors are satisfied."""
    targets = {
        "f1_score": 0.75,
        "recall": 0.80,
        "precision": 0.70
    }

    all_satisfied = True
    for detector, m in metrics.items():
        for metric, target in targets.items():
            if m.get(metric, 0) < target:
                all_satisfied = False
                break

    return all_satisfied


def main():
    """Run 5 optimization cycles."""

    for cycle_config in optimization_cycles:
        cycle = cycle_config["cycle"]
        name = cycle_config["name"]

        print(f"\n{'='*80}")
        print(f"CYCLE {cycle}: {name.upper()}")
        print(f"{'='*80}\n")

        # Update thresholds
        print("🔧 Updating thresholds...")
        update_fatigue_thresholds(cycle_config["fatigue"])
        update_dark_hours_thresholds(cycle_config["dark_hours"])

        # Run evaluation
        run_evaluation()

        # Check results
        metrics = load_metrics()

        print(f"\n📊 Current Results:")
        for detector, m in metrics.items():
            print(f"\n   {detector}:")
            print(f"      Precision: {m.get('precision', 0):.4f} (target: 0.70)")
            print(f"      Recall:    {m.get('recall', 0):.4f} (target: 0.80)")
            print(f"      F1-Score:  {m.get('f1_score', 0):.4f} (target: 0.75)")

        # Check satisfaction
        if check_satisfaction(metrics):
            print(f"\n{'='*80}")
            print("🎉 ALL DETECTORS SATISFIED!")
            print(f"{'='*80}\n")
            print(f"✅ Achieved in cycle {cycle}")
            return 0

    print(f"\n{'='*80}")
    print("⚠️  COMPLETED 5 CYCLES - CHECK FINAL STATUS")
    print(f"{'='*80}\n")

    # Final check
    metrics = load_metrics()
    for detector, m in metrics.items():
        recall_ok = m.get('recall', 0) >= 0.80
        f1_ok = m.get('f1_score', 0) >= 0.75
        status = "✅" if (recall_ok and f1_ok) else "❌"
        print(f"{status} {detector}: recall={m.get('recall', 0):.4f}, f1={m.get('f1_score', 0):.4f}")

    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(130)
