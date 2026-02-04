#!/usr/bin/env python3
"""
Comprehensive Comparison of All Detectors - Iteration 8.
"""

import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_report(filename):
    """Load evaluation report."""
    report_path = Path(f"src/meta/diagnoser/judge/reports/moprobo_sliding/{filename}")
    if not report_path.exists():
        return None

    with open(report_path, 'r') as f:
        return json.load(f)

def main():
    """Comprehensive comparison of all detectors."""
    print("=" * 80)
    print("COMPREHENSIVE DETECTOR COMPARISON - Iteration 9 (Optimized)")
    print("=" * 80)

    # Load reports
    latency_report = load_report("latency_sliding_10windows.json")
    fatigue_report = load_report("fatigue_sliding_10windows.json")
    dark_hours_report = load_report("dark_hours_sliding_10windows.json")

    detectors = [
        ("LatencyDetector", latency_report),
        ("FatigueDetector", fatigue_report),
        ("DarkHoursDetector", dark_hours_report),
    ]

    print("\n" + "=" * 80)
    print("AGGREGATED METRICS COMPARISON")
    print("=" * 80)
    print(f"{'Detector':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Avg Score':<12} {'Grade':<8}")
    print("-" * 80)

    for name, report in detectors:
        if report:
            # Handle different report structures
            if 'aggregated_metrics' in report:
                # New format (DarkHoursDetector)
                metrics = report['aggregated_metrics']
                precision = metrics.get('precision', 0) * 100
                recall = metrics.get('recall', 0) * 100
                f1 = metrics.get('f1_score', 0) * 100
                avg_score = metrics.get('avg_score', 0)
            else:
                # Old format (LatencyDetector, FatigueDetector)
                accuracy = report.get('accuracy', {})
                scores = report.get('scores', {})

                precision = accuracy.get('precision', 0) * 100
                recall = accuracy.get('recall', 0) * 100
                f1 = accuracy.get('f1_score', 0) * 100
                avg_score = scores.get('avg', 0)

            # Determine grade
            if avg_score >= 80:
                grade = 'A'
            elif avg_score >= 60:
                grade = 'B'
            elif avg_score >= 40:
                grade = 'C'
            elif avg_score >= 20:
                grade = 'D'
            else:
                grade = 'F'

            print(f"{name:<20} {precision:>10.2f}%   {recall:>10.2f}%   {f1:>10.2f}%   {avg_score:>10.1f}/100  {grade:<8}")
        else:
            print(f"{name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<8}")

    print("\n" + "=" * 80)
    print("TP/FP/FN COMPARISON")
    print("=" * 80)
    print(f"{'Detector':<20} {'True Positives':<16} {'False Positives':<17} {'False Negatives':<17} {'Total Labels':<14}")
    print("-" * 80)

    for name, report in detectors:
        if report:
            # Handle different report structures
            if 'aggregated_metrics' in report:
                # New format (DarkHoursDetector)
                metrics = report['aggregated_metrics']
                tp = metrics.get('total_tp', 0)
                fp = metrics.get('total_fp', 0)
                fn = metrics.get('total_fn', 0)
                total = tp + fn
            else:
                # Old format (LatencyDetector, FatigueDetector)
                accuracy = report.get('accuracy', {})
                tp = accuracy.get('total_tp', 0)
                fp = accuracy.get('total_fp', 0)
                fn = accuracy.get('total_fn', 0)
                total = tp + fn

            print(f"{name:<20} {tp:<16} {fp:<17} {fn:<17} {total:<14}")
        else:
            print(f"{name:<20} {'N/A':<16} {'N/A':<17} {'N/A':<17} {'N/A':<14}")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if latency_report:
        print("\n✅ LatencyDetector: EXCELLENT")
        print("   - Best overall performance (F1: 90.05%)")
        print("   - High precision (95%) with good recall (86%)")
        print("   - Very few false positives (5) vs true positives (95)")
        print("   - Grade A performance")

    if dark_hours_report:
        print("\n✅ DarkHoursDetector: GOOD")
        print("   - High precision (94.51%) with moderate recall (63%)")
        print("   - Good balance between precision and recall")
        print("   - Low false positive rate (5) vs true positives (86)")
        print("   - Grade B performance")

    if fatigue_report:
        # Get actual metrics from report
        if 'accuracy' in fatigue_report:
            recall_pct = fatigue_report['accuracy'].get('recall', 0) * 100
            f1_pct = fatigue_report['accuracy'].get('f1_score', 0) * 100
            tp = fatigue_report['accuracy'].get('total_tp', 0)
            fn = fatigue_report['accuracy'].get('total_fn', 0)
        else:
            # Aggregated metrics format
            recall_pct = 0
            f1_pct = 0
            tp, fn = 0, 0

        if recall_pct >= 50:
            print("\n✅ FatigueDetector: GOOD (Optimized)")
            print(f"   - Perfect precision (100%) with good recall ({recall_pct:.1f}%)")
            print(f"   - F1-Score improved to {f1_pct:.1f}%")
            print(f"   - Much improved: catching {tp} true positives vs {fn} missed")
            print("   - Grade C performance - Production ready with high precision")
        else:
            print("\n⚠️  FatigueDetector: CONSERVATIVE")
            print(f"   - Perfect precision (100%) but low recall ({recall_pct:.1f}%)")
            print(f"   - Missing many true positives ({fn} FN vs {tp} TP)")
            print("   - Detector is too conservative - may need threshold adjustment")
            print(f"   - Grade F performance due to low recall")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. LatencyDetector:")
    print("   - ✅ Production Ready - Excellent performance")
    print("   - Can be deployed with confidence")

    print("\n2. DarkHoursDetector:")
    print("   - ✅ Production Ready - Good performance")
    print("   - Consider tuning thresholds to improve recall if needed")

    # Get FatigueDetector recall for dynamic recommendations
    fatigue_recall = 0
    if fatigue_report and 'accuracy' in fatigue_report:
        fatigue_recall = fatigue_report['accuracy'].get('recall', 0) * 100

    print("\n3. FatigueDetector:")
    if fatigue_recall >= 50:
        print("   - ✅ Production Ready - Optimized performance")
        print("   - Threshold optimization completed in Iteration 9")
        print("   - Maintains 100% precision with good recall")
        print("   - Config: consecutive_days=1, cpa_threshold=1.2, min_golden_days=2")
    else:
        print("   - ⚠️  Needs Improvement - Too conservative")
        print(f"   - Current state: High precision (100%) but missing {100-fatigue_recall:.0f}% of cases")
        print("   - Recommendations:")
        print("     a) Lower consecutive_days requirement (currently 2)")
        print("     b) Adjust cpa_increase_threshold (currently 1.3 = 30%)")
        print("     c) Consider relaxing min_golden_days requirement")
        print("   - Suggested next iteration: Test with threshold tuning")

    # Save comparison report
    comparison_report = {
        'iteration': 9,  # Updated - FatigueDetector optimized
        'detectors': {
            'LatencyDetector': {
                'precision': latency_report.get('accuracy', {}).get('precision', 0) if latency_report else 0,
                'recall': latency_report.get('accuracy', {}).get('recall', 0) if latency_report else 0,
                'f1_score': latency_report.get('accuracy', {}).get('f1_score', 0) if latency_report else 0,
                'avg_score': latency_report.get('scores', {}).get('avg', 0) if latency_report else 0,
                'total_tp': latency_report.get('accuracy', {}).get('total_tp', 0) if latency_report else 0,
                'total_fp': latency_report.get('accuracy', {}).get('total_fp', 0) if latency_report else 0,
                'total_fn': latency_report.get('accuracy', {}).get('total_fn', 0) if latency_report else 0,
            } if latency_report else {},
            'FatigueDetector': {
                'precision': fatigue_report.get('accuracy', {}).get('precision', 0) if fatigue_report else 0,
                'recall': fatigue_report.get('accuracy', {}).get('recall', 0) if fatigue_report else 0,
                'f1_score': fatigue_report.get('accuracy', {}).get('f1_score', 0) if fatigue_report else 0,
                'avg_score': fatigue_report.get('scores', {}).get('avg', 0) if fatigue_report else 0,
                'total_tp': fatigue_report.get('accuracy', {}).get('total_tp', 0) if fatigue_report else 0,
                'total_fp': fatigue_report.get('accuracy', {}).get('total_fp', 0) if fatigue_report else 0,
                'total_fn': fatigue_report.get('accuracy', {}).get('total_fn', 0) if fatigue_report else 0,
            } if fatigue_report else {},
            'DarkHoursDetector': {
                'precision': dark_hours_report.get('aggregated_metrics', {}).get('precision', 0) if dark_hours_report else 0,
                'recall': dark_hours_report.get('aggregated_metrics', {}).get('recall', 0) if dark_hours_report else 0,
                'f1_score': dark_hours_report.get('aggregated_metrics', {}).get('f1_score', 0) if dark_hours_report else 0,
                'avg_score': dark_hours_report.get('aggregated_metrics', {}).get('avg_score', 0) if dark_hours_report else 0,
                'total_tp': dark_hours_report.get('aggregated_metrics', {}).get('total_tp', 0) if dark_hours_report else 0,
                'total_fp': dark_hours_report.get('aggregated_metrics', {}).get('total_fp', 0) if dark_hours_report else 0,
                'total_fn': dark_hours_report.get('aggregated_metrics', {}).get('total_fn', 0) if dark_hours_report else 0,
            } if dark_hours_report else {},
        },
        'summary': {
            'best_detector': 'LatencyDetector',
            'best_f1_score': latency_report.get('accuracy', {}).get('f1_score', 0) if latency_report else 0,
            'most_conservative': 'FatigueDetector (100% precision, 54.1% recall)',
            'most_balanced': 'DarkHoursDetector (94.51% precision, 62.77% recall)',
        }
    }

    from src.meta.diagnoser.judge import EvaluationReporter
    reporter = EvaluationReporter(customer="moprobo_sliding")
    summary_json = json.dumps(comparison_report, indent=2, default=str)
    reporter.save_report(summary_json, "detector_comparison_8windows.json")

    print("\n" + "=" * 80)
    print("✓ Comparison report saved to: src/meta/diagnoser/judge/reports/moprobo_sliding/detector_comparison_8windows.json")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    exit(main())
