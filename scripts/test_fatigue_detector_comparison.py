"""
Compare Fatigue Detector v1 vs v2

Demonstrates lookahead bias in v1 and shows how v2 fixes it.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector
from src.meta.diagnoser.detectors.fatigue_detector_v2 import FatigueDetectorV2


def load_moprobo_ad_data(ad_id):
    """Load data for a specific ad."""
    ad_daily = pd.read_csv('datasets/moprobo/meta/raw/ad_daily_insights_2024-12-17_2025-12-17.csv')

    # Filter for specific ad
    ad_data = ad_daily[ad_daily['ad_id'] == ad_id].copy()

    # Preprocess
    numeric_cols = ['spend', 'impressions', 'reach', 'purchase_roas', 'clicks']
    for col in numeric_cols:
        if col in ad_data.columns:
            ad_data[col] = pd.to_numeric(ad_data[col], errors='coerce').fillna(0)
        else:
            ad_data[col] = 0

    # Parse conversions
    def parse_conversions(actions_str):
        import json
        if pd.isna(actions_str) or actions_str == "":
            return 0
        try:
            cleaned = actions_str.replace("'", '"')
            actions = json.loads(cleaned)
            purchase_keys = ["offsite_conversion.fb_pixel_purchase", "omni_purchase", "purchase"]
            total = 0.0
            for action in actions:
                if action.get("action_type", "") in purchase_keys:
                    total += float(action.get("value", 0))
            return total
        except:
            return 0

    if 'actions' in ad_data.columns:
        ad_data['conversions'] = ad_data['actions'].apply(parse_conversions)
    else:
        ad_data['conversions'] = 0

    ad_data['date_start'] = pd.to_datetime(ad_data['date_start'], errors='coerce')
    ad_data = ad_data.sort_values('date_start').reset_index(drop=True)

    return ad_data


def compare_detectors(ad_id):
    """Compare v1 and v2 detectors on the same ad."""
    print(f"\n{'='*80}")
    print(f"Comparing Detectors for Ad ID: {ad_id}")
    print('='*80)

    # Load data
    data = load_moprobo_ad_data(ad_id)
    print(f"\n📊 Data: {len(data)} days")
    print(f"Date range: {data['date_start'].min()} to {data['date_start'].max()}")

    # Run v1 detector (with lookahead bias)
    print(f"\n{'─'*80}")
    print("🔍 V1 DETECTOR (with lookahead bias)")
    print('='*80)

    detector_v1 = FatigueDetector()
    issues_v1 = detector_v1.detect(data, str(ad_id))

    if issues_v1:
        for issue in issues_v1:
            print(f"\n✅ Issue Detected:")
            print(f"   Title: {issue.title}")
            print(f"   Severity: {issue.severity.value}")
            print(f"   Description: {issue.description}")
            print(f"\n   Metrics:")
            for key, value in issue.metrics.items():
                if isinstance(value, (int, float, np.number)):
                    print(f"   - {key}: {value:.2f}")
                else:
                    print(f"   - {key}: {value}")
    else:
        print("\n✅ No fatigue detected")

    # Run v2 detector (no lookahead bias)
    print(f"\n{'─'*80}")
    print("🔍 V2 DETECTOR (rolling window, no lookahead bias)")
    print('='*80)

    detector_v2 = FatigueDetectorV2()
    issues_v2 = detector_v2.detect(data, str(ad_id))

    if issues_v2:
        for issue in issues_v2:
            print(f"\n✅ Issue Detected:")
            print(f"   Title: {issue.title}")
            print(f"   Severity: {issue.severity.value}")
            print(f"   Description: {issue.description}")
            print(f"\n   Metrics:")
            for key, value in issue.metrics.items():
                if isinstance(value, (int, float, np.number)):
                    print(f"   - {key}: {value:.2f}")
                else:
                    print(f"   - {key}: {value}")
    else:
        print("\n✅ No fatigue detected")

    # Comparison
    print(f"\n{'─'*80}")
    print("📊 COMPARISON")
    print('='*80)

    if issues_v1 and issues_v2:
        severity_v1 = issues_v1[0].metrics.get('severity_score', 0)
        severity_v2 = issues_v2[0].metrics.get('severity_score', 0)

        print(f"\nV1 Severity Score: {severity_v1:.1f}/100")
        print(f"V2 Severity Score: {severity_v2:.1f}/100")
        print(f"Difference: {abs(severity_v1 - severity_v2):.1f} points")

        if abs(severity_v1 - severity_v2) > 10:
            print(f"\n⚠️  SIGNIFICANT DIFFERENCE detected!")
            print(f"   This demonstrates lookahead bias in V1.")
    elif issues_v1:
        print("\n⚠️  V1 detected fatigue, but V2 did not.")
        print("   This suggests V1's detection may be a false positive due to lookahead bias.")
    elif issues_v2:
        print("\n✅ V2 detected fatigue (more conservative, no lookahead bias)")
    else:
        print("\n✅ Both detectors agree: No fatigue detected")


def analyze_lookahead_bias():
    """Demonstrate lookahead bias with concrete example."""
    print(f"\n{'='*80}")
    print("DEMONSTRATING LOOKAHEAD BIAS")
    print('='*80)

    # Create synthetic example
    print("\n📊 Synthetic Example:")
    print("Imagine an ad with the following data:")
    print("")
    print("Day 1-30:   Golden period (freq 1.5-2.0x, CPA $40)")
    print("Day 31-60:  Frequency increases to 3.5x")
    print("Day 61-90:  CPA spikes to $80 (fatigue!)")
    print("Day 91-100: CPA drops back to $45 (recovery)")
    print("")

    print("V1 Detector (with lookahead bias):")
    print("  - Uses ALL data (Day 1-100)")
    print("  - Calculates cum_freq using Day 1-100")
    print("  - Finds golden period in Day 1-30")
    print("  - Compares Day 61-90 CPA ($80) to golden CPA ($40)")
    print("  - Result: ✅ Detects fatigue on Day 61")
    print("  ❌ Problem: On Day 61, it shouldn't know about Day 91-100 recovery!")

    print("")
    print("V2 Detector (rolling window, no lookahead bias):")
    print("  - Uses 60-day rolling window")
    print("  - On Day 61: window = Day 1-60")
    print("  - On Day 90: window = Day 30-89")
    print("  - Each window independently calculates cum_freq")
    print("  - Result: ✅ Detects fatigue on Day 61-90")
    print("  ✅ No lookahead bias: only uses historical data")

    print("")
    print("Key Difference:")
    print("  V1: Sees the 'whole movie' before making predictions")
    print("  V2: Only sees 'past scenes', predicts 'next scene'")


def main():
    print("🚀 Fatigue Detector Comparison: V1 vs V2")
    print("="*80)

    # Demonstrate lookahead bias concept
    analyze_lookahead_bias()

    # Test on real moprobo data
    test_ad_ids = [
        120215767568610310,  # Showed fatigue in v1
        120215893529500310,  # Highest severity in v1
        120215808390050310,  # Mid severity in v1
    ]

    for ad_id in test_ad_ids:
        try:
            compare_detectors(ad_id)
        except Exception as e:
            print(f"\n❌ Error testing ad {ad_id}: {e}")
            continue

    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print("""
V1 (Current Implementation):
  ❌ Uses ALL data (lookahead bias)
  ❌ Knows 'future' when making predictions
  ❌ Overly optimistic results
  ❌ May produce false positives

V2 (Recommended):
  ✅ Uses rolling window (no lookahead bias)
  ✅ Only uses historical data
  ✅ Realistic predictions
  ✅ Suitable for production deployment

Recommendation: Migrate to V2 for production use.
""")


if __name__ == "__main__":
    main()
