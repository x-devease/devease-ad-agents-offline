"""
Test Diagnosers with Moprobo Data

Runs all three detectors on moprobo data and displays results.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector
from src.meta.diagnoser.detectors.latency_detector import LatencyDetector, infer_status_changes
from src.meta.diagnoser.detectors.dark_hours_detector import DarkHoursDetector


def load_and_preprocess_data():
    """Load and preprocess moprobo data."""
    base_path = "datasets/moprobo/meta/raw"

    print("📂 Loading moprobo data...")

    # Load daily data
    ad_daily = pd.read_csv(f"{base_path}/ad_daily_insights_2024-12-17_2025-12-17.csv")
    adset_daily = pd.read_csv(f"{base_path}/adset_daily_insights_2024-12-17_2025-12-17.csv")

    # Load hourly data (if available)
    try:
        adset_hourly = pd.read_csv(f"{base_path}/adset_hourly_insights_2025-09-01_2025-12-11.csv")
        print(f"✅ Loaded adset hourly data: {len(adset_hourly)} rows")
    except:
        adset_hourly = None
        print("⚠️  No hourly data available")

    print(f"✅ Loaded ad_daily: {len(ad_daily)} rows")
    print(f"✅ Loaded adset_daily: {len(adset_daily)} rows")

    # Preprocess ad_daily for fatigue detector
    ad_daily_processed = ad_daily.copy()

    # Convert numeric columns (including purchase_roas)
    numeric_cols = ['spend', 'impressions', 'reach', 'purchase_roas', 'clicks']
    for col in numeric_cols:
        if col in ad_daily_processed.columns:
            ad_daily_processed[col] = pd.to_numeric(ad_daily_processed[col], errors='coerce').fillna(0)
        else:
            ad_daily_processed[col] = 0

    # Parse conversions
    if 'actions' in ad_daily_processed.columns:
        ad_daily_processed['conversions'] = ad_daily_processed['actions'].apply(
            lambda x: parse_conversions_from_json(x) if pd.notna(x) else 0
        )
    else:
        ad_daily_processed['conversions'] = 0

    # Preprocess adset_daily for latency and dark_hours detectors
    adset_daily_processed = adset_daily.copy()

    # Convert numeric columns
    numeric_cols = ['spend', 'purchase_roas']
    for col in numeric_cols:
        if col in adset_daily_processed.columns:
            adset_daily_processed[col] = pd.to_numeric(adset_daily_processed[col], errors='coerce').fillna(0)

    # Calculate purchase_roas if not present or invalid
    if 'purchase_roas' not in adset_daily_processed.columns or adset_daily_processed['purchase_roas'].isna().all():
        if 'action_values' in adset_daily_processed.columns:
            purchase_value = adset_daily_processed['action_values'].apply(
                lambda x: parse_purchase_value_from_json(x) if pd.notna(x) else 0
            )
            adset_daily_processed['purchase_roas'] = purchase_value / adset_daily_processed['spend'].replace(0, np.nan)
        else:
            adset_daily_processed['purchase_roas'] = 0

    adset_daily_processed['purchase_roas'] = adset_daily_processed['purchase_roas'].fillna(0).replace([np.inf, -np.inf], 0)

    # Convert date_start to datetime if present
    if 'date_start' in ad_daily_processed.columns:
        ad_daily_processed['date_start'] = pd.to_datetime(ad_daily_processed['date_start'], errors='coerce')
    if 'date_start' in adset_daily_processed.columns:
        adset_daily_processed['date_start'] = pd.to_datetime(adset_daily_processed['date_start'], errors='coerce')

    # Preprocess hourly data if available
    if adset_hourly is not None:
        adset_hourly_processed = adset_hourly.copy()
        # Convert numeric columns
        numeric_cols = ['spend', 'purchase_roas', 'hour', 'conversions', 'clicks']
        for col in numeric_cols:
            if col in adset_hourly_processed.columns:
                adset_hourly_processed[col] = pd.to_numeric(adset_hourly_processed[col], errors='coerce').fillna(0)
            else:
                adset_hourly_processed[col] = 0

        # Convert date_start to datetime
        if 'date_start' in adset_hourly_processed.columns:
            adset_hourly_processed['date_start'] = pd.to_datetime(adset_hourly_processed['date_start'], errors='coerce')
    else:
        adset_hourly_processed = None

    return {
        'ad_daily': ad_daily_processed,
        'adset_daily': adset_daily_processed,
        'adset_hourly': adset_hourly_processed,
    }


def parse_conversions_from_json(actions_str):
    """Parse conversions from actions JSON string."""
    import json
    if pd.isna(actions_str) or actions_str == "":
        return 0

    try:
        cleaned = actions_str.replace("'", '"')
        actions = json.loads(cleaned)

        purchase_keys = [
            "offsite_conversion.fb_pixel_purchase",
            "omni_purchase",
            "purchase",
            "onsite_web_purchase",
        ]

        total = 0.0
        for action in actions:
            action_type = action.get("action_type", "")
            if action_type in purchase_keys:
                total += float(action.get("value", 0))

        return total
    except:
        return 0


def parse_purchase_value_from_json(action_values_str):
    """Parse purchase value from action_values JSON string."""
    import json
    if pd.isna(action_values_str) or action_values_str == "":
        return 0

    try:
        cleaned = action_values_str.replace("'", '"')
        action_values = json.loads(cleaned)

        purchase_keys = [
            "offsite_conversion.fb_pixel_purchase",
            "omni_purchase",
            "purchase",
        ]

        for action in action_values:
            action_type = action.get("action_type", "")
            if action_type in purchase_keys:
                return float(action.get("value", 0))

        return 0
    except:
        return 0


def test_fatigue_detector(data):
    """Test FatigueDetector."""
    print("\n" + "="*60)
    print("🔍 TESTING FATIGUE DETECTOR (Rolling Window, No Lookahead Bias)")
    print("="*60)

    # Use 30-day rolling window (default)
    detector = FatigueDetector()

    ad_daily = data['ad_daily']

    # Get unique ads - test more ads
    ad_ids = ad_daily['ad_id'].unique() if 'ad_id' in ad_daily.columns else []
    print(f"\n📊 Testing {min(20, len(ad_ids))} ads (sample)...")

    all_issues = []
    tested_count = 0
    skipped_count = 0

    for i, ad_id in enumerate(ad_ids[:20]):
        ad_data = ad_daily[ad_daily['ad_id'] == ad_id] if 'ad_id' in ad_daily.columns else ad_daily.iloc[:100]

        if len(ad_data) < 33:  # Need 30 days + 3 consecutive
            skipped_count += 1
            continue

        tested_count += 1
        issues = detector.detect(ad_data, str(ad_id))
        all_issues.extend(issues)

        if issues:
            for issue in issues:
                print(f"\n{'─'*60}")
                print(f"📢 Ad ID: {ad_id}")
                print(f"Title: {issue.title}")
                print(f"Severity: {issue.severity.value}")
                print(f"Description: {issue.description}")
                print(f"Metrics: {issue.metrics}")

    print(f"\n📊 Tested: {tested_count} ads, Skipped: {skipped_count} ads (insufficient data)")
    print(f"   Window size: 30 days (rolling)")
    print(f"   Consecutive days required: 3")
    print(f"✅ Total fatigue issues found: {len(all_issues)}")
    return all_issues


def test_latency_detector(data):
    """Test LatencyDetector."""
    print("\n" + "="*60)
    print("🔍 TESTING LATENCY DETECTOR")
    print("="*60)

    # Use lower thresholds for testing
    detector = LatencyDetector(config={
        "thresholds": {
            "roas_threshold": 0.8,  # Lower from 1.0 to 0.8
            "min_drop_ratio": 0.15,  # Lower from 0.2 to 0.15
            "rolling_window_days": 2,  # Lower from 3 to 2
        }
    })

    adset_daily = data['adset_daily']

    # Infer status changes
    if 'adset_status' in adset_daily.columns:
        status_changes = infer_status_changes(adset_daily)
        print(f"✅ Inferred {len(status_changes)} status changes")
    else:
        status_changes = None
        print("⚠️  No status column available")

    # Get unique adsets - test more adsets
    adset_ids = adset_daily['adset_id'].unique() if 'adset_id' in adset_daily.columns else []
    print(f"\n📊 Testing {min(20, len(adset_ids))} adsets (sample)...")

    all_issues = []
    tested_count = 0
    skipped_count = 0

    for i, adset_id in enumerate(adset_ids[:20]):
        adset_data = adset_daily[adset_daily['adset_id'] == adset_id] if 'adset_id' in adset_daily.columns else adset_daily.iloc[:100]

        if len(adset_data) < 10:
            skipped_count += 1
            continue

        tested_count += 1
        issues = detector.detect(adset_data, str(adset_id), status_changes)
        all_issues.extend(issues)

        if issues:
            for issue in issues:
                print(f"\n{'─'*60}")
                print(f"📢 AdSet ID: {adset_id}")
                print(f"Title: {issue.title}")
                print(f"Severity: {issue.severity.value}")
                print(f"Description: {issue.description}")
                print(f"Metrics: {issue.metrics}")

    print(f"\n📊 Tested: {tested_count} adsets, Skipped: {skipped_count} adsets (insufficient data)")
    print(f"✅ Total latency issues found: {len(all_issues)}")
    return all_issues


def test_dark_hours_detector(data):
    """Test DarkHoursDetector."""
    print("\n" + "="*60)
    print("🔍 TESTING DAY & HOUR PERFORMANCE DETECTOR")
    print("="*60)

    # Use lower thresholds for testing
    detector = DarkHoursDetector(config={
        "thresholds": {
            "min_days": 14,  # Lower from 21 to 14 days
            "cvr_threshold_ratio": 0.3,  # Increase (easier to detect) from 0.2 to 0.3
            "min_spend_ratio_hourly": 0.03,  # Lower from 0.05 to 0.03
            "min_spend_ratio_daily": 0.08,  # Lower from 0.10 to 0.08
        }
    })

    adset_daily = data['adset_daily']
    adset_hourly = data.get('adset_hourly')

    # Get unique adsets - test more adsets
    adset_ids = adset_daily['adset_id'].unique() if 'adset_id' in adset_daily.columns else []
    print(f"\n📊 Testing {min(20, len(adset_ids))} adsets (sample)...")

    all_issues = []
    tested_count = 0
    skipped_count = 0

    for i, adset_id in enumerate(adset_ids[:20]):
        adset_data = adset_daily[adset_daily['adset_id'] == adset_id] if 'adset_id' in adset_daily.columns else adset_daily.iloc[:100]

        if len(adset_data) < 14:
            skipped_count += 1
            continue

        tested_count += 1

        # Get hourly data for this adset
        hourly_data = None
        if adset_hourly is not None and 'adset_id' in adset_hourly.columns:
            hourly_data = adset_hourly[adset_hourly['adset_id'] == adset_id]
            if len(hourly_data) > 0:
                print(f"  📊 AdSet {adset_id}: {len(hourly_data)} hourly records")

        issues = detector.detect(adset_data, str(adset_id), hourly_data)
        all_issues.extend(issues)

        if issues:
            for issue in issues:
                print(f"\n{'─'*60}")
                print(f"📢 AdSet ID: {adset_id}")
                print(f"Title: {issue.title}")
                print(f"Severity: {issue.severity.value}")
                print(f"Description: {issue.description}")
                print(f"Metrics: {issue.metrics}")

    print(f"\n📊 Tested: {tested_count} adsets, Skipped: {skipped_count} adsets (insufficient data)")
    print(f"✅ Total performance issues found: {len(all_issues)}")
    return all_issues


def print_summary(fatigue_issues, latency_issues, performance_issues):
    """Print summary of all detections."""
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    print(f"\n🔴 Fatigue Issues: {len(fatigue_issues)}")
    if fatigue_issues:
        scores = [issue.metrics.get('severity_score', 0) for issue in fatigue_issues]
        print(f"   - Avg Severity Score: {np.mean(scores):.1f}/100")
        print(f"   - Score Range: {min(scores):.0f} - {max(scores):.0f}")

    print(f"\n🟡 Latency Issues: {len(latency_issues)}")
    if latency_issues:
        scores = [issue.metrics.get('avg_responsiveness_score', 0) for issue in latency_issues]
        delays = [issue.metrics.get('avg_delay_days', 0) for issue in latency_issues]
        print(f"   - Avg Responsiveness Score: {np.mean(scores):.1f}/100")
        print(f"   - Avg Delay: {np.mean(delays):.1f} days")

    print(f"\n🟢 Performance Issues: {len(performance_issues)}")
    if performance_issues:
        weekly_scores = [
            issue.metrics.get('efficiency_score', 0)
            for issue in performance_issues
            if issue.metrics.get('analysis_type') == 'weekly'
        ]
        hourly_scores = [
            issue.metrics.get('efficiency_score', 0)
            for issue in performance_issues
            if issue.metrics.get('analysis_type') == 'hourly'
        ]
        if weekly_scores:
            print(f"   - Weekly Efficiency: {np.mean(weekly_scores):.1f}/100 ({len(weekly_scores)} analyses)")
        if hourly_scores:
            print(f"   - Hourly Efficiency: {np.mean(hourly_scores):.1f}/100 ({len(hourly_scores)} analyses)")

    total_issues = len(fatigue_issues) + len(latency_issues) + len(performance_issues)
    print(f"\n🎯 Total Issues Detected: {total_issues}")

    # Severity breakdown
    severity_counts = {}
    for issue in fatigue_issues + latency_issues + performance_issues:
        sev = issue.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    if severity_counts:
        print(f"\n📈 Severity Breakdown:")
        for sev, count in sorted(severity_counts.items()):
            print(f"   - {sev.upper()}: {count}")


if __name__ == "__main__":
    print("🚀 DevEase Diagnoser - Moprobo Data Test")
    print("="*60)

    # Load data
    data = load_and_preprocess_data()

    # Run tests
    fatigue_issues = test_fatigue_detector(data)
    latency_issues = test_latency_detector(data)
    performance_issues = test_dark_hours_detector(data)

    # Print summary
    print_summary(fatigue_issues, latency_issues, performance_issues)

    print("\n✅ Test Complete!")
