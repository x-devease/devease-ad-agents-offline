#!/usr/bin/env python3
"""
Debug script to understand detector vs label mismatch.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.detectors import LatencyDetector
from src.meta.diagnoser.judge import ZeroCostLabelGenerator


def load_moprobo_data():
    """加载moprobo数据"""
    daily_path = Path("datasets/moprobo/meta/raw/ad_daily_insights_2024-12-17_2025-12-17.csv")
    ad_daily = pd.read_csv(daily_path)

    numeric_cols = ['spend', 'impressions', 'reach', 'purchase_roas', 'clicks']
    for col in numeric_cols:
        if col in ad_daily.columns:
            ad_daily[col] = pd.to_numeric(ad_daily[col], errors='coerce').fillna(0)

    if 'date_start' in ad_daily.columns:
        ad_daily['date'] = pd.to_datetime(ad_daily['date_start'], errors='coerce')
        ad_daily = ad_daily.sort_values('date')
        ad_daily = ad_daily.dropna(subset=['date'])

    # Get last 30 days
    latest_date = ad_daily['date'].max()
    thirty_days_ago = latest_date - pd.Timedelta(days=30)
    daily_sample = ad_daily[ad_daily['date'] >= thirty_days_ago].copy()

    return daily_sample


def debug_latency_detection():
    """Debug LatencyDetector vs zero-cost labels"""
    print("=" * 60)
    print("Debugging Latency Detection")
    print("=" * 60)

    data = load_moprobo_data()
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Generate zero-cost latency labels
    generator = ZeroCostLabelGenerator()
    all_labels = generator.generate(data, method="rule_based")
    latency_labels = [l for l in all_labels if l['issue_type'] == 'latency']

    print(f"\nZero-cost latency labels: {len(latency_labels)}")

    if latency_labels:
        print("\nSample zero-cost labels:")
        for i, label in enumerate(latency_labels[:5], 1):
            entity = label['affected_entities'][0] if label['affected_entities'] else 'unknown'
            print(f"\n{i}. Entity: {entity}")
            print(f"   Date: {label['date']}")
            print(f"   Severity: {label['severity']}")
            print(f"   Metrics: {label['metrics']}")

    # Run LatencyDetector on sample entities
    detector = LatencyDetector(config={
        "thresholds": {
            "roas_threshold": 1.0,
            "rolling_window_days": 3,
            "min_daily_spend": 50,
            "min_drop_ratio": 0.2,
        }
    })

    # Get unique entities
    unique_entities = data['ad_id'].unique()
    print(f"\n\nTotal unique entities in data: {len(unique_entities)}")

    # Test on first few entities
    detections = []
    for entity_id in unique_entities[:50]:  # Test first 50 entities
        entity_data = data[data['ad_id'] == entity_id]
        try:
            result = detector.detect(entity_data, entity_id=entity_id)
            if result:
                detections.extend(result)
        except Exception as e:
            pass  # Skip entities with errors

    print(f"\n\nLatencyDetector detections (from 50 entities): {len(detections)}")

    if detections:
        print("\nSample detections:")
        for i, detection in enumerate(detections[:5], 1):
            print(f"\n{i}. {detection}")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if latency_labels:
        # Get unique entities from labels
        label_entities = set()
        for label in latency_labels:
            if label['affected_entities']:
                label_entities.update(label['affected_entities'])

        print(f"\nEntities with latency labels: {len(label_entities)}")
        print(f"Sample entities: {list(label_entities)[:5]}")

    if detections:
        # Get unique entities from detections
        detection_entities = set()
        for detection in detections:
            if 'affected_entities' in detection:
                detection_entities.update(detection['affected_entities'])

        print(f"\nEntities with detections: {len(detection_entities)}")
        print(f"Sample entities: {list(detection_entities)[:5]}")

    # Check specific entity
    if latency_labels and label_entities:
        sample_entity = list(label_entities)[0]
        print(f"\n\nDetailed analysis for entity: {sample_entity}")

        entity_data = data[data['ad_id'] == sample_entity].sort_values('date')
        print(f"Entity data points: {len(entity_data)}")

        # Find low ROAS periods
        if 'purchase_roas' in entity_data.columns:
            entity_data = entity_data.copy()
            entity_data['low_roas'] = entity_data['purchase_roas'] < 1.0

            low_roas_days = entity_data[entity_data['low_roas']]
            print(f"\nDays with ROAS < 1.0: {len(low_roas_days)}")

            if len(low_roas_days) > 0:
                print("\nSample low ROAS days:")
                for idx, row in low_roas_days.head(5).iterrows():
                    print(f"  {row['date'].strftime('%Y-%m-%d')}: ROAS={row['purchase_roas']:.2f}, Spend=${row['spend']:.2f}")


if __name__ == "__main__":
    debug_latency_detection()
