#!/usr/bin/env python3
"""
Debug FatigueDetector vs Label Generator mismatch.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Set logging level BEFORE importing the detector
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.meta.diagnoser.detectors import FatigueDetector
from src.meta.diagnoser.judge import ZeroCostLabelGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_moprobo_data():
    """Load and preprocess moprobo data."""
    import json

    daily_path = Path("datasets/moprobo/meta/raw/ad_daily_insights_2024-12-17_2025-12-17.csv")
    ad_daily = pd.read_csv(daily_path)

    numeric_cols = ['spend', 'impressions', 'reach', 'clicks', 'actions']
    for col in numeric_cols:
        if col in ad_daily.columns:
            ad_daily[col] = pd.to_numeric(ad_daily[col], errors='coerce').fillna(0)

    if 'purchase_roas' in ad_daily.columns:
        def extract_roas_value(roas_str):
            if pd.isna(roas_str) or roas_str == '':
                return 0.0
            try:
                data = json.loads(roas_str)
                if isinstance(data, list) and len(data) > 0:
                    return float(data[0].get('value', 0))
                return 0.0
            except:
                return 0.0

        ad_daily['purchase_roas'] = ad_daily['purchase_roas'].apply(extract_roas_value)

    if 'date_start' in ad_daily.columns:
        ad_daily['date'] = pd.to_datetime(ad_daily['date_start'], errors='coerce')
        ad_daily = ad_daily.sort_values('date').dropna(subset=['date'])

    return ad_daily


def main():
    """Debug FatigueDetector vs Label Generator."""
    logger.info("=" * 80)
    logger.info("Debug: FatigueDetector vs Label Generator")
    logger.info("=" * 80)

    # 1. Load data
    daily_data = load_moprobo_data()
    logger.info(f"Loaded {len(daily_data)} rows")

    # 2. Take first window
    min_date = daily_data['date'].min()
    window_start = min_date
    window_end = window_start + timedelta(days=29)

    window_data = daily_data[
        (daily_data['date'] >= window_start) &
        (daily_data['date'] <= window_end)
    ].copy()

    logger.info(f"Window: {window_start.date()} to {window_end.date()}")
    logger.info(f"Window rows: {len(window_data)}")

    # 3. Find first entity with enough data
    entity_col = "ad_id"
    entities_with_data = []

    for entity_id, entity_data in window_data.groupby(entity_col):
        if len(entity_data) >= 23:  # Need 21 + 2 days
            entities_with_data.append((entity_id, entity_data))
            if len(entities_with_data) >= 3:
                break

    logger.info(f"\nFound {len(entities_with_data)} entities with enough data")

    # 4. Test first entity
    for i, (entity_id, entity_data) in enumerate(entities_with_data[:2]):
        logger.info("\n" + "=" * 80)
        logger.info(f"Testing Entity {i+1}: {entity_id}")
        logger.info("=" * 80)
        logger.info(f"Data points: {len(entity_data)}")
        logger.info(f"Date range: {entity_data['date'].min()} to {entity_data['date'].max()}")

        # Check required columns
        logger.info(f"\nColumns: {entity_data.columns.tolist()}")
        logger.info(f"Has conversions: {'conversions' in entity_data.columns}")
        logger.info(f"Has actions: {'actions' in entity_data.columns}")

        # Run detector (using default thresholds)
        detector = FatigueDetector()

        issues = detector.detect(entity_data.copy(), entity_id)
        logger.info(f"\nDetector found {len(issues)} issues")

        for issue in issues:
            logger.info(f"  Issue: {issue.title}")
            logger.info(f"  Metrics: {issue.metrics}")

        # Run label generator
        label_gen = ZeroCostLabelGenerator()
        labels = label_gen._apply_fatigue_rules(entity_data, entity_id)
        logger.info(f"\nLabel generator found {len(labels)} labels")

        for label in labels:
            logger.info(f"  Label: {label.get('issue_type')}")
            logger.info(f"  Date: {label.get('date')}")
            logger.info(f"  Severity: {label.get('severity')}")
            logger.info(f"  Metrics: {label.get('metrics')}")

        # Check conversions
        if 'conversions' in entity_data.columns:
            total_conversions = entity_data['conversions'].sum()
            logger.info(f"\nTotal conversions: {total_conversions}")
            logger.info(f"Daily conversions:\n{entity_data['conversions'].tail(10).tolist()}")

        # Check if data has impressions and reach for frequency calculation
        if 'impressions' in entity_data.columns and 'reach' in entity_data.columns:
            logger.info(f"\nLast 10 days impressions: {entity_data['impressions'].tail(10).tolist()}")
            logger.info(f"Last 10 days reach: {entity_data['reach'].tail(10).tolist()}")

    return 0


if __name__ == "__main__":
    exit(main())
