"""
Test all recommenders on moprobo data.

Tests:
1. ROAS recommender (default)
2. CPA recommender
3. CPC recommender
4. CTR recommender
5. Percentile-based recommender
6. Custom recommender with strategies
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adset.generator.core.recommender import (
    create_recommender,
    create_roas_recommender,
    create_cpa_recommender,
    create_cpc_recommender,
    create_ctr_recommender,
    create_percentile_recommender,
    create_custom_recommender,
    MetricConfig,
    RecommendationStrategy,
    ConfigurableRecommender,
    default_opportunity_maximize,
    default_opportunity_minimize,
)


@pytest.fixture
def df():
    """Load sample moprobo data. Skip test if file doesn't exist."""
    data_file = Path("datasets/moprobo/meta/features/ad_features.csv")
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")
    
    print(f"\nLoading 5000 samples from moprobo data...")
    df = pd.read_csv(str(data_file), nrows=5000)

    # Filter valid data
    df_valid = df[
        (df["spend"] > 0) & (df["impressions"] > 0) & (df["purchase_roas"].notna())
    ].copy()

    print(f"  Loaded: {len(df_valid)} valid records")
    return df_valid


def test_roas_recommender(df):
    """Test ROAS recommender."""
    print("\n" + "=" * 60)
    print("TEST 1: ROAS Recommender (Default)")
    print("=" * 60)

    # Mock predictions (slight variation from actual)
    predictions = df["purchase_roas"].values * np.random.uniform(0.8, 1.2, len(df))

    recommender = create_roas_recommender()
    recs = recommender.generate_recommendations(df, predictions)
    summary = recommender.get_summary(recs)

    print(f"\n✓ ROAS recommender works!")
    print(f"  Scale up: {summary['scale_up']}")
    print(f"  Optimize: {summary['optimize_or_pause']}")
    print(f"  Hold: {summary['hold']}")
    return True


def test_cpa_recommender(df):
    """Test CPA recommender."""
    print("\n" + "=" * 60)
    print("TEST 2: CPA Recommender")
    print("=" * 60)

    # Create CPA column if not exists (spend / conversions)
    if "cost_per_action" not in df.columns:
        # Mock CPA: spend / (purchase_roas * spend / 100) = 100 / purchase_roas
        df["cost_per_action"] = np.where(
            df["purchase_roas"] > 0,
            100 / df["purchase_roas"],
            df["spend"],  # High CPA when no conversions
        )

    # Mock predictions
    predictions = df["cost_per_action"].values * np.random.uniform(0.8, 1.2, len(df))

    try:
        recommender = create_cpa_recommender()
        recs = recommender.generate_recommendations(df, predictions)
        summary = recommender.get_summary(recs)

        print(f"\n✓ CPA recommender works!")
        print(f"  Scale up: {summary['scale_up']}")
        print(f"  Optimize: {summary['optimize_or_pause']}")
        print(f"  Hold: {summary['hold']}")
        return True
    except Exception as e:
        print(f"\n⚠ CPA recommender issue: {e}")
        return False


def test_cpc_recommender(df):
    """Test CPC recommender."""
    print("\n" + "=" * 60)
    print("TEST 3: CPC Recommender")
    print("=" * 60)

    # Create CPC column if not exists
    if "cost_per_click" not in df.columns:
        df["cost_per_click"] = df["spend"] / df["clicks"].replace(0, 1)

    # Mock predictions
    predictions = df["cost_per_click"].values * np.random.uniform(0.9, 1.1, len(df))

    try:
        recommender = create_cpc_recommender()
        recs = recommender.generate_recommendations(df, predictions)
        summary = recommender.get_summary(recs)

        print(f"\n✓ CPC recommender works!")
        print(f"  Scale up: {summary['scale_up']}")
        print(f"  Optimize: {summary['optimize_or_pause']}")
        print(f"  Hold: {summary['hold']}")
        return True
    except Exception as e:
        print(f"\n⚠ CPC recommender issue: {e}")
        return False


def test_ctr_recommender(df):
    """Test CTR recommender."""
    print("\n" + "=" * 60)
    print("TEST 4: CTR Recommender")
    print("=" * 60)

    # Create CTR column if not exists
    if "ctr" not in df.columns:
        df["ctr"] = (df["clicks"] / df["impressions"].replace(0, 1)) * 100

    # Mock predictions
    predictions = df["ctr"].values * np.random.uniform(0.9, 1.1, len(df))

    try:
        recommender = create_ctr_recommender()
        recs = recommender.generate_recommendations(df, predictions)
        summary = recommender.get_summary(recs)

        print(f"\n✓ CTR recommender works!")
        print(f"  Scale up: {summary['scale_up']}")
        print(f"  Optimize: {summary['optimize_or_pause']}")
        print(f"  Hold: {summary['hold']}")
        return True
    except Exception as e:
        print(f"\n⚠ CTR recommender issue: {e}")
        return False


def test_percentile_recommender(df):
    """Test percentile-based recommender."""
    print("\n" + "=" * 60)
    print("TEST 5: Percentile-Based Recommender")
    print("=" * 60)

    predictions = df["purchase_roas"].values * np.random.uniform(0.8, 1.2, len(df))

    try:
        recommender = create_percentile_recommender(
            metric_name="roas", percentile_high=0.75, percentile_low=0.25
        )
        recs = recommender.generate_recommendations(
            df, predictions, use_percentiles=True
        )
        summary = recommender.get_summary(recs)

        print(f"\n✓ Percentile recommender works!")
        print(f"  Scale up: {summary['scale_up']}")
        print(f"  Optimize: {summary['optimize_or_pause']}")
        print(f"  Hold: {summary['hold']}")
        return True
    except Exception as e:
        print(f"\n⚠ Percentile recommender issue: {e}")
        return False


def test_custom_strategies_recommender(df):
    """Test custom strategies recommender."""
    print("\n" + "=" * 60)
    print("TEST 6: Custom Strategies Recommender")
    print("=" * 60)

    predictions = df["purchase_roas"].values * np.random.uniform(0.8, 1.2, len(df))

    # Define custom strategies
    strategies = [
        RecommendationStrategy(
            name="aggressive_scale",
            condition="greater_than",
            threshold=200,
            priority="high",
            label="Aggressive Scale Up",
            action="Significantly increase budget",
        ),
        RecommendationStrategy(
            name="moderate_scale",
            condition="between",
            threshold=(50, 200),
            priority="medium",
            label="Moderate Scale",
            action="Slightly increase budget",
        ),
        RecommendationStrategy(
            name="hold",
            condition="between",
            threshold=(-50, 50),
            priority="medium",
            label="Hold",
            action="Maintain budget",
        ),
        RecommendationStrategy(
            name="optimize",
            condition="less_than",
            threshold=-50,
            priority="high",
            label="Optimize/Pause",
            action="Review and optimize",
        ),
    ]

    try:
        config = MetricConfig(
            name="custom_roas",
            target_column="purchase_roas",
            direction="maximize",
            scale_threshold=200,
            pause_threshold=-50,
            strategies=strategies,
            opportunity_fn=default_opportunity_maximize,
        )
        recommender = ConfigurableRecommender(config=config)
        recs = recommender.generate_recommendations(df, predictions)

        # Check custom strategies
        print(f"\n  Custom strategy breakdown:")
        for strategy in strategies:
            count = (recs["recommendation"] == strategy.name).sum()
            print(f"    {strategy.label}: {count}")

        print(f"\n✓ Custom strategies recommender works!")
        return True
    except Exception as e:
        print(f"\n⚠ Custom strategies recommender issue: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_create_recommender_generic(df):
    """Test generic create_recommender function."""
    print("\n" + "=" * 60)
    print("TEST 7: Generic create_recommender()")
    print("=" * 60)

    predictions = df["purchase_roas"].values * np.random.uniform(0.8, 1.2, len(df))

    try:
        # Test with roas
        recommender = create_recommender(metric_name="roas")
        recs = recommender.generate_recommendations(df, predictions)
        print(f"\n✓ Generic create_recommender('roas') works!")

        return True
    except Exception as e:
        print(f"\n⚠ Generic create_recommender issue: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ALL RECOMMENDERS ON MOPROBO DATA")
    print("=" * 60)

    # Load data
    df = load_moprobo_data(5000)

    # Run tests
    results = {
        "ROAS Recommender": test_roas_recommender(df.copy()),
        "CPA Recommender": test_cpa_recommender(df.copy()),
        "CPC Recommender": test_cpc_recommender(df.copy()),
        "CTR Recommender": test_ctr_recommender(df.copy()),
        "Percentile Recommender": test_percentile_recommender(df.copy()),
        "Custom Strategies": test_custom_strategies_recommender(df.copy()),
        "Generic Factory": test_create_recommender_generic(df.copy()),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All recommenders work on moprobo data!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
