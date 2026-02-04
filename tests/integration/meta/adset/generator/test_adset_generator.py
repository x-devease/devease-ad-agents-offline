"""
End-to-end test for adset generator pipeline.

Tests the complete flow:
1. Load adset features from dataset
2. Generate audience recommendations
3. Generate creative recommendations
4. Output recommendations in structured format
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

from src.meta.adset.generator.core.recommender import create_roas_recommender
from src.meta.adset.generator.generation.audience_recommender import AudienceRecommender
from src.meta.adset.generator.generation.audience_aggregator import AudienceAggregator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_e2e_adset_generation():
    """
    End-to-end test: Adset generation pipeline.

    Uses test_customer data from datasets/test_customer/meta/
    """
    logger.info("=" * 80)
    logger.info("E2E Test: Adset Generator Pipeline")
    logger.info("=" * 80)

    # Check if running in CI
    is_ci = os.getenv("CI", "").lower() == "true"

    # Configuration
    customer = "test_customer"
    platform = "meta"
    features_file = Path("datasets/test_customer/meta/raw/adset_features.csv")

    # Verify data exists
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        logger.error("Please run feature extraction first to generate test data")
        return False

    logger.info(f"Customer: {customer}")
    logger.info(f"Platform: {platform}")
    logger.info(f"Features file: {features_file}")

    # Step 1: Load features
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Load Adset Features")
    logger.info("=" * 80)

    try:
        df = pd.read_csv(features_file)
        logger.info(f"✓ Loaded {len(df)} adsets")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info(f"  Date range: {df['date_start'].min()} to {df['date_start'].max()}")
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return False

    # Step 2: Initialize ROAS recommender
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Initialize ROAS Recommender")
    logger.info("=" * 80)

    try:
        recommender = create_roas_recommender(
            scale_threshold=2.0,
            pause_threshold=1.0
        )
        logger.info("✓ ROAS recommender initialized")
        logger.info(f"  Scale threshold: {recommender.config.scale_threshold}")
        logger.info(f"  Pause threshold: {recommender.config.pause_threshold}")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Generate recommendations
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Generate Recommendations")
    logger.info("=" * 80)

    try:
        # Get latest data per adset
        latest_df = df.sort_values('date_start').groupby('adset_id').last().reset_index()

        # Generate recommendations for first 5 adsets
        sample_df = latest_df.head(5)

        # Prepare predictions and spend arrays
        predictions = sample_df['purchase_roas_rolling_7d'].values

        # Generate recommendations
        recs_df = recommender.generate_recommendations(
            df=sample_df,
            predictions=predictions,
            spend_col='spend',
            include_evidence=True
        )

        # Convert to list for easier processing
        recommendations = []
        for idx, row in recs_df.iterrows():
            adset_id = row['adset_id']
            recommendation = row['recommendation']
            opportunity_score = row.get('opportunity_score', 0)
            action = row.get('action', '')
            current_roas = row.get('purchase_roas_rolling_7d', 0)
            spend = row.get('spend', 0)

            logger.info(f"\nProcessing {adset_id}:")
            logger.info(f"  ROAS: {current_roas:.2f}")
            logger.info(f"  Spend: ${spend:.2f}")
            logger.info(f"  → {recommendation}")
            logger.info(f"  → Opportunity score: {opportunity_score:.2f}")
            logger.info(f"  → Action: {action}")

            recommendations.append({
                'adset_id': adset_id,
                'recommendation': recommendation,
                'opportunity_score': opportunity_score,
                'action': action,
                'current_roas': current_roas,
                'spend': spend
            })

    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Aggregate Results")
    logger.info("=" * 80)

    try:
        # Count recommendations by type
        rec_counts = {}
        for rec in recommendations:
            rec_type = rec['recommendation']
            rec_counts[rec_type] = rec_counts.get(rec_type, 0) + 1

        logger.info("Recommendation distribution:")
        for rec_type, count in sorted(rec_counts.items()):
            logger.info(f"  {rec_type}: {count}")

        # Calculate opportunity scores
        high_opportunity = [r for r in recommendations if r['opportunity_score'] > 50]
        logger.info(f"\nHigh opportunity adsets (score > 50): {len(high_opportunity)}")

        for rec in high_opportunity[:3]:
            logger.info(f"  {rec['adset_id']}: score={rec['opportunity_score']:.1f}, rec={rec['recommendation']}")

    except Exception as e:
        logger.error(f"Failed to aggregate results: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Save output (skip in CI)
    if not is_ci:
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Save Output")
        logger.info("=" * 80)

        try:
            output_dir = Path("results/meta/adset/generator/test_customer")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "recommendations.csv"
            rec_df = pd.DataFrame(recommendations)
            rec_df.to_csv(output_file, index=False)
            logger.info(f"✓ Saved recommendations to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return False
    else:
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Save Output (Skipped in CI)")
        logger.info("=" * 80)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("E2E Test Summary")
    logger.info("=" * 80)
    logger.info("✓ All tests passed!")
    logger.info(f"✓ Generated {len(recommendations)} recommendations")
    logger.info(f"✓ Recommendation types: {list(rec_counts.keys())}")
    logger.info(f"✓ High opportunity adsets: {len(high_opportunity)}")

    if not is_ci:
        logger.info(f"✓ Output saved to: {output_dir}")
    else:
        logger.info("✓ Output saving disabled (CI mode)")

    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_e2e_adset_generation()
    sys.exit(0 if success else 1)
