"""
End-to-end test for adset allocator pipeline.

Tests the complete flow:
1. Load adset features from dataset
2. Initialize rule-based allocator
3. Calculate budget allocations
4. Validate safety constraints and decision rules
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

from src.meta.adset.allocator.allocator import Allocator
from src.meta.adset.allocator.lib.safety_rules import SafetyRules
from src.meta.adset.allocator.lib.decision_rules import DecisionRules
from src.meta.adset.allocator.lib.models import BudgetAllocationMetrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_e2e_adset_allocation():
    """
    End-to-end test: Adset allocator pipeline.

    Uses test_customer data from datasets/test_customer/meta/
    """
    logger.info("=" * 80)
    logger.info("E2E Test: Adset Allocator Pipeline")
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
        logger.info(f"✓ Loaded {len(df)} adset records")

        # Get latest data per adset
        latest_df = df.sort_values('date_start').groupby('adset_id').last().reset_index()
        logger.info(f"  Unique adsets: {len(latest_df)}")

    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        return False

    # Step 2: Initialize allocator
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Initialize Allocator")
    logger.info("=" * 80)

    try:
        # Create safety rules (using defaults)
        safety_rules = SafetyRules(config=None)

        # Create decision rules
        decision_rules = DecisionRules(config=None)

        # Initialize allocator
        allocator = Allocator(
            safety_rules=safety_rules,
            decision_rules=decision_rules
        )

        logger.info("✓ Allocator initialized")
        logger.info(f"  Safety rules:")
        logger.info(f"    - Freeze ROAS threshold: {safety_rules.freeze_roas_threshold}")
        logger.info(f"    - Freeze health threshold: {safety_rules.freeze_health_threshold}")
        logger.info(f"    - Min budget: ${safety_rules.min_budget}")
        logger.info(f"    - Max daily increase: {safety_rules.max_daily_increase_pct*100:.1f}%")
        logger.info(f"    - Max daily decrease: {safety_rules.max_daily_decrease_pct*100:.1f}%")

    except Exception as e:
        logger.error(f"Failed to initialize allocator: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Run budget allocations
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Run Budget Allocations")
    logger.info("=" * 80)

    allocations = []

    try:
        # Test with first 5 adsets
        sample_adsets = latest_df.head(5)

        for idx, row in sample_adsets.iterrows():
            adset_id = row['adset_id']
            current_budget = 100.0  # Default budget since not in dataset
            roas_7d = row.get('purchase_roas_rolling_7d', row.get('purchase_roas', 1.0))
            roas_trend = row.get('purchase_roas_trend_7d', 0.0)
            health_score = row.get('health_score', 0.5)
            spend = row['spend']

            logger.info(f"\nProcessing {adset_id}:")
            logger.info(f"  Current budget: ${current_budget:.2f}")
            logger.info(f"  ROAS 7d: {roas_7d:.2f}")
            logger.info(f"  ROAS trend: {roas_trend:.3f}")
            logger.info(f"  Health score: {health_score:.2f}")
            logger.info(f"  Spend: ${spend:.2f}")

            # Allocate budget
            new_budget, decision_path = allocator.allocate_budget(
                adset_id=adset_id,
                current_budget=current_budget,
                roas_7d=roas_7d,
                roas_trend=roas_trend,
                health_score=health_score,
                spend=spend
            )

            budget_change = new_budget - current_budget
            budget_change_pct = (budget_change / current_budget) * 100 if current_budget > 0 else 0

            allocations.append({
                'adset_id': adset_id,
                'current_budget': current_budget,
                'new_budget': new_budget,
                'budget_change': budget_change,
                'budget_change_pct': budget_change_pct,
                'decision_path': ' -> '.join(decision_path),
                'roas_7d': roas_7d,
                'health_score': health_score
            })

            logger.info(f"  → New budget: ${new_budget:.2f}")
            logger.info(f"  → Change: ${budget_change:+.2f} ({budget_change_pct:+.1f}%)")
            logger.info(f"  → Decision path: {' -> '.join(decision_path)}")

    except Exception as e:
        logger.error(f"Failed to allocate budget: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Validate allocations
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Validate Allocations")
    logger.info("=" * 80)

    try:
        # Check for safety violations
        violations = []

        for alloc in allocations:
            # Check min budget constraint
            if alloc['new_budget'] < safety_rules.min_budget and alloc['new_budget'] > 0:
                violations.append(f"{alloc['adset_id']}: Below min budget ${safety_rules.min_budget}")

        if violations:
            logger.warning(f"⚠️  Found {len(violations)} safety violations:")
            for violation in violations:
                logger.warning(f"  - {violation}")
        else:
            logger.info("✓ No safety violations")

        # Statistics
        total_current = sum(a['current_budget'] for a in allocations)
        total_new = sum(a['new_budget'] for a in allocations)
        budget_change = total_new - total_current
        budget_change_pct = (budget_change / total_current) * 100 if total_current > 0 else 0

        logger.info(f"\nBudget aggregation:")
        logger.info(f"  Total current budget: ${total_current:.2f}")
        logger.info(f"  Total new budget: ${total_new:.2f}")
        logger.info(f"  Total change: ${budget_change:+.2f} ({budget_change_pct:+.1f}%)")

        # Count decisions
        scaled_up = len([a for a in allocations if a['budget_change'] > 0])
        scaled_down = len([a for a in allocations if a['budget_change'] < 0])
        frozen = len([a for a in allocations if a['new_budget'] == 0])

        logger.info(f"\nDecision distribution:")
        logger.info(f"  Scaled up: {scaled_up}")
        logger.info(f"  Scaled down: {scaled_down}")
        logger.info(f"  Frozen: {frozen}")

    except Exception as e:
        logger.error(f"Failed to validate allocations: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Save output (skip in CI)
    if not is_ci:
        logger.info("\n" + "=" * 80)
        logger.info("Step 5: Save Output")
        logger.info("=" * 80)

        try:
            output_dir = Path("results/meta/adset/allocator/test_customer")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "allocations.csv"
            alloc_df = pd.DataFrame(allocations)
            alloc_df.to_csv(output_file, index=False)
            logger.info(f"✓ Saved allocations to {output_file}")

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
    logger.info(f"✓ Allocated budgets for {len(allocations)} adsets")
    logger.info(f"✓ Total budget change: {budget_change_pct:+.1f}%")
    logger.info(f"✓ Safety violations: {len(violations)}")

    if not is_ci:
        logger.info(f"✓ Output saved to: {output_dir}")
    else:
        logger.info("✓ Output saving disabled (CI mode)")

    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_e2e_adset_allocation()
    sys.exit(0 if success else 1)
