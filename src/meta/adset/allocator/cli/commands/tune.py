#!/usr/bin/env python3
"""
Tuning Script for Rules-Based Budget Allocation

This script provides Bayesian optimization for rule-based allocation parameters.

Usage:
    # Rule parameter tuning
    python3 tune.py rules --customer moprobo
    python3 tune.py rules --customer moprobo --iterations 100

    # Legacy mode (tunes all customers)
    python3 tune.py --iterations 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
from src.config.manager import get_path_manager
from src.utils.logger_config import setup_logging

logger = setup_logging()


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Unified tuning and diagnostics for budget allocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rule parameter tuning
  python3 tune.py rules --customer moprobo
  python3 tune.py rules --customer moprobo --iterations 100
  python3 tune.py rules --no-update

  # Legacy (tunes rules, same as 'tune.py rules')
  python3 tune.py --customer moprobo
  python3 tune.py --iterations 50
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Command",
        required=False,  # For backward compatibility with legacy mode
        metavar="COMMAND",
    )

    # Rules subcommand
    _add_rules_subcommand(subparsers)

    # Legacy arguments (when no subcommand is provided)
    parser.add_argument(
        "--customer",
        type=str,
        default=None,
        help="Customer name (legacy mode - tunes rules)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of optimization iterations (legacy mode)",
    )
    parser.add_argument(
        "--initial-points",
        type=int,
        default=10,
        help="Number of initial random evaluations (legacy mode)",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't update config (legacy mode)",
    )

    args = parser.parse_args()

    # Handle legacy mode (no subcommand)
    if args.command is None:
        return _cmd_rules_legacy(args)

    # Execute subcommand
    try:
        return args.func(args)
    except Exception as err:
        logger.exception("Command failed: %s", err)
        return 1


def _add_rules_subcommand(subparsers):
    """Add rules tuning subcommand."""
    parser = subparsers.add_parser(
        "rules",
        help="Tune rule parameters using Bayesian optimization (CV by default)",
        description="""
Tune rule-based allocation parameters using Bayesian optimization.

DEFAULT BEHAVIOR:
- Uses 5-fold time-series cross-validation (prevents overfitting)
- Evaluates with forward-looking ROAS (next period performance)
- Takes ~5x longer than non-CV mode but provides realistic estimates

Optimizes thresholds for:
- Low ROAS adjustments
- High/low efficiency adjustments
- Health score thresholds
- Budget utilization targets
- Trend scaling factors
- And 50+ more parameters...

Use --no-cv for faster exploration (not recommended for production tuning).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--customer",
        type=str,
        default=None,
        help="Tune specific customer only (default: tune all customers)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of Bayesian optimization iterations (default: 50)",
    )

    parser.add_argument(
        "--initial-points",
        type=int,
        default=10,
        help="Number of initial random evaluations (default: 10)",
    )

    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't auto-update config (only report results)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/adset/allocator/rules.yaml",
        help="Path to rules.yaml configuration file",
    )

    parser.set_defaults(func=_cmd_rules)


# ============================================================================
# COMMAND HANDLERS
# ============================================================================


def _cmd_rules_legacy(args):
    """Handle legacy mode (no subcommand)."""
    logger.info("=" * 70)
    logger.info("LEGACY MODE: Tuning Rule Parameters")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Note: Using 'tune.py rules' is recommended for clarity")
    logger.info("")

    config_path = "config/adset/allocator/rules.yaml"

    if args.customer:
        _tune_single_customer(
            customer_name=args.customer,
            config_path=config_path,
            n_calls=args.iterations,
            n_initial_points=args.initial_points,
            update_config=not args.no_update,
        )
    else:
        _tune_all_customers(
            config_path=config_path,
            n_calls=args.iterations,
            n_initial_points=args.initial_points,
            update_config=not args.no_update,
            generate_report=True,
        )

    return 0


def _cmd_rules(args):
    """Handle rules tuning command."""
    logger.info("=" * 70)
    logger.info("RULE PARAMETER TUNING")
    logger.info("=" * 70)

    # CV is enabled by default (use_cv=True unless --no-cv flag)
    use_cv = not args.no_cv

    if args.customer:
        _tune_single_customer(
            customer_name=args.customer,
            config_path=args.config,
            n_calls=args.iterations,
            n_initial_points=args.initial_points,
            update_config=not args.no_update,
            use_cv=use_cv,
            n_folds=args.n_folds,
            train_ratio=args.train_ratio,
        )
    else:
        _tune_all_customers(
            config_path=args.config,
            n_calls=args.iterations,
            n_initial_points=args.initial_points,
            update_config=not args.no_update,
            generate_report=True,
            use_cv=use_cv,
            n_folds=args.n_folds,
            train_ratio=args.train_ratio,
        )

    return 0


def _tune_single_customer(
    customer_name: str,
    config_path: str = "config/adset/allocator/rules.yaml",
    n_calls: int = 50,
    n_initial_points: int = 10,
    update_config: bool = True,
    use_cv: bool = True,
    n_folds: int = 5,
    train_ratio: float = 0.7,
) -> None:
    """Tune parameters for a single customer using Bayesian optimization.

    Args:
        customer_name: Customer to tune.
        config_path: Path to config file.
        n_calls: Number of optimization iterations.
        n_initial_points: Number of initial random evaluations.
        update_config: Whether to update config file.
        use_cv: If True, use time-series cross-validation (reduces
               overfitting). Default: True.
        n_folds: Number of CV folds. Default: 5.
        train_ratio: Training ratio for CV folds.
    """
    from src.meta.adset.allocator.optimizer.lib.bayesian_tuner import BayesianTuner
    from src.meta.adset.allocator.optimizer.tuning import TuningConstraints

    print("=" * 80)
    if use_cv:
        print(
            f"AUTOMATED BAYESIAN TUNING WITH {n_folds}-FOLD CV FOR: {customer_name.upper()}"
        )
    else:
        print(f"AUTOMATED BAYESIAN TUNING FOR: {customer_name.upper()}")
        print("[WARN]  WARNING: CV disabled - may overfit to historical data")
    print("=" * 80)
    print()

    if use_cv:
        print("[SCIENCE] Using time-series cross-validation (default)")
        print(f"   - {n_folds}-fold CV with forward-looking ROAS evaluation")
        print("   - Reduces overfitting to historical patterns")
        print("   - Provides realistic production performance estimates")
        print(f"   - Estimated time: ~{n_folds}x longer than non-CV mode")
        print()
    else:
        print("[WARN]  Cross-validation DISABLED (--no-cv flag)")
        print("   - Uses historical ROAS for evaluation (faster but may overfit)")
        print("   - Recommended for exploration only")
        print("   - Use CV mode for production tuning")
        print()

    # Define constraints
    constraints = TuningConstraints(
        min_budget_change_rate=0.10,
        max_budget_change_rate=0.50,
        min_total_budget_utilization=0.85,
        max_total_budget_utilization=1.05,
        min_avg_roas=1.5,
        min_revenue_efficiency=0.8,
    )

    # Initialize tuner
    tuner = BayesianTuner(
        config_path=config_path,
        constraints=constraints,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
    )

    # Tune customer (with or without CV)
    if use_cv:
        result = tuner.tune_customer_with_cv(
            customer_name,
            n_folds=n_folds,
            train_ratio=train_ratio,
        )
    else:
        result = tuner.tune_customer(customer_name)

    if result is None:
        print(f"\n[WARN]  Failed to tune {customer_name}")
        return

    # Update config
    if update_config:
        print("\n" + "=" * 80)
        print("UPDATING CONFIG")
        print("=" * 80)
        tuner.update_config_with_results({customer_name: result})

    # Show results
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)
    for key, value in result.param_config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)
    print(f"  Weighted Avg ROAS: {result.weighted_avg_roas:.4f}")
    print(f"  Budget Utilization: {result.budget_utilization:.2%}")
    print(f"  Change Rate: {result.change_rate:.2%}")
    print(f"  Revenue Efficiency: {result.revenue_efficiency:.4f}")
    print(f"  Total Revenue: ${result.total_revenue:,.2f}")
    print(f"  Budget Gini: {result.budget_gini:.3f}")
    print(f"  Budget Entropy: {result.budget_entropy:.3f}")


def _tune_all_customers(
    config_path: str = "config/adset/allocator/rules.yaml",
    n_calls: int = 50,
    n_initial_points: int = 10,
    update_config: bool = True,
    generate_report: bool = True,
    use_cv: bool = False,
    n_folds: int = 3,
    train_ratio: float = 0.7,
) -> None:
    """Tune parameters for all customers using Bayesian optimization.

    Note: CV mode is not yet supported for batch tuning. Use --customer flag
    with --use-cv to tune individual customers with cross-validation.
    """
    from src.meta.adset.allocator.optimizer.lib.bayesian_tuner import BayesianTuner
    from src.meta.adset.allocator.optimizer.tuning import TuningConstraints

    if use_cv:
        logger.info("=" * 80)
        logger.info("AUTOMATED BAYESIAN TUNING FOR ALL CUSTOMERS (with CV)")
        logger.info("=" * 80)
        logger.info("")
        logger.info("[WARN]  CV mode is not yet supported for batch tuning.")
        logger.info("   Please tune customers individually:")
        logger.info("   python3 -m src.tools.tune rules " "--customer <name> --use-cv")
        logger.info("")
        return

    print("=" * 80)
    print("AUTOMATED BAYESIAN TUNING FOR ALL CUSTOMERS")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Discover all customers in datasets/")
    print("  2. Load their feature data")
    print("  3. Run Bayesian optimization to find optimal parameters")
    print("  4. Auto-update config/adset/allocator/rules.yaml with best parameters")
    print()

    # Define constraints
    constraints = TuningConstraints(
        min_budget_change_rate=0.10,
        max_budget_change_rate=0.50,
        min_total_budget_utilization=0.85,
        max_total_budget_utilization=1.05,
        min_avg_roas=1.5,
        min_revenue_efficiency=0.8,
    )

    # Initialize tuner
    tuner = BayesianTuner(
        config_path=config_path,
        constraints=constraints,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
    )

    # Tune all customers
    results = tuner.tune_all_customers(datasets_dir="datasets")

    if not results:
        print("\n[WARN]  No customers were tuned successfully")
        return

    # Update config
    if update_config:
        print("\n" + "=" * 80)
        print("UPDATING CONFIG")
        print("=" * 80)
        tuner.update_config_with_results(results)

    # Generate report
    if generate_report:
        print("\n" + "=" * 80)
        print("TUNING REPORT")
        print("=" * 80)
        tuner.generate_report(results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully tuned {len(results)} customer(s)")
    for customer_name, result in results.items():
        print(f"  {customer_name}:")
        print(f"    Weighted ROAS: {result.weighted_avg_roas:.4f}")
        print(f"    Budget Util: {result.budget_utilization:.2%}")
