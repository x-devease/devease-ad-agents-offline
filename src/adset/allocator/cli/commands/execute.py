#!/usr/bin/env python3
"""
Run Rule-Based Budget Allocator

This script demonstrates how to use the rule-based budget allocation system
for Meta ads budget allocation based on 21 important features.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.adset import Allocator, DecisionRules, SafetyRules
from src.adset.allocator.utils.parser import Parser
from src.config.path_manager import get_path_manager, PathManager
from src.utils.customer_paths import get_all_customers
from src.utils.logger_config import setup_logging
from src.utils.script_helpers import (
    add_config_argument,
    add_customer_argument,
    add_input_argument,
    add_output_argument,
    add_platform_argument,
)

logger = setup_logging()

# Import budget tracking module
from src.adset.allocator.budget import MonthlyBudgetTracker, MonthlyBudgetState

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main function for rule-based budget allocation."""
    args = _parse_arguments()

    if args.customer == "all":
        return _run_all_customers(args)

    # Customer is required
    if not args.customer:
        logger.error("--customer parameter is required (or use --customer all)")
        return 1

    # Set default input/output paths based on customer
    # if not explicitly provided
    path_manager = get_path_manager(customer=args.customer, platform=args.platform)
    # Create directories if needed
    path_manager.features_dir(args.customer, args.platform).mkdir(
        parents=True, exist_ok=True
    )
    path_manager.results_dir(args.customer, args.platform).mkdir(
        parents=True, exist_ok=True
    )

    if args.input is None:
        args.input = str(
            path_manager.features_dir(args.customer, args.platform)
            / "adset_features.csv"
        )
    if args.output is None:
        args.output = str(
            path_manager.allocations_path(
                args.customer, args.platform, "adset_budget_allocations.csv"
            )
        )

    config = _load_config(args.config, args.customer, args.platform)
    if config is None:
        return 1

    allocator = _initialize_allocator(config)
    if args.test:
        _run_test_allocation(allocator, args.customer)
    else:
        return _run_full_allocation(allocator, args, args.customer)

    return 0


def _parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run rule-based budget allocation for Meta ads"
    )
    add_config_argument(parser)
    add_customer_argument(parser)
    add_platform_argument(parser)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test example instead of full allocation",
    )
    add_input_argument(
        parser,
        help_text=(
            "Input CSV file with adset features "
            "(default: datasets/{customer}/features/adset_features.csv "
            "or datasets/{customer}/adset_features.csv if no features/)"
        ),
    )
    add_output_argument(
        parser,
        help_text=(
            "Output CSV file for allocation results "
            "(default: results/{customer}/rules/"
            "adset_budget_allocations.csv)"
        ),
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Total monthly budget to allocate (overrides config)",
    )
    return parser.parse_args()


def _load_config(config_path, customer_name="moprobo", platform="meta"):
    """Load configuration from file"""
    # Use default customer/platform-specific path if config_path is None
    if config_path is None:
        config_path = f"config/adset/allocator/{customer_name}/{platform}/rules.yaml"
        logger.debug("Using default config path: %s", config_path)

    try:
        config = Parser(
            config_path=config_path,
            customer_name=customer_name,
            platform=platform,
        )
        logger.info(
            "Loaded configuration from %s (customer: %s, platform: %s)",
            config.config_path,
            config.customer_name,
            config.platform,
        )
        return config
    except FileNotFoundError as err:
        logger.error("%s", err)
        return None
    except ValueError as err:
        logger.error("%s", err)
        return None


def _initialize_allocator(config):
    """Initialize rules and allocator"""
    safety_rules = SafetyRules(config)
    decision_rules = DecisionRules(config)
    return Allocator(safety_rules, decision_rules, config)


def _handle_flat_config():
    """Handle flat config structure (backward compatibility)."""
    logger.error("Flat config structure is not supported anymore")
    logger.error("Please specify --customer <name> when running allocation")
    logger.error("Or reorganize your config to have customer-specific sections")
    return 1


def _process_customer_allocation(customer, args, prefix_output):
    """Process allocation for a single customer."""
    logger.info("")
    logger.info("-" * 70)
    logger.info("Processing customer: %s", customer)
    logger.info("-" * 70)

    config = _load_config(args.config, customer, args.platform)
    if config is None:
        logger.warning(
            "Skipping customer %s due to config error",
            customer,
        )
        return False

    allocator = _initialize_allocator(config)

    # Create customer-specific args
    customer_args = argparse.Namespace(**vars(args))
    customer_args.customer = customer

    # Set customer-specific input/output paths
    path_manager = get_path_manager(customer=customer, platform=args.platform)
    if args.input is None:
        customer_args.input = str(
            path_manager.features_dir(customer, args.platform) / "adset_features.csv"
        )
    if args.output is None:
        customer_args.output = str(
            path_manager.allocations_path(
                customer, args.platform, "adset_budget_allocations.csv"
            )
        )
    elif prefix_output:
        # If output was explicitly provided and multiple customers,
        # prefix it
        output_path = Path(args.output)
        customer_output = output_path.parent / f"{customer}_{output_path.name}"
        customer_args.output = str(customer_output)

    if args.test:
        _run_test_allocation(allocator, customer)
        return True
    result = _run_full_allocation(allocator, customer_args, customer)
    if result != 0:
        logger.warning("Customer %s allocation failed", customer)
        return False
    return True


def _run_all_customers(args):
    """Run allocation for all customers"""
    customers = get_all_customers(args.config)
    if not customers:
        return _handle_flat_config()

    logger.info("=" * 70)
    logger.info("Running allocation for all customers")
    logger.info("=" * 70)
    logger.info("Found %d customer(s): %s", len(customers), ", ".join(customers))

    # If only one customer, don't prefix output
    # (behave like single customer mode)
    prefix_output = len(customers) > 1
    has_failures = False

    for customer in customers:
        if not _process_customer_allocation(customer, args, prefix_output):
            has_failures = True

    logger.info("")
    logger.info("=" * 70)
    logger.info("All customers processed")
    logger.info("=" * 70)
    return 1 if has_failures else 0


def _run_test_allocation(allocator, customer_name="unknown"):
    """Run test allocation example"""
    logger.info("=" * 70)
    logger.info("TEST: Rule-Based Budget Allocation (Customer: %s)", customer_name)
    logger.info("=" * 70)

    example_metrics = {
        "adset_id": "test_adset_001",
        "current_budget": 50.0,
        "previous_budget": 45.0,
        "roas_7d": 3.2,
        "roas_trend": 0.12,
        "adset_roas": 2.5,
        "roas_vs_adset": 1.28,
        "roas_vs_campaign": 1.20,
        "efficiency": 0.12,
        "revenue_per_impression": 0.09,
        "revenue_per_click": 2.8,
        "spend": 80.0,
        "spend_rolling_7d": 70.0,
        "impressions": 6000,
        "clicks": 60,
        "reach": 1200,
        "health_score": 0.85,
        "days_active": 20,
        "is_weekend": False,
        "week_of_year": 45,
        "adaptive_target_roas": 2.2,
        "static_target_roas": 2.0,
        "budget_utilization": 0.92,
        "marginal_roas": 3.0,
    }

    logger.info("Example Adset Metrics:")
    logger.info("Adset ID: %s", example_metrics["adset_id"])
    logger.info("Current Budget: $%.2f", example_metrics["current_budget"])
    logger.info("ROAS 7d: %.2fx", example_metrics["roas_7d"])
    logger.info("Trend: %.1f%%", example_metrics["roas_trend"] * 100)
    logger.info("Health Score: %.2f", example_metrics["health_score"])
    logger.info("Days Active: %d", example_metrics["days_active"])

    new_budget, decision_path = allocator.allocate_budget(**example_metrics)

    logger.info("Allocation Result:")
    logger.info("New Budget: $%.2f", new_budget)
    change_pct = ((new_budget / example_metrics["current_budget"]) - 1) * 100
    logger.info("Change: %+.1f%%", change_pct)
    logger.info("Decision Path: %s", decision_path)

    logger.info("=" * 70)
    logger.info("SUCCESS: Test complete!")
    logger.info("=" * 70)


def _run_full_allocation(allocator, args, customer_name="unknown"):
    """Run full allocation from input file"""
    from datetime import datetime

    logger.info("=" * 70)
    logger.info("Rule-Based Budget Allocator (Customer: %s)", customer_name)
    logger.info("=" * 70)

    df = _load_data(args.input)
    if df is None:
        return 1

    if "adset_id" not in df.columns:
        logger.error("'adset_id' column not found in input data")
        return 1

    # Aggregate by adset: use mean for numeric columns, first for categorical
    # This ensures we get the average ROAS per adset instead of just the first row
    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(exclude=["number"]).columns

    agg_dict = {}
    for col in numeric_cols:
        if col != "adset_id":
            agg_dict[col] = "mean"
    for col in categorical_cols:
        if col != "adset_id":
            agg_dict[col] = "first"

    adset_groups = df.groupby("adset_id").agg(agg_dict)
    logger.info(
        "Processing %d adsets (aggregated from %d total rows)...",
        len(adset_groups),
        len(df),
    )

    # === Budget Tracking Integration ===
    config = allocator.config

    # Get monthly budget (required)
    monthly_budget = _get_monthly_budget(args, config, customer_name)

    # Load or create state (monthly tracking always enabled)
    state = MonthlyBudgetState.load_or_create(
        customer=customer_name,
        platform=args.platform,
        monthly_budget=monthly_budget,
    )
    tracker = MonthlyBudgetTracker(state)

    # Check if budget exhausted
    if tracker.is_budget_exhausted():
        logger.error("=" * 70)
        logger.error("MONTHLY BUDGET EXHAUSTED")
        logger.error("=" * 70)
        logger.error(f"Monthly Budget: ${monthly_budget:.2f}")
        logger.error(f"Total Spent: ${state.tracking['total_spent']:.2f}")
        logger.error("NO ALLOCATION PERFORMED")
        return 1

    # Calculate today's budget
    total_budget_today = tracker.calculate_daily_budget(datetime.now())

    logger.info(f"Monthly tracking active for {state.month}")
    logger.info(f"Spent: ${state.tracking['total_spent']:.2f} of ${monthly_budget:.2f}")
    logger.info(f"Today's budget: ${total_budget_today:.2f}")

    # Perform allocation
    results = _process_adsets(allocator, adset_groups, total_budget_today)

    results_df = _process_results(results, total_budget_today)
    _save_results(results_df, args.output)
    _print_summary(results_df)

    # Record allocation
    actual_spend = results_df["current_budget"].sum()

    # Archive allocation (always enabled)
    archive_path = _archive_allocation(
        args.output, datetime.now(), customer_name, args.platform
    )

    # Record execution
    tracker.record_allocation(
        execution_date=datetime.now(),
        allocated=total_budget_today,
        spent=actual_spend,
        num_adsets=len(results_df),
        allocation_file=archive_path,
    )

    # Save state
    state.save()
    logger.info(f"State saved to: {state.state_path}")

    return 0


def _load_data(input_file):
    """Load data from CSV file"""
    try:
        logger.info("Loading data from %s...", input_file)
        df = pd.read_csv(input_file)
        logger.info("Loaded %d rows", len(df))
        return df
    except FileNotFoundError:
        logger.error("Input file '%s' not found", input_file)
        return None
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
        logger.error("Error loading data: %s", err, exc_info=True)
        return None
    except (IOError, OSError) as err:
        logger.error("Unexpected error loading data: %s", err, exc_info=True)
        return None


def _extract_metrics_from_row(row, adset_id, current_budget, total_budget_today):
    """Extract metrics from DataFrame row"""
    return {
        "adset_id": str(adset_id),
        "current_budget": float(current_budget),
        "previous_budget": (
            float(row.get("previous_budget", 0))
            if pd.notna(row.get("previous_budget"))
            else None
        ),
        "roas_7d": (
            float(row.get("purchase_roas_rolling_7d", 0))
            if pd.notna(row.get("purchase_roas_rolling_7d"))
            else 0.0
        ),
        "roas_trend": (
            float(row.get("roas_trend", 0)) if pd.notna(row.get("roas_trend")) else 0.0
        ),
        "adset_roas": (
            float(row.get("adset_roas", 0)) if pd.notna(row.get("adset_roas")) else None
        ),
        "campaign_roas": (
            float(row.get("campaign_roas", 0))
            if pd.notna(row.get("campaign_roas"))
            else None
        ),
        "account_roas": (
            float(row.get("account_roas", 0))
            if pd.notna(row.get("account_roas"))
            else None
        ),
        "roas_vs_adset": (
            float(row.get("roas_vs_adset", 0))
            if pd.notna(row.get("roas_vs_adset"))
            else None
        ),
        "roas_vs_campaign": (
            float(row.get("roas_vs_campaign", 0))
            if pd.notna(row.get("roas_vs_campaign"))
            else None
        ),
        "roas_vs_account": (
            float(row.get("roas_vs_account", 0))
            if pd.notna(row.get("roas_vs_account"))
            else None
        ),
        "efficiency": (
            float(row.get("efficiency", 0)) if pd.notna(row.get("efficiency")) else None
        ),
        "revenue_per_impression": (
            float(row.get("revenue_per_impression", 0))
            if pd.notna(row.get("revenue_per_impression"))
            else None
        ),
        "revenue_per_click": (
            float(row.get("revenue_per_click", 0))
            if pd.notna(row.get("revenue_per_click"))
            else None
        ),
        "spend": (float(row.get("spend", 0)) if pd.notna(row.get("spend")) else None),
        "spend_rolling_7d": (
            float(row.get("spend_rolling_7d", 0))
            if pd.notna(row.get("spend_rolling_7d"))
            else None
        ),
        "impressions": (
            int(row.get("impressions", 0)) if pd.notna(row.get("impressions")) else None
        ),
        "clicks": (int(row.get("clicks", 0)) if pd.notna(row.get("clicks")) else None),
        "reach": (int(row.get("reach", 0)) if pd.notna(row.get("reach")) else None),
        "adset_spend": (
            float(row.get("adset_spend", 0))
            if pd.notna(row.get("adset_spend"))
            else None
        ),
        "campaign_spend": (
            float(row.get("campaign_spend", 0))
            if pd.notna(row.get("campaign_spend"))
            else None
        ),
        "expected_clicks": (
            float(row.get("expected_clicks", 0))
            if pd.notna(row.get("expected_clicks"))
            else None
        ),
        "health_score": (
            float(row.get("health_score", 0.5))
            if pd.notna(row.get("health_score"))
            else 0.5
        ),
        "days_active": (
            int(row.get("days_since_start", 0))
            if pd.notna(row.get("days_since_start"))
            else 0
        ),
        "day_of_week": (
            int(row.get("day_of_week", 0)) if pd.notna(row.get("day_of_week")) else None
        ),
        "is_weekend": (
            bool(row.get("is_weekend", False))
            if pd.notna(row.get("is_weekend"))
            else None
        ),
        "week_of_year": (
            int(row.get("week_of_year", 0))
            if pd.notna(row.get("week_of_year"))
            else None
        ),
        "adaptive_target_roas": (
            float(row.get("adaptive_target_roas", 0))
            if pd.notna(row.get("adaptive_target_roas"))
            else None
        ),
        "static_target_roas": (
            float(row.get("static_target_roas", 0))
            if pd.notna(row.get("static_target_roas"))
            else None
        ),
        "budget_utilization": (
            float(row.get("budget_utilization", 0))
            if pd.notna(row.get("budget_utilization"))
            else None
        ),
        "marginal_roas": (
            float(row.get("marginal_roas", 0))
            if pd.notna(row.get("marginal_roas"))
            else None
        ),
        "total_budget_today": total_budget_today,
        # Ad-level statistics (newly added)
        "num_ads": (int(row.get("num_ads", 1)) if pd.notna(row.get("num_ads")) else 1),
        "num_active_ads": (
            int(row.get("num_active_ads", 0))
            if pd.notna(row.get("num_active_ads"))
            else 0
        ),
        "ad_diversity": (
            int(row.get("ad_diversity", 1)) if pd.notna(row.get("ad_diversity")) else 1
        ),
        "ad_roas_mean": (
            float(row.get("ad_roas_mean", 0.0))
            if pd.notna(row.get("ad_roas_mean"))
            else 0.0
        ),
        "ad_roas_std": (
            float(row.get("ad_roas_std", 0.0))
            if pd.notna(row.get("ad_roas_std"))
            else 0.0
        ),
        "ad_roas_range": (
            float(row.get("ad_roas_range", 0.0))
            if pd.notna(row.get("ad_roas_range"))
            else 0.0
        ),
        "ad_spend_gini": (
            float(row.get("ad_spend_gini", 0.0))
            if pd.notna(row.get("ad_spend_gini"))
            else 0.0
        ),
        "top_ad_spend_pct": (
            float(row.get("top_ad_spend_pct", 1.0))
            if pd.notna(row.get("top_ad_spend_pct"))
            else 1.0
        ),
        "video_ads_ratio": (
            float(row.get("video_ads_ratio", 0.0))
            if pd.notna(row.get("video_ads_ratio"))
            else 0.0
        ),
        "format_diversity_score": (
            int(row.get("format_diversity_score", 1))
            if pd.notna(row.get("format_diversity_score"))
            else 1
        ),
    }


def _process_adsets(allocator, adset_groups, total_budget_today):
    """Process all adsets and collect allocation results"""
    results = []
    for adset_id, row in adset_groups.iterrows():
        current_budget = row.get("spend", 0) or (total_budget_today / len(adset_groups))
        metrics = _extract_metrics_from_row(
            row, str(adset_id), current_budget, total_budget_today
        )
        new_budget, decision_path = allocator.allocate_budget(**metrics)
        results.append(
            {
                "adset_id": adset_id,
                "current_budget": current_budget,
                "new_budget": new_budget,
                "change_pct": (
                    ((new_budget / current_budget) - 1) * 100
                    if current_budget > 0
                    else 0
                ),
                "roas_7d": metrics["roas_7d"],
                "health_score": metrics["health_score"],
                "days_active": metrics["days_active"],
                "decision_path": (
                    " -> ".join(decision_path) if decision_path else "default"
                ),
            }
        )
    return results


def _process_results(results, total_budget_today):
    """Process and scale results"""
    results_df = pd.DataFrame(results)
    total_new_budget = results_df["new_budget"].sum()
    if total_new_budget > 0:
        scale_factor = total_budget_today / total_new_budget
        results_df["new_budget"] = results_df["new_budget"] * scale_factor
        results_df["change_pct"] = (
            (results_df["new_budget"] / results_df["current_budget"]) - 1
        ) * 100
    return results_df


def _save_results(results_df, output_file):
    """Save results to CSV file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    logger.info("Results saved to: %s", output_file)


def _print_summary(results_df):
    """Print allocation summary"""
    logger.info("=" * 70)
    logger.info("ALLOCATION SUMMARY")
    logger.info("=" * 70)
    logger.info("Total adsets: %d", len(results_df))
    total_allocated = results_df["new_budget"].sum()
    logger.info("Total allocated budget: $%.2f", total_allocated)
    avg_budget = results_df["new_budget"].mean()
    logger.info("Average budget per adset: $%.2f", avg_budget)
    logger.info("Budget changes:")
    logger.info("Increases: %d adsets", (results_df["change_pct"] > 0).sum())
    logger.info("Decreases: %d adsets", (results_df["change_pct"] < 0).sum())
    logger.info("No change: %d adsets", (results_df["change_pct"] == 0).sum())
    logger.info("Top 5 increases:")
    columns = [
        "adset_id",
        "current_budget",
        "new_budget",
        "change_pct",
        "roas_7d",
    ]
    top_increases = results_df.nlargest(5, "change_pct")[columns]
    for _, row in top_increases.iterrows():
        logger.info(
            "%s: $%.2f → $%.2f (%+.1f%%), ROAS: %.2fx",
            row["adset_id"],
            row["current_budget"],
            row["new_budget"],
            row["change_pct"],
            row["roas_7d"],
        )
    logger.info("=" * 70)
    logger.info("SUCCESS: Allocation complete!")
    logger.info("=" * 70)


def _get_monthly_budget(args, config, customer_name: str) -> float:
    """Get monthly budget from CLI arg or config.

    Args:
        args: Command line arguments
        config: Configuration parser instance
        customer_name: Customer name

    Returns:
        Monthly budget amount

    Raises:
        ValueError: If no monthly budget specified
    """
    from datetime import datetime

    # CLI override takes priority
    if args.budget is not None:
        logger.info(f"Using CLI budget: ${args.budget:.2f}")
        return args.budget

    # Check config for monthly_budget
    config_budget = config.get_monthly_setting("monthly_budget_cap", None)
    if config_budget is not None:
        logger.info(f"Using config budget: ${config_budget:.2f}")
        return config_budget

    # Error: No budget specified
    logger.error("No monthly budget specified!")
    logger.error("Either:")
    logger.error("  1. Use --budget CLI argument")
    logger.error("  2. Add monthly_budget.monthly_budget_cap to config")
    raise ValueError("Monthly budget required but not specified")


def _archive_allocation(
    output_file: str, execution_date, customer: str, platform: str
) -> str:
    """Archive allocation file with date suffix.

    Args:
        output_file: Path to allocation output file
        execution_date: Date of execution
        customer: Customer name
        platform: Platform name

    Returns:
        Path to archived file
    """
    path_manager = get_path_manager(customer, platform)
    archive_dir = path_manager.archive_dir(
        customer, platform, execution_date.strftime("%Y-%m")
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_file)
    archive_name = (
        f"{output_path.stem}_{execution_date.strftime('%Y-%m-%d')}{output_path.suffix}"
    )
    archive_path = archive_dir / archive_name

    import shutil

    shutil.copy2(output_file, archive_path)
    logger.info(f"Archived allocation to: {archive_path}")
    return str(archive_path)


if __name__ == "__main__":
    sys.exit(main())
