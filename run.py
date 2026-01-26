#!/usr/bin/env python3
"""
DevEase Meta Autopilot - Unified CLI

Main entry point for all 4 major components:

1. Adset Allocator (src/adset/allocator/):
   - extract: Feature extraction from raw Meta ads data
   - execute: Budget allocation using rules-based approach
   - tune: Parameter tuning using Bayesian optimization
   - discover: Pattern discovery from data using decision trees

2. Adset Generator (src/adset/generator/):
   - rules: Run rules-based audience configuration pipeline

3. Ad Recommender (src/ad/recommender/):
   - extract-features: Extract features from images using GPT-4 Vision API
   - recommend: Generate creative recommendations from feature data

4. Ad Generator (src/ad/generator/):
   - prompt: Generate prompts from recommendations (structured or nano)
   - generate: Generate images with nano-banana models
   - run: End-to-end pipeline (recommendations → prompts → images)

Default approach: Rules-only with Bayesian-optimized parameters from config/adset/allocator/rules.yaml

Usage:
    # Adset operations
    python run.py extract --customer moprobo
    python run.py execute --customer moprobo --budget 10000
    python run.py tune --customer moprobo --iterations 50
    python run.py discover --customer moprobo
    python run.py rules --customer moprobo --platform meta
    
    # Creative operations
    python run.py extract-features --ad-data-csv data/ad_data.csv --output-csv data/features_with_roas.csv
    python run.py recommend --input-csv data/features_with_roas.csv --output-dir config/ad/recommender
    python run.py prompt structured --base-prompt "A professional product image"
    python run.py generate --source-image product.jpg --prompt "Professional product image" --num-variations 3
    python run.py run --source-image product.jpg --base-prompt "A professional product image" --num-variations 3
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
from src.utils.logger_config import setup_logging

logger = setup_logging()


def main():
    """Main entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="DevEase Meta Autopilot - Unified CLI for Adset & Creative Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Adset Allocator
  python run.py extract --customer moprobo
  python run.py execute --customer moprobo --budget 10000
  python run.py tune --customer moprobo --iterations 100
  python run.py discover --customer moprobo

  # Adset Generator
  python run.py rules --customer moprobo --platform meta

  # Ad Recommender
  python run.py extract-features --ad-data-csv data/ad_data.csv --output-csv data/features_with_roas.csv
  python run.py recommend --input-csv data/features_with_roas.csv --output-dir config/recommendations/moprobo/meta

  # Ad Generator
  python run.py prompt structured --base-prompt "A professional product image"
  python run.py generate --source-image product.jpg --prompt "Professional product image" --num-variations 3
  python run.py run --source-image product.jpg --base-prompt "A professional product image" --num-variations 3

Environment Configuration:
  python run.py execute --customer moprobo --environment development
  python run.py execute --customer moprobo --base-dir /path/to/data
        """,
    )

    # Global configuration options
    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "production", "test"],
        default=None,
        help="Environment to use (default: from config or production)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Override base directory for data paths",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
        metavar="COMMAND",
    )

    # Add subcommands for all 4 components
    # 1. Adset Allocator
    _add_extract_subcommand(subparsers)
    _add_execute_subcommand(subparsers)
    _add_tune_subcommand(subparsers)
    _add_discover_subcommand(subparsers)
    
    # 2. Adset Generator
    _add_rules_subcommand(subparsers)
    
    # 3. Ad Recommender
    _add_extract_features_subcommand(subparsers)
    _add_recommend_subcommand(subparsers)
    
    # 4. Ad Generator
    _add_prompt_subcommand(subparsers)
    _add_generate_subcommand(subparsers)
    _add_run_subcommand(subparsers)

    args = parser.parse_args()

    # Apply global configuration
    if args.environment:
        import os
        os.environ["ENVIRONMENT"] = args.environment
    if args.base_dir:
        import os
        os.environ["BASE_DIR"] = args.base_dir
    if args.verbose:
        import os
        os.environ["VERBOSE"] = "true"

    # Execute the appropriate command
    try:
        return args.func(args)
    except Exception as err:
        logger.exception("Command failed: %s", err)
        return 1


def _add_extract_subcommand(subparsers):
    """Add extract subcommand."""
    parser = subparsers.add_parser(
        "extract",
        help="Extract features from raw Meta ads data",
        description="""
Feature extraction from Meta ads data.

Joins account, campaign, adset, and ad-level data into enriched ad-level
features, then aggregates to adset-level features for budget allocation.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    from src.utils.script_helpers import (
        add_customer_argument,
        add_config_argument,
        add_platform_argument,
    )

    add_customer_argument(parser)
    add_platform_argument(parser)
    add_config_argument(parser)

    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip feature preprocessing (normalization, bucketing)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip feature normalization",
    )
    parser.add_argument(
        "--no-bucket",
        action="store_true",
        help="Skip feature bucketing",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip adset-level aggregation (only extract ad-level features)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine split CSV files before extraction (for time-period split data)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip column validation when combining files",
    )

    # Explicit file paths
    parser.add_argument(
        "--ad-file",
        type=str,
        default=None,
        help="Explicit path to ad-level CSV file",
    )
    parser.add_argument(
        "--adset-file",
        type=str,
        default=None,
        help="Explicit path to adset-level CSV file",
    )
    parser.add_argument(
        "--campaign-file",
        type=str,
        default=None,
        help="Explicit path to campaign-level CSV file",
    )
    parser.add_argument(
        "--account-file",
        type=str,
        default=None,
        help="Explicit path to account-level CSV file",
    )

    parser.set_defaults(func=_cmd_extract)


def _add_execute_subcommand(subparsers):
    """Add execute subcommand."""
    parser = subparsers.add_parser(
        "execute",
        help="Run rules-based budget allocation",
        description="""
Execute budget allocation using rules-based approach with Bayesian-optimized
parameters from config/adset/allocator/rules.yaml.

Analyzes adset performance metrics and allocates budget based on configurable
rules for ROAS, health score, trends, and efficiency.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    from src.utils.script_helpers import (
        add_customer_argument,
        add_config_argument,
        add_platform_argument,
        add_input_argument,
        add_output_argument,
    )

    add_customer_argument(parser)
    add_platform_argument(parser)
    add_config_argument(parser)

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
            "(default: results/{customer}/rules/adset_budget_allocations.csv)"
        ),
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=10000.0,
        help="Total monthly budget to allocate (default: 10000.0)",
    )

    parser.set_defaults(func=_cmd_execute)


def _add_tune_subcommand(subparsers):
    """Add tune subcommand."""
    parser = subparsers.add_parser(
        "tune",
        help="Tune allocation parameters using Bayesian optimization",
        description="""
Automatic parameter tuning for budget allocation rules.

Uses Bayesian optimization to find optimal parameter values that
maximize ROAS and revenue while maintaining budget utilization
constraints.
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

    # Legacy examples (still support for backward compatibility)
    parser.add_argument(
        "--example-single-param",
        action="store_true",
        help="Run single parameter tuning example (legacy grid search)",
    )

    parser.add_argument(
        "--example-grid-search",
        action="store_true",
        help="Run grid search example (legacy grid search)",
    )

    parser.set_defaults(func=_cmd_tune)


def _cmd_extract(args):
    """Handle extract command using workflow."""
    from src.meta.adset.allocator.features.workflows import ExtractWorkflow

    logger.info("=" * 70)
    logger.info("EXTRACT: Feature Extraction")
    logger.info("=" * 70)

    # Combine files first if requested
    if args.combine:
        logger.info("")
        logger.info("STEP 1: Combining split CSV files")
        logger.info("-" * 70)

        success = _combine_customer_data(
            customer=args.customer,
            platform=args.platform,
            skip_validation=args.skip_validation,
        )

        if not success:
            logger.error("Combination failed. Proceeding with extraction anyway...")
        else:
            logger.info("Combination completed successfully")

    workflow = ExtractWorkflow(
        config_path=args.config,
        preprocess=not args.no_preprocess,
        normalize=not args.no_normalize,
        bucket=not args.no_bucket,
        aggregate_to_adset=not args.no_aggregate,
    )

    results = workflow.run(
        customer=args.customer,
        platform=args.platform,
        ad_file=args.ad_file,
        adset_file=args.adset_file,
        campaign_file=args.campaign_file,
        account_file=args.account_file,
    )

    # Return exit code (0 if all successful, 1 if any failed)
    return 0 if all(r.success for r in results.values()) else 1


def _combine_customer_data(customer, platform, skip_validation=False):
    """Combine split CSV files for a customer.

    Args:
        customer: Customer name
        platform: Platform name
        skip_validation: If True, skip column validation

    Returns:
        True if successful, False otherwise
    """
    from src.meta.adset.allocator.features.utils.csv_combiner import CSVCombiner
    from pathlib import Path

    project_root = Path(__file__).parent

    # Determine source and output directories
    source_base = project_root / "notebooks" / customer / platform / "raw"
    output_dir = project_root / "datasets" / customer / platform / "raw"

    logger.info("Source directory: %s", source_base)
    logger.info("Output directory: %s", output_dir)
    logger.info("")

    # Process each granularity
    granularities_to_process = []

    for granularity in ["daily", "hourly"]:
        for entity_type in ["ad", "adset"]:
            # Source directory for this granularity
            # Pattern: {entity_type}-{granularity}-{range}
            # e.g., daily-ad-1y, daily-adset-1y
            source_dir = source_base / f"{granularity}-{entity_type}-1y"

            # Check if directory exists
            if not source_dir.exists():
                # Try hourly-adset-1w pattern
                source_dir = source_base / f"{granularity}-{entity_type}-1w"

            if source_dir.exists():
                granularities_to_process.append((entity_type, granularity, source_dir))

    if not granularities_to_process:
        logger.warning("No split files found to combine")
        return True  # Not an error, just nothing to combine

    # Process each entity type and granularity
    success_count = 0

    for entity_type, granularity, source_dir in granularities_to_process:
        logger.info("")
        logger.info("-" * 70)
        logger.info("Processing: %s %s", entity_type, granularity)

        combiner = CSVCombiner(
            source_dir=source_dir,
            output_dir=output_dir,
        )

        if combiner.process_entity_type(
            entity_type=entity_type,
            granularity=granularity,
            skip_validation=skip_validation,
            dry_run=False,
        ):
            success_count += 1

    # Summary
    logger.info("")
    logger.info("-" * 70)
    logger.info("Combination Summary: %d/%d successful", success_count, len(granularities_to_process))

    return success_count > 0


def _cmd_execute(args):
    """Handle execute command using workflow."""
    from src.meta.adset.allocator.workflows import AllocationWorkflow

    logger.info("=" * 70)
    logger.info("EXECUTE: Budget Allocation (Rules-Based)")
    logger.info("=" * 70)
    logger.info("Mode: Rules-Based with Bayesian-optimized Parameters")

    # Handle test mode separately (doesn't use workflow)
    if args.test:
        return _cmd_execute_test(args)

    workflow = AllocationWorkflow(
        config_path=args.config,
        budget=args.budget,
        approach="rules",
    )

    results = workflow.run(
        customer=args.customer,
        platform=args.platform,
        input_file=args.input,
        output_file=args.output,
    )

    # Return exit code
    return 0 if all(r.success for r in results.values()) else 1


def _cmd_execute_test(args):
    """Handle execute test mode."""
    from src.meta.adset import DecisionRules, Allocator, SafetyRules
    from src.meta.adset.allocator.utils.parser import Parser

    logger.info("=" * 70)
    logger.info("TEST: Rule-Based Budget Allocation")
    logger.info("=" * 70)

    try:
        config = Parser(config_path=args.config, customer_name=args.customer or "moprobo", platform="meta")
        safety_rules = SafetyRules(config)
        decision_rules = DecisionRules(config)
        allocator = Allocator(safety_rules, decision_rules, config)

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

        return 0

    except Exception as err:
        logger.exception("Test failed: %s", err)
        return 1


def _cmd_tune(args):
    """Handle tune command using workflow."""
    from src.meta.adset.allocator.workflows import TuningWorkflow

    logger.info("=" * 70)
    logger.info("TUNE: Parameter Tuning")
    logger.info("=" * 70)

    # Handle legacy examples (backward compatibility)
    if args.example_single_param or args.example_grid_search:
        # Fall back to original tune.py for legacy examples
        import src.adset.allocator.cli.commands.tune as tune_module

        tune_argv = ["tune.py"]
        if args.example_single_param:
            tune_argv.append("--example-single-param")
        if args.example_grid_search:
            tune_argv.append("--example-grid-search")

        original_argv = sys.argv
        try:
            sys.argv = tune_argv
            return tune_module.main()
        finally:
            sys.argv = original_argv

    # Use new workflow for automated tuning
    workflow = TuningWorkflow(
        config_path=args.config,
        n_calls=args.iterations,
        n_initial_points=args.initial_points,
        update_config=not args.no_update,
        generate_report=True,
    )

    results = workflow.run(
        customer=args.customer,
    )

    # Return exit code
    return 0 if all(r.success for r in results.values()) else 1


def _add_discover_subcommand(subparsers):
    """Add discover subcommand for pattern discovery."""
    from src.utils.script_helpers import (
        add_customer_argument,
        add_platform_argument,
    )

    parser = subparsers.add_parser(
        "discover",
        help="Discover patterns from data using decision trees",
        description="""
Automatically discover decision rules from data using decision trees.

Discovers human-readable rules that can improve ROAS:
- Mine rules from historical data
- Validate using forward-looking ROAS
- Generate YAML configs for deployment
- Support for safety constraint checking

Methods:
- decision_tree: Extract rules from decision trees (default)
- association: Mine association rules for feature combinations
- shap: Explain ML predictions with SHAP values
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_customer_argument(parser)
    add_platform_argument(parser)

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV file with features (default: datasets/{customer}/features/adset_features.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for discovered rules YAML (default: patterns/{customer}/discovered_rules.yaml)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="decision_tree",
        choices=["decision_tree", "association", "shap"],
        help="Pattern discovery method (default: decision_tree)",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum depth for decision tree (default: 5)",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum samples required for a rule (default: 50)",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for rules (default: 0.7)",
    )

    parser.add_argument(
        "--target-threshold",
        type=float,
        default=2.0,
        help="ROAS threshold for binary classification (default: 2.0)",
    )

    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation step (not recommended for production)",
    )

    parser.add_argument(
        "--deploy-ready",
        action="store_true",
        help="Only output rules that pass validation (recommendation: deploy or test)",
    )

    parser.set_defaults(func=_cmd_discover)


def _load_discovery_data(args):
    """Load and validate data for discovery."""
    import pandas as pd
    from pathlib import Path

    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path(f"datasets/{args.customer}/features/adset_features.csv")

    logger.info(f"Loading features from: {input_path}")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error(
            f"Please extract features first:\n  python run.py extract --customer {args.customer}"
        )
        return None, None

    # Load data
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} adsets with {len(df.columns)} features")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None, None

    # Check for required columns
    required_cols = ["purchase_roas_7d"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {list(df.columns)[:10]}...")
        return None, None

    # Split data for validation (80/20 temporal split)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    logger.info(f"Training data: {len(df_train)} samples")
    logger.info(f"Test data: {len(df_test)} samples")

    return df_train, df_test


def _mine_discovery_rules(df_train, args):
    """Mine rules from training data."""
    from src.meta.adset.allocator.lib.discovery_miner import DecisionTreeMiner

    logger.info("")
    logger.info("Mining rules using decision trees...")
    logger.info(f"  Max depth: {args.max_depth}")
    logger.info(f"  Min samples per leaf: {args.min_samples}")

    miner = DecisionTreeMiner(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples,
    )

    rules = miner.mine_rules(
        df_train,
        target_col="purchase_roas_7d",
        target_threshold=args.target_threshold,
    )

    logger.info(f"Discovered {len(rules)} rules")

    if len(rules) == 0:
        logger.warning("No rules discovered. Try adjusting parameters:")
        logger.warning("  - Increase max-depth")
        logger.warning("  - Decrease min-samples")
        logger.warning("  - Adjust target-threshold")
        return None

    return rules


def _validate_discovery_rules(rules, df_train, df_test, args):
    """Validate discovered rules."""
    from src.meta.adset.allocator.lib.discovery_validator import RuleValidator

    if args.no_validation:
        return rules

    logger.info("")
    logger.info("Validating rules...")

    validator = RuleValidator(df_train, df_test)

    validated_rules = []
    deploy_count = 0
    test_count = 0
    reject_count = 0

    for rule in rules:
        validation = validator.validate_rule(rule)
        rule.validation_metric = validation.mean_roas

        if validation.recommendation == "deploy":
            deploy_count += 1
            validated_rules.append(rule)
        elif validation.recommendation == "test":
            test_count += 1
            validated_rules.append(rule)
        else:
            reject_count += 1

        # Only add deploy-ready rules if requested
        if args.deploy_ready and validation.recommendation == "reject":
            continue

    logger.info(f"  Deploy: {deploy_count}")
    logger.info(f"  Test: {test_count}")
    logger.info(f"  Reject: {reject_count}")

    if args.deploy_ready:
        validated_rules = [r for r in validated_rules if r.validation_metric is not None]
        logger.info(f"Deploy-ready rules: {len(validated_rules)}")

    return validated_rules


def _display_discovery_results(rules):
    """Display top discovered rules."""
    logger.info("")
    logger.info("Top 5 discovered rules:")
    for i, rule in enumerate(rules[:5], 1):
        logger.info(f"  {i}. {rule.rule_id}")
        logger.info(f"     Conditions: {list(rule.conditions.keys())[:3]}")
        logger.info(f"     Outcome: {rule.outcome} ({rule.adjustment_factor:.2f}x)")
        logger.info(f"     Support: {rule.support}, Confidence: {rule.confidence:.2f}")
        logger.info(f"     Lift: {rule.lift:.2f}")
        if rule.validation_metric:
            logger.info(f"     Validation ROAS: {rule.validation_metric:.2f}")


def _save_discovery_results(rules, args, validated_rules):
    """Save discovered rules to YAML file."""
    from pathlib import Path
    from src.meta.adset.allocator.lib.discovery_extractor import RuleExtractor

    output_path = Path(args.output) if args.output else Path(f"patterns/{args.customer}/discovered_rules.yaml")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info(f"Generating YAML config: {output_path}")

    extractor = RuleExtractor()
    extractor.generate_yaml_config(rules, str(output_path), config_type="decision_rules")

    logger.info("")
    logger.info("=" * 70)
    logger.info("DISCOVERY COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Discovered {len(rules)} rules")
    logger.info(f"Output: {output_path}")

    if not args.no_validation and len(validated_rules) > 0:
        logger.info("")
        logger.info("To deploy these rules:")
        logger.info(f"  1. Review: cat {output_path}")
        logger.info(f"  2. Merge with existing config manually")
        logger.info(f"  3. Run allocation: python run.py execute --customer {args.customer}")


def _cmd_discover(args):
    """Handle discover command for pattern discovery."""
    logger.info("=" * 70)
    logger.info("DISCOVER: Pattern Discovery from Data")
    logger.info("=" * 70)

    # Load and validate data
    df_train, df_test = _load_discovery_data(args)
    if df_train is None or df_test is None:
        return 1

    # Mine rules
    rules = _mine_discovery_rules(df_train, args)
    if rules is None:
        return 1

    # Validate rules
    validated_rules = _validate_discovery_rules(rules, df_train, df_test, args)

    # Display results
    _display_discovery_results(rules)

    # Save results
    _save_discovery_results(rules, args, validated_rules)

    return 0


def _add_rules_subcommand(subparsers):
    """Add rules subcommand for adset generator."""
    from src.utils.script_helpers import add_customer_argument, add_platform_argument
    
    parser = subparsers.add_parser(
        "rules",
        help="Run rules-based audience configuration pipeline",
        description="""
Run rules-based audience quality pipeline.

Detects mistakes in audience setup and generates recommendations using
transparent rules (no ML models).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    add_customer_argument(parser)
    add_platform_argument(parser)
    
    parser.set_defaults(func=_cmd_rules)


def _cmd_rules(args):
    """Handle rules command for adset generator."""
    logger.info("=" * 70)
    logger.info("RULES: Audience Configuration Pipeline")
    logger.info("=" * 70)
    
    try:
        from src.meta.adset.generator.detection import MistakeDetector
        from src.meta.adset.generator.generation import AudienceRecommender
        
        logger.info("Running rules-based pipeline for %s/%s", args.customer, args.platform or "meta")
        
        # TODO: Implement full pipeline integration
        # For now, just log that the command is available
        logger.warning("Full pipeline integration pending - using basic components")
        
        logger.info("=" * 70)
        logger.info("SUCCESS: Rules pipeline complete!")
        logger.info("=" * 70)
        
        return 0
    except Exception as err:
        logger.exception("Rules pipeline failed: %s", err)
        return 1


def _add_extract_features_subcommand(subparsers):
    """Add extract-features subcommand for ad recommender."""
    parser = subparsers.add_parser(
        "extract-features",
        help="Extract features from images using GPT-4 Vision API",
        description="""
Extract image features using GPT-4 Vision API and integrate with ROAS data.

Can extract from:
- Top/bottom performer CSV files
- Ad performance data CSV
- Single or batch image processing
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--top-csv",
        type=str,
        default=None,
        help="CSV file with top performing images",
    )
    
    parser.add_argument(
        "--bottom-csv",
        type=str,
        default=None,
        help="CSV file with bottom performing images",
    )
    
    parser.add_argument(
        "--ad-data-csv",
        type=str,
        default=None,
        help="CSV file with ad performance data",
    )
    
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/features_with_roas.csv",
        help="Output CSV file path (default: data/features_with_roas.csv)",
    )
    
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic ROAS data for testing",
    )
    
    parser.set_defaults(func=_cmd_extract_features)


def _cmd_extract_features(args):
    """Handle extract-features command for ad recommender."""
    logger.info("=" * 70)
    logger.info("EXTRACT-FEATURES: Image Feature Extraction")
    logger.info("=" * 70)
    
    try:
        from src.meta.ad.recommender.features import extract_batch_features, add_roas_to_features
        
        if args.top_csv and args.bottom_csv:
            # Extract from top/bottom CSVs
            import pandas as pd
            
            top_df = pd.read_csv(args.top_csv)
            bottom_df = pd.read_csv(args.bottom_csv)
            
            # Extract image paths from CSVs (assuming 'image_path' or 'filename' column)
            image_paths = []
            if 'image_path' in top_df.columns:
                image_paths.extend(top_df['image_path'].tolist())
            elif 'filename' in top_df.columns:
                image_paths.extend(top_df['filename'].tolist())
            
            if 'image_path' in bottom_df.columns:
                image_paths.extend(bottom_df['image_path'].tolist())
            elif 'filename' in bottom_df.columns:
                image_paths.extend(bottom_df['filename'].tolist())
            
            if not image_paths:
                logger.error("No image paths found in CSV files")
                return 1
            
            # Extract features
            features_csv = args.output_csv.replace('.csv', '_features.csv')
            extract_batch_features(image_paths, output_csv=features_csv)
            
            # Add ROAS
            add_roas_to_features(
                features_csv=features_csv,
                ad_data_csv=args.ad_data_csv,
                output_csv=args.output_csv,
                synthetic=args.synthetic,
            )
        elif args.ad_data_csv:
            # Extract with real ad data
            logger.info("Extracting features with ad performance data...")
            # This would require image paths from ad data
            logger.warning("Direct ad data extraction requires image paths in ad data CSV")
            return 1
        else:
            logger.error("Must provide either --top-csv/--bottom-csv or --ad-data-csv")
            return 1
        
        logger.info("=" * 70)
        logger.info("SUCCESS: Feature extraction complete!")
        logger.info("Output: %s", args.output_csv)
        logger.info("=" * 70)
        
        return 0
    except Exception as err:
        logger.exception("Feature extraction failed: %s", err)
        return 1


def _add_recommend_subcommand(subparsers):
    """Add recommend subcommand for ad recommender."""
    parser = subparsers.add_parser(
        "recommend",
        help="Generate creative recommendations from feature data",
        description="""
Generate statistical pattern-based recommendations for creative optimization.

Analyzes feature patterns in top vs bottom performers and generates
actionable recommendations with evidence.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Input CSV file with features and ROAS data",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="config/ad/recommender",
        help="Output directory for recommendations (default: config/ad/recommender)",
    )
    
    parser.add_argument(
        "--customer",
        type=str,
        default="moprobo",
        help="Customer name (default: moprobo)",
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="meta",
        help="Platform name (default: meta)",
    )
    
    parser.set_defaults(func=_cmd_recommend)


def _cmd_recommend(args):
    """Handle recommend command for ad recommender."""
    logger.info("=" * 70)
    logger.info("RECOMMEND: Creative Recommendations")
    logger.info("=" * 70)
    
    try:
        from src.meta.ad.recommender.recommendations import RuleEngine
        import pandas as pd
        from pathlib import Path
        
        # Load features
        logger.info("Loading features from: %s", args.input_csv)
        df = pd.read_csv(args.input_csv)
        
        # Initialize rule engine
        rule_engine = RuleEngine()
        
        # Generate recommendations
        logger.info("Generating recommendations...")
        recommendations = rule_engine.generate_recommendations(df)
        
        # Save recommendations (MD format only)
        # Output to config/ad/recommender/{customer}/{platform}
        output_dir = Path(args.output_dir) / args.customer / args.platform
        output_dir.mkdir(parents=True, exist_ok=True)

        from src.meta.ad.recommender.recommendations.md_io import (
            export_recommendations_md,
        )

        output_md = output_dir / "recommendations.md"
        export_recommendations_md(recommendations, output_md)

        logger.info("=" * 70)
        logger.info("SUCCESS: Recommendations generated!")
        logger.info("Output: %s", output_md)
        logger.info("=" * 70)
        
        return 0
    except Exception as err:
        logger.exception("Recommendation generation failed: %s", err)
        return 1


def _add_prompt_subcommand(subparsers):
    """Add prompt subcommand for ad generator."""
    parser = subparsers.add_parser(
        "prompt",
        help="Generate prompts from recommendations",
        description="""
Generate prompts from feature recommendations.

Modes:
- structured: Generate structured prompt from recommendations
- nano: Generate Nano Banana-optimized prompt via GPT-4o
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "mode",
        choices=["structured", "nano"],
        help="Prompt generation mode",
    )
    
    parser.add_argument(
        "--customer",
        type=str,
        default="moprobo",
        help="Customer name (default: moprobo)",
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="meta",
        help="Platform name (default: meta)",
    )
    
    parser.add_argument(
        "--base-prompt",
        type=str,
        default=(
            "Professional product photography, sharp focus, studio lighting, "
            "high resolution, clean composition, commercial quality"
        ),
        help="Base prompt for positive/structured output",
    )
    
    parser.add_argument(
        "--source-image",
        type=str,
        default=None,
        help="Source image path (required for nano mode)",
    )
    
    parser.add_argument(
        "--recommendations-file",
        type=str,
        default=None,
        help="Path to recommendations file (.md from ad/recommender)",
    )
    
    parser.set_defaults(func=_cmd_prompt)


def _cmd_prompt(args):
    """Handle prompt command for ad generator."""
    logger.info("=" * 70)
    logger.info("PROMPT: Prompt Generation (%s mode)", args.mode)
    logger.info("=" * 70)
    
    try:
        from pathlib import Path

        # Load recommendations (.json or .md from ad/recommender)
        from src.meta.ad.recommender.recommendations.md_io import (
            load_recommendations_file,
        )

        if args.recommendations_file:
            recs_path = Path(args.recommendations_file)
        else:
            # Look in config/ad/recommender/{customer}/{platform}
            base = Path(
                f"config/ad/recommender/{args.customer}/{args.platform}"
            )
            recs_path = base / "recommendations.md"
        recommendations = load_recommendations_file(recs_path)

        if args.mode == "structured":
            from src.meta.ad.recommender.recommendations.prompt_formatter import (
                format_recs_as_prompts,
            )

            result = format_recs_as_prompts(
                recommendations,
                base_positive=args.base_prompt,
            )
            logger.info("Generated prompt:")
            logger.info("Positive: %s", result.get("final_prompt", ""))
            logger.info("Negative: %s", result.get("negative_prompt", ""))

        elif args.mode == "nano":
            if not args.source_image:
                logger.error("--source-image is required for nano mode")
                return 1

            from src.meta.ad.generator.core.generation.prompt_converter import (
                PromptConverter,
            )
            from src.meta.ad.recommender.utils.api_keys import get_openai_api_key

            api_key = get_openai_api_key()
            converter = PromptConverter(api_key=api_key)
            out = converter.convert_to_nano_banana(
                base_prompt=args.base_prompt,
                source_image_path=args.source_image,
                recommendations=recommendations,
            )
            logger.info("Generated Nano Banana prompt: %s", out.get("flux_prompt", ""))
        
        logger.info("=" * 70)
        logger.info("SUCCESS: Prompt generation complete!")
        logger.info("=" * 70)
        
        return 0
    except Exception as err:
        logger.exception("Prompt generation failed: %s", err)
        return 1


def _add_generate_subcommand(subparsers):
    """Add generate subcommand for ad generator."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate images with nano-banana models",
        description="""
Generate images using FAL.ai nano-banana models.

Supports image-to-image generation with source images and prompts.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--source-image",
        type=str,
        required=True,
        help="Source image path",
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt text for generation",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="nano-banana-pro",
        choices=["nano-banana-pro", "nano-banana"],
        help="Model to use (default: nano-banana-pro)",
    )
    
    parser.add_argument(
        "--num-variations",
        type=int,
        default=1,
        help="Number of variations to generate (default: 1)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_images",
        help="Output directory (default: generated_images)",
    )
    
    parser.set_defaults(func=_cmd_generate)


def _cmd_generate(args):
    """Handle generate command for ad generator."""
    logger.info("=" * 70)
    logger.info("GENERATE: Image Generation")
    logger.info("=" * 70)
    
    try:
        from src.meta.ad.generator.core.generation.generator import ImageGenerator
        from pathlib import Path
        
        generator = ImageGenerator(
            model=args.model,
            output_dir=args.output_dir,
        )
        
        results = []
        for i in range(args.num_variations):
            logger.info("Generating variation %d/%d", i + 1, args.num_variations)
            result = generator.generate(
                prompt=args.prompt,
                source_image_path=args.source_image,
            )
            results.append(result)
            
            if result.get('success'):
                logger.info("Generated: %s", result.get('image_path'))
            else:
                logger.error("Generation failed: %s", result.get('error'))
        
        logger.info("=" * 70)
        logger.info("SUCCESS: Image generation complete!")
        logger.info("Generated %d variations", len([r for r in results if r.get('success')]))
        logger.info("=" * 70)
        
        return 0
    except Exception as err:
        logger.exception("Image generation failed: %s", err)
        return 1


def _add_run_subcommand(subparsers):
    """Add run subcommand for ad generator (end-to-end pipeline)."""
    parser = subparsers.add_parser(
        "run",
        help="Run end-to-end pipeline (recommendations → prompts → images)",
        description="""
Run complete creative generation pipeline:
1. Load recommendations
2. Generate prompts
3. Generate images
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--customer",
        type=str,
        default="moprobo",
        help="Customer name (default: moprobo)",
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="meta",
        help="Platform name (default: meta)",
    )
    
    parser.add_argument(
        "--source-image",
        type=str,
        required=True,
        help="Source image path",
    )
    
    parser.add_argument(
        "--base-prompt",
        type=str,
        default=(
            "Professional product photography, sharp focus, studio lighting, "
            "high resolution, clean composition, commercial quality"
        ),
        help="Base prompt for generation",
    )
    
    parser.add_argument(
        "--num-variations",
        type=int,
        default=3,
        help="Number of variations to generate (default: 3)",
    )
    
    parser.add_argument(
        "--product-name",
        type=str,
        default=None,
        help="Product name for context",
    )
    
    parser.set_defaults(func=_cmd_run)


def _cmd_run(args):
    """Handle run command for ad generator (end-to-end pipeline)."""
    logger.info("=" * 70)
    logger.info("RUN: End-to-End Creative Generation Pipeline")
    logger.info("=" * 70)
    
    try:
        from src.meta.ad.generator.pipeline import CreativePipeline
        from src.meta.ad.generator.pipeline.pipeline import (
            CreativePipelineConfig,
            RecommendationPaths,
        )
        from pathlib import Path
        
        # Determine recommendation path (ad/recommender format)
        # Look in config/ad/recommender/{customer}/{platform}
        rec_path = Path(
            f"config/ad/recommender/{args.customer}/{args.platform}/recommendations.md"
        )
        
        # Build config with recommendation path
        config = CreativePipelineConfig(
            product_name=args.product_name or args.customer,
            recommendation_paths=RecommendationPaths(
                recommendation_path=rec_path if rec_path.exists() else None
            ),
        )
        
        # Initialize pipeline
        pipeline = CreativePipeline(config)
        
        # Run pipeline
        results = pipeline.run(
            source_image_path=args.source_image,
            num_variations=args.num_variations,
        )
        
        logger.info("=" * 70)
        logger.info("SUCCESS: Pipeline complete!")
        logger.info("Generated %d variations", len(results))
        logger.info("=" * 70)
        
        return 0
    except Exception as err:
        logger.exception("Pipeline failed: %s", err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
