"""Run rules-based pipeline command for CLI."""

import argparse
import logging

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def run_pipeline(
    customer: str = "moprobo",
    platform: str = "meta",
):
    """
    Run rules-based audience quality pipeline.

    Detects mistakes in audience setup and generates recommendations using
    transparent rules (no ML models).

    Args:
        customer: Customer name (default: moprobo)
        platform: Platform name (default: meta)
    """
    logger.info(f"Running rules-based pipeline for {customer}/{platform}")

    from src.rules import RulesPipeline

    pipeline = RulesPipeline(customer=customer, platform=platform)
    results = pipeline.run()
    return results


def add_parser(subparsers):
    """Add rules command parser."""
    parser = subparsers.add_parser(
        "rules",
        help="Run rules-based audience quality pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py rules
  python run.py rules --customer moprobo --platform meta
        """,
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

    parser.set_defaults(func=execute)


def execute(args):
    """Execute rules command."""
    setup_logging(level=logging.INFO)

    try:
        run_pipeline(
            customer=args.customer,
            platform=args.platform,
        )
        logger.info("Pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
