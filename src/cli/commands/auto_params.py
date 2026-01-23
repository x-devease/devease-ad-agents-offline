"""Auto-calculate rule parameters from customer data command."""

import argparse
import logging

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def calculate_params(
    customer: str = "moprobo",
    platform: str = "meta",
):
    """
    Auto-calculate rule parameters from customer data.

    Analyzes customer's adset and Shopify data to determine optimal thresholds
    for detection rules using percentile-based approach.

    Args:
        customer: Customer name (default: moprobo)
        platform: Platform name (default: meta)
    """
    logger.info(f"Auto-calculating parameters for {customer}/{platform}")

    from src.utils.auto_params import AutoParams

    params = AutoParams.calculate_from_data(customer, platform)

    logger.info("Parameters calculated successfully")
    return params


def add_parser(subparsers):
    """Add auto-params command parser."""
    parser = subparsers.add_parser(
        "auto-params",
        help="Auto-calculate rule parameters from customer data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py auto-params
  python run.py auto-params --customer moprobo --platform meta

This command analyzes your customer's data to determine optimal thresholds
for detection rules. It uses percentile-based calculations:

- ROAS thresholds (75th/50th/25th percentiles)
- Spend thresholds (75th/50th/25th percentiles)
- CTR, CPC, CPM thresholds (percentiles)
- Shopify buyer demographics (age, geo concentration)

Non-calculable parameters (platform constraints, business rules) use
domain knowledge defaults.
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
    """Execute auto-params command."""
    setup_logging(level=logging.INFO)

    try:
        calculate_params(
            customer=args.customer,
            platform=args.platform,
        )
        logger.info("Auto-params completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Auto-params failed: {e}", exc_info=True)
        return 1
