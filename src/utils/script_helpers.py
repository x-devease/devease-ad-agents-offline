"""
Common utilities for command-line scripts.

This module provides shared functionality for scripts that process
customer data, including CLI argument handling.
"""

import logging

logger = logging.getLogger(__name__)


def add_customer_argument(parser):
    """
    Add --customer argument to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--customer",
        type=str,
        default="all",
        help=(
            "Customer name for customer-specific data paths, or 'all' to run "
            "all customers (default: all)"
        ),
    )


def add_platform_argument(parser):
    """
    Add --platform argument to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--platform",
        type=str,
        default="meta",
        help=(
            "Platform name (e.g., 'meta', 'google') for platform-specific data paths. "
            "Data and config will be looked up in {customer}/{platform}/. "
            "(default: meta)"
        ),
    )


def add_config_argument(parser):
    """
    Add --config argument to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to rules.yaml configuration file. "
            "If None, looks for config/{customer}/{platform}/rules.yaml "
            "(default: None, uses customer/platform path)"
        ),
    )


def add_input_argument(parser, help_text=None):
    """
    Add --input argument to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
        help_text: Optional custom help text. If None, uses default.
    """
    default_help = "Input CSV file path"
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=help_text if help_text else default_help,
    )


def add_output_argument(parser, help_text=None):
    """
    Add --output argument to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
        help_text: Optional custom help text. If None, uses default.
    """
    default_help = "Output CSV file path"
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=help_text if help_text else default_help,
    )


def process_all_customers(config_path, action_name, process_customer_fn, args=None):
    """
    Process all customers with a common loop pattern.

    Args:
        config_path: Path to configuration file
        action_name: Name of the action (e.g., "aggregation", "extraction")
        process_customer_fn: Callback function that takes (customer, args) and
            returns True on success, False on failure
        args: Optional args object to pass to the callback

    Returns:
        0 on success, 1 if any customer processing failed
    """
    from src.utils.customer_paths import (
        get_all_customers,
    )  # pylint: disable=import-outside-toplevel

    customers = get_all_customers(config_path)
    if not customers:
        logger.error("No customers found in configuration file")
        return 1

    logger.info("=" * 70)
    logger.info("Running %s for all customers", action_name)
    logger.info("=" * 70)
    logger.info("Found %d customer(s): %s", len(customers), ", ".join(customers))

    has_failures = False
    for customer in customers:
        logger.info("")
        logger.info("-" * 70)
        logger.info("Processing customer: %s", customer)
        logger.info("-" * 70)

        if args is not None:
            if not process_customer_fn(customer, args):
                has_failures = True
        else:
            if not process_customer_fn(customer):
                has_failures = True

    logger.info("")
    logger.info("=" * 70)
    logger.info("All customers processed")
    logger.info("=" * 70)
    return 1 if has_failures else 0
