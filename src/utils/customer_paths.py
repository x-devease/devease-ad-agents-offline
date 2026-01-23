"""
Customer path utilities for managing multi-customer data and results.

This module now uses the new PathManager internally while maintaining
backward compatibility with existing code.
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


# Import PathManager with graceful fallback for backward compatibility
try:
    from ..config.path_manager import get_path_manager, PathManager

    _PATHMANAGER_AVAILABLE = True
except ImportError:
    _PATHMANAGER_AVAILABLE = False
    logger.warning("PathManager not available, using legacy path resolution")


def _get_path_manager(customer: str, platform: str) -> Optional[PathManager]:
    """Get PathManager instance if available."""
    if _PATHMANAGER_AVAILABLE:
        return get_path_manager(customer=customer, platform=platform)
    return None


def get_customer_data_dir(
    customer: Optional[str] = None, platform: Optional[str] = None
) -> Path:
    """
    Get the data directory path for a customer.

    Args:
        customer: Customer name. Required parameter.
        platform: Platform name (e.g., 'meta', 'google'). If provided, data will be
            looked up in datasets/{customer}/{platform}/raw or datasets/{customer}/{platform}/.

    Returns:
        Path to customer-specific data directory (raw subdirectory if exists)

    Raises:
        ValueError: If customer is None, empty string, "all"
    """
    if not customer or customer == "all":
        raise ValueError("customer parameter is required and cannot be None or 'all'")

    # Use PathManager if available
    path_manager = _get_path_manager(customer, platform or "meta")
    if path_manager:
        return path_manager.raw_data_dir(customer, platform or "meta")

    # Legacy fallback
    base_dir = Path("datasets")
    customer_dir = base_dir / customer

    # If platform is specified, use platform-specific paths
    if platform:
        platform_dir = customer_dir / platform
        # Check for platform/raw subdirectory structure first
        platform_raw_dir = platform_dir / "raw"
        if platform_raw_dir.exists():
            return platform_raw_dir
        # If no raw subdirectory, return platform directory
        return platform_dir

    # Legacy: Check for meta/raw subdirectory structure (backward compatibility)
    meta_raw_dir = customer_dir / "meta" / "raw"
    if meta_raw_dir.exists():
        return meta_raw_dir
    # Check if raw subdirectory exists (for reorganized structure)
    raw_dir = customer_dir / "raw"
    if raw_dir.exists():
        return raw_dir
    return customer_dir


def get_customer_results_dir(customer: Optional[str] = None) -> Path:
    """
    Get the results directory path for a customer.

    Args:
        customer: Customer name. Required parameter.

    Returns:
        Path to customer-specific results directory

    Raises:
        ValueError: If customer is None, empty string, or "all"
    """
    if not customer or customer == "all":
        raise ValueError("customer parameter is required and cannot be None or 'all'")

    # Use PathManager if available
    path_manager = _get_path_manager(customer, "meta")
    if path_manager:
        return path_manager.results_base / customer

    # Legacy fallback
    base_dir = Path("results")
    return base_dir / customer


def get_customer_ad_features_path(
    customer: Optional[str] = None, platform: Optional[str] = None
) -> Path:
    """
    Get the path to ad_features.csv for a customer.

    Args:
        customer: Customer name. Required parameter.
        platform: Platform name (e.g., 'meta', 'google'). If provided, data will be
            looked up in datasets/{customer}/{platform}/features/ or datasets/{customer}/{platform}/.

    Returns:
        Path to customer-specific ad_features.csv (in features subdirectory if exists)

    Raises:
        ValueError: If customer is None, empty string, or "all"
    """
    if not customer or customer == "all":
        raise ValueError("customer parameter is required and cannot be None or 'all'")

    # Use PathManager if available
    path_manager = _get_path_manager(customer, platform or "meta")
    if path_manager:
        # Try features directory first
        features_path = (
            path_manager.features_dir(customer, platform or "meta") / "ad_features.csv"
        )
        if features_path.exists():
            return features_path
        # Fall back to data directory
        return (
            path_manager.raw_data_dir(customer, platform or "meta") / "ad_features.csv"
        )

    # Legacy fallback
    base_dir = Path("datasets")
    customer_dir = base_dir / customer

    # If platform is specified, use platform-specific paths
    if platform:
        platform_dir = customer_dir / platform
        # Check if features subdirectory exists
        platform_features_dir = platform_dir / "features"
        if platform_features_dir.exists():
            return platform_features_dir / "ad_features.csv"
        return platform_dir / "ad_features.csv"

    # Legacy: Check if features subdirectory exists (for reorganized structure)
    features_dir = customer_dir / "features"
    if features_dir.exists():
        return features_dir / "ad_features.csv"
    return customer_dir / "ad_features.csv"


def get_customer_adset_features_path(  # pylint: disable=invalid-name
    customer: Optional[str] = None,
    platform: Optional[str] = None,
) -> Path:
    """
    Get the path to adset_features.csv for a customer.

    Args:
        customer: Customer name. Required parameter.
        platform: Platform name (e.g., 'meta', 'google'). If provided, data will be
            looked up in datasets/{customer}/{platform}/features/ or datasets/{customer}/{platform}/.

    Returns:
        Path to customer-specific adset_features.csv (in features subdirectory if exists)

    Raises:
        ValueError: If customer is None, empty string, or "all"
    """
    if not customer or customer == "all":
        raise ValueError("customer parameter is required and cannot be None or 'all'")

    # Use PathManager if available
    path_manager = _get_path_manager(customer, platform or "meta")
    if path_manager:
        # Try features directory first
        features_path = (
            path_manager.features_dir(customer, platform or "meta")
            / "adset_features.csv"
        )
        if features_path.exists():
            return features_path
        # Fall back to data directory
        return (
            path_manager.raw_data_dir(customer, platform or "meta")
            / "adset_features.csv"
        )

    # Legacy fallback
    base_dir = Path("datasets")
    customer_dir = base_dir / customer

    # If platform is specified, use platform-specific paths
    if platform:
        platform_dir = customer_dir / platform
        # Check if features subdirectory exists
        platform_features_dir = platform_dir / "features"
        if platform_features_dir.exists():
            return platform_features_dir / "adset_features.csv"
        return platform_dir / "adset_features.csv"

    # Legacy: Check if features subdirectory exists (for reorganized structure)
    features_dir = customer_dir / "features"
    if features_dir.exists():
        return features_dir / "adset_features.csv"
    return customer_dir / "adset_features.csv"


def get_customer_allocations_path(
    customer: Optional[str] = None,
    filename: str = "adset_budget_allocations.csv",
    platform: Optional[str] = None,
) -> Path:
    """
    Get the path to allocation results for a customer.

    Args:
        customer: Customer name. Required parameter.
        filename: Name of the output file.
        platform: Platform name (e.g., 'meta', 'google'). If provided, results will be
            saved in results/{customer}/{platform}/rules/.

    Returns:
        Path to customer-specific allocation results

    Raises:
        ValueError: If customer is None, empty string, or "all"
    """
    if not customer or customer == "all":
        raise ValueError("customer parameter is required and cannot be None or 'all'")

    # Use PathManager if available
    path_manager = _get_path_manager(customer, platform or "meta")
    if path_manager:
        return path_manager.allocations_path(
            customer, platform or "meta", filename=filename
        )

    # Legacy fallback
    results_dir = get_customer_results_dir(customer)

    # If platform is specified, include platform in path
    if platform:
        return results_dir / platform / "rules" / filename

    return results_dir / "rules" / filename


def ensure_customer_dirs(
    customer: Optional[str] = None, platform: Optional[str] = None
) -> None:
    """
    Ensure customer-specific directories exist.

    Args:
        customer: Customer name. Required parameter for creating customer directories.
        platform: Platform name (e.g., 'meta', 'google'). If provided, directories will be
            created in datasets/{customer}/{platform}/. Defaults to "meta" if None.

    Raises:
        ValueError: If customer is None, empty string, or "all"
    """
    if not customer or customer == "all":
        raise ValueError("customer parameter is required and cannot be None or 'all'")

    # Default platform to "meta" if not specified
    if platform is None:
        platform = "meta"

    # Always use legacy logic for ensure_customer_dirs
    # This ensures directories are created in the current working directory
    # (important for tests running in tmp_path)
    data_dir = get_customer_data_dir(customer, platform)
    results_dir = get_customer_results_dir(customer)

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure rules directory exists
    # Platform-specific rules directory
    platform_rules_dir = results_dir / platform / "rules"
    platform_rules_dir.mkdir(parents=True, exist_ok=True)

    # Ensure features directory exists
    base_dir = Path("datasets")
    customer_dir = base_dir / customer

    # Platform-specific features directory
    platform_dir = customer_dir / platform
    platform_features_dir = platform_dir / "features"
    platform_features_dir.mkdir(parents=True, exist_ok=True)


def get_all_customers(config_path: Optional[str] = None) -> List[str]:
    """
    Get all customer names from config directory.

    With new config structure (config/adset/allocator/{customer}/{platform}/rules.yaml),
    scans the config directory to discover all customers.

    For backward compatibility, if config_path points to an existing file
    with nested customer structure, parses it for customer names.

    Args:
        config_path: Optional path to config file. If None, scans config/ directory.

    Returns:
        List of customer names found.
        Returns empty list if none found or on error.
    """
    # New structure: scan config/ directory
    if config_path is None:
        try:
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

            if not config_dir.exists():
                return []

            customers = []
            for customer_path in config_dir.iterdir():
                # Skip non-directories
                if not customer_path.is_dir():
                    continue
                # Skip hidden directories
                if customer_path.name.startswith("."):
                    continue
                # Check if it has platform subdirectories with rules.yaml
                has_platforms = False
                for platform_path in customer_path.iterdir():
                    if platform_path.is_dir():
                        rules_file = platform_path / "rules.yaml"
                        if rules_file.exists():
                            has_platforms = True
                            break
                if has_platforms:
                    customers.append(customer_path.name)

            return sorted(customers)
        except (OSError, IOError) as err:
            logger.error("Error scanning config directory: %s", err)
            return []

    # Backward compatibility: parse nested structure from file
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file) or {}
        # Filter out keys that are not customer configs
        customers = []
        for key, value in config_data.items():
            if isinstance(value, dict) and any(
                section in value
                for section in [
                    "safety_rules",
                    "decision_rules",
                    "advanced_concepts",
                ]
            ):
                customers.append(key)
        return customers
    except FileNotFoundError:
        return []
    except (yaml.YAMLError, IOError, OSError) as err:
        logger.error("Error reading config file: %s", err)
        return []
