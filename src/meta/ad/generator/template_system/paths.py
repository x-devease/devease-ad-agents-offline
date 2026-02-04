"""
Path Management: Shared Config Per Customer/Platform.

This module provides centralized path management for Template-Driven Ad Generator
following the shared config architecture:

Config (Shared Across Miner, Generator, Reviewer):
  config/{customer}/{platform}/
  ├── config.yaml                    # SHARED config (miner + generator + allocator + reviewer)
  ├── patterns.yaml                  # Ad miner output patterns
  └── campaign_content.yaml          # Optional: Campaign-specific content

Results (Per Platform):
  results/{customer}/{platform}/
  ├── ad_generator/
  │   ├── generated/{product}/{date}/
  │   ├── backgrounds/{product}/{date}/
  │   └── composited/{product}/{date}/

Author: Ad System
Date: 2026-01-30
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, List


logger = logging.getLogger(__name__)

# Default directories
DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_OUTPUT_DIR = Path("results")


class GeneratorPaths:
    """
    Path management with single config per customer, platform-specific outputs.

    Example:
        paths = GeneratorPaths(
            customer="moprobo",
            platform="facebook"
        )
        blueprint_path = paths.get_blueprint_path()
        output_dir = paths.get_generated_output("Power Station")
    """

    def __init__(
        self,
        customer: str,
        platform: str,
        config_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize path manager.

        Args:
            customer: Customer/account name (e.g., "moprobo")
            platform: Platform name (facebook, tiktok, instagram)
            config_dir: Base config directory (default: config/ad/)
            output_dir: Base output directory (default: results/)
        """
        self.customer = customer.lower().replace("-", "_")
        self.platform = platform.lower().replace("-", "_")
        self.config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    # ============================================================
    # CONFIG PATHS (Shared Per Customer/Platform)
    # ============================================================

    def get_config_path(self) -> Path:
        """
        Get customer/platform config path (shared across miner, generator, reviewer).

        Returns:
            Path to customer/platform config directory

        Example:
            paths.get_config_path()
            # → config/moprobo/facebook/
        """
        return self.config_dir / self.customer / self.platform

    def get_blueprint_path(self) -> Path:
        """
        Get master blueprint path from shared customer config.

        Returns:
            Path to config.yaml (shared config)

        Example:
            paths.get_blueprint_path()
            # → config/moprobo/facebook/config.yaml

        Note:
            This now points to the shared config.yaml which contains
            settings for miner, generator, and reviewer.
        """
        return self.get_config_path() / "config.yaml"

    def get_campaign_content_path(self, platform_specific: bool = True) -> Path:
        """
        Get campaign content path with platform-specific override support.

        Args:
            platform_specific: If True, check for platform-specific override first

        Returns:
            Path to campaign content YAML

        Example:
            paths.get_campaign_content_path()
            # → config/moprobo/facebook/campaign_content.yaml
        """
        # Note: Platform-specific is now implicit in the path structure
        # Config is already at {customer}/{platform}/ level
        base_path = self.get_config_path() / "campaign_content.yaml"
        logger.debug(f"Using campaign content: {base_path}")
        return base_path

    def get_generator_config_path(self) -> Path:
        """
        Get path to consolidated customer config (contains generator settings).

        The customer config.yaml contains all settings including:
        - psychology_catalog (14 psychology types) - Part P
        - psychology_templates (text overlay templates)
        - generator_settings (model, generation parameters)
        - All other miner, allocator, and reviewer settings

        Returns:
            Path to customer config.yaml

        Example:
            paths.get_generator_config_path()
            # → config/moprobo/meta/config.yaml
        """
        return self.get_config_path() / "config.yaml"

    # ============================================================
    # AD MINER INTEGRATION PATHS
    # ============================================================

    def get_ad_miner_output_path(
        self,
        product: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path to Ad Miner output for this customer/platform.

        Args:
            product: Optional product name
            date: Optional date string (YYYY-MM-DD), defaults to today

        Returns:
            Path to Ad Miner output directory

        Example:
            paths.get_ad_miner_output_path(product="Power Station")
            # → results/moprobo/facebook/ad_miner/
        """
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        base_path = self.output_dir / self.customer / self.platform / "ad_miner"

        if product:
            product_clean = product.lower().replace(" ", "_")
            base_path = base_path / product_clean / date_str

        return base_path

    def get_ad_miner_blueprint_path(
        self,
        product: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path to Ad Miner's master blueprint output.

        Note: This is the OUTPUT from Ad Miner, not the source config.
        The actual config should be read from get_blueprint_path().

        Args:
            product: Optional product name
            date: Optional date string

        Returns:
            Path to Ad Miner's master blueprint output

        Example:
            paths.get_ad_miner_blueprint_path(product="Power Station")
            # → results/moprobo/facebook/ad_miner/Power_Station/2026-01-30/master_blueprint.yaml
        """
        return self.get_ad_miner_output_path(product, date) / "master_blueprint.yaml"

    # ============================================================
    # OUTPUT PATHS (Per Platform)
    # ============================================================

    def get_ad_generator_base_path(self) -> Path:
        """
        Get base path for Ad Generator outputs (platform-specific).

        Returns:
            Path to ad_generator base directory

        Example:
            paths.get_ad_generator_base_path()
            # → results/moprobo/facebook/ad_generator/
        """
        return self.output_dir / self.customer / self.platform / "ad_generator"

    def get_generated_output_path(
        self,
        product: str,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path for generated ad candidates (platform-specific).

        Args:
            product: Product name
            date: Optional date string (YYYY-MM-DD), defaults to today

        Returns:
            Path to generated output directory

        Example:
            paths.get_generated_output_path("Power Station")
            # → results/moprobo/facebook/ad_generator/generated/Power_Station/2026-01-30/
        """
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        product_clean = product.lower().replace(" ", "_")
        return (
            self.get_ad_generator_base_path()
            / "generated"
            / product_clean
            / date_str
        )

    def get_backgrounds_output_path(
        self,
        product: str,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path for generated background images (platform-specific).

        Args:
            product: Product name
            date: Optional date string

        Returns:
            Path to backgrounds output directory

        Example:
            paths.get_backgrounds_output_path("Power Station")
            # → results/moprobo/facebook/ad_generator/backgrounds/Power_Station/2026-01-30/
        """
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        product_clean = product.lower().replace(" ", "_")
        return (
            self.get_ad_generator_base_path()
            / "backgrounds"
            / product_clean
            / date_str
        )

    def get_composited_output_path(
        self,
        product: str,
        date: Optional[str] = None,
    ) -> Path:
        """
        Get path for composited final outputs (platform-specific).

        Args:
            product: Product name
            date: Optional date string

        Returns:
            Path to composited output directory

        Example:
            paths.get_composited_output_path("Power Station")
            # → results/moprobo/facebook/ad_generator/composited/Power_Station/2026-01-30/
        """
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        product_clean = product.lower().replace(" ", "_")
        return (
            self.get_ad_generator_base_path()
            / "composited"
            / product_clean
            / date_str
        )

    # ============================================================
    # ASSET INPUT PATHS
    # ============================================================

    def get_product_input_path(
        self,
        product: str,
        filename: str = "product_raw.png"
    ) -> Path:
        """
        Get path to product input image.

        Args:
            product: Product name
            filename: Product image filename (default: product_raw.png)

        Returns:
            Path to product input image

        Example:
            paths.get_product_input_path("Power Station")
            # → config/ad/moprobo/products/Power_Station/product_raw.png
        """
        product_clean = product.lower().replace(" ", "_")
        return self.get_config_path() / "products" / product_clean / filename

    # ============================================================
    # DIRECTORY MANAGEMENT
    # ============================================================

    def ensure_directories(self, product: Optional[str] = None) -> List[Path]:
        """
        Ensure all necessary directories exist for this customer/platform.

        Args:
            product: Optional product name to create product-specific directories

        Returns:
            List of created/verified directories

        Example:
            paths.ensure_directories("Power Station")
        """
        directories = [
            self.get_config_path(),
            self.get_ad_generator_base_path(),
        ]

        # Add product-specific directories if product provided
        if product:
            directories.extend([
                self.get_generated_output_path(product),
                self.get_backgrounds_output_path(product),
                self.get_composited_output_path(product),
            ])

        created = []
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
            logger.debug("Ensured directory exists: %s", directory)

        logger.info("Ensured %d directories exist for %s/%s", len(created), self.customer, self.platform)
        return created

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GeneratorPaths(customer={self.customer}, platform={self.platform})"
        )


def create_customer_structure(
    customer: str,
    platforms: List[str],
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> dict[str, Path]:
    """
    Create standard directory structure for a new customer.

    Args:
        customer: Customer name
        platforms: List of platforms to create
        config_dir: Base config directory

    Returns:
        Dict with created directory paths

    Example:
        dirs = create_customer_structure("moprobo", ["facebook", "tiktok"])
        # Creates:
        # - config/ad/moprobo/
        # - results/moprobo/facebook/ad_generator/
        # - results/moprobo/tiktok/ad_generator/
    """
    customer_clean = customer.lower().replace("-", "_")
    directories = {}

    # Create customer config directory
    customer_config = config_dir / customer_clean
    customer_config.mkdir(parents=True, exist_ok=True)
    directories["customer_config"] = customer_config
    logger.info("Created customer config directory: %s", customer_config)

    # Create platform-specific output directories
    for platform in platforms:
        platform_clean = platform.lower().replace("-", "_")
        platform_output = DEFAULT_OUTPUT_DIR / customer_clean / platform_clean / "ad_generator"
        platform_output.mkdir(parents=True, exist_ok=True)
        directories[f"{platform}_output"] = platform_output
        logger.info("Created platform output directory: %s", platform_output)

    return directories
