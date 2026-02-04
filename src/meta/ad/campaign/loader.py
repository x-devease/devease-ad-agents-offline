"""
Campaign Configuration Loader.

Utilities for loading and working with campaign configurations from
consolidated customer config files.

Author: Ad System
Date: 2026-01-30
Version: 3.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from .schemas import Campaign, CampaignsConfig, CampaignStatus, CampaignObjective

logger = logging.getLogger(__name__)


class CampaignLoader:
    """
    Load campaign configurations from consolidated customer config.

    Usage:
        loader = CampaignLoader()
        campaigns = loader.load_campaigns(
            customer="moprobo",
            platform="meta"
        )

        # Get specific campaign
        campaign = loader.get_campaign("ps_launch_q1_2026")

        # Get campaigns by status
        active_campaigns = loader.get_campaigns_by_status(CampaignStatus.ACTIVE)

        # Get campaigns by objective
        conversion_campaigns = loader.get_campaigns_by_objective(CampaignObjective.CONVERSION)
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
    ):
        """
        Initialize campaign loader.

        Args:
            config_dir: Base config directory (defaults to "config")
        """
        self.config_dir = config_dir or Path("config")

        # Cache loaded campaigns
        self._campaigns_cache: Dict[str, CampaignsConfig] = {}

    def load_campaigns(
        self,
        customer: str,
        platform: str,
        force_reload: bool = False,
    ) -> List[Campaign]:
        """
        Load all campaigns from customer config.

        Args:
            customer: Customer name
            platform: Platform name
            force_reload: Force reload from disk

        Returns:
            List of Campaign objects
        """
        config_path = self.config_dir / customer / platform / "config.yaml"

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return []

        # Check cache
        cache_key = f"{customer}/{platform}"
        if not force_reload and cache_key in self._campaigns_cache:
            return self._campaigns_cache[cache_key].campaigns

        # Load config file
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Parse campaigns section
        campaigns_config = CampaignsConfig.from_dict(config_data)

        # Cache result
        self._campaigns_cache[cache_key] = campaigns_config

        logger.info(
            f"Loaded {len(campaigns_config.campaigns)} campaigns "
            f"from {customer}/{platform}"
        )

        return campaigns_config.campaigns

    def get_campaign(
        self,
        customer: str,
        platform: str,
        campaign_id: str,
    ) -> Optional[Campaign]:
        """
        Get specific campaign by ID.

        Args:
            customer: Customer name
            platform: Platform name
            campaign_id: Campaign ID

        Returns:
            Campaign if found, None otherwise
        """
        campaigns = self.load_campaigns(customer, platform)

        for campaign in campaigns:
            if campaign.campaign_id == campaign_id:
                return campaign

        return None

    def get_campaigns_by_status(
        self,
        customer: str,
        platform: str,
        status: CampaignStatus,
    ) -> List[Campaign]:
        """
        Get campaigns filtered by status.

        Args:
            customer: Customer name
            platform: Platform name
            status: Campaign status to filter by

        Returns:
            List of matching Campaigns
        """
        campaigns = self.load_campaigns(customer, platform)

        return [c for c in campaigns if c.status == status]

    def get_campaigns_by_objective(
        self,
        customer: str,
        platform: str,
        objective: CampaignObjective,
    ) -> List[Campaign]:
        """
        Get campaigns filtered by objective.

        Args:
            customer: Customer name
            platform: Platform name
            objective: Campaign objective to filter by

        Returns:
            List of matching Campaigns
        """
        campaigns = self.load_campaigns(customer, platform)

        return [
            c for c in campaigns
            if c.metadata and c.metadata.objective == objective
        ]

    def get_ad_sets(
        self,
        customer: str,
        platform: str,
        campaign_id: str,
    ) -> List[Any]:
        """
        Get all ad sets for a specific campaign.

        Args:
            customer: Customer name
            platform: Platform name
            campaign_id: Campaign ID

        Returns:
            List of AdSet objects
        """
        campaign = self.get_campaign(customer, platform, campaign_id)

        if not campaign or not campaign.ad_sets:
            return []

        return campaign.ad_sets

    def get_active_products(
        self,
        customer: str,
        platform: str,
        campaign_id: Optional[str] = None,
    ) -> List[str]:
        """
        Get all active product IDs from campaigns or specific campaign.

        Args:
            customer: Customer name
            platform: Platform name
            campaign_id: Optional campaign ID to filter by

        Returns:
            List of product IDs
        """
        if campaign_id:
            campaigns = [self.get_campaign(customer, platform, campaign_id)]
        else:
            campaigns = self.load_campaigns(customer, platform)

        products = set()
        for campaign in campaigns:
            if campaign.ad_sets:
                for ad_set in campaign.ad_sets:
                    if ad_set.products:
                        for product_ref in ad_set.products:
                            products.add(product_ref.product_id)

        return sorted(list(products))


# ==========================================
# Convenience Functions
# ==========================================

def load_campaigns(
    customer: str,
    platform: str,
    config_dir: Optional[Path] = None,
) -> List[Campaign]:
    """
    Convenience function to load all campaigns for a customer/platform.

    Args:
        customer: Customer name
        platform: Platform name
        config_dir: Optional custom config directory

    Returns:
        List of Campaign objects
    """
    loader = CampaignLoader(config_dir=config_dir)
    return loader.load_campaigns(customer, platform)


def get_campaign(
    customer: str,
    platform: str,
    campaign_id: str,
    config_dir: Optional[Path] = None,
) -> Optional[Campaign]:
    """
    Convenience function to get a specific campaign.

    Args:
        customer: Customer name
        platform: Platform name
        campaign_id: Campaign ID
        config_dir: Optional custom config directory

    Returns:
        Campaign if found, None otherwise
    """
    loader = CampaignLoader(config_dir=config_dir)
    return loader.get_campaign(customer, platform, campaign_id)
