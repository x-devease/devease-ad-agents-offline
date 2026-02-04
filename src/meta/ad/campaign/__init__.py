"""
Campaign Configuration Module.

This module provides comprehensive campaign configuration management
including:
- Campaign metadata (name, goal, dates)
- Target audience (demographics, interests, behaviors)
- Budget & bidding strategy
- Ad sets & ad groups
- Creative settings
- Goals & KPIs

Usage:
    from src.meta.ad.campaign import CampaignLoader, load_campaigns

    # Load all campaigns
    loader = CampaignLoader()
    campaigns = loader.load_campaigns(
        customer="moprobo",
        platform="meta"
    )

    # Get specific campaign
    campaign = loader.get_campaign("ps_launch_q1_2026")

    # Get active campaigns
    active = loader.get_campaigns_by_status(CampaignStatus.ACTIVE)

Author: Ad System
Date: 2026-01-30
Version: 3.0 (Fully consolidated with campaigns)
"""

from .schemas import (
    # Enums
    CampaignStatus,
    CampaignObjective,
    BiddingStrategy,

    # Targeting
    AgeRange,
    Location,
    Demographics,
    CustomAudience,
    TargetAudience,

    # Budget
    BiddingConfig,
    BudgetPacing,
    Budget,

    # Ad Sets
    Schedule,
    ProductReference,
    AdSet,

    # Creative
    BrandColors,
    LogoSettings,
    BrandSettings,
    ContentTemplates,
    CreativeSettings,

    # Goals
    ConversionRateTargets,
    ClickThroughRateTargets,
    ROASTargets,
    CostPerConversionTargets,
    KPITargets,
    CampaignGoals,

    # Metadata
    CampaignMetadata,

    # Main
    Campaign,
    CampaignsConfig,
)

from .loader import (
    CampaignLoader,
    load_campaigns,
    get_campaign,
)

__all__ = [
    # Enums
    "CampaignStatus",
    "CampaignObjective",
    "BiddingStrategy",

    # Targeting
    "AgeRange",
    "Location",
    "Demographics",
    "CustomAudience",
    "TargetAudience",

    # Budget
    "BiddingConfig",
    "BudgetPacing",
    "Budget",

    # Ad Sets
    "Schedule",
    "ProductReference",
    "AdSet",

    # Creative
    "BrandColors",
    "LogoSettings",
    "BrandSettings",
    "ContentTemplates",
    "CreativeSettings",

    # Goals
    "ConversionRateTargets",
    "ClickThroughRateTargets",
    "ROASTargets",
    "CostPerConversionTargets",
    "KPITargets",
    "CampaignGoals",

    # Metadata
    "CampaignMetadata",

    # Main
    "Campaign",
    "CampaignsConfig",

    # Loader
    "CampaignLoader",
    "load_campaigns",
    "get_campaign",
]
