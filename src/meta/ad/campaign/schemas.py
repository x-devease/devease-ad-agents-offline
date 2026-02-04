"""
Campaign Configuration Schemas.

Dataclass definitions for campaign configuration including:
- Campaign metadata
- Target audience (demographics, interests, behaviors)
- Budget & bidding strategy
- Ad sets & ad groups
- Creative settings
- Goals & KPIs

Author: Ad System
Date: 2026-01-30
Version: 3.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from datetime import datetime


class CampaignStatus(str, Enum):
    """Campaign status types."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    SCHEDULED = "scheduled"


class CampaignObjective(str, Enum):
    """Campaign objective types."""
    AWARENESS = "awareness"
    TRAFFIC = "traffic"
    ENGAGEMENT = "engagement"
    LEADS = "leads"
    CONVERSION = "conversion"
    APP_INSTALLATION = "app_installation"
    STORE_VISITS = "store_visits"


class BiddingStrategy(str, Enum):
    """Bidding strategy types."""
    LOWEST_COST_WITH_MIN_ROAS = "lowest_cost_with_min_roas"
    LOWEST_COST_WITHOUT_CAP = "lowest_cost_without_cap"
    TARGET_COST = "target_cost"
    COST_CAP = "cost_cap"
    BID_CAP = "bid_cap"


# ==========================================
# Demographics & Targeting
# ==========================================

@dataclass
class AgeRange:
    """Age range targeting."""
    min: int
    max: int


@dataclass
class Location:
    """Geographic location targeting."""
    country: str
    regions: Optional[List[str]] = None
    cities: Optional[List[str]] = None


@dataclass
class Demographics:
    """Demographic targeting criteria."""
    age_range: Optional[AgeRange] = None
    genders: Optional[List[str]] = None
    locations: Optional[List[Location]] = None
    languages: Optional[List[str]] = None


@dataclass
class CustomAudience:
    """Custom audience definition."""
    name: str
    type: str  # website_traffic, email_list, lookalike, etc.
    retention_days: Optional[int] = None
    source_audience: Optional[str] = None  # For lookalike audiences


@dataclass
class TargetAudience:
    """Target audience configuration."""
    # Demographics
    demographics: Optional[Demographics] = None

    # Interests & behaviors
    interests: Optional[List[str]] = None
    behaviors: Optional[List[str]] = None

    # Custom audiences
    custom_audiences: Optional[List[CustomAudience]] = None

    # Exclusions
    excluded_audiences: Optional[List[str]] = None


# ==========================================
# Budget & Bidding
# ==========================================

@dataclass
class BiddingConfig:
    """Bidding strategy configuration."""
    strategy: BiddingStrategy
    roas_target: Optional[float] = None
    bid_cap: Optional[float] = None
    cost_cap: Optional[float] = None
    target_cost: Optional[float] = None


@dataclass
class BudgetPacing:
    """Budget pacing configuration."""
    type: str = "standard"  # standard, accelerated, no_pacing
    spend_equally: bool = True


@dataclass
class Budget:
    """Budget configuration."""
    daily_budget: float
    lifetime_budget: Optional[float] = None
    bidding: Optional[BiddingConfig] = None
    pacing: Optional[BudgetPacing] = None


# ==========================================
# Schedule
# ==========================================

@dataclass
class Schedule:
    """Ad set schedule configuration."""
    start_time: str  # ISO 8601 format
    end_time: Optional[str] = None
    days_of_week: Optional[List[str]] = None
    hours: Optional[List[int]] = None
    run_continuously: Optional[bool] = None
    timezone: Optional[str] = None


# ==========================================
# Products
# ==========================================

@dataclass
class ProductReference:
    """Product reference in ad set."""
    product_id: str
    product_name: str
    priority: str = "medium"  # low, medium, high


# ==========================================
# Ad Sets
# ==========================================

@dataclass
class AdSet:
    """Ad set (Ad Group) configuration."""
    ad_set_id: str
    ad_set_name: str
    status: CampaignStatus = CampaignStatus.ACTIVE

    # Targeting (can override campaign-level)
    target_audience: Optional[TargetAudience] = None

    # Budget
    budget: Optional[Budget] = None

    # Schedule
    schedule: Optional[Schedule] = None

    # Products to feature
    products: Optional[List[ProductReference]] = None


# ==========================================
# Creative Settings
# ==========================================

@dataclass
class BrandColors:
    """Brand color palette."""
    primary: str
    secondary: str
    accent: str


@dataclass
class LogoSettings:
    """Logo placement settings."""
    path: str
    placement: str = "top_left"  # top_left, top_right, bottom_left, etc.
    size: str = "small"  # small, medium, large


@dataclass
class BrandSettings:
    """Brand guidelines."""
    brand_colors: Optional[BrandColors] = None
    logo: Optional[LogoSettings] = None
    tone: Optional[str] = None  # professional_authoritative, friendly, etc.
    voice: Optional[str] = None  # expert_reassuring, casual, etc.


@dataclass
class ContentTemplates:
    """Content templates for creative generation."""
    headlines: Optional[List[str]] = None
    primary_text: Optional[List[str]] = None
    descriptions: Optional[List[str]] = None


@dataclass
class CreativeSettings:
    """Campaign-level creative settings."""
    primary_psychology: str = "trust"
    fallback_psychology: str = "social_proof"

    brand: Optional[BrandSettings] = None
    content_templates: Optional[ContentTemplates] = None
    negative_prompts: Optional[List[str]] = None


# ==========================================
# Goals & KPIs
# ==========================================

@dataclass
class ConversionRateTargets:
    """Conversion rate KPI targets."""
    target: float  # e.g., 0.035 for 3.5%
    minimum: float


@dataclass
class ClickThroughRateTargets:
    """Click-through rate KPI targets."""
    target: float  # e.g., 0.025 for 2.5%
    minimum: float


@dataclass
class ROASTargets:
    """Return on ad spend KPI targets."""
    target: float  # e.g., 3.5x
    minimum: float


@dataclass
class CostPerConversionTargets:
    """Cost per conversion targets."""
    target: float  # e.g., 25.00 USD
    maximum: float


@dataclass
class KPITargets:
    """KPI targets for campaign."""
    conversion_rate: Optional[ConversionRateTargets] = None
    click_through_rate: Optional[ClickThroughRateTargets] = None
    return_on_ad_spend: Optional[ROASTargets] = None
    cost_per_conversion: Optional[CostPerConversionTargets] = None


@dataclass
class CampaignGoals:
    """Campaign goals and KPIs."""
    primary_goal: CampaignObjective
    secondary_goals: List[CampaignObjective] = field(default_factory=list)
    kpi_targets: Optional[KPITargets] = None


# ==========================================
# Campaign Metadata
# ==========================================

@dataclass
class CampaignMetadata:
    """Campaign metadata."""
    objective: CampaignObjective
    goal: str
    start_date: str  # ISO 8601 date
    end_date: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ==========================================
# Full Campaign Configuration
# ==========================================

@dataclass
class Campaign:
    """
    Complete campaign configuration.

    This represents a full campaign with all its settings including:
    - Metadata (name, objective, dates)
    - Target audience (demographics, interests, behaviors)
    - Budget & bidding
    - Ad sets (ad groups)
    - Creative settings
    - Goals & KPIs
    """
    campaign_id: str
    campaign_name: str
    status: CampaignStatus = CampaignStatus.ACTIVE

    # Metadata
    metadata: Optional[CampaignMetadata] = None

    # Target audience
    target_audience: Optional[TargetAudience] = None

    # Budget
    budget: Optional[Budget] = None

    # Ad sets
    ad_sets: Optional[List[AdSet]] = None

    # Creative settings
    creative_settings: Optional[CreativeSettings] = None

    # Goals
    goals: Optional[CampaignGoals] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
        """Create Campaign from dictionary."""
        # Handle nested objects
        if data.get("metadata"):
            metadata = CampaignMetadata(**data["metadata"])
        else:
            metadata = None

        if data.get("target_audience"):
            audience_data = data["target_audience"]

            # Parse demographics
            demographics = None
            if audience_data.get("demographics"):
                demo_data = audience_data["demographics"]
                if demo_data.get("age_range"):
                    demo_data["age_range"] = AgeRange(**demo_data["age_range"])
                if demo_data.get("locations"):
                    demo_data["locations"] = [
                        Location(**loc) for loc in demo_data["locations"]
                    ]
                demographics = Demographics(**demo_data)

            # Parse custom audiences
            custom_audiences = None
            if audience_data.get("custom_audiences"):
                custom_audiences = [
                    CustomAudience(**aud) for aud in audience_data["custom_audiences"]
                ]

            target_audience = TargetAudience(
                demographics=demographics,
                interests=audience_data.get("interests"),
                behaviors=audience_data.get("behaviors"),
                custom_audiences=custom_audiences,
            )
        else:
            target_audience = None

        # Parse budget
        if data.get("budget"):
            budget_data = data["budget"]

            # Parse bidding config
            bidding = None
            if budget_data.get("bidding"):
                bid_data = budget_data["bidding"]
                # Convert string to enum if needed
                if isinstance(bid_data.get("strategy"), str):
                    bid_data["strategy"] = BiddingStrategy(bid_data["strategy"])
                bidding = BiddingConfig(**bid_data)

            # Parse pacing
            pacing = None
            if budget_data.get("pacing"):
                pacing = BudgetPacing(**budget_data["pacing"])

            budget = Budget(
                daily_budget=budget_data["daily_budget"],
                lifetime_budget=budget_data.get("lifetime_budget"),
                bidding=bidding,
                pacing=pacing,
            )
        else:
            budget = None

        # Parse ad sets
        ad_sets = None
        if data.get("ad_sets"):
            ad_sets = []
            for ad_set_data in data["ad_sets"]:
                # Convert status string to enum
                if isinstance(ad_set_data.get("status"), str):
                    ad_set_data["status"] = CampaignStatus(ad_set_data["status"])

                # Parse schedule if present
                schedule = None
                if ad_set_data.get("schedule"):
                    schedule = Schedule(**ad_set_data["schedule"])

                # Parse products
                products = None
                if ad_set_data.get("products"):
                    products = [ProductReference(**p) for p in ad_set_data["products"]]

                # Parse budget
                ad_set_budget = None
                if ad_set_data.get("budget"):
                    bid_data = ad_set_data["budget"]
                    if isinstance(bid_data.get("bidding", {}).get("strategy"), str):
                        bid_data["bidding"]["strategy"] = BiddingStrategy(
                            bid_data["bidding"]["strategy"]
                        )
                    ad_set_budget = Budget(**bid_data)

                ad_sets.append(
                    AdSet(
                        ad_set_id=ad_set_data["ad_set_id"],
                        ad_set_name=ad_set_data["ad_set_name"],
                        status=ad_set_data["status"],
                        schedule=schedule,
                        products=products,
                        budget=ad_set_budget,
                    )
                )

        # Parse goals
        if data.get("goals"):
            goals_data = data["goals"]

            # Convert objective strings to enums
            primary_goal = CampaignObjective(goals_data["primary_goal"])
            secondary_goals = [
                CampaignObjective(g) for g in goals_data.get("secondary_goals", [])
            ]

            # Parse KPI targets
            kpi_targets = None
            if goals_data.get("kpi_targets"):
                kpi_data = goals_data["kpi_targets"]

                # Parse nested KPI objects
                if kpi_data.get("conversion_rate"):
                    kpi_data["conversion_rate"] = ConversionRateTargets(**kpi_data["conversion_rate"])
                if kpi_data.get("click_through_rate"):
                    kpi_data["click_through_rate"] = ClickThroughRateTargets(**kpi_data["click_through_rate"])
                if kpi_data.get("return_on_ad_spend"):
                    kpi_data["return_on_ad_spend"] = ROASTargets(**kpi_data["return_on_ad_spend"])
                if kpi_data.get("cost_per_conversion"):
                    kpi_data["cost_per_conversion"] = CostPerConversionTargets(**kpi_data["cost_per_conversion"])

                kpi_targets = KPITargets(**kpi_data)

            goals = CampaignGoals(
                primary_goal=primary_goal,
                secondary_goals=secondary_goals,
                kpi_targets=kpi_targets,
            )
        else:
            goals = None

        return cls(
            campaign_id=data["campaign_id"],
            campaign_name=data["campaign_name"],
            status=CampaignStatus(data.get("status", "active")),
            metadata=metadata,
            target_audience=target_audience,
            budget=budget,
            ad_sets=ad_sets,
            creative_settings=data.get("creative_settings"),  # TODO: Parse fully
            goals=goals,
        )


# ==========================================
# Campaigns List Wrapper
# ==========================================

@dataclass
class CampaignsConfig:
    """Wrapper for campaigns list configuration."""
    campaigns: List[Campaign] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CampaignsConfig":
        """Create CampaignsConfig from dictionary."""
        campaigns = []
        if data.get("campaigns"):
            for campaign_data in data["campaigns"]:
                campaigns.append(Campaign.from_dict(campaign_data))

        return cls(campaigns=campaigns)
