"""
Unit tests for Campaign Configuration.

Tests campaign schema parsing, validation, and loading.
"""

import pytest
from pathlib import Path
import yaml

from src.meta.ad.campaign import (
    Campaign,
    CampaignLoader,
    CampaignStatus,
    CampaignObjective,
    load_campaigns,
    get_campaign,
    # Schemas
    AgeRange,
    Location,
    CustomAudience,
    Budget,
    BiddingConfig,
    BiddingStrategy,
)


@pytest.fixture
def sample_campaign_config(tmp_path):
    """Create sample campaign config file."""
    config_data = {
        "campaigns": [
            {
                "campaign_id": "test_campaign_001",
                "campaign_name": "Test Campaign Q1 2026",
                "status": "active",

                "metadata": {
                    "objective": "conversion",
                    "goal": "Drive test conversions",
                    "start_date": "2026-01-01",
                    "end_date": "2026-03-31",
                    "created_at": "2025-12-01",
                    "updated_at": "2026-01-01",
                },

                "target_audience": {
                    "demographics": {
                        "age_range": {
                            "min": 25,
                            "max": 55
                        },
                        "genders": ["male", "female"],
                        "locations": [
                            {
                                "country": "US",
                                "regions": ["California", "Texas"]
                            }
                        ]
                    },
                    "interests": [
                        "Technology",
                        "Camping",
                        "Outdoor activities"
                    ],
                    "custom_audiences": [
                        {
                            "name": "Website visitors",
                            "type": "website_traffic",
                            "retention_days": 30
                        }
                    ]
                },

                "budget": {
                    "daily_budget": 100.00,
                    "lifetime_budget": 9000.00,
                    "bidding": {
                        "strategy": "lowest_cost_with_min_roas",
                        "roas_target": 3.5
                    },
                    "pacing": {
                        "type": "standard",
                        "spend_equally": True
                    }
                },

                "ad_sets": [
                    {
                        "ad_set_id": "ad_set_001",
                        "ad_set_name": "Test Ad Set 1",
                        "status": "active",
                        "budget": {
                            "daily_budget": 50.00
                        },
                        "schedule": {
                            "start_time": "2026-01-01T09:00:00",
                            "end_time": "2026-03-31T23:59:59",
                            "days_of_week": ["monday", "wednesday", "friday"]
                        },
                        "products": [
                            {
                                "product_id": "prod_001",
                                "product_name": "Test Product 1",
                                "priority": "high"
                            }
                        ]
                    }
                ],

                "creative_settings": {
                    "primary_psychology": "trust",
                    "fallback_psychology": "social_proof",
                    "brand": {
                        "brand_colors": {
                            "primary": "#003366",
                            "secondary": "#1E8449",
                            "accent": "#F39C12"
                        }
                    },
                    "content_templates": {
                        "headlines": [
                            "Test Headline 1",
                            "Test Headline 2"
                        ]
                    },
                    "negative_prompts": [
                        "cartoon",
                        "blurry"
                    ]
                },

                "goals": {
                    "primary_goal": "conversion",
                    "secondary_goals": ["traffic", "engagement"],
                    "kpi_targets": {
                        "conversion_rate": {
                            "target": 0.035,
                            "minimum": 0.025
                        },
                        "click_through_rate": {
                            "target": 0.025,
                            "minimum": 0.015
                        },
                        "return_on_ad_spend": {
                            "target": 3.5,
                            "minimum": 2.5
                        },
                        "cost_per_conversion": {
                            "target": 25.00,
                            "maximum": 35.00
                        }
                    }
                }
            }
        ]
    }

    # Create customer directory structure
    customer_dir = tmp_path / "config" / "test_customer" / "facebook"
    customer_dir.mkdir(parents=True)

    config_path = customer_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


class TestCampaignSchemas:
    """Test campaign schema parsing."""

    def test_campaign_from_dict(self):
        """Test creating Campaign from dict."""
        data = {
            "campaign_id": "test_001",
            "campaign_name": "Test Campaign",
            "status": "active",
            "metadata": {
                "objective": "conversion",
                "goal": "Test goal",
                "start_date": "2026-01-01"
            }
        }

        campaign = Campaign.from_dict(data)

        assert campaign.campaign_id == "test_001"
        assert campaign.campaign_name == "Test Campaign"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.metadata.objective == CampaignObjective.CONVERSION

    def test_age_range_schema(self):
        """Test AgeRange parsing."""
        data = {"min": 25, "max": 55}
        age_range = AgeRange(**data)

        assert age_range.min == 25
        assert age_range.max == 55

    def test_location_schema(self):
        """Test Location parsing."""
        data = {
            "country": "US",
            "regions": ["California", "Texas"]
        }
        location = Location(**data)

        assert location.country == "US"
        assert location.regions == ["California", "Texas"]

    def test_custom_audience_schema(self):
        """Test CustomAudience parsing."""
        data = {
            "name": "Website visitors",
            "type": "website_traffic",
            "retention_days": 30
        }
        audience = CustomAudience(**data)

        assert audience.name == "Website visitors"
        assert audience.type == "website_traffic"
        assert audience.retention_days == 30

    def test_budget_schema(self):
        """Test Budget parsing with bidding."""
        data = {
            "daily_budget": 100.00,
            "bidding": {
                "strategy": "lowest_cost_with_min_roas",
                "roas_target": 3.5
            }
        }

        budget = Budget(
            daily_budget=100.00,
            bidding=BiddingConfig(
                strategy=BiddingStrategy.LOWEST_COST_WITH_MIN_ROAS,
                roas_target=3.5
            )
        )

        assert budget.daily_budget == 100.00
        assert budget.bidding.strategy == BiddingStrategy.LOWEST_COST_WITH_MIN_ROAS


class TestCampaignLoader:
    """Test CampaignLoader functionality."""

    def test_load_campaigns(self, sample_campaign_config):
        """Test loading campaigns from config."""
        loader = CampaignLoader(config_dir=sample_campaign_config.parent.parent.parent)

        campaigns = loader.load_campaigns("test_customer", "facebook")

        assert len(campaigns) == 1
        assert campaigns[0].campaign_id == "test_campaign_001"
        assert campaigns[0].campaign_name == "Test Campaign Q1 2026"

    def test_get_campaign(self, sample_campaign_config):
        """Test getting specific campaign by ID."""
        loader = CampaignLoader(config_dir=sample_campaign_config.parent.parent.parent)

        campaign = loader.get_campaign(
            "test_customer",
            "facebook",
            "test_campaign_001"
        )

        assert campaign is not None
        assert campaign.campaign_id == "test_campaign_001"

    def test_get_campaign_not_found(self, sample_campaign_config):
        """Test getting non-existent campaign returns None."""
        loader = CampaignLoader(config_dir=sample_campaign_config.parent.parent.parent)

        campaign = loader.get_campaign(
            "test_customer",
            "facebook",
            "nonexistent"
        )

        assert campaign is None

    def test_get_campaigns_by_status(self, sample_campaign_config):
        """Test filtering campaigns by status."""
        loader = CampaignLoader(config_dir=sample_campaign_config.parent.parent.parent)

        active_campaigns = loader.get_campaigns_by_status(
            "test_customer",
            "facebook",
            CampaignStatus.ACTIVE
        )

        assert len(active_campaigns) == 1
        assert active_campaigns[0].status == CampaignStatus.ACTIVE

    def test_get_campaigns_by_objective(self, sample_campaign_config):
        """Test filtering campaigns by objective."""
        loader = CampaignLoader(config_dir=sample_campaign_config.parent.parent.parent)

        conversion_campaigns = loader.get_campaigns_by_objective(
            "test_customer",
            "facebook",
            CampaignObjective.CONVERSION
        )

        assert len(conversion_campaigns) == 1
        assert conversion_campaigns[0].metadata.objective == CampaignObjective.CONVERSION

    def test_get_ad_sets(self, sample_campaign_config):
        """Test getting ad sets for a campaign."""
        loader = CampaignLoader(config_dir=sample_campaign_config.parent.parent.parent)

        ad_sets = loader.get_ad_sets(
            "test_customer",
            "facebook",
            "test_campaign_001"
        )

        assert len(ad_sets) == 1
        assert ad_sets[0].ad_set_id == "ad_set_001"

    def test_get_active_products(self, sample_campaign_config):
        """Test getting active products from campaigns."""
        loader = CampaignLoader(config_dir=sample_campaign_config.parent.parent.parent)

        products = loader.get_active_products("test_customer", "facebook")

        assert "prod_001" in products

    def test_get_active_products_from_campaign(self, sample_campaign_config):
        """Test getting products from specific campaign."""
        loader = CampaignLoader(config_dir=sample_campaign_config.parent.parent.parent)

        products = loader.get_active_products(
            "test_customer",
            "facebook",
            "test_campaign_001"
        )

        assert "prod_001" in products


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_campaigns_function(self, sample_campaign_config):
        """Test load_campaigns convenience function."""
        campaigns = load_campaigns(
            "test_customer",
            "facebook",
            config_dir=sample_campaign_config.parent.parent.parent
        )

        assert len(campaigns) == 1
        assert campaigns[0].campaign_id == "test_campaign_001"

    def test_get_campaign_function(self, sample_campaign_config):
        """Test get_campaign convenience function."""
        campaign = get_campaign(
            "test_customer",
            "facebook",
            "test_campaign_001",
            config_dir=sample_campaign_config.parent.parent.parent
        )

        assert campaign is not None
        assert campaign.campaign_id == "test_campaign_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
