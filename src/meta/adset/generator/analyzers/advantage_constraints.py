"""
Advantage+ Constraint Generator

Generates hard constraints for Meta Audience Controls to prevent Advantage+
from drifting away from Shopify buyer profiles.
"""

import json
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path

from src.meta.adset.generator.analyzers.shopify_analyzer import ShopifyAnalyzer


@dataclass
class AudienceConstraint:
    """Single audience constraint configuration."""

    constraint_type: str  # age_min, geo_exclusion, custom_audience
    value: any
    enforce: bool  # True = hard constraint, False = suggestion only
    reason: str
    compliance_note: str = None


class AdvantageConstraintGenerator:
    """
    Generate hard constraints for Meta Advantage+ campaigns.

    Purpose: Set boundaries to prevent Advantage+ from targeting
    low-value audiences (e.g., 18-year-olds chasing cheap clicks).
    """

    def __init__(self, shopify_data_path: str = None):
        """
        Initialize constraint generator.

        Args:
            shopify_data_path: Path to Shopify orders CSV
        """
        self.shopify_analyzer = ShopifyAnalyzer(shopify_data_path)

    def generate_age_constraint(
        self,
        min_age: int = None,
        business_type: str = "ecommerce",
    ) -> AudienceConstraint:
        """
        Generate minimum age constraint.

        This is a HARD constraint that will be written to Audience Controls
        (not Suggestions).

        Args:
            min_age: Minimum buyer age (auto-detected from Shopify if None)
            business_type: Business category for compliance

        Returns:
            AudienceConstraint for minimum age
        """
        if min_age is None:
            orders_df = self.shopify_analyzer.load_shopify_orders()
            min_age = self.shopify_analyzer.extract_minimum_buyer_age(orders_df)

        # Compliance adjustments for special categories
        if business_type in ["housing", "credit", "employment"]:
            # Special Ad Categories have stricter requirements
            min_age = max(min_age, 18)

        return AudienceConstraint(
            constraint_type="age_min",
            value=min_age,
            enforce=True,  # HARD constraint
            reason=f"Minimum buyer age from Shopify data: {min_age}+",
            compliance_note=None,
        )

    def generate_geo_exclusions(
        self,
        zip_blacklist: List[str] = None,
        business_description: str = "",
    ) -> List[AudienceConstraint]:
        """
        Generate geographic exclusion constraints.

        Args:
            zip_blacklist: List of ZIP codes to exclude (auto-detected if None)
            business_description: Business type for compliance check

        Returns:
            List of AudienceConstraint objects for geo exclusions
        """
        constraints = []

        # Check if special ad category (compliance)
        is_special = self.shopify_analyzer.detect_special_ad_category(
            business_description
        )

        if is_special:
            # Compliance: NO exclusions for special categories
            return [
                AudienceConstraint(
                    constraint_type="geo_exclusion",
                    value=[],
                    enforce=False,
                    reason="Geo exclusions disabled for Special Ad Category compliance",
                    compliance_note=f"Special Category: {self.shopify_analyzer._detect_category_type(business_description)}",
                )
            ]

        # Generate blacklist if not provided
        if zip_blacklist is None:
            orders_df = self.shopify_analyzer.load_shopify_orders()
            zip_blacklist = self.shopify_analyzer.build_zip_blacklist(orders_df)

        if not zip_blacklist:
            return []

        # Meta API format for location exclusions
        # Must be structured as: country + ZIP codes
        constraint = AudienceConstraint(
            constraint_type="geo_exclusion",
            value={
                "countries": ["US"],
                "zip_codes": zip_blacklist,
                "exclusion_type": "zip",  # Exclude by ZIP code
            },
            enforce=True,  # HARD constraint
            reason=f"Exclude {len(zip_blacklist)} low-performing ZIP codes (Shopify analysis)",
            compliance_note=None,
        )

        return [constraint]

    def generate_lal_constraints(
        self,
        business_description: str = "",
    ) -> Dict[str, AudienceConstraint]:
        """
        Generate Lookalike Audience signal layering constraints.

        Three-tier signal strategy:
        1. Gold Seed (LTV): High LTV customers → Advantage+ Prioritized Suggestion
        2. Velocity: Recent high-frequency customers → Test creative seed
        3. Negative: Recent purchasers → Exclusion from targeting

        Args:
            business_description: Business type description

        Returns:
            Dictionary mapping signal type to constraint
        """
        orders_df = self.shopify_analyzer.load_shopify_orders()
        gold_customers, velocity_customers, negative_customers = (
            self.shopify_analyzer.segment_customers_by_ltv(orders_df)
        )

        constraints = {}

        # Gold Seed - LTV: Top 10% by spend
        constraints["gold_seed"] = AudienceConstraint(
            constraint_type="lal_gold_seed",
            value={
                "customer_ids": (
                    gold_customers["customer_id"].tolist()
                    if not gold_customers.empty
                    else []
                ),
                "percentage": 1,  # 1% LAL
                "meta_setting": "prioritized_suggestion",  # Advantage+ specific
                "purpose": "High LTV signal for Advantage+ expansion",
            },
            enforce=False,  # Suggestion (not hard constraint)
            reason=f"Top {len(gold_customers)} high-LTV customers from Shopify",
        )

        # Velocity: Recent high-frequency
        constraints["velocity"] = AudienceConstraint(
            constraint_type="lal_velocity",
            value={
                "customer_ids": (
                    velocity_customers["customer_id"].tolist()
                    if not velocity_customers.empty
                    else []
                ),
                "percentage": 1,
                "purpose": "Test new creative with engaged customers",
            },
            enforce=False,  # Suggestion
            reason=f"Top {len(velocity_customers)} recent purchasers by frequency",
        )

        # Negative: Exclude recent purchasers
        constraints["negative"] = AudienceConstraint(
            constraint_type="exclusion_audience",
            value={
                "customer_ids": (
                    negative_customers["customer_id"].tolist()
                    if not negative_customers.empty
                    else []
                ),
                "exclusion_type": "custom_audience",
            },
            enforce=True,  # HARD constraint
            reason=f"Exclude {len(negative_customers)} recent purchasers (prevent waste)",
        )

        return constraints

    def generate_all_constraints(
        self,
        business_description: str = "",
        business_type: str = "ecommerce",
        output_path: str = None,
    ) -> Dict:
        """
        Generate complete constraint configuration for Advantage+.

        Args:
            business_description: Business description for category detection
            business_type: Business category
            output_path: Optional path to save constraints JSON

        Returns:
            Complete constraint configuration
        """
        # 1. Age constraint (HARD)
        age_constraint = self.generate_age_constraint(business_type=business_type)

        # 2. Geo exclusions (HARD, unless special category)
        geo_constraints = self.generate_geo_exclusions(
            business_description=business_description
        )

        # 3. LAL signal layering (Mixed)
        lal_constraints = self.generate_lal_constraints(
            business_description=business_description
        )

        config = {
            "constraints": {
                "age": age_constraint.__dict__,
                "geo": [c.__dict__ for c in geo_constraints],
                "lal": {
                    key: constraint.__dict__
                    for key, constraint in lal_constraints.items()
                },
            },
            "metadata": {
                "business_type": business_type,
                "is_special_category": any(
                    c.compliance_note is not None for c in geo_constraints
                ),
                "total_constraints": 1 + len(geo_constraints) + len(lal_constraints),
            },
            "usage": {
                "meta_api": "Use these values in Meta Audience Controls API",
                "advantage_plus": "Apply to Advantage+ campaigns",
                "manual": "Or manually set in Ads Manager",
            },
        }

        # Save if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(config, f, indent=2, default=str)

        return config

    def format_for_meta_api(self, constraints: Dict) -> Dict:
        """
        Format constraints for Meta API payload.

        Transforms internal format to Meta Audience Controls API format.

        Args:
            constraints: Constraint configuration from generate_all_constraints()

        Returns:
            Meta API formatted payload
        """
        api_payload = {
            "age_min": constraints["constraints"]["age"]["value"],
            "exclusions": [],
        }

        # Add geo exclusions
        for geo in constraints["constraints"]["geo"]:
            if geo["enforce"] and geo["value"]:
                api_payload["exclusions"].append(
                    {
                        "type": "geo",
                        "geo": {
                            "countries": geo["value"]["countries"],
                            "zip_codes": geo["value"]["zip_codes"],
                        },
                    }
                )

        # Add custom audience exclusions
        if "negative" in constraints["constraints"]["lal"]:
            negative = constraints["constraints"]["lal"]["negative"]
            if negative["enforce"] and negative["value"]["customer_ids"]:
                api_payload["exclusions"].append(
                    {
                        "type": "custom_audience",
                        "custom_audience_ids": negative["value"]["customer_ids"],
                    }
                )

        return api_payload

    def generate_dashboard_commands(self, constraints: Dict) -> Dict[str, str]:
        """
        Generate one-click dashboard commands.

        Three key functions:
        1. Signal Booster: Shopify high-LTV → Meta prioritized signals
        2. Auto-Tightener: Write hard min age + ZIP exclusions to Controls
        3. Drift Prevention: Monitoring alert setup

        Args:
            constraints: Constraint configuration

        Returns:
            Dictionary of command names to descriptions
        """
        gold_count = len(
            constraints["constraints"]["lal"]["gold_seed"]["value"]["customer_ids"]
        )

        # Handle geo exclusions (may be empty for special categories)
        geo_count = 0
        if constraints["constraints"]["geo"]:
            geo_list = constraints["constraints"]["geo"]
            if geo_list and len(geo_list) > 0 and geo_list[0].get("value"):
                geo_count = len(geo_list[0]["value"].get("zip_codes", []))

        return {
            "signal_booster": f"Push {gold_count} high-LTV customers to Advantage+ as Prioritized Suggestions",
            "auto_tightener": f"Set age_min={constraints['constraints']['age']['value']} and exclude {geo_count} ZIP codes in Audience Controls",
            "drift_prevention": "Enable monitoring: Alert if Advantage+ CPM/CVR deviates >30% from Shopify buyer profile",
        }
