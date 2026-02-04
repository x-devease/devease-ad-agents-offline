"""
Mining Strategy Selector for Ad Miner V1.8

Intelligently selects winner quantile from customer config using 6-level priority hierarchy.
"""
import logging
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Hardcoded system defaults (no global config needed)
SYSTEM_DEFAULTS = {
    "mining_profiles": {
        "balanced": {
            "base_quantile": 0.90,
            "min_sample_size": 50
        }
    },
    "default_profile": "balanced",
    "min_sample_size_requirements": {
        "winner_quantile_0.98": 200,
        "winner_quantile_0.95": 100,
        "winner_quantile_0.90": 50,
        "winner_quantile_0.80": 30,
        "winner_quantile_0.70": 20
    }
}


class MiningStrategySelector:
    """
    V1.8 Mining Strategy Selector.

    Intelligently resolves winner_quantile from customer config:
    1. Manual override (input_config.yaml)
    2. Profile selection (input_config.yaml)
    3. Product-level override (customer config)
    4. Platform-level override (customer config)
    5. Customer default (customer config)
    6. System hardcoded default
    """

    def __init__(self, config_root: Path):
        """
        Initialize selector with config root path.

        Args:
            config_root: Root path for customer config files
        """
        self.config_root = Path(config_root)

    def determine_winner_quantile(
        self,
        customer_id: str,
        platform: str,
        product: Optional[str],
        daily_budget_cents: int,
        manual_quantile: Optional[float] = None,
        manual_profile: Optional[str] = None
    ) -> float:
        """
        Determine winner quantile using 6-level hierarchy.

        Args:
            customer_id: Customer identifier
            platform: Platform name (meta, tiktok, google)
            product: Product name (optional)
            daily_budget_cents: Daily budget in cents
            manual_quantile: Manual override from input_config
            manual_profile: Manual profile selection from input_config

        Returns:
            Winner quantile (0.0-1.0)
        """
        # Load customer config
        customer_config = self._load_customer_config(customer_id)

        # Priority 1: Manual override (highest priority)
        if manual_quantile is not None:
            logger.info(f"Priority 1: Using manual quantile {manual_quantile}")
            return self._validate_and_adjust_quantile(
                manual_quantile, daily_budget_cents, customer_config
            )

        # Priority 2: Manual profile selection
        if manual_profile is not None:
            logger.info(f"Priority 2: Using manual profile '{manual_profile}'")
            profile = self._load_profile_from_dict(
                manual_profile,
                customer_config.get("mining_profiles", {}) if customer_config else {}
            )
            if profile:
                return profile["base_quantile"]

        # If no customer config, use system defaults
        if customer_config is None:
            logger.warning(f"No customer config for '{customer_id}', using system default")
            return SYSTEM_DEFAULTS["mining_profiles"]["balanced"]["base_quantile"]

        # Priority 3: Product-level override (highest specificity)
        if product and product in customer_config.get("product_overrides", {}):
            product_config = customer_config["product_overrides"][product]

            # Check platform-specific override
            if platform in product_config.get("platform_overrides", {}):
                platform_config = product_config["platform_overrides"][platform]
                if "winner_quantile" in platform_config:
                    quantile = platform_config["winner_quantile"]
                    logger.info(
                        f"Priority 3: Using product-specific override: "
                        f"{product}@{platform} → {quantile}"
                    )
                    return self._validate_and_adjust_quantile(quantile, daily_budget_cents, customer_config)

            # Check product default
            if "default_profile" in product_config:
                profile_name = product_config["default_profile"]
                profile = self._load_profile_from_dict(
                    profile_name,
                    customer_config.get("mining_profiles", {})
                )
                if profile:
                    logger.info(f"Priority 3: Using product default profile '{profile_name}'")
                    return profile["base_quantile"]

        # Priority 4: Platform-level override
        if platform in customer_config.get("platform_overrides", {}):
            platform_config = customer_config["platform_overrides"][platform]

            # Check budget tiers
            if "budget_tiers" in platform_config:
                for tier in sorted(
                    platform_config["budget_tiers"],
                    key=lambda x: x["min_budget_cents"],
                    reverse=True
                ):
                    if daily_budget_cents >= tier["min_budget_cents"]:
                        quantile = tier["winner_quantile"]
                        logger.info(
                            f"Priority 4: Using platform budget tier: "
                            f"{platform} (${daily_budget_cents/100:.0f}) → {quantile}"
                        )
                        return self._validate_and_adjust_quantile(quantile, daily_budget_cents, customer_config)

            # Check platform profile
            if "profile" in platform_config:
                profile_name = platform_config["profile"]
                profile = self._load_profile_from_dict(
                    profile_name,
                    customer_config.get("mining_profiles", {})
                )
                if profile:
                    logger.info(f"Priority 4: Using platform profile '{profile_name}'")
                    return profile["base_quantile"]

        # Priority 5: Customer default profile
        if "default_profile" in customer_config:
            profile_name = customer_config["default_profile"]
            profile = self._load_profile_from_dict(
                profile_name,
                customer_config.get("mining_profiles", {})
            )
            if profile:
                logger.info(f"Priority 5: Using customer default profile '{profile_name}'")
                return profile["base_quantile"]

        # Priority 6: System hardcoded default
        logger.info("Priority 6: Using system default profile 'balanced'")
        return SYSTEM_DEFAULTS["mining_profiles"]["balanced"]["base_quantile"]

    def _load_customer_config(self, customer_id: str) -> Optional[dict]:
        """Load customer configuration file."""
        path = self.config_root / customer_id / "config.yaml"
        if not path.exists():
            logger.warning(f"Customer config not found at {path}")
            return None

        with open(path) as f:
            return yaml.safe_load(f)

    def _load_profile_from_dict(self, profile_name: str, profiles_dict: dict) -> Optional[dict]:
        """Load profile from a dictionary of profiles."""
        if profile_name in profiles_dict:
            return profiles_dict[profile_name]

        # Fall back to system defaults
        if profile_name in SYSTEM_DEFAULTS["mining_profiles"]:
            return SYSTEM_DEFAULTS["mining_profiles"][profile_name]

        logger.error(f"Profile '{profile_name}' not found")
        return None

    def _validate_and_adjust_quantile(
        self,
        quantile: float,
        daily_budget_cents: int,
        customer_config: Optional[dict]
    ) -> float:
        """
        Validate and adjust quantile.

        Args:
            quantile: Proposed winner quantile
            daily_budget_cents: Daily budget (for logging)
            customer_config: Customer config (for override rules)

        Returns:
            Adjusted quantile (if needed)
        """
        # Check customer-specific override rules
        if customer_config:
            for rule in customer_config.get("auto_fallback_rules", []):
                condition = rule.get("condition", "")
                if f"daily_budget_cents > {daily_budget_cents}" in condition:
                    override_quantile = rule["winner_quantile"]
                    logger.info(
                        f"Customer override triggered: {condition} → {override_quantile}"
                    )
                    return override_quantile

        # Validate quantile range
        if not 0.5 <= quantile <= 0.99:
            logger.warning(f"Quantile {quantile} out of range [0.5, 0.99], clamping")
            return max(0.5, min(0.99, quantile))

        return quantile

    def get_min_sample_size(self, winner_quantile: float, customer_config: Optional[dict] = None) -> int:
        """
        Get minimum sample size requirement for a given quantile.

        Args:
            winner_quantile: Winner quantile
            customer_config: Customer config (for custom requirements)

        Returns:
            Minimum sample size
        """
        # Check customer-specific requirements first
        if customer_config:
            requirements = customer_config.get("min_sample_size_requirements", {})
            quantile_key = f"winner_quantile_{winner_quantile:.2f}"
            if quantile_key in requirements:
                return requirements[quantile_key]

        # Use system defaults
        requirements = SYSTEM_DEFAULTS["min_sample_size_requirements"]
        quantile_key = f"winner_quantile_{winner_quantile:.2f}"
        if quantile_key in requirements:
            return requirements[quantile_key]

        # Interpolate or use default
        if winner_quantile >= 0.95:
            return 100
        elif winner_quantile >= 0.85:
            return 50
        else:
            return 30
