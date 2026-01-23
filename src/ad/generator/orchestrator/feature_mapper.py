"""
Feature Mapper: Extract feature values from visual_formula and map to template placeholders.

This module implements the priority-based feature extraction logic:
1. Check headroom_features first (highest priority)
2. Check entrance_features second
3. Use defaults if not found

Critical ROAS drivers (product_position, color_balance) are handled with special care.

UNIFIED FEATURE REGISTRY: All feature values are normalized using the centralized
feature_registry to ensure consistency across formula selection, prompt injection,
and feature extraction.
"""

import logging
from typing import Any, Dict, Optional

from .defaults import DEFAULT_VALUES
from .feature_registry import normalize_feature_value


logger = logging.getLogger(__name__)


class FeatureMapper:
    """
    Maps visual formula features to template placeholder values.

    Priority order:
    1. headroom_features (checked first - highest priority)
    2. entrance_features (checked if not in headroom)
    3. defaults (used if feature not found)
    """

    def __init__(self, defaults: Optional[Dict[str, str]] = None):
        """
        Initialize feature mapper.

        Args:
            defaults: Optional dict of default values (uses module defaults if None)
        """
        self.defaults = defaults if defaults is not None else DEFAULT_VALUES

    def get_feature_value(
        self,
        visual_formula: Dict[str, Any],
        feature_name: str,
    ) -> Optional[str]:
        """
        Extract feature value from visual_formula using priority rules.

        Priority:
        1. headroom_features (checked first)
        2. entrance_features (checked if not in headroom)
        3. Returns None if not found (caller should use default)

        Args:
            visual_formula: Visual formula JSON dict
            feature_name: Name of feature to extract (e.g., "product_position")

        Returns:
            Feature value string, or None if not found
        """
        # Priority 1: Check headroom_features first
        headroom_features = visual_formula.get("headroom_features", [])
        for feature in headroom_features:
            if feature.get("feature_name") == feature_name:
                value = feature.get("feature_value")
                if value:
                    logger.debug(
                        "Found %s='%s' in headroom_features",
                        feature_name,
                        value,
                    )
                    return self._normalize_value(value, feature_name)
        # Priority 2: Check entrance_features
        entrance_features = visual_formula.get("entrance_features", [])
        for feature in entrance_features:
            if feature.get("feature_name") == feature_name:
                value = feature.get("feature_value")
                if value:
                    logger.debug(
                        "Found %s='%s' in entrance_features",
                        feature_name,
                        value,
                    )
                    return self._normalize_value(value, feature_name)
        # Not found in either
        logger.debug("Feature '%s' not found in visual_formula", feature_name)
        return None

    def get_feature_value_with_default(
        self,
        visual_formula: Dict[str, Any],
        feature_name: str,
        placeholder_name: Optional[str] = None,
    ) -> str:
        """
        Get feature value with automatic fallback to default.

        Args:
            visual_formula: Visual formula JSON dict
            feature_name: Name of feature to extract
            placeholder_name: Name of template placeholder (defaults to feature_name)

        Returns:
            Feature value string (never None - uses default if needed)

        Logs the full fallback chain for traceability:
            1. headroom_features -> 2. entrance_features -> 3. defaults
        """
        # Priority 1: Check headroom_features
        headroom_features = visual_formula.get("headroom_features", [])
        for feature in headroom_features:
            if feature.get("feature_name") == feature_name:
                value = feature.get("feature_value")
                if value:
                    normalized = self._normalize_value(value, feature_name)
                    logger.info(
                        "Feature '%s': using headroom value '%s' (source: headroom_features)",
                        feature_name,
                        normalized,
                    )
                    return normalized

        # Priority 2: Check entrance_features
        entrance_features = visual_formula.get("entrance_features", [])
        for feature in entrance_features:
            if feature.get("feature_name") == feature_name:
                value = feature.get("feature_value")
                if value:
                    normalized = self._normalize_value(value, feature_name)
                    logger.info(
                        "Feature '%s': using entrance value '%s' (source: entrance_features)",
                        feature_name,
                        normalized,
                    )
                    return normalized

        # Priority 3: Use default
        placeholder = placeholder_name or feature_name
        default_value = self.defaults.get(placeholder)

        if default_value is None:
            logger.warning(
                "Feature '%s': no default for placeholder '%s', using 'N/A' "
                "(fallback chain exhausted)",
                feature_name,
                placeholder,
            )
            return "N/A"

        logger.info(
            "Feature '%s': using default value '%s' (source: defaults, placeholder: '%s')",
            feature_name,
            default_value,
            placeholder,
        )
        return default_value

    def _normalize_value(
        self, value: str, feature_name: Optional[str] = None
    ) -> str:
        """
        Normalize feature value using unified feature registry.

        Args:
            value: Raw feature value string
            feature_name: Optional feature name for registry lookup

        Returns:
            Normalized canonical value from registry
        """
        if not value:
            return ""
        if feature_name:
            # Use unified registry for normalization
            return normalize_feature_value(feature_name, value)
        # Fallback to simple normalization
        return str(value).strip()

    def get_product_position(
        self,
        visual_formula: Dict[str, Any],
    ) -> str:
        """
        Get product_position with special handling for critical ROAS driver.

        This method ensures consistent mapping of product_position, which is
        a critical ROAS driver. Handles multiple values in headroom_features
        by selecting the highest ROAS value.

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Product position value (e.g., "bottom-right", "left", "center")
        """
        headroom_features = visual_formula.get("headroom_features", [])
        entrance_features = visual_formula.get("entrance_features", [])
        # Collect all product_position values with their ROAS
        candidates = []
        # Check headroom_features (priority)
        for feature in headroom_features:
            if feature.get("feature_name") == "product_position":
                value = feature.get("feature_value")
                roas = feature.get("avg_roas", 0.0)
                if value:
                    candidates.append((value, roas, "headroom"))
        # Check entrance_features (lower priority)
        for feature in entrance_features:
            if feature.get("feature_name") == "product_position":
                value = feature.get("feature_value")
                roas = feature.get("avg_roas", 0.0)
                if value:
                    # Only add if not already in headroom
                    if not any(c[0] == value for c in candidates):
                        candidates.append((value, roas, "entrance"))

        if not candidates:
            # Use default
            default = self.defaults.get("product_position", "center")
            logger.info(
                "No product_position found, using default: %s",
                default,
            )
            return default
        # If multiple candidates, select highest ROAS
        if len(candidates) > 1:
            candidates.sort(
                key=lambda x: x[1], reverse=True
            )  # Sort by ROAS desc
            selected = candidates[0]
            logger.info(
                "Multiple product_position values found. Selected '%s' "
                "(ROAS: %.2f, source: %s) from %d candidates",
                selected[0],
                selected[1],
                selected[2],
                len(candidates),
            )
            return self._normalize_value(selected[0], "product_position")
        # Single candidate
        selected = candidates[0]
        logger.debug(
            "Selected product_position: '%s' (ROAS: %.2f, source: %s)",
            selected[0],
            selected[1],
            selected[2],
        )
        return self._normalize_value(selected[0], "product_position")

    def get_color_balance(
        self,
        visual_formula: Dict[str, Any],
    ) -> str:
        """
        Get color_balance with special handling for critical ROAS driver.

        This method ensures consistent mapping of color_balance, which is
        a critical ROAS driver. Handles cases where it appears in both
        entrance and headroom features.

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Color balance value (e.g., "warm-dominant", "cool-dominant", "balanced")
        """
        # Priority: headroom > entrance
        headroom_features = visual_formula.get("headroom_features", [])
        entrance_features = visual_formula.get("entrance_features", [])
        # Check headroom first
        for feature in headroom_features:
            if feature.get("feature_name") == "color_balance":
                value = feature.get("feature_value")
                if value:
                    logger.debug(
                        "Found color_balance='%s' in headroom_features",
                        value,
                    )
                    return self._normalize_value(value, "color_balance")
        # Check entrance
        for feature in entrance_features:
            if feature.get("feature_name") == "color_balance":
                value = feature.get("feature_value")
                if value:
                    logger.debug(
                        "Found color_balance='%s' in entrance_features",
                        value,
                    )
                    return self._normalize_value(value, "color_balance")
        # Use default
        default = self.defaults.get("color_balance", "balanced")
        logger.info(
            "No color_balance found, using default: %s",
            default,
        )
        return default

    def get_interaction_context(
        self,
        visual_formula: Dict[str, Any],
    ) -> str:
        """
        Get interaction_context from human_elements feature.

        Maps human_elements values to fixed, high-quality interaction descriptions
        for deterministic prompt generation. This follows the modular design:
        Subject (Complete) -> Background/Interaction (Simplified) -> Style (Bright/Meta).

        Priority: headroom > entrance > default

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Interaction context string (empty if no human elements)
        """
        # Priority: headroom > entrance
        headroom_features = visual_formula.get("headroom_features", [])
        entrance_features = visual_formula.get("entrance_features", [])

        human_elements_value = None
        source = None
        # Check headroom first
        for feature in headroom_features:
            if feature.get("feature_name") == "human_elements":
                value = feature.get("feature_value")
                if value:
                    human_elements_value = self._normalize_value(
                        value, "human_elements"
                    )
                    source = "headroom"
                    logger.debug(
                        "Found human_elements='%s' in headroom_features",
                        human_elements_value,
                    )
                    break
        # Check entrance if not found in headroom
        if not human_elements_value:
            for feature in entrance_features:
                if feature.get("feature_name") == "human_elements":
                    value = feature.get("feature_value")
                    if value:
                        human_elements_value = self._normalize_value(
                            value, "human_elements"
                        )
                        source = "entrance"
                        logger.debug(
                            "Found human_elements='%s' in entrance_features",
                            human_elements_value,
                        )
                        break
        # Map human_elements value to interaction context string
        if not human_elements_value:
            # No human elements - return empty (product-focused)
            default = self.defaults.get("interaction_context", "")
            logger.debug("No human_elements found, using default: empty")
            return default

        interaction = self._map_human_elements_to_interaction(
            human_elements_value, source
        )
        return interaction

    def _map_human_elements_to_interaction(
        self, human_elements_value: str, source: Optional[str]
    ) -> str:
        """
        Map human elements value to interaction context string.

        Args:
            human_elements_value: Normalized human elements value
            source: Source of the value ("headroom" or "entrance")

        Returns:
            Interaction context string
        """
        value_lower = human_elements_value.lower()

        # Use lookup table for mapping
        interaction_map = {
            "lifestyle": (
                "Interaction: A person visible in the background, walking "
                "barefoot or interacting with a pet, demonstrating floor hygiene. "
                "Natural, realistic interaction without posed appearance. "
            ),
            "person visible": (
                "Interaction: A person visible in the background, walking "
                "barefoot or interacting with a pet, demonstrating floor hygiene. "
                "The product remains the primary focus with natural interaction. "
            ),
            "face-visible": (
                "Interaction: A person visible in the background, walking "
                "barefoot or interacting with a pet, demonstrating floor hygiene. "
                "The product remains the primary focus with natural interaction. "
            ),
            "silhouette": (
                "Interaction: Human silhouette or figure in background, "
                "providing context without detail. Product remains dominant. "
            ),
            "none": "",
            "": "",
        }

        # Check for matches in priority order
        for key, interaction in interaction_map.items():
            if key in value_lower:
                if interaction:
                    logger.info(
                        "Mapped human_elements='%s' (source: %s) to interaction context",
                        human_elements_value,
                        source,
                    )
                return interaction

        # Unknown value - use generic lifestyle context
        logger.warning(
            "Unknown human_elements value '%s', using generic lifestyle context",
            human_elements_value,
        )
        return (
            "Interaction: Subtle lifestyle context suggesting product use "
            "without prominent human subjects. "
        )

    def get_composition_style(
        self,
        visual_formula: Dict[str, Any],
    ) -> str:
        """
        Extract composition style from negative_space_usage feature.

        Uses ROAS-based selection: highest ROAS value with positive lift.
        Maps to descriptive photographic instruction.

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Composition style string (e.g., "generous negative space")
        """
        # Find all negative_space_usage features
        candidates = []
        # Check headroom_features
        for feature in visual_formula.get("headroom_features", []):
            if feature.get("feature_name") == "negative_space_usage":
                roas = feature.get("avg_roas", 0)
                roas_lift = feature.get("roas_lift_pct", 0)
                value = feature.get("feature_value", "")
                if value and roas_lift >= 0:  # Only positive or neutral lift
                    candidates.append(
                        {
                            "value": value,
                            "roas": roas,
                            "roas_lift": roas_lift,
                            "source": "headroom",
                        }
                    )
        # Check entrance_features
        for feature in visual_formula.get("entrance_features", []):
            if feature.get("feature_name") == "negative_space_usage":
                roas = feature.get("avg_roas", 0)
                value = feature.get("feature_value", "")
                if value:
                    candidates.append(
                        {
                            "value": value,
                            "roas": roas,
                            "roas_lift": 0,  # Entrance features don't have lift
                            "source": "entrance",
                        }
                    )

        if not candidates:
            # Use default
            default = self.defaults.get("composition_style", "")
            logger.debug(
                "No negative_space_usage found, using default: %s", default
            )
            return default
        # Select highest ROAS
        best = max(candidates, key=lambda x: x["roas"])
        value = best["value"].lower()
        # Map to descriptive instruction
        if "generous" in value:
            composition = "generous negative space"
        elif "balanced" in value:
            composition = "balanced composition"
        elif "minimal" in value:
            composition = "minimal negative space"
        else:
            composition = f"{value} negative space"

        logger.info(
            "Selected composition_style='%s' (ROAS: %.2f, source: %s)",
            composition,
            best["roas"],
            best["source"],
        )
        return composition

    def get_lighting_detail(
        self,
        visual_formula: Dict[str, Any],
    ) -> str:
        """
        Extract lighting detail from brightness_distribution feature.

        Uses ROAS-based selection: highest ROAS value.
        Maps to descriptive photographic instruction.

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Lighting detail string (e.g., "subtle light-to-shadow gradient across the floor")
        """
        # Find all brightness_distribution features
        candidates = []
        # Check headroom_features
        for feature in visual_formula.get("headroom_features", []):
            if feature.get("feature_name") == "brightness_distribution":
                roas = feature.get("avg_roas", 0)
                roas_lift = feature.get("roas_lift_pct", 0)
                value = feature.get("feature_value", "")
                if value and roas_lift >= 0:
                    candidates.append(
                        {
                            "value": value,
                            "roas": roas,
                            "roas_lift": roas_lift,
                            "source": "headroom",
                        }
                    )
        # Check entrance_features
        for feature in visual_formula.get("entrance_features", []):
            if feature.get("feature_name") == "brightness_distribution":
                roas = feature.get("avg_roas", 0)
                value = feature.get("feature_value", "")
                if value:
                    candidates.append(
                        {
                            "value": value,
                            "roas": roas,
                            "roas_lift": 0,
                            "source": "entrance",
                        }
                    )

        if not candidates:
            # Use default
            default = self.defaults.get("lighting_detail", "")
            logger.debug(
                "No brightness_distribution found, using default: %s", default
            )
            return default
        # Select highest ROAS
        best = max(candidates, key=lambda x: x["roas"])
        value = best["value"].lower()
        # Map to descriptive instruction
        if "gradient" in value:
            lighting = "subtle light-to-shadow gradient across the floor"
        elif "uniform" in value:
            lighting = "uniform brightness distribution"
        elif "spotlight" in value:
            lighting = "focused spotlight on product"
        else:
            lighting = f"{value} brightness distribution"

        logger.info(
            "Selected lighting_detail='%s' (ROAS: %.2f, source: %s)",
            lighting,
            best["roas"],
            best["source"],
        )
        return lighting

    def get_environment_objects(
        self,
        visual_formula: Dict[str, Any],
    ) -> str:
        """
        Extract environment objects from relationship_depiction feature.

        Uses ROAS-based selection: highest ROAS value with positive lift.
        Maps to descriptive photographic instruction.

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Environment objects string (e.g., "placed near minimalist lifestyle items")
        """
        # Find all relationship_depiction features
        candidates = []
        # Check headroom_features
        for feature in visual_formula.get("headroom_features", []):
            if feature.get("feature_name") == "relationship_depiction":
                roas = feature.get("avg_roas", 0)
                roas_lift = feature.get("roas_lift_pct", 0)
                value = feature.get("feature_value", "")
                if value and roas_lift >= 0:
                    candidates.append(
                        {
                            "value": value,
                            "roas": roas,
                            "roas_lift": roas_lift,
                            "source": "headroom",
                        }
                    )
        # Check entrance_features
        for feature in visual_formula.get("entrance_features", []):
            if feature.get("feature_name") == "relationship_depiction":
                roas = feature.get("avg_roas", 0)
                value = feature.get("feature_value", "")
                if value:
                    candidates.append(
                        {
                            "value": value,
                            "roas": roas,
                            "roas_lift": 0,
                            "source": "entrance",
                        }
                    )

        if not candidates:
            # Use default
            default = self.defaults.get("environment_objects", "")
            logger.debug(
                "No relationship_depiction found, using default: %s", default
            )
            return default
        # Select highest ROAS
        best = max(candidates, key=lambda x: x["roas"])
        value = best["value"].lower()
        # Map to descriptive instruction
        if "product-with-objects" in value or "with-objects" in value:
            environment = (
                "placed near minimalist lifestyle items like designer indoor "
                "plants or pet accessories"
            )
        elif "product-in-environment" in value or "in-environment" in value:
            environment = "integrated into a natural home environment with contextual elements"
        elif "product-alone" in value or "alone" in value:
            environment = "isolated product focus without additional objects"
        else:
            environment = f"contextual relationship: {value}"

        logger.info(
            "Selected environment_objects='%s' (ROAS: %.2f, source: %s)",
            environment,
            best["roas"],
            best["source"],
        )
        return environment

    def get_placement_target(
        self,
        visual_formula: Dict[str, Any],
    ) -> str:
        """
        Extract placement target from product_placement or relationship_depiction.

        Uses ROAS-based selection: highest ROAS value.
        Maps to descriptive photographic instruction.

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Placement target string (e.g., "modern sofa or wall")
        """
        candidates = self._collect_placement_candidates(visual_formula)

        if not candidates:
            # Use default
            default = self.defaults.get(
                "placement_target", "modern sofa or wall"
            )
            logger.debug(
                "No placement target found, using default: %s", default
            )
            return default

        # Select highest ROAS and map to placement
        best = max(candidates, key=lambda x: x["roas"])
        placement = self._map_value_to_placement(best["value"])
        feature = best.get("feature", "unknown")

        logger.info(
            "Selected placement_target='%s' (ROAS: %.2f, source: %s, feature: %s)",
            placement,
            best["roas"],
            best["source"],
            feature,
        )
        return placement

    def _collect_placement_candidates(
        self, visual_formula: Dict[str, Any]
    ) -> list:
        """
        Collect placement candidates from headroom features.

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            List of candidate dicts with value, roas, source, feature
        """
        candidates = []
        # Check product_placement first
        candidates.extend(
            self._extract_feature_candidates(
                visual_formula, "product_placement", "headroom"
            )
        )
        # If no product_placement, check relationship_depiction
        if not candidates:
            candidates.extend(
                self._extract_feature_candidates(
                    visual_formula, "relationship_depiction", "headroom"
                )
            )
        return candidates

    def _extract_feature_candidates(
        self,
        visual_formula: Dict[str, Any],
        feature_name: str,
        source: str,
    ) -> list:
        """
        Extract candidates for a specific feature from headroom features.

        Args:
            visual_formula: Visual formula JSON dict
            feature_name: Name of feature to extract
            source: Source label (e.g., "headroom")

        Returns:
            List of candidate dicts
        """
        candidates = []
        for feature in visual_formula.get("headroom_features", []):
            if feature.get("feature_name") == feature_name:
                roas = feature.get("avg_roas", 0)
                roas_lift = feature.get("roas_lift_pct", 0)
                value = feature.get("feature_value", "")
                if value and roas_lift >= 0:
                    candidates.append(
                        {
                            "value": value,
                            "roas": roas,
                            "roas_lift": roas_lift,
                            "source": source,
                            "feature": feature_name,
                        }
                    )
        return candidates

    def _map_value_to_placement(self, value: str) -> str:
        """
        Map placement value to descriptive instruction.

        Args:
            value: Placement value string

        Returns:
            Descriptive placement string
        """
        value_lower = value.lower()

        # Use lookup table for mapping
        placement_map = {
            "left": "modern sofa or wall",
            "right": "minimalist wall or furniture",
            "product-with-objects": "lifestyle furniture or decorative elements",
            "product-in-environment": "natural home environment elements",
        }

        # Check for matches
        for key, placement in placement_map.items():
            if key in value_lower:
                return placement

        # Safe default
        return "modern sofa or wall"
