"""
Prompt Builder: Generates prompts from scorer recommendations and product context.

This module builds deterministic prompts by:
1. Extracting features from scorer recommendations (visual_formula)
2. Extracting product context information
3. Mapping features to template placeholders
4. Rendering templates with resolved values
"""

# pylint: disable=too-many-lines

from dataclasses import asdict, dataclass
import logging
import re
from typing import Any, Dict, Optional

from .defaults import DEFAULT_VALUES
from .feature_mapper import FeatureMapper
from .feature_registry import (
    get_branch_synergy_features,
    normalize_feature_value,
)
from .scene_config import get_scene_overview
from .template_engine import TemplateEngine


logger = logging.getLogger(__name__)


@dataclass
class PromptBuilderConfig:
    """Configuration for PromptBuilder initialization."""

    defaults: Optional[Dict[str, str]] = None
    template: Optional[str] = None
    lean_mode: bool = False
    v2_mode: bool = False
    branch_name: Optional[str] = None
    step2_mode: bool = False
    product_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return asdict(self)


@dataclass
class BranchProcessingConfig:
    """Configuration for branch processing methods."""

    visual_formula: Dict[str, Any]
    product_context: Dict[str, Any]
    placeholder_values: Dict[str, Any]
    headroom_temperature: Optional[str] = None
    headroom_color_balance: Optional[str] = None
    merged_synergies: Dict[str, str] = None

    def __post_init__(self):
        if self.merged_synergies is None:
            self.merged_synergies = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return asdict(self)


@dataclass
class AtmosphereDeterminationConfig:
    """Configuration for atmosphere determination with headroom."""

    visual_formula: Dict[str, Any]
    headroom_temperature: Optional[str] = None
    headroom_color_balance: Optional[str] = None
    merged_synergies: Dict[str, str] = None
    branch_name: str = ""

    def __post_init__(self):
        if self.merged_synergies is None:
            self.merged_synergies = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return asdict(self)


@dataclass
class WideScenePlaceholdersConfig:
    """Configuration for wide scene placeholders."""

    placement_target: str
    composition_style: str
    lighting_detail: str
    environment_objects: str
    placeholder_values: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of the config
        """
        return asdict(self)


class PromptBuilder:
    """
    Builds prompts from scorer recommendations and product context.

    Builds prompts by:
    - Extracting features from visual_formula (scorer recommendations)
    - Extracting product context information
    - Mapping to template placeholders
    - Rendering templates with resolved values
    """

    def __init__(self, config: Optional[PromptBuilderConfig] = None):
        """
        Initialize orchestrator.

        Args:
            config: PromptBuilderConfig object with all builder parameters
        """
        # Use default config if none provided
        if config is None:
            config = PromptBuilderConfig()
        # Store config (1 attribute)
        self._config = config
        # Initialize helpers (2 attributes)
        self.defaults = config.defaults or DEFAULT_VALUES
        self.feature_mapper = FeatureMapper(defaults=self.defaults)
        # Template engine (lazy-loaded, 1 attribute)
        self._template_engine = None

    @property
    def template_engine(self) -> TemplateEngine:
        """Get or create template engine."""
        if self._template_engine is None:
            self._template_engine = TemplateEngine(
                template=self._config.template,
                lean_mode=self._config.lean_mode,
                v2_mode=self._config.v2_mode,
                branch_name=self._config.branch_name,
            )
        return self._template_engine

    @property
    def lean_mode(self) -> bool:
        """Get lean mode from config."""
        return self._config.lean_mode

    @property
    def v2_mode(self) -> bool:
        """Get v2 mode from config."""
        return self._config.v2_mode

    @property
    def branch_name(self) -> Optional[str]:
        """Get branch name from config."""
        return self._config.branch_name

    @property
    def step2_mode(self) -> bool:
        """Get step2 mode from config."""
        return self._config.step2_mode

    @property
    def product_context(self) -> Dict[str, Any]:
        """Get product context from config."""
        return self._config.product_context or {}

    def _extract_global_view_definition(
        self, visual_formula: Dict[str, Any]
    ) -> str:
        """
        Extract global view definition based on branch/feature context.

        Tri-Template Architecture:
        - golden_ratio (V2_WIDE_SCENE): Wide-angle lifestyle photography
        - high_efficiency (V2_MACRO_DETAIL): Professional close-up (handled in template)
        - cool_peak (V2_FLAT_TECH): Low-profile flat-lay (handled in template)

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Global view definition string (only used for legacy V2 or fallback)
        """
        # Check for branch name first
        if self.branch_name == "golden_ratio":
            # V2_WIDE_SCENE: Wide-angle lifestyle (10.7 ROAS Hero)
            return "Wide-angle lifestyle photography"
        if self.branch_name == "high_efficiency":
            # V2_MACRO_DETAIL: Template handles this directly
            # Return empty string as template has its own header
            return ""
        if self.branch_name == "cool_peak":
            # V2_FLAT_TECH: Template handles this directly
            # Return empty string as template has its own header
            return ""
        # Fallback: Check product_visibility to infer view type
        product_visibility = self.feature_mapper.get_feature_value(
            visual_formula, "product_visibility"
        )
        if product_visibility == "partial":
            return (
                "Professional close-up photography focusing on the textured "
                "base and metallic finishes"
            )
        # Default: Wide-angle lifestyle
        return "Wide-angle lifestyle photography"

    def _extract_lean_subject_description(
        self, product_context: Dict[str, Any]
    ) -> str:
        """
        Extract minimal subject description for lean mode.

        Uses broader architectural terms instead of detailed component catalog.

        Args:
            product_context: Product context dict

        Returns:
            Minimal subject description string
        """
        product_name = product_context.get(
            "product_name"
        ) or product_context.get("name")

        if not product_name:
            raise ValueError(
                "product_name is required but missing from product_context. "
                "Provide 'product_name' or 'name' in product_context."
            )
        # Minimal description: product name with optional handle text
        # Add explicit uniqueness to prevent double products (configurable)
        uniqueness_text = product_context.get(
            "uniqueness_text", "a single, solitary unit"
        )
        handle_text = product_context.get("handle_text")
        if handle_text:
            return (
                f'{product_name}, {uniqueness_text} with the text "{handle_text}" '
                "crisply printed on the handle, oriented vertically along the handle's axis, "
                "arranged from top to bottom"
            )

        return f"{product_name}, {uniqueness_text}"

    def generate_p0_prompt(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],
    ) -> str:
        """
        Generate P0 prompt using hard-coded template mapping.

        This is the main entry point. It:
        1. Extracts features from visual_formula (with priority rules)
        2. Extracts product info from product_context
        3. Maps to template placeholders
        4. Renders template
        5. Returns deterministic prompt string

        Args:
            visual_formula: Visual formula JSON (same format as current system)
            product_context: Product context dict (same format as current system)

        Returns:
            Deterministic prompt string

        Raises:
            ValueError: If critical placeholders (subject_name) are missing
        """
        logger.info(
            "Generating P0 prompt with Mask-Template Orchestrator "
            "(lean_mode=%s, v2_mode=%s)",
            self.lean_mode,
            self.v2_mode,
        )
        # Build placeholder values dictionary
        placeholder_values = {}
        # Extract base product and context info
        self._extract_base_product_info(
            visual_formula, product_context, placeholder_values
        )
        # Process V2 branch-specific features
        if self.v2_mode and self.branch_name:
            self._process_v2_branch_features(
                visual_formula, product_context, placeholder_values
            )
        else:
            # Non-V2 or no branch: Use standard extraction
            self._process_standard_features(
                visual_formula, product_context, placeholder_values
            )
        # Extract color balance (independent from atmosphere)
        self._extract_color_balance(visual_formula, placeholder_values)
        # Extract layout and composition features
        self._extract_layout_features(visual_formula, placeholder_values)
        # Extract interaction context
        self._extract_interaction_context(
            visual_formula, product_context, placeholder_values
        )
        # Extract static context/background
        self._extract_static_context_with_mode(visual_formula, placeholder_values)
        # Extract physical state and grounding
        self._extract_physical_state_and_grounding(
            visual_formula, product_context, placeholder_values
        )
        # Build CMF core
        self._build_cmf_core(product_context, placeholder_values)
        # Extract scene overview
        placeholder_values["scene_overview"] = self._extract_scene_overview(
            visual_formula, placeholder_values, product_context
        )
        # Process V2 Enhanced ROAS features
        if self.v2_mode:
            self._process_v2_enhanced_features(
                visual_formula, product_context, placeholder_values
            )
        # Step 2: Inject Aesthetic Modifiers (if enabled)
        if self.step2_mode:
            modifiers = self._get_step2_modifiers(placeholder_values)
            placeholder_values.update(modifiers)
        # Validate critical placeholders
        if not placeholder_values.get("subject_description"):
            raise ValueError(
                "subject_description is required but missing from product_context. "
                "Provide 'product_name' or 'name' in product_context."
            )
        # Render template
        prompt = self.template_engine.render(placeholder_values, strict=False)
        # Cleanup prompt: Remove hanging prepositions and broken syntax
        prompt = self.cleanup_prompt(prompt)

        logger.info(
            "Generated P0 prompt (%d chars). product_position=%s, color_balance=%s",
            len(prompt),
            placeholder_values.get("product_position"),
            placeholder_values.get("color_balance"),
        )

        return prompt

    def _extract_base_product_info(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract base product information and global view."""
        # 0. Extract global view definition based on branch (for dynamic image reference)
        # Note: V2_MACRO_DETAIL and V2_FLAT_TECH templates have their own headers
        # Only V2_WIDE_SCENE and legacy V2 use global_view_definition
        global_view = self._extract_global_view_definition(visual_formula)
        if global_view or not (
            self.v2_mode and self.branch_name in ["high_efficiency", "cool_peak"]
        ):
            placeholder_values["global_view_definition"] = global_view
        # 1. Extract product information from product_context
        if self.v2_mode or self.lean_mode:
            placeholder_values["subject_description"] = (
                self._extract_lean_subject_description(product_context)
            )
        else:
            placeholder_values["subject_description"] = (
                self._extract_subject_description(product_context)
            )
        placeholder_values["material_finish"] = self._extract_material_finish(
            product_context
        )
        # Color constraint and grounding instruction are handled below based on mode
        if not self.v2_mode and not self.lean_mode:
            placeholder_values["completeness_instruction"] = (
                self._extract_completeness_instruction(product_context)
            )

    def cleanup_prompt(self, prompt: str) -> str:
        """
        Cleanup prompt by removing hanging prepositions and broken syntax.

        Fixes:
        - Double commas (e.g., ", ,")
        - Trailing prepositions before punctuation (e.g., "with .")
        - Isolated commas/periods from empty placeholders
        - Multiple spaces
        - Awkward phrasing (e.g., "featuring placed" -> "featuring")

        Args:
            prompt: Raw prompt string

        Returns:
            Cleaned prompt string
        """
        # Remove double commas
        cleaned = re.sub(r",\s*,", ",", prompt)
        # Remove trailing prepositions before comma or period if followed by empty content
        # Pattern: "with/featuring/and/near" followed by comma or period
        cleaned = re.sub(
            r"\s+(with|featuring|and|near)\s*(\.|,)",
            r"\2",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Remove isolated commas or periods
        cleaned = re.sub(r"\s*,\s*\.", ".", cleaned)
        cleaned = re.sub(r"\s*\.\s*,", ".", cleaned)
        # Remove multiple spaces
        cleaned = re.sub(r"\s\s+", " ", cleaned).strip()
        # Remove trailing prepositions at end of string
        cleaned = re.sub(
            r"\s+(with|featuring|and|near)\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Remove trailing comma or period if redundant
        if cleaned.endswith(",") or cleaned.endswith("."):
            cleaned = cleaned.rstrip(",.").strip() + "."

        return cleaned
    # ========== NEW HELPER METHODS FOR REFACTORING ==========
    # These methods extract logic from generate_p0_prompt to reduce local variables
    def _process_v2_branch_features(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Process V2 branch-specific features (golden_ratio, high_efficiency, cool_peak)."""
        # V2 PRODUCTION STRATEGY: ROAS Synergy with Configuration Override
        # UNIFIED FEATURE REGISTRY: All values use centralized registry for consistency
        # UNIVERSAL CONFIGURATION: Allow override via product_context["branch_synergies"]
        # HEADROOM PRIORITY: Headroom features take priority over branch synergies
        # PRIORITY 1: Check headroom features first (highest priority)
        headroom_temperature = self.feature_mapper.get_feature_value(
            visual_formula, "temperature"
        )
        headroom_color_balance = self.feature_mapper.get_color_balance(
            visual_formula
        )
        # Get branch-specific ROAS synergies from unified registry (defaults)
        branch_synergies = get_branch_synergy_features(self.branch_name)
        # Check for configuration override (universal)
        config_synergies = {}
        if "branch_synergies" in product_context:
            branch_config = product_context["branch_synergies"]
            if isinstance(branch_config, dict):
                config_synergies = branch_config.get(self.branch_name, {})
        # Merge: Configuration override > ROAS defaults
        merged_synergies = {**branch_synergies, **config_synergies}

        if config_synergies:
            logger.info(
                "Branch %s: Using configuration override for branch_synergies",
                self.branch_name,
            )

        if self.branch_name == "golden_ratio":
            self._process_golden_ratio_branch(
                BranchProcessingConfig(
                    visual_formula=visual_formula,
                    product_context=product_context,
                    placeholder_values=placeholder_values,
                    headroom_temperature=headroom_temperature,
                    headroom_color_balance=headroom_color_balance,
                    merged_synergies=merged_synergies,
                )
            )
        elif self.branch_name == "high_efficiency":
            self._process_high_efficiency_branch(
                BranchProcessingConfig(
                    visual_formula=visual_formula,
                    product_context=product_context,
                    placeholder_values=placeholder_values,
                    headroom_color_balance=headroom_color_balance,
                    merged_synergies=merged_synergies,
                )
            )
        elif self.branch_name == "cool_peak":
            self._process_cool_peak_branch(
                BranchProcessingConfig(
                    visual_formula=visual_formula,
                    product_context=product_context,
                    placeholder_values=placeholder_values,
                    headroom_temperature=headroom_temperature,
                    headroom_color_balance=headroom_color_balance,
                    merged_synergies=merged_synergies,
                )
            )
        else:
            # Fallback: Use standard extraction with registry normalization
            self._process_v2_fallback(
                visual_formula, product_context, placeholder_values,
                headroom_color_balance
            )

    def _process_golden_ratio_branch(
        self, config: BranchProcessingConfig
    ) -> None:
        """Process golden_ratio branch features (10.7 ROAS Synergy)."""
        # Branch 1: 10.7 ROAS Synergy (bottom-right + product-in-environment)
        # + 7.07 ROAS Pair (warm-dominant + Person visible)
        # SCORER PRIORITY: Check scorer recommendations first, then fallback to hardcoded
        scorer_product_position = self.feature_mapper.get_feature_value(
            config.visual_formula, "product_position"
        )
        if scorer_product_position:
            config.placeholder_values["product_position"] = scorer_product_position
            logger.info(
                "Branch 1 (golden_ratio): Using scorer recommendation for "
                "product_position='%s' (overriding hardcoded)",
                scorer_product_position,
            )
        else:
            config.placeholder_values["product_position"] = (
                config.merged_synergies.get("product_position", "bottom-right")
            )
            logger.warning(
                "Branch 1 (golden_ratio): No scorer recommendation for "
                "product_position, using hardcoded fallback: '%s'",
                config.placeholder_values["product_position"],
            )

        scorer_relationship = self.feature_mapper.get_feature_value(
            config.visual_formula, "relationship_depiction"
        )
        if scorer_relationship:
            config.placeholder_values["relationship_depiction"] = scorer_relationship
            logger.info(
                "Branch 1 (golden_ratio): Using scorer recommendation for "
                "relationship_depiction='%s' (overriding hardcoded)",
                scorer_relationship,
            )
        else:
            config.placeholder_values["relationship_depiction"] = (
                config.merged_synergies.get("relationship_depiction", "product-in-environment")
            )
            logger.warning(
                "Branch 1 (golden_ratio): No scorer recommendation for "
                "relationship_depiction, using hardcoded fallback: '%s'",
                config.placeholder_values["relationship_depiction"],
            )
        # HEADROOM PRIORITY: Use headroom temperature if available
        atmosphere_value = self._determine_atmosphere_with_headroom(
            AtmosphereDeterminationConfig(
                visual_formula=config.visual_formula,
                headroom_temperature=config.headroom_temperature,
                headroom_color_balance=config.headroom_color_balance,
                merged_synergies=config.merged_synergies,
                branch_name="golden_ratio",
            )
        )
        config.placeholder_values["atmosphere"] = atmosphere_value
        # Incorporate color_balance into atmosphere description if available
        color_balance_value = (
            config.headroom_color_balance
            or self.feature_mapper.get_color_balance(config.visual_formula)
        )
        config.placeholder_values["atmosphere_description"] = (
            self._map_atmosphere_to_description(
                atmosphere_value,
                self.branch_name,
                config.product_context,
                color_balance_value,
            )
        )
        # SCORER PRIORITY: Check scorer recommendation for human_elements
        scorer_human_elements = self.feature_mapper.get_feature_value(
            config.visual_formula, "human_elements"
        )
        if scorer_human_elements:
            config.placeholder_values["human_elements"] = scorer_human_elements
            logger.info(
                "Branch 1 (golden_ratio): Using scorer recommendation for "
                "human_elements='%s' (overriding hardcoded 'Person visible')",
                scorer_human_elements,
            )
        else:
            config.placeholder_values["human_elements"] = config.merged_synergies.get(
                "human_elements", "Person visible"
            )
            logger.warning(
                "Branch 1 (golden_ratio): No scorer recommendation for "
                "human_elements, using hardcoded fallback: '%s'",
                config.placeholder_values["human_elements"],
            )

    def _process_high_efficiency_branch(
        self, config: BranchProcessingConfig
    ) -> None:
        """Process high_efficiency branch features (5.60 ROAS Synergy)."""
        # Branch 2: 5.60 ROAS Synergy (partial visibility + strong visual impact)
        scorer_product_visibility = self.feature_mapper.get_feature_value(
            config.visual_formula, "product_visibility"
        )
        if scorer_product_visibility:
            config.placeholder_values["product_visibility"] = scorer_product_visibility
            logger.info(
                "Branch 2 (high_efficiency): Using scorer recommendation for "
                "product_visibility='%s' (overriding hardcoded 'partial')",
                scorer_product_visibility,
            )
        else:
            config.placeholder_values["product_visibility"] = (
                config.merged_synergies.get("product_visibility", "partial")
            )
            logger.warning(
                "Branch 2 (high_efficiency): No scorer recommendation for "
                "product_visibility, using hardcoded fallback: '%s'",
                config.placeholder_values["product_visibility"],
            )

        scorer_visual_impact = self.feature_mapper.get_feature_value(
            config.visual_formula, "visual_impact"
        )
        if scorer_visual_impact:
            config.placeholder_values["visual_impact"] = scorer_visual_impact
            logger.info(
                "Branch 2 (high_efficiency): Using scorer recommendation for "
                "visual_impact='%s' (overriding hardcoded 'strong')",
                scorer_visual_impact,
            )
        else:
            config.placeholder_values["visual_impact"] = config.merged_synergies.get(
                "visual_impact", "strong"
            )
            logger.warning(
                "Branch 2 (high_efficiency): No scorer recommendation for "
                "visual_impact, using hardcoded fallback: '%s'",
                config.placeholder_values["visual_impact"],
            )
        # Check scorer for atmosphere recommendation
        scorer_atmosphere = self.feature_mapper.get_feature_value(
            config.visual_formula, "atmosphere"
        )
        if scorer_atmosphere:
            atmosphere_value = scorer_atmosphere
            logger.info(
                "Branch 2 (high_efficiency): Using scorer recommendation for "
                "atmosphere='%s' (overriding hardcoded 'Neutral')",
                scorer_atmosphere,
            )
        else:
            atmosphere_value = config.merged_synergies.get("atmosphere", "Neutral")
            logger.warning(
                "Branch 2 (high_efficiency): No scorer recommendation for "
                "atmosphere, using hardcoded fallback: '%s'",
                atmosphere_value,
            )
        config.placeholder_values["atmosphere"] = atmosphere_value
        # Incorporate color_balance into atmosphere description if available
        color_balance_value = (
            config.headroom_color_balance
            or self.feature_mapper.get_color_balance(config.visual_formula)
        )
        config.placeholder_values["atmosphere_description"] = (
            self._map_atmosphere_to_description(
                atmosphere_value,
                self.branch_name,
                config.product_context,
                color_balance_value,
            )
        )

    def _process_cool_peak_branch(
        self, config: BranchProcessingConfig
    ) -> None:
        """Process cool_peak branch features (8.34 ROAS Critical)."""
        # Branch 3: 8.34 ROAS Critical (Temperature: Cool)
        if config.headroom_temperature:
            temp_lower = config.headroom_temperature.lower()
            if "cool" in temp_lower:
                atmosphere_value = "Cool"
            elif "warm" in temp_lower:
                atmosphere_value = "Warm"
            else:
                atmosphere_value = "Neutral"
            logger.info(
                "Branch 3 (cool_peak): Using headroom temperature='%s' "
                "(highest priority)",
                atmosphere_value,
            )
        else:
            scorer_atmosphere = self.feature_mapper.get_feature_value(
                config.visual_formula, "atmosphere"
            )
            if scorer_atmosphere:
                atmosphere_value = scorer_atmosphere
                logger.info(
                    "Branch 3 (cool_peak): Using scorer recommendation for "
                    "atmosphere='%s' (overriding hardcoded 'Cool')",
                    scorer_atmosphere,
                )
            else:
                atmosphere_value = config.merged_synergies.get("atmosphere", "Cool")
                logger.warning(
                    "Branch 3 (cool_peak): No scorer recommendation for "
                    "atmosphere, using hardcoded fallback: '%s'",
                    atmosphere_value,
                )
        config.placeholder_values["atmosphere"] = atmosphere_value
        # Incorporate color_balance into atmosphere description if available
        color_balance_value = (
            config.headroom_color_balance
            or self.feature_mapper.get_color_balance(config.visual_formula)
        )
        config.placeholder_values["atmosphere_description"] = (
            self._map_atmosphere_to_description(
                atmosphere_value,
                self.branch_name,
                config.product_context,
                color_balance_value,
            )
        )

    def _process_v2_fallback(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],
        placeholder_values: Dict[str, Any],
        headroom_color_balance: Optional[str],  # pylint: disable=unused-argument
    ) -> None:
        """Fallback V2 processing when branch doesn't match specific handlers."""
        product_pos = self.feature_mapper.get_product_position(visual_formula)
        placeholder_values["product_position"] = (
            normalize_feature_value("product_position", product_pos)
            if product_pos
            else product_pos
        )
        atmosphere_value = self._extract_atmosphere(visual_formula)
        placeholder_values["atmosphere"] = (
            normalize_feature_value("atmosphere", atmosphere_value)
            if atmosphere_value
            else atmosphere_value
        )
        # Incorporate color_balance into atmosphere description if available
        color_balance_value = self.feature_mapper.get_color_balance(visual_formula)
        placeholder_values["atmosphere_description"] = (
            self._map_atmosphere_to_description(
                atmosphere_value,
                self.branch_name,
                product_context,
                color_balance_value,
            )
        )

    def _determine_atmosphere_with_headroom(
        self, config: AtmosphereDeterminationConfig
    ) -> str:
        """Determine atmosphere value with headroom priority."""
        # HEADROOM PRIORITY: Use headroom temperature if available
        if config.headroom_temperature:
            return self._map_temperature_to_atmosphere(
                config.headroom_temperature, config.branch_name
            )

        if config.headroom_color_balance and config.headroom_color_balance != "balanced":
            return self._map_color_balance_to_atmosphere(
                config.headroom_color_balance,
                config.visual_formula,
                config.merged_synergies,
                config.branch_name,
            )
        # Fallback to scorer recommendation or hardcoded
        return self._get_fallback_atmosphere(
            config.visual_formula, config.merged_synergies, config.branch_name
        )

    def _map_temperature_to_atmosphere(
        self, temperature: str, branch_name: str
    ) -> str:
        """Map temperature value to atmosphere."""
        temp_lower = temperature.lower()

        if "cool" in temp_lower:
            atmosphere = "Cool"
            logger.info(
                "Branch %s: Using headroom temperature='Cool' (highest priority)",
                branch_name,
            )
        elif "warm" in temp_lower:
            atmosphere = "Warm"
            logger.info(
                "Branch %s: Using headroom temperature='Warm' (highest priority)",
                branch_name,
            )
        else:
            atmosphere = "Neutral"
            logger.info(
                "Branch %s: Using headroom temperature='Neutral' (highest priority)",
                branch_name,
            )

        return atmosphere

    def _map_color_balance_to_atmosphere(
        self,
        color_balance: str,
        visual_formula: Dict[str, Any],
        merged_synergies: Dict[str, str],
        branch_name: str,
    ) -> str:
        """Map color balance value to atmosphere with scorer fallback."""
        cb_lower = color_balance.lower()

        if "cool" in cb_lower:
            atmosphere = "Cool"
            logger.info(
                "Branch %s: Using headroom color_balance='cool' -> 'Cool'",
                branch_name,
            )
            return atmosphere
        if "warm" in cb_lower:
            atmosphere = "Warm"
            logger.info(
                "Branch %s: Using headroom color_balance='warm' -> 'Warm'",
                branch_name,
            )
            return atmosphere
        # No match in color balance, check scorer
        return self._get_fallback_atmosphere(
            visual_formula, merged_synergies, branch_name
        )

    def _get_fallback_atmosphere(
        self,
        visual_formula: Dict[str, Any],
        merged_synergies: Dict[str, str],
        branch_name: str,
    ) -> str:
        """Get fallback atmosphere from scorer or hardcoded defaults."""
        scorer_atmosphere = self.feature_mapper.get_feature_value(
            visual_formula, "atmosphere"
        )
        if scorer_atmosphere:
            logger.info(
                "Branch %s: Using scorer recommendation for "
                "atmosphere='%s' (overriding hardcoded)",
                branch_name,
                scorer_atmosphere,
            )
            return scorer_atmosphere

        atmosphere = merged_synergies.get("atmosphere", "Warm")
        logger.warning(
            "Branch %s: No scorer recommendation for "
            "atmosphere, using hardcoded fallback: '%s'",
            branch_name,
            atmosphere,
        )
        return atmosphere

    def _process_standard_features(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Process non-V2 or no-branch features using standard extraction."""
        # Non-V2 or no branch: Use standard extraction with registry normalization
        product_pos = self.feature_mapper.get_product_position(visual_formula)
        placeholder_values["product_position"] = (
            normalize_feature_value("product_position", product_pos)
            if product_pos
            else product_pos
        )
        atmosphere_value = self._extract_atmosphere(visual_formula)
        placeholder_values["atmosphere"] = (
            normalize_feature_value("atmosphere", atmosphere_value)
            if atmosphere_value
            else atmosphere_value
        )
        # Incorporate color_balance into atmosphere description if available
        color_balance_value = self.feature_mapper.get_color_balance(
            visual_formula
        )
        placeholder_values["atmosphere_description"] = (
            self._map_atmosphere_to_description(
                atmosphere_value, None, product_context, color_balance_value
            )
        )

    def _extract_color_balance(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract color balance with headroom priority."""
        # Color balance: Use headroom > entrance priority (independent from atmosphere)
        # HEADROOM PRIORITY: Extract color_balance with headroom priority
        extracted_color_balance = self.feature_mapper.get_color_balance(
            visual_formula
        )
        if extracted_color_balance:
            placeholder_values["color_balance"] = extracted_color_balance
            logger.info(
                "Using color_balance='%s' (headroom > entrance priority)",
                extracted_color_balance,
            )
        else:
            # Fallback to atmosphere_description for backward compatibility
            placeholder_values["color_balance"] = placeholder_values[
                "atmosphere_description"
            ]

    def _extract_layout_features(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract layout and composition features."""
        # Layout Logic: Check scorer recommendations first for composition/negative space
        # MACRO BRANCH: Use visual filling (tight framing) instead of negative space
        # Avoid 'balanced' space usage (negative performer)
        # UNIFIED REGISTRY: Use canonical value from registry
        if self.v2_mode:
            self._extract_v2_layout_features(visual_formula, placeholder_values)

        self._extract_product_visibility(visual_formula, placeholder_values)
        self._extract_visual_impact(visual_formula, placeholder_values)
        self._extract_remaining_layout_features(visual_formula, placeholder_values)

    def _extract_v2_layout_features(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract V2-specific layout features."""
        if self.branch_name == "high_efficiency":
            # Branch 2 (Macro): Visual filling logic - tight framing, no negative space
            placeholder_values["composition_style"] = ""
            placeholder_values["negative_space_usage"] = ""
            logger.info(
                "Branch 2 (Macro): Using visual filling logic (tight framing), "
                "no negative space"
            )
        else:
            # Branch 1 & 3: Check scorer recommendation for composition_style
            self._extract_v2_composition_style(visual_formula, placeholder_values)
            self._extract_v2_negative_space(visual_formula, placeholder_values)

    def _extract_v2_composition_style(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract composition style for V2 mode."""
        scorer_composition_style = self.feature_mapper.get_feature_value(
            visual_formula, "composition_style"
        )

        if scorer_composition_style:
            placeholder_values["composition_style"] = (
                normalize_feature_value(
                    "composition_style", scorer_composition_style
                )
            )
            logger.info(
                "V2 Production: Using scorer recommendation for "
                "composition_style='%s' (overriding hardcoded 'generous')",
                scorer_composition_style,
            )
        else:
            placeholder_values["composition_style"] = (
                normalize_feature_value(
                    "composition_style", "generous negative space"
                )
            )
            logger.warning(
                "V2 Production: No scorer recommendation for composition_style, "
                "using hardcoded fallback: 'generous negative space'"
            )

    def _extract_v2_negative_space(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract negative space usage for V2 mode."""
        scorer_negative_space = self.feature_mapper.get_feature_value(
            visual_formula, "negative_space_usage"
        )

        if scorer_negative_space:
            placeholder_values["negative_space_usage"] = (
                normalize_feature_value(
                    "negative_space_usage", scorer_negative_space
                )
            )
            logger.info(
                "V2 Production: Using scorer recommendation for "
                "negative_space_usage='%s' (overriding hardcoded 'generous')",
                scorer_negative_space,
            )
        else:
            placeholder_values["negative_space_usage"] = (
                normalize_feature_value(
                    "negative_space_usage", "generous"
                )
            )
            logger.warning(
                "V2 Production: No scorer recommendation for negative_space_usage, "
                "using hardcoded fallback: 'generous'"
            )

    def _extract_product_visibility(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract product visibility with branch-specific handling."""
        # product_visibility: Branch 2 hardcodes "partial" for 5.60 ROAS synergy
        if (
            self.v2_mode
            and self.branch_name == "high_efficiency"
            and placeholder_values.get("product_visibility") == "partial"
        ):
            # Already hardcoded above, use it
            product_visibility = placeholder_values["product_visibility"]
        else:
            product_visibility = (
                self.feature_mapper.get_feature_value_with_default(
                    visual_formula,
                    "product_visibility",
                    "product_visibility",
                )
            )
            # If partial visibility, clarify it means partial view angle, not partial product
            if product_visibility == "partial":
                product_visibility = (
                    "partial view angle (complete product visible, not cropped)"
                )
            placeholder_values["product_visibility"] = product_visibility

    def _extract_visual_impact(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract visual impact with branch-specific handling."""
        # visual_impact: Branch 2 hardcodes "strong" for 5.60 ROAS synergy
        if (
            self.v2_mode
            and self.branch_name == "high_efficiency"
            and placeholder_values.get("visual_impact") == "strong"
        ):
            # Already hardcoded, ensure it's preserved
            logger.info(
                "Branch 2: Hardcoded visual_impact='strong' for 5.60 ROAS synergy"
            )
        else:
            # Extract visual_impact if needed (for tracking/reporting)
            visual_impact = self.feature_mapper.get_feature_value(
                visual_formula, "visual_impact"
            )
            if visual_impact:
                placeholder_values["visual_impact"] = visual_impact

    def _extract_remaining_layout_features(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract remaining layout features with standard mapping."""
        placeholder_values["brightness_distribution"] = (
            self.feature_mapper.get_feature_value_with_default(
                visual_formula,
                "brightness_distribution",
                "brightness_distribution",
            )
        )
        placeholder_values["visual_prominence"] = (
            self.feature_mapper.get_feature_value_with_default(
                visual_formula,
                "visual_prominence",
                "visual_prominence",
            )
        )

    def _extract_interaction_context(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract interaction context with priority rules."""
        # Interaction context: Universal configuration-driven approach
        # Priority: 1. Explicit interaction_context from product_context (highest)
        #          2. Branch-specific hardcoding (ROAS synergies)
        #          3. Feature-based extraction (fallback)
        # V2_MACRO_DETAIL: Skip interaction_context (stripped for close-up focus)
        interaction_context = self._determine_interaction_context(
            visual_formula, product_context
        )
        placeholder_values["interaction_context"] = interaction_context

    def _determine_interaction_context(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],
    ) -> str:
        """Determine interaction context based on priority rules."""
        if self.v2_mode and self.branch_name == "high_efficiency":
            # Branch 2: Macro detail - no interaction context
            return ""
        # Check explicit product_context
        interaction_context = self._get_explicit_interaction_context(
            product_context
        )
        if interaction_context is not None:
            return interaction_context
        # Check scenario configuration
        interaction_context = self._get_scenario_interaction_context(
            product_context
        )
        if interaction_context is not None:
            return interaction_context
        # Check branch-specific hardcoding
        if self.v2_mode and self.branch_name == "golden_ratio":
            return self._get_golden_ratio_interaction_context(product_context)
        # Fallback: Extract from visual_formula features
        return self._get_feature_interaction_context(visual_formula)

    def _get_explicit_interaction_context(
        self, product_context: Dict[str, Any]
    ) -> Optional[str]:
        """Get explicit interaction context from product_context."""
        if (
            "interaction_context" in product_context
            and product_context["interaction_context"]
        ):
            logger.info(
                "Using explicit interaction_context from product_context"
            )
            return product_context["interaction_context"]
        return None

    def _get_scenario_interaction_context(
        self, product_context: Dict[str, Any]
    ) -> Optional[str]:
        """Get interaction context from scenario configuration."""
        if not product_context.get("scenario_name"):
            return None

        scenario_name = product_context.get("scenario_name")
        scenario_config = self._get_scenario_config(
            product_context, scenario_name
        )
        if scenario_config and "interaction_context" in scenario_config:
            interaction_context = scenario_config["interaction_context"]
            logger.info(
                "Using scenario '%s' interaction_context from product config",
                scenario_name,
            )
            return interaction_context

        return None

    def _get_golden_ratio_interaction_context(
        self, product_context: Dict[str, Any]
    ) -> str:
        """Get hardcoded interaction context for golden_ratio branch."""
        market = (
            product_context.get("market", "").upper()
            if product_context
            else ""
        )
        if market == "US":
            interaction_context = (
                "Bare feet walking in the soft-focus background, "
                "signaling floor safety and lifestyle without distracting from the product. "
                "Classic US cleaning ad aesthetic with natural, realistic interaction."
            )
        else:
            interaction_context = (
                "A person visible in the background, walking "
                "barefoot or interacting with a pet, demonstrating floor hygiene. "
                "Natural, realistic interaction without posed appearance."
            )
        logger.info(
            "Branch 1: Hardcoded Person visible interaction for 7.07 ROAS pair "
            "(market: %s)",
            market,
        )
        return interaction_context

    def _get_feature_interaction_context(
        self, visual_formula: Dict[str, Any]
    ) -> str:
        """Get interaction context from visual_formula features."""
        interaction_context = self.feature_mapper.get_interaction_context(
            visual_formula
        )
        # For lean/V2 mode, use concise interaction description
        if self.v2_mode or self.lean_mode:
            interaction_context = self._simplify_interaction_context(
                interaction_context
            )
        return interaction_context

    def _simplify_interaction_context(self, interaction_context: str) -> str:
        """Simplify interaction context for lean/V2 mode."""
        if not interaction_context:
            return ""

        if "barefoot" in interaction_context.lower():
            return "person walking barefoot or with pet in background"
        return "person in background"

    def _extract_static_context_with_mode(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract static context/background based on mode."""
        # Background/context mapping
        # V2_MACRO_DETAIL: Skip static_context (stripped for close-up focus)
        if self.v2_mode and self.branch_name == "high_efficiency":
            placeholder_values["static_context"] = ""
        elif self.lean_mode:
            # Lean Anchor: Concise background
            placeholder_values["static_context"] = "sunlit minimalist home"
        else:
            placeholder_values["static_context"] = self._extract_static_context(
                visual_formula
            )

    def _extract_physical_state_and_grounding(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Extract physical state and grounding information."""
        # STRICT MODULAR TEMPLATE: Physical State Slot
        # Locked to: Leaning, Flat, or Macro-focused (no contradictions)
        # NOTE: Physical state is extracted AFTER branch-specific hardcoding
        # to ensure correct mapping (e.g., Branch 3 -> Flat, not Leaning)
        physical_state = self._extract_physical_state(product_context)
        placeholder_values["physical_state"] = physical_state
        # Get grounding_method (config file has highest priority)
        grounding_method = self._get_grounding_method(product_context)
        # Get placement_target for branch-specific logic
        placement_target = self._get_placement_target_for_branch(
            visual_formula, product_context, grounding_method, placeholder_values
        )
        # Map physical_state to descriptive string for template
        placeholder_values["physical_state_description"] = (
            self._map_physical_state_to_description(
                physical_state,
                placement_target,
                placeholder_values,
                grounding_method,
            )
        )
        # Get grounding instruction
        placeholder_values["grounding_instruction"] = (
            self._get_grounding_instruction(placeholder_values)
        )

    def _get_grounding_method(
        self, product_context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get grounding_method with priority: config file > scenario config.

        Priority:
        1. Top-level product_context (from config file)
        2. Scenario config (from product_context JSON)
        """
        grounding_method = product_context.get("grounding_method")
        if not grounding_method:
            scenario_name = product_context.get("scenario_name")
            scenario_config = self._get_scenario_config(
                product_context, scenario_name
            )
            if scenario_config:
                grounding_method = scenario_config.get("grounding_method")
        return grounding_method

    def _get_placement_target_for_branch(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],  # pylint: disable=unused-argument
        grounding_method: Optional[str],
        placeholder_values: Dict[str, Any],
    ) -> str:
        """
        Get placement_target based on V2 branch and grounding method.

        Branch 1: Will be hardcoded to "natural home environment elements"
        Branch 3: Must be hardcoded to "low-clearance furniture" BEFORE physical state mapping
        """
        if self.v2_mode and self.branch_name == "cool_peak":
            # Branch 3: Set placement_target BEFORE physical state mapping
            placement_target = "low-clearance furniture"
        elif self.v2_mode and self.branch_name == "golden_ratio":
            # Branch 1: Set placement_target for product-in-environment
            # Override with scenario-specific grounding if available
            if grounding_method == "base_on_floor":
                placement_target = "floor surface"
            else:
                placement_target = "modern sofa, wall, or kitchen counter"
        elif placeholder_values.get("placement_target"):
            placement_target = placeholder_values["placement_target"]
        else:
            placement_target = self.feature_mapper.get_placement_target(
                visual_formula
            )
            if not placement_target:
                placement_target = "modern sofa or wall"

        placeholder_values["placement_target"] = placement_target
        return placement_target

    def _get_grounding_instruction(
        self, placeholder_values: Dict[str, Any]
    ) -> str:
        """
        Get grounding instruction based on mode and interaction context.

        V2_MACRO_DETAIL: Skip grounding_instruction (not used in macro template)
        V2_FLAT_TECH: Use sliding-under-furniture logic (handled in template)
        """
        if self.v2_mode and self.branch_name == "high_efficiency":
            # Macro detail doesn't use grounding_instruction
            return ""

        if not (self.v2_mode or self.lean_mode):
            # Not used in full template, but provide empty string to avoid errors
            return ""
        # Default to "resting/leaning" state for lifestyle context
        interaction_context = placeholder_values.get("interaction_context", "")
        return self._determine_grounding_instruction_for_interaction(
            interaction_context
        )

    def _determine_grounding_instruction_for_interaction(
        self, interaction_context: str
    ) -> str:
        """Determine grounding instruction based on interaction context."""
        if not interaction_context or "person" not in interaction_context.lower():
            # Default: Resting position
            return (
                "mop shown in realistic leaning position against modern "
                "sofa or wall, firmly grounded with contact shadows"
            )
        # Person present - check if actively using or in background
        if "background" in interaction_context.lower():
            # Option B: Self-Standing/Resting (Lifestyle)
            return (
                "mop shown in realistic leaning position against modern "
                "sofa or wall, firmly grounded with firm contact shadows "
                "that define the unit's shape and grounding"
            )
        # Option A: In-Use (High Conversion)
        return (
            "person's hand visible gripping handle, actively using "
            "mop to clean floor, firmly grounded with contact shadows"
        )

    def _build_cmf_core(
        self,
        product_context: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Build CMF core with material and color constraint."""
        # STRICT MODULAR TEMPLATE: CMF_Core Slot
        # Always at position 200-300 for subject primacy
        # Combines material_finish and color_constraint
        # CRITICAL: Prevent white drift by explicitly defining silver/black parts
        material_finish = placeholder_values.get("material_finish", "").strip()
        # Priority 1: Check for explicit color_constraint field
        explicit_color_constraint = product_context.get("color_constraint")
        if explicit_color_constraint:
            color_constraint = str(explicit_color_constraint).strip()
            logger.debug(
                "Using explicit color_constraint from product_context: %s",
                (
                    explicit_color_constraint[:50]
                    if len(explicit_color_constraint) > 50
                    else explicit_color_constraint
                ),
            )
        else:
            # Priority 2: No color constraints (universal)
            color_constraint = ""
            logger.debug(
                "No explicit color_constraint field found, using empty string. "
                "To add color constraints, set 'color_constraint' in product_context."
            )
        # Check for prevent_white_drift flag
        product_ctx = product_context or (
            self.product_context if hasattr(self, "product_context") else {}
        )
        scenario_name = product_ctx.get("scenario_name")
        prevent_white_drift = False
        if scenario_name:
            scenario_config = self._get_scenario_config(
                product_ctx, scenario_name
            )
            if scenario_config:
                prevent_white_drift = scenario_config.get(
                    "prevent_white_drift", False
                )
        prevent_white_drift = prevent_white_drift or product_ctx.get(
            "prevent_white_drift", False
        )

        if color_constraint:
            cmf_core = f"{material_finish}; {color_constraint}"
        else:
            # If color_constraint is empty, use only material_finish
            cmf_core = material_finish if material_finish else "premium finish"
        # Strengthen CMF to prevent white drift: explicitly define silver/black parts
        if (
            prevent_white_drift
            and "silver" in material_finish.lower()
            and "black" in material_finish.lower()
        ):
            # Add explicit color definitions to prevent white drift
            cmf_core = (
                f"{cmf_core}. "
                f"CRITICAL: Exterior casing must be premium brushed metallic "
                f"silver and deep charcoal black ONLY - "
                f"Maintain distinct silver and black color separation with "
                f"defined structural shadows."
            )

        placeholder_values["cmf_core"] = cmf_core
        placeholder_values["color_constraint"] = color_constraint

    def _process_v2_enhanced_features(
        self,
        visual_formula: Dict[str, Any],
        product_context: Dict[str, Any],  # pylint: disable=unused-argument
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Process V2 Enhanced ROAS features (branch-specific feature handling)."""
        # V2 Enhanced ROAS features (must be before template rendering)
        # Tri-Template Architecture: Branch-specific feature handling
        if self.branch_name == "high_efficiency":
            self._process_v2_macro_features(placeholder_values)
        elif self.branch_name == "cool_peak":
            self._process_v2_flat_tech_features(
                visual_formula, placeholder_values
            )
        elif self.branch_name == "golden_ratio":
            self._process_v2_wide_scene_features(
                visual_formula, placeholder_values
            )
        else:
            # Legacy V2: Use standard extraction
            self._process_v2_legacy_features(
                visual_formula, placeholder_values
            )

    def _process_v2_macro_features(
        self,
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Process V2_MACRO_DETAIL features (high_efficiency branch)."""
        # V2_MACRO_DETAIL: SCENE-FIRST NARRATIVE STRUCTURE
        # Layer 1: Scene - Global environment and lighting setup
        # Layer 2: Action - Product interaction with floor
        # Layer 3: Anchor - Consistency descriptors (Geometry & CMF)
        # Layer 4: Polish - Step 2 aesthetic modifiers
        # FEATURE-EMBEDDED: All layers include feature tags for extractor visibility
        action_desc = self._extract_macro_action_description(placeholder_values)
        consistency_anchor = self._extract_macro_consistency_anchor(
            placeholder_values
        )
        aesthetic_polish = self._extract_macro_aesthetic_polish(
            placeholder_values
        )

        placeholder_values["scene_description"] = ""
        placeholder_values["action_description"] = action_desc
        placeholder_values["consistency_anchor"] = consistency_anchor
        placeholder_values["aesthetic_polish"] = aesthetic_polish
        # Geometric Constraint: Placed as second sentence for maximum attention
        placeholder_values["geometric_constraint"] = (
            "Strictly maintain the exact geometric structure and proportions of Image 1. "
        )

        logger.info(
            "V2_MACRO_DETAIL: Scene-First Narrative Structure applied "
            "(Scene -> Action -> Anchor -> Polish) with Feature-Embedded tags"
        )

    def _process_v2_flat_tech_features(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Process V2_FLAT_TECH features (cool_peak branch)."""
        # V2_FLAT_TECH: placement_target already set BEFORE physical_state mapping
        if not placeholder_values.get("placement_target"):
            placeholder_values["placement_target"] = "low-clearance furniture"
        # Branch 3: Force generous negative space (ROAS 2.94)
        if placeholder_values.get("composition_style"):
            composition_style = placeholder_values["composition_style"]
        else:
            composition_style = self.feature_mapper.get_composition_style(
                visual_formula
            )
            # Force generous if not already set and avoid balanced
            if (
                not composition_style
                or "balanced" in composition_style.lower()
            ):
                composition_style = "generous negative space"
                logger.info(
                    "Branch 3: Forced 'generous negative space' (ROAS 2.94), "
                    "avoiding 'balanced'"
                )

        lighting_detail = self.feature_mapper.get_lighting_detail(
            visual_formula
        )
        environment_objects = (
            self.feature_mapper.get_environment_objects(visual_formula)
        )
        # Conditional suffixes
        if composition_style:
            placeholder_values["composition_style_suffix"] = (
                f" with {composition_style}"
            )
        else:
            placeholder_values["composition_style_suffix"] = ""

        if lighting_detail:
            placeholder_values["lighting_detail_suffix"] = f", {lighting_detail}"
        else:
            placeholder_values["lighting_detail_suffix"] = ""

        if environment_objects:
            env_text = environment_objects
            for prefix in ["placed ", "positioned ", "situated "]:
                if env_text.lower().startswith(prefix):
                    env_text = env_text[len(prefix) :]
                    break
            placeholder_values["environment_objects_suffix"] = (
                f", featuring {env_text}"
            )
        else:
            placeholder_values["environment_objects_suffix"] = ""

        logger.info(
            "V2_FLAT_TECH: Physical logic synced "
            "(sliding under furniture, not leaning)"
        )

    def _process_v2_wide_scene_features(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Process V2_WIDE_SCENE features (golden_ratio branch)."""
        # V2_WIDE_SCENE (golden_ratio): Full feature injection
        # Hardcode product-in-environment for 10.7 ROAS synergy
        # Force product-in-environment relationship
        # NOTE: placement_target already set in physical state section above
        environment_objects = (
            "integrated into a natural home environment with contextual "
            "elements"
        )
        # Ensure placement_target is set (already set above, but verify)
        placement_target = self._ensure_placement_target(placeholder_values)
        composition_style = self._get_wide_scene_composition_style(
            visual_formula, placeholder_values
        )
        lighting_detail = self.feature_mapper.get_lighting_detail(
            visual_formula
        )
        # Cleanup and set placeholder values
        self._set_wide_scene_placeholders(
            WideScenePlaceholdersConfig(
                placement_target=placement_target,
                composition_style=composition_style,
                lighting_detail=lighting_detail,
                environment_objects=environment_objects,
                placeholder_values=placeholder_values,
            )
        )

        logger.info(
            "V2_WIDE_SCENE: Full feature injection for environment integration"
        )

    def _ensure_placement_target(
        self, placeholder_values: Dict[str, Any]
    ) -> str:
        """Ensure placement_target is set, using default if needed."""
        if not placeholder_values.get("placement_target"):
            placement_target = "natural home environment elements"
            placeholder_values["placement_target"] = placement_target
            logger.info(
                "Branch 1: Hardcoded product-in-environment for 10.7 ROAS synergy"
            )
            return placement_target
        return placeholder_values["placement_target"]

    def _get_wide_scene_composition_style(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> str:
        """Get composition style for wide scene, forcing generous if needed."""
        if placeholder_values.get("composition_style"):
            return placeholder_values["composition_style"]

        composition_style = self.feature_mapper.get_composition_style(
            visual_formula
        )
        # Force generous if not already set and avoid balanced
        if (
            not composition_style
            or "balanced" in composition_style.lower()
        ):
            composition_style = "generous negative space"
            logger.info(
                "Forced 'generous negative space' (ROAS 2.94), "
                "avoiding 'balanced'"
            )
        return composition_style

    def _set_wide_scene_placeholders(
        self, config: WideScenePlaceholdersConfig
    ) -> None:
        """Set all wide scene placeholder values with proper formatting."""
        # Handle empty values to prevent broken syntax
        config.placeholder_values["placement_target"] = (
            config.placement_target if config.placement_target else "modern sofa or wall"
        )
        # Conditional suffixes to avoid trailing prepositions and double commas
        config.placeholder_values["composition_style_suffix"] = (
            f" with {config.composition_style}" if config.composition_style else ""
        )
        config.placeholder_values["lighting_detail_suffix"] = (
            f", {config.lighting_detail}" if config.lighting_detail else ""
        )
        config.placeholder_values["environment_objects_suffix"] = (
            self._format_environment_objects_suffix(config.environment_objects)
        )

    def _format_environment_objects_suffix(self, environment_objects: str) -> str:
        """Format environment objects suffix, removing leading prefixes."""
        if not environment_objects:
            return ""
        # Remove leading verbs/prepositions from environment_objects
        env_text = environment_objects
        for prefix in ["placed ", "positioned ", "situated "]:
            if env_text.lower().startswith(prefix):
                env_text = env_text[len(prefix) :]
                break
        return f", featuring {env_text}"

    def _process_v2_legacy_features(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, Any],
    ) -> None:
        """Process legacy V2 features (no specific branch)."""
        # Legacy V2: Use standard extraction
        placement_target = self.feature_mapper.get_placement_target(
            visual_formula
        )
        environment_objects = (
            self.feature_mapper.get_environment_objects(visual_formula)
        )
        # Composition style: Use hardcoded generous negative space if set
        # Otherwise extract from formula
        if placeholder_values.get("composition_style"):
            composition_style = placeholder_values["composition_style"]
        else:
            composition_style = self.feature_mapper.get_composition_style(
                visual_formula
            )
            # Force generous if not already set and avoid balanced
            if (
                not composition_style
                or "balanced" in composition_style.lower()
            ):
                composition_style = "generous negative space"
                logger.info(
                    "Forced 'generous negative space' (ROAS 2.94), "
                    "avoiding 'balanced'"
                )

        lighting_detail = self.feature_mapper.get_lighting_detail(
            visual_formula
        )
        # Cleanup: Handle empty values to prevent broken syntax
        placeholder_values["placement_target"] = (
            placement_target
            if placement_target
            else "modern sofa or wall"
        )
        # Conditional suffixes to avoid trailing prepositions and double commas
        if composition_style:
            placeholder_values["composition_style_suffix"] = (
                f" with {composition_style}"
            )
        else:
            placeholder_values["composition_style_suffix"] = ""

        if lighting_detail:
            placeholder_values["lighting_detail_suffix"] = f", {lighting_detail}"
        else:
            placeholder_values["lighting_detail_suffix"] = ""

        if environment_objects:
            # Remove leading verbs/prepositions from environment_objects
            env_text = environment_objects
            for prefix in ["placed ", "positioned ", "situated "]:
                if env_text.lower().startswith(prefix):
                    env_text = env_text[len(prefix) :]
                    break
            placeholder_values["environment_objects_suffix"] = (
                f", featuring {env_text}"
            )
        else:
            placeholder_values["environment_objects_suffix"] = ""


    def _extract_subject_description(
        self, product_context: Dict[str, Any]
    ) -> str:
        """
        Extract expanded subject description with product completeness.

        Builds a descriptive block that mentions:
        - Product name
        - Complete assembled unit (handle, body, tanks, mop head)
        - Prevents showing only individual parts

        Priority:
        1. Build from detailed_product_context if available
        2. Build from product_name with standard completeness description
        3. Raises ValueError if product_name missing

        Args:
            product_context: Product context dict

        Returns:
            Expanded subject description string

        Raises:
            ValueError: If product_name is missing
        """
        product_name = self._get_product_name(product_context)
        # Check for detailed product context that mentions components
        detailed_context = product_context.get("detailed_product_context") or ""
        additional_context = product_context.get("additional_context") or ""
        context_text = (detailed_context + " " + additional_context).lower()

        if self._has_completeness_info(context_text):
            return self._build_detailed_subject_description(
                product_name, context_text, product_context
            )
        # Default completeness description
        return self._build_default_subject_description(
            product_name, product_context
        )

    def _get_product_name(self, product_context: Dict[str, Any]) -> str:
        """Get product name from context, raising error if missing."""
        product_name = product_context.get("product_name") or product_context.get(
            "name"
        )

        if not product_name:
            raise ValueError(
                "product_name is required but missing from product_context. "
                "Provide 'product_name' or 'name' in product_context."
            )
        return product_name

    def _has_completeness_info(self, context_text: str) -> bool:
        """Check if context contains component information."""
        keywords = [
            "handle",
            "body",
            "tank",
            "assembled",
            "complete",
            "unit",
            "components",
        ]
        return any(keyword in context_text for keyword in keywords)

    def _build_detailed_subject_description(
        self, product_name: str, context_text: str, product_context: Dict[str, Any]
    ) -> str:
        """Build subject description from detailed context."""
        components = self._extract_components_from_context(context_text)
        components_str = ", ".join(components)

        uniqueness_text = product_context.get(
            "uniqueness_text", "a single, solitary unit"
        )

        description = (
            f"{product_name}, {uniqueness_text}, complete assembled unit "
            f"featuring {components_str}, "
            "all components connected and functioning as one integrated product"
        )
        # Add maintenance-free architecture emphasis if mentioned
        description = self._add_maintenance_info(description, context_text)
        # Add handle text if specified
        description = self._add_handle_text(description, product_context)

        return description

    def _extract_components_from_context(self, context_text: str) -> list:
        """Extract component descriptions from context text."""
        components = []
        # Upper Section: Clean water tank
        if "clean water" in context_text or "1l" in context_text:
            components.append(
                "Upper Section: slim, clear 1L clean water tank "
                "integrated into the metallic handle assembly"
            )
        # Base Section: Dirty water tank
        if "dirty water" in context_text or "500ml" in context_text:
            tank_description = self._build_tank_description(context_text)
            components.append(tank_description)
        # Mop Head
        if "mop head" in context_text or "mop base" in context_text:
            components.append(
                "mop head with transparent front cover showcasing the "
                "signature vibrant emerald green cleaning belt"
            )
        # Fallback if no tank components found
        if not components or "tank" not in " ".join(components):
            components.append("water tanks")

        return components

    def _build_tank_description(self, context_text: str) -> str:
        """Build dirty water tank description based on context."""
        # Check for pipe-free design
        if self._has_pipe_free_design(context_text):
            tank_description = (
                "Base Section: innovative 500ml pipe-free dirty water tank "
                "at the very bottom, integrated with the mop head, "
                "featuring a direct-intake module at floor level"
            )
            # Add triple-waste separation if mentioned
            if self._has_triple_waste_separation(context_text):
                tank_description += (
                    " with visible triple-waste separation "
                    "(wet waste, dry debris, and hair)"
                )
            # Add CMF for tank
            tank_description += (
                ", featuring a transparent smoke-gray viewing window "
                "to reveal the modular internal structure"
            )
            return tank_description
        # Fallback to standard description
        return "Base Section: 500ml dirty water tank at the bottom"

    def _has_pipe_free_design(self, context_text: str) -> bool:
        """Check if context mentions pipe-free design."""
        pipe_free_keywords = [
            "pipe-free",
            "pipe free",
            "zero-distance",
            "modular",
        ]
        return any(keyword in context_text for keyword in pipe_free_keywords)

    def _has_triple_waste_separation(self, context_text: str) -> bool:
        """Check if context mentions triple-waste separation."""
        waste_keywords = [
            "triple-waste",
            "triple waste",
            "waste separation",
        ]
        return any(keyword in context_text for keyword in waste_keywords)

    def _has_maintenance_info(self, context_text: str) -> bool:
        """Check if context mentions maintenance-free features."""
        maintenance_keywords = [
            "maintenance-free",
            "maintenance free",
            "zero-maintenance",
            "minute-to-rinse",
        ]
        return any(keyword in context_text for keyword in maintenance_keywords)

    def _add_maintenance_info(self, description: str, context_text: str) -> str:
        """Add maintenance-free architecture description if mentioned."""
        if self._has_maintenance_info(context_text):
            description += (
                ". The design emphasizes a hygienic, maintenance-free "
                "architecture with easily detachable, modular components "
                "that showcase zero-maintenance ease of use"
            )
        return description

    def _add_handle_text(
        self, description: str, product_context: Dict[str, Any]
    ) -> str:
        """Add handle text description if specified."""
        handle_text = product_context.get("handle_text")
        if handle_text:
            description += (
                f', with the text "{handle_text}" crisply printed on the handle, '
                "oriented vertically along the handle's axis, arranged from top to bottom"
            )
        return description

    def _build_default_subject_description(
        self, product_name: str, product_context: Dict[str, Any]
    ) -> str:
        """Build default subject description without detailed context."""
        uniqueness_text = product_context.get(
            "uniqueness_text", "a single, solitary unit"
        )
        description = (
            f"{product_name}, {uniqueness_text}, complete assembled unit "
            "with handle, main body, "
            "water tanks, and mop head, all components connected and functioning "
            "as one integrated product"
        )
        # Add handle text if specified
        description = self._add_handle_text(description, product_context)
        return description

    def _extract_material_finish(
        self,
        product_context: Dict[str, Any],
    ) -> str:
        """
        Extract material/finish from product context.

        Priority:
        1. material (explicit field)
        2. finish (explicit field)
        3. Extract from detailed_product_context (look for "premium", "hygienic")
        4. default

        Args:
            product_context: Product context dict

        Returns:
            Material/finish string
        """
        # Check explicit fields first (early return acceptable for direct value)
        material = product_context.get("material") or product_context.get("finish")
        if material:
            return str(material).strip()
        # Extract from detailed context
        detailed_context = product_context.get("detailed_product_context") or ""
        additional_context = product_context.get("additional_context") or ""
        context_text = (detailed_context + " " + additional_context).lower()
        # Build result through condition matching
        result = self._determine_material_from_context(context_text)
        if result:
            return result
        # Default: Generic premium finish
        return self.defaults.get("material_finish", "premium finish")

    def _determine_material_from_context(self, context_text: str) -> Optional[str]:
        """Determine material finish from context text."""
        has_metallic = "metallic" in context_text or "silver" in context_text
        has_black = "black" in context_text or "charcoal" in context_text
        has_white = "matte" in context_text or "white" in context_text
        has_premium = "premium" in context_text
        has_hygienic = "hygienic" in context_text

        if self.lean_mode:
            return self._get_lean_material(has_metallic, has_white, has_black, has_premium)
        # V2 mode: More detailed descriptions
        return self._get_v2_material(has_metallic, has_black, has_premium, has_hygienic)

    def _get_lean_material(
        self, has_metallic: bool, has_white: bool, has_black: bool, has_premium: bool  # pylint: disable=unused-argument
    ) -> str:
        """Get lean mode material description."""
        if not has_metallic:
            return "premium finish"
        if has_white and has_black:
            return "multi-toned silver/white/black casing"
        if has_white:
            return "silver/white casing"
        return "silver casing"

    def _get_v2_material(
        self, has_metallic: bool, has_black: bool, has_premium: bool, has_hygienic: bool
    ) -> str:
        """Get V2 mode material description."""
        if has_metallic and has_black:
            return "premium brushed metallic silver and deep charcoal black casing"
        if has_metallic:
            return "premium brushed metallic silver casing"
        if has_premium and has_hygienic:
            return "premium, hygienic appearance"
        if has_premium:
            return "premium finish"
        if has_hygienic:
            return "hygienic, clean appearance"
        return None

    def _extract_static_context(
        self,
        visual_formula: Dict[str, Any],
    ) -> str:
        """
        Extract static context (background) from visual_formula.

        Maps from background_type feature or uses default.

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Static context string
        """
        # Try to get from background_type feature
        background_type = self.feature_mapper.get_feature_value(
            visual_formula,
            "background_type",
        )

        if background_type:
            # Map common background_type values to descriptive strings
            background_mapping = {
                "solid-color": "clean solid color background",
                "gradient": "clean gradient background",
                "pattern": "subtle pattern background",
                "photographic": "professional photographic background",
                "abstract": "abstract artistic background",
                "textured": "textured background",
            }
            return background_mapping.get(
                background_type.lower(),
                f"{background_type} background",
            )
        # Use default (Meta Ad optimized: sunlit, modern minimalist home)
        default_context = "sunlit, modern minimalist home environment"
        return self.defaults.get("static_context", default_context)

    def _extract_completeness_instruction(
        self, product_context: Dict[str, Any]
    ) -> str:
        """
        Extract product completeness instruction to prevent hallucinations.

        Ensures the prompt explicitly states that the complete product
        must be shown, not just individual parts.

        Args:
            product_context: Product context dict

        Returns:
            Completeness instruction string
        """
        detailed_context = product_context.get("detailed_product_context") or ""
        additional_context = product_context.get("additional_context") or ""
        context_text = (detailed_context + " " + additional_context).lower()
        # Check if context has specific completeness requirements
        if any(
            phrase in context_text
            for phrase in [
                "complete product",
                "assembled unit",
                "all components",
                "integrated",
            ]
        ):
            return (
                "Show the complete product as an assembled unit with all components "
                "connected. DO NOT show only individual parts separately. The product "
                "must appear as one integrated, functional unit"
            )
        # Default completeness instruction (universal)
        return (
            "Show the complete product as an assembled unit with all components "
            "connected. DO NOT show only individual parts separately. "
            "All components must appear as one integrated, functional product"
        )

    def _extract_atmosphere(self, visual_formula: Dict[str, Any]) -> str:
        """
        STRICT MODULAR TEMPLATE: Atmosphere Slot Normalization.

        Consolidates color_balance and temperature into a single 'Atmosphere Slot'.
        Priority logic: temperature > color_balance (prevents 'Balanced' drift).

        Returns locked value: "Cool", "Warm", or "Neutral".

        Args:
            visual_formula: Visual formula JSON dict

        Returns:
            Atmosphere value: "Cool", "Warm", or "Neutral"
        """
        # Priority 1: Check temperature (highest priority)
        temperature_feature = self.feature_mapper.get_feature_value(
            visual_formula, "temperature"
        )

        if temperature_feature:
            temp_lower = temperature_feature.lower()
            if "cool" in temp_lower:
                logger.info(
                    "Atmosphere Slot: Temperature='Cool' -> locked to 'Cool' "
                    "(ignoring color_balance)"
                )
                return "Cool"
            if "warm" in temp_lower:
                logger.info(
                    "Atmosphere Slot: Temperature='Warm' -> locked to 'Warm' "
                    "(ignoring color_balance)"
                )
                return "Warm"
        # Priority 2: Check color_balance (only if temperature not provided)
        color_balance_feature = self.feature_mapper.get_color_balance(
            visual_formula
        )

        if color_balance_feature and color_balance_feature != "balanced":
            cb_lower = color_balance_feature.lower()
            if "cool" in cb_lower:
                logger.info(
                    "Atmosphere Slot: color_balance='cool' -> locked to 'Cool'"
                )
                return "Cool"
            if "warm" in cb_lower:
                logger.info(
                    "Atmosphere Slot: color_balance='warm' -> locked to 'Warm'"
                )
                return "Warm"
        # Default: Neutral (no contradictory adjectives)
        logger.info(
            "Atmosphere Slot: No temperature/color_balance -> locked to 'Neutral'"
        )
        return "Neutral"

    def _extract_physical_state(
        self, product_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        STRICT MODULAR TEMPLATE: Physical State Slot with Configuration Override.

        Priority:
        1. product_context["physical_states"][branch_name] (branch-specific override)
        2. product_context["physical_state"] (generic override)
        3. Branch defaults (for template compatibility)

        Default values based on branch:
        - high_efficiency -> "Macro-focused" (default)
        - cool_peak -> "Flat" (default)
        - golden_ratio (or default) -> "Leaning" (default)

        Args:
            product_context: Optional product context for configuration override

        Returns:
            Physical state value: "Leaning", "Flat", or "Macro-focused"
        """
        # Check for configuration override (universal)
        product_ctx = product_context or (
            self.product_context if hasattr(self, "product_context") else {}
        )
        # Priority 1: Branch-specific override
        if product_ctx and "physical_states" in product_ctx:
            branch_physical = product_ctx["physical_states"].get(
                self.branch_name
            )
            if branch_physical:
                logger.info(
                    "Physical State: Using configured override for %s: %s",
                    self.branch_name,
                    branch_physical,
                )
                return branch_physical
        # Priority 2: Generic override
        if product_ctx and "physical_state" in product_ctx:
            logger.info(
                "Physical State: Using configured override: %s",
                product_ctx["physical_state"],
            )
            return product_ctx["physical_state"]
        # Priority 3: Branch defaults (for template compatibility)
        if self.branch_name == "high_efficiency":
            logger.info(
                "Physical State Slot: high_efficiency -> default 'Macro-focused' "
                "(template compatibility)"
            )
            return "Macro-focused"
        if self.branch_name == "cool_peak":
            logger.info(
                "Physical State Slot: cool_peak -> default 'Flat' (template compatibility)"
            )
            return "Flat"
        # golden_ratio or default
        logger.info(
            "Physical State Slot: golden_ratio/default -> default 'Leaning' "
            "(template compatibility)"
        )
        return "Leaning"

    def _get_scenario_config(
        self,
        product_context: Dict[str, Any],
        scenario_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get scenario configuration from product_context.

        Scenarios are defined in product_context["scenarios"] and can be selected
        by name. This allows product-specific scenarios without hardcoding.

        Args:
            product_context: Product context dictionary
            scenario_name: Optional scenario name to look up

        Returns:
            Scenario configuration dict or None
        """
        scenarios = product_context.get("scenarios", {})
        if scenario_name and scenario_name in scenarios:
            return scenarios[scenario_name]
        return None

    def _map_atmosphere_to_description(
        self,
        atmosphere: str,
        branch_name: Optional[str] = None,
        product_context: Optional[Dict[str, Any]] = None,
        color_balance: Optional[str] = None,
    ) -> str:
        """
        Map atmosphere value to lighting temperature description (NOT color properties).

        CRITICAL: Atmosphere controls LIGHTING TEMPERATURE only, NOT color tinting.
        - Contrast and saturation are ALWAYS high (separate from atmosphere)
        - Atmosphere only affects lighting color temperature (warm/cool light)
        - Product colors must remain unchanged regardless of atmosphere

        Args:
            atmosphere: Atmosphere value ("Cool", "Warm", or "Neutral")
            branch_name: Optional branch identifier for scene-specific tones
            product_context: Product context dict
            color_balance: Color balance for palette

        Returns:
            Descriptive string for lighting temperature and color properties
        """
        product_ctx = product_context or (
            self.product_context if hasattr(self, "product_context") else {}
        )
        market = product_ctx.get("market", "").upper() if product_ctx else ""
        # Build base lighting for branch
        base_lighting = self._get_base_lighting_for_branch(
            branch_name, product_ctx
        )
        # Add color palette suffix if specified
        color_suffix = self._get_color_palette_suffix(color_balance)
        # Apply atmosphere adjustments
        return self._apply_atmosphere_to_lighting(
            base_lighting, atmosphere, market, color_suffix
        )

    def _get_base_lighting_for_branch(
        self, branch_name: Optional[str], product_ctx: Dict[str, Any]
    ) -> str:
        """Get base lighting description for specific branch."""
        if branch_name == "golden_ratio":
            return self._get_golden_ratio_lighting(product_ctx)
        if branch_name == "cool_peak":
            return self._get_cool_peak_lighting()
        if branch_name == "high_efficiency":
            return self._get_high_efficiency_lighting()
        # Default lighting
        return (
            "ultra-bright high-key aesthetics with over-illuminated interiors, "
            "eliminate heavy shadows, vibrant rich color grading with technicolor palette, "
            "powerful and vivid colors, high-key lighting with clean, vibrant highlights"
        )

    def _get_golden_ratio_lighting(self, product_ctx: Dict[str, Any]) -> str:
        """Get lighting for golden_ratio (life scene)."""
        enhanced = product_ctx.get("enhanced_lighting", False)
        lighting_modes = product_ctx.get("lighting_modes", {})
        mode_name = product_ctx.get(
            "lighting_mode", "commercial_high_key" if enhanced else None
        )
        market = product_ctx.get("market", "").upper()

        if market == "US":
            return self._get_us_golden_ratio_lighting(
                enhanced, lighting_modes, mode_name
            )
        return self._get_global_golden_ratio_lighting(
            enhanced, lighting_modes, mode_name
        )

    def _get_us_golden_ratio_lighting(
        self, enhanced: bool, lighting_modes: Dict, mode_name: Optional[str]
    ) -> str:
        """Get US market golden_ratio lighting."""
        if mode_name and mode_name in lighting_modes:
            style = lighting_modes[mode_name].get('style', 'commercial_grade')
            return (
                f"completely transparent lighting aesthetic with highlights overflowing, "
                f"super-luminous commercial high-key lighting with "
                f"luminous ambient fill that feels like natural daylight, "
                f"eliminating any dim or muddy areas, maximum brightness "
                f"creating an airy, bright, and expensive feel, "
                f"transparent highlights spilling over edges creating a "
                f"luminous glow, "
                f"firm contact shadows define product shape and "
                f"grounding, "
                f"vibrant rich color grading with technicolor palette, "
                f"powerful and vivid colors, "
                f"{style} presentation"
            )
        if enhanced:
            return (
                "completely transparent lighting aesthetic with highlights overflowing, "
                "super-luminous commercial high-key lighting with "
                "luminous ambient fill that feels like natural daylight, "
                "eliminating any dim or muddy areas, maximum brightness "
                "creating an airy, bright, and expensive feel, "
                "transparent highlights spilling over edges creating a "
                "luminous glow, "
                "firm contact shadows define product shape and grounding, "
                "vibrant rich color grading with technicolor palette, "
                "powerful and vivid colors, commercial-grade presentation"
            )
        return (
            "completely transparent lighting aesthetic with highlights overflowing, "
            "super-luminous ultra-bright high-key aesthetics with "
            "luminous ambient fill that feels like natural daylight, "
            "eliminating any dim or muddy areas, over-illuminated "
            "interiors creating an airy, bright, and expensive feel, "
            "transparent highlights spilling over edges creating a "
            "luminous glow, "
            "eliminate heavy shadows, vibrant rich color grading "
            "with technicolor palette, "
            "powerful and vivid colors"
        )

    def _get_global_golden_ratio_lighting(
        self, enhanced: bool, lighting_modes: Dict, mode_name: Optional[str]
    ) -> str:
        """Get global market golden_ratio lighting."""
        if mode_name and mode_name in lighting_modes:
            style = lighting_modes[mode_name].get('style', 'commercial_grade')
            return (
                f"commercial high-key lighting with maximum brightness, "
                f"firm contact shadows define product shape and "
                f"grounding, "
                f"vibrant rich color grading with technicolor palette, "
                f"powerful and vivid colors, "
                f"{style} presentation"
            )
        if enhanced:
            return (
                "commercial high-key lighting with maximum brightness, "
                "firm contact shadows define product shape and grounding, "
                "vibrant rich color grading with technicolor palette, "
                "powerful and vivid colors, commercial-grade presentation"
            )
        return (
            "ultra-bright high-key aesthetics with over-illuminated interiors, "
            "eliminate heavy shadows, vibrant rich color grading "
            "with technicolor palette, "
            "powerful and vivid colors"
        )

    def _get_cool_peak_lighting(self) -> str:
        """Get lighting for cool_peak (creative scene)."""
        return (
            "ultra-bright high-key aesthetics with over-illuminated interiors, "
            "eliminate heavy shadows, vibrant rich color grading with technicolor palette, "
            "powerful and vivid colors, high contrast lighting with "
            "vibrant highlights for visual impact"
        )

    def _get_high_efficiency_lighting(self) -> str:
        """Get lighting for high_efficiency (close-up scene)."""
        return (
            "texture-focused lighting with high contrast emphasizing "
            "material details and surface textures, "
            "high saturation for rich color rendering"
        )

    def _get_color_palette_suffix(self, color_balance: Optional[str]) -> str:
        """Get color palette suffix from color_balance."""
        if not color_balance:
            return ""
        cb_lower = color_balance.lower()
        if "warm" in cb_lower and "dominant" in cb_lower:
            return (
                " with warm-dominant color palette (rich reds, "
                "oranges, and yellows in the scene, separate from "
                "lighting temperature)"
            )
        if "cool" in cb_lower and "dominant" in cb_lower:
            return (
                " with cool-dominant color palette (blues, teals, and "
                "purples in the scene, separate from lighting "
                "temperature)"
            )
        return ""

    def _apply_atmosphere_to_lighting(
        self, base_lighting: str, atmosphere: str, market: str, color_suffix: str
    ) -> str:
        """Apply atmosphere temperature adjustments to base lighting."""
        if atmosphere == "Cool":
            return self._format_cool_lighting(base_lighting, market, color_suffix)
        if atmosphere == "Warm":
            return (
                f"{base_lighting}, warm-toned lighting "
                f"(lighting temperature only, NO color tinting){color_suffix}"
            )
        # Neutral
        return f"{base_lighting}, neutral-toned lighting{color_suffix}"

    def _format_cool_lighting(
        self, base_lighting: str, market: str, color_suffix: str
    ) -> str:
        """Format cool atmosphere lighting with market-specific adjustments."""
        if market == "US":
            return (
                f"{base_lighting}, North-facing daylight with "
                f"Navy-slate shadows (Steel & Glass premium tech "
                f"aesthetic, lighting temperature only, NO color "
                f"tinting){color_suffix}"
            )
        return (
            f"{base_lighting}, cool-toned lighting (lighting "
            f"temperature only, NO color tinting){color_suffix}"
        )
    def _map_physical_state_to_description(
        self,
        physical_state: str,
        placement_target: str = "modern sofa or wall",
        placeholder_values: Optional[Dict[str, str]] = None,
        grounding_method: Optional[str] = None,
    ) -> str:
        """
        Map physical_state value to descriptive string for template.

        Args:
            physical_state: Physical state value ("Leaning", "Flat", or "Macro-focused")
            placement_target: Placement target for Leaning state
            placeholder_values: Optional dict containing subject_description
                for dynamic product name
            grounding_method: Optional grounding method from scenario config
                ("base_on_floor", "leaning_against_support", or "held_by_hand")

        Returns:
            Descriptive string for physical state
        """
        # Get product name dynamically (universal)
        subject = "product"
        if placeholder_values:
            subject = placeholder_values.get("subject_description", "product")
            # Extract just the product name if subject_description is long
            if ", " in subject:
                subject = subject.split(", ")[0]

        if physical_state == "Macro-focused":
            # V2_MACRO_DETAIL: No physical state description (close-up focus)
            return ""
        if physical_state == "Flat":
            # V2_FLAT_TECH: Universal flat-lay state
            return (
                f"{subject} lying 180-degree flat against the floor, sliding under "
                "low-clearance furniture, demonstrating flat-lay capability, "
                "firmly grounded with contact shadows"
            )
        # Leaning (golden_ratio or default)
        # V2_WIDE_SCENE: Use grounding_method to determine specific grounding
        if grounding_method == "held_by_hand":
            # Held by hand: Product is being actively held by a person
            return (
                f"{subject} being held by a person's hand with the handle "
                f"making clear physical contact, "
                "showing active use and demonstrating the product's "
                "ergonomic design. "
                "The person's hand must be clearly visible gripping the "
                "handle. "
                "The product is at an angle, not standing upright, and is "
                "supported entirely by the person's hand."
            )
        if grounding_method == "base_on_floor":
            # Direct: Base on floor (for kitchen cleanup, etc.)
            # CRITICAL: Product must NOT stand upright alone - it MUST be leaning or held
            # The phrase "base on floor" means the base touches the floor,
            # but the product MUST lean against something
            return (
                f"{subject} with its base touching the floor surface, "
                "MUST be leaning at an angle against a visible support surface "
                "(kitchen counter, wall, or furniture) "
                "OR MUST be held by a person's hand with the product handle "
                "making clear physical contact. "
                "The product CANNOT and MUST NOT stand upright alone without "
                "support - it will fall over. "
                "The product MUST be at an angle, either leaning against a "
                "support OR being held. "
                "Visible firm contact shadows must prove the product is "
                "grounded and supported."
            )
        # Default: Leaning against support (for artful test track, daily life, etc.)
        if not placement_target or placement_target.strip() == "":
            placement_target = "modern sofa, wall, or kitchen counter"
        # Ensure placement_target is specific, not vague
        if "natural home environment elements" in placement_target:
            placement_target = "modern sofa, wall, or kitchen counter"

        return (
            f"{subject} shown in realistic leaning position against {placement_target}, "
            "with firm contact shadows that prove the product is grounded."
        )

    def _get_step2_modifiers(
        self, placeholder_values: Dict[str, str]
    ) -> Dict[str, str]:
        """
        STEP 2: Aesthetic Enhancement Modifiers.

        Returns pre-authorized aesthetic modifiers for premium, grounded, in-use product:
        - Metallic texture enhancement (with product anchor)
        - Cinematic lighting
        - Render quality
        - Interaction scene (cleaning motion, marble floor reflections)

        Args:
            placeholder_values: Current placeholder values (for product name anchor)

        Returns:
            Dict of Step 2 placeholder values
        """
        # Get product name for anchor (to specify which silver is being referred to)
        product_name = placeholder_values.get("subject_description", "product")
        if ", " in product_name:
            product_name = product_name.split(", ")[0]

        modifiers = {
            "metallic_texture_enhancement": (
                f". The {product_name}'s metallic components feature hyper-realistic "
                "brushed silver grain with anisotropic highlights"
            ),
            "lighting_enhancement": (
                "completely transparent lighting with highlights overflowing, "
                "diffused high-key natural window light with subtle rim light "
                "for component separation, luminous glow spilling over edges"
            ),
            "render_quality": (
                "Octane render, 8K, ray-traced floor reflections, ultra-sharp "
                "CMF textures, completely transparent lighting aesthetic with "
                "highlights overflowing creating luminous glow"
            ),
            "interaction_scene_enhancement": "",
        }
        # Branch-specific interaction scene enhancements
        if self.branch_name == "golden_ratio":
            # Branch 1: Add cleaning motion context
            modifiers["interaction_scene_enhancement"] = (
                ". Subtle cleaning motion context: product positioned as if recently used, "
                "with natural wear patterns and marble floor reflections showing cleaning path"
            )
        elif self.branch_name == "high_efficiency":
            # Branch 2: Macro - Scene-First structure handles this separately
            # No interaction_scene_enhancement needed (handled in _extract_macro_aesthetic_polish)
            modifiers["interaction_scene_enhancement"] = ""
        elif self.branch_name == "cool_peak":
            # Branch 3: Flat-lay - emphasize floor integration
            modifiers["interaction_scene_enhancement"] = (
                ". Premium floor surface with subtle reflections, product seamlessly "
                "integrated into environment"
            )

        logger.info("Step 2 Aesthetic Modifiers: Activated")
        return modifiers

    def _extract_scene_overview(
        self,
        visual_formula: Dict[str, Any],
        placeholder_values: Dict[str, str],
        product_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Extract comprehensive scene overview - opening sentence describing the overall picture.

        This describes what the overall scene looks like:
        - Wide-angle lifestyle scene
        - Close-up detail shot (macro)
        - Flat-lay product photography (flat)

        The overview describes the overall effect when the product is placed in a certain scene.

        Supports custom scenarios through product_context.scene_overviews configuration.

        Args:
            visual_formula: Visual formula JSON dict
            placeholder_values: Current placeholder values (for context)
            product_context: Product context dict (merged with self.product_context)

        Returns:
            Scene overview string (opening sentence)
        """
        # Merge passed product_context with self.product_context (passed takes priority)
        merged_context = {**self.product_context}
        if product_context:
            merged_context.update(product_context)
        # Use configurable scene overview system
        if self.v2_mode and self.branch_name:
            return get_scene_overview(
                self.branch_name, merged_context, placeholder_values
            )
        # Fallback: Generic overview based on product_visibility
        product_visibility = self.feature_mapper.get_feature_value(
            visual_formula, "product_visibility"
        )
        if product_visibility == "partial":
            return (
                "A professional close-up photography scene focusing on "
                "product details and textures, "
                "with shallow depth of field emphasizing precision and craftsmanship."
            )
        return (
            "A wide-angle lifestyle photography scene showing the product integrated into "
            "a modern home environment, capturing the complete product in a wide-angle view."
        )

    def _extract_macro_scene_description(self) -> str:
        """
        SCENE-FIRST NARRATIVE: Layer 1 - Scene Description.

        Global environment and lighting setup for Branch 2 (Macro).
        FEATURE-EMBEDDED: Includes visual_impact and brightness_distribution tags.

        Returns:
            Scene description string with feature tags
        """
        # Note: This is no longer used in template (replaced by scene_overview)
        # Keeping for backward compatibility
        return ""

    def _extract_macro_action_description(
        self, placeholder_values: Dict[str, str]
    ) -> str:
        """
        SCENE-FIRST NARRATIVE: Layer 2 - Action Description.

        How the product interacts with the floor, including the visual bridge
        that makes grounding look natural.
        FEATURE-EMBEDDED: Includes shadow_direction tag.

        Args:
            placeholder_values: Dict containing subject_description for dynamic product name

        Returns:
            Action description string with feature tags
        """
        # Get product name dynamically (universal)
        subject = placeholder_values.get("subject_description", "product")
        # Extract just the product name if subject_description is long
        if ", " in subject:
            subject = subject.split(", ")[0]

        return (
            f"The {subject} is positioned on premium marble flooring, creating realistic, "
            "ray-traced reflections that demonstrate natural grounding and contact shadows."
        )

    def _extract_macro_consistency_anchor(
        self, placeholder_values: Dict[str, str]
    ) -> str:
        """
        SCENE-FIRST NARRATIVE: Layer 3 - Consistency Anchor.

        Hard-coded consistency descriptors (CMF and positioning) to maintain product accuracy.
        FEATURE-EMBEDDED: Includes visual_prominence tag.
        NOTE: Geometric constraint is handled separately as second sentence in template.

        Args:
            placeholder_values: Current placeholder values dict

        Returns:
            Consistency anchor string with feature tags (without redundant geometric constraint)
        """
        subject = placeholder_values.get("subject_description", "product")
        material_finish = placeholder_values.get(
            "material_finish",
            "premium finish",
        )
        # Extract just the product name if subject_description is long
        product_name = subject
        if ", " in subject:
            product_name = subject.split(", ")[0]

        return (
            f"Featuring polished {material_finish}. "
            f"Ensure the {product_name} occupies a prominent position in the frame."
        )

    def _extract_macro_aesthetic_polish(
        self, placeholder_values: Optional[Dict[str, str]] = None
    ) -> str:
        """
        SCENE-FIRST NARRATIVE: Layer 4 - Aesthetic Polish.

        Step 2 aesthetic modifiers with natural lighting and depth.
        MACRO LOGIC: Focus on visual filling (tight framing), not negative space.
        Includes product-anchored metallic texture enhancement.

        Args:
            placeholder_values: Optional dict for product name anchor

        Returns:
            Aesthetic polish string
        """
        # Get product name for anchor (to specify which metallic surfaces)
        product_name = "product"
        if placeholder_values:
            product_name = placeholder_values.get(
                "subject_description", "product"
            )
            if ", " in product_name:
                product_name = product_name.split(", ")[0]

        return (
            f"Hyper-realistic 8k textures with anisotropic highlights on {product_name}'s "
            "metallic surfaces. Tight framing with visual filling, ensuring the product base "
            "and cleaning mechanism occupy the majority of the frame with shallow depth of field "
            "blurring the background."
        )
