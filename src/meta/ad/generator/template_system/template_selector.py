"""
Psychology-Driven Template Selector.

Auto-selects text overlay templates based on psychology_driver from
Ad Miner's master blueprint. Supports 14 psychology types with
complete visual specifications.

Reads from customer config:
- config/{customer}/{platform}/config.yaml (CONSOLIDATED config containing
  psychology_catalog, psychology_templates, and all other settings)

Author: Ad System
Date: 2026-01-30
Version: 3.0 (Fully consolidated customer config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml


logger = logging.getLogger(__name__)


@dataclass
class TemplateSpec:
    """
    Text template specification loaded from YAML.

    Attributes:
        template_id: Unique template identifier
        display_name: Human-readable name
        description: Template description
        psychology_driver: Associated psychology type
        layout: Layout configuration (position, margins, alignment)
        typography: Font specifications
        style: Styling (colors, effects, shapes)
    """
    template_id: str
    display_name: str
    description: str
    psychology_driver: str
    layout: Dict[str, Any]
    typography: Dict[str, Any]
    style: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateSpec":
        """Create TemplateSpec from dictionary (YAML deserialization)."""
        return cls(
            template_id=data["template_id"],
            display_name=data["display_name"],
            description=data.get("description", ""),
            psychology_driver=data.get("psychology_driver", "trust"),
            layout=data.get("layout", {}),
            typography=data.get("typography", {}),
            style=data.get("style", {}),
        )


@dataclass
class PsychologySpec:
    """
    Psychology driver specification from catalog.

    Attributes:
        psychology_id: Psychology type identifier
        full_name: Full human-readable name
        category: Category (authority_trust, urgency_scarcity, etc.)
        colors: Color palette
        typography: Typography specifications
        layout: Layout preferences
        copy_patterns: Example copy patterns
    """
    psychology_id: str
    full_name: str
    category: str
    colors: Dict[str, str]
    typography: Dict[str, Any]
    layout: Dict[str, Any]
    copy_patterns: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PsychologySpec":
        """Create PsychologySpec from dictionary (YAML deserialization)."""
        return cls(
            psychology_id=data["psychology_id"],
            full_name=data["full_name"],
            category=data["category"],
            colors=data.get("colors", {}),
            typography=data.get("typography", {}),
            layout=data.get("layout", {}),
            copy_patterns=data.get("copy_patterns", []),
        )


class TemplateLoader:
    """
    Load and manage text templates from consolidated YAML configuration.

    Usage:
        loader = TemplateLoader()
        templates = loader.load_templates()
        template = loader.get_template("trust_authority")
    """

    def __init__(
        self,
        customer: str = "moprobo",
        platform: str = "meta",
        config_path: Optional[Path] = None,
    ):
        """
        Initialize template loader.

        Args:
            customer: Customer name
            platform: Platform name (e.g., meta, google)
            config_path: Optional direct path to config.yaml (overrides customer/platform)
        """
        # Default: load from customer config
        if config_path is None:
            config_path = Path(f"config/{customer}/{platform}/config.yaml")

        self.config_path = Path(config_path)
        self.customer = customer
        self.platform = platform

        # Cache loaded data
        self._templates_cache: Optional[List[TemplateSpec]] = None
        self._psychology_cache: Optional[Dict[str, PsychologySpec]] = None
        self._config_cache: Optional[Dict[str, Any]] = None

    def _load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load the consolidated config file.

        Args:
            force_reload: Force reload from disk even if cached

        Returns:
            Full config dict
        """
        if self._config_cache is not None and not force_reload:
            return self._config_cache

        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            self._config_cache = {}
            return self._config_cache

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config_cache = yaml.safe_load(f)

        logger.info(f"Loaded config from {self.config_path}")
        return self._config_cache

    def load_templates(self, force_reload: bool = False) -> List[TemplateSpec]:
        """
        Load all templates from consolidated config.yaml.

        Args:
            force_reload: Force reload from disk even if cached

        Returns:
            List of TemplateSpec objects
        """
        if self._templates_cache is not None and not force_reload:
            return self._templates_cache

        config = self._load_config(force_reload)

        templates = []
        for template_data in config.get("psychology_templates", []):
            templates.append(TemplateSpec.from_dict(template_data))

        self._templates_cache = templates
        logger.info(f"Loaded {len(templates)} templates from {self.config_path}")

        return templates

    def load_psychology_catalog(self, force_reload: bool = False) -> Dict[str, PsychologySpec]:
        """
        Load psychology catalog from consolidated config.yaml.

        Args:
            force_reload: Force reload from disk even if cached

        Returns:
            Dict mapping psychology_id to PsychologySpec
        """
        if self._psychology_cache is not None and not force_reload:
            return self._psychology_cache

        config = self._load_config(force_reload)

        catalog = {}
        psych_catalog = config.get("psychology_catalog", {})

        # Handle both nested structure (with "types" key) and flat structure (list)
        if isinstance(psych_catalog, list):
            # Old format: psychology_catalog is directly a list
            psych_types = psych_catalog
        else:
            # New format: psychology_catalog.types contains the list
            psych_types = psych_catalog.get("types", [])

        for psych_data in psych_types:
            psych = PsychologySpec.from_dict(psych_data)
            catalog[psych.psychology_id] = psych

        self._psychology_cache = catalog
        logger.info(f"Loaded {len(catalog)} psychology types from {self.config_path}")

        return catalog

    def get_template(
        self,
        template_id: str,
    ) -> Optional[TemplateSpec]:
        """
        Get template by ID.

        Args:
            template_id: Template identifier

        Returns:
            TemplateSpec if found, None otherwise
        """
        templates = self.load_templates()
        for template in templates:
            if template.template_id == template_id:
                return template
        return None

    def get_templates_by_psychology(
        self,
        psychology_driver: str,
    ) -> List[TemplateSpec]:
        """
        Get all templates for a specific psychology driver.

        Args:
            psychology_driver: Psychology type (e.g., "trust", "fomo")

        Returns:
            List of matching templates
        """
        templates = self.load_templates()
        return [
            t for t in templates
            if t.psychology_driver == psychology_driver.lower()
        ]

    def get_psychology_spec(
        self,
        psychology_id: str,
    ) -> Optional[PsychologySpec]:
        """
        Get psychology specification by ID.

        Args:
            psychology_id: Psychology type identifier

        Returns:
            PsychologySpec if found, None otherwise
        """
        catalog = self.load_psychology_catalog()
        return catalog.get(psychology_id.lower())


class PsychologyTemplateSelector:
    """
    Auto-select templates based on psychology_driver from master blueprint.

    Supports:
    1. Psychology-driven auto-selection (psychology_driven: true)
    2. Manual template_id override
    3. Fallback to default template

    Usage:
        selector = PsychologyTemplateSelector()
        template = selector.select_from_blueprint(master_blueprint)
    """

    # Default fallback template
    DEFAULT_TEMPLATE_ID = "trust_authority"
    DEFAULT_PSYCHOLOGY = "trust"

    def __init__(self, loader: Optional[TemplateLoader] = None):
        """
        Initialize template selector.

        Args:
            loader: TemplateLoader instance (creates new if None)
        """
        self.loader = loader or TemplateLoader()

    def select_from_blueprint(
        self,
        blueprint: Dict[str, Any],
    ) -> TemplateSpec:
        """
        Select template based on master blueprint configuration.

        Selection logic:
        1. If blueprint.text_overlay.template_id is set → use that template
        2. If blueprint.text_overlay.psychology_driven is true → auto-select based on psychology_driver
        3. Otherwise → use default template

        Args:
            blueprint: Master blueprint dict from Ad Miner

        Returns:
            Selected TemplateSpec

        Raises:
            ValueError: If no suitable template found
        """
        text_overlay_config = blueprint.get("text_overlay", {})

        # Option 1: Explicit template_id
        if "template_id" in text_overlay_config:
            template_id = text_overlay_config["template_id"]
            template = self.loader.get_template(template_id)
            if template:
                logger.info(f"Selected template by ID: {template_id}")
                return template
            else:
                logger.warning(f"Template ID not found: {template_id}, falling back to psychology-driven")

        # Option 2: Psychology-driven auto-selection
        if text_overlay_config.get("psychology_driven", False):
            psychology_driver = self._extract_psychology_driver(blueprint)
            template = self._select_by_psychology(psychology_driver)
            logger.info(f"Selected template by psychology: {psychology_driver} → {template.template_id}")
            return template

        # Option 3: Fallback to default
        logger.warning("No template selection criteria found, using default")
        return self._get_default_template()

    def _extract_psychology_driver(self, blueprint: Dict[str, Any]) -> str:
        """
        Extract psychology_driver from blueprint.

        Looks in blueprint.strategy_rationale.psychology_driver first,
        then falls back to default.

        Args:
            blueprint: Master blueprint dict

        Returns:
            Psychology driver string (e.g., "trust", "fomo")
        """
        strategy = blueprint.get("strategy_rationale", {})
        psychology_driver = strategy.get("psychology_driver", self.DEFAULT_PSYCHOLOGY)

        logger.debug(f"Extracted psychology_driver: {psychology_driver}")
        return psychology_driver.lower()

    def _select_by_psychology(self, psychology_driver: str) -> TemplateSpec:
        """
        Select best template for a psychology driver.

        Strategy:
        1. Look for template with matching template_id (e.g., "trust_authority")
        2. Look for templates with matching psychology_driver field
        3. Fall back to default template

        Args:
            psychology_driver: Psychology type (e.g., "trust", "fomo")

        Returns:
            Selected TemplateSpec

        Raises:
            ValueError: If no template found
        """
        # Try exact template_id match first (e.g., "trust_authority")
        template_id = f"{psychology_driver}_authority" if psychology_driver == "trust" else f"{psychology_driver}"
        template = self.loader.get_template(template_id)

        if template:
            return template

        # Try psychology_driver field match
        templates = self.loader.get_templates_by_psychology(psychology_driver)
        if templates:
            # Return first match (could be enhanced to score/rank templates)
            return templates[0]

        # Fall back to default
        logger.warning(f"No template found for psychology: {psychology_driver}, using default")
        return self._get_default_template()

    def _get_default_template(self) -> TemplateSpec:
        """
        Get default fallback template.

        Returns:
            Default TemplateSpec

        Raises:
            ValueError: If default template not found
        """
        template = self.loader.get_template(self.DEFAULT_TEMPLATE_ID)
        if template is None:
            # Try to load any template
            templates = self.loader.load_templates()
            if templates:
                template = templates[0]
            else:
                raise ValueError(
                    f"No templates available. Default template '{self.DEFAULT_TEMPLATE_ID}' not found."
                )

        return template

    def get_psychology_spec(self, psychology_id: str) -> Optional[PsychologySpec]:
        """
        Get psychology specification for a psychology type.

        Args:
            psychology_id: Psychology type identifier

        Returns:
            PsychologySpec if found, None otherwise
        """
        return self.loader.get_psychology_spec(psychology_id)


# Convenience functions
def select_template_from_blueprint(
    blueprint: Dict[str, Any],
) -> TemplateSpec:
    """
    Convenience function for quick template selection from blueprint.

    Args:
        blueprint: Master blueprint dict

    Returns:
        Selected TemplateSpec

    Example:
        template = select_template_from_blueprint(master_blueprint)
        print(f"Selected: {template.display_name}")
    """
    selector = PsychologyTemplateSelector()
    return selector.select_from_blueprint(blueprint)


def load_template_by_id(template_id: str) -> Optional[TemplateSpec]:
    """
    Convenience function to load template by ID.

    Args:
        template_id: Template identifier

    Returns:
        TemplateSpec if found, None otherwise
    """
    loader = TemplateLoader()
    return loader.get_template(template_id)
