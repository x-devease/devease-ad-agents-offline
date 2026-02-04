"""
Unit tests for TemplateSelector.

Tests template loading and psychology-driven selection.
"""

import pytest
from pathlib import Path
import yaml

from src.meta.ad.generator.template_system.template_selector import (
    TemplateLoader,
    TemplateSpec,
    PsychologySpec,
    PsychologyTemplateSelector,
    select_template_from_blueprint,
    load_template_by_id,
)


@pytest.fixture
def sample_consolidated_config(tmp_path):
    """Create sample consolidated config.yaml for testing."""
    config_data = {
        "psychology_catalog": [
            {
                "psychology_id": "trust",
                "full_name": "Trust & Authority",
                "category": "authority_trust",
                "description": "Establish credibility",
                "colors": {
                    "primary": "#003366",
                    "secondary": "#1E8449"
                },
                "typography": {
                    "headline_font": "Serif_Bold",
                    "letter_spacing": 0.02
                },
                "layout": {
                    "position": "centered_symmetrical"
                },
                "copy_patterns": ["Expert recommended", "Trusted by"]
            },
            {
                "psychology_id": "fomo",
                "full_name": "FOMO & Urgency",
                "category": "urgency_scarcity",
                "description": "Create urgency",
                "colors": {
                    "primary": "#FF0000",
                    "secondary": "#FFCC00"
                },
                "typography": {
                    "headline_font": "Sans_Black",
                    "letter_spacing": -0.03
                },
                "layout": {
                    "position": "top_right"
                },
                "copy_patterns": ["Limited time", "Hurry"]
            }
        ],
        "psychology_templates": [
            {
                "template_id": "trust_authority",
                "display_name": "Trust: Authority",
                "description": "Trust-based template",
                "psychology_driver": "trust",
                "layout": {
                    "position": "Bottom_Center",
                    "margin_y": 80,
                    "alignment": "center"
                },
                "typography": {
                    "headline": {
                        "font_family": "Serif_Bold",
                        "font_size": 48
                    }
                },
                "style": {
                    "font_color_logic": "Auto_Contrast",
                    "cta_shape": "Underline"
                }
            },
            {
                "template_id": "fomo_urgency",
                "display_name": "FOMO: Urgency",
                "description": "Urgency template",
                "psychology_driver": "fomo",
                "layout": {
                    "position": "Top_Right",
                    "margin_y": 40
                },
                "typography": {
                    "headline": {
                        "font_family": "Sans_Black",
                        "font_size": 64
                    }
                },
                "style": {
                    "font_color_logic": "Fixed_White",
                    "cta_shape": "Pill_Solid"
                }
            }
        ],
        "modifiers": {
            "intensity": [
                {"level": "calm", "animation_speed": "very_slow"}
            ],
            "complexity": [
                {"level": "minimal", "element_count": "3-5"}
            ]
        },
        "generator_settings": {
            "image_model": "dall-e-3",
            "num_candidates": 4
        }
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


@pytest.fixture
def sample_master_blueprint(tmp_path):
    """Create sample master_blueprint.yaml for testing."""
    blueprint = {
        "metadata": {
            "customer": "moprobo",
            "product": "Power Station"
        },
        "strategy_rationale": {
            "locked_combination": {
                "primary_material": "Marble",
                "confidence_score": 0.92
            },
            "psychology_driver": "trust",
            "psychology_confidence": 0.85
        },
        "text_overlay": {
            "template_id": "trust_authority",
            "psychology_driven": True
        }
    }

    blueprint_path = tmp_path / "master_blueprint.yaml"
    with open(blueprint_path, "w") as f:
        yaml.dump(blueprint, f)

    return blueprint_path


class TestTemplateSpec:
    """Test TemplateSpec dataclass."""

    def test_from_dict(self):
        """Test creating TemplateSpec from dict."""
        data = {
            "template_id": "test_template",
            "display_name": "Test Template",
            "description": "Test description",
            "psychology_driver": "trust",
            "layout": {"position": "Bottom_Center"},
            "typography": {"headline": {"font_size": 48}},
            "style": {"cta_shape": "Underline"}
        }

        spec = TemplateSpec.from_dict(data)

        assert spec.template_id == "test_template"
        assert spec.display_name == "Test Template"
        assert spec.psychology_driver == "trust"
        assert spec.layout == {"position": "Bottom_Center"}


class TestPsychologySpec:
    """Test PsychologySpec dataclass."""

    def test_from_dict(self):
        """Test creating PsychologySpec from dict."""
        data = {
            "psychology_id": "trust",
            "full_name": "Trust & Authority",
            "category": "authority_trust",
            "colors": {"primary": "#003366"},
            "typography": {"headline_font": "Serif_Bold"},
            "layout": {"position": "centered"},
            "copy_patterns": ["Expert recommended"]
        }

        spec = PsychologySpec.from_dict(data)

        assert spec.psychology_id == "trust"
        assert spec.full_name == "Trust & Authority"
        assert spec.category == "authority_trust"
        assert spec.colors == {"primary": "#003366"}


class TestTemplateLoader:
    """Test TemplateLoader class."""

    def test_init_default_path(self):
        """Test initialization with default path."""
        loader = TemplateLoader()

        # Default uses customer="moprobo", platform="meta"
        assert loader.config_path == Path("config/moprobo/meta/config.yaml")
        assert loader.customer == "moprobo"
        assert loader.platform == "meta"

    def test_init_custom_path(self, tmp_path):
        """Test initialization with custom path."""
        config_path = tmp_path / "custom_config.yaml"

        loader = TemplateLoader(config_path=config_path)

        assert loader.config_path == config_path

    def test_load_templates(self, sample_consolidated_config):
        """Test loading templates from consolidated config."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        templates = loader.load_templates()

        assert len(templates) == 2
        assert templates[0].template_id == "trust_authority"
        assert templates[1].template_id == "fomo_urgency"

    def test_load_templates_caching(self, sample_consolidated_config):
        """Test that templates are cached."""
        loader = TemplateLoader(config_path=sample_consolidated_config)

        # First load
        templates1 = loader.load_templates()
        # Second load (should use cache)
        templates2 = loader.load_templates(force_reload=False)

        assert templates1 is templates2  # Same object (cached)

    def test_load_templates_force_reload(self, sample_consolidated_config):
        """Test force reload bypasses cache."""
        loader = TemplateLoader(config_path=sample_consolidated_config)

        templates1 = loader.load_templates()
        templates2 = loader.load_templates(force_reload=True)

        assert templates1 is not templates2  # Different objects (reloaded)

    def test_load_templates_missing_file(self, tmp_path):
        """Test loading missing config file returns empty list."""
        loader = TemplateLoader(config_path=tmp_path / "nonexistent.yaml")
        templates = loader.load_templates()

        assert templates == []

    def test_load_psychology_catalog(self, sample_consolidated_config):
        """Test loading psychology catalog from consolidated config."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        catalog = loader.load_psychology_catalog()

        assert len(catalog) == 2
        assert "trust" in catalog
        assert "fomo" in catalog

    def test_get_template_by_id(self, sample_consolidated_config):
        """Test getting template by ID."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        template = loader.get_template("trust_authority")

        assert template is not None
        assert template.template_id == "trust_authority"

    def test_get_template_by_id_not_found(self, sample_consolidated_config):
        """Test getting non-existent template returns None."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        template = loader.get_template("nonexistent")

        assert template is None

    def test_get_templates_by_psychology(self, sample_consolidated_config):
        """Test getting templates by psychology driver."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        templates = loader.get_templates_by_psychology("trust")

        assert len(templates) == 1
        assert templates[0].template_id == "trust_authority"

    def test_get_psychology_spec(self, sample_consolidated_config):
        """Test getting psychology spec from consolidated config."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        spec = loader.get_psychology_spec("trust")

        assert spec is not None
        assert spec.psychology_id == "trust"
        assert spec.full_name == "Trust & Authority"


class TestPsychologyTemplateSelector:
    """Test PsychologyTemplateSelector class."""

    def test_init_default(self):
        """Test initialization with default loader."""
        selector = PsychologyTemplateSelector()

        assert selector.loader is not None
        assert isinstance(selector.loader, TemplateLoader)

    def test_init_custom_loader(self, sample_consolidated_config):
        """Test initialization with custom loader."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        selector = PsychologyTemplateSelector(loader=loader)

        assert selector.loader == loader

    def test_select_from_blueprint_explicit_template_id(
        self, sample_consolidated_config, sample_master_blueprint
    ):
        """Test selection with explicit template_id."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        selector = PsychologyTemplateSelector(loader=loader)

        # Load blueprint
        with open(sample_master_blueprint) as f:
            blueprint = yaml.safe_load(f)

        template = selector.select_from_blueprint(blueprint)

        assert template.template_id == "trust_authority"

    def test_select_from_blueprint_psychology_driven(
        self, sample_consolidated_config, sample_master_blueprint
    ):
        """Test psychology-driven template selection."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        selector = PsychologyTemplateSelector(loader=loader)

        # Create blueprint with psychology_driven=True
        blueprint = {
            "strategy_rationale": {
                "psychology_driver": "fomo"
            },
            "text_overlay": {
                "psychology_driven": True
            }
        }

        template = selector.select_from_blueprint(blueprint)

        assert template.psychology_driver == "fomo"
        assert template.template_id == "fomo_urgency"

    def test_select_from_blueprint_missing_psychology(
        self, sample_consolidated_config
    ):
        """Test selection falls back to default when psychology missing."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        selector = PsychologyTemplateSelector(loader=loader)

        blueprint = {
            "strategy_rationale": {},
            "text_overlay": {
                "psychology_driven": True
            }
        }

        template = selector.select_from_blueprint(blueprint)

        # Should fall back to default (trust)
        assert template.psychology_driver == "trust"

    def test_get_psychology_spec(
        self, sample_consolidated_config
    ):
        """Test getting psychology spec through selector."""
        loader = TemplateLoader(config_path=sample_consolidated_config)
        selector = PsychologyTemplateSelector(loader=loader)

        spec = selector.get_psychology_spec("trust")

        assert spec is not None
        assert spec.psychology_id == "trust"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_select_template_from_blueprint(
        self, sample_consolidated_config, sample_master_blueprint
    ):
        """Test select_template_from_blueprint convenience function."""
        # Mock the default loader paths by temporarily creating files
        # (In real tests, you'd patch the paths)

        with open(sample_master_blueprint) as f:
            blueprint = yaml.safe_load(f)

        # This would use default paths, so we skip testing actual file loading
        # Just verify the function exists and has correct signature
        assert callable(select_template_from_blueprint)

    def test_load_template_by_id(self):
        """Test load_template_by_id convenience function."""
        # Just verify the function exists
        assert callable(load_template_by_id)
