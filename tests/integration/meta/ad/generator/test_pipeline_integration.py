"""
Integration tests for Template-Driven Ad Generator Pipeline.

Tests end-to-end functionality across multiple modules.
"""

import pytest
from pathlib import Path
import yaml
from PIL import Image
import numpy as np

from src.meta.ad.generator.template_system.pipeline import (
    TemplatePipeline,
    PipelineConfig,
    generate_ads,
)
from src.meta.ad.generator.template_system.product_preprocessor import (
    preprocess_product,
    PerspectiveType,
)
from src.meta.ad.generator.template_system.template_selector import (
    select_template_from_blueprint,
)


@pytest.fixture
def test_customer_config(tmp_path):
    """Create test customer configuration."""
    config_dir = tmp_path / "config" / "ad" / "test_customer"
    config_dir.mkdir(parents=True)

    # Create master blueprint
    blueprint = {
        "metadata": {
            "customer": "test_customer",
            "product": "Test Product",
            "campaign_goal": "conversion"
        },
        "strategy_rationale": {
            "locked_combination": {
                "primary_material": "Marble",
                "paired_lighting": "Window Light",
                "confidence_score": 0.92
            },
            "psychology_driver": "trust",
            "psychology_confidence": 0.85
        },
        "nano_generation_rules": {
            "prompt_template_structure": "Product on table, {atmosphere}",
            "prompt_slots": {
                "atmosphere": "luxury home environment"
            },
            "negative_prompt": "cartoon, illustration",
            "inference_config": {
                "model": "nanobanana_pro",
                "steps": 8,
                "cfg_scale": 3.5,
                "batch_size": 1
            }
        },
        "compositing": {
            "shadow_direction": "left",
            "light_wrap_intensity": 0.3,
            "light_match_mode": "soft_light",
            "light_match_opacity": 0.25
        },
        "text_overlay": {
            "template_id": "trust_authority",
            "psychology_driven": True,
            "smart_color_enabled": True,
            "collision_detection_enabled": True
        }
    }

    blueprint_path = config_dir / "master_blueprint.yaml"
    with open(blueprint_path, "w") as f:
        yaml.dump(blueprint, f)

    # Create campaign content
    campaign = {
        "campaign_content": {
            "headline": "Test Headline",
            "sub_text": "Test Subtext",
            "cta_text": "Shop Now",
            "brand_color": "#FF5733"
        }
    }

    campaign_path = config_dir / "campaign_content.yaml"
    with open(campaign_path, "w") as f:
        yaml.dump(campaign, f)

    # Create template catalog
    generator_dir = tmp_path / "config" / "ad" / "generator"
    generator_dir.mkdir(parents=True)

    templates_data = {
        "templates": [
            {
                "template_id": "trust_authority",
                "display_name": "Trust: Authority",
                "psychology_driver": "trust",
                "layout": {
                    "position": "Bottom_Center",
                    "margin_y": 80,
                    "alignment": "center"
                },
                "typography": {
                    "headline": {
                        "font_family": "Sans_Bold",
                        "font_size": 48
                    },
                    "sub_text": {
                        "font_family": "Sans_Regular",
                        "font_size": 32
                    },
                    "cta": {
                        "font_family": "Sans_Medium",
                        "font_size": 24,
                        "padding_x": 40,
                        "padding_y": 16
                    }
                },
                "style": {
                    "font_color_logic": "Auto_Contrast",
                    "cta_shape": "Pill_Solid",
                    "cta_bg_color": "Transparent"
                }
            }
        ]
    }

    templates_path = generator_dir / "text_templates.yaml"
    with open(templates_path, "w") as f:
        yaml.dump(templates_data, f)

    # Create psychology catalog
    psychology_data = {
        "psychology_catalog": [
            {
                "psychology_id": "trust",
                "full_name": "Trust & Authority",
                "category": "authority_trust",
                "description": "Establish credibility",
                "colors": {"primary": "#003366"},
                "typography": {"headline_font": "Serif_Bold"},
                "layout": {"position": "centered"},
                "copy_patterns": ["Expert recommended"]
            }
        ]
    }

    psychology_path = generator_dir / "psychology_catalog.yaml"
    with open(psychology_path, "w") as f:
        yaml.dump(psychology_data, f)

    return {
        "config_dir": config_dir,
        "generator_dir": generator_dir,
        "blueprint_path": blueprint_path,
        "campaign_path": campaign_path
    }


@pytest.fixture
def sample_product_image(tmp_path):
    """Create sample product image."""
    # Create RGBA image with product in center
    image = Image.new("RGBA", (400, 400), (0, 0, 0, 0))

    # Add product (opaque red square with some transparency)
    product_layer = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
    image.paste(product_layer, (100, 100), product_layer)

    image_path = tmp_path / "product.png"
    image.save(image_path)

    return image_path


class TestProductPreprocessingIntegration:
    """Integration tests for product preprocessing."""

    def test_preprocess_realistic_product(self, sample_product_image):
        """Test preprocessing realistic product image."""
        result = preprocess_product(sample_product_image)

        assert result.trimmed_image is not None
        assert result.mask is not None
        assert result.perspective in [PerspectiveType.HIGH_ANGLE, PerspectiveType.EYE_LEVEL]
        assert result.original_size == (400, 400)

        # Trimmed should be smaller
        assert result.trimmed_image.width < 400
        assert result.trimmed_image.height < 400

    def test_preprocess_result_serialization(self, sample_product_image):
        """Test that preprocess result can be serialized."""
        result = preprocess_product(sample_product_image)

        # Should serialize to dict
        result_dict = result.to_dict()
        assert "perspective" in result_dict
        assert "bbox" in result_dict
        assert "original_size" in result_dict


class TestTemplateSelectionIntegration:
    """Integration tests for template selection."""

    def test_select_from_blueprint_file(self, test_customer_config):
        """Test template selection from blueprint file."""
        with open(test_customer_config["blueprint_path"]) as f:
            blueprint = yaml.safe_load(f)

        template = select_template_from_blueprint(blueprint)

        assert template is not None
        assert template.template_id == "trust_authority"
        assert template.psychology_driver == "trust"

    def test_template_has_required_fields(self, test_customer_config):
        """Test that selected template has all required fields."""
        with open(test_customer_config["blueprint_path"]) as f:
            blueprint = yaml.safe_load(f)

        template = select_template_from_blueprint(blueprint)

        # Check required fields
        assert hasattr(template, "template_id")
        assert hasattr(template, "layout")
        assert hasattr(template, "typography")
        assert hasattr(template, "style")

        # Check layout fields
        assert "position" in template.layout
        assert "margin_y" in template.layout

        # Check typography fields
        assert "headline" in template.typography
        assert "font_family" in template.typography["headline"]
        assert "font_size" in template.typography["headline"]


class TestPipelineIntegration:
    """Integration tests for complete pipeline."""

    def test_pipeline_initialization(self, test_customer_config):
        """Test pipeline can be initialized."""
        # This test would require mocking paths or using actual config
        # For now, just test PipelineConfig creation
        config = PipelineConfig(
            customer="test_customer",
            platform="facebook",
            product="Test Product",
            num_variants=1,
            generate_backgrounds=False  # Skip background generation
        )

        assert config.customer == "test_customer"
        assert config.platform == "facebook"
        assert config.num_variants == 1

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig(
            customer="test",
            platform="facebook",
            product="test"
        )

        assert config.num_variants == 1
        assert config.generate_backgrounds is True
        assert config.save_intermediates is True


@pytest.mark.integration
class TestEndToEndGeneration:
    """End-to-end integration tests (marked as slow)."""

    def test_full_generation_flow_description(self):
        """
        Test complete generation flow.

        NOTE: This test requires:
        1. Actual config files in place
        2. Product image
        3. Optional: Background generation API

        This is a placeholder for the full integration test.
        The actual implementation would test:
        - Loading master blueprint
        - Preprocessing product
        - Selecting template
        - Generating/compositing backgrounds
        - Rendering text overlay
        - Saving outputs

        Example:
            results = generate_ads(
                customer="moprobo",
                platform="facebook",
                product="Power Station",
                num_variants=1
            )

            assert len(results.generated_images) == 1
            image, metadata = results.generated_images[0]
            assert image is not None
            assert metadata["psychology_driver"] == "trust"
        """
        pass  # Placeholder

    def test_pipeline_with_mock_backgrounds(self, test_customer_config, sample_product_image):
        """Test pipeline with mocked backgrounds (no API required)."""
        # This would test the pipeline with pre-generated backgrounds
        # Skipped for now as it requires full implementation
        pytest.skip("Requires full pipeline implementation")


class TestPathResolution:
    """Test path resolution across modules."""

    def test_config_paths_consistency(self, test_customer_config):
        """Test that config paths are consistent across modules."""
        blueprint_path = test_customer_config["blueprint_path"]
        campaign_path = test_customer_config["campaign_path"]

        # Verify files exist
        assert blueprint_path.exists()
        assert campaign_path.exists()

        # Verify we can load them
        with open(blueprint_path) as f:
            blueprint = yaml.safe_load(f)

        with open(campaign_path) as f:
            campaign = yaml.safe_load(f)

        assert blueprint["metadata"]["customer"] == "test_customer"
        assert "campaign_content" in campaign
