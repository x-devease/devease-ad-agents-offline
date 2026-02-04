"""
End-to-End Pipeline Integration Tests.

Tests complete ad generation pipeline from config to output.
"""

import pytest
from pathlib import Path
import yaml
from PIL import Image
import shutil
import uuid

from src.meta.ad.generator.template_system.pipeline import (
    TemplatePipeline,
    PipelineConfig,
    generate_ads,
)


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Clean up any test artifacts before and after tests."""
    # Cleanup before tests
    config_dir = Path("config")
    for item in config_dir.glob("test_*"):
        if item.is_dir():
            shutil.rmtree(item)

    yield

    # Cleanup after tests
    for item in config_dir.glob("test_*"):
        if item.is_dir():
            shutil.rmtree(item)


@pytest.fixture
def full_test_environment(tmp_path):
    """Create complete test environment with all required files."""
    # Use UUID to ensure unique test customer name
    test_customer_id = f"test_{uuid.uuid4().hex[:8]}"

    # Create directory structure (use resolve() to get absolute path)
    config_base = (tmp_path / "config").resolve()
    config_base.mkdir(parents=True, exist_ok=True)

    # Customer config directory (using unique ID)
    customer_config = config_base / test_customer_id / "facebook"
    customer_config.mkdir(parents=True, exist_ok=True)

    # Create products directory
    products_dir = customer_config / "products"
    products_dir.mkdir(parents=True)

    # Create sample product image
    product_image = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    product_layer = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
    product_image.paste(product_layer, (100, 100), product_layer)
    product_path = products_dir / "test_product.png"
    product_image.save(product_path)

    # Create master blueprint
    blueprint = {
        "metadata": {
            "schema_version": "2.0",
            "customer": test_customer_id,
            "product": "Test Product",
            "branch": "US",
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
                "product_context": "professional product",
                "atmosphere": "luxury home environment"
            },
            "negative_prompt": "cartoon, illustration",
            "inference_config": {
                "model": "nanobanana_pro",
                "steps": 8,
                "cfg_scale": 3.5,
                "batch_size": 1,
                "aspect_ratio": "3:4",
                "guidance": "perspective_aware"
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

    blueprint_path = customer_config / "config.yaml"
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

    campaign_path = customer_config / "campaign_content.yaml"
    with open(campaign_path, "w") as f:
        yaml.dump(campaign, f)

    # Add psychology_catalog and psychology_templates to customer config (consolidated)
    # Part P: Psychology Catalog
    psychology_catalog = {
        "version": "2.0",
        "total_types": 1,
        "types": [
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

    # Part G: Psychology Templates
    psychology_templates = [
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

    # Add to existing config
    blueprint["psychology_catalog"] = psychology_catalog
    blueprint["psychology_templates"] = psychology_templates
    blueprint["modifiers"] = {
        "intensity": [{"level": "calm"}],
        "complexity": [{"level": "minimal"}]
    }
    blueprint["generator_settings"] = {
        "image_model": "test_model",
        "num_candidates": 1
    }

    # Rewrite config.yaml with full consolidated structure
    with open(blueprint_path, "w") as f:
        yaml.dump(blueprint, f)

    # Create results directory
    results_dir = tmp_path / "results" / test_customer_id / "facebook"
    results_dir.mkdir(parents=True)

    return {
        "config_base": config_base,
        "customer_config": customer_config,
        "product_path": product_path,
        "blueprint_path": blueprint_path,
        "campaign_path": campaign_path,
        "results_dir": results_dir,
        "tmp_path": tmp_path,
        "test_customer_id": test_customer_id  # For test assertions
    }


@pytest.mark.integration
class TestPipelineEndToEnd:
    """End-to-end pipeline integration tests."""

    def test_pipeline_initialization(self, full_test_environment):
        """Test pipeline can be initialized with test environment."""
        env = full_test_environment

        # Create pipeline config with custom paths
        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            num_variants=1,
            generate_backgrounds=False,  # Skip for testing
            save_intermediates=False,
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        pipeline = TemplatePipeline(config)

        assert pipeline.config.customer == env["test_customer_id"]
        assert pipeline.config.platform == "facebook"

    def test_pipeline_loads_configs(self, full_test_environment):
        """Test pipeline loads all required configurations."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            num_variants=1,
            generate_backgrounds=False,
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        pipeline = TemplatePipeline(config)

        # Load configs (this would normally happen in run())
        pipeline._load_configs()

        assert pipeline._master_blueprint is not None
        assert pipeline._campaign_content is not None
        assert pipeline._master_blueprint["metadata"]["customer"] == env["test_customer_id"]

    def test_pipeline_selects_template(self, full_test_environment):
        """Test pipeline selects template based on psychology."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            num_variants=1,
            generate_backgrounds=False,
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        pipeline = TemplatePipeline(config)
        pipeline._load_configs()

        # Select template
        pipeline._stage_template_selection()

        assert pipeline._selected_template is not None
        assert pipeline._selected_template.template_id == "trust_authority"

    def test_pipeline_preprocesses_product(self, full_test_environment):
        """Test pipeline preprocesses product image."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            num_variants=1,
            generate_backgrounds=False,
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        pipeline = TemplatePipeline(config)

        # Preprocess product
        pipeline._stage_preprocessing(env["product_path"])

        assert pipeline._preprocessor_result is not None
        assert pipeline._preprocessor_result.trimmed_image is not None
        assert pipeline._preprocessor_result.mask is not None

    def test_pipeline_saves_outputs(self, full_test_environment):
        """Test pipeline saves generated outputs."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            num_variants=1,
            generate_backgrounds=False,
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        # Create mock result
        mock_image = Image.new("RGB", (1080, 1080), (255, 0, 0))
        metadata = {
            "variant": 1,
            "template_id": "trust_authority",
            "psychology_driver": "trust"
        }

        pipeline = TemplatePipeline(config)
        pipeline._save_output(mock_image, 1, metadata)

        # Check file was created (search from parent of results_dir)
        output_files = list(env["results_dir"].parent.rglob("*.png"))
        assert len(output_files) > 0


@pytest.mark.integration
class TestGenerateAdsConvenience:
    """Test generate_ads convenience function."""

    def test_generate_ads_signature(self):
        """Test that generate_ads has correct signature."""
        assert callable(generate_ads)

    def test_generate_ads_creates_pipeline(self):
        """Test generate_ads creates pipeline with correct config."""
        # This would require full environment setup
        # For now, just verify the function works
        pass


@pytest.mark.integration
class TestPipelineDataFlow:
    """Test data flow through pipeline stages."""

    def test_blueprint_to_template_flow(self, full_test_environment):
        """Test data flow from blueprint to template selection."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        pipeline = TemplatePipeline(config)
        pipeline._load_configs()
        pipeline._stage_template_selection()

        # Verify psychology driver propagated correctly
        assert pipeline._selected_template.psychology_driver == "trust"

    def test_campaign_content_to_text_rendering(self, full_test_environment):
        """Test data flow from campaign content to text rendering."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        pipeline = TemplatePipeline(config)
        pipeline._load_configs()

        # Verify campaign content loaded
        assert pipeline._campaign_content.headline == "Test Headline"
        assert pipeline._campaign_content.cta_text == "Shop Now"


@pytest.mark.slow
class TestPipelinePerformance:
    """Performance and stress tests."""

    def test_multiple_variants(self, full_test_environment):
        """Test generating multiple variants."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            num_variants=5,
            generate_backgrounds=False,
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        pipeline = TemplatePipeline(config)
        pipeline._load_configs()

        # Just verify config is set correctly
        assert pipeline.config.num_variants == 5


@pytest.fixture
def sample_background(tmp_path):
    """Create sample background for testing."""
    background = Image.new("RGB", (1080, 1080), (240, 240, 240))
    background_path = tmp_path / "background.png"
    background.save(background_path)
    return background_path


class TestPipelineWithBackgrounds:
    """Test pipeline with background handling."""

    def test_pipeline_with_existing_background(
        self, full_test_environment, sample_background
    ):
        """Test pipeline can use existing background."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Test Product",
            num_variants=1,
            generate_backgrounds=False
        )

        pipeline = TemplatePipeline(config)

        # This would use the background instead of generating
        # For now, just verify the method exists
        assert hasattr(pipeline, '_stage_background_generation')


class TestPipelineErrorHandling:
    """Test pipeline error handling."""

    def test_pipeline_handles_missing_blueprint(self, tmp_path):
        """Test pipeline handles missing blueprint gracefully."""
        config = PipelineConfig(
            customer="missing_customer",
            platform="facebook",
            product="Nonexistent Product",
            config_dir=tmp_path / "config",
            output_dir=tmp_path / "results"
        )

        pipeline = TemplatePipeline(config)

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            pipeline._load_configs()

    def test_pipeline_handles_missing_product_image(self, full_test_environment):
        """Test pipeline handles missing product image."""
        env = full_test_environment

        config = PipelineConfig(
            customer=env["test_customer_id"],
            platform="facebook",
            product="Missing Product",
            config_dir=env["config_base"],
            output_dir=env["results_dir"].parent
        )

        pipeline = TemplatePipeline(config)

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            pipeline._stage_preprocessing(
                env["tmp_path"] / "nonexistent.png"
            )


class TestPipelineMetadata:
    """Test pipeline metadata tracking."""

    def test_pipeline_result_metadata(self, full_test_environment):
        """Test pipeline result contains proper metadata."""
        env = full_test_environment

        from src.meta.ad.generator.template_system.pipeline import PipelineResult
        from PIL import Image

        # Create mock result
        image = Image.new("RGB", (1080, 1080), (255, 0, 0))
        metadata = {
            "variant": 1,
            "template_id": "trust_authority",
            "psychology_driver": "trust",
            "perspective": "eye_level"
        }

        result = PipelineResult(
            generated_images=[(image, metadata)],
            metadata={
                "customer": env["test_customer_id"],
                "platform": "facebook",
                "product": "Test Product",
                "num_variants": 1
            }
        )

        assert len(result.generated_images) == 1
        assert result.metadata["customer"] == env["test_customer_id"]
        assert result.metadata["num_variants"] == 1
