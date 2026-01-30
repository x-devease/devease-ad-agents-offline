"""
Unit tests for Background Generator.

Tests perspective-aware background generation with NanoBanana Pro.
"""

import pytest
from pathlib import Path
import yaml
from PIL import Image

from src.meta.ad.generator.template_system.background_generator import (
    GenerationConfig,
    BackgroundPrompt,
    GeneratedBackground,
    NanoBackgroundGenerator,
    generate_backgrounds_from_blueprint,
    AspectRatio,
)


@pytest.fixture
def sample_master_blueprint(tmp_path):
    """Create sample master blueprint for testing."""
    blueprint = {
        "metadata": {
            "customer": "moprobo",
            "product": "Power Station"
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
                "batch_size": 20,
                "aspect_ratio": "3:4",
                "guidance": "perspective_aware"
            }
        }
    }

    blueprint_path = tmp_path / "blueprint.yaml"
    with open(blueprint_path, "w") as f:
        yaml.dump(blueprint, f)

    return blueprint_path


class TestAspectRatio:
    """Test AspectRatio enum."""

    def test_aspect_ratio_values(self):
        """Test aspect ratio enum values."""
        assert AspectRatio.SQUARE.value == "1:1"
        assert AspectRatio.PORTRAIT.value == "3:4"
        assert AspectRatio.LANDSCAPE.value == "4:3"
        assert AspectRatio.STORY.value == "9:16"


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_from_dict(self):
        """Test creating GenerationConfig from blueprint."""
        blueprint_data = {
            "nano_generation_rules": {
                "inference_config": {
                    "model": "nanobanana_pro",
                    "steps": 8,
                    "cfg_scale": 3.5,
                    "batch_size": 20,
                    "aspect_ratio": "3:4",
                    "guidance": "perspective_aware"
                }
            }
        }

        config = GenerationConfig.from_dict(blueprint_data)

        assert config.model == "nanobanana_pro"
        assert config.steps == 8
        assert config.cfg_scale == 3.5
        assert config.batch_size == 20
        assert config.aspect_ratio == "3:4"
        assert config.guidance == "perspective_aware"

    def test_from_dict_with_defaults(self):
        """Test GenerationConfig with missing values uses defaults."""
        blueprint_data = {
            "nano_generation_rules": {
                "inference_config": {}
            }
        }

        config = GenerationConfig.from_dict(blueprint_data)

        assert config.model == "nanobanana_pro"
        assert config.steps == 8
        assert config.cfg_scale == 3.5
        assert config.batch_size == 20

    def test_from_dict_custom_values(self):
        """Test GenerationConfig with custom values."""
        blueprint_data = {
            "nano_generation_rules": {
                "inference_config": {
                    "model": "custom_model",
                    "steps": 20,
                    "cfg_scale": 5.0,
                    "batch_size": 10
                }
            }
        }

        config = GenerationConfig.from_dict(blueprint_data)

        assert config.model == "custom_model"
        assert config.steps == 20
        assert config.cfg_scale == 5.0
        assert config.batch_size == 10


class TestBackgroundPrompt:
    """Test BackgroundPrompt class."""

    def test_build_final_prompt_high_angle(self):
        """Test prompt building with HIGH_ANGLE perspective."""
        prompt = BackgroundPrompt(
            base_prompt="Product on table",
            perspective="high_angle",
            negative_prompt="cartoon"
        )

        final = prompt.build_final_prompt()

        assert "Product on table" in final
        assert "top-down view" in final
        assert "looking down" in final

    def test_build_final_prompt_eye_level(self):
        """Test prompt building with EYE_LEVEL perspective."""
        from src.meta.ad.generator.template_system.product_preprocessor import PerspectiveType

        prompt = BackgroundPrompt(
            base_prompt="Product on table",
            perspective=PerspectiveType.EYE_LEVEL,
            negative_prompt="cartoon"
        )

        final = prompt.build_final_prompt()

        assert "Product on table" in final
        assert "eye-level view" in final
        assert "horizontal perspective" in final

    def test_build_final_prompt_preserves_base(self):
        """Test that base prompt is preserved in final prompt."""
        prompt = BackgroundPrompt(
            base_prompt="Luxury product on marble table",
            perspective="eye_level"
        )

        final = prompt.build_final_prompt()

        assert "Luxury product on marble table" in final


class TestNanoBackgroundGenerator:
    """Test NanoBackgroundGenerator class."""

    def test_init_default(self):
        """Test default initialization."""
        generator = NanoBackgroundGenerator()

        assert generator.api_key is None
        assert generator.config is not None
        assert generator.config.model == "nanobanana_pro"

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = GenerationConfig(
            model="custom_model",
            steps=10,
            batch_size=5
        )
        generator = NanoBackgroundGenerator(config=config)

        assert generator.config == config

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        generator = NanoBackgroundGenerator(api_key="test_key")

        assert generator.api_key == "test_key"

    def test_generate_from_blueprint_loads_config(self, sample_master_blueprint):
        """Test that generate_from_blueprint loads blueprint correctly."""
        generator = NanoBackgroundGenerator()

        with open(sample_master_blueprint) as f:
            blueprint = yaml.safe_load(f)

        # This would normally generate backgrounds
        # For testing, we just verify it doesn't crash
        # (actual generation is mocked/stubbed)
        results = generator.generate_from_blueprint(
            blueprint=blueprint,
            perspective="eye_level",
            save_images=False
        )

        # Should return empty list (placeholder implementation)
        assert isinstance(results, list)

    def test_generate_single(self):
        """Test generating single background."""
        generator = NanoBackgroundGenerator()

        result = generator.generate_single(
            prompt="Product on table",
            perspective="eye_level"
        )

        # Should return None (placeholder implementation)
        assert result is None or isinstance(result, GeneratedBackground)

    def test_generate_batch(self):
        """Test batch generation."""
        generator = NanoBackgroundGenerator()

        prompt = BackgroundPrompt(
            base_prompt="Test product",
            perspective="eye_level"
        )
        config = GenerationConfig(batch_size=5)

        results = generator.generate_batch(
            prompt=prompt,
            config=config,
            save_images=False
        )

        # Should return list (empty in placeholder)
        assert isinstance(results, list)


class TestGeneratedBackground:
    """Test GeneratedBackground dataclass."""

    def test_creation(self):
        """Test creating GeneratedBackground."""
        image = Image.new("RGB", (100, 100), (255, 0, 0))

        bg = GeneratedBackground(
            image=image,
            prompt="Test prompt",
            index=0,
            perspective="eye_level",
            metadata={"test": "data"}
        )

        assert bg.image == image
        assert bg.prompt == "Test prompt"
        assert bg.index == 0
        assert bg.perspective == "eye_level"
        assert bg.metadata == {"test": "data"}

    def test_creation_with_defaults(self):
        """Test creating GeneratedBackground with default metadata."""
        image = Image.new("RGB", (100, 100), (255, 0, 0))

        bg = GeneratedBackground(
            image=image,
            prompt="Test prompt",
            index=0,
            perspective="eye_level"
        )

        assert bg.metadata == {}


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_generate_backgrounds_from_blueprint_function(self, sample_master_blueprint):
        """Test generate_backgrounds_from_blueprint convenience function."""
        with open(sample_master_blueprint) as f:
            blueprint = yaml.safe_load(f)

        results = generate_backgrounds_from_blueprint(
            blueprint=blueprint,
            perspective="eye_level",
            save_images=False
        )

        assert isinstance(results, list)


@pytest.mark.parametrize("perspective", ["high_angle", "eye_level"])
def test_perspective_injection(perspective):
    """Test that perspective is correctly injected into prompts."""
    prompt = BackgroundPrompt(
        base_prompt="Product on table",
        perspective=perspective
    )

    final = prompt.build_final_prompt()

    if perspective == "high_angle":
        assert "top-down" in final
    else:
        assert "eye-level" in final


@pytest.mark.parametrize("steps,cfg", [
    (4, 2.0),
    (8, 3.5),
    (20, 5.0),
])
def test_generation_config_variations(steps, cfg):
    """Test various generation config combinations."""
    config = GenerationConfig(
        steps=steps,
        cfg_scale=cfg
    )

    assert config.steps == steps
    assert config.cfg_scale == cfg


@pytest.mark.parametrize("aspect_ratio", [
    "1:1",
    "3:4",
    "4:3",
    "9:16",
])
def test_aspect_ratio_support(aspect_ratio):
    """Test support for various aspect ratios."""
    config = GenerationConfig(aspect_ratio=aspect_ratio)

    assert config.aspect_ratio == aspect_ratio
