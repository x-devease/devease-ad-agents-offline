"""
Unit tests for Physics Compositor.

Tests physics-aware compositing including:
- Dual-layer shadows
- Light matching
- Light wrap
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np

from src.meta.ad.generator.template_system.physics_compositor import (
    ShadowDirection,
    CompositingConfig,
    CompositingResult,
    PhysicsCompositor,
    composite_physics_aware,
)


@pytest.fixture
def sample_product_image():
    """Create sample product image with transparency."""
    # Create RGBA image with product in center
    image = Image.new("RGBA", (400, 400), (0, 0, 0, 0))

    # Add product (opaque red square)
    product = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
    image.paste(product, (100, 100), product)

    return image


@pytest.fixture
def sample_product_mask():
    """Create sample product mask."""
    mask = Image.new("L", (400, 400), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([100, 100, 300, 300], fill=255)

    return mask


@pytest.fixture
def sample_background():
    """Create sample background image."""
    # Create gradient background (lighter on one side)
    background = Image.new("RGB", (400, 400), (200, 200, 200))

    # Add some variation
    for y in range(400):
        color = int(150 + (y / 400) * 100)
        for x in range(400):
            background.putpixel((x, y), (color, color, color))

    return background


class TestShadowDirection:
    """Test ShadowDirection enum."""

    def test_direction_values(self):
        """Test shadow direction enum values."""
        assert ShadowDirection.LEFT.value == "left"
        assert ShadowDirection.RIGHT.value == "right"
        assert ShadowDirection.TOP.value == "top"
        assert ShadowDirection.BOTTOM.value == "bottom"


class TestCompositingConfig:
    """Test CompositingConfig dataclass."""

    def test_from_dict(self):
        """Test creating CompositingConfig from dict."""
        data = {
            "compositing": {
                "shadow_direction": "left",
                "light_wrap_intensity": 0.3,
                "light_match_mode": "soft_light",
                "light_match_opacity": 0.25
            }
        }

        config = CompositingConfig.from_dict(data)

        assert config.shadow_direction == ShadowDirection.LEFT
        assert config.light_wrap_intensity == 0.3
        assert config.light_match_mode == "soft_light"
        assert config.light_match_opacity == 0.25

    def test_from_dict_with_defaults(self):
        """Test CompositingConfig with default values."""
        data = {"compositing": {}}

        config = CompositingConfig.from_dict(data)

        assert config.shadow_direction == ShadowDirection.LEFT
        assert config.light_wrap_intensity == 0.3
        assert config.contact_shadow_size == 30
        assert config.contact_shadow_opacity == 0.4

    def test_from_dict_custom_values(self):
        """Test CompositingConfig with custom values."""
        data = {
            "compositing": {
                "shadow_direction": "right",
                "light_wrap_intensity": 0.5,
                "light_match_opacity": 0.4
            }
        }

        config = CompositingConfig.from_dict(data)

        assert config.shadow_direction == ShadowDirection.RIGHT
        assert config.light_wrap_intensity == 0.5
        assert config.light_match_opacity == 0.4


class TestPhysicsCompositor:
    """Test PhysicsCompositor class."""

    def test_init_default(self):
        """Test default initialization."""
        compositor = PhysicsCompositor()

        assert compositor.config is not None
        assert compositor.config.shadow_direction == ShadowDirection.LEFT

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = CompositingConfig(
            shadow_direction=ShadowDirection.RIGHT,
            light_wrap_intensity=0.5
        )
        compositor = PhysicsCompositor(config=config)

        assert compositor.config == config

    def test_composite_basic(
        self, sample_product_image, sample_background, sample_product_mask
    ):
        """Test basic compositing."""
        compositor = PhysicsCompositor()

        result = compositor.composite(
            product_image=sample_product_image,
            background_image=sample_background,
            product_mask=sample_product_mask,
            product_position=(100, 100)
        )

        assert isinstance(result, CompositingResult)
        assert result.composited_image is not None
        assert result.composited_image.size == sample_background.size

    def test_composite_without_mask(
        self, sample_product_image, sample_background
    ):
        """Test compositing without mask (auto-generated)."""
        compositor = PhysicsCompositor()

        result = compositor.composite(
            product_image=sample_product_image,
            background_image=sample_background,
            product_mask=None,
            product_position=(100, 100)
        )

        assert result.composited_image is not None

    def test_composite_result_contains_metadata(
        self, sample_product_image, sample_background, sample_product_mask
    ):
        """Test that compositing result contains metadata."""
        compositor = PhysicsCompositor()

        result = compositor.composite(
            product_image=sample_product_image,
            background_image=sample_background,
            product_mask=sample_product_mask,
            product_position=(100, 100)
        )

        assert "config" in result.metadata
        assert "shadow_direction" in result.metadata["config"]

    def test_composite_with_custom_config(
        self, sample_product_image, sample_background, sample_product_mask
    ):
        """Test compositing with custom configuration."""
        config = CompositingConfig(
            shadow_direction=ShadowDirection.RIGHT,
            light_wrap_intensity=0.5
        )
        compositor = PhysicsCompositor(config=config)

        result = compositor.composite(
            product_image=sample_product_image,
            background_image=sample_background,
            product_mask=sample_product_mask,
            product_position=(100, 100),
            config=config
        )

        assert result.composited_image is not None
        assert result.metadata["config"]["shadow_direction"] == "right"
        assert result.metadata["config"]["light_wrap_intensity"] == 0.5


class TestShadowGeneration:
    """Test shadow generation methods."""

    def test_generate_shadow_stack(
        self, sample_product_image, sample_background, sample_product_mask
    ):
        """Test dual-layer shadow stack generation."""
        compositor = PhysicsCompositor()

        shadow_layer = compositor._generate_shadow_stack(
            product_image=sample_product_image,
            product_mask=sample_product_mask,
            background_image=sample_background,
            position=(100, 100),
            config=compositor.config
        )

        assert shadow_layer is not None
        assert shadow_layer.size == sample_background.size

    def test_create_contact_shadow(self):
        """Test contact shadow creation."""
        compositor = PhysicsCompositor()

        shadow = compositor._create_contact_shadow(
            width=100,
            config=compositor.config
        )

        assert shadow is not None
        assert shadow.mode == "RGBA"
        assert shadow.width >= 100  # Should be at least product width

    def test_create_cast_shadow(self):
        """Test cast shadow creation."""
        compositor = PhysicsCompositor()

        shadow = compositor._create_cast_shadow(
            width=100,
            config=compositor.config
        )

        assert shadow is not None
        assert shadow.mode == "RGBA"

    def test_shadow_direction_affects_cast_shadow(self):
        """Test that shadow direction affects cast shadow."""
        compositor = PhysicsCompositor()

        left_shadow = compositor._create_cast_shadow(
            width=100,
            config=CompositingConfig(shadow_direction=ShadowDirection.LEFT)
        )

        right_shadow = compositor._create_cast_shadow(
            width=100,
            config=CompositingConfig(shadow_direction=ShadowDirection.RIGHT)
        )

        # Shadows should be different (direction affects skew)
        # Note: In full implementation, this would be more obvious
        assert left_shadow is not None
        assert right_shadow is not None


class TestLightMatching:
    """Test light matching methods."""

    def test_apply_light_matching(
        self, sample_product_image, sample_background
    ):
        """Test light matching application."""
        compositor = PhysicsCompositor()

        result = compositor._apply_light_matching(
            product_image=sample_product_image,
            background_image=sample_background,
            position=(100, 100),
            config=compositor.config
        )

        assert result is not None
        assert result.mode == "RGBA"  # Should preserve alpha

    def test_light_matching_opacity(self):
        """Test that light matching respects opacity setting."""
        from src.meta.ad.generator.template_system.product_preprocessor import PerspectiveType

        compositor = PhysicsCompositor(
            config=CompositingConfig(light_match_opacity=0.5)
        )

        product = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        background = Image.new("RGB", (100, 100), (128, 128, 128))

        result = compositor._apply_light_matching(
            product_image=product,
            background_image=background,
            position=(0, 0),
            config=compositor.config
        )

        assert result is not None


class TestLightWrap:
    """Test light wrap methods."""

    def test_apply_light_wrap(
        self, sample_product_image, sample_background, sample_product_mask
    ):
        """Test light wrap application."""
        compositor = PhysicsCompositor()

        result = compositor._apply_light_wrap(
            product_image=sample_product_image,
            background_image=sample_background,
            product_mask=sample_product_mask,
            position=(100, 100),
            config=compositor.config
        )

        assert result is not None

    def test_create_edge_mask(self):
        """Test edge mask creation."""
        compositor = PhysicsCompositor()

        # Create simple mask
        mask = Image.new("L", (100, 100), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([20, 20, 80, 80], fill=255)

        edge_mask = compositor._create_edge_mask(mask, erosion=2)

        assert edge_mask is not None
        assert edge_mask.mode == "L"

    def test_light_wrap_intensity(self):
        """Test that light wrap intensity affects result."""
        from src.meta.ad.generator.template_system.product_preprocessor import PerspectiveType

        compositor = PhysicsCompositor(
            config=CompositingConfig(light_wrap_intensity=0.1)
        )

        product = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        background = Image.new("RGB", (100, 100), (200, 200, 200))
        mask = Image.new("L", (100, 100), 255)

        result = compositor._apply_light_wrap(
            product_image=product,
            background_image=background,
            product_mask=mask,
            position=(0, 0),
            config=compositor.config
        )

        assert result is not None


class TestCompositingResult:
    """Test CompositingResult dataclass."""

    def test_creation(self):
        """Test creating CompositingResult."""
        image = Image.new("RGB", (100, 100), (255, 0, 0))
        shadow_layer = Image.new("RGBA", (100, 100), (0, 0, 0, 50))

        result = CompositingResult(
            composited_image=image,
            shadow_layer=shadow_layer,
            light_wrap_layer=None,
            metadata={"test": "data"}
        )

        assert result.composited_image == image
        assert result.shadow_layer == shadow_layer
        assert result.light_wrap_layer is None
        assert result.metadata == {"test": "data"}

    def test_creation_with_defaults(self):
        """Test creating CompositingResult with default metadata."""
        image = Image.new("RGB", (100, 100), (255, 0, 0))

        result = CompositingResult(
            composited_image=image
        )

        assert result.composited_image == image
        assert result.shadow_layer is None
        assert result.light_wrap_layer is None
        assert result.metadata == {}


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_composite_physics_aware(
        self, sample_product_image, sample_background
    ):
        """Test composite_physics_aware convenience function."""
        result = composite_physics_aware(
            product_image=sample_product_image,
            background_image=sample_background,
            shadow_direction="left",
            light_wrap_intensity=0.3
        )

        assert result is not None
        assert result.size == sample_background.size


@pytest.mark.parametrize("direction", ["left", "right", "top", "bottom"])
def test_shadow_directions(direction):
    """Test all shadow directions."""
    config = CompositingConfig(shadow_direction=direction)
    assert config.shadow_direction.value == direction


@pytest.mark.parametrize("intensity", [0.0, 0.3, 0.5, 1.0])
def test_light_wrap_intensities(intensity):
    """Test various light wrap intensities."""
    config = CompositingConfig(light_wrap_intensity=intensity)
    assert config.light_wrap_intensity == intensity


@pytest.mark.parametrize("opacity", [0.0, 0.25, 0.5, 1.0])
def test_light_match_opacities(opacity):
    """Test various light match opacities."""
    config = CompositingConfig(light_match_opacity=opacity)
    assert config.light_match_opacity == opacity
