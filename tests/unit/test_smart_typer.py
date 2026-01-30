"""
Unit tests for SmartTyper.

Tests smart text rendering including:
- Auto-contrast color calculation
- Collision detection
- Psychology-enhanced rendering
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np

from src.meta.ad.generator.template_system.smart_typer import (
    CampaignContent,
    SmartColorCalculator,
    CollisionDetector,
    SmartTyper,
    TextAlignment,
    PositionType,
    render_text_overlay,
)


@pytest.fixture
def sample_campaign_content():
    """Create sample campaign content."""
    return CampaignContent(
        headline="Test Headline",
        sub_text="Test Subtext",
        cta_text="Shop Now",
        brand_color="#FF5733"
    )


@pytest.fixture
def sample_template_spec():
    """Create sample template specification."""
    class MockTemplateSpec:
        template_id = "test_template"
        display_name = "Test Template"
        psychology_driver = "trust"
        layout = {
            "position": "Bottom_Center",
            "margin_y": 80,
            "margin_x": 0,
            "alignment": "center"
        }
        typography = {
            "headline": {
                "font_family": "Sans_Bold",
                "font_size": 48,
                "line_height": 1.2
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
        }
        style = {
            "font_color_logic": "Auto_Contrast",
            "cta_shape": "Pill_Solid",
            "cta_bg_color": "Transparent",
            "shadow_effect": False
        }

    return MockTemplateSpec()


class TestCampaignContent:
    """Test CampaignContent dataclass."""

    def test_from_dict(self):
        """Test creating CampaignContent from dict."""
        data = {
            "campaign_content": {
                "headline": "Test Headline",
                "sub_text": "Test Subtext",
                "cta_text": "Shop Now",
                "brand_color": "#FF5733"
            }
        }

        content = CampaignContent.from_dict(data)

        assert content.headline == "Test Headline"
        assert content.sub_text == "Test Subtext"
        assert content.cta_text == "Shop Now"
        assert content.brand_color == "#FF5733"

    def test_from_dict_with_missing_fields(self):
        """Test creating CampaignContent with optional fields missing."""
        data = {
            "campaign_content": {
                "headline": "Test Headline",
                "brand_color": "#000000"
            }
        }

        content = CampaignContent.from_dict(data)

        assert content.headline == "Test Headline"
        assert content.sub_text is None
        assert content.cta_text is None
        assert content.brand_color == "#000000"

    def test_direct_creation(self):
        """Test creating CampaignContent directly."""
        content = CampaignContent(
            headline="Test",
            sub_text="Sub",
            cta_text="CTA",
            brand_color="#ABCDEF"
        )

        assert content.headline == "Test"
        assert content.sub_text == "Sub"
        assert content.cta_text == "CTA"
        assert content.brand_color == "#ABCDEF"


class TestSmartColorCalculator:
    """Test SmartColorCalculator class."""

    def test_calculate_luminance_dark_image(self):
        """Test luminance calculation for dark image."""
        # Create dark image
        image = Image.new("RGB", (100, 100), (0, 0, 0))
        luminance = SmartColorCalculator.calculate_luminance(image)

        assert luminance == 0.0

    def test_calculate_luminance_light_image(self):
        """Test luminance calculation for light image."""
        # Create light image
        image = Image.new("RGB", (100, 100), (255, 255, 255))
        luminance = SmartColorCalculator.calculate_luminance(image)

        assert luminance == 255.0

    def test_calculate_luminance_mixed_image(self):
        """Test luminance calculation for mixed image."""
        # Create image with gray
        image = Image.new("RGB", (100, 100), (128, 128, 128))
        luminance = SmartColorCalculator.calculate_luminance(image)

        assert luminance == 128.0

    def test_get_contrast_color_light_background(self):
        """Test contrast color for light background."""
        color = SmartColorCalculator.get_contrast_color(
            background_luminance=200.0  # Light
        )

        assert color == "#000000"  # Black text

    def test_get_contrast_color_dark_background(self):
        """Test contrast color for dark background."""
        color = SmartColorCalculator.get_contrast_color(
            background_luminance=50.0  # Dark
        )

        assert color == "#FFFFFF"  # White text

    def test_get_contrast_color_threshold(self):
        """Test contrast color at threshold boundary."""
        # At threshold (128), condition is > 128, so returns white (else branch)
        color = SmartColorCalculator.get_contrast_color(
            background_luminance=128.0
        )

        assert color == "#FFFFFF"  # White text (not > threshold)

        # Just above threshold, returns black
        color = SmartColorCalculator.get_contrast_color(
            background_luminance=129.0
        )

        assert color == "#000000"  # Black text (> threshold)

    def test_get_contrast_color_fixed_override(self):
        """Test fixed color override."""
        color = SmartColorCalculator.get_contrast_color(
            background_luminance=200.0,
            fixed_color="#FF0000"
        )

        assert color == "#FF0000"


class TestCollisionDetector:
    """Test CollisionDetector class."""

    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        box1 = (0, 0, 50, 50)  # x, y, w, h
        box2 = (100, 100, 50, 50)

        iou = CollisionDetector.calculate_iou(box1, box2)

        assert iou == 0.0

    def test_calculate_iou_complete_overlap(self):
        """Test IoU calculation with complete overlap."""
        box1 = (0, 0, 50, 50)
        box2 = (0, 0, 50, 50)

        iou = CollisionDetector.calculate_iou(box1, box2)

        assert iou == 1.0

    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 100, 100)

        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 ≈ 0.143
        iou = CollisionDetector.calculate_iou(box1, box2)

        assert 0.14 < iou < 0.15

    def test_detect_collision_no_collision(self):
        """Test collision detection with no collision."""
        # Create mask with product in center
        mask = Image.new("L", (300, 300), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 200, 200], fill=255)

        # Text box far away
        text_box = (0, 0, 50, 50)

        collision = CollisionDetector.detect_collision(
            text_box=text_box,
            product_mask=mask,
            threshold=0.0
        )

        assert collision is False

    def test_detect_collision_with_collision(self):
        """Test collision detection with collision."""
        # Create mask with product in center
        mask = Image.new("L", (300, 300), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 200, 200], fill=255)

        # Text box overlapping product
        text_box = (90, 90, 50, 50)

        collision = CollisionDetector.detect_collision(
            text_box=text_box,
            product_mask=mask,
            threshold=0.0
        )

        assert collision is True

    def test_detect_collision_with_threshold(self):
        """Test collision detection with threshold."""
        # Create mask with product
        mask = Image.new("L", (300, 300), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 200, 200], fill=255)

        # Text box that overlaps significantly with product
        # Product: 100x100 at (100, 100)
        # Text: 80x80 at (110, 110) - mostly overlapping
        text_box = (110, 110, 80, 80)

        # Calculate expected IoU:
        # Intersection: 70x70 = 4900
        # Union: 6400 + 10000 - 4900 = 11500
        # IoU = 4900 / 11500 = 0.426 (42.6%)
        # With threshold=0.3, this should be a collision
        collision = CollisionDetector.detect_collision(
            text_box=text_box,
            product_mask=mask,
            threshold=0.3  # 30% overlap required
        )

        # Should have collision (overlap > 30%)
        assert collision is True

    def test_find_non_colliding_position_simple(self):
        """Test finding non-colliding position."""
        # Create mask with product in center
        mask = Image.new("L", (300, 300), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 200, 200], fill=255)

        # Text size that would collide at center
        text_size = (150, 50)
        initial_pos = (75, 75)  # Would collide

        position = CollisionDetector.find_non_colliding_position(
            text_size=text_size,
            product_mask=mask,
            canvas_size=(300, 300),
            initial_position=initial_pos,
            max_attempts=10
        )

        # Should find a position
        assert position is not None
        assert isinstance(position, tuple)
        assert len(position) == 2


class TestSmartTyper:
    """Test SmartTyper class."""

    def test_init_default(self):
        """Test default initialization."""
        typer = SmartTyper()

        assert typer.smart_color_enabled is True
        assert typer.collision_detection_enabled is True
        assert typer.color_calculator is not None
        assert typer.collision_detector is not None

    def test_init_custom(self):
        """Test initialization with custom settings."""
        typer = SmartTyper(
            smart_color_enabled=False,
            collision_detection_enabled=False
        )

        assert typer.smart_color_enabled is False
        assert typer.collision_detection_enabled is False

    def test_render_text_basic(self, sample_campaign_content, sample_template_spec):
        """Test basic text rendering."""
        typer = SmartTyper()

        # Create simple background
        background = Image.new("RGB", (1080, 1080), (240, 240, 240))

        result = typer.render_text(
            image=background,
            product_mask=None,
            campaign_content=sample_campaign_content,
            template_spec=sample_template_spec,
            product_position=(100, 100),
            product_size=(200, 200)
        )

        assert result.image is not None
        assert result.image.size == (1080, 1080)
        assert "headline" in result.text_positions
        assert "headline" in result.colors_used

    def test_render_text_without_subtext(self, sample_template_spec):
        """Test rendering without subtext."""
        typer = SmartTyper()

        content = CampaignContent(
            headline="Test Headline",
            cta_text="Shop Now",
            brand_color="#FF5733"
        )

        background = Image.new("RGB", (1080, 1080), (240, 240, 240))

        result = typer.render_text(
            image=background,
            product_mask=None,
            campaign_content=content,
            template_spec=sample_template_spec,
            product_position=(100, 100),
            product_size=(200, 200)
        )

        assert "headline" in result.text_positions
        # sub_text should not be in positions
        assert "sub_text" not in result.text_positions or result.text_positions.get("sub_text") is None

    def test_render_text_without_cta(self, sample_campaign_content, sample_template_spec):
        """Test rendering without CTA."""
        typer = SmartTyper()

        content = CampaignContent(
            headline="Test Headline",
            sub_text="Test Subtext",
            brand_color="#FF5733"
        )

        background = Image.new("RGB", (1080, 1080), (240, 240, 240))

        result = typer.render_text(
            image=background,
            product_mask=None,
            campaign_content=content,
            template_spec=sample_template_spec,
            product_position=(100, 100),
            product_size=(200, 200)
        )

        assert "headline" in result.text_positions
        # CTA should not be in positions
        assert "cta" not in result.text_positions or result.text_positions.get("cta") is None


class TestRenderTextOverlay:
    """Test render_text_overlay convenience function."""

    def test_render_text_overlay_convenience(self, sample_campaign_content, sample_template_spec):
        """Test convenience function for text rendering."""
        background = Image.new("RGB", (1080, 1080), (200, 200, 200))

        result = render_text_overlay(
            image=background,
            product_mask=None,
            campaign_content=sample_campaign_content,
            template_spec=sample_template_spec,
            product_position=(100, 100)
        )

        assert result is not None
        assert result.size == (1080, 1080)

    def test_render_text_overlay_with_dict_content(self, sample_template_spec):
        """Test convenience function with dict content."""
        background = Image.new("RGB", (1080, 1080), (200, 200, 200))

        content_dict = {
            "campaign_content": {
                "headline": "Test",
                "cta_text": "Shop Now",
                "brand_color": "#FF5733"
            }
        }

        result = render_text_overlay(
            image=background,
            product_mask=None,
            campaign_content=content_dict,
            template_spec=sample_template_spec
        )

        assert result is not None
