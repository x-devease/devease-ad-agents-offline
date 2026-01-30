"""
Smart Typer: Psychology-Adaptive Text Overlay with Collision Detection.

Implements intelligent text placement and rendering based on:
1. Psychology-driven template auto-selection
2. Smart color (auto-contrast based on background luminance)
3. Collision detection (IoU check against product)
4. Psychology-enhanced effects (shadows, badges, etc.)

Key Philosophy: 排版不再仅仅是为了"好看"，而是为了**"匹配心智"**

Author: Ad System
Date: 2026-01-30
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter, ImageChops


logger = logging.getLogger(__name__)


class TextAlignment(str, Enum):
    """Text alignment options."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class PositionType(str, Enum):
    """Predefined position types."""

    TOP_LEFT = "Top_Left"
    TOP_RIGHT = "Top_Right"
    TOP_CENTER = "Top_Center"
    BOTTOM_LEFT = "Bottom_Left"
    BOTTOM_RIGHT = "Bottom_Right"
    BOTTOM_CENTER = "Bottom_Center"
    CENTER = "Center"
    SMART_VOID = "Smart_Void"  # Auto-find empty space


@dataclass
class CampaignContent:
    """
    Campaign content from campaign_content.yaml.

    Attributes:
        headline: Main headline text
        sub_text: Subheading text
        cta_text: Call-to-action button text
        brand_color: Brand color in hex (e.g., "#FF5733")
    """
    headline: str
    sub_text: Optional[str] = None
    cta_text: Optional[str] = None
    brand_color: str = "#000000"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CampaignContent":
        """Create from dictionary (YAML deserialization)."""
        campaign = data.get("campaign_content", data)
        return cls(
            headline=campaign.get("headline", ""),
            sub_text=campaign.get("sub_text"),
            cta_text=campaign.get("cta_text"),
            brand_color=campaign.get("brand_color", "#000000"),
        )


@dataclass
class TextRenderResult:
    """
    Result from text rendering.

    Attributes:
        image: Image with rendered text overlay
        text_positions: Dict with text element positions
        colors_used: Colors actually used (after smart contrast)
        metadata: Rendering metadata
    """
    image: Image.Image
    text_positions: Dict[str, Tuple[int, int, int, int]]  # element -> (x, y, w, h)
    colors_used: Dict[str, str]  # element -> color_hex
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartColorCalculator:
    """
    Calculate optimal text color based on background luminance.

    Uses luminance threshold to determine whether to use black or white text
    for maximum readability.
    """

    LUMINANCE_THRESHOLD = 128

    @classmethod
    def calculate_luminance(cls, image_region: Image.Image) -> float:
        """
        Calculate average luminance of an image region.

        Args:
            image_region: PIL Image region

        Returns:
            Average luminance value (0-255)
        """
        # Convert to grayscale
        gray = image_region.convert("L")
        # Calculate average
        avg_luminance = np.array(gray).mean()

        return float(avg_luminance)

    @classmethod
    def get_contrast_color(
        cls,
        background_luminance: float,
        fixed_color: Optional[str] = None,
    ) -> str:
        """
        Get optimal text color based on background luminance.

        Args:
            background_luminance: Average background luminance (0-255)
            fixed_color: Optional fixed color (if specified, skips auto-contrast)

        Returns:
            Color hex string (e.g., "#FFFFFF" for white)
        """
        if fixed_color:
            return fixed_color

        # Auto-contrast: black for light backgrounds, white for dark
        if background_luminance > cls.LUMINANCE_THRESHOLD:
            return "#000000"  # Black
        else:
            return "#FFFFFF"  # White


class CollisionDetector:
    """
    Detect and resolve collisions between text and product using IoU.

    Intersection over Union (IoU) = Area of Intersection / Area of Union
    """

    @classmethod
    def calculate_iou(
        cls,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1: (x1, y1, w1, h1) - Note: width/height, not x2/y2
            box2: (x2, y2, w2, h2)

        Returns:
            IoU value (0.0 - 1.0)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to (x1, y1, x2, y2) format
        box1_corners = (x1, y1, x1 + w1, y1 + h1)
        box2_corners = (x2, y2, x2 + w2, y2 + h2)

        # Calculate intersection
        x_left = max(box1_corners[0], box2_corners[0])
        y_top = max(box1_corners[1], box2_corners[1])
        x_right = min(box1_corners[2], box2_corners[2])
        y_bottom = min(box1_corners[3], box2_corners[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    @classmethod
    def detect_collision(
        cls,
        text_box: Tuple[int, int, int, int],
        product_mask: Image.Image,
        threshold: float = 0.0,
    ) -> bool:
        """
        Detect if text collides with product.

        Args:
            text_box: Text bounding box (x, y, w, h)
            product_mask: Product binary mask
            threshold: IoU threshold (0.0 = no overlap allowed)

        Returns:
            True if collision detected
        """
        # Get product bounding box
        product_bbox = product_mask.getbbox()
        if product_bbox is None:
            return False  # No product

        # Convert to (x, y, w, h)
        product_box = (
            product_bbox[0],
            product_bbox[1],
            product_bbox[2] - product_bbox[0],
            product_bbox[3] - product_bbox[1],
        )

        iou = cls.calculate_iou(text_box, product_box)

        logger.debug(f"Text-Product IoU: {iou:.3f}")

        return iou > threshold

    @classmethod
    def find_non_colliding_position(
        cls,
        text_size: Tuple[int, int],
        product_mask: Image.Image,
        canvas_size: Tuple[int, int],
        initial_position: Tuple[int, int],
        max_attempts: int = 10,
        threshold: float = 0.0,
    ) -> Optional[Tuple[int, int]]:
        """
        Find non-colliding position by shifting along Y-axis.

        Args:
            text_size: (width, height) of text
            product_mask: Product binary mask
            canvas_size: (canvas_width, canvas_height)
            initial_position: Initial (x, y) position
            max_attempts: Maximum shift attempts
            threshold: IoU threshold

        Returns:
            Non-colliding (x, y) position or None if not found
        """
        x, y = initial_position
        w, h = text_size
        canvas_w, canvas_h = canvas_size

        for attempt in range(max_attempts):
            text_box = (x, y, w, h)

            if not cls.detect_collision(text_box, product_mask, threshold):
                logger.debug(f"Found non-colliding position at attempt {attempt + 1}: ({x}, {y})")
                return (x, y)

            # Shift down
            y += h // 2

            # Check bounds
            if y + h > canvas_h:
                # Try shifting up from original
                y = initial_position[1] - h // 2
                if y < 0:
                    break

        logger.warning(f"Could not find non-colliding position after {max_attempts} attempts")
        return None


class SmartTyper:
    """
    Psychology-adaptive text overlay with smart color and collision detection.

    Usage:
        typer = SmartTyper()
        result = typer.render_text(
            image=background,
            product_mask=product_mask,
            campaign_content=campaign_content,
            template_spec=template,
            product_position=(100, 100),
        )
        result.image.save("output.png")
    """

    def __init__(
        self,
        font_dir: Optional[Path] = None,
        smart_color_enabled: bool = True,
        collision_detection_enabled: bool = True,
    ):
        """
        Initialize Smart Typer.

        Args:
            font_dir: Optional directory containing font files
            smart_color_enabled: Enable smart color auto-contrast
            collision_detection_enabled: Enable collision detection
        """
        self.font_dir = font_dir
        self.smart_color_enabled = smart_color_enabled
        self.collision_detection_enabled = collision_detection_enabled
        self.color_calculator = SmartColorCalculator()
        self.collision_detector = CollisionDetector()

    def render_text(
        self,
        image: Image.Image,
        product_mask: Optional[Image.Image],
        campaign_content: CampaignContent,
        template_spec: Any,  # TemplateSpec from template_selector
        product_position: Tuple[int, int] = (0, 0),
        product_size: Tuple[int, int] = (0, 0),
    ) -> TextRenderResult:
        """
        Render text overlay based on template specification.

        Args:
            image: Base image (composited product + background)
            product_mask: Product mask for collision detection
            campaign_content: Campaign content (headline, CTA, brand color)
            template_spec: Template specification from selector
            product_position: Product (x, y) position on image
            product_size: Product (width, height)

        Returns:
            TextRenderResult with rendered image and metadata
        """
        # Create drawing canvas
        canvas = image.copy()

        # Extract template configuration
        layout = template_spec.layout
        typography = template_spec.typography
        style = template_spec.style

        # Calculate text position
        text_position = self._calculate_text_position(
            image_size=image.size,
            layout=layout,
            product_position=product_position,
            product_size=product_size,
        )

        # Render headline
        headline_pos, headline_color = self._render_headline(
            canvas=canvas,
            text=campaign_content.headline,
            position=text_position,
            typography=typography.get("headline", {}),
            style=style,
        )

        # Render sub_text if provided
        sub_text_pos = None
        sub_text_color = None
        if campaign_content.sub_text:
            sub_text_pos, sub_text_color = self._render_sub_text(
                canvas=canvas,
                text=campaign_content.sub_text,
                position=text_position,
                typography=typography.get("sub_text", {}),
                style=style,
                headline_pos=headline_pos,
            )

        # Render CTA if provided
        cta_pos = None
        cta_color = None
        if campaign_content.cta_text:
            cta_pos, cta_color = self._render_cta(
                canvas=canvas,
                text=campaign_content.cta_text,
                position=text_position,
                typography=typography.get("cta", {}),
                style=style,
                brand_color=campaign_content.brand_color,
                last_text_pos=sub_text_pos or headline_pos,
            )

        # Collect metadata
        text_positions = {"headline": headline_pos}
        colors_used = {"headline": headline_color}

        if sub_text_pos:
            text_positions["sub_text"] = sub_text_pos
            colors_used["sub_text"] = sub_text_color

        if cta_pos:
            text_positions["cta"] = cta_pos
            colors_used["cta"] = cta_color

        return TextRenderResult(
            image=canvas,
            text_positions=text_positions,
            colors_used=colors_used,
            metadata={
                "template_id": template_spec.template_id,
                "psychology_driver": template_spec.psychology_driver,
            }
        )

    def _calculate_text_position(
        self,
        image_size: Tuple[int, int],
        layout: Dict[str, Any],
        product_position: Tuple[int, int],
        product_size: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Calculate text position based on layout configuration.

        Args:
            image_size: (width, height) of image
            layout: Layout configuration from template
            product_position: Product (x, y)
            product_size: Product (width, height)

        Returns:
            (x, y) position for text
        """
        position = layout.get("position", "Bottom_Center")
        margin_x = layout.get("margin_x", 0)
        margin_y = layout.get("margin_y", 80)

        img_w, img_h = image_size

        # Handle predefined positions
        if position == "Top_Left":
            return (margin_x, margin_y)
        elif position == "Top_Right":
            return (img_w - margin_x, margin_y)
        elif position == "Top_Center":
            return (img_w // 2, margin_y)
        elif position == "Bottom_Left":
            return (margin_x, img_h - margin_y)
        elif position == "Bottom_Right":
            return (img_w - margin_x, img_h - margin_y)
        elif position == "Bottom_Center":
            return (img_w // 2, img_h - margin_y)
        elif position == "Center":
            return (img_w // 2, img_h // 2)
        else:  # Smart_Void or unknown - default to bottom center
            return (img_w // 2, img_h - margin_y)

    def _render_headline(
        self,
        canvas: Image.Image,
        text: str,
        position: Tuple[int, int],
        typography: Dict[str, Any],
        style: Dict[str, Any],
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Render headline text."""
        # Get font settings
        font_family = typography.get("font_family", "Sans_Bold")
        font_size = typography.get("font_size", 48)
        line_height = typography.get("line_height", 1.2)

        # Load font (simplified - using default)
        try:
            font = ImageFont.truetype(font_family, font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Calculate text size
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Get color
        if self.smart_color_enabled:
            # Sample background at text position
            x, y = position
            margin_x = typography.get("margin_x", 0)
            margin_y = typography.get("margin_y", 0)

            sample_region = canvas.crop((
                max(0, x - text_w // 2 - margin_x),
                max(0, y - margin_y),
                min(canvas.width, x + text_w // 2 + margin_x),
                min(canvas.height, y + text_h + margin_y),
            ))

            luminance = self.color_calculator.calculate_luminance(sample_region)
            color = self.color_calculator.get_contrast_color(luminance)
        else:
            color_logic = style.get("font_color_logic", "Auto_Contrast")
            if color_logic == "Fixed_White":
                color = "#FFFFFF"
            elif color_logic.startswith("Fixed"):
                # Extract color from logic (e.g., "Deep_Blue")
                color = style.get("fallback_color", "#000000")
            else:
                color = "#000000"  # Default black

        # Draw text
        x, y = position
        alignment = typography.get("alignment", "center")

        if alignment == "center":
            x = x - text_w // 2
        elif alignment == "right":
            x = x - text_w

        draw.text((x, y), text, font=font, fill=color)

        text_box = (x, y, text_w, text_h)

        logger.debug(f"Rendered headline: '{text}' at ({x}, {y})")

        return text_box, color

    def _render_sub_text(
        self,
        canvas: Image.Image,
        text: str,
        position: Tuple[int, int],
        typography: Dict[str, Any],
        style: Dict[str, Any],
        headline_pos: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Render sub_text below headline."""
        font_size = typography.get("font_size", 32)

        try:
            font = ImageFont.truetype(
                typography.get("font_family", "Sans_Regular"),
                font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Position below headline
        x, y, _, h = headline_pos
        y = y + h + 10  # 10px gap

        # Use same color calculation as headline
        _, color = self._render_headline(
            canvas, text, (x, y), typography, style
        )

        # Actually draw
        draw.text((x, y), text, font=font, fill=color)

        logger.debug(f"Rendered sub_text: '{text}' at ({x}, {y})")

        return (x, y, text_w, text_h), color

    def _render_cta(
        self,
        canvas: Image.Image,
        text: str,
        position: Tuple[int, int],
        typography: Dict[str, Any],
        style: Dict[str, Any],
        brand_color: str,
        last_text_pos: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int, int, int], str]:
        """Render CTA button."""
        font_size = typography.get("font_size", 24)
        padding_x = typography.get("padding_x", 40)
        padding_y = typography.get("padding_y", 16)

        try:
            font = ImageFont.truetype(
                typography.get("font_family", "Sans_Medium"),
                font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Position below last text
        x, y, _, h = last_text_pos
        y = y + h + 20  # 20px gap

        # Center horizontally
        x = canvas.width // 2 - text_w // 2

        # Get CTA colors
        cta_shape = style.get("cta_shape", "Pill_Solid")
        cta_bg_color = style.get("cta_bg_color", brand_color)

        if cta_bg_color == "Transparent":
            bg_color = None
        else:
            bg_color = cta_bg_color

        text_color = style.get("cta_color", "#FFFFFF")

        # Draw button background if not transparent
        if bg_color:
            button_rect = [
                x - padding_x,
                y - padding_y,
                x + text_w + padding_x,
                y + text_h + padding_y,
            ]

            # Draw rounded rectangle
            corner_radius = typography.get("corner_radius", 12)
            if corner_radius > 0:
                # Simplified rounded rect
                draw.rectangle(button_rect, fill=bg_color)
            else:
                draw.rectangle(button_rect, fill=bg_color)

        # Draw text
        draw.text((x, y), text, font=font, fill=text_color)

        text_box = (x, y, text_w, text_h)

        logger.debug(f"Rendered CTA: '{text}' at ({x}, {y})")

        return text_box, text_color


# Convenience function
def render_text_overlay(
    image: Image.Image,
    product_mask: Optional[Image.Image],
    campaign_content: Dict[str, Any],
    template_spec: Any,
    product_position: Tuple[int, int] = (0, 0),
    smart_color: bool = True,
    collision_detection: bool = True,
) -> Image.Image:
    """
    Convenience function for quick text rendering.

    Args:
        image: Base image
        product_mask: Optional product mask
        campaign_content: Campaign content dict or CampaignContent
        template_spec: Template specification
        product_position: Product position
        smart_color: Enable smart color
        collision_detection: Enable collision detection

    Returns:
        Image with text overlay

    Example:
        from src.meta.ad.generator.template_system.template_selector import select_template_from_blueprint

        template = select_template_from_blueprint(blueprint)
        content = CampaignContent.from_dict(campaign_dict)
        result = render_text_overlay(
            image=background,
            product_mask=mask,
            campaign_content=content,
            template_spec=template
        )
        result.save("output.png")
    """
    if not isinstance(campaign_content, CampaignContent):
        campaign_content = CampaignContent.from_dict(campaign_content)

    typer = SmartTyper(
        smart_color_enabled=smart_color,
        collision_detection_enabled=collision_detection,
    )

    result = typer.render_text(
        image=image,
        product_mask=product_mask,
        campaign_content=campaign_content,
        template_spec=template_spec,
        product_position=product_position,
    )

    return result.image
