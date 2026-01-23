"""
Text overlay module for adding custom text to generated images.
Supports multiple text elements with configurable positioning and styling.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)


class TextPosition(Enum):
    """Text position anchors."""

    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    CENTER = "center"


@dataclass
class TextElementConfig:
    """Configuration for a single text element."""

    text: str
    position: Union[str, TextPosition] = TextPosition.BOTTOM_RIGHT
    font_size: int = 24
    font_path: Optional[str] = None  # None = use default
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA white
    background_color: Optional[Tuple[int, int, int, int]] = (
        None  # None = transparent
    )
    background_padding: int = 10  # pixels
    background_corner_radius: int = 8
    margin_x: int = 20  # pixels from edge
    margin_y: int = 20  # pixels from edge
    shadow_offset: int = 2
    shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 128)
    max_width: Optional[int] = None  # For text wrapping (None = no limit)


@dataclass
class TextOverlayConfig:
    """Configuration for text overlay system."""

    enabled: bool = False
    elements: List[TextElementConfig] = None
    # Default font for all elements (can be overridden per-element)
    default_font_size: int = 24
    default_font_path: Optional[str] = None
    # Default styling
    default_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    default_background_color: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 180)
    default_background_padding: int = 10
    default_background_corner_radius: int = 8
    default_margin: int = 20
    default_shadow: bool = True

    def __post_init__(self):
        """Initialize elements list if None."""
        if self.elements is None:
            self.elements = []


class TextOverlay:
    """
    Text overlay system for adding customizable text to images.
    Supports multiple text elements with independent positioning and styling.
    """

    # Font loading fallback chain (same as watermark)
    FONT_FALLBACKS = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/Avenir.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    def __init__(self, config: TextOverlayConfig):
        """
        Initialize text overlay with configuration.

        Args:
            config: TextOverlayConfig with all settings
        """
        self.config = config
        self._font_cache: Dict[Tuple[Optional[str], int], Optional[ImageFont.FreeTypeFont]] = {}

    def apply_text_overlay(
        self, image_path: str, output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Apply text overlay to an image.

        Args:
            image_path: Path to input image
            output_path: Path to save output (None = overwrite input)

        Returns:
            Path to output image or None if failed
        """
        if not self.config.enabled or not self.config.elements:
            logger.debug("Text overlay disabled or no elements configured")
            return image_path

        try:
            with Image.open(image_path) as img:
                # Convert to RGBA for transparency support
                if img.mode != "RGBA":
                    img = img.convert("RGBA")

                # Create overlay layer
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                # Apply each text element
                for element_config in self.config.elements:
                    self._apply_text_element(draw, overlay.size, element_config)

                # Composite overlay onto image
                result = Image.alpha_composite(img, overlay)

                # Convert back to RGB for JPEG
                result = result.convert("RGB")

                # Save
                output = output_path or image_path
                result.save(output, "JPEG", quality=95, optimize=True)

                logger.info("Text overlay applied: %s", output)
                return output

        except Exception as exc:
            logger.error("Failed to apply text overlay: %s", exc)
            return None

    def _apply_text_element(
        self,
        draw: ImageDraw.ImageDraw,
        image_size: Tuple[int, int],
        config: TextElementConfig,
    ) -> None:
        """Apply a single text element."""
        # Load font
        font = self._load_font(
            config.font_path or self.config.default_font_path,
            config.font_size or self.config.default_font_size,
        )

        if not font:
            logger.warning("Failed to load font, skipping text element")
            return

        # Get text bounding box for sizing
        bbox = draw.textbbox((0, 0), config.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position
        x, y = self._calculate_position(
            image_size,
            text_width,
            text_height,
            config.position,
            config.margin_x,
            config.margin_y,
        )

        # Draw background if specified
        bg_color = config.background_color or self.config.default_background_color
        if bg_color:
            padding = (
                config.background_padding
                or self.config.default_background_padding
            )
            corner_radius = (
                config.background_corner_radius
                or self.config.default_background_corner_radius
            )

            bg_x1 = x - padding
            bg_y1 = y - padding
            bg_x2 = x + text_width + padding
            bg_y2 = y + text_height + padding

            draw.rounded_rectangle(
                [bg_x1, bg_y1, bg_x2, bg_y2], radius=corner_radius, fill=bg_color
            )

        # Draw shadow (if enabled)
        if self.config.default_shadow and config.shadow_offset > 0:
            draw.text(
                (x + config.shadow_offset, y + config.shadow_offset),
                config.text,
                font=font,
                fill=config.shadow_color,
            )

        # Draw main text
        draw.text(
            (x, y),
            config.text,
            font=font,
            fill=config.color,
        )

    def _calculate_position(
        self,
        image_size: Tuple[int, int],
        text_width: int,
        text_height: int,
        position: Union[str, TextPosition],
        margin_x: int,
        margin_y: int,
    ) -> Tuple[int, int]:
        """Calculate (x, y) position based on anchor."""
        img_width, img_height = image_size

        # Convert string to enum if needed
        if isinstance(position, str):
            position = TextPosition(position)

        if position == TextPosition.TOP_LEFT:
            return (margin_x, margin_y)
        if position == TextPosition.TOP_RIGHT:
            return (img_width - text_width - margin_x, margin_y)
        if position == TextPosition.BOTTOM_LEFT:
            return (margin_x, img_height - text_height - margin_y)
        if position == TextPosition.BOTTOM_RIGHT:
            return (
                img_width - text_width - margin_x,
                img_height - text_height - margin_y,
            )
        if position == TextPosition.CENTER:
            return ((img_width - text_width) // 2, (img_height - text_height) // 2)

        logger.warning("Unknown position: %s, using bottom-right", position)
        return self._calculate_position(
            image_size, text_width, text_height, TextPosition.BOTTOM_RIGHT, margin_x, margin_y
        )

    def _load_font(
        self, font_path: Optional[str], size: int
    ) -> Optional[ImageFont.FreeTypeFont]:
        """Load font with fallback chain."""
        cache_key = (font_path, size)

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font = None

        # Try specified font first
        if font_path:
            try:
                font = ImageFont.truetype(font_path, size)
                self._font_cache[cache_key] = font
                return font
            except (OSError, IOError):
                logger.warning("Failed to load font from %s", font_path)

        # Try fallback fonts
        for fallback_path in self.FONT_FALLBACKS:
            try:
                font = ImageFont.truetype(fallback_path, size)
                self._font_cache[cache_key] = font
                return font
            except (OSError, IOError):
                continue

        # Use default font as last resort
        try:
            font = ImageFont.load_default()
            self._font_cache[cache_key] = font
            return font
        except Exception:
            logger.error("Failed to load any font")
            return None
