"""
Premium watermark module for image generation.
Provides adaptive watermarking with modern design and logo integration.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .constants import (
    WATERMARK_MARGIN_PCT,
    WATERMARK_OPACITY,
    WATERMARK_SIZE_PCT,
)


logger = logging.getLogger(__name__)


class PremiumWatermark:
    """
    Premium watermark with modern solid design and DevEase logo.
    Features separate logo placement (top-right) and text watermark
    (bottom-right).
    """

    def __init__(self):
        """Initialize watermark with default settings."""
        # Size constraints for text watermark (in pixels)
        self.min_width = 250
        self.max_width = 600
        self.min_height = 50
        self.max_height = 100
        # Logo size constraints
        self.logo_min_size = 40
        self.logo_max_size = 80
        # Percentage of image size (from constants)
        self.width_percent = (
            WATERMARK_SIZE_PCT  # % of image width for text watermark
        )
        self.height_percent = WATERMARK_OPACITY  # opacity as height ratio
        # Margin from edges (percentage)
        self.margin_percent = WATERMARK_MARGIN_PCT  # margin
        # Text content
        self.line1_text = "FOR PREVIEW ONLY"
        self.line2_text = "Powered by DevEase"
        # Logo path - try to find logo in media directory
        logo_paths = [
            Path(__file__).parent.parent.parent / "media" / "devease_logo.png",
            Path(__file__).parent.parent / "media" / "devease_logo.png",
        ]
        self.logo_path = None
        for path in logo_paths:
            if path.exists():
                self.logo_path = path
                break
        self._logo_cache = None

    def _load_logo(self) -> Optional[Image.Image]:
        """Load and cache the DevEase logo."""
        if (
            self._logo_cache is None
            and self.logo_path
            and self.logo_path.exists()
        ):
            try:
                with Image.open(self.logo_path) as logo:
                    if logo.mode != "RGBA":
                        logo = logo.convert("RGBA")
                    # Copy image data to keep after context closes
                    self._logo_cache = logo.copy()
                logger.debug("Logo loaded successfully from %s", self.logo_path)
            except (OSError, IOError, ValueError) as exc:
                logger.warning("Failed to load logo: %s", exc)
                self._logo_cache = False  # Mark as failed
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Unexpected error loading logo: %s", exc)
                self._logo_cache = False  # Mark as failed

        return self._logo_cache if self._logo_cache else None

    def calculate_dimensions(
        self, img_width: int, img_height: int
    ) -> Tuple[int, int, int, int, dict]:
        """
        Calculate adaptive watermark dimensions based on image size.

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (watermark_width, watermark_height, x_position,
            y_position, font_sizes)
        """
        # Calculate base dimensions as percentage of image
        wm_width = img_width * self.width_percent
        wm_height = img_height * self.height_percent
        # Apply size constraints
        wm_width = max(self.min_width, min(self.max_width, wm_width))
        wm_height = max(self.min_height, min(self.max_height, wm_height))
        # Adjust for very small images (< 800px width)
        if img_width < 800:
            wm_width = min(img_width * 0.35, self.max_width)
            wm_height = min(img_height * 0.08, self.max_height)
        # Adjust for very large images (> 3000px width)
        elif img_width > 3000:
            wm_width = min(img_width * 0.22, self.max_width)
            wm_height = min(img_height * 0.055, self.max_height)
        # Convert to integers
        wm_width = int(wm_width)
        wm_height = int(wm_height)
        # Calculate position (bottom-right with margin)
        margin = int(min(img_width, img_height) * self.margin_percent)
        x_pos = img_width - wm_width - margin
        y_pos = img_height - wm_height - margin
        # Calculate font sizes based on watermark height
        font_sizes = {
            "line1": max(10, int(wm_height * 0.24)),  # FOR PREVIEW ONLY
            "line2": max(12, int(wm_height * 0.32)),  # Powered by DevEase
        }

        return wm_width, wm_height, x_pos, y_pos, font_sizes

    def calculate_logo_dimensions(
        self, img_width: int, img_height: int
    ) -> Tuple[int, int, int]:
        """
        Calculate logo dimensions and position.

        Args:
            img_width: Image width
            img_height: Image height

        Returns:
            Tuple of (logo_size, x_position, y_position)
        """
        # Calculate logo size based on image dimensions
        base_size = min(img_width, img_height) * 0.04
        logo_size = max(self.logo_min_size, min(self.logo_max_size, base_size))
        # Adjust for very small or large images
        if img_width < 800:
            logo_size = min(img_width * 0.06, self.logo_max_size)
        elif img_width > 3000:
            logo_size = min(img_width * 0.03, self.logo_max_size)

        logo_size = int(logo_size)
        # Position in top-right corner
        margin = int(min(img_width, img_height) * self.margin_percent)
        x_pos = img_width - logo_size - margin
        y_pos = margin

        return logo_size, x_pos, y_pos

    def create_text_watermark(
        self,
        width: int,
        height: int,
        wm_width: int,
        wm_height: int,
        wm_x: int,
        wm_y: int,
        font_sizes: dict,
    ) -> Image.Image:
        """Create the text watermark with solid modern design."""
        # Create transparent overlay
        watermark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        # Corner radius
        corner_radius = int(wm_height * 0.18)
        # Create solid background with gradient effect
        # Dark background with good contrast
        main_color = (15, 15, 20, 230)  # Almost black with high opacity
        # Draw main background
        main_bounds = [wm_x, wm_y, wm_x + wm_width, wm_y + wm_height]
        draw.rounded_rectangle(
            main_bounds, radius=corner_radius, fill=main_color
        )
        # Add subtle border for definition
        draw.rounded_rectangle(
            main_bounds,
            radius=corner_radius,
            outline=(255, 255, 255, 40),
            width=1,
        )
        # Load fonts
        fonts = self._load_fonts(font_sizes)
        # Calculate text positions with equal padding
        text_x = wm_x + wm_width // 2
        # Visually equal top and bottom padding:
        # 24% from top and 18% from bottom
        text1_y = wm_y + int(wm_height * 0.24)
        text2_y = wm_y + int(wm_height * 0.58)
        # Draw text with subtle shadow
        # Line 1: FOR PREVIEW ONLY
        # Shadow
        draw.text(
            (text_x + 1, text1_y + 1),
            self.line1_text,
            font=fonts["line1"],
            fill=(0, 0, 0, 100),
            anchor="mt",
        )
        # Main text
        draw.text(
            (text_x, text1_y),
            self.line1_text,
            font=fonts["line1"],
            fill=(255, 255, 255, 200),
            anchor="mt",
        )
        # Line 2: Powered by DevEase
        # Shadow
        draw.text(
            (text_x + 1, text2_y + 1),
            self.line2_text,
            font=fonts["line2"],
            fill=(0, 0, 0, 100),
            anchor="mt",
        )
        # Main text
        draw.text(
            (text_x, text2_y),
            self.line2_text,
            font=fonts["line2"],
            fill=(255, 255, 255, 255),
            anchor="mt",
        )

        return watermark

    def create_logo_watermark(
        self, width: int, height: int, logo_size: int, logo_x: int, logo_y: int
    ) -> Optional[Image.Image]:
        """Create the logo watermark for top-right corner."""
        logo = self._load_logo()
        if not logo:
            return None
        # Create transparent overlay
        watermark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # Extract just the D icon (square portion from the left)
        logo_cropped = logo.crop(
            (0, 0, min(logo.height, logo.width), logo.height)
        )
        # Resize and paste logo
        logo_resized = logo_cropped.resize(
            (logo_size, logo_size), Image.Resampling.LANCZOS
        )
        watermark.paste(logo_resized, (logo_x, logo_y), logo_resized)

        return watermark

    def create_watermark_overlay(self, width: int, height: int) -> Image.Image:
        """
        Create complete watermark overlay with logo and text.

        Args:
            width: Image width
            height: Image height

        Returns:
            RGBA image with watermark
        """
        # Create base transparent overlay
        final_watermark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # Calculate text watermark dimensions and create it
        wm_width, wm_height, wm_x, wm_y, font_sizes = self.calculate_dimensions(
            width, height
        )
        text_watermark = self.create_text_watermark(
            width, height, wm_width, wm_height, wm_x, wm_y, font_sizes
        )
        # Composite text watermark
        final_watermark = Image.alpha_composite(final_watermark, text_watermark)
        # Calculate logo dimensions and create it
        logo_size, logo_x, logo_y = self.calculate_logo_dimensions(
            width, height
        )
        logo_watermark = self.create_logo_watermark(
            width, height, logo_size, logo_x, logo_y
        )
        # Composite logo watermark if available
        if logo_watermark:
            final_watermark = Image.alpha_composite(
                final_watermark, logo_watermark
            )

        return final_watermark

    def _load_fonts(self, font_sizes: dict) -> dict:
        """
        Load fonts with fallback to default.

        Args:
            font_sizes: Dictionary of font sizes

        Returns:
            Dictionary of loaded fonts
        """
        fonts = {}
        # Try to load modern, clean fonts
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Avenir Next.ttc",
            "/System/Library/Fonts/Avenir.ttc",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/liberation/"
            "LiberationSans-Regular.ttf",  # Linux
        ]
        # Try bold variants for line2
        bold_font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Avenir Next.ttc",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/liberation/"
            "LiberationSans-Bold.ttf",  # Linux
        ]
        # Load regular font for line1
        font_loaded = False
        for font_path in font_paths:
            try:
                fonts["line1"] = ImageFont.truetype(
                    font_path, font_sizes["line1"]
                )
                font_loaded = True
                break
            except (OSError, IOError):
                continue

        if not font_loaded:
            try:
                fonts["line1"] = ImageFont.load_default()
            except (OSError, IOError):
                fonts["line1"] = None
        # Load bold/medium font for line2
        font_loaded = False
        for font_path in bold_font_paths:
            try:
                fonts["line2"] = ImageFont.truetype(
                    font_path, font_sizes["line2"]
                )
                font_loaded = True
                break
            except (OSError, IOError):
                continue

        if not font_loaded:
            # Use regular font as fallback
            fonts["line2"] = fonts.get("line1")

        return fonts

    def apply_watermark(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        quality: int = 95,
    ) -> Optional[str]:
        """
        Apply watermark to an image.

        Args:
            image_path: Path to input image
            output_path: Path to save watermarked image
                (if None, overwrites input)
            quality: JPEG quality (1-100)

        Returns:
            Path to watermarked image or None if failed
        """
        # Validate quality parameter
        if not 1 <= quality <= 100:
            logger.warning(
                "Quality must be between 1 and 100, got %d. Using default 95.",
                quality,
            )
            quality = 95

        try:
            # Open image with context manager for proper cleanup
            with Image.open(image_path) as img:
                width, height = img.size

                logger.debug(
                    "Applying watermark to %s (%dx%d)", image_path, width, height
                )
                # Create watermark overlay
                watermark = self.create_watermark_overlay(width, height)
                # Convert to RGBA if needed
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                # Composite watermark onto image
                watermarked = Image.alpha_composite(img, watermark)
                # Convert back to RGB for JPEG
                watermarked = watermarked.convert("RGB")
                # Determine output path
                if output_path is None:
                    output_path = image_path
                # Save with high quality
                watermarked.save(
                    output_path, "JPEG", quality=quality, optimize=True
                )

            logger.debug("Watermark applied successfully: %s", output_path)
            return output_path

        except (OSError, IOError, ValueError) as exc:
            logger.error("Failed to apply watermark to %s: %s", image_path, exc)
            return None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "Unexpected error applying watermark to %s: %s", image_path, exc
            )
            return None
