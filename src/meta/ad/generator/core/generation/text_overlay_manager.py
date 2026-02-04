"""
Text Overlay Manager - Integrates text extraction and overlay.

Automatically extracts product information from prompts and overlays
text onto generated images using smart positioning and collision detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .text_extractor import TextExtractor, ExtractedText
from .text_overlay import TextOverlay, TextPosition


logger = logging.getLogger(__name__)


@dataclass
class TextOverlayConfig:
    """Configuration for text overlay."""
    template: str = "minimal"
    font_family: str = "Arial"
    font_size: int = 24
    font_weight: str = "bold"
    text_color: str = "#000000"
    background_color: str = "rgba(255,255,255,0.8)"
    position: str = "top_center"
    max_length: int = 50


class TextOverlayManager:
    """
    Manages automatic text overlay on generated images.

    Integrates TextExtractor with TextOverlay for smart text positioning.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        enabled: bool = True,
    ):
        """
        Initialize text overlay manager.

        Args:
            config_path: Path to text overlay templates config
            enabled: Whether text overlay is enabled
        """
        self.enabled = enabled
        self.config_path = config_path
        self.extractor = TextExtractor()

        # Load text overlay templates
        self.templates = {}
        if config_path and config_path.exists():
            self._load_templates(config_path)

        logger.info(
            f"TextOverlayManager initialized: enabled={enabled}, "
            f"templates_loaded={len(self.templates)}"
        )

    def _load_templates(self, config_path: Path) -> None:
        """Load text overlay templates from config."""
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.templates = config.get('text_overlay', {}).get('templates', {})
        logger.info(f"Loaded {len(self.templates)} text overlay templates")

    def select_template(
        self,
        features: Dict[str, str],
        default: str = "minimal"
    ) -> str:
        """
        Select appropriate template based on features.

        Args:
            features: Feature dict from prompts
            default: Default template if no rules match

        Returns:
            Template name
        """
        # Load template selection rules
        if self.templates:
            template_config = self.templates.get('template_selection', {})
            rules = template_config.get('rules', [])

            # Check each rule
            for rule in rules:
                condition = rule.get('condition', '')
                # Simple condition evaluation (can be enhanced)
                if self._evaluate_condition(condition, features):
                    template = rule.get('template')
                    logger.debug(f"Selected template '{template}' based on rule: {rule.get('reason', '')}")
                    return template

        return default

    def _evaluate_condition(self, condition: str, features: Dict[str, str]) -> bool:
        """Evaluate template selection condition."""
        # Simple key=value check (can be enhanced with proper expression parsing)
        if '=' in condition:
            key, value = condition.split('=', 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            return features.get(key) == value

        return False

    def extract_and_format(
        self,
        prompt: str,
        features: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract text and format for overlay.

        Args:
            prompt: The prompt text
            features: Optional features dict
            metadata: Optional metadata
            template: Optional template name (auto-select if None)

        Returns:
            Dict with formatted text elements ready for overlay
        """
        # Select template if not provided
        if template is None and features:
            template = self.select_template(features)
        elif template is None:
            template = "minimal"

        # Extract text elements
        extracted = self.extractor.extract_from_prompt(
            prompt=prompt,
            features=features,
            metadata=metadata,
        )

        # Format for overlay based on template
        formatted = self.extractor.format_for_overlay(extracted, template)

        logger.info(
            f"Extracted and formatted text: template={template}, "
            f"product={extracted.product_name}, "
            f"features={len(extracted.key_features)}"
        )

        return formatted

    def overlay_text_on_image(
        self,
        image_path: str,
        formatted_text: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Overlay formatted text onto image.

        Args:
            image_path: Path to source image
            formatted_text: Dict with formatted text from extract_and_format
            output_path: Optional output path (defaults to source_path with _overlay suffix)

        Returns:
            Path to image with text overlay
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, cannot overlay text")
            return image_path

        # Load image
        image = Image.open(image_path).convert("RGBA")

        # Create overlay layer
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Load font (use default if custom font not available)
        try:
            font_size = formatted_text.get('font_size', 24)
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()

        # Get text and position
        headline = formatted_text.get('headline', '')
        position_str = formatted_text.get('position', 'top_center')

        # Calculate position
        img_width, img_height = image.size
        x, y = self._calculate_position(position_str, img_width, img_height)

        # Add text color/background
        text_color = self._parse_color(formatted_text.get('text_color', '#000000'))
        bg_color = self._parse_color(formatted_text.get('background_color', 'rgba(255,255,255,0.8)'))

        # Draw background if specified
        if bg_color and bg_color[3] > 0:  # Alpha > 0
            # Get text bounding box
            bbox = draw.textbbox((x, y), headline, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw background rectangle
            padding = 10
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=bg_color
            )

        # Draw text
        draw.text((x, y), headline, fill=text_color, font=font)

        # Composite overlay onto image
        composite = Image.alpha_composite(image, overlay)

        # Save result
        if output_path is None:
            base_path = Path(image_path).stem
            output_path = str(Path(image_path).parent / f"{base_path}_overlay.png")

        composite.convert('RGB').save(output_path)

        logger.info(f"Text overlay saved to: {output_path}")

        return output_path

    def _calculate_position(
        self,
        position_str: str,
        img_width: int,
        img_height: int
    ) -> tuple[int, int]:
        """Calculate (x, y) coordinates from position string."""
        if position_str == "top_center":
            return (img_width // 2, 50)

        elif position_str == "top_third":
            return (img_width // 2, img_height // 6)

        elif position_str == "bottom_left":
            return (50, img_height - 100)

        elif position_str == "bottom_right":
            return (img_width - 150, img_height - 100)

        elif position_str == "center":
            return (img_width // 2, img_height // 2)

        # Default: top center
        return (img_width // 2, 50)

    def _parse_color(self, color_str: str) -> tuple:
        """Parse color string to RGBA tuple."""
        color_str = color_str.strip()

        # Handle hex colors
        if color_str.startswith('#'):
            hex_color = color_str.lstrip('#')
            if len(hex_color) == 6:
                # Add alpha if not present
                hex_color += 'ff'

            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(hex_color[6:8], 16) if len(hex_color) >= 8 else 255

            return (r, g, b, a)

        # Handle rgba() format
        elif color_str.startswith('rgba'):
            import re
            match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)', color_str)
            if match:
                return (
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(float(match.group(4)) * 255)
                )

        # Default black
        return (0, 0, 0, 255)

    def process_generation_result(
        self,
        generation_result: Dict[str, Any],
        prompt: str,
        features: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process generation result and add text overlay.

        Args:
            generation_result: Result from generator.generate()
            prompt: Original prompt
            features: Optional features dict
            metadata: Optional metadata

        Returns:
            Updated result with text_overlay_path
        """
        if not self.enabled:
            return generation_result

        if not generation_result.get('success'):
            return generation_result

        # Extract and format text
        formatted_text = self.extract_and_format(
            prompt=prompt,
            features=features,
            metadata=metadata,
        )

        # Overlay text on generated image
        image_path = generation_result.get('image_path')
        if image_path:
            overlay_path = self.overlay_text_on_image(
                image_path=image_path,
                formatted_text=formatted_text,
            )

            generation_result['text_overlay_path'] = overlay_path
            generation_result['text_overlay_applied'] = True

        return generation_result
