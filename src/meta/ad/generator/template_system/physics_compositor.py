"""
Physics-Aware Compositor: Dual-Layer Shadows, Light Matching, Light Wrap.

Implements photorealistic compositing with:
1. Dual-Layer Shadow Stack:
   - Contact Shadow: Hard black ellipse (AO occlusion)
   - Cast Shadow: Affine skew based on shadow_direction
2. Light Matching: Soft light blend mode for ambient light integration
3. Light Wrap: Screen blend mode edge glow for environmental light simulation

Goal: 0 deformation, 100% photorealistic fusion

Author: Ad System
Date: 2026-01-30
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageChops


logger = logging.getLogger(__name__)


class ShadowDirection(str, Enum):
    """Shadow direction for affine skew."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class CompositingConfig:
    """
    Physics-based compositing configuration.

    Attributes:
        shadow_direction: Direction for cast shadow affine skew
        light_wrap_intensity: Light wrap strength (0.0 - 1.0)
        light_match_mode: Blend mode for light matching (soft_light)
        light_match_opacity: Opacity for light matching (0.0 - 1.0)
        contact_shadow_size: Size of contact shadow ellipse (pixels)
        contact_shadow_opacity: Opacity of contact shadow (0.0 - 1.0)
        cast_shadow_length: Length of cast shadow (pixels)
        cast_shadow_opacity: Opacity of cast shadow (0.0 - 1.0)
    """
    shadow_direction: ShadowDirection = ShadowDirection.LEFT
    light_wrap_intensity: float = 0.3
    light_match_mode: str = "soft_light"
    light_match_opacity: float = 0.25
    contact_shadow_size: int = 30
    contact_shadow_opacity: float = 0.4
    cast_shadow_length: int = 60
    cast_shadow_opacity: float = 0.2

    def __post_init__(self):
        """Convert string shadow_direction to ShadowDirection enum if needed."""
        if isinstance(self.shadow_direction, str):
            self.shadow_direction = ShadowDirection(self.shadow_direction)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompositingConfig":
        """Create from dictionary (YAML deserialization)."""
        # Handle nested "compositing" key
        if "compositing" in data:
            comp_data = data["compositing"]
        else:
            comp_data = data

        return cls(
            shadow_direction=comp_data.get("shadow_direction", "left"),
            light_wrap_intensity=comp_data.get("light_wrap_intensity", 0.3),
            light_match_mode=comp_data.get("light_match_mode", "soft_light"),
            light_match_opacity=comp_data.get("light_match_opacity", 0.25),
        )


@dataclass
class CompositingResult:
    """
    Result from physics-based compositing.

    Attributes:
        composited_image: Final composited image
        shadow_layer: Shadow layer (for debugging)
        light_wrap_layer: Light wrap layer (for debugging)
        metadata: Compositing metadata
    """
    composited_image: Image.Image
    shadow_layer: Optional[Image.Image] = None
    light_wrap_layer: Optional[Image.Image] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PhysicsCompositor:
    """
    Physics-aware compositing for photorealistic product integration.

    Usage:
        compositor = PhysicsCompositor()
        result = compositor.composite(
            product_image=product,
            background_image=background,
            product_mask=mask,
            config=compositing_config
        )
        result.composited_image.save("output.png")
    """

    def __init__(self, config: Optional[CompositingConfig] = None):
        """
        Initialize physics compositor.

        Args:
            config: Optional compositing configuration
        """
        self.config = config or CompositingConfig()

    def composite(
        self,
        product_image: Image.Image,
        background_image: Image.Image,
        product_mask: Optional[Image.Image] = None,
        product_position: Tuple[int, int] = (0, 0),
        config: Optional[CompositingConfig] = None,
    ) -> CompositingResult:
        """
        Composite product onto background with physics-aware rendering.

        Args:
            product_image: Product image (RGBA)
            background_image: Background image (RGB or RGBA)
            product_mask: Optional product mask (binary, same size as product)
            product_position: (x, y) position to place product on background
            config: Optional compositing config (uses instance config if None)

        Returns:
            CompositingResult with composited image
        """
        config = config or self.config

        # Ensure images are in correct format
        if product_image.mode != "RGBA":
            product_image = product_image.convert("RGBA")
        if background_image.mode != "RGB":
            background_image = background_image.convert("RGB")

        # Create mask if not provided
        if product_mask is None:
            product_mask = self._create_mask_from_alpha(product_image)

        logger.info(
            f"Compositing {product_image.size} product onto "
            f"{background_image.size} background at {product_position}"
        )

        # Create composite canvas
        composite = background_image.copy()

        # Step 1: Generate dual-layer shadows
        shadow_layer = self._generate_shadow_stack(
            product_image=product_image,
            product_mask=product_mask,
            background_image=background_image,
            position=product_position,
            config=config,
        )

        # Step 2: Apply light matching
        product_with_light_match = self._apply_light_matching(
            product_image=product_image,
            background_image=background_image,
            position=product_position,
            config=config,
        )

        # Step 3: Apply light wrap
        product_with_light_wrap = self._apply_light_wrap(
            product_image=product_with_light_match,
            background_image=background_image,
            product_mask=product_mask,
            position=product_position,
            config=config,
        )

        # Step 4: Composite all layers
        # First, add shadows
        composite = self._composite_layer(composite, shadow_layer, product_position)

        # Then, add product with light effects
        composite = self._composite_layer(composite, product_with_light_wrap, product_position)

        logger.info("Compositing complete")

        return CompositingResult(
            composited_image=composite,
            shadow_layer=shadow_layer,
            light_wrap_layer=product_with_light_wrap,
            metadata={
                "config": {
                    "shadow_direction": config.shadow_direction.value,
                    "light_wrap_intensity": config.light_wrap_intensity,
                    "light_match_opacity": config.light_match_opacity,
                }
            }
        )

    def _create_mask_from_alpha(self, image: Image.Image) -> Image.Image:
        """Create binary mask from image alpha channel."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        alpha = image.split()[-1]
        # Threshold at 10
        mask = alpha.point(lambda x: 255 if x > 10 else 0)
        return mask.convert("L")

    def _generate_shadow_stack(
        self,
        product_image: Image.Image,
        product_mask: Image.Image,
        background_image: Image.Image,
        position: Tuple[int, int],
        config: CompositingConfig,
    ) -> Image.Image:
        """
        Generate dual-layer shadow stack.

        Layer 1: Contact Shadow (hard black ellipse, AO occlusion)
        Layer 2: Cast Shadow (affine skew stretched ellipse)
        """
        # Get product bounding box
        bbox = product_mask.getbbox()
        if bbox is None:
            logger.warning("No valid mask bbox, skipping shadow generation")
            return Image.new("RGBA", background_image.size)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Create shadow layer (transparent canvas)
        shadow_layer = Image.new("RGBA", background_image.size, (0, 0, 0, 0))

        # Contact shadow: Hard ellipse at base of product
        contact_shadow = self._create_contact_shadow(
            width=width,
            config=config,
        )

        # Cast shadow: Stretched ellipse with affine skew
        cast_shadow = self._create_cast_shadow(
            width=width,
            config=config,
        )

        # Position shadows
        x, y = position

        # Contact shadow position (at product base)
        contact_x = x + width // 2 - contact_shadow.width // 2
        contact_y = y + height - 10

        # Cast shadow position (offset in direction)
        cast_x, cast_y = self._calculate_cast_shadow_position(
            x, y, width, height, config.shadow_direction
        )

        # Blend shadows onto layer
        shadow_layer.paste(contact_shadow, (contact_x, contact_y), contact_shadow)
        shadow_layer.paste(cast_shadow, (cast_x, cast_y), cast_shadow)

        logger.debug("Generated dual-layer shadow stack")

        return shadow_layer

    def _create_contact_shadow(
        self,
        width: int,
        config: CompositingConfig,
    ) -> Image.Image:
        """Create hard contact shadow ellipse."""
        size = config.contact_shadow_size
        shadow = Image.new("RGBA", (width + size, size // 2), (0, 0, 0, 0))

        draw = ImageDraw.Draw(shadow)
        bbox = [
            size // 4,  # left
            0,         # top
            width + size * 3 // 4,  # right
            size // 2,  # bottom
        ]

        # Draw ellipse with alpha
        alpha = int(255 * config.contact_shadow_opacity)
        draw.ellipse(bbox, fill=(0, 0, 0, alpha))

        # Blur slightly
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=2))

        return shadow

    def _create_cast_shadow(
        self,
        width: int,
        config: CompositingConfig,
    ) -> Image.Image:
        """Create stretched cast shadow with affine skew."""
        length = config.cast_shadow_length

        # Create stretched ellipse
        shadow = Image.new("RGBA", (width + length, length // 3), (0, 0, 0, 0))

        draw = ImageDraw.Draw(shadow)
        bbox = [0, 0, width + length, length // 3]
        alpha = int(255 * config.cast_shadow_opacity)
        draw.ellipse(bbox, fill=(0, 0, 0, alpha))

        # Apply affine skew based on direction
        shadow = self._apply_affine_skew(shadow, config.shadow_direction)

        # Blur for realism
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))

        return shadow

    def _apply_affine_skew(
        self,
        image: Image.Image,
        direction: ShadowDirection,
    ) -> Image.Image:
        """Apply affine skew transformation for shadow perspective."""
        # This is a simplified implementation
        # For production, use proper affine transform with OpenCV

        # For now, just resize to create stretch effect
        width, height = image.size

        if direction in [ShadowDirection.LEFT, ShadowDirection.RIGHT]:
            # Stretch horizontally
            new_width = int(width * 1.3)
            image = image.resize((new_width, height), Image.Resampling.LANCZOS)
        elif direction in [ShadowDirection.TOP, ShadowDirection.BOTTOM]:
            # Stretch vertically
            new_height = int(height * 1.3)
            image = image.resize((width, new_height), Image.Resampling.LANCZOS)

        return image

    def _calculate_cast_shadow_position(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        direction: ShadowDirection,
    ) -> Tuple[int, int]:
        """Calculate cast shadow position based on direction."""
        offset = 20

        if direction == ShadowDirection.LEFT:
            return x - offset, y + height - 20
        elif direction == ShadowDirection.RIGHT:
            return x + width + offset - 60, y + height - 20
        elif direction == ShadowDirection.TOP:
            return x + width // 2 - 30, y - offset
        else:  # BOTTOM
            return x + width // 2 - 30, y + height + offset

    def _apply_light_matching(
        self,
        product_image: Image.Image,
        background_image: Image.Image,
        position: Tuple[int, int],
        config: CompositingConfig,
    ) -> Image.Image:
        """
        Apply light matching using soft light blend mode.

        Calculates background average color and applies it to product
        at specified opacity using soft light blend mode.
        """
        # Get region of background under product
        x, y = position
        product_region = product_image.size
        bg_crop = background_image.crop((
            x, y,
            x + product_region[0],
            y + product_region[1],
        ))

        # Calculate average background color
        bg_array = np.array(bg_crop)
        avg_color = bg_array.mean(axis=(0, 1)).astype(int)

        # Create color overlay
        overlay = Image.new("RGB", product_region, tuple(avg_color))

        # Apply soft light blend mode
        # For soft light: if top < 0.5, 2*top*bottom; else 1-2*(1-top)*(1-bottom)
        product_rgb = product_image.convert("RGB")

        # Simple approximation: blend with overlay
        blended = Image.blend(
            product_rgb,
            overlay,
            alpha=config.light_match_opacity,
        )

        # Preserve alpha channel
        result = Image.composite(
            product_image,
            blended.convert("RGBA"),
            product_image.split()[-1]
        )

        logger.debug(f"Applied light matching: opacity={config.light_match_opacity}")

        return result

    def _apply_light_wrap(
        self,
        product_image: Image.Image,
        background_image: Image.Image,
        product_mask: Image.Image,
        position: Tuple[int, int],
        config: CompositingConfig,
    ) -> Image.Image:
        """
        Apply light wrap effect (edge glow) using screen blend mode.

        Simulates environmental light wrapping around product edges
        to eliminate cutout white edges.
        """
        # Get background region
        x, y = position
        product_size = product_image.size
        bg_crop = background_image.crop((
            x, y,
            x + product_size[0],
            y + product_size[1],
        ))

        # Create edge mask (eroded 2px from edge)
        edge_mask = self._create_edge_mask(product_mask, erosion=2)

        # Blur background
        bg_blur = bg_crop.filter(ImageFilter.GaussianBlur(radius=8))

        # Apply screen blend mode
        # Screen: 1 - (1-top)*(1-bottom) = top + bottom - top*bottom
        product_rgb = product_image.convert("RGB")
        bg_array = np.array(bg_blur)
        product_array = np.array(product_rgb)

        # Screen blend
        screen_array = (
            product_array.astype(float) +
            bg_array.astype(float) -
            (product_array.astype(float) * bg_array.astype(float) / 255)
        ).clip(0, 255).astype(np.uint8)

        screen_layer = Image.fromarray(screen_array, mode="RGB")

        # Composite only on edges using edge mask
        result = Image.composite(
            screen_layer.convert("RGBA"),
            product_image,
            edge_mask,
        )

        # Apply intensity
        if config.light_wrap_intensity < 1.0:
            result_with_product = Image.blend(
                product_image,
                result,
                alpha=config.light_wrap_intensity,
            )
            result = result_with_product

        logger.debug(f"Applied light wrap: intensity={config.light_wrap_intensity}")

        return result

    def _create_edge_mask(
        self,
        mask: Image.Image,
        erosion: int = 2,
    ) -> Image.Image:
        """Create edge mask by eroding product mask."""
        # Invert mask
        inverted = ImageChops.invert(mask)

        # Erode inverted mask (this expands the white area)
        eroded = inverted.filter(ImageFilter.MinFilter(size=erosion * 2 + 1))

        # Invert back to get edges
        edge_mask = ImageChops.invert(eroded)

        return edge_mask

    def _composite_layer(
        self,
        base: Image.Image,
        layer: Image.Image,
        position: Tuple[int, int],
    ) -> Image.Image:
        """Composite a layer onto base at position."""
        if layer.mode != "RGBA":
            layer = layer.convert("RGBA")

        base_rgba = base.convert("RGBA")

        # Paste layer onto base using alpha channel
        base_rgba.paste(layer, position, layer)

        return base_rgba.convert("RGB")


# Convenience function
def composite_physics_aware(
    product_image: Image.Image,
    background_image: Image.Image,
    product_mask: Optional[Image.Image] = None,
    position: Tuple[int, int] = (0, 0),
    shadow_direction: str = "left",
    light_wrap_intensity: float = 0.3,
) -> Image.Image:
    """
    Convenience function for physics-aware compositing.

    Args:
        product_image: Product image (RGBA)
        background_image: Background image (RGB)
        product_mask: Optional product mask
        position: (x, y) position for product
        shadow_direction: Shadow direction (left, right, top, bottom)
        light_wrap_intensity: Light wrap strength (0.0 - 1.0)

    Returns:
        Composited image

    Example:
        result = composite_physics_aware(
            product_image=product,
            background_image=background,
            shadow_direction="left"
        )
        result.save("composited.png")
    """
    config = CompositingConfig(
        shadow_direction=ShadowDirection(shadow_direction),
        light_wrap_intensity=light_wrap_intensity,
    )

    compositor = PhysicsCompositor(config)
    result = compositor.composite(
        product_image=product_image,
        background_image=background_image,
        product_mask=product_mask,
        product_position=position,
    )

    return result.composited_image
