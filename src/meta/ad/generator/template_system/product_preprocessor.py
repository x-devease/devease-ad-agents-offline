"""
Product Preprocessor: Trim Transparency, Perspective Detection, Mask Generation.

This module handles the initial processing of product images:
1. Trim Transparency: Crop to bounding box, remove empty space
2. Perspective Analyzer: Detect camera angle (high_angle vs eye_level)
3. Product Extraction: Create mask layer for collision detection

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
from PIL import Image, ImageChops


logger = logging.getLogger(__name__)


class PerspectiveType(str, Enum):
    """Camera perspective types for background matching."""

    HIGH_ANGLE = "high_angle"  # 俯视 - looking down at product
    EYE_LEVEL = "eye_level"    # 平视 - level with product


@dataclass
class PreprocessorResult:
    """
    Result from product preprocessing.

    Attributes:
        trimmed_image: Product image cropped to bounding box
        mask: Binary mask for collision detection
        perspective: Detected camera perspective
        bbox: Bounding box (x, y, width, height)
        original_size: Original image size (width, height)
    """
    trimmed_image: Image.Image
    mask: Image.Image
    perspective: PerspectiveType
    bbox: Tuple[int, int, int, int]
    original_size: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "perspective": self.perspective.value,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
            "original_size": {
                "width": self.original_size[0],
                "height": self.original_size[1],
            },
            "trimmed_size": {
                "width": self.trimmed_image.width,
                "height": self.trimmed_image.height,
            }
        }


class ProductPreprocessor:
    """
    Product image preprocessing for Template-Driven Ad Generator.

    Handles transparency trimming, perspective detection, and mask generation.

    Usage:
        preprocessor = ProductPreprocessor()
        result = preprocessor.process(product_image_path)
        print(f"Detected perspective: {result.perspective}")
    """

    def __init__(
        self,
        padding: int = 10,
        min_alpha_threshold: int = 10,
    ):
        """
        Initialize preprocessor.

        Args:
            padding: Padding around trimmed product (pixels)
            min_alpha_threshold: Minimum alpha value to consider transparent (0-255)
        """
        self.padding = padding
        self.min_alpha_threshold = min_alpha_threshold

    def process(
        self,
        image_path: str | Path,
    ) -> PreprocessorResult:
        """
        Process product image: trim, detect perspective, generate mask.

        Args:
            image_path: Path to product image (PNG with alpha channel)

        Returns:
            PreprocessorResult with trimmed image, mask, and perspective

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image doesn't have alpha channel
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Product image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert("RGBA")
        logger.info(f"Loaded product image: {image.size[0]}x{image.size[1]}")

        original_size = image.size

        # Step 1: Trim transparency
        trimmed, bbox = self._trim_transparency(image)
        logger.info(f"Trimmed to bounding box: {bbox}")

        # Step 2: Detect perspective
        perspective = self._detect_perspective(trimmed)
        logger.info(f"Detected perspective: {perspective.value}")

        # Step 3: Generate mask
        mask = self._generate_mask(trimmed)
        logger.info(f"Generated mask: {mask.size}")

        return PreprocessorResult(
            trimmed_image=trimmed,
            mask=mask,
            perspective=perspective,
            bbox=bbox,
            original_size=original_size,
        )

    def _trim_transparency(
        self,
        image: Image.Image,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Crop image to content bounding box, removing transparent edges.

        Args:
            image: PIL Image with alpha channel

        Returns:
            Tuple of (trimmed_image, bbox) where bbox is (x, y, w, h)
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Get alpha channel
        alpha = np.array(image.split()[-1])

        # Find non-transparent pixels
        non_transparent = alpha > self.min_alpha_threshold

        if not np.any(non_transparent):
            # Image is fully transparent, return as-is
            logger.warning("Image is fully transparent, returning original")
            return image, (0, 0, image.width, image.height)

        # Get bounding box of non-transparent region
        rows = np.any(non_transparent, axis=1)
        cols = np.any(non_transparent, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Add padding
        x_min = max(0, x_min - self.padding)
        y_min = max(0, y_min - self.padding)
        x_max = min(image.width, x_max + self.padding + 1)
        y_max = min(image.height, y_max + self.padding + 1)

        # Crop image
        bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        trimmed = image.crop(bbox)

        logger.debug(f"Trimmed from {image.size} to {trimmed.size}")

        return trimmed, bbox

    def _detect_perspective(
        self,
        image: Image.Image,
    ) -> PerspectiveType:
        """
        Detect camera perspective based on product shape analysis.

        Strategy:
        - HIGH_ANGLE: Product appears wider than tall (looking down)
        - EYE_LEVEL: Product height ≈ width (level view)

        Args:
            image: Trimmed product image

        Returns:
            PerspectiveType (HIGH_ANGLE or EYE_LEVEL)
        """
        # Get aspect ratio
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0

        # Simple heuristic based on aspect ratio
        # This can be enhanced with more sophisticated analysis (vanishing points, etc.)
        if aspect_ratio > 1.3:
            # Wider than tall → likely top-down view
            perspective = PerspectiveType.HIGH_ANGLE
        elif aspect_ratio < 0.7:
            # Taller than wide → likely eye-level with vertical orientation
            perspective = PerspectiveType.EYE_LEVEL
        else:
            # Roughly square → assume eye-level (most common for product shots)
            perspective = PerspectiveType.EYE_LEVEL

        logger.debug(f"Aspect ratio: {aspect_ratio:.2f} → {perspective.value}")

        return perspective

    def _generate_mask(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Generate binary mask for collision detection.

        Mask is white (255) where product exists, black (0) where transparent.

        Args:
            image: Product image with alpha channel

        Returns:
            Grayscale mask image (L mode)
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Extract alpha channel
        alpha = image.split()[-1]

        # Convert to binary mask (0 or 255)
        # Pixels with alpha > threshold become white (255)
        mask = alpha.point(lambda x: 255 if x > self.min_alpha_threshold else 0)

        logger.debug(f"Generated mask: {mask.size}, mode={mask.mode}")

        return mask


def preprocess_product(
    image_path: str | Path,
    padding: int = 10,
) -> PreprocessorResult:
    """
    Convenience function for quick product preprocessing.

    Args:
        image_path: Path to product image
        padding: Padding around trimmed product

    Returns:
        PreprocessorResult

    Example:
        result = preprocess_product("product.png")
        result.trimmed_image.save("product_trimmed.png")
        result.mask.save("product_mask.png")
        print(f"Perspective: {result.perspective.value}")
    """
    preprocessor = ProductPreprocessor(padding=padding)
    return preprocessor.process(image_path)


# Main execution for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python product_preprocessor.py <product_image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    result = preprocess_product(image_path)

    # Save outputs
    output_dir = image_path.parent / "preprocessed"
    output_dir.mkdir(exist_ok=True)

    result.trimmed_image.save(output_dir / f"{image_path.stem}_trimmed.png")
    result.mask.save(output_dir / f"{image_path.stem}_mask.png")

    # Print info
    print(f"\nPreprocessing Results:")
    print(f"  Original size: {result.original_size[0]}x{result.original_size[1]}")
    print(f"  Trimmed size: {result.trimed_image.width}x{result.trimmed_image.height}")
    print(f"  Bounding box: x={result.bbox[0]}, y={result.bbox[1]}, w={result.bbox[2]}, h={result.bbox[3]}")
    print(f"  Perspective: {result.perspective.value}")

    print(f"\nOutputs saved to: {output_dir}")
