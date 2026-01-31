"""
Reference Image Manager: Angle-aware product image selection.

Manages multiple product reference images and selects appropriate images
based on camera angle specifications in generation patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class ReferenceImage:
    """Single reference image with angle metadata."""

    path: Path
    angle_category: str  # "front", "back", "45_left", "45_right", "top", "side", "flat", "rotation"
    filename: str
    priority: int  # Lower = higher priority (0 = primary)


class ReferenceImageManager:
    """
    Manages product reference images with angle-aware selection.

    Maps Chinese filenames to angle categories and provides intelligent
    image selection based on requested camera angles.
    """

    # Filename to angle category mapping
    FILENAME_MAPPINGS: Dict[str, tuple[str, int]] = {
        # Front/Back views
        "正面.png": ("front", 0),
        "背面.png": ("back", 10),
        # 45-degree angles
        "左侧45.png": ("45_left", 1),
        "右侧45.png": ("45_right", 2),
        # Top views (high-angle)
        "俯视.1.png": ("top", 3),
        "俯视.2.png": ("top", 4),
        # Side angles
        "侧仰.png": ("side", 5),
        "侧俯.png": ("side", 6),
        # Rotations (negative angles)
        "-120.png": ("rotation", 7),
        "-150.png": ("rotation", 8),
        "-30.png": ("rotation", 9),
        # Rotations (positive angles)
        "30.png": ("rotation", 10),
        "60.png": ("rotation", 11),
        "90.png": ("rotation", 12),
        "120.png": ("rotation", 13),
        "150.png": ("rotation", 14),
        # Flat view
        "180躺平.png": ("flat", 15),
    }

    # Pattern camera angle to reference image mapping
    # Order matters: first = highest priority
    ANGLE_SELECTION_MAP: Dict[Optional[str], List[str]] = {
        "45-degree": ["45_left", "45_right", "front"],
        "45-degree High-Angle Shot": ["top", "45_left", "front"],
        "Eye-Level Shot": ["front", "45_left", "45_right"],
        "High-Angle Shot": ["top", "side", "front"],
        "Top-Down": ["top", "top", "flat"],
        "Side View": ["side", "45_left", "45_right"],
        "Low-Angle Shot": ["front", "side", "45_left"],
        None: ["front", "45_left", "45_right"],  # Default fallback
    }

    def __init__(
        self,
        reference_images_dir: Path,
        fallback_image_path: Optional[Path] = None,
    ):
        """
        Initialize reference image manager.

        Args:
            reference_images_dir: Directory containing reference images
            fallback_image_path: Fallback image if reference dir doesn't exist
        """
        self.reference_images_dir = Path(reference_images_dir)
        self.fallback_image_path = Path(fallback_image_path) if fallback_image_path else None
        self.reference_images: Dict[str, ReferenceImage] = {}

        self._load_reference_images()

    def _load_reference_images(self) -> None:
        """Scan and categorize reference images from directory."""
        if not self.reference_images_dir.exists():
            logger.warning(
                f"Reference images directory not found: {self.reference_images_dir}"
            )
            return

        for filename in self.reference_images_dir.glob("*.png"):
            if filename.name in self.FILENAME_MAPPINGS:
                angle_category, priority = self.FILENAME_MAPPINGS[filename.name]
                self.reference_images[angle_category] = ReferenceImage(
                    path=filename,
                    angle_category=angle_category,
                    filename=filename.name,
                    priority=priority,
                )

        logger.info(
            f"Loaded {len(self.reference_images)} reference images "
            f"from {self.reference_images_dir}"
        )

        # Log loaded images by category
        if self.reference_images:
            categories_by_priority = sorted(
                [(cat, img.filename, img.priority) for cat, img in self.reference_images.items()],
                key=lambda x: x[2]
            )
            logger.debug(f"Reference images by category: {categories_by_priority}")

    def select_images_for_angle(
        self,
        camera_angle: Optional[str],
        max_images: int = 3,
    ) -> List[Path]:
        """
        Select appropriate reference images for a given camera angle.

        Args:
            camera_angle: Camera angle from pattern (e.g., "45-degree")
            max_images: Maximum number of images to return

        Returns:
            List of image paths (sorted by priority)
        """
        # Get angle category preferences
        angle_prefs = self.ANGLE_SELECTION_MAP.get(
            camera_angle,
            self.ANGLE_SELECTION_MAP[None]  # Default
        )

        selected_images = []
        for category in angle_prefs[:max_images]:
            if category in self.reference_images:
                selected_images.append(self.reference_images[category].path)

        # Fallback: if no images selected, use fallback_image_path
        if not selected_images and self.fallback_image_path:
            logger.warning(
                f"No reference images found for angle '{camera_angle}', "
                f"using fallback: {self.fallback_image_path}"
            )
            return [self.fallback_image_path]

        if selected_images:
            logger.info(
                f"Selected {len(selected_images)} images for angle '{camera_angle}': "
                f"{[p.name for p in selected_images]}"
            )
        else:
            logger.warning(f"No reference images available for angle '{camera_angle}'")

        return selected_images

    def get_all_image_paths(self) -> List[Path]:
        """Get all available reference image paths."""
        return [img.path for img in self.reference_images.values()]

    def has_images(self) -> bool:
        """Check if any reference images are loaded."""
        return len(self.reference_images) > 0

    def get_available_categories(self) -> List[str]:
        """Get list of available angle categories."""
        return list(self.reference_images.keys())
