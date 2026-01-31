"""
Unit tests for ReferenceImageManager.

Tests angle-aware reference image selection including:
- Filename mapping (Chinese names to categories)
- Angle-based selection logic
- Fallback behavior
- Priority ordering
"""

import pytest
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from src.meta.ad.generator.core.generation.reference_image_manager import (
    ReferenceImage,
    ReferenceImageManager,
)


class TestReferenceImage:
    """Test ReferenceImage dataclass."""

    def test_reference_image_creation(self):
        """Test creating a ReferenceImage."""
        path = Path("/tmp/test.png")
        img = ReferenceImage(
            path=path,
            angle_category="front",
            filename="test.png",
            priority=0,
        )

        assert img.path == path
        assert img.angle_category == "front"
        assert img.filename == "test.png"
        assert img.priority == 0


class TestReferenceImageManager:
    """Test ReferenceImageManager class."""

    @pytest.fixture
    def temp_reference_dir(self):
        """Create a temporary directory with test reference images."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        # Create test images with Chinese filenames
        test_images = [
            "正面.png",  # front
            "背面.png",  # back
            "左侧45.png",  # 45_left
            "右侧45.png",  # 45_right
            "俯视.1.png",  # top
            "俯视.2.png",  # top
            "侧仰.png",  # side
            "侧俯.png",  # side
        ]

        for filename in test_images:
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            img.save(temp_path / filename)

        yield temp_path

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def empty_reference_dir(self):
        """Create an empty temporary directory."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        yield temp_path
        shutil.rmtree(temp_dir)

    def test_load_reference_images(self, temp_reference_dir):
        """Test loading reference images from directory."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        # Should load 6 unique categories (8 files but some map to same category)
        assert len(manager.reference_images) == 6
        assert manager.has_images()

        # Check specific categories
        assert "front" in manager.reference_images
        assert "back" in manager.reference_images
        assert "45_left" in manager.reference_images
        assert "45_right" in manager.reference_images
        assert "top" in manager.reference_images
        assert "side" in manager.reference_images

    def test_filename_mappings(self, temp_reference_dir):
        """Test filename to angle category mapping."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        # Check front mapping
        front_img = manager.reference_images["front"]
        assert front_img.filename == "正面.png"
        assert front_img.priority == 0

        # Check back mapping
        back_img = manager.reference_images["back"]
        assert back_img.filename == "背面.png"
        assert back_img.priority == 10

        # Check 45-degree mappings
        left_45 = manager.reference_images["45_left"]
        assert left_45.filename == "左侧45.png"
        assert left_45.priority == 1

        right_45 = manager.reference_images["45_right"]
        assert right_45.filename == "右侧45.png"
        assert right_45.priority == 2

    def test_select_images_for_45_degree_angle(self, temp_reference_dir):
        """Test selecting images for 45-degree camera angle."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        selected = manager.select_images_for_angle("45-degree", max_images=3)

        # Should return 3 images: 45_left, 45_right, front
        assert len(selected) == 3
        assert selected[0].name == "左侧45.png"  # 45_left (priority 1)
        assert selected[1].name == "右侧45.png"  # 45_right (priority 2)
        assert selected[2].name == "正面.png"  # front (priority 0)

    def test_select_images_for_high_angle_shot(self, temp_reference_dir):
        """Test selecting images for high-angle shot."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        selected = manager.select_images_for_angle("45-degree High-Angle Shot", max_images=3)

        # Should return: top, 45_left, front
        assert len(selected) == 3
        assert "俯视" in selected[0].name  # top view
        assert "45" in selected[1].name  # 45-degree
        assert selected[2].name == "正面.png"  # front

    def test_select_images_for_eye_level_shot(self, temp_reference_dir):
        """Test selecting images for eye-level shot."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        selected = manager.select_images_for_angle("Eye-Level Shot", max_images=3)

        # Should return: front, 45_left, 45_right
        assert len(selected) == 3
        assert selected[0].name == "正面.png"  # front
        assert "45" in selected[1].name
        assert "45" in selected[2].name

    def test_select_images_with_max_images_limit(self, temp_reference_dir):
        """Test selecting images with max_images limit."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        # Request only 2 images
        selected = manager.select_images_for_angle("45-degree", max_images=2)

        assert len(selected) == 2
        assert selected[0].name == "左侧45.png"
        assert selected[1].name == "右侧45.png"

    def test_select_images_with_no_camera_angle(self, temp_reference_dir):
        """Test selecting images when camera_angle is None (default)."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        selected = manager.select_images_for_angle(None, max_images=3)

        # Should use default selection: front, 45_left, 45_right
        assert len(selected) == 3
        assert selected[0].name == "正面.png"  # front
        assert "45" in selected[1].name
        assert "45" in selected[2].name

    def test_select_images_with_empty_directory(self, empty_reference_dir):
        """Test selecting images from empty directory."""
        manager = ReferenceImageManager(reference_images_dir=empty_reference_dir)

        assert not manager.has_images()
        assert len(manager.reference_images) == 0

        # Should return empty list (no fallback configured)
        selected = manager.select_images_for_angle("45-degree", max_images=3)
        assert len(selected) == 0

    def test_select_images_with_fallback(self, empty_reference_dir):
        """Test fallback behavior when directory is empty."""
        # Create a fallback image
        temp_fallback = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        img.save(temp_fallback.name)
        temp_fallback.close()

        try:
            manager = ReferenceImageManager(
                reference_images_dir=empty_reference_dir,
                fallback_image_path=Path(temp_fallback.name),
            )

            # Should use fallback image
            selected = manager.select_images_for_angle("45-degree", max_images=3)
            assert len(selected) == 1
            assert selected[0] == Path(temp_fallback.name)
        finally:
            # Cleanup
            Path(temp_fallback.name).unlink()

    def test_get_all_image_paths(self, temp_reference_dir):
        """Test getting all reference image paths."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        all_paths = manager.get_all_image_paths()

        assert len(all_paths) == 6
        assert all(isinstance(p, Path) for p in all_paths)

    def test_get_available_categories(self, temp_reference_dir):
        """Test getting available angle categories."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        categories = manager.get_available_categories()

        assert len(categories) == 6  # front, back, 45_left, 45_right, top, side
        assert "front" in categories
        assert "back" in categories
        assert "45_left" in categories
        assert "45_right" in categories
        assert "top" in categories
        assert "side" in categories

    def test_nonexistent_directory(self):
        """Test initialization with non-existent directory."""
        manager = ReferenceImageManager(
            reference_images_dir=Path("/nonexistent/path"),
        )

        assert not manager.has_images()
        assert len(manager.reference_images) == 0

    def test_angle_selection_map_completeness(self):
        """Test that all pattern angles have mappings."""
        # This test ensures we don't miss any pattern angles
        expected_angles = [
            "45-degree",
            "45-degree High-Angle Shot",
            "Eye-Level Shot",
            "High-Angle Shot",
            "Top-Down",
            "Side View",
            "Low-Angle Shot",
        ]

        for angle in expected_angles:
            assert angle in ReferenceImageManager.ANGLE_SELECTION_MAP
            preferences = ReferenceImageManager.ANGLE_SELECTION_MAP[angle]
            assert len(preferences) > 0
            assert all(isinstance(p, str) for p in preferences)

    def test_priority_ordering(self, temp_reference_dir):
        """Test that images are selected by priority order."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        # For 45-degree, order should be: 45_left (priority 1), 45_right (priority 2), front (priority 0)
        # Note: front has priority 0 but comes last in the preference list
        selected = manager.select_images_for_angle("45-degree", max_images=3)

        # Check that selection follows ANGLE_SELECTION_MAP order
        assert selected[0].name == "左侧45.png"  # 45_left
        assert selected[1].name == "右侧45.png"  # 45_right
        assert selected[2].name == "正面.png"  # front
