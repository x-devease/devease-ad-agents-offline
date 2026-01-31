"""
Integration test for multi-image generation with angle-aware reference selection.

Tests end-to-end flow:
1. Load config with reference_images section
2. Initialize ImageGenerator with multi-image enabled
3. Generate images with different camera angles
4. Verify correct reference images are selected
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from src.meta.ad.generator.core.generation.generator import ImageGenerator
from src.meta.ad.generator.core.generation.reference_image_manager import ReferenceImageManager


@pytest.fixture
def multi_image_config():
    """Create a test config with reference_images section."""
    return {
        "reference_images": {
            "enabled": True,
            "directory": "config/moprobo/product",
            "max_images": 3,
            "fallback_to_single_image": True,
        },
        "prompt_building": {
            "enable_nano_enhancement": False,  # Disable for faster testing
        },
    }


@pytest.fixture
def temp_reference_dir():
    """Create a temporary directory with test reference images."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Create test images simulating moprobo product images
    test_images = [
        "正面.png",  # front
        "背面.png",  # back
        "左侧45.png",  # 45_left
        "右侧45.png",  # 45_right
        "俯视.1.png",  # top view 1
        "俯视.2.png",  # top view 2
        "侧仰.png",  # side up
        "侧俯.png",  # side down
    ]

    for filename in test_images:
        img = Image.new("RGB", (512, 512), color=(255, 0, 0))
        img.save(temp_path / filename)

    yield temp_path

    # Cleanup
    shutil.rmtree(temp_path)


@pytest.fixture
def source_image(temp_reference_dir):
    """Create a source image for generation."""
    source_path = temp_reference_dir / "正面.png"
    return str(source_path)


class TestMultiImageGeneration:
    """Test multi-image generation integration."""

    def test_reference_manager_with_test_images(self, temp_reference_dir):
        """Test ReferenceImageManager with test images."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        # Should load 6 unique categories (8 files but some map to same category)
        assert manager.has_images()
        assert len(manager.get_all_image_paths()) == 6

        # Verify all expected categories are present
        expected_categories = {"front", "back", "45_left", "45_right", "top", "side"}
        actual_categories = {img.angle_category for img in manager.reference_images.values()}
        assert actual_categories == expected_categories

        # Test angle selection
        selected_45 = manager.select_images_for_angle("45-degree", max_images=3)
        assert len(selected_45) == 3
        assert "左侧45.png" in [p.name for p in selected_45]
        assert "右侧45.png" in [p.name for p in selected_45]

        selected_eye_level = manager.select_images_for_angle("Eye-Level Shot", max_images=3)
        assert len(selected_eye_level) == 3
        assert "正面.png" in [p.name for p in selected_eye_level]

    def test_image_generator_initialization_with_multi_image(
        self, temp_reference_dir, source_image
    ):
        """Test ImageGenerator initialization with multi-image enabled."""
        # Note: This test doesn't require actual API calls
        # It just tests the initialization logic

        generator = ImageGenerator(
            model="nano-banana-pro",
            reference_images_dir=str(temp_reference_dir),
            enable_multi_image=True,
            enable_watermark=False,
            enable_upscaling=False,
            use_gpt4o_conversion=False,  # Disable for testing
        )

        # Verify multi-image is enabled
        assert generator.enable_multi_image is True
        assert generator.reference_manager is not None
        assert generator.reference_manager.has_images()

    def test_image_generator_fallback_to_single_image(
        self, temp_reference_dir, source_image
    ):
        """Test ImageGenerator fallback when multi-image disabled."""
        generator = ImageGenerator(
            model="nano-banana-pro",
            reference_images_dir=str(temp_reference_dir),
            enable_multi_image=False,  # Disabled
            enable_watermark=False,
            enable_upscaling=False,
            use_gpt4o_conversion=False,
        )

        # Verify multi-image is disabled
        assert generator.enable_multi_image is False
        assert generator.reference_manager is None

    def test_image_generator_with_nonexistent_directory(self, source_image):
        """Test ImageGenerator with non-existent reference directory."""
        generator = ImageGenerator(
            model="nano-banana-pro",
            reference_images_dir="/nonexistent/path",
            enable_multi_image=True,
            enable_watermark=False,
            enable_upscaling=False,
            use_gpt4o_conversion=False,
        )

        # Should still initialize, but reference_manager should have no images
        assert generator.enable_multi_image is True
        assert generator.reference_manager is not None
        assert not generator.reference_manager.has_images()

    def test_camera_angle_parameter_in_generate(
        self, temp_reference_dir, source_image
    ):
        """Test that camera_angle parameter is accepted by generate method."""
        generator = ImageGenerator(
            model="nano-banana-pro",
            reference_images_dir=str(temp_reference_dir),
            enable_multi_image=True,
            enable_watermark=False,
            enable_upscaling=False,
            use_gpt4o_conversion=False,
        )

        # Test that the method accepts camera_angle parameter
        # Note: We can't actually test generation without API keys
        # This is a parameter validation test
        import inspect

        sig = inspect.signature(generator.generate)
        assert "camera_angle" in sig.parameters

    def test_angle_selection_for_all_pattern_angles(self, temp_reference_dir):
        """Test angle selection for all pattern camera angles."""
        manager = ReferenceImageManager(reference_images_dir=temp_reference_dir)

        # Test all pattern angles
        test_cases = [
            ("45-degree", ["左侧45.png", "右侧45.png", "正面.png"]),
            ("45-degree High-Angle Shot", ["俯视"]),
            ("Eye-Level Shot", ["正面.png"]),
            ("High-Angle Shot", ["俯视"]),
            ("Top-Down", ["俯视"]),
            ("Side View", ["侧仰.png"]),  # 侧仰.png has priority 5 (higher than 侧俯.png at 6)
            (None, ["正面.png"]),  # Default
        ]

        for camera_angle, expected_filenames_substring in test_cases:
            selected = manager.select_images_for_angle(camera_angle, max_images=3)

            # Should return some images
            assert len(selected) > 0, f"No images selected for angle '{camera_angle}'"

            # Check that expected files are included
            selected_names = [p.name for p in selected]
            for expected_substring in expected_filenames_substring:
                found = any(expected_substring in name for name in selected_names)
                assert found, f"Expected '{expected_substring}' in selection for angle '{camera_angle}', got: {selected_names}"


class TestMultiImageGenerationConfig:
    """Test multi-image generation configuration."""

    def test_config_has_reference_images_section(self, multi_image_config):
        """Test that config has reference_images section."""
        assert "reference_images" in multi_image_config
        assert multi_image_config["reference_images"]["enabled"] is True
        assert multi_image_config["reference_images"]["max_images"] == 3

    def test_config_directory_path(self, multi_image_config):
        """Test that config directory path is correct."""
        directory = multi_image_config["reference_images"]["directory"]
        assert directory == "config/moprobo/product"

    def test_config_max_images_range(self, multi_image_config):
        """Test that max_images is within valid range."""
        max_images = multi_image_config["reference_images"]["max_images"]
        assert 1 <= max_images <= 14  # API supports up to 14 images


@pytest.mark.integration
class TestMultiImageGenerationEndToEnd:
    """
    End-to-end integration tests for multi-image generation.

    Note: These tests require valid FAL.ai API credentials.
    They are marked as integration tests and should be run separately.
    """

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.mark.skip(reason="Requires FAL.ai API credentials")
    def test_generate_with_45_degree_angle(
        self, temp_reference_dir, source_image, output_dir
    ):
        """
        Test actual image generation with 45-degree camera angle.

        This test requires valid FAL.ai credentials and will make actual API calls.
        """
        generator = ImageGenerator(
            model="nano-banana-pro",
            reference_images_dir=str(temp_reference_dir),
            enable_multi_image=True,
            output_dir=str(output_dir),
            enable_watermark=False,
            enable_upscaling=False,
            use_gpt4o_conversion=False,
        )

        result = generator.generate(
            prompt="Professional product photograph",
            source_image_path=source_image,
            camera_angle="45-degree",
        )

        # Verify generation succeeded
        assert result["success"] is True
        assert "image_path" in result
        assert Path(result["image_path"]).exists()

    @pytest.mark.skip(reason="Requires FAL.ai API credentials")
    def test_generate_with_eye_level_angle(
        self, temp_reference_dir, source_image, output_dir
    ):
        """
        Test actual image generation with eye-level camera angle.

        This test requires valid FAL.ai credentials and will make actual API calls.
        """
        generator = ImageGenerator(
            model="nano-banana-pro",
            reference_images_dir=str(temp_reference_dir),
            enable_multi_image=True,
            output_dir=str(output_dir),
            enable_watermark=False,
            enable_upscaling=False,
            use_gpt4o_conversion=False,
        )

        result = generator.generate(
            prompt="Professional product photograph at eye level",
            source_image_path=source_image,
            camera_angle="Eye-Level Shot",
        )

        # Verify generation succeeded
        assert result["success"] is True
        assert "image_path" in result

    @pytest.mark.skip(reason="Requires FAL.ai API credentials")
    def test_generate_single_vs_multi_image_comparison(
        self, temp_reference_dir, source_image, output_dir
    ):
        """
        Compare single image vs multi-image generation quality.

        This test requires valid FAL.ai credentials and will make actual API calls.
        """
        # Single image generation
        single_generator = ImageGenerator(
            model="nano-banana-pro",
            reference_images_dir=str(temp_reference_dir),
            enable_multi_image=False,  # Single image mode
            output_dir=str(output_dir / "single"),
            enable_watermark=False,
            enable_upscaling=False,
            use_gpt4o_conversion=False,
        )

        single_result = single_generator.generate(
            prompt="Professional product photograph",
            source_image_path=source_image,
            camera_angle="45-degree",
        )

        # Multi-image generation
        multi_generator = ImageGenerator(
            model="nano-banana-pro",
            reference_images_dir=str(temp_reference_dir),
            enable_multi_image=True,  # Multi-image mode
            output_dir=str(output_dir / "multi"),
            enable_watermark=False,
            enable_upscaling=False,
            use_gpt4o_conversion=False,
        )

        multi_result = multi_generator.generate(
            prompt="Professional product photograph",
            source_image_path=source_image,
            camera_angle="45-degree",
        )

        # Both should succeed
        assert single_result["success"] is True
        assert multi_result["success"] is True

        # Both should produce images
        assert Path(single_result["image_path"]).exists()
        assert Path(multi_result["image_path"]).exists()

        # Note: Quality comparison would require visual assessment
        # or automated metrics (e.g., FID score, perceptual similarity)
