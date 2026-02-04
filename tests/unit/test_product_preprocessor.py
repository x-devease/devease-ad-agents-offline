"""
Unit tests for ProductPreprocessor.

Tests product image preprocessing including:
- Transparency trimming
- Perspective detection
- Mask generation
"""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from src.meta.ad.generator.template_system.product_preprocessor import (
    ProductPreprocessor,
    PreprocessorResult,
    PerspectiveType,
    preprocess_product,
)


class TestPerspectiveType:
    """Test PerspectiveType enum."""

    def test_perspective_type_values(self):
        """Test perspective type enum values."""
        assert PerspectiveType.HIGH_ANGLE.value == "high_angle"
        assert PerspectiveType.EYE_LEVEL.value == "eye_level"

    def test_perspective_type_string_conversion(self):
        """Test perspective type to string conversion."""
        # For str-based enums, use .value to get the string value
        assert PerspectiveType.HIGH_ANGLE == "high_angle"
        assert PerspectiveType.EYE_LEVEL == "eye_level"
        # Or access via .value
        assert PerspectiveType.HIGH_ANGLE.value == "high_angle"
        assert PerspectiveType.EYE_LEVEL.value == "eye_level"


class TestPreprocessorResult:
    """Test PreprocessorResult dataclass."""

    def test_preprocessor_result_creation(self):
        """Test creating a PreprocessorResult."""
        # Create dummy image
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        mask = Image.new("L", (100, 100), 255)

        result = PreprocessorResult(
            trimmed_image=image,
            mask=mask,
            perspective=PerspectiveType.EYE_LEVEL,
            bbox=(10, 10, 80, 80),
            original_size=(100, 100)
        )

        assert result.trimmed_image == image
        assert result.mask == mask
        assert result.perspective == PerspectiveType.EYE_LEVEL
        assert result.bbox == (10, 10, 80, 80)
        assert result.original_size == (100, 100)

    def test_preprocessor_result_to_dict(self):
        """Test PreprocessorResult serialization to dict."""
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        mask = Image.new("L", (100, 100), 255)

        result = PreprocessorResult(
            trimmed_image=image,
            mask=mask,
            perspective=PerspectiveType.HIGH_ANGLE,
            bbox=(10, 10, 80, 80),
            original_size=(100, 100)
        )

        result_dict = result.to_dict()

        assert result_dict["perspective"] == "high_angle"
        assert result_dict["bbox"]["x"] == 10
        assert result_dict["bbox"]["y"] == 10
        assert result_dict["bbox"]["width"] == 80
        assert result_dict["bbox"]["height"] == 80
        assert result_dict["original_size"]["width"] == 100
        assert result_dict["original_size"]["height"] == 100
        assert result_dict["trimmed_size"]["width"] == 100
        assert result_dict["trimmed_size"]["height"] == 100


class TestProductPreprocessor:
    """Test ProductPreprocessor class."""

    def test_init_default(self):
        """Test default initialization."""
        preprocessor = ProductPreprocessor()

        assert preprocessor.padding == 10
        assert preprocessor.min_alpha_threshold == 10

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        preprocessor = ProductPreprocessor(
            padding=20,
            min_alpha_threshold=5
        )

        assert preprocessor.padding == 20
        assert preprocessor.min_alpha_threshold == 5

    def test_process_missing_file(self, tmp_path):
        """Test processing non-existent file raises FileNotFoundError."""
        preprocessor = ProductPreprocessor()

        with pytest.raises(FileNotFoundError):
            preprocessor.process(tmp_path / "nonexistent.png")

    def test_process_no_alpha_channel(self, tmp_path):
        """Test processing image without alpha channel."""
        # Create RGB image (no alpha)
        image = Image.new("RGB", (100, 100), (255, 0, 0))
        image_path = tmp_path / "rgb_image.png"
        image.save(image_path)

        preprocessor = ProductPreprocessor()
        result = preprocessor.process(image_path)

        assert isinstance(result, PreprocessorResult)
        assert result.perspective in [PerspectiveType.HIGH_ANGLE, PerspectiveType.EYE_LEVEL]

    def test_trim_transparency_square_image(self):
        """Test trimming transparency from square image."""
        preprocessor = ProductPreprocessor()

        # Create image with transparent border
        image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        # Add opaque square in center
        opaque = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        image.paste(opaque, (50, 50))

        trimmed, bbox = preprocessor._trim_transparency(image)

        # Should be trimmed to ~100x100 (plus padding)
        assert trimmed.width < 200
        assert trimmed.height < 200
        assert bbox[2] < 200  # width
        assert bbox[3] < 200  # height

    def test_trim_transparency_fully_transparent(self):
        """Test trimming fully transparent image."""
        preprocessor = ProductPreprocessor()

        # Create fully transparent image
        image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))

        trimmed, bbox = preprocessor._trim_transparency(image)

        # Should return original image dimensions
        assert trimmed.size == image.size

    def test_detect_perspective_wide_image(self):
        """Test perspective detection for wide image."""
        preprocessor = ProductPreprocessor()

        # Create wide image (width > height * 1.3)
        image = Image.new("RGBA", (200, 100), (255, 0, 0, 255))

        perspective = preprocessor._detect_perspective(image)

        assert perspective == PerspectiveType.HIGH_ANGLE

    def test_detect_perspective_tall_image(self):
        """Test perspective detection for tall image."""
        preprocessor = ProductPreprocessor()

        # Create tall image (height > width * 1.3)
        image = Image.new("RGBA", (100, 200), (255, 0, 0, 255))

        perspective = preprocessor._detect_perspective(image)

        assert perspective == PerspectiveType.EYE_LEVEL

    def test_detect_perspective_square_image(self):
        """Test perspective detection for square image."""
        preprocessor = ProductPreprocessor()

        # Create square image
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))

        perspective = preprocessor._detect_perspective(image)

        # Square images default to EYE_LEVEL
        assert perspective == PerspectiveType.EYE_LEVEL

    def test_generate_mask(self):
        """Test mask generation."""
        preprocessor = ProductPreprocessor()

        # Create image with semi-transparent center
        image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        center = Image.new("RGBA", (50, 50), (255, 0, 0, 200))
        image.paste(center, (25, 25))

        mask = preprocessor._generate_mask(image)

        # Mask should be grayscale
        assert mask.mode == "L"
        # Should have some white pixels (where alpha > threshold)
        mask_array = np.array(mask)
        assert np.any(mask_array > 0)


class TestPreprocessProduct:
    """Test preprocess_product convenience function."""

    def test_preprocess_product_convenience(self, tmp_path):
        """Test preprocess_product convenience function."""
        # Create test image
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        image_path = tmp_path / "product.png"
        image.save(image_path)

        result = preprocess_product(image_path)

        assert isinstance(result, PreprocessorResult)
        assert result.trimmed_image is not None
        assert result.mask is not None
        assert result.perspective in [PerspectiveType.HIGH_ANGLE, PerspectiveType.EYE_LEVEL]

    def test_preprocess_product_with_padding(self, tmp_path):
        """Test preprocess_product with custom padding."""
        image = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        image_path = tmp_path / "product.png"
        image.save(image_path)

        result = preprocess_product(image_path, padding=20)

        assert isinstance(result, PreprocessorResult)


@pytest.fixture
def sample_product_image(tmp_path):
    """Create a sample product image for testing."""
    # Create image with product in center
    image = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    # Add product (opaque red square)
    product = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    image.paste(product, (100, 100))

    image_path = tmp_path / "product.png"
    image.save(image_path)

    return image_path


class TestProductPreprocessorIntegration:
    """Integration tests for product preprocessing."""

    def test_full_preprocessing_pipeline(self, sample_product_image):
        """Test complete preprocessing pipeline."""
        preprocessor = ProductPreprocessor()
        result = preprocessor.process(sample_product_image)

        # Verify all components
        assert result.trimmed_image is not None
        assert result.mask is not None
        assert result.perspective in [PerspectiveType.HIGH_ANGLE, PerspectiveType.EYE_LEVEL]
        assert result.bbox is not None
        assert result.original_size == (300, 300)

        # Verify trimmed image is smaller than original
        trimmed_w, trimmed_h = result.trimmed_image.size
        original_w, original_h = result.original_size
        assert trimmed_w <= original_w
        assert trimmed_h <= original_h

    def test_result_serialization(self, sample_product_image):
        """Test PreprocessorResult can be serialized to dict."""
        preprocessor = ProductPreprocessor()
        result = preprocessor.process(sample_product_image)

        # Should serialize without errors
        result_dict = result.to_dict()
        assert "perspective" in result_dict
        assert "bbox" in result_dict
        assert "original_size" in result_dict
        assert "trimmed_size" in result_dict
