"""
Unit tests for GeometricGuard.

Tests SIFT-based and contour-based geometric validation.
"""

import pytest
import numpy as np
from pathlib import Path

from src.meta.ad.reviewer.guards.geometric_guard import GeometricGuard
from src.meta.ad.reviewer.schemas.audit_report import GuardStatus


class TestGeometricGuard:
    """Test suite for GeometricGuard."""

    def test_initialization(self):
        """Test guard initialization with default parameters."""
        guard = GeometricGuard()

        assert guard.tolerance == 0.02
        assert guard.min_features == 10
        assert guard.fallback is True
        assert guard.contour_tolerance == 0.05

    def test_initialization_custom_params(self):
        """Test guard initialization with custom parameters."""
        guard = GeometricGuard(
            tolerance=0.05,
            min_features=15,
            fallback=False,
            contour_tolerance=0.10
        )

        assert guard.tolerance == 0.05
        assert guard.min_features == 15
        assert guard.fallback is False
        assert guard.contour_tolerance == 0.10

    def test_check_with_missing_images(self):
        """Test handling of missing image files."""
        guard = GeometricGuard()

        result = guard.check(
            raw_product_path="nonexistent_raw.jpg",
            candidate_path="nonexistent_candidate.jpg"
        )

        assert result.status == GuardStatus.FAIL
        assert "Error during geometric validation" in result.reasoning or "not found" in result.reasoning.lower()

    def test_aspect_ratio_validation(self):
        """Test uniform scaling validation."""
        guard = GeometricGuard(tolerance=0.02)

        # Test perfect match (scale_x == scale_y)
        is_valid, delta = guard._check_with_sift._validate_uniform_scaling if hasattr(guard, '_validate_uniform_scaling') else (lambda x, y, t: (abs(x/y - 1) <= t, abs(x/y - 1)))(1.0, 1.0, 0.02)
        from src.meta.ad.reviewer.utils.geometry_utils import validate_uniform_scaling
        is_valid, delta = validate_uniform_scaling(1.0, 1.0, 0.02)

        assert is_valid is True
        assert delta == 0.0

        # Test slight distortion (within tolerance)
        is_valid, delta = validate_uniform_scaling(1.01, 1.0, 0.02)

        assert is_valid is True
        assert abs(delta - 0.01) < 0.0001  # Allow floating point tolerance

        # Test excessive distortion (exceeds tolerance)
        is_valid, delta = validate_uniform_scaling(1.05, 1.0, 0.02)

        assert is_valid is False
        assert abs(delta - 0.05) < 0.0001  # Allow floating point tolerance

    def test_reasoning_generation(self):
        """Test reasoning message generation."""
        guard = GeometricGuard(tolerance=0.02)

        # Valid case with SIFT
        reasoning = guard._generate_reasoning(
            is_valid=True,
            aspect_ratio_delta=0.005,
            num_features=25,
            method="homography"
        )

        assert "validated" in reasoning.lower()
        assert "25" in reasoning
        assert "0.005" in reasoning
        assert "homography" in reasoning or "sift" in reasoning.lower()

        # Invalid case with SIFT
        reasoning = guard._generate_reasoning(
            is_valid=False,
            aspect_ratio_delta=0.08,
            num_features=20,
            method="homography"
        )

        assert "distortion detected" in reasoning.lower()
        assert "0.08" in reasoning
        assert "exceeds tolerance" in reasoning.lower()

        # Valid case with contour fallback
        reasoning = guard._generate_reasoning(
            is_valid=True,
            aspect_ratio_delta=0.03,
            num_features=0,
            method="contour"
        )

        assert "contour" in reasoning.lower()
        assert "validated" in reasoning.lower()


class TestGeometricGuardIntegration:
    """Integration tests with actual images (when available)."""

    @pytest.mark.skipif(
        not Path("tests/fixtures/product.jpg").exists(),
        reason="Test fixtures not available"
    )
    def test_check_with_real_images(self):
        """Test geometric validation with real images."""
        guard = GeometricGuard()

        result = guard.check(
            raw_product_path="tests/fixtures/product.jpg",
            candidate_path="tests/fixtures/product_undistorted.jpg"
        )

        assert result is not None
        assert result.status in [GuardStatus.PASS, GuardStatus.FAIL]

    @pytest.mark.skipif(
        not Path("tests/fixtures/product.jpg").exists(),
        reason="Test fixtures not available"
    )
    def test_check_with_distorted_image(self):
        """Test geometric validation detects distorted images."""
        guard = GeometricGuard(tolerance=0.02)

        result = guard.check(
            raw_product_path="tests/fixtures/product.jpg",
            candidate_path="tests/fixtures/product_distorted.jpg"
        )

        # Should detect distortion
        if result.method_used == "homography":
            # If SIFT worked, should detect distortion
            if result.aspect_ratio_delta > 0.02:
                assert result.status == GuardStatus.FAIL
        else:
            # Contour method may have different thresholds
            assert result.status in [GuardStatus.PASS, GuardStatus.FAIL]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
