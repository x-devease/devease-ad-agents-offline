"""
Geometric Guard - Product integrity validation.

This guard validates that the product has not been distorted
during image generation using SIFT feature matching and
homography matrix decomposition.
"""

import time
import numpy as np
from pathlib import Path
from typing import Tuple

from ..schemas.audit_report import GuardStatus, GeometricResult
from ..utils.image_processor import (
    load_image,
    convert_to_grayscale,
    resize_to_match,
    get_image_dimensions
)
from ..utils.geometry_utils import (
    extract_sift_features,
    match_features,
    compute_homography,
    decompose_homography,
    validate_uniform_scaling,
    get_product_contour_aspect_ratio,
    compare_aspect_ratios
)


class GeometricGuard:
    """
    Validates product integrity using geometric analysis.

    Uses SIFT feature matching with homography decomposition
    to detect non-uniform scaling (product distortion).
    Falls back to contour-based aspect ratio comparison for
    products with insufficient features.
    """

    def __init__(
        self,
        tolerance: float = 0.02,
        min_features: int = 10,
        fallback: bool = True,
        contour_tolerance: float = 0.05
    ):
        """
        Initialize GeometricGuard.

        Args:
            tolerance: Allowed aspect ratio difference (0.02 = 2%)
            min_features: Minimum SIFT features for homography calculation
            fallback: Enable contour-based fallback method
            contour_tolerance: Tolerance for contour comparison fallback
        """
        self.tolerance = tolerance
        self.min_features = min_features
        self.fallback = fallback
        self.contour_tolerance = contour_tolerance

    def check(
        self,
        raw_product_path: str,
        candidate_path: str
    ) -> GeometricResult:
        """
        Check if product has been geometrically distorted.

        Args:
            raw_product_path: Path to original product image
            candidate_path: Path to generated candidate image

        Returns:
            GeometricResult with validation status
        """
        start_time = time.time()

        try:
            # Load images
            raw_image = load_image(raw_product_path)
            candidate_image = load_image(candidate_path)

            # Convert to grayscale
            raw_gray = convert_to_grayscale(raw_image)
            candidate_gray = convert_to_grayscale(candidate_image)

            # Try SIFT-based method first
            result = self._check_with_sift(raw_gray, candidate_gray)

            # If SIFT fails and fallback is enabled, use contour method
            if not result and self.fallback:
                result = self._check_with_contours(raw_gray, candidate_gray)

            execution_time = (time.time() - start_time) * 1000

            if result:
                is_valid, aspect_ratio_delta, num_features, method = result

                return GeometricResult(
                    guard_name="geometric",
                    status=GuardStatus.PASS if is_valid else GuardStatus.FAIL,
                    reasoning=self._generate_reasoning(
                        is_valid, aspect_ratio_delta, num_features, method
                    ),
                    metrics={
                        "aspect_ratio_delta": aspect_ratio_delta,
                        "num_matched_features": num_features,
                        "method_used": method,
                        "tolerance": self.tolerance if method == "homography" else self.contour_tolerance
                    },
                    execution_time_ms=execution_time,
                    aspect_ratio_delta=aspect_ratio_delta,
                    num_matched_features=num_features,
                    method_used=method
                )
            else:
                # Both methods failed
                return GeometricResult(
                    guard_name="geometric",
                    status=GuardStatus.FAIL,
                    reasoning="Failed to validate geometry: insufficient features and contour method unavailable",
                    metrics={},
                    execution_time_ms=execution_time,
                    aspect_ratio_delta=float('inf'),
                    num_matched_features=0,
                    method_used="none"
                )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return GeometricResult(
                guard_name="geometric",
                status=GuardStatus.FAIL,
                reasoning=f"Error during geometric validation: {str(e)}",
                metrics={"error": str(e)},
                execution_time_ms=execution_time,
                aspect_ratio_delta=float('inf'),
                num_matched_features=0,
                method_used="error"
            )

    def _check_with_sift(
        self,
        raw_gray: np.ndarray,
        candidate_gray: np.ndarray
    ) -> Tuple[bool, float, int, str]:
        """
        Check geometry using SIFT feature matching.

        Returns:
            Tuple of (is_valid, aspect_ratio_delta, num_features, method)
            Returns (False, 0, 0, "") if insufficient features
        """
        # Resize candidate to match raw if needed
        if raw_gray.shape != candidate_gray.shape:
            candidate_gray = resize_to_match(candidate_gray, raw_gray.shape)

        # Extract SIFT features
        _, kp1, desc1 = extract_sift_features(raw_gray)
        _, kp2, desc2 = extract_sift_features(candidate_gray)

        # Check if we have valid descriptors
        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            return None

        # Match features
        matches = match_features(desc1, desc2)

        num_features = len(matches)
        if num_features < self.min_features:
            # Not enough features for reliable homography
            return None

        # Compute homography
        H = compute_homography(kp1, kp2, matches)

        if H is None:
            # Homography computation failed
            return None

        # Decompose homography
        result = decompose_homography(H)

        # Validate uniform scaling
        is_valid, delta = validate_uniform_scaling(
            result.scale_x,
            result.scale_y,
            self.tolerance
        )

        return is_valid, delta, num_features, "homography"

    def _check_with_contours(
        self,
        raw_gray: np.ndarray,
        candidate_gray: np.ndarray
    ) -> Tuple[bool, float, int, str]:
        """
        Check geometry using contour-based aspect ratio comparison.

        Returns:
            Tuple of (is_valid, aspect_ratio_delta, num_features, method)
            Returns (False, 0, 0, "") if contours not found
        """
        # Get aspect ratios from contours
        raw_ratio = get_product_contour_aspect_ratio(raw_gray)
        candidate_ratio = get_product_contour_aspect_ratio(candidate_gray)

        if raw_ratio is None or candidate_ratio is None:
            return None

        # Compare aspect ratios
        is_valid, delta = compare_aspect_ratios(
            raw_ratio,
            candidate_ratio,
            self.contour_tolerance
        )

        # Return with num_features = 0 to indicate contour method was used
        return is_valid, delta, 0, "contour"

    def _generate_reasoning(
        self,
        is_valid: bool,
        aspect_ratio_delta: float,
        num_features: int,
        method: str
    ) -> str:
        """Generate human-readable reasoning for the result."""
        if method == "homography":
            if is_valid:
                return f"Product geometry validated using {num_features} SIFT feature matches. Aspect ratio delta: {aspect_ratio_delta:.4f} (within tolerance {self.tolerance})"
            else:
                return f"Product distortion detected. Aspect ratio delta: {aspect_ratio_delta:.4f} exceeds tolerance {self.tolerance}. Non-uniform scaling detected from {num_features} feature matches."

        elif method == "contour":
            if is_valid:
                return f"Product geometry validated using contour comparison (fallback method). Aspect ratio delta: {aspect_ratio_delta:.4f} (within tolerance {self.contour_tolerance})"
            else:
                return f"Product distortion detected using contour comparison. Aspect ratio delta: {aspect_ratio_delta:.4f} exceeds tolerance {self.contour_tolerance}."

        else:
            return "Geometric validation failed - unable to determine product distortion."
