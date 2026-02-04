"""
Geometry utilities for product integrity validation.

This module provides SIFT feature matching, homography matrix decomposition,
and contour-based fallback methods for validating product geometry.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class HomographyResult:
    """Result from homography matrix analysis."""
    scale_x: float
    scale_y: float
    rotation: float  # in degrees
    translation: Tuple[float, float]  # (tx, ty)
    matrix: np.ndarray  # 3x3 homography matrix


def extract_sift_features(
    image: np.ndarray,
    max_features: int = 0
) -> Tuple[cv2.SIFT, np.ndarray, np.ndarray]:
    """
    Extract SIFT features from an image.

    Args:
        image: Grayscale input image
        max_features: Maximum number of features (0 = no limit)

    Returns:
        Tuple of (sift_object, keypoints, descriptors)
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_features)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return sift, keypoints, descriptors


def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio_threshold: float = 0.75
) -> list:
    """
    Match SIFT features using BFMatcher with ratio test.

    Args:
        descriptors1: Descriptors from first image
        descriptors2: Descriptors from second image
        ratio_threshold: Ratio test threshold (Lowe's ratio test)

    Returns:
        List of DMatch objects representing good matches
    """
    # Use BFMatcher with default params (normType=NORM_L2)
    bf = cv2.BFMatcher()

    # Apply ratio test
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def compute_homography(
    keypoints1: list,
    keypoints2: list,
    matches: list,
    ransac_threshold: float = 5.0
) -> Optional[np.ndarray]:
    """
    Compute homography matrix from matched keypoints.

    Args:
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: List of DMatch objects
        ransac_threshold: RANSAC reprojection threshold in pixels

    Returns:
        3x3 homography matrix, or None if computation failed
    """
    if len(matches) < 4:
        # Need at least 4 points to compute homography
        return None

    # Extract matched point coordinates
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_threshold
    )

    return H


def decompose_homography(H: np.ndarray) -> HomographyResult:
    """
    Decompose homography matrix to extract scaling, rotation, and translation.

    Args:
        H: 3x3 homography matrix

    Returns:
        HomographyResult with decomposed parameters
    """
    # Normalize the matrix
    H = H / H[2, 2]

    # Extract scale from the upper-left 2x2 submatrix
    A = H[:2, :2]

    # Perform SVD to get scale and rotation
    U, S, Vt = np.linalg.svd(A)

    # Scale factors (singular values)
    scale_x = S[0]
    scale_y = S[1]

    # Rotation angle
    rotation = np.arctan2(U[1, 0], U[0, 0]) * 180 / np.pi

    # Translation from the third column
    translation = (H[0, 2], H[1, 2])

    return HomographyResult(
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        rotation=float(rotation),
        translation=translation,
        matrix=H
    )


def validate_uniform_scaling(
    scale_x: float,
    scale_y: float,
    tolerance: float = 0.02
) -> Tuple[bool, float]:
    """
    Validate if scaling is uniform (no distortion).

    Args:
        scale_x: X-axis scale factor
        scale_y: Y-axis scale factor
        tolerance: Allowed ratio difference (e.g., 0.02 = 2%)

    Returns:
        Tuple of (is_valid, ratio_difference)
    """
    # Avoid division by zero
    if scale_y == 0:
        return False, float('inf')

    ratio = scale_x / scale_y
    delta = abs(ratio - 1.0)

    is_valid = delta <= tolerance
    return is_valid, delta


def find_contours(
    image: np.ndarray,
    mode: int = cv2.RETR_EXTERNAL,
    method: int = cv2.CHAIN_APPROX_SIMPLE
) -> list:
    """
    Find contours in a binary image.

    Args:
        image: Binary input image
        mode: Contour retrieval mode
        method: Contour approximation method

    Returns:
        List of contours
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply threshold if not already binary
    if gray.dtype != np.uint8 or gray.max() > 1:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = (gray * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary, mode, method)

    return contours


def get_bounding_box_aspect_ratio(
    contour: np.ndarray
) -> float:
    """
    Calculate aspect ratio of contour's bounding box.

    Args:
        contour: Single contour as numpy array

    Returns:
        Aspect ratio (width / height)
    """
    x, y, w, h = cv2.boundingRect(contour)

    if h == 0:
        return 0.0

    return w / h


def find_largest_contour(
    contours: list
) -> Optional[np.ndarray]:
    """
    Find the largest contour by area.

    Args:
        contours: List of contours

    Returns:
        Largest contour, or None if list is empty
    """
    if not contours:
        return None

    # Sort by area (descending)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours_sorted[0]


def compare_aspect_ratios(
    ratio1: float,
    ratio2: float,
    tolerance: float = 0.05
) -> Tuple[bool, float]:
    """
    Compare two aspect ratios.

    Args:
        ratio1: First aspect ratio
        ratio2: Second aspect ratio
        tolerance: Allowed difference

    Returns:
        Tuple of (are_similar, difference)
    """
    delta = abs(ratio1 - ratio2)
    are_similar = delta <= tolerance

    return are_similar, delta


def get_product_contour_aspect_ratio(
    image: np.ndarray,
    min_area: int = 1000
) -> Optional[float]:
    """
    Get the aspect ratio of the largest contour in an image.
    Useful as a fallback for products with few SIFT features.

    Args:
        image: Input image (grayscale or color)
        min_area: Minimum contour area to consider

    Returns:
        Aspect ratio, or None if no suitable contour found
    """
    # Find contours
    contours = find_contours(image)

    if not contours:
        return None

    # Filter by area and get largest
    large_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not large_contours:
        return None

    largest = find_largest_contour(large_contours)

    if largest is None:
        return None

    # Calculate aspect ratio
    ratio = get_bounding_box_aspect_ratio(largest)

    return ratio


def compute_geometric_validation_result(
    aspect_ratio_delta: float,
    num_features: int,
    method_used: str,
    is_valid: bool,
    tolerance: float,
    execution_time_ms: float
) -> Dict[str, Any]:
    """
    Create a geometric validation result dictionary.

    Args:
        aspect_ratio_delta: Difference in aspect ratios
        num_features: Number of matched features
        method_used: "homography" or "contour"
        is_valid: Whether validation passed
        tolerance: Tolerance used for validation
        execution_time_ms: Execution time in milliseconds

    Returns:
        Dictionary with validation results
    """
    return {
        "aspect_ratio_delta": aspect_ratio_delta,
        "num_matched_features": num_features,
        "method_used": method_used,
        "is_valid": is_valid,
        "tolerance": tolerance,
        "execution_time_ms": execution_time_ms
    }
