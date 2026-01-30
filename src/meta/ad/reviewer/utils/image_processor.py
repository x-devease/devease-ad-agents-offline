"""
Image processing utilities for the Ad Reviewer.

This module provides helper functions for loading, preprocessing,
and analyzing images using OpenCV and PIL.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


def load_image(
    image_path: Union[str, Path],
    grayscale: bool = False,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to image file
        grayscale: If True, load as grayscale
        target_size: Optional (width, height) to resize to

    Returns:
        Image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    if grayscale:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize if needed
    if target_size is not None:
        image = cv2.resize(image, target_size)

    return image


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save an image to disk.

    Args:
        image: Image as numpy array
        output_path: Path to save image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)


def get_image_dimensions(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Get image dimensions.

    Args:
        image: Image as numpy array

    Returns:
        Tuple of (height, width, channels)
    """
    if len(image.shape) == 2:
        # Grayscale
        h, w = image.shape
        return h, w, 1
    else:
        # Color
        h, w, c = image.shape
        return h, w, c


def calculate_aspect_ratio(image: np.ndarray) -> float:
    """
    Calculate aspect ratio of an image.

    Args:
        image: Image as numpy array

    Returns:
        Aspect ratio (width / height)
    """
    h, w = image.shape[:2]
    return w / h if h > 0 else 0.0


def resize_to_match(
    image: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resize image to match target shape.

    Args:
        image: Source image
        target_shape: Target (height, width)

    Returns:
        Resized image
    """
    target_h, target_w = target_shape[:2]
    return cv2.resize(image, (target_w, target_h))


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert color image to grayscale.

    Args:
        image: Color image (BGR format)

    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to [0, 1] range.

    Args:
        image: Image with values in [0, 255]

    Returns:
        Normalized image with values in [0, 1]
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert normalized image back to [0, 255] range.

    Args:
        image: Image with values in [0, 1]

    Returns:
        Image with values in [0, 255]
    """
    return (image * 255.0).astype(np.uint8)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Apply Gaussian blur to image.

    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation (0 = auto-calculate)

    Returns:
        Blurred image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make odd

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def detect_edges(
    image: np.ndarray,
    low_threshold: float = 50,
    high_threshold: float = 150
) -> np.ndarray:
    """
    Detect edges using Canny edge detector.

    Args:
        image: Grayscale input image
        low_threshold: Lower threshold for edge linking
        high_threshold: Upper threshold for edge linking

    Returns:
        Binary edge map
    """
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)

    return cv2.Canny(image, low_threshold, high_threshold)


def get_image_hash(image: np.ndarray) -> str:
    """
    Compute a simple hash of an image for comparison.

    Args:
        image: Input image

    Returns:
        Hex string hash
    """
    import hashlib

    # Resize to small fixed size for hashing
    small = cv2.resize(image, (8, 8))
    gray = convert_to_grayscale(small) if len(small.shape) == 3 else small

    # Compute hash
    h = hashlib.md5(gray.tobytes()).hexdigest()
    return h


def are_images_similar(
    image1: np.ndarray,
    image2: np.ndarray,
    threshold: float = 0.95
) -> bool:
    """
    Check if two images are similar using structural similarity.

    Args:
        image1: First image
        image2: Second image
        threshold: Similarity threshold (0-1)

    Returns:
        True if images are similar above threshold
    """
    # Convert to grayscale if needed
    gray1 = convert_to_grayscale(image1) if len(image1.shape) == 3 else image1
    gray2 = convert_to_grayscale(image2) if len(image2.shape) == 3 else image2

    # Resize to match
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # Compute structural similarity
    # Note: This requires scikit-image - for now use simple MSE
    mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
    max_mse = 255.0 ** 2
    similarity = 1.0 - (mse / max_mse)

    return similarity >= threshold
