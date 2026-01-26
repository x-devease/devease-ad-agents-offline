"""Constants for image generation."""

from pathlib import Path


# ============================================================================
# File Paths
# ============================================================================

# Repo root (expected: .../devease-creative-gen-offline)
# This file lives at:
#   <repo>/src/core/generation/constants.py
# so parents[3] is the repo root directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
# Repo .env path (preferred)
REPO_ENV_FILE = REPO_ROOT / ".env"
# Environment file path
ENV_FILE = Path.home() / ".devease" / "keys"
# Temporary directory path
TEMP_DIR = Path.home() / ".devease" / "temp"


# ============================================================================
# FAL.ai Polling Constants
# ============================================================================

# Initial delay before first status check (seconds)
# FAL.ai needs time to start processing
POLLING_INITIAL_DELAY_SECONDS = 20

# Interval between status checks (seconds)
POLLING_INTERVAL_SECONDS = 5

# Maximum time to wait for image generation (seconds)
POLLING_MAX_WAIT_SECONDS = 300


# ============================================================================
# Image Generation Constants
# ============================================================================

# Source image preservation strength (0.0-1.0)
# Lower = more transformation, Higher = more faithful to source
STRENGTH_DEFAULT = 0.85

# Strength for high_efficiency macro close-ups
STRENGTH_MACRO_CLOSEUP = 0.6

# Texture enhancement factor for macro close-ups
TEXTURE_ENHANCEMENT_FACTOR = 1.3


# ============================================================================
# Frame Occupancy Constants
# ============================================================================

# Frame occupancy mapping from visual_prominence to frame percentage
# Used in templates for consistent product sizing
FRAME_OCCUPANCY_MAP = {
    "dominant": {
        "percentage": 45,
        "description": "Product occupying 45% of frame (dominant prominence).",
        "use_case": "Hero product, single-product focus",
    },
    "balanced": {
        "percentage": 30,
        "description": "Product occupying 30% of frame (balanced).",
        "use_case": "Standard commercial shots, multi-product",
    },
    "subtle": {
        "percentage": 20,
        "description": "Product occupying 20% of frame (subtle, lifestyle focus).",
        "use_case": "Environmental context, lifestyle scenes",
    },
}


# ============================================================================
# Watermark Constants
# ============================================================================

# Watermark size as percentage of image dimensions
WATERMARK_SIZE_PCT = 0.28

# Watermark opacity (0.0-1.0)
WATERMARK_OPACITY = 0.065

# Watermark margin as percentage of image size
WATERMARK_MARGIN_PCT = 0.025


# ============================================================================
# Text Overlay Constants
# ============================================================================

# Default text overlay settings
TEXT_OVERLAY_ENABLED = False
TEXT_OVERLAY_DEFAULT_FONT_SIZE = 24
TEXT_OVERLAY_DEFAULT_COLOR = (255, 255, 255, 255)  # White
TEXT_OVERLAY_DEFAULT_BACKGROUND = (0, 0, 0, 180)  # Semi-transparent black
TEXT_OVERLAY_DEFAULT_MARGIN = 20  # pixels
TEXT_OVERLAY_DEFAULT_PADDING = 10  # pixels
TEXT_OVERLAY_DEFAULT_CORNER_RADIUS = 8  # pixels


# ============================================================================
# Default Generation Config
# ============================================================================

DEFAULT_GENERATION_CONFIG = {
    "product_name": "Product",
    "brand": "",
    "category": "",
    "market": "US",
    "generation": {
        "source_image": "",
        "num_variations": 1,
        "aspect_ratio": "3:4",
        "model": "nano-banana-pro",
        "resolution": "1K",
        "enable_upscaling": False,
        "enable_watermark": False,
        "enable_text_overlay": False,
        "text_overlay_config": None,
        "include_negative_guidance": True,
        "use_llm": True,
        "use_orchestrator": False,
        "branch_name": None,  # "golden_ratio" | "high_efficiency" | "cool_peak"
        "step2_mode": False,
        "strength": 0.85,
        "guidance_scale": 8.0,
        "num_inference_steps": 30,
        # Feature flags (all default to True for professional quality)
        "anti_hallucination_enhanced": True,
        "camera_specs": True,
        "material_textures": True,
        "three_point_lighting": True,
        "depth_of_field": True,
        "post_processing": True,
        "shadow_specification": True,
        "frame_occupancy": True,
        "visual_flow": True,
        "color_accuracy_tolerance": True,
    },
}
