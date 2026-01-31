"""
Technical Specification Layer for Nano Banana Pro Agent.

Adds technical quality specifications (resolution, lighting, camera, style)
with professional product photography parameters for high-fidelity output.
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Dict, Any

from src.agents.nano.core.types import (
    TechnicalSpecs,
    Resolution,
    LightingStyle,
    CameraStyle,
)


logger = logging.getLogger(__name__)


class TechnicalSpecLayer:
    """Add technical specifications to the prompt."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducibility."""
        self.seed = seed or random.randint(1, 999999)

    def generate_specs(
        self,
        intent,
        preferred_resolution: Optional[Resolution] = None,
    ) -> Dict[str, Any]:
        """Generate technical specifications for the output."""

        # Basic specs
        resolution = preferred_resolution or self._default_resolution_for_intent(intent)
        lighting = self._default_lighting_for_intent(intent)
        camera = self._default_camera_for_intent(intent)
        style = self._default_style_for_intent(intent)

        # Professional photography enhancements
        pro_specs = self._pro_photography_specs_for_intent(intent)

        return {
            "resolution": resolution.value,
            "aspect_ratio": "3:4",  # Product photography standard
            "focal_length": pro_specs["focal_length"],
            "aperture": pro_specs["aperture"],
            "color_temperature": pro_specs["color_temperature"],
            "lighting_setup": pro_specs["lighting_setup"],
            "lighting_ratio": pro_specs["lighting_ratio"],
            "depth_of_field": pro_specs["depth_of_field"],
            "product_angle": pro_specs["product_angle"],
            "color_space": "Adobe RGB",
            "bit_depth": "16-bit",
            "seed": self.seed,
            "lighting": lighting.value.replace('_', ' '),
            "camera": camera.value.replace('_', ' '),
            "style": style,
        }

    def _pro_photography_specs_for_intent(self, intent) -> Dict[str, str]:
        """Get professional photography specs based on intent."""

        # Default professional product photography settings
        return {
            "focal_length": "85mm",  # Classic product photography lens
            "aperture": "f/5.6",  # Sweet spot for sharpness + slight DOF
            "color_temperature": "5600K",  # Daylight balanced
            "lighting_setup": "3-point lighting (key, fill, rim)",
            "lighting_ratio": "3:1 (key:fill)",
            "depth_of_field": "Front-to-back sharpness",
            "product_angle": "45-degree angle with 3/4 view",
        }

    def _default_resolution_for_intent(self, intent) -> Resolution:
        """Get default resolution based on intent."""

        if hasattr(intent, 'value') and 'product' in intent.value.lower():
            return Resolution.K4  # Product photos need high res
        return Resolution.K2  # Default to 2K

    def _default_lighting_for_intent(self, intent) -> LightingStyle:
        """Get default lighting based on intent."""

        if hasattr(intent, 'value') and 'lifestyle' in intent.value.lower():
            return LightingStyle.NATURAL_DAYLIGHT
        return LightingStyle.STUDIO_SOFT

    def _default_camera_for_intent(self, intent) -> CameraStyle:
        """Get default camera style based on intent."""

        return CameraStyle.EYE_LEVEL

    def _default_style_for_intent(self, intent) -> str:
        """Get default style declaration based on intent."""

        return "Professional photography, photorealistic rendering"

    def format_specs(self, specs: Dict[str, Any]) -> str:
        """Format technical specs as text for the prompt."""

        lines = []
        lines.append("Technical Specifications:")

        # Core specs
        lines.append(f"  - Resolution: {specs['resolution']}")
        lines.append(f"  - Aspect Ratio: {specs['aspect_ratio']}")

        # Camera parameters
        lines.append(f"  - Camera: {specs['camera']} at {specs['focal_length']}")
        lines.append(f"  - Aperture: {specs['aperture']} (optimal sharpness)")
        lines.append(f"  - Depth of Field: {specs['depth_of_field']}")

        # Lighting details
        lines.append(f"  - Lighting: {specs['lighting']}")
        lines.append(f"  - Color Temperature: {specs['color_temperature']}")
        lines.append(f"  - Lighting Setup: {specs['lighting_setup']}")
        lines.append(f"  - Lighting Ratio: {specs['lighting_ratio']}")

        # Composition
        lines.append(f"  - Product Angle: {specs['product_angle']}")

        # Image quality
        lines.append(f"  - Color Space: {specs['color_space']}")
        lines.append(f"  - Bit Depth: {specs['bit_depth']}")

        # Generation parameters
        lines.append(f"  - Seed: {specs['seed']} (for reproducibility)")

        # Style
        lines.append(f"  - Style: {specs['style']}")

        return "\n".join(lines)
