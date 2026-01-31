"""
Technical Specification Layer for Nano Banana Pro Agent.

Adds technical quality specifications (resolution, lighting, camera, style).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents.nano.core.types import (
    TechnicalSpecs,
    Resolution,
    LightingStyle,
    CameraStyle,
)


logger = logging.getLogger(__name__)


class TechnicalSpecLayer:
    """Add technical specifications to the prompt."""

    def generate_specs(
        self,
        intent,
        preferred_resolution: Optional[Resolution] = None,
    ) -> TechnicalSpecs:
        """Generate technical specifications for the output."""

        # Resolution
        resolution = preferred_resolution or self._default_resolution_for_intent(intent)

        # Lighting style
        lighting = self._default_lighting_for_intent(intent)

        # Camera style
        camera = self._default_camera_for_intent(intent)

        # Style declaration
        style = self._default_style_for_intent(intent)

        return TechnicalSpecs(
            resolution=resolution,
            lighting_style=lighting,
            camera_style=camera,
            style_declaration=style,
        )

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

    def format_specs(self, specs: TechnicalSpecs) -> str:
        """Format technical specs as text for the prompt."""

        parts = []

        parts.append(f"Resolution: {specs.resolution.value}")
        parts.append(f"Aspect Ratio: {specs.aspect_ratio}")

        if specs.lighting_style:
            parts.append(f"Lighting: {specs.lighting_style.value.replace('_', ' ')}")

        if specs.camera_style:
            parts.append(f"Camera: {specs.camera_style.value.replace('_', ' ')}")

        if specs.style_declaration:
            parts.append(f"Style: {specs.style_declaration}")

        return "Technical Specifications:\n" + "\n".join(f"  - {p}" for p in parts)
