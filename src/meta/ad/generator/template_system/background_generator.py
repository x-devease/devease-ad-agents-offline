"""
NanoBanana Pro Background Generator with Perspective Awareness.

Generates photorealistic background images with perspective matching
product camera angle for seamless compositing.

Key Features:
- Reads nano_generation_rules from master_blueprint
- Injects perspective parameter (high_angle vs eye_level)
- Generates with NanoBanana Pro (8 steps, CFG 3.5)
- Batch generation for parallel processing
- Ensures angle consistency (table matches product perspective)

Author: Ad System
Date: 2026-01-30
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

from .product_preprocessor import PerspectiveType


logger = logging.getLogger(__name__)


class AspectRatio(str, Enum):
    """Supported aspect ratios for background generation."""

    SQUARE = "1:1"
    PORTRAIT = "3:4"
    LANDSCAPE = "4:3"
    STORY = "9:16"


@dataclass
class GenerationConfig:
    """
    NanoBanana Pro generation configuration.

    Attributes:
        model: Model identifier (default: nanobanana_pro)
        steps: Number of diffusion steps (default: 8)
        cfg_scale: CFG scale (default: 3.5)
        batch_size: Number of images to generate in parallel (default: 20)
        aspect_ratio: Output aspect ratio (default: 3:4 for portrait ads)
        guidance: Special guidance mode (perspective_aware)
    """
    model: str = "nanobanana_pro"
    steps: int = 8
    cfg_scale: float = 3.5
    batch_size: int = 20
    aspect_ratio: str = "3:4"
    guidance: str = "perspective_aware"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Create from dictionary (YAML deserialization)."""
        # Handle both nested (nano_generation_rules) and flat structure
        if "nano_generation_rules" in data:
            inference_config = data["nano_generation_rules"].get("inference_config", {})
        else:
            inference_config = data.get("inference_config", {})

        return cls(
            model=inference_config.get("model", "nanobanana_pro"),
            steps=inference_config.get("steps", 8),
            cfg_scale=inference_config.get("cfg_scale", 3.5),
            batch_size=inference_config.get("batch_size", 20),
            aspect_ratio=inference_config.get("aspect_ratio", "3:4"),
            guidance=inference_config.get("guidance", "perspective_aware"),
        )


@dataclass
class BackgroundPrompt:
    """
    Background generation prompt with perspective injection.

    Attributes:
        base_prompt: Base prompt from master_blueprint
        perspective: Camera perspective (high_angle or eye_level)
        negative_prompt: Negative prompt
    """
    base_prompt: str
    perspective: PerspectiveType
    negative_prompt: str = "cartoon, illustration, text, watermark, distorted"

    def __post_init__(self):
        """Convert string perspective to PerspectiveType enum if needed."""
        if isinstance(self.perspective, str):
            self.perspective = PerspectiveType(self.perspective)

    def build_final_prompt(self) -> str:
        """
        Build final prompt with perspective injection.

        Adds perspective-specific modifiers to ensure generated background
        matches product camera angle.

        Returns:
            Final prompt string with perspective injection
        """
        # Perspective-specific modifiers
        perspective_modifiers = {
            PerspectiveType.HIGH_ANGLE: ", top-down view, looking down, 45-degree angle",
            PerspectiveType.EYE_LEVEL: ", eye-level view, straight-on angle, horizontal perspective",
        }

        modifier = perspective_modifiers.get(self.perspective, "")
        final_prompt = f"{self.base_prompt}{modifier}"

        logger.debug(f"Built prompt with perspective: {self.perspective.value}")
        return final_prompt


@dataclass
class GeneratedBackground:
    """
    Result from background generation.

    Attributes:
        image: Generated image data (PIL Image or bytes)
        prompt: Prompt used for generation
        index: Index in batch
        perspective: Perspective type used
    """
    image: Any  # PIL.Image.Image or bytes
    prompt: str
    index: int
    perspective: PerspectiveType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string perspective to PerspectiveType enum if needed."""
        if isinstance(self.perspective, str):
            self.perspective = PerspectiveType(self.perspective)


class NanoBackgroundGenerator:
    """
    Generate backgrounds using NanoBanana Pro with perspective awareness.

    Usage:
        generator = NanoBackgroundGenerator()
        results = generator.generate_from_blueprint(
            blueprint=master_blueprint,
            perspective=PerspectiveType.EYE_LEVEL,
            output_dir=output_path
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize background generator.

        Args:
            api_key: Optional API key for image generation service
            config: Optional generation config (uses defaults if None)
        """
        self.api_key = api_key
        self.config = config or GenerationConfig()

    def generate_from_blueprint(
        self,
        blueprint: Dict[str, Any],
        perspective: PerspectiveType,
        output_dir: Optional[Path] = None,
        save_images: bool = True,
    ) -> List[GeneratedBackground]:
        """
        Generate backgrounds from master blueprint configuration.

        Args:
            blueprint: Master blueprint dict from Ad Miner
                Should contain nano_generation_rules section
            perspective: Product camera perspective (from ProductPreprocessor)
            output_dir: Optional output directory for generated images
            save_images: Whether to save images to disk

        Returns:
            List of GeneratedBackground objects
        """
        # Extract generation rules
        nano_rules = blueprint.get("nano_generation_rules", {})
        config = GenerationConfig.from_dict(blueprint)

        # Build prompt with perspective injection
        prompt_template = nano_rules.get("prompt_template_structure", "")
        prompt_slots = nano_rules.get("prompt_slots", {})

        # Fill prompt slots
        base_prompt = self._fill_prompt_template(prompt_template, prompt_slots)
        negative_prompt = nano_rules.get("negative_prompt", "")

        # Create background prompt with perspective
        bg_prompt = BackgroundPrompt(
            base_prompt=base_prompt,
            perspective=perspective,
            negative_prompt=negative_prompt,
        )

        # Generate batch
        results = self.generate_batch(
            prompt=bg_prompt,
            config=config,
            output_dir=output_dir,
            save_images=save_images,
        )

        # Get perspective value safely
        if isinstance(perspective, str):
            persp_value = perspective
        else:
            persp_value = perspective.value

        logger.info(
            f"Generated {len(results)} backgrounds with perspective: {persp_value}"
        )

        return results

    def _fill_prompt_template(
        self,
        template: str,
        slots: Dict[str, str],
    ) -> str:
        """
        Fill prompt template with slot values.

        Args:
            template: Prompt template string with {slot} placeholders
            slots: Dict of slot_name → value

        Returns:
            Filled prompt string
        """
        try:
            return template.format(**slots)
        except KeyError as e:
            logger.warning(f"Missing prompt slot: {e}, using template as-is")
            return template

    def generate_batch(
        self,
        prompt: BackgroundPrompt,
        config: Optional[GenerationConfig] = None,
        output_dir: Optional[Path] = None,
        save_images: bool = True,
    ) -> List[GeneratedBackground]:
        """
        Generate a batch of background images.

        Args:
            prompt: BackgroundPrompt with perspective injection
            config: Optional generation config (uses instance config if None)
            output_dir: Optional output directory
            save_images: Whether to save images to disk

        Returns:
            List of GeneratedBackground objects

        Note:
            This is a placeholder implementation. The actual image generation
            API integration should be implemented based on your specific API.
        """
        gen_config = config or self.config

        # Build final prompt with perspective injection
        final_prompt = prompt.build_final_prompt()

        logger.info(f"Generating {gen_config.batch_size} backgrounds...")
        logger.debug(f"Prompt: {final_prompt[:200]}...")
        logger.debug(f"Config: model={gen_config.model}, steps={gen_config.steps}, cfg={gen_config.cfg_scale}")

        # Placeholder: In actual implementation, call image generation API here
        # Example API call structure:
        # response = api.generate(
        #     prompt=final_prompt,
        #     negative_prompt=prompt.negative_prompt,
        #     num_images=gen_config.batch_size,
        #     steps=gen_config.steps,
        #     cfg_scale=gen_config.cfg_scale,
        #     aspect_ratio=gen_config.aspect_ratio,
        # )

        # For now, return empty list (implement actual API call)
        logger.warning(
            "Background generation is a placeholder. "
            "Implement actual API integration with NanoBanana Pro."
        )

        results: List[GeneratedBackground] = []

        # TODO: Implement actual API call
        # Pseudo-code:
        # for i in range(gen_config.batch_size):
        #     image = response.images[i]
        #     result = GeneratedBackground(
        #         image=image,
        #         prompt=final_prompt,
        #         index=i,
        #         perspective=prompt.perspective,
        #     )
        #     results.append(result)
        #
        #     if save_images and output_dir:
        #         image_path = output_dir / f"background_{i+1:03d}.png"
        #         image.save(image_path)

        return results

    def generate_single(
        self,
        prompt: str,
        perspective: PerspectiveType = PerspectiveType.EYE_LEVEL,
        negative_prompt: str = "cartoon, illustration, text, watermark",
        config: Optional[GenerationConfig] = None,
    ) -> Optional[GeneratedBackground]:
        """
        Generate a single background image.

        Args:
            prompt: Base prompt string
            perspective: Camera perspective
            negative_prompt: Negative prompt
            config: Optional generation config

        Returns:
            GeneratedBackground or None if generation failed
        """
        bg_prompt = BackgroundPrompt(
            base_prompt=prompt,
            perspective=perspective,
            negative_prompt=negative_prompt,
        )

        results = self.generate_batch(
            prompt=bg_prompt,
            config=config,
            save_images=False,
        )

        return results[0] if results else None


# Convenience functions
def generate_backgrounds_from_blueprint(
    blueprint: Dict[str, Any],
    perspective: PerspectiveType,
    output_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    save_images: bool = True,
) -> List[GeneratedBackground]:
    """
    Convenience function for quick background generation from blueprint.

    Args:
        blueprint: Master blueprint dict
        perspective: Product camera perspective
        output_dir: Optional output directory
        api_key: Optional API key
        save_images: Whether to save images to disk

    Returns:
        List of GeneratedBackground objects

    Example:
        results = generate_backgrounds_from_blueprint(
            blueprint=master_blueprint,
            perspective=PerspectiveType.EYE_LEVEL,
            output_dir=Path("backgrounds")
        )
    """
    generator = NanoBackgroundGenerator(api_key=api_key)
    return generator.generate_from_blueprint(
        blueprint=blueprint,
        perspective=perspective,
        output_dir=output_dir,
        save_images=save_images,
    )
