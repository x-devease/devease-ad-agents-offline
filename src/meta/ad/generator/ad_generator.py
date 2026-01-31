"""
Ad Generator - V1.0

Generates ad creatives using mined patterns and NanoBanana Pro.

Pipeline:
1. Read patterns.yaml from ad miner
2. Generate prompts using PromptBuilder
3. Save prompts.yaml for reference
4. Generate images using NanoBanana Pro
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from .prompt_builder import PromptBuilder

# Import ImageGenerator for NanoBanana Pro integration
try:
    from .core.generation.generator import ImageGenerator
    IMAGE_GENERATOR_AVAILABLE = True
except ImportError:
    IMAGE_GENERATOR_AVAILABLE = False
    ImageGenerator = None

logger = logging.getLogger(__name__)


class AdGenerator:
    """
    Ad Generator orchestrates prompt generation and image creation.

    Reads mined patterns from ad miner and generates ad creatives.
    """

    def __init__(
        self,
        customer: str,
        platform: str,
        patterns_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize ad generator.

        Args:
            customer: Customer name (e.g., "moprobo")
            platform: Platform name (e.g., "meta")
            patterns_path: Path to patterns.yaml (default: results/{customer}/{platform}/ad_miner/patterns.yaml)
            output_dir: Base output directory (default: results/{customer}/{platform})
        """
        self.customer = customer
        self.platform = platform

        # Set paths
        if output_dir is None:
            output_dir = Path(f"results/{customer}/{platform}")
        self.output_dir = Path(output_dir)

        if patterns_path is None:
            patterns_path = self.output_dir / "ad_miner" / "patterns.yaml"
        self.patterns_path = Path(patterns_path)

        # Output paths
        self.prompts_path = self.output_dir / "ad_miner" / "prompts.yaml"
        self.creatives_dir = self.output_dir / "ad" / "creatives"

        # Initialize PromptBuilder (will be loaded when patterns are available)
        self.prompt_builder = None

        logger.info(f"AdGenerator initialized:")
        logger.info(f"  Customer: {customer}")
        logger.info(f"  Platform: {platform}")
        logger.info(f"  Patterns: {self.patterns_path}")
        logger.info(f"  Prompts output: {self.prompts_path}")
        logger.info(f"  Creatives output: {self.creatives_dir}")

    def load_patterns(self) -> bool:
        """
        Load patterns from patterns.yaml and initialize PromptBuilder.

        Returns:
            True if patterns loaded successfully, False otherwise
        """
        if not self.patterns_path.exists():
            logger.error(f"Patterns file not found: {self.patterns_path}")
            logger.error("Run ad miner first to generate patterns.yaml")
            return False

        try:
            self.prompt_builder = PromptBuilder(self.patterns_path)
            logger.info(f"✓ Loaded patterns from {self.patterns_path}")

            # Log pattern summary
            with open(self.patterns_path, 'r') as f:
                patterns = yaml.safe_load(f)

            comb_count = len(patterns.get("combinatorial_patterns", []))
            ind_count = len(patterns.get("individual_features", []))
            psych_count = len(patterns.get("psychology_patterns", []))
            anti_count = len(patterns.get("anti_patterns", []))

            logger.info(f"  Combinatorial patterns: {comb_count}")
            logger.info(f"  Individual features: {ind_count}")
            logger.info(f"  Psychology patterns: {psych_count}")
            logger.info(f"  Anti-patterns: {anti_count}")

            return True

        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return False

    def generate_prompts(
        self,
        max_supporting: int = 3,
        max_individual: int = 5,
        max_psychology: int = 2
    ) -> Dict[str, List[Dict]]:
        """
        Generate all prompts from patterns.

        Args:
            max_supporting: Maximum supporting combination prompts
            max_individual: Maximum individual feature prompts
            max_psychology: Maximum psychology prompts

        Returns:
            Dict with all generated prompts
        """
        if self.prompt_builder is None:
            logger.error("PromptBuilder not initialized. Call load_patterns() first.")
            return {}

        logger.info("Generating prompts from patterns...")

        all_prompts = self.prompt_builder.build_all_prompts()

        # Log summary
        total = sum(len(v) for v in all_prompts.values())
        logger.info(f"✓ Generated {total} prompts:")
        for category, prompts in all_prompts.items():
            logger.info(f"  {category}: {len(prompts)} prompts")

        return all_prompts

    def save_prompts(self, prompts: Dict[str, List[Dict]]) -> Path:
        """
        Save generated prompts to prompts.yaml.

        Args:
            prompts: Dict of generated prompts

        Returns:
            Path to saved prompts.yaml
        """
        # Create output directory if needed
        self.prompts_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare output data
        output_data = {
            "metadata": {
                "customer": self.customer,
                "platform": self.platform,
                "source_patterns": str(self.patterns_path),
                "total_prompts": sum(len(v) for v in prompts.values()),
                "categories": list(prompts.keys())
            },
            "prompts": prompts
        }

        # Save to YAML
        with open(self.prompts_path, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"✓ Saved prompts to {self.prompts_path}")
        logger.info(f"  File size: {self.prompts_path.stat().st_size} bytes")

        return self.prompts_path

    def generate_images(
        self,
        prompts: Dict[str, List[Dict]],
        batch_size: int = 20,
        num_images_per_prompt: int = 1
    ) -> Dict[str, List[Path]]:
        """
        Generate images using NanoBanana Pro.

        Args:
            prompts: Dict of generated prompts
            batch_size: Number of images to generate per batch (default from config)
            num_images_per_prompt: Number of images to generate per prompt (default: 1)

        Returns:
            Dict mapping prompt_id to list of generated image paths
        """
        if not IMAGE_GENERATOR_AVAILABLE:
            logger.error("ImageGenerator not available. Install fal-client:")
            logger.error("  pip install fal-client")
            return {}

        logger.info("=" * 80)
        logger.info("GENERATING IMAGES WITH NANO BANANA PRO")
        logger.info("=" * 80)

        # Create creatives directory
        self.creatives_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ImageGenerator
        # Get FAL_KEY from environment or ~/.devease/keys
        import os
        fal_key = os.getenv("FAL_KEY")
        if not fal_key:
            # Try loading from keys file
            from src.meta.ad.miner.utils.api_keys import get_fal_api_key
            fal_key = get_fal_api_key()
            if fal_key:
                os.environ["FAL_KEY"] = fal_key

        if not os.getenv("FAL_KEY"):
            logger.error("FAL_KEY not found in environment or ~/.devease/keys")
            logger.error("Set FAL_KEY in ~/.devease/keys file:")
            logger.error("  FAL_KEY=your_key_here")
            return {}

        try:
            # Pass explicit output_dir to ImageGenerator
            # We want: results/{customer}/{platform}/ad/creatives
            image_gen = ImageGenerator(
                model="nano-banana-pro",
                aspect_ratio="3:4",  # Default from config
                resolution="2K",
                enable_upscaling=False,  # Disable for speed
                enable_watermark=False,  # No watermark for ads
                enable_text_overlay=False,  # Text overlay later
                output_dir=str(self.creatives_dir),  # Use our creatives directory
                strength=0.85,
                guidance_scale=8.0,
                num_inference_steps=30
            )
            logger.info("✓ ImageGenerator initialized")
            logger.info(f"  Output directory: {self.creatives_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ImageGenerator: {e}")
            logger.error("Make sure fal-client is installed: pip install fal-client")
            return {}

        # Generate images for each prompt
        results = {}
        total_generated = 0
        total_failed = 0

        for category, prompt_list in prompts.items():
            logger.info(f"\n📂 Category: {category} ({len(prompt_list)} prompts)")

            for prompt_dict in prompt_list:
                prompt_id = prompt_dict.get("prompt_id", "unknown")
                nano_prompt = prompt_dict.get("nano_prompt", "")

                # Get generation config
                gen_config = prompt_dict.get("generation_config", {})
                aspect_ratio = gen_config.get("aspect_ratio", "3:4")
                num_inference_steps = gen_config.get("steps", 30)
                cfg_scale = gen_config.get("cfg_scale", 3.5)

                logger.info(f"\n🎨 Generating: {prompt_id}")
                logger.info(f"  Prompt: {nano_prompt[:100]}...")
                logger.info(f"  Aspect Ratio: {aspect_ratio}")
                logger.info(f"  Steps: {num_inference_steps}")

                try:
                    # Generate image
                    result = image_gen.generate(
                        prompt=nano_prompt,
                        aspect_ratio=aspect_ratio,
                    )

                    if result and result.get("success"):
                        image_path = result.get("image_path")
                        logger.info(f"  ✓ Generated: {image_path}")

                        # Store result
                        if prompt_id not in results:
                            results[prompt_id] = []
                        results[prompt_id].append(Path(image_path))
                        total_generated += 1
                    else:
                        logger.warning(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                        total_failed += 1

                except Exception as e:
                    logger.error(f"  ✗ Exception: {e}")
                    total_failed += 1
                    continue

        logger.info("\n" + "=" * 80)
        logger.info("IMAGE GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Successfully generated: {total_generated} images")
        if total_failed > 0:
            logger.warning(f"✗ Failed: {total_failed} images")
        logger.info(f"📁 Output directory: {self.creatives_dir}")

        return results

    def run(
        self,
        save_prompts: bool = True,
        generate_images: bool = False
    ) -> Dict[str, Any]:
        """
        Run full ad generation pipeline.

        Args:
            save_prompts: Whether to save prompts to prompts.yaml
            generate_images: Whether to generate actual images (requires FAL_KEY)

        Returns:
            Dict with pipeline results
        """
        logger.info("=" * 80)
        logger.info("AD GENERATOR PIPELINE")
        logger.info("=" * 80)

        # Step 1: Load patterns
        if not self.load_patterns():
            return {"success": False, "error": "Failed to load patterns"}

        # Step 2: Generate prompts
        prompts = self.generate_prompts()
        if not prompts:
            return {"success": False, "error": "Failed to generate prompts"}

        # Step 3: Save prompts
        if save_prompts:
            self.save_prompts(prompts)

        # Step 4: Generate images (optional)
        image_paths = {}
        if generate_images:
            image_paths = self.generate_images(prompts)

        logger.info("=" * 80)
        logger.info("✅ AD GENERATOR PIPELINE COMPLETE")
        logger.info("=" * 80)

        return {
            "success": True,
            "patterns_path": str(self.patterns_path),
            "prompts_path": str(self.prompts_path) if save_prompts else None,
            "creatives_dir": str(self.creatives_dir),
            "total_prompts": sum(len(v) for v in prompts.values()),
            "images_generated": len(image_paths) if generate_images else 0,
            "categories": {k: len(v) for k, v in prompts.items()},
            "image_paths": image_paths
        }


def generate_ads(
    customer: str = "moprobo",
    platform: str = "meta",
    patterns_path: Optional[Path] = None,
    generate_images: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to generate ads from patterns.

    Args:
        customer: Customer name
        platform: Platform name
        patterns_path: Optional path to patterns.yaml
        generate_images: Whether to generate actual images

    Returns:
        Dict with generation results
    """
    generator = AdGenerator(
        customer=customer,
        platform=platform,
        patterns_path=patterns_path
    )

    return generator.run(save_prompts=True, generate_images=generate_images)
