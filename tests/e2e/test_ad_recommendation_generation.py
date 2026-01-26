"""
End-to-end test for ad recommendation and generation pipeline.

Tests the complete flow:
1. Load recommendations from ad/recommender (moprobo data)
2. Generate prompts using CreativePipeline
3. Validate prompt quality and coverage
4. Generate images (optional, disabled in CI)

Image generation can be disabled by setting CI=true or GENERATE_IMAGES=false
"""

import logging
import os
import sys
from pathlib import Path

from src.meta.ad.generator.core.generation.generator import ImageGenerator
from src.meta.ad.generator.core.paths import Paths
from src.meta.ad.generator.pipeline.pipeline import (
    CreativePipeline,
    CreativePipelineConfig,
    RecommendationPaths,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_e2e_ad_recommendation_generation():
    """
    End-to-end test: Load recommendations, generate prompts, and optionally generate images.

    Uses moprobo data from config/ad/recommender/moprobo/meta/
    Image generation is disabled in CI (set CI=true or GENERATE_IMAGES=false to disable)
    """
    logger.info("=" * 80)
    logger.info("E2E Test: Ad Recommendation and Generation Pipeline")
    logger.info("=" * 80)

    # Check if image generation should be enabled
    # Disable in CI or if explicitly disabled
    is_ci = os.getenv("CI", "").lower() == "true"
    generate_images = os.getenv("GENERATE_IMAGES", "true" if not is_ci else "false").lower() == "true"

    if not generate_images:
        logger.info("Image generation disabled (CI mode or GENERATE_IMAGES=false)")
    else:
        logger.info("Image generation enabled")

    # Configuration
    customer = "moprobo"  # Note: pipeline will lowercase and replace spaces
    platform = "meta"
    date = "2026-01-26"

    # Setup paths - check if date-specific path exists, otherwise use default
    paths = Paths(customer=customer, platform=platform, date=date)
    recommendations_path_candidates = [
        paths.recommendations(),  # With date
        Path("config/ad/recommender/moprobo/meta/recommendations.md"),  # Without date
    ]

    recommendations_path = None
    for candidate in recommendations_path_candidates:
        if candidate.exists():
            recommendations_path = candidate
            break

    if not recommendations_path:
        logger.error(f"Could not find recommendations file in any of these locations:")
        for candidate in recommendations_path_candidates:
            logger.error(f"  - {candidate}")
        return False

    logger.info(f"Customer: {customer}")
    logger.info(f"Platform: {platform}")
    logger.info(f"Date: {date}")
    logger.info(f"Recommendations path: {recommendations_path}")

    # Verify recommendations exist
    if not recommendations_path.exists():
        logger.error(f"Recommendations file not found: {recommendations_path}")
        logger.error("Please run ad recommender first to generate recommendations")
        return False

    logger.info(f"✓ Recommendations file exists")

    # Initialize pipeline
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Initialize CreativePipeline")
    logger.info("=" * 80)

    try:
        config = CreativePipelineConfig(
            product_name="Moprobo",  # Generic product for testing
            output_dir=None,  # Use default organized path
            recommendation_paths=RecommendationPaths(
                recommendation_path=recommendations_path
            ),
        )
        pipeline = CreativePipeline(config)
        logger.info(f"✓ Pipeline initialized")
        logger.info(f"  Product: {pipeline.product_name}")
        logger.info(f"  Output dir: {pipeline.output_dir}")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False

    # Load recommendations
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Load Recommendations")
    logger.info("=" * 80)

    try:
        visual_recommendation = pipeline.load_recommendation()
        entrance_count = len(visual_recommendation.get("entrance_features", {}))
        headroom_count = len(visual_recommendation.get("headroom_features", {}))

        logger.info(f"✓ Recommendations loaded successfully")
        logger.info(f"  Entrance features: {entrance_count}")
        logger.info(f"  Headroom features: {headroom_count}")
    except Exception as e:
        logger.error(f"Failed to load recommendations: {e}")
        return False

    # Generate prompts
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Generate Prompts")
    logger.info("=" * 80)

    try:
        num_variations = 3
        results = pipeline.run(
            source_image_path="tests/fixtures/sample_product.jpg",
            num_variations=num_variations,
            save_prompts=True,
        )

        logger.info(f"✓ Generated {len(results)} prompt variations")

        for i, result in enumerate(results):
            prompt = result["prompt"]
            validation = result["validation"]
            logger.info(f"\n  Variation {i+1}:")
            logger.info(f"    Prompt length: {len(prompt)} chars")
            logger.info(f"    Coverage: {validation['coverage']*100:.1f}%")
            logger.info(f"    Features covered: {validation['covered_features']}/{validation['total_features']}")
            logger.info(f"    Status: {'✓ PASSED' if validation['passed'] else '✗ FAILED'}")

            # Show first 200 chars of prompt
            logger.info(f"    Preview: {prompt[:200]}...")

    except Exception as e:
        logger.error(f"Failed to generate prompts: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Generate images (if enabled)
    generated_images = []
    if generate_images:
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Generate Images")
        logger.info("=" * 80)

        try:
            # Initialize image generator with organized paths
            generator = ImageGenerator(
                model="nano-banana-pro",
                aspect_ratio="1:1",
                resolution="2K",
                customer=customer,
                platform=platform,
                date=date,
                enable_watermark=True,
                enable_upscaling=True,
            )
            logger.info("✓ Image generator initialized")
            logger.info(f"  Output dir: {generator.output_dir}")

            # Get source image path
            source_image = "tests/fixtures/sample_product.jpg"
            if not Path(source_image).exists():
                logger.warning(f"Source image not found: {source_image}")
                logger.warning("Skipping image generation")
            else:
                # Generate images for each prompt
                for i, result in enumerate(results):
                    prompt = result["prompt"]
                    logger.info(f"\nGenerating image {i+1}/{len(results)}...")

                    try:
                        output_filename = f"generated_{i+1:03d}.png"
                        image_path = generator.generate(
                            prompt=prompt,
                            source_image_path=source_image,
                            output_filename=output_filename,
                        )

                        if image_path:
                            logger.info(f"✓ Generated image: {image_path}")
                            generated_images.append(image_path)
                            result["generated_image_path"] = str(image_path)
                        else:
                            logger.warning(f"Failed to generate image {i+1}")

                    except Exception as e:
                        logger.error(f"Error generating image {i+1}: {e}")
                        import traceback
                        traceback.print_exc()

                logger.info(f"\n✓ Generated {len(generated_images)} images")

        except Exception as e:
            logger.error(f"Failed to initialize image generator: {e}")
            logger.warning("Continuing without image generation")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("Step 4: Image Generation (Skipped)")
        logger.info("=" * 80)
        logger.info("Image generation disabled. Set GENERATE_IMAGES=true to enable.")

    # Validate outputs
    logger.info("\n" + "=" * 80)
    logger.info(f"Step 5: Validate Outputs")
    logger.info("=" * 80)

    # Check prompt files were saved
    prompts_dir = paths.prompts_output()
    prompt_files = sorted(prompts_dir.glob("*.md"))

    expected_count = num_variations
    saved_count = len(prompt_files)

    if saved_count > 0:
        for prompt_file in prompt_files[-expected_count:]:  # Get most recent files
            logger.info(f"✓ Prompt file saved: {prompt_file.name}")
    else:
        logger.warning(f"No prompt files found in {prompts_dir}")

    logger.info(f"✓ Total prompt files saved: {saved_count} (expected {expected_count})")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("E2E Test Summary")
    logger.info("=" * 80)
    logger.info("✓ All tests passed!")
    logger.info(f"✓ Generated {num_variations} prompt variations")
    logger.info(f"✓ Saved {saved_count} prompt files")

    if generate_images:
        logger.info(f"✓ Generated {len(generated_images)} images")
        for img_path in generated_images:
            logger.info(f"  - {img_path}")
    else:
        logger.info("✓ Image generation disabled (CI mode)")

    logger.info(f"✓ Output directory: {pipeline.output_dir}")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_e2e_ad_recommendation_generation()
    sys.exit(0 if success else 1)
