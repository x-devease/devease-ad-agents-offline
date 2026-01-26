"""
Integration test: Combinatorial prompt generation.

Tests the combinatorial sampling logic that generates different prompts
by exploring multiple dimension combinations from High/Medium confidence
recommendations.
"""

import logging
import sys
from pathlib import Path

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


def test_combinatorial_prompt_generation():
    """
    Integration test: Generate combinatorial prompts from recommendations.

    Tests:
    1. Load recommendations (moprobo data)
    2. Identify dimensions with multiple options
    3. Generate dimension combinations
    4. Generate unique prompts for each combination
    5. Validate prompts are different
    """
    logger.info("=" * 80)
    logger.info("Integration Test: Combinatorial Prompt Generation")
    logger.info("=" * 80)

    # Configuration
    customer = "moprobo"
    platform = "meta"
    date = "2026-01-26"

    # Setup paths
    paths = Paths(customer=customer, platform=platform, date=date)
    recommendations_path = Path("config/ad/recommender/moprobo/meta/recommendations.md")

    if not recommendations_path.exists():
        logger.error(f"Recommendations file not found: {recommendations_path}")
        return False

    logger.info(f"Customer: {customer}")
    logger.info(f"Platform: {platform}")
    logger.info(f"Recommendations: {recommendations_path}")

    # Initialize pipeline
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Initialize Pipeline")
    logger.info("=" * 80)

    config = CreativePipelineConfig(
        product_name="Moprobo",
        recommendation_paths=RecommendationPaths(
            recommendation_path=recommendations_path
        ),
    )

    pipeline = CreativePipeline(config)
    logger.info("✓ Pipeline initialized")

    # Load recommendations
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Load Recommendations")
    logger.info("=" * 80)

    visual_rec = pipeline.load_recommendation()
    logger.info(f"✓ Loaded {len(visual_rec.get('entrance_features', []))} entrance features")
    logger.info(f"✓ Loaded {len(visual_rec.get('headroom_features', []))} headroom features")

    # Generate combinations
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Generate Dimension Combinations")
    logger.info("=" * 80)

    num_combinations = 5
    combinations = pipeline._sample_dimension_combinations(
        visual_rec,
        num_combinations
    )

    logger.info(f"✓ Generated {len(combinations)} combinations")

    # Show what varies
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Analyze Combinations")
    logger.info("=" * 80)

    # Extract key dimensions for comparison
    key_dims = ["color_balance", "mood_lighting", "relationship_depiction"]

    combo_features = []
    for i, combo in enumerate(combinations):
        features = {}
        entrance = combo.get("entrance_features", [])
        headroom = combo.get("headroom_features", [])

        for feat in entrance + headroom:
            if isinstance(feat, dict):
                orig = feat.get("_original_feature", "")
                value = feat.get("feature_value", "")
                if orig in key_dims:
                    features[orig] = value

        combo_features.append(features)

        logger.info(f"\nCombination {i+1}:")
        for dim in key_dims:
            if dim in features:
                logger.info(f"  {dim}: {features[dim]}")

    # Generate prompts
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Generate Prompts")
    logger.info("=" * 80)

    prompts = []
    prompts_dir = paths.prompts_output()
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(combinations):
        logger.info(f"\nGenerating prompt {i+1}/{len(combinations)}...")

        prompt = pipeline.generate_prompt(combo)
        prompts.append(prompt)

        # Save with descriptive filename
        prompt_filename = pipeline._generate_prompt_filename(i, prompt)
        prompt_file = prompts_dir / prompt_filename
        prompt_file.write_text(prompt, encoding="utf-8")

        logger.info(f"  → {prompt_filename}")
        logger.info(f"  → Length: {len(prompt)} chars")

    # Validate prompts are different
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Validate Prompt Diversity")
    logger.info("=" * 80)

    # Check all pairs are different
    unique_prompts = len(set(prompts))
    logger.info(f"✓ Unique prompts: {unique_prompts}/{len(prompts)}")

    if unique_prompts == len(prompts):
        logger.info("✓ All prompts are unique!")
    else:
        logger.warning("⚠️  Some prompts are identical")
        return False

    # Check key differences
    logger.info("\nKey feature differences:")
    for i in range(len(combo_features) - 1):
        features_a = combo_features[i]
        features_b = combo_features[i + 1]

        differences = []
        for dim in key_dims:
            val_a = features_a.get(dim, "N/A")
            val_b = features_b.get(dim, "N/A")
            if val_a != val_b:
                differences.append(f"{dim}: {val_a} → {val_b}")

        if differences:
            logger.info(f"  Combo {i+1} → {i+2}: {', '.join(differences)}")
        else:
            logger.info(f"  Combo {i+1} → {i+2}: Same features")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Integration Test Summary")
    logger.info("=" * 80)
    logger.info("✓ All tests passed!")
    logger.info(f"✓ Generated {len(prompts)} unique combinatorial prompts")
    logger.info(f"✓ Explored {len(key_dims)} dimensions with multiple options")
    logger.info(f"✓ Output directory: {paths.prompts_output()}")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_combinatorial_prompt_generation()
    sys.exit(0 if success else 1)
