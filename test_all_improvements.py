#!/usr/bin/env python3
"""
End-to-End Test: All Ad Creative Improvements

Tests all three priority improvements:
1. Logo Deformation Prevention (anti-hallucination + logo reference locking)
2. Product Description Overlay (TextExtractor + TextOverlayManager)
3. Background Library Expansion (surface_material-based background selection)

This test demonstrates:
- Multi-image generation with angle-aware selection
- Background reference selection based on surface_material
- Text extraction and overlay
- Full pipeline integration
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from meta.ad.generator.core.generation.generator import ImageGenerator
from meta.ad.generator.core.generation.reference_image_manager import ReferenceImageManager
from meta.ad.generator.core.generation.text_overlay_manager import TextOverlayManager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prompts() -> List[Dict[str, Any]]:
    """Load test prompts from prompts.yaml."""
    import yaml

    prompts_file = Path("results/moprobo/meta/ad_miner/prompts.yaml")
    if not prompts_file.exists():
        logger.error(f"Prompts file not found: {prompts_file}")
        return []

    with open(prompts_file, 'r') as f:
        data = yaml.safe_load(f)

    # Flatten nested structure
    all_prompts = []
    prompts_by_category = data.get('prompts', {})

    for category, prompts in prompts_by_category.items():
        if isinstance(prompts, list):
            all_prompts.extend(prompts)

    logger.info(f"Loaded {len(all_prompts)} prompts from {prompts_file}")
    return all_prompts


def test_reference_image_manager():
    """Test ReferenceImageManager with angle-aware selection."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Reference Image Manager - Angle-Aware Selection")
    logger.info("="*80)

    # Initialize manager
    reference_dir = Path("config/moprobo/product")
    background_dir = Path("config/moprobo/backgrounds")

    manager = ReferenceImageManager(
        reference_images_dir=reference_dir,
        background_dir=background_dir,
    )

    # Test different camera angles
    test_cases = [
        {
            "name": "45-degree with Marble",
            "camera_angle": "45-degree",
            "surface_material": "Marble",
            "expected_product_images": 2,
            "expected_background": True,
        },
        {
            "name": "Eye-Level Shot with Wood",
            "camera_angle": "Eye-Level Shot",
            "surface_material": "Wood",
            "expected_product_images": 2,
            "expected_background": True,
        },
        {
            "name": "High-Angle Shot with Concrete",
            "camera_angle": "High-Angle Shot",
            "surface_material": "Concrete",
            "expected_product_images": 2,
            "expected_background": True,
        },
        {
            "name": "Top-Down (no surface_material)",
            "camera_angle": "Top-Down",
            "surface_material": None,
            "expected_product_images": 2,
            "expected_background": False,
        },
    ]

    results = []
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        logger.info(f"  Camera Angle: {test_case['camera_angle']}")
        logger.info(f"  Surface Material: {test_case['surface_material']}")

        selected = manager.select_images_for_angle(
            camera_angle=test_case['camera_angle'],
            surface_material=test_case['surface_material'],
            max_images=3,
        )

        logger.info(f"  Selected {len(selected)} images:")
        for img_path in selected:
            logger.info(f"    - {img_path.name}")

        # Verify expectations
        product_images = [p for p in selected if p.parent == reference_dir]
        background_images = [p for p in selected if background_dir in p.parents]

        logger.info(f"  Product images: {len(product_images)} (expected: {test_case['expected_product_images']})")
        logger.info(f"  Background images: {len(background_images)} (expected: {1 if test_case['expected_background'] else 0})")

        passed = (
            len(product_images) == test_case['expected_product_images'] and
            (len(background_images) > 0) == test_case['expected_background']
        )

        results.append({
            "test": test_case['name'],
            "passed": passed,
            "total_images": len(selected),
            "product_images": len(product_images),
            "background_images": len(background_images),
        })

    # Summary
    logger.info("\n" + "-"*80)
    logger.info("Reference Image Manager Test Summary:")
    passed_count = sum(1 for r in results if r['passed'])
    logger.info(f"  Passed: {passed_count}/{len(results)}")

    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        logger.info(f"  {status}: {result['test']} ({result['total_images']} images)")

    return all(r['passed'] for r in results)


def test_text_extraction():
    """Test TextExtractor and TextOverlayManager."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Text Extraction and Overlay")
    logger.info("="*80)

    # Initialize manager
    template_path = Path("config/moprobo/meta/text_overlay_templates.yaml")
    overlay_manager = TextOverlayManager(
        config_path=template_path,
        enabled=True,
    )

    # Test prompts with different features
    test_cases = [
        {
            "name": "Marble + Window Light",
            "prompt": "A professional product photograph showing the mop in detail. Materials include microfiber and plastic.",
            "features": {
                "surface_material": "Marble",
                "lighting_style": "Window Light",
                "camera_angle": "45-degree",
            },
        },
        {
            "name": "Wood + Natural Light",
            "prompt": "Professional product photography showing cleaning tool with absorbent head and handle.",
            "features": {
                "surface_material": "Wood",
                "lighting_style": "Natural Light",
                "camera_angle": "Eye-Level Shot",
            },
        },
    ]

    results = []
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        logger.info(f"  Features: {test_case['features']}")

        # Extract and format text
        formatted = overlay_manager.extract_and_format(
            prompt=test_case['prompt'],
            features=test_case['features'],
            template="minimal",
        )

        logger.info(f"  Extracted:")
        logger.info(f"    Headline: {formatted.get('headline', 'N/A')}")
        logger.info(f"    Features: {formatted.get('features', [])}")
        logger.info(f"    Template: {formatted.get('template', 'N/A')}")

        passed = bool(formatted.get('headline'))
        results.append({
            "test": test_case['name'],
            "passed": passed,
            "headline": formatted.get('headline', 'N/A'),
        })

    # Summary
    logger.info("\n" + "-"*80)
    logger.info("Text Extraction Test Summary:")
    passed_count = sum(1 for r in results if r['passed'])
    logger.info(f"  Passed: {passed_count}/{len(results)}")

    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        logger.info(f"  {status}: {result['test']} - Headline: {result['headline']}")

    return all(r['passed'] for r in results)


def test_multi_image_generation():
    """Test full generation pipeline with all improvements."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Full Generation Pipeline with All Improvements")
    logger.info("="*80)

    # Load prompts
    prompts = load_prompts()
    if not prompts:
        logger.error("No prompts loaded, skipping generation test")
        return False

    # Select test prompts with different features
    test_prompts = [
        prompts[0],  # Top combination: Marble + Window Light + 45-degree
        prompts[4],  # Individual feature: Window Light + 45-degree
    ]

    logger.info(f"\nTesting {len(test_prompts)} generations with different feature combinations")

    # Initialize generator
    generator = ImageGenerator(
        reference_images_dir=Path("config/moprobo/product"),
        background_dir=Path("config/moprobo/backgrounds"),
        enable_multi_image=True,
    )

    results = []
    output_dir = Path("results/test_improvements")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt_data in enumerate(test_prompts):
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {i+1}/{len(test_prompts)}: {prompt_data.get('prompt_name', 'Unknown')}")
        logger.info(f"{'='*60}")

        # Extract prompt info
        nano_prompt = prompt_data.get('nano_prompt', '')
        features_used = prompt_data.get('features_used', {})
        roas_lift = prompt_data.get('roas_lift', 0)

        logger.info(f"  ROAS Lift: {roas_lift}x")
        logger.info(f"  Features: {features_used}")

        # Generate with all improvements
        start_time = time.time()
        result = generator.generate(
            prompt=nano_prompt,
            source_image_path=str(Path("config/moprobo/product/正面.png")),
            output_filename=f"test_improvement_{i+1}.png",
            camera_angle=features_used.get('camera_angle'),
            features=features_used,
        )
        generation_time = time.time() - start_time

        # Log results
        logger.info(f"\n  Generation Results:")
        logger.info(f"    Success: {result.get('success')}")
        logger.info(f"    Time: {generation_time:.1f}s")
        logger.info(f"    Image Path: {result.get('image_path', 'N/A')}")
        logger.info(f"    Image Size: {result.get('image_size_mb', 'N/A')} MB")

        if result.get('reference_images_used'):
            logger.info(f"    Reference Images: {len(result['reference_images_used'])}")
            for ref in result['reference_images_used']:
                logger.info(f"      - {Path(ref).name}")

        # Check for improvements
        improvements = []
        if 'ANTI-HALLUCINATION' in nano_prompt:
            improvements.append("Anti-hallucination constraints")

        if result.get('reference_images_used'):
            if len(result['reference_images_used']) > 1:
                improvements.append(f"Multi-image ({len(result['reference_images_used'])} refs)")

        passed = result.get('success', False)
        results.append({
            "test": prompt_data.get('prompt_name', f'Test {i+1}'),
            "passed": passed,
            "time": generation_time,
            "improvements": improvements,
            "roas_lift": roas_lift,
        })

    # Summary
    logger.info("\n" + "-"*80)
    logger.info("Full Generation Pipeline Test Summary:")
    passed_count = sum(1 for r in results if r['passed'])
    logger.info(f"  Passed: {passed_count}/{len(results)}")

    for result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        logger.info(f"  {status}: {result['test']}")
        logger.info(f"         ROAS: {result['roas_lift']}x | Time: {result['time']:.1f}s")
        logger.info(f"         Improvements: {', '.join(result['improvements'])}")

    return all(r['passed'] for r in results)


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("END-TO-END TEST: ALL AD CREATIVE IMPROVEMENTS")
    logger.info("="*80)
    logger.info("\nTesting 3 Priority Improvements:")
    logger.info("  1. Logo Deformation Prevention (anti-hallucination + logo reference)")
    logger.info("  2. Product Description Overlay (TextExtractor + TextOverlay)")
    logger.info("  3. Background Library Expansion (surface_material-based selection)")
    logger.info("")

    results = {}

    # Test 1: Reference Image Manager
    results['reference_manager'] = test_reference_image_manager()

    # Test 2: Text Extraction
    results['text_extraction'] = test_text_extraction()

    # Test 3: Full Generation Pipeline
    results['full_pipeline'] = test_multi_image_generation()

    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✓ ALL TESTS PASSED - All improvements working correctly!")
    else:
        logger.info("\n✗ SOME TESTS FAILED - Review logs above for details")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
