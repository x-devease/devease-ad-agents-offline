#!/usr/bin/env python3
"""
Test multi-image generation with real prompts and reference images.

This script:
1. Loads prompts from results/moprobo/meta/ad_miner/prompts.yaml
2. Initializes ImageGenerator with multi-image enabled
3. Generates creatives for a few prompts with different camera angles
4. Outputs to results/moprobo/meta/ad/creatives/test_multi_image/
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.meta.ad.generator.core.generation.generator import ImageGenerator


def load_prompts():
    """Load prompts from prompts.yaml and flatten by category."""
    prompts_path = Path("results/moprobo/meta/ad_miner/prompts.yaml")
    with open(prompts_path, 'r') as f:
        data = yaml.safe_load(f)

    # Prompts are organized by category
    prompts_by_category = data.get('prompts', {})
    all_prompts = []

    # Flatten into a single list
    for category, prompt_list in prompts_by_category.items():
        for prompt in prompt_list:
            all_prompts.append({
                'category': category,
                **prompt
            })

    return all_prompts


def get_source_image():
    """Get source image path."""
    # Use the front view as source
    return "config/moprobo/product/正面.png"


def main():
    """Main test function."""
    print("=" * 80)
    print("Multi-Image Generation Test")
    print("=" * 80)

    # Load prompts
    print("\n1. Loading prompts from prompts.yaml...")
    prompts = load_prompts()
    print(f"   Loaded {len(prompts)} prompts")

    # Select test prompts with different camera angles
    test_prompts = []
    camera_angles_seen = set()

    for prompt in prompts:
        features = prompt.get('features_used', {})
        camera_angle = features.get('camera_angle')

        # Pick first prompt with each unique camera angle
        if camera_angle and camera_angle not in camera_angles_seen:
            test_prompts.append(prompt)
            camera_angles_seen.add(camera_angle)

        # Stop when we have 3 different angles
        if len(test_prompts) >= 3:
            break

    print(f"\n2. Selected {len(test_prompts)} test prompts with different camera angles:")
    for i, prompt in enumerate(test_prompts):
        features = prompt.get('features_used', {})
        prompt_text = prompt.get('nano_prompt', prompt.get('prompt', ''))
        print(f"   Prompt {i+1}:")
        print(f"     - Category: {prompt.get('category')}")
        print(f"     - Name: {prompt.get('prompt_name', 'N/A')}")
        print(f"     - Camera Angle: {features.get('camera_angle')}")
        print(f"     - Text: {prompt_text[:80]}...")

    # Get source image
    source_image = get_source_image()
    print(f"\n3. Source image: {source_image}")

    # Initialize generator with multi-image enabled
    print("\n4. Initializing ImageGenerator with multi-image enabled...")
    reference_images_dir = "config/moprobo/product"
    output_dir = "results/moprobo/meta/ad/creatives/test_multi_image"

    generator = ImageGenerator(
        model="nano-banana-pro",
        reference_images_dir=reference_images_dir,
        enable_multi_image=True,
        output_dir=output_dir,
        enable_watermark=False,
        enable_upscaling=False,
        use_gpt4o_conversion=False,  # Use prompts as-is
    )

    print(f"   Reference images dir: {reference_images_dir}")
    print(f"   Multi-image enabled: {generator.enable_multi_image}")
    print(f"   Reference manager loaded: {generator.reference_manager is not None}")
    if generator.reference_manager:
        print(f"   Reference images loaded: {len(generator.reference_manager.get_all_image_paths())}")

    # Generate images
    print("\n5. Generating images...")
    print("-" * 80)

    results = []
    for i, prompt_data in enumerate(test_prompts):
        prompt_text = prompt_data.get('nano_prompt', prompt_data.get('prompt', ''))
        features = prompt_data.get('features_used', {})
        camera_angle = features.get('camera_angle')
        category = prompt_data.get('category', 'unknown')
        prompt_name = prompt_data.get('prompt_name', 'N/A')

        print(f"\nGenerating {i+1}/{len(test_prompts)}: {prompt_name}")
        print(f"  Category: {category}")
        print(f"  Camera Angle: {camera_angle}")
        print(f"  Prompt: {prompt_text[:100]}...")

        # Create output filename
        safe_name = prompt_name.replace(' ', '_').replace('/', '_')[:50]
        output_filename = f"test_{i+1}_{safe_name}.jpg"

        # Generate
        try:
            result = generator.generate(
                prompt=prompt_text,
                source_image_path=source_image,
                output_filename=output_filename,
                camera_angle=camera_angle,
            )

            if result.get('success'):
                image_path = result.get('image_path')
                print(f"  ✓ Success: {image_path}")
                results.append({
                    'prompt': prompt_data,
                    'result': result,
                    'status': 'success'
                })
            else:
                error = result.get('error', 'Unknown error')
                print(f"  ✗ Failed: {error}")
                results.append({
                    'prompt': prompt_data,
                    'result': result,
                    'status': 'failed'
                })
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'prompt': prompt_data,
                'result': None,
                'status': 'error',
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 80)
    print("Generation Summary")
    print("=" * 80)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"Total attempted: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated images:")
    for i, r in enumerate(results):
        if r['status'] == 'success':
            image_path = r['result'].get('image_path')
            features = r['prompt'].get('features_used', {})
            print(f"  {i+1}. {image_path}")
            print(f"     Camera Angle: {features.get('camera_angle')}")
            print(f"     Category: {r['prompt'].get('category')}")
            print(f"     Name: {r['prompt'].get('prompt_name')}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
