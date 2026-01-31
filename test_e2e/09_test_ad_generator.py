#!/usr/bin/env python3
"""
TEST: Ad Generator

Tests the ad generator pipeline from patterns to prompts.

Usage:
  python3 test_e2e/09_test_ad_generator.py           # Generate prompts only
  python3 test_e2e/09_test_ad_generator.py --images  # Generate prompts + images
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 80)
    print("TEST: AD GENERATOR")
    print("=" * 80)

    # Check command line args
    generate_images_flag = "--images" in sys.argv or "-i" in sys.argv

    customer = "moprobo"
    platform = "meta"

    # Check if patterns.yaml exists
    patterns_path = Path(f"results/{customer}/{platform}/ad_miner/patterns.yaml")

    if not patterns_path.exists():
        print(f"\n❌ patterns.yaml not found at {patterns_path}")
        print("   Run test_e2e/07_test_ad_miner_real_data.py first")
        return 1

    print(f"\n📂 Patterns found:")
    print(f"  {patterns_path}")
    print(f"  Size: {patterns_path.stat().st_size} bytes")

    if generate_images_flag:
        print(f"\n🎨 Mode: Generate prompts + images")
        print(f"  This will call NanoBanana Pro API (requires FAL_KEY)")
    else:
        print(f"\n📝 Mode: Generate prompts only")

    # Import AdGenerator
    print(f"\n🏗️  Initializing AdGenerator...")
    from src.meta.ad.generator import AdGenerator

    generator = AdGenerator(
        customer=customer,
        platform=platform,
        patterns_path=patterns_path
    )

    # Run the pipeline
    print(f"\n" + "=" * 80)
    print("RUNNING AD GENERATOR PIPELINE")
    print("=" * 80)

    result = generator.run(save_prompts=True, generate_images=generate_images_flag)

    if not result.get("success"):
        print(f"\n❌ Pipeline failed: {result.get('error')}")
        return 1

    # Check generated prompts.yaml
    prompts_path = Path(result.get("prompts_path"))
    print(f"\n📝 Generated prompts:")
    print(f"  Path: {prompts_path}")
    print(f"  Size: {prompts_path.stat().st_size} bytes")

    # Show summary
    print(f"\n📊 Summary:")
    print(f"  Total prompts: {result.get('total_prompts')}")
    print(f"  Categories:")
    for category, count in result.get('categories', {}).items():
        print(f"    {category}: {count} prompts")

    if generate_images_flag:
        images_generated = result.get('images_generated', 0)
        print(f"\n🎨 Images generated: {images_generated}")

        if images_generated > 0:
            print(f"\n📁 Generated images:")
            image_paths = result.get('image_paths', {})
            for prompt_id, paths in image_paths.items():
                for path in paths:
                    print(f"  {prompt_id}: {path}")
        else:
            print(f"\n⚠️  No images generated")
            print(f"  Make sure FAL_KEY is set in ~/.devease/keys")

    # Show a sample prompt
    print(f"\n🎨 Sample prompt (top combination):")
    import yaml
    with open(prompts_path, 'r') as f:
        prompts_data = yaml.safe_load(f)

    top_prompts = prompts_data.get("prompts", {}).get("top_combination", [])
    if top_prompts:
        sample = top_prompts[0]
        print(f"  ID: {sample.get('prompt_id')}")
        print(f"  Name: {sample.get('prompt_name')}")
        print(f"  Strategy: {sample.get('strategy')}")
        print(f"  Confidence: {sample.get('confidence')}")
        print(f"  ROAS Lift: {sample.get('roas_lift')}x")
        print(f"  Features: {sample.get('features_used')}")
        print(f"  Prompt: {sample.get('nano_prompt')}")

    print(f"\n📁 Output directories:")
    print(f"  Patterns: {result.get('patterns_path')}")
    print(f"  Prompts: {result.get('prompts_path')}")
    print(f"  Creatives: {result.get('creatives_dir')}")

    print(f"\n" + "=" * 80)
    print("✅ TEST COMPLETE: Ad Generator working correctly")
    print("=" * 80)

    if not generate_images_flag:
        print(f"\n💡 To generate images:")
        print(f"  python3 {sys.argv[0]} --images")

    return 0


if __name__ == "__main__":
    sys.exit(main())
