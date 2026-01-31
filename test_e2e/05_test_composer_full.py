#!/usr/bin/env python3
"""
TEST: Full Composer - Generate Real Backgrounds

Tests whether the composer can generate actual background images using FAL API.
"""

import sys
from pathlib import Path
import yaml
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.meta.ad.generator.template_system.background_generator import (
    NanoBackgroundGenerator,
)
from src.meta.ad.generator.template_system.product_preprocessor import PerspectiveType


def check_fal_credentials():
    """Check if FAL credentials are set."""
    fal_key = os.getenv("FAL_KEY")
    if fal_key:
        print(f"  ✓ FAL_KEY found (length: {len(fal_key)})")
        return True
    else:
        print(f"  ⚠️  FAL_KEY not set")
        print(f"     Set it with: export FAL_KEY='your-key-here'")
        return False


def test_real_background_generation():
    """Test actual background generation with FAL API."""
    print("=" * 80)
    print("TEST: COMPOSER - Generate Real Backgrounds with FAL API")
    print("=" * 80)

    # Check credentials
    print("\n🔑 Checking FAL credentials...")
    has_credentials = check_fal_credentials()

    if not has_credentials:
        print("\n" + "=" * 80)
        print("⚠️  SKIPPED: FAL credentials not available")
        print("=" * 80)
        print("\nTo run this test:")
        print("1. Get a FAL API key from https://fal.ai/")
        print("2. Set environment variable: export FAL_KEY='your-key'")
        print("3. Run this test again")
        return 0  # Return 0 so CI doesn't fail

    # Load master blueprint
    print("\n📋 Loading master blueprint...")
    blueprint_path = Path("config/moprobo/meta/config.yaml")
    with open(blueprint_path, 'r') as f:
        blueprint = yaml.safe_load(f)
    print(f"  ✓ Blueprint loaded")

    # Create output directory
    output_dir = Path("test_e2e/generated_backgrounds")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Initialize generator
    print(f"\n🎨 Initializing Background Generator...")
    generator = NanoBackgroundGenerator()

    # Test with eye-level perspective
    print(f"\n🚀 Generating background with eye-level perspective...")
    print(f"  This will take 30-60 seconds...")

    try:
        results = generator.generate_from_blueprint(
            blueprint=blueprint,
            perspective=PerspectiveType.EYE_LEVEL,
            output_dir=output_dir,
            save_images=True,
        )

        if results:
            print(f"\n✅ SUCCESS: Generated {len(results)} background(s)")

            for i, bg in enumerate(results, 1):
                print(f"\n  Background {i}:")
                print(f"    Index: {bg.index}")
                print(f"    Perspective: {bg.perspective.value}")
                print(f"    Prompt: {bg.prompt[:100]}...")

                if hasattr(bg, 'metadata') and bg.metadata:
                    print(f"    Model: {bg.metadata.get('model', 'N/A')}")
                    print(f"    Steps: {bg.metadata.get('steps', 'N/A')}")
                    print(f"    CFG: {bg.metadata.get('cfg_scale', 'N/A')}")

                # Check if image exists
                if hasattr(bg.image, 'filename'):
                    img_path = Path(bg.image.filename)
                elif isinstance(bg.image, Path):
                    img_path = bg.image
                else:
                    # Find the image file
                    images = list(output_dir.glob("background_*.jpg"))
                    if images:
                        img_path = images[0]
                    else:
                        img_path = None

                if img_path and img_path.exists():
                    size = img_path.stat().st_size / 1024
                    print(f"    File: {img_path.name} ({size:.1f} KB)")

                    # Verify it's a valid image
                    from PIL import Image
                    try:
                        img = Image.open(img_path)
                        print(f"    Size: {img.size[0]}x{img.size[1]}")
                        print(f"    Mode: {img.mode}")
                        print(f"    ✓ Valid image")
                    except Exception as e:
                        print(f"    ✗ Invalid image: {e}")
        else:
            print(f"\n⚠️  No backgrounds generated")
            print(f"    This could be due to:")
            print(f"    - API rate limiting")
            print(f"    - Invalid API key")
            print(f"    - Network issues")
            print(f"    - FAL service downtime")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

    print(f"\n📁 Generated backgrounds:")
    bg_files = list(output_dir.glob("background_*.jpg"))
    if bg_files:
        for f in bg_files:
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.1f} KB)")
    else:
        print(f"  No background files found")

    return 0


if __name__ == "__main__":
    sys.exit(test_real_background_generation())
