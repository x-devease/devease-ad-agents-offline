#!/usr/bin/env python3
"""
TEST 2: Ad Generator - Complete Flow Test

Generates ad images using the template system.
"""

import sys
import yaml
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.meta.ad.generator.template_system.pipeline import (
    TemplatePipeline,
    PipelineConfig,
    generate_ads,
)


def create_sample_background():
    """Create a sample background for testing."""
    # Create a nice gradient background
    background = Image.new("RGB", (1080, 1080), (240, 240, 235))

    # Add some subtle texture/gradient
    draw = ImageDraw.Draw(background)
    for y in range(0, 1080, 2):
        color = int(235 + (y / 1080) * 20)
        draw.line([(0, y), (1080, y)], fill=(color, color, color))

    return background


def main():
    print("=" * 80)
    print("TEST 2: AD GENERATOR - Complete Flow Test")
    print("=" * 80)

    customer = "moprobo"
    platform = "meta"
    product = "Power Station"

    print(f"\n📋 Configuration:")
    print(f"  Customer: {customer}")
    print(f"  Platform: {platform}")
    print(f"  Product: {product}")

    # Check required files
    print(f"\n📂 Checking required files...")
    blueprint_path = Path(f"config/{customer}/{platform}/config.yaml")
    campaign_path = Path(f"config/{customer}/{platform}/campaign_content.yaml")
    product_path = Path(f"config/{customer}/{platform}/products/{product.lower().replace(' ', '_')}.png")

    print(f"  Blueprint: {blueprint_path}")
    print(f"    Exists: {blueprint_path.exists()}")
    print(f"  Campaign: {campaign_path}")
    print(f"    Exists: {campaign_path.exists()}")
    print(f"  Product: {product_path}")
    print(f"    Exists: {product_path.exists()}")

    if not all([blueprint_path.exists(), campaign_path.exists(), product_path.exists()]):
        print(f"\n❌ Missing required files!")
        return 1

    # Create background
    print(f"\n🎨 Creating sample background...")
    background = create_sample_background()
    background_path = Path("test_e2e/tmp/background.png")
    background_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(background_path)
    print(f"  ✓ Background created: {background_path}")

    # Initialize pipeline
    print(f"\n🔧 Initializing Template Pipeline...")
    config = PipelineConfig(
        customer=customer,
        platform=platform,
        product=product,
        num_variants=3,  # Generate 3 variants
        generate_backgrounds=False,  # Use our sample background
        save_intermediates=True,
    )

    pipeline = TemplatePipeline(config)
    print(f"  ✓ Pipeline initialized")

    # Run generation
    print(f"\n🚀 Generating ad variants...")
    try:
        result = pipeline.run(
            product_image_path=str(product_path),
            background_image_path=str(background_path),
            num_variants=3,
        )

        print(f"\n✅ Generation completed successfully!")

        # Show results
        print(f"\n📊 Results:")
        print(f"  Total images generated: {len(result.generated_images)}")

        # Save test copies
        test_output_dir = Path("test_e2e/tmp/generated")
        test_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🖼️  Generated Images:")
        for i, (image, metadata) in enumerate(result.generated_images, 1):
            print(f"\n  Variant {i}:")
            print(f"    Size: {image.size}")
            print(f"    Template: {metadata.get('template_id', 'N/A')}")
            print(f"    Psychology: {metadata.get('psychology_driver', 'N/A')}")
            print(f"    Perspective: {metadata.get('perspective', 'N/A')}")

            # Save test copy
            test_path = test_output_dir / f"ad_variant_{i}.png"
            image.save(test_path)
            print(f"    Saved: {test_path} ({test_path.stat().st_size / 1024:.1f} KB)")

        # Show output directory
        print(f"\n📁 Output Directory:")
        output_dir = pipeline.paths.get_generated_output_path(product)
        print(f"  {output_dir}")

        # List generated files
        if output_dir.exists():
            files = list(output_dir.glob("*.png"))
            print(f"\n📄 Files in output directory ({len(files)} total):")
            for f in files:
                print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

        print(f"\n" + "=" * 80)
        print("✅ TEST 2 COMPLETED: Ad Generator working correctly!")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
