# Test Fixtures

This directory contains test fixtures for e2e tests.

## Required Files

### `sample_product.jpg`
A sample product image for testing image generation.

**Setup Instructions:**

For local testing with image generation, add a product image:

```bash
# Option 1: Copy an existing product image
cp /path/to/your/product/image.jpg tests/fixtures/sample_product.jpg

# Option 2: Download a sample image
curl -o tests/fixtures/sample_product.jpg https://placehold.co/800x800/cccccc/666666?text=Product+Image

# Option 3: Use ImageMagick to create a placeholder
convert -size 800x800 xc:#cccccc -gravity center -pointsize 48 -fill #666666 -annotate 0 "Product Image" tests/fixtures/sample_product.jpg
```

## Image Generation Testing

Image generation tests require:
1. A sample product image in this directory
2. Valid FAL API credentials (FAL_KEY)
3. CI=false environment variable (enabled by default for local testing)

To run with image generation:
```bash
GENERATE_IMAGES=true python3 -m tests.e2e.test_ad_recommendation_generation
```

To skip image generation (CI mode):
```bash
CI=true python3 -m tests.e2e.test_ad_recommendation_generation
```
