# Template-Driven Ad Generator Usage Examples

## Quick Start

### Method 1: Using the Convenience Function

```python
from pathlib import Path
from src.meta.ad.generator.template_system import generate_ads

# Generate ads with default configuration
results = generate_ads(
    customer="moprobo",
    platform="facebook",
    product="Power Station",
    num_variants=3,
)

# Save outputs
for i, (image, metadata) in enumerate(results.generated_images):
    image.save(f"ad_candidate_{i+1}.png")
    print(f"Variant {i+1}: {metadata['psychology_driver']}")
```

### Method 2: Using the Pipeline Class

```python
from src.meta.ad.generator.template_system import TemplatePipeline, PipelineConfig

# Create pipeline configuration
config = PipelineConfig(
    customer="moprobo",
    platform="facebook",
    product="Power Station",
    num_variants=3,
    generate_backgrounds=True,  # Generate new backgrounds
    save_intermediates=True,    # Save intermediate outputs
)

# Initialize and run pipeline
pipeline = TemplatePipeline(config)
results = pipeline.run(
    product_image_path=Path("product.png"),
    background_image_path=None,  # Generate backgrounds
    num_variants=3,
)

# Access results
for i, (image, metadata) in enumerate(results.generated_images):
    print(f"Variant {i+1}:")
    print(f"  Template: {metadata['template_id']}")
    print(f"  Psychology: {metadata['psychology_driver']}")
    print(f"  Perspective: {metadata['perspective']}")
    image.save(f"output_{i+1}.png")
```

## Individual Component Usage

### Product Preprocessing

```python
from src.meta.ad.generator.template_system import preprocess_product

# Preprocess product image
result = preprocess_product("product.png")

# Access results
print(f"Original size: {result.original_size}")
print(f"Trimmed size: {result.trimmed_image.size}")
print(f"Perspective: {result.perspective.value}")

# Save outputs
result.trimmed_image.save("product_trimmed.png")
result.mask.save("product_mask.png")
```

### Template Selection

```python
from src.meta.ad.generator.template_system import select_template_from_blueprint
import yaml

# Load master blueprint
with open("config/ad/moprobo/master_blueprint.yaml") as f:
    blueprint = yaml.safe_load(f)

# Select template based on psychology_driver
template = select_template_from_blueprint(blueprint)

print(f"Selected: {template.display_name}")
print(f"Psychology: {template.psychology_driver}")
print(f"Font: {template.typography['headline']['font_family']}")
```

### Background Generation

```python
from src.meta.ad.generator.template_system import generate_backgrounds_from_blueprint
from src.meta.ad.generator.template_system import PerspectiveType
import yaml

# Load blueprint
with open("config/ad/moprobo/master_blueprint.yaml") as f:
    blueprint = yaml.safe_load(f)

# Generate backgrounds
results = generate_backgrounds_from_blueprint(
    blueprint=blueprint,
    perspective=PerspectiveType.EYE_LEVEL,
    output_dir=Path("backgrounds"),
)

for i, bg in enumerate(results):
    bg.image.save(f"background_{i+1}.png")
```

### Physics-Aware Compositing

```python
from PIL import Image
from src.meta.ad.generator.template_system import composite_physics_aware

# Load images
product = Image.open("product_trimmed.png")
background = Image.open("background.png")

# Composite with physics
result = composite_physics_aware(
    product_image=product,
    background_image=background,
    shadow_direction="left",
    light_wrap_intensity=0.3,
)

result.save("composited.png")
```

### Smart Text Overlay

```python
from PIL import Image
from src.meta.ad.generator.template_system import (
    render_text_overlay,
    CampaignContent,
)

# Load images and template
background = Image.open("composited.png")
# ... load template_spec ...

# Create campaign content
campaign = CampaignContent(
    headline="Power Your Adventures",
    sub_text="Portable Power Station",
    cta_text="Shop Now",
    brand_color="#FF5733",
)

# Render text
result = render_text_overlay(
    image=background,
    product_mask=None,  # Optional
    campaign_content=campaign,
    template_spec=template_spec,
    smart_color=True,
    collision_detection=True,
)

result.save("final_ad.png")
```

## Configuration File Setup

### Directory Structure

```
config/
  ad/
    moprobo/                          # Customer directory
      ├── master_blueprint.yaml       # Single config, shared across platforms
      ├── campaign_content.yaml       # Base campaign content
      ├── campaign_content_facebook.yaml   # Optional: Facebook overrides
      └── campaign_content_tiktok.yaml     # Optional: TikTok overrides

    generator/
      ├── psychology_catalog.yaml     # System: 14 psychology types
      └── text_templates.yaml         # System: Template definitions

results/
  moprobo/
    ├── facebook/                     # Platform-specific outputs
    │   └── ad_generator/
    │       ├── generated/
    │       ├── backgrounds/
    │       └── composited/
    └── tiktok/                       # Another platform
        └── ad_generator/
            └── ...
```

### Master Blueprint Example

See: `config/ad/moprobo/master_blueprint.yaml`

Key fields:
- `strategy_rationale.psychology_driver`: Auto-selects template
- `nano_generation_rules`: Background generation config
- `compositing`: Physics compositing config
- `text_overlay`: Text overlay config

### Campaign Content Example

See: `config/ad/moprobo/campaign_content.yaml`

Fields:
- `headline`: Main text
- `sub_text`: Optional subheading
- `cta_text`: Call-to-action
- `brand_color`: CTA button color

## Command-Line Usage

```bash
# Run pipeline from command line
python -m src.meta.ad.generator.template_system.pipeline moprobo facebook "Power Station" 3

# Arguments:
# 1. customer: Customer name
# 2. platform: Platform (facebook, tiktok, instagram)
# 3. product: Product name
# 4. num_variants: Number of variants (optional, default: 1)
```

## Advanced Usage

### Custom Path Management

```python
from src.meta.ad.generator.template_system import GeneratorPaths

paths = GeneratorPaths(
    customer="moprobo",
    platform="facebook",
)

# Get paths
blueprint_path = paths.get_blueprint_path()
campaign_path = paths.get_campaign_content_path(platform_specific=True)
output_path = paths.get_generated_output_path("Power Station")

print(f"Blueprint: {blueprint_path}")
print(f"Campaign: {campaign_path}")
print(f"Output: {output_path}")
```

### Custom Compositing Config

```python
from src.meta.ad.generator.template_system import (
    PhysicsCompositor,
    CompositingConfig,
    ShadowDirection,
)

config = CompositingConfig(
    shadow_direction=ShadowDirection.LEFT,
    light_wrap_intensity=0.4,  # Stronger light wrap
    light_match_opacity=0.3,   # More light matching
)

compositor = PhysicsCompositor(config)
result = compositor.composite(
    product_image=product,
    background_image=background,
    product_mask=mask,
)
```

### Platform-Specific Overrides

Create platform-specific campaign content:

```yaml
# config/ad/moprobo/campaign_content_tiktok.yaml
campaign_content:
  headline: "Short & Punchy"  # TikTok needs shorter text
  sub_text: null  # No subtext for TikTok
  cta_text: "Buy Now"  # More urgent CTA
  brand_color: "#FF5733"
```

The pipeline will automatically use the platform-specific file when available.

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
cd /path/to/devease-ad-agents-offline
python -m src.meta.ad.generator.template_system.pipeline ...
```

### Missing Fonts

The system uses PIL's default fonts. To use custom fonts, install them:

```python
typer = SmartTyper(font_dir=Path("/path/to/fonts"))
```

### Background Generation Not Working

Background generation requires API integration. Implement the actual API call in:

```python
# src/meta/ad/generator/template_system/background_generator.py
# See NanoBackgroundGenerator.generate_batch()
```

For now, it returns empty list (fallback to blank backgrounds).

## Next Steps

1. Implement actual NanoBanana Pro API integration
2. Add font loading from custom directories
3. Implement proper affine skew for shadows (using OpenCV)
4. Add more psychology templates (currently 3, design supports 14)
5. Add unit tests for all modules

See: `docs/ad_generator_v1.9_design.md` for complete design documentation.
