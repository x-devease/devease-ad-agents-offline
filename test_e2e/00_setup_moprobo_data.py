#!/usr/bin/env python3
"""
Setup Moprobo Test Data

Creates the necessary config files for testing the ad miner pipeline.
"""

import sys
from pathlib import Path
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

def create_config_directories():
    """Create config directory structure."""
    config_dir = Path("config/moprobo/meta")
    config_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {config_dir}")

    products_dir = config_dir / "products"
    products_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {products_dir}")

    return config_dir

def create_master_blueprint():
    """Create master blueprint for testing."""
    blueprint = {
        "metadata": {
            "schema_version": "2.0",
            "customer": "moprobo",
            "product": "Power Station",
            "platform": "meta",
            "branch": "US",
            "campaign_goal": "conversion",
            "generated_at": "2026-01-30"
        },
        "strategy_rationale": {
            "psychology_driver": "trust",
            "psychology_confidence": 0.85,
            "rationale": "High ROAS winners use marble surfaces with window light",
            "locked_combination": {
                "primary_material": "Marble",
                "paired_lighting": "Window Light",
                "confidence_score": 0.92
            }
        },
        "nano_generation_rules": {
            "inference_config": {
                "aspect_ratio": "3:4",
                "batch_size": 20,
                "cfg_scale": 3.5,
                "guidance": "perspective_aware",
                "model": "nanobanana_pro",
                "steps": 8
            },
            "negative_prompt": "cartoon, illustration, text, watermark, distorted",
            "prompt_slots": {
                "atmosphere": "luxury home environment",
                "product_context": "professional power station"
            },
            "prompt_template_structure": "Product on table, {atmosphere}"
        },
        "compositing": {
            "light_match_mode": "soft_light",
            "light_match_opacity": 0.25,
            "light_wrap_intensity": 0.3,
            "shadow_direction": "left"
        },
        "text_overlay": {
            "template_id": "trust_authority",
            "psychology_driven": True,
            "smart_color_enabled": True,
            "collision_detection_enabled": True
        },
        "psychology_catalog": {
            "total_types": 1,
            "types": [{
                "category": "authority_trust",
                "colors": {"primary": "#003366"},
                "copy_patterns": ["Expert recommended"],
                "description": "Establish credibility",
                "full_name": "Trust & Authority",
                "layout": {"position": "centered"},
                "psychology_id": "trust",
                "typography": {"headline_font": "Serif_Bold"}
            }]
        },
        "psychology_templates": [{
            "display_name": "Trust: Authority",
            "layout": {
                "alignment": "center",
                "margin_y": 80,
                "position": "Bottom_Center"
            },
            "psychology_driver": "trust",
            "style": {
                "cta_bg_color": "Transparent",
                "cta_shape": "Pill_Solid",
                "font_color_logic": "Auto_Contrast"
            },
            "template_id": "trust_authority",
            "typography": {
                "cta": {
                    "font_family": "Sans_Medium",
                    "font_size": 24,
                    "padding_x": 40,
                    "padding_y": 16
                },
                "headline": {
                    "font_family": "Sans_Bold",
                    "font_size": 48
                },
                "sub_text": {
                    "font_family": "Sans_Regular",
                    "font_size": 32
                }
            }
        }]
    }

    config_dir = Path("config/moprobo/meta")
    blueprint_path = config_dir / "config.yaml"
    with open(blueprint_path, 'w') as f:
        yaml.dump(blueprint, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created master blueprint: {blueprint_path}")
    return blueprint

def create_campaign_content():
    """Create campaign content for testing."""
    content = {
        "campaign_content": {
            "headline": "Power Your Adventures",
            "sub_text": "Reliable portable power for anywhere",
            "cta_text": "Shop Now",
            "brand_color": "#FF6600"
        }
    }

    config_dir = Path("config/moprobo/meta")
    content_path = config_dir / "campaign_content.yaml"
    with open(content_path, 'w') as f:
        yaml.dump(content, f, default_flow_style=False)

    print(f"✓ Created campaign content: {content_path}")
    return content

def create_product_image():
    """Create a sample product image for testing."""
    products_dir = Path("config/moprobo/meta/products")
    products_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple product image (power station)
    img = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Draw a simple power station shape
    draw.rectangle([100, 150, 300, 280], fill=(255, 140, 0, 255), outline=(200, 100, 0, 255))
    draw.rectangle([120, 100, 280, 150], fill=(255, 160, 0, 255), outline=(200, 100, 0, 255))
    draw.rectangle([180, 200, 220, 250], fill=(50, 50, 50, 255))  # Port

    product_path = products_dir / "power_station.png"
    img.save(product_path)
    print(f"✓ Created product image: {product_path}")
    return product_path

def main():
    print("=" * 80)
    print("SETUP: Creating Moprobo Test Data")
    print("=" * 80)

    create_config_directories()
    create_master_blueprint()
    create_campaign_content()
    create_product_image()

    print("\n" + "=" * 80)
    print("✅ SETUP COMPLETE: All test data created")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
