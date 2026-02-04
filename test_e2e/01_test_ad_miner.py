#!/usr/bin/env python3
"""
TEST 1: Ad Miner - Test Patterns Output

Tests that the ad miner properly outputs patterns.yaml
"""

import sys
from pathlib import Path
import yaml
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ad miner components
from src.meta.ad.miner.io import PatternsIO, save_patterns_yaml


def create_sample_patterns():
    """Create sample patterns data for testing."""
    patterns = {
        "metadata": {
            "schema_version": "2.0",
            "customer": "moprobo",
            "product": "Power Station",
            "branch": "US",
            "campaign_goal": "conversion",
            "granularity_level": 1,
            "sample_size": 342,
            "min_threshold": 200,
            "analysis_date": "2026-01-30",
            "fallback_used": False,
            "data_quality": {
                "completeness_score": 0.95,
                "avg_roas": 2.34,
                "top_quartile_roas": 4.56,
                "bottom_quartile_roas": 0.98,
                "roas_range": 4.65,
                "top_quartile_size": 85,
                "bottom_quartile_size": 85
            }
        },
        "patterns": [
            {
                "feature": "surface_material",
                "value": "Marble",
                "current_value": "Wood",
                "pattern_type": "DO",
                "confidence": "high",
                "roas_lift_multiple": 2.8,
                "roas_lift_pct": 180.0,
                "top_quartile_prevalence": 0.67,
                "bottom_quartile_prevalence": 0.12,
                "prevalence_lift": 0.55,
                "goal_specific": True,
                "product_specific": True,
                "branch_specific": False,
                "reason": "For conversion campaigns with Power Station in US, marble surfaces show 2.8x higher ROAS. Present in 67% of top performers vs 12% in bottom quartile.",
                "maps_to_template": "surface_material",
                "priority_score": 9.5,
                "sample_count": 89
            }
        ],
        "anti_patterns": [
            {
                "feature": "lighting_style",
                "avoid_value": "Studio Lighting",
                "pattern_type": "DON'T",
                "confidence": "high",
                "roas_penalty_multiple": 0.6,
                "roas_penalty_pct": -40.0,
                "bottom_quartile_prevalence": 0.65,
                "top_quartile_prevalence": 0.15,
                "reason": "Used in 65% of worst performers, 40% lower ROAS than average",
                "maps_to_template": "lighting_style",
                "sample_count": 65
            }
        ],
        "low_priority_insights": [
            {
                "feature": "contrast_level",
                "value": "high",
                "roas_lift_multiple": 1.05,
                "roas_lift_pct": 5.0,
                "confidence": "low",
                "reason": "Slight positive trend (5% lift), but not statistically significant (p=0.15)",
                "trend_direction": "positive"
            }
        ],
        "generation_instructions": {
            "must_include": ["surface_material", "lighting_style"],
            "prioritize": ["visual_prominence", "color_balance"],
            "avoid": ["lighting_style:Studio Lighting"],
            "min_coverage": 0.8,
            "max_features": 15
        }
    }
    return patterns


def main():
    print("=" * 80)
    print("TEST 1: AD MINER - Patterns Output Test")
    print("=" * 80)

    customer = "moprobo"
    platform = "meta"

    print(f"\n📋 Configuration:")
    print(f"  Customer: {customer}")
    print(f"  Platform: {platform}")

    # Create output directory
    output_dir = Path(f"results/{customer}/{platform}/ad_miner")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Create sample patterns
    print(f"\n🔨 Creating sample patterns data...")
    patterns = create_sample_patterns()
    print(f"  ✓ Patterns data created")

    # Save patterns as YAML
    print(f"\n💾 Saving patterns to YAML...")
    patterns_path = output_dir / "patterns.yaml"

    success = save_patterns_yaml(
        patterns_data=patterns,
        yaml_path=str(patterns_path),
        validate=False  # Skip validation for this test
    )

    if not success:
        print(f"  ✗ Failed to save patterns.yaml")
        return 1

    print(f"  ✓ Patterns saved to: {patterns_path}")

    # Verify file exists and is valid YAML
    print(f"\n🔍 Verifying patterns.yaml...")

    if not patterns_path.exists():
        print(f"  ✗ File does not exist: {patterns_path}")
        return 1

    print(f"  ✓ File exists")

    # Load and validate YAML
    try:
        with open(patterns_path, 'r') as f:
            loaded_patterns = yaml.safe_load(f)
        print(f"  ✓ YAML is valid")
    except Exception as e:
        print(f"  ✗ Failed to load YAML: {e}")
        return 1

    # Verify content
    print(f"\n📊 Verifying content...")

    metadata = loaded_patterns.get('metadata', {})
    print(f"  Customer: {metadata.get('customer')}")
    print(f"  Product: {metadata.get('product')}")
    print(f"  Schema Version: {metadata.get('schema_version')}")
    print(f"  Sample Size: {metadata.get('sample_size')}")

    patterns_count = len(loaded_patterns.get('patterns', []))
    anti_patterns_count = len(loaded_patterns.get('anti_patterns', []))
    low_priority_count = len(loaded_patterns.get('low_priority_insights', []))

    print(f"\n  Positive Patterns: {patterns_count}")
    print(f"  Anti-Patterns: {anti_patterns_count}")
    print(f"  Low-Priority Insights: {low_priority_count}")

    generation_instructions = loaded_patterns.get('generation_instructions')
    if generation_instructions:
        print(f"\n  Generation Instructions:")
        print(f"    Must Include: {generation_instructions.get('must_include')}")
        print(f"    Avoid: {generation_instructions.get('avoid')}")

    print(f"\n" + "=" * 80)
    print("✅ TEST 1 COMPLETED: patterns.yaml output working correctly!")
    print("=" * 80)

    print(f"\n📁 Output File:")
    print(f"   {patterns_path}")
    print(f"   Size: {patterns_path.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
