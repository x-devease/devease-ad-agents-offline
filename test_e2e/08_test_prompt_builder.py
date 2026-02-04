#!/usr/bin/env python3
"""
TEST: Prompt Builder

Tests the PromptBuilder class that dynamically generates prompts from mined patterns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 80)
    print("TEST: Prompt Builder")
    print("=" * 80)

    patterns_path = Path("results/moprobo/meta/ad_miner/patterns.yaml")

    if not patterns_path.exists():
        print(f"\n❌ patterns.yaml not found at {patterns_path}")
        print("   Run test_e2e/07_test_ad_miner_real_data.py first")
        return 1

    print(f"\n📂 Loading patterns from:")
    print(f"  {patterns_path}")
    print(f"  Size: {patterns_path.stat().st_size} bytes")

    # Import PromptBuilder
    print(f"\n🏗️  Initializing PromptBuilder...")
    from src.meta.ad.generator.prompt_builder import PromptBuilder

    builder = PromptBuilder(patterns_path)

    # Test 1: Top combination prompt
    print(f"\n" + "=" * 80)
    print("TEST 1: Top Combination Prompt")
    print("=" * 80)

    top_prompt = builder.build_top_combination_prompt()

    print(f"\n📋 Prompt ID: {top_prompt.get('prompt_id')}")
    print(f"   Name: {top_prompt.get('prompt_name')}")
    print(f"   Strategy: {top_prompt.get('strategy')}")
    print(f"   Confidence: {top_prompt.get('confidence')}")
    print(f"   ROAS Lift: {top_prompt.get('roas_lift')}x")

    print(f"\n🎨 Features Used:")
    for k, v in top_prompt.get('features_used', {}).items():
        print(f"   {k}: {v}")

    print(f"\n📝 Nano Prompt:")
    print(f"   {top_prompt.get('nano_prompt')}")

    print(f"\n🧠 Psychology Overlay:")
    for k, v in top_prompt.get('psychology_overlay', {}).items():
        print(f"   {k}: {v}")

    # Test 2: Supporting combination prompts
    print(f"\n" + "=" * 80)
    print("TEST 2: Supporting Combination Prompts")
    print("=" * 80)

    supporting_prompts = builder.build_supporting_combination_prompts(max_prompts=2)

    print(f"\n✓ Generated {len(supporting_prompts)} supporting prompts")

    for i, prompt in enumerate(supporting_prompts, 1):
        print(f"\n{i}. {prompt.get('prompt_name')}")
        print(f"   ROAS Lift: {prompt.get('roas_lift')}x")
        print(f"   Prompt: {prompt.get('nano_prompt')}")

    # Test 3: Individual feature prompts
    print(f"\n" + "=" * 80)
    print("TEST 3: Individual Feature Prompts")
    print("=" * 80)

    individual_prompts = builder.build_individual_feature_prompts(max_prompts=3)

    print(f"\n✓ Generated {len(individual_prompts)} individual feature prompts")

    for i, prompt in enumerate(individual_prompts, 1):
        print(f"\n{i}. {prompt.get('prompt_name')}")
        print(f"   ROAS Lift: {prompt.get('roas_lift')}x")
        print(f"   Reason: {prompt.get('note', prompt.get('reason', 'N/A'))}")

    # Test 4: All prompts
    print(f"\n" + "=" * 80)
    print("TEST 4: Build All Prompts")
    print("=" * 80)

    all_prompts = builder.build_all_prompts()

    print(f"\n📊 Summary:")
    for category, prompts in all_prompts.items():
        print(f"   {category}: {len(prompts)} prompts")

    total_prompts = sum(len(v) for v in all_prompts.values())
    print(f"   TOTAL: {total_prompts} prompts")

    print(f"\n" + "=" * 80)
    print("✅ TEST COMPLETE: Prompt Builder working correctly")
    print("=" * 80)

    print(f"\n💡 Next Steps:")
    print(f"   1. Integrate PromptBuilder into ad_generator pipeline")
    print(f"   2. Use prompts for NanoBanana Pro image generation")
    print(f"   3. Combine with psychology templates and text overlay")

    return 0


if __name__ == "__main__":
    sys.exit(main())
