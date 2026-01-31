#!/usr/bin/env python3
"""
Regenerate prompts.yaml from patterns.yaml using updated PromptBuilder.

This script tests the new PromptBuilder features:
1. Configurable max_prompts from config.yaml
2. Anti-pattern validation
3. Category metadata for A/B testing
4. Confidence threshold filtering
"""

import yaml
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    customer = "moprobo"
    platform = "meta"

    logger.info("=" * 80)
    logger.info("REGENERATING PROMPTS.YAML FROM PATTERNS")
    logger.info("=" * 80)

    # Load config
    config_path = Path(f"config/{customer}/{platform}/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get product name from config
    product_name = config.get("metadata", {}).get("product", "product")

    logger.info(f"\n📋 Config loaded from: {config_path}")
    logger.info(f"   Product: {product_name}")
    logger.info(f"   Customer: {customer}")
    logger.info(f"   Platform: {platform}")

    # Show prompt building configuration
    prompt_building = config.get("prompt_building", {})
    max_prompts = prompt_building.get("max_prompts", {})
    confidence_thresholds = prompt_building.get("confidence_thresholds", {})

    logger.info(f"\n⚙️  Prompt Building Configuration:")
    logger.info(f"   Max prompts:")
    logger.info(f"     - Top combination: {max_prompts.get('top_combination', 1)}")
    logger.info(f"     - Supporting combinations: {max_prompts.get('supporting_combinations', 3)}")
    logger.info(f"     - Individual features: {max_prompts.get('individual_features', 5)}")
    logger.info(f"     - Psychology patterns: {max_prompts.get('psychology_patterns', 2)}")
    logger.info(f"   Confidence thresholds:")
    logger.info(f"     - Individual features: {confidence_thresholds.get('individual_features', 0.0)}")
    logger.info(f"   Nano enhancement: {prompt_building.get('enable_nano_enhancement', True)}")

    # Load patterns
    patterns_path = Path(f"results/{customer}/{platform}/ad_miner/patterns.yaml")
    with open(patterns_path, 'r') as f:
        patterns = yaml.safe_load(f)

    logger.info(f"\n📊 Patterns loaded from: {patterns_path}")
    logger.info(f"   Combinatorial patterns: {len(patterns.get('combinatorial_patterns', []))}")
    logger.info(f"   Individual features: {len(patterns.get('individual_features', []))}")
    logger.info(f"   Psychology patterns: {len(patterns.get('psychology_patterns', []))}")
    logger.info(f"   Anti-patterns: {len(patterns.get('anti_patterns', []))}")

    # Build prompts using PromptBuilder
    logger.info(f"\n🔨 Building prompts...")

    from src.meta.ad.generator.prompt_builder import PromptBuilder

    # Prepare config with both sections intact
    full_config = config.copy()
    # Ensure prompt_building section exists with proper structure
    if 'prompt_building' not in full_config:
        full_config['prompt_building'] = {}
    # Add nano generation rules to config
    full_config['prompt_building']['max_prompts'] = prompt_building.get('max_prompts', {})
    full_config['prompt_building']['confidence_thresholds'] = prompt_building.get('confidence_thresholds', {})
    full_config['prompt_building']['enable_nano_enhancement'] = prompt_building.get('enable_nano_enhancement', True)

    builder = PromptBuilder(
        patterns_path=patterns_path,
        config=full_config
    )

    all_prompts = builder.build_all_prompts()

    # Count prompts
    total_prompts = sum(len(v) for v in all_prompts.values())

    # Count anti-pattern violations
    violations = []
    for category, prompts in all_prompts.items():
        for prompt in prompts:
            if not prompt.get("passed_anti_pattern_check", True):
                violations.append({
                    "prompt_id": prompt.get("prompt_id"),
                    "category": prompt.get("category"),
                    "violations": prompt.get("anti_pattern_violations", [])
                })

    logger.info(f"\n✅ Prompts generated:")
    logger.info(f"   Top combination: {len(all_prompts['top_combination'])}")
    logger.info(f"   Supporting combinations: {len(all_prompts['supporting_combinations'])}")
    logger.info(f"   Individual features: {len(all_prompts['individual_features'])}")
    logger.info(f"   Psychology patterns: {len(all_prompts['psychology'])}")
    logger.info(f"   Total: {total_prompts}")

    if violations:
        logger.warning(f"\n⚠️  Anti-pattern violations detected: {len(violations)}")
        for v in violations:
            logger.warning(f"     - {v['prompt_id']}: {', '.join(v['violations'])}")
    else:
        logger.info(f"\n✓ No anti-pattern violations detected")

    # Prepare output
    output = {
        "metadata": {
            "customer": customer,
            "platform": platform,
            "source_patterns": str(patterns_path),
            "total_prompts": total_prompts,
            "categories": list(all_prompts.keys()),
            "generated_at": datetime.now().isoformat(),
            "prompt_building_config": {
                "max_prompts": max_prompts,
                "confidence_thresholds": confidence_thresholds,
                "enable_nano_enhancement": prompt_building.get('enable_nano_enhancement', True)
            }
        },
        "prompts": all_prompts
    }

    # Save prompts
    output_path = Path(f"results/{customer}/{platform}/ad_miner/prompts.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, width=1000)

    logger.info(f"\n💾 Saved prompts to: {output_path}")

    # Show sample prompts
    logger.info(f"\n📋 Sample prompts:")
    for category, prompts in all_prompts.items():
        if prompts:
            sample = prompts[0]
            logger.info(f"\n{category.upper()}:")
            logger.info(f"   ID: {sample.get('prompt_id')}")
            logger.info(f"   Name: {sample.get('prompt_name')}")
            logger.info(f"   Category: {sample.get('category')}")
            logger.info(f"   Strategy: {sample.get('strategy')}")
            logger.info(f"   Confidence: {sample.get('confidence')}")
            logger.info(f"   ROAS lift: {sample.get('roas_lift')}")
            logger.info(f"   Passed anti-pattern check: {sample.get('passed_anti_pattern_check')}")
            if sample.get('anti_pattern_violations'):
                logger.warning(f"   Violations: {sample['anti_pattern_violations']}")

    logger.info(f"\n" + "=" * 80)
    logger.info(f"✅ COMPLETE: Prompts regenerated with new features")
    logger.info(f"=" * 80)

if __name__ == "__main__":
    main()
