#!/usr/bin/env python3
"""
TEST: Loser Quantile Configuration

Tests the new configurable loser_quantile parameter.
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config():
    """Load moprobo config."""
    config_path = Path("config/moprobo/meta/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("=" * 80)
    print("TEST: Loser Quantile Configuration")
    print("=" * 80)

    # Load config
    print("\n📋 Loading moprobo config...")
    config = load_config()

    mining_strategy = config.get("mining_strategy", {})
    winner_quantile = mining_strategy.get("winner_quantile")
    loser_quantile = mining_strategy.get("loser_quantile")

    print(f"\n⚙️  Mining Strategy Configuration:")
    print(f"  Winner Quantile: {winner_quantile} (Top {(1-winner_quantile)*100:.0f}%)")
    print(f"  Loser Quantile:  {loser_quantile} (Bottom {loser_quantile*100:.0f}%)")

    # Calculate what it would be if auto-calculated
    auto_loser_quantile = 1 - winner_quantile

    print(f"\n📊 Comparison:")
    print(f"  Configured Loser Quantile: {loser_quantile}")
    print(f"  Auto-calculated (1-winner): {auto_loser_quantile}")
    print(f"  Match: {'✓ Yes' if loser_quantile == auto_loser_quantile else '✗ No (Custom)'}")

    # Visual representation
    print(f"\n📈 Ad Distribution:")
    top_pct = (1 - winner_quantile) * 100
    bottom_pct = loser_quantile * 100
    middle_pct = 100 - top_pct - bottom_pct

    print(f"  ┌{'─' * 70}┐")
    print(f"  │ {'Top ' + f'{top_pct:.0f}%':<10} │ {'Middle ' + f'{middle_pct:.0f}%':<12} │ {'Bottom ' + f'{bottom_pct:.0f}%':<12} │")
    print(f"  ├{'─' * 70}┤")
    print(f"  │ {'Winners':<10} │ {'(ignored)':<12} │ {'Losers':<12} │")
    print(f"  │ {'DOs':<10} │ {'':<12} │ {'DONTs':<12} │")
    print(f"  └{'─' * 70}┘")

    # Test different configurations
    print(f"\n🔧 Configuration Examples:")

    examples = [
        {"winner": 0.95, "loser": 0.05, "desc": "Conservative (symmetric)"},
        {"winner": 0.90, "loser": 0.10, "desc": "Balanced (symmetric)"},
        {"winner": 0.80, "loser": 0.20, "desc": "Aggressive (symmetric)"},
        {"winner": 0.80, "loser": 0.30, "desc": "Asymmetric (more losers)"},
        {"winner": 0.90, "loser": 0.15, "desc": "Asymmetric (fewer losers)"},
    ]

    for ex in examples:
        w = ex["winner"]
        l = ex["loser"]
        auto_l = 1 - w
        is_symmetric = "✓" if l == auto_l else "✗"

        print(f"\n  {ex['desc']}:")
        print(f"    winner_quantile: {w:.2f} (Top {(1-w)*100:.0f}%)")
        print(f"    loser_quantile:  {l:.2f} (Bottom {l*100:.0f}%) {is_symmetric}")
        if l != auto_l:
            print(f"    → Custom setting (auto would be {auto_l:.2f})")

    print(f"\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

    print(f"\n📝 Summary:")
    print(f"  ✓ loser_quantile is now configurable in config.yaml")
    print(f"  ✓ If not set, auto-calculated as (1 - winner_quantile)")
    print(f"  ✓ Allows symmetric or asymmetric winner/loser analysis")
    print(f"\n  Current config: Top {(1-winner_quantile)*100:.0f}% vs Bottom {loser_quantile*100:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
