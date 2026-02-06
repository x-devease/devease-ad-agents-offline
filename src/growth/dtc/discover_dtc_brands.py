#!/usr/bin/env python3
"""
DTC Brand Discovery: Smart Pairing Matrix System

V2.0 - Intent-Verified Discovery:
- Search tasks generated from Smart Pairing Matrix strategies
- Each search = pre-combined niche + intent trigger
- Results have embedded strategy tags (no post-processing)

Example:
  Search: "Mushroom coffee" + "Subscribe & Save" → LTV_HUNTERS
  Result: Brand actively running subscription ads (verified intent)

Target: $1M-$15M ARR DTC brands with active direct response marketing
"""

import asyncio
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import argparse

# Get the script's directory for absolute paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data"


class DTCBrandHunter:
    """
    Hunter for $1M-$15M ARR DTC brands using Smart Pairing Matrix.

    Single-pass discovery: Search = Filtering
    - Strategy definitions in smart_pairing_matrix.json
    - Search tasks generated from strategy combinations
    - Each result pre-tagged with outreach strategy
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load smart pairing matrix (single source of truth)
        self.config = self._load_smart_pairing_matrix()
        self.exclude_keywords = set(self.config.get('exclude_keywords', []))
        self.signals = self.config.get('signals', {})

        # Strategy definitions
        self.strategies = self.config.get('strategies', {})

    def _load_smart_pairing_matrix(self) -> Dict:
        """Load the smart pairing matrix configuration."""
        config_path = SCRIPT_DIR / "config" / "smart_pairing_matrix.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"✗ Smart pairing matrix not found: {config_path}")
            raise

    def generate_search_tasks(self) -> List[Dict]:
        """
        Generate search tasks from Smart Pairing Matrix strategies.

        Each strategy defines search_combinations: [niche, intent] pairs.
        Converts these into executable search tasks with embedded strategy.

        Returns:
            List of search task dicts with query, strategy_tag, outreach_template, pain_point
        """
        print("\n" + "="*70)
        print("GENERATING SEARCH TASKS FROM SMART PAIRING MATRIX")
        print("="*70)

        tasks = []

        # Iterate through each strategy
        for strategy_name, strategy_config in self.strategies.items():
            combinations = strategy_config.get('search_combinations', [])

            for niche, intent in combinations:
                # Create combined query with quotes for exact match
                query = f'"{niche}" "{intent}"'

                tasks.append({
                    'query': query,
                    'strategy_tag': strategy_name,
                    'niche': niche,
                    'intent_trigger': intent,
                    'outreach_template': strategy_config.get('outreach_template'),
                    'pain_point': strategy_config.get('pain_point'),
                    'description': strategy_config.get('description')
                })

        # Show stats
        strategy_counts = {}
        for task in tasks:
            s = task.get('strategy_tag', 'UNKNOWN')
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        print(f"✓ Generated {len(tasks)} intent-verified search tasks")

        print(f"\nStrategy breakdown:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} tasks")

        # Show sample tasks
        print(f"\nSample search tasks:")
        for task in list(tasks)[:5]:
            query = task.get('query')
            strategy = task.get('strategy_tag', 'UNKNOWN')
            print(f"  [{strategy}] {query}")

        return tasks

    async def search_meta_ad_library(
        self,
        query: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Search Meta ad library for brands using search query.

        Note: This is a placeholder for actual Meta ad library scraping.
        The real implementation would need to handle:
        - Facebook login/authentication
        - Ad library pagination
        - Rate limiting
        - Domain extraction from ads

        Args:
            query: Search query string
            max_results: Maximum brands to return

        Returns:
            List of discovered brand dicts with domain, ad_copy, etc.
        """
        # Placeholder - would implement actual scraping
        # Returns empty list for now
        return []

    async def discover_brands(
        self,
        max_queries: int = None,
        max_results_per_query: int = 50
    ) -> List[Dict]:
        """
        Intent-Verified Brand Discovery: Search = Filtering

        V2.0 Logic:
        1. Each search task = pre-combined niche + intent
        2. Results already have verified strategy (no post-processing)
        3. Strategy embedded in task, direct mapping to brand

        Example:
        - Task: "Mushroom coffee" + "Subscribe & Save" → LTV_HUNTERS
        - Result: Four Sigmatic found running subscription ads
        - Output: strategy_tag=LTV_HUNTERS (verified, not guessed)

        Args:
            max_queries: Limit number of search tasks (None = all)
            max_results_per_query: Max brands per query

        Returns:
            List of discovered brands with pre-tagged strategy
        """
        print("\n" + "="*70)
        print("INTENT-VERIFIED BRAND DISCOVERY")
        print("="*70)
        print("Each search = pre-filtered intent (no post-processing needed)")

        # Generate search tasks (already have strategy embedded)
        tasks = self.generate_search_tasks()
        if max_queries:
            tasks = tasks[:max_queries]

        print(f"\nExecuting {len(tasks)} intent-verified search tasks...")

        all_discovered = {}
        for i, task in enumerate(tasks, 1):
            query = task.get('query')
            strategy = task.get('strategy_tag', 'UNKNOWN')
            print(f"[{i}/{len(tasks)}] [{strategy}] {query}...", end=' ', flush=True)

            results = await self.search_meta_ad_library(query, max_results_per_query)

            # Filter out excluded keywords
            filtered = [
                r for r in results
                if not any(
                    neg.lower() in r.get('domain', '').lower() or
                    neg.lower() in r.get('ad_copy', '').lower()
                    for neg in self.exclude_keywords
                )
            ]

            # Add to discovered set (deduplicate by domain)
            for brand in filtered:
                domain = brand.get('domain')
                if domain and domain not in all_discovered:
                    # Strategy already embedded in task - NO LOOKUP NEEDED
                    all_discovered[domain] = {
                        **brand,
                        'source': 'meta_ads',
                        'discovery_query': query,
                        'strategy_tag': task.get('strategy_tag'),
                        'niche': task.get('niche'),
                        'intent_trigger': task.get('intent_trigger'),
                        'outreach_template': task.get('outreach_template'),
                        'pain_point': task.get('pain_point'),
                        'strategy_description': task.get('description'),
                        'discovered_at': datetime.now().isoformat(),
                    }

            print(f"✓ Found {len(results)} brands ({len(filtered)} after filters)")

            # Rate limiting
            await asyncio.sleep(1)

        # Convert to list
        discovered_brands = list(all_discovered.values())

        print(f"\n{'='*70}")
        print(f"DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"Total unique brands discovered: {len(discovered_brands)}")

        # Show strategy breakdown
        strategy_counts = {}
        for brand in discovered_brands:
            s = brand.get('strategy_tag', 'UNKNOWN')
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        print(f"\nStrategy breakdown:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} brands")

        return discovered_brands

    def save_results(self, brands: List[Dict], filename: str = "dtc_brands"):
        """
        Save discovered brands to JSON and CSV.

        Args:
            brands: List of discovered brand dicts
            filename: Output filename (without extension)
        """
        if not brands:
            print("\n⚠️  No brands to save")
            return

        # Sort by discovery time (most recent first)
        brands_sorted = sorted(
            brands,
            key=lambda x: x.get('discovered_at', ''),
            reverse=True
        )

        # Save JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(brands_sorted, f, indent=2)

        # Save CSV
        csv_path = self.output_dir / f"{filename}.csv"
        if brands_sorted:
            fieldnames = [
                'domain',
                'strategy_tag',
                'niche',
                'intent_trigger',
                'discovery_query',
                'pain_point',
                'outreach_template',
                'strategy_description',
                'discovered_at'
            ]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(brands_sorted)

        print(f"\n✓ Saved:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")
        print(f"   Total: {len(brands_sorted)} brands")

        # Show top brands
        print(f"\n🎯 DISCOVERED BRANDS:")
        print(f"   {'Domain':<30} | {'Strategy':<15} | {'Niche':<20} | {'Intent'}")
        print(f"   {'-'*30}-+-{'-'*15}-+-{'-'*20}-+-{'-'*25}")
        for brand in brands_sorted[:20]:
            domain = brand.get('domain', 'N/A')[:30]
            strategy = brand.get('strategy_tag', 'UNKNOWN')[:15]
            niche = brand.get('niche', 'N/A')[:20]
            intent = brand.get('intent_trigger', 'N/A')[:25]

            print(f"   {domain:<30} | {strategy:<15} | {niche:<20} | {intent}")


async def main():
    parser = argparse.ArgumentParser(
        description='DTC Brand Discovery: Smart Pairing Matrix System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run brand discovery with smart pairing strategy tags
  python3 discover_dtc_brands.py

  # Limit search tasks for testing
  python3 discover_dtc_brands.py --max-tasks 10

  # Custom output filename
  python3 discover_dtc_brands.py --output my_brands
        """
    )

    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Limit search tasks (default: all 40)')
    parser.add_argument('--output', type=str, default='dtc_brands',
                        help='Output filename (default: dtc_brands)')

    args = parser.parse_args()

    hunter = DTCBrandHunter()

    # Intent-verified discovery using Smart Pairing Matrix
    print("\n🔍 Smart Pairing Matrix Brand Discovery")
    print("   Each search = verified intent (niche + trigger)")
    print("   Results pre-tagged with outreach strategy")

    discovered = await hunter.discover_brands(max_queries=args.max_tasks)

    if discovered:
        hunter.save_results(discovered, args.output)
    else:
        print("\n⚠️  No brands discovered")


if __name__ == "__main__":
    asyncio.run(main())
