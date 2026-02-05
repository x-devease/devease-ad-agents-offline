#!/usr/bin/env python3
"""
Safe batch tweet deletion with anti-rate-limiting.

Features:
- Random delays between deletions (30-60s default)
- Screenshot before each deletion
- Progress tracking
- Can pause/resume
- Detailed logging
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.growth.twitter.agents.browser_agent import BrowserAgent
from src.growth.twitter.core.types import TwitterKeys, TwitterConfig

# Load keys
keys_path = Path.home() / ".devease" / "keys"
env_vars = {}
with open(keys_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            key, value = line.split('=', 1)
            env_vars[key.strip()] = value.strip()

api_key = env_vars.get('OPENAI_API_KEY', '')
org_id = env_vars.get('OPENAI_ORG_ID', '')


def safe_batch_delete():
    """Demonstrate safe batch deletion."""
    print("\n" + "=" * 80)
    print("Safe Batch Tweet Deletion")
    print("=" * 80)
    print()
    print("SAFETY FEATURES:")
    print("  ✓ Random delays between deletions (anti-rate-limit)")
    print("  ✓ Screenshot before each deletion")
    print("  ✓ Progress tracking")
    print("  ✓ Can pause/resume")
    print()
    print("RATE LIMIT PROTECTION:")
    print("  • Default delay: 30-60 seconds between deletions")
    print("  • Twitter API limit: ~100-300 deletions/hour (estimate)")
    print("  • Browser automation: More conservative (safer)")
    print()
    print("RECOMMENDED DELAYS:")
    print("  • Conservative: 60-120s (1-2 min) - Very safe")
    print("  • Default: 30-60s - Safe for batch operations")
    print("  • Fast: 15-30s - For small batches (<10 tweets)")
    print("  • Risky: 5-10s - May trigger rate limits")
    print()

    # Example tweet URLs (replace with real ones)
    tweets_to_delete = [
        "https://twitter.com/your/status/1234567890",
        "https://twitter.com/your/status/1234567891",
        "https://twitter.com/your/status/1234567892",
    ]

    print(f"Example tweets to delete: {len(tweets_to_delete)}")
    print()

    # Choose delay mode
    print("Select delay mode:")
    print("  1. Conservative (60-120s per deletion)")
    print("  2. Default (30-60s per deletion)")
    print("  3. Fast (15-30s per deletion)")
    print("  4. Custom")
    print()

    choice = input("Choose mode (1-4, or 0 to cancel): ").strip()

    if choice == '0':
        print("Cancelled")
        return
    elif choice == '1':
        delay_range = (60, 120)
        print("✓ Conservative mode selected (60-120s delays)")
    elif choice == '2':
        delay_range = (30, 60)
        print("✓ Default mode selected (30-60s delays)")
    elif choice == '3':
        delay_range = (15, 30)
        print("✓ Fast mode selected (15-30s delays)")
    elif choice == '4':
        min_delay = int(input("Min seconds between deletions: "))
        max_delay = int(input("Max seconds between deletions: "))
        delay_range = (min_delay, max_delay)
        print(f"✓ Custom mode selected ({min_delay}-{max_delay}s delays)")
    else:
        print("Invalid choice, using default (30-60s)")
        delay_range = (30, 60)

    print()
    print("=" * 80)
    print("DELETION PLAN")
    print("=" * 80)
    print(f"Tweets to delete: {len(tweets_to_delete)}")
    print(f"Delay range: {delay_range[0]}-{delay_range[1]} seconds")
    import random
    avg_delay = sum(delay_range) / 2
    estimated_time = len(tweets_to_delete) * avg_delay
    print(f"Estimated time: {estimated_time / 60:.1f} minutes")
    print()

    confirm = input("Proceed with deletion? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("Cancelled")
        return

    print()
    print("=" * 80)
    print("STARTING BATCH DELETION")
    print("=" * 80)
    print()

    # Initialize agent
    keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
    config = TwitterConfig(llm_model="gpt-4o")
    agent = BrowserAgent(keys, config, headless=False)

    # Delete tweets with rate limiting
    results = agent.delete_tweets_batch(tweets_to_delete, delay_range=delay_range)

    print()
    print("=" * 80)
    print("DELETION COMPLETE")
    print("=" * 80)
    print()

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    print(f"Total: {len(results)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")

    if failed > 0:
        print()
        print("Failed deletions:")
        for i, result in enumerate(results):
            if not result.success:
                print(f"  {i+1}. {result.error}")


if __name__ == "__main__":
    safe_batch_delete()
