#!/usr/bin/env python3
"""
Delete and un-repost 100 latest tweets - ROBUST VERSION.

Better error handling and recovery.
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playwright.sync_api import sync_playwright
import time
import random

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


def get_latest_tweets_robust(page, count=100):
    """Get latest tweet URLs with retries."""
    tweets = []

    # Try direct profile navigation
    print("📍 Navigating to your profile...")

    # Try to find a link to your profile
    try:
        # Try getting profile from home page
        page.goto("https://twitter.com/home", timeout=15000)
        time.sleep(3)

        # Look for your handle/profile link
        profile_link = page.query_selector('a[href*="/"]')
        if profile_link:
            href = profile_link.get_attribute("href")
            if href and '/status/' not in href and href.startswith("/"):
                profile_url = "https://twitter.com" + href
                print(f"✓ Found profile: {profile_url}")
                page.goto(profile_url, timeout=15000)
                time.sleep(3)

    except Exception as e:
        print(f"⚠️  Could not navigate to profile: {e}")
        print("Trying to get tweets from current page...")

    # Get current tweets
    try:
        tweet_elems = page.query_selector_all('[data-testid="tweet"]')
        print(f"✓ Found {len(tweet_elems)} tweets on current page")

        # Limit to requested count
        tweet_elems = tweet_elems[:count]

        for i, tweet_elem in enumerate(tweet_elems):
            try:
                # Get tweet URL
                tweet_link = tweet_elem.query_selector('a[href*="/status/"]')
                if tweet_link:
                    tweet_url = tweet_link.get_attribute("href")
                    if tweet_url and not tweet_url.startswith("http"):
                        tweet_url = "https://twitter.com" + tweet_url

                    # Get tweet text
                    tweet_text_elem = tweet_elem.query_selector('[data-testid="tweetText"]')
                    content = tweet_text_elem.inner_text() if tweet_text_elem else ""

                    tweets.append({
                        "url": tweet_url,
                        "content": content[:80] + "..." if len(content) > 80 else content,
                        "index": i + 1
                    })

            except Exception as e:
                print(f"  ⚠️  Could not extract tweet {i+1}: {e}")
                continue

    except Exception as e:
        print(f"❌ Error getting tweets: {e}")

    return tweets


def delete_tweet_safe(page, tweet_url, index):
    """Delete tweet with better error handling."""
    try:
        print(f"  Navigating to: {tweet_url}")
        page.goto(tweet_url, timeout=30000, wait_until="domcontentloaded")
        time.sleep(2)

        # Screenshot
        screenshots_dir = Path("logs/twitter/screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = screenshots_dir / f"delete_{index}_{timestamp}.png"

        page.screenshot(path=str(screenshot_path))
        print(f"  📸 Screenshot saved: {screenshot_path.name}")

        # Check for retweet button - unretweet if present
        try:
            unretweet_button = page.query_selector('[data-testid="unretweet"]')
            if unretweet_button:
                print("  🔓 Found retweet - unretweeting...")
                unretweet_button.click()
                time.sleep(1)

                # Confirm unretweet
                confirm = page.query_selector('[data-testid="confirmationSheetConfirm"]')
                if confirm:
                    confirm.click()
                    time.sleep(2)
                    print("  ✓ Unretweeted")
        except:
            pass  # Not a retweet, that's fine

        # Delete tweet
        print("  🗑️  Deleting...")

        # Click more button
        more_button = page.wait_for_selector('[data-testid="caret"]', timeout=5000)
        more_button.click()
        time.sleep(1)

        # Click delete
        delete_button = page.query_selector('text=Delete')
        if delete_button:
            delete_button.click()
            time.sleep(1)

            # Confirm
            confirm_button = page.query_selector('[data-testid="confirmationSheetConfirm"]')
            if confirm_button:
                confirm_button.click()
                time.sleep(2)
                print("  ✓ ✓ ✓ DELETED SUCCESSFULLY")
                return True
        else:
            print("  ⚠️  Delete button not found (already deleted?)")
            return True  # Consider it success if already deleted

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Delete 100 latest tweets")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--count", type=int, default=100, help="Number of tweets to delete")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"Delete & Un-repost {args.count} Latest Tweets")
    print("=" * 80)
    print()
    print("⚠️  THIS WILL DELETE TWEETS")
    print(f"   Count: {args.count}")
    print()
    print("Rate limiting:")
    print("  • Delay: 60-90 seconds between tweets")
    print(f"  • Estimated time: {args.count * 75 / 60:.1f} minutes")
    print("  • Screenshots saved before each deletion")
    print()

    if not args.yes:
        input("Press Enter to continue or Ctrl+C to cancel...")
    else:
        print("--yes flag set, proceeding automatically...")

    print()

    try:
        with sync_playwright() as p:
            print("🔄 Connecting to Chrome...")
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            default_context = browser.contexts[0]

            # Use first available page or create new one
            if default_context.pages:
                page = default_context.pages[0]
            else:
                page = default_context.new_page()

            print("✓ Connected")
            print()

            # Get tweets
            tweets = get_latest_tweets_robust(page, count=args.count)

            if not tweets:
                print("❌ No tweets found. Make sure Chrome is on your profile page.")
                print()
                print("Try:")
                print("  1. Open Chrome manually")
                print("  2. Go to your profile: https://twitter.com/yourhandle")
                print("  3. Run this script again")
                return

            print()
            print("=" * 80)
            print(f"Found {len(tweets)} tweets")
            print("=" * 80)
            print()

            # Show preview
            print("Preview (first 5):")
            for tweet in tweets[:5]:
                print(f"  {tweet['index']}. {tweet['content']}")
            print()

            if not args.yes:
                input("Press Enter to start deletion or Ctrl+C to cancel...")
            else:
                print("Starting deletion...")

            print()

            # Process tweets
            successful = 0
            failed = 0

            for i, tweet in enumerate(tweets, 1):
                print()
                print("=" * 80)
                print(f"Tweet {i}/{len(tweets)}")
                print("=" * 80)
                print(f"URL: {tweet['url']}")
                print()

                if delete_tweet_safe(page, tweet['url'], i):
                    successful += 1
                else:
                    failed += 1

                # Rate limiting delay (skip after last)
                if i < len(tweets):
                    delay = random.uniform(60, 90)
                    print(f"⏸️  Waiting {delay:.1f}s before next tweet... ({i}/{len(tweets)} done)")
                    time.sleep(delay)

            # Summary
            print()
            print("=" * 80)
            print("DELETION COMPLETE!")
            print("=" * 80)
            print()
            print(f"Total tweets processed: {len(tweets)}")
            print(f"✓ Successful: {successful}")
            print(f"✗ Failed: {failed}")
            print(f"Screenshots: logs/twitter/screenshots/")
            print()

            browser.close()

    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("⚠️  INTERRUPTED")
        print("=" * 80)
        print()
        print(f"Stopped after {successful} tweets deleted")
        print("You can restart the script to continue")
        sys.exit(0)


if __name__ == "__main__":
    main()
