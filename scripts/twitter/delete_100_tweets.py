#!/usr/bin/env python3
"""
Delete and un-repost 100 latest tweets.

This script will:
1. List your latest 100 tweets
2. For each tweet:
   - If it's a retweet/repost → unretweet
   - Then delete the tweet
3. Use conservative rate limiting (60-120s delays)
4. Take screenshots before deletion
5. Full progress tracking

Estimated time: 2-3 hours for 100 tweets
"""

import sys
from pathlib import Path
from datetime import datetime

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


def get_latest_tweets(page, count=100):
    """Get latest tweet URLs from profile."""
    tweets = []

    # Navigate to your profile
    print("📍 Navigating to your profile...")
    page.goto("https://twitter.com/home", wait_until="networkidle")
    time.sleep(2)

    # Click profile icon
    profile_icon = page.query_selector('[data-testid="UserAvatar"]')
    if profile_icon:
        profile_icon.click()
        time.sleep(1)

        # Click profile link
        profile_link = page.query_selector('a[href*="/status"]:not([href*="/status/"])')
        if profile_link:
            profile_link.click()
            time.sleep(3)

    # Scroll to load more tweets
    print("📜 Loading tweets...")
    for i in range(5):  # Scroll 5 times to load tweets
        page.evaluate("window.scrollBy(0, window.innerHeight)")
        time.sleep(1)

    # Get tweet elements
    tweet_elems = page.query_selector_all('[data-testid="tweet"]')[:count]

    print(f"✓ Found {len(tweet_elems)} tweets")

    for i, tweet_elem in enumerate(tweet_elems):
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

    return tweets


def unretweet(page, tweet_url):
    """Unretweet a retweeted tweet."""
    try:
        page.goto(tweet_url)
        time.sleep(2)

        # Check if it's a retweet
        unretweet_button = page.query_selector('[data-testid="unretweet"]')

        if unretweet_button:
            print("  🔓 Unretweeting...")
            unretweet_button.click()
            time.sleep(1)

            # Confirm
            confirm_button = page.query_selector('[data-testid="confirmationSheetConfirm"]')
            if confirm_button:
                confirm_button.click()
                time.sleep(2)
                print("  ✓ Unretweeted")
                return True

        return False

    except Exception as e:
        print(f"  ✗ Unretweet failed: {e}")
        return False


def delete_tweet(page, tweet_url, index):
    """Delete a tweet with screenshot."""
    try:
        page.goto(tweet_url)
        time.sleep(2)

        # Screenshot before deletion
        screenshots_dir = Path("logs/twitter/screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = screenshots_dir / f"delete_{index}_{timestamp}.png"

        page.screenshot(path=str(screenshot_path))
        print(f"  📸 Screenshot: {screenshot_path.name}")

        # Click more menu
        more_button = page.wait_for_selector('[data-testid="caret"]', timeout=5000)
        more_button.click()
        time.sleep(0.5)

        # Click delete
        delete_button = page.wait_for_selector('text=Delete', timeout=5000)
        if delete_button:
            delete_button.click()
            time.sleep(0.5)

            # Confirm
            confirm_button = page.wait_for_selector('[data-testid="confirmationSheetConfirm"]', timeout=5000)
            if confirm_button:
                confirm_button.click()
                time.sleep(2)
                print("  ✓ Deleted")
                return True

        return False

    except Exception as e:
        print(f"  ✗ Delete failed: {e}")
        return False


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("Delete & Un-repost 100 Latest Tweets")
    print("=" * 80)
    print()
    print("⚠️  THIS WILL DELETE 100 TWEETS")
    print("   Starting from your latest tweets")
    print()
    print("Rate limiting:")
    print("  • Delay: 60-90 seconds between tweets")
    print("  • Estimated time: 2-2.5 hours")
    print("  • Screenshots saved before each deletion")
    print()

    input("Press Enter to continue or Ctrl+C to cancel...")
    print()

    with sync_playwright() as p:
        print("🔄 Connecting to Chrome...")
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        default_context = browser.contexts[0]
        page = default_context.pages[0]

        print("✓ Connected")
        print()

        # Get latest tweets
        tweets = get_latest_tweets(page, count=100)

        if not tweets:
            print("❌ No tweets found")
            return

        print()
        print("=" * 80)
        print(f"Found {len(tweets)} tweets")
        print("=" * 80)
        print()

        # Show first few
        print("First 5 tweets:")
        for tweet in tweets[:5]:
            print(f"  {tweet['index']}. {tweet['content']}")

        print()
        input("Press Enter to start deletion or Ctrl+C to cancel...")
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
            print(f"Content: {tweet['content']}")
            print()

            # Check if retweet and unretweet
            unretweeted = unretweet(page, tweet['url'])

            if unretweeted:
                time.sleep(random.uniform(2, 4))

            # Delete tweet
            print("  🗑️  Deleting...")
            deleted = delete_tweet(page, tweet['url'], i)

            if deleted:
                successful += 1
            else:
                failed += 1

            # Rate limiting delay (skip after last)
            if i < len(tweets):
                delay = random.uniform(60, 90)
                print(f"  ⏸️  Waiting {delay:.1f}s before next tweet...")
                time.sleep(delay)

        # Summary
        print()
        print("=" * 80)
        print("COMPLETE!")
        print("=" * 80)
        print()
        print(f"Total tweets processed: {len(tweets)}")
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        print()

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 80)
        print("⚠️  INTERRUPTED BY USER")
        print("=" * 80)
        print()
        print("You can check the screenshots in: logs/twitter/screenshots/")
        sys.exit(0)
