#!/usr/bin/env python3
"""
Non-stop reply deletion - clean all replies from your profile.

This will:
1. Navigate to your profile's Replies tab
2. For each reply:
   - Unlike it (if you liked it)
   - Unretweet it (if it's a repost)
   - Delete it
3. Auto-recover if navigated away from replies page
4. Continue until NO MORE REPLIES FOUND or safety limit reached

Anti-bot features:
- Random delays between actions (simulates human behavior)
- Efficient scanning through all visible tweets
- Auto-recovery from navigation errors

Usage:
    python clean_replies.py --port 9223 --max 10000

    --port: Chrome remote debugging port (default 9223)
    --max: Safety limit (default 10000 replies)
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import time
import random

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(line_buffering=True)


def random_delay(min_seconds: float = 1.0, max_seconds: float = 3.0):
    """Add random delay to simulate human behavior."""
    time.sleep(random.uniform(min_seconds, max_seconds))


def human_like_scroll(page, direction: str = "down", intensity: float = 1.0):
    """
    Scroll in a more human-like pattern with variable speed.

    Args:
        page: Playwright page object
        direction: "down" or "up"
        intensity: Multiplier for scroll distance (0.5 to 2.0)
    """
    # Random number of scroll movements
    num_movements = random.randint(2, 4)

    for _ in range(num_movements):
        # Variable scroll distance
        if direction == "down":
            distance = random.randint(200, 500) * intensity
        else:
            distance = random.randint(-500, -200) * intensity

        page.evaluate(f"window.scrollBy(0, {distance})")

        # Random delay between movements
        time.sleep(random.uniform(0.1, 0.4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9223, help="Chrome remote debugging port (default 9223)")
    parser.add_argument("--max", type=int, default=10000, help="Maximum replies to process (safety limit)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("NON-STOP Reply Deletion (Until all replies are clean)")
    print("=" * 80)
    print("Will: Unlike → Unretweet (if needed) → Delete")
    print(f"Chrome port: {args.port}")
    print(f"Safety limit: {args.max} replies maximum")
    print()

    screenshots_dir = Path("logs/twitter/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        print("🔄 Connecting...")
        browser = p.chromium.connect_over_cdp(f"http://localhost:{args.port}")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()
        print("✓ Connected")
        print()

        # Go to profile
        print("📍 Navigating to profile...")
        page.goto("https://x.com/0xJ_builder", timeout=15000)
        random_delay(0.1, 0.3)
        print(f"  ✓ On profile")
        print()

        # Navigate to Replies tab
        print("📍 Navigating to Replies tab...")
        replies_tab_selectors = [
            'a[href*="/with_replies"]',
            'a[role="tab"]:has-text("Replies")',
            'div[data-testid="UserDescription"]:hover + div a[href*="/with_replies"]',  # Fallback
        ]

        navigated = False
        for selector in replies_tab_selectors:
            try:
                replies_tab = page.query_selector(selector)
                if replies_tab:
                    replies_tab.click()
                    random_delay(0.1, 0.3)
                    navigated = True
                    print("  ✓ On Replies tab")
                    break
            except:
                continue

        if not navigated:
            # Try direct URL
            page.goto("https://x.com/0xJ_builder/with_replies", timeout=15000)
            random_delay(0.1, 0.3)
            print("  ✓ On Replies tab (direct URL)")
        print()

        # Function to ensure we're on replies page
        def ensure_on_replies():
            """Check if on replies page, navigate back if not."""
            current_url = page.url

            # Check if navigated away from replies
            if '/status/' in current_url or current_url == 'https://x.com/home' or '/i/' in current_url:
                print(f"  ⚠️  Navigated away! Current: {current_url}")
                print("  🔙 Returning to replies...")
                page.goto("https://x.com/0xJ_builder/with_replies", timeout=15000)
                random_delay(0.1, 0.3)
                print("  ✓ Back on replies")
                return True
            elif not current_url.startswith('https://x.com/0xJ_builder/with_replies'):
                print(f"  ⚠️  Wrong page! Current: {current_url}")
                print("  🔙 Going to replies...")
                page.goto("https://x.com/0xJ_builder/with_replies", timeout=15000)
                random_delay(0.1, 0.3)
                print("  ✓ Back on replies")
                return True

            return False

        print("=" * 80)
        print("Starting non-stop deletion...")
        print("Will continue until no more replies found or safety limit reached")
        print("=" * 80)
        print()

        success = 0
        failed = 0
        processed = 0  # Only count tweets we attempt to delete
        skipped = 0  # Count other people's tweets we skip
        unliked = 0  # Count of replies unliked
        unretweeted = 0  # Count of retweets unretweeted
        scroll_count = 0
        no_replies_count = 0  # Count consecutive times no replies found
        max_no_replies = 3  # Stop after 3 consecutive scans with no replies
        processed_tweets = set()  # Track processed tweets to avoid duplicates

        while processed < args.max and no_replies_count < max_no_replies:
            try:
                # Check if we're still on replies page
                if ensure_on_replies():
                    # Page was refreshed/navigated, reset replies
                    random_delay(0.1, 0.2)
                    continue

                # Get current replies
                tweets = page.query_selector_all('[data-testid="tweet"]')

                if len(tweets) == 0:
                    no_replies_count += 1
                    print(f"  No replies found (scan #{no_replies_count}/{max_no_replies}), scrolling to load more...")
                    human_like_scroll(page, "down", intensity=2.0)
                    random_delay(0.1, 0.3)

                    if no_replies_count >= max_no_replies:
                        print("  ✓✓✓ No more replies found after multiple scans - DONE!")
                        break

                    scroll_count += 1
                    continue

                # Reset no_replies_count when we find replies
                no_replies_count = 0

                # Process ONE tweet at a time (top to bottom)
                if len(tweets) > 0:
                    # Get the first visible tweet
                    tweet = tweets[0]

                    # Check author
                    author_elem = tweet.query_selector('[data-testid="User-Name"] a')
                    author_handle = ""
                    if author_elem:
                        author_href = author_elem.get_attribute("href")
                        author_handle = author_href.strip("/") if author_href else ""

                    # Get content for display AND unique identifier
                    text_elem = tweet.query_selector('[data-testid="tweetText"]')
                    content = text_elem.inner_text()[:40] if text_elem else "(No text)"

                    # Create unique identifier: author + first 30 chars of content
                    tweet_id = f"{author_handle}:{content[:30]}"

                    # Skip if already processed this tweet
                    if tweet_id in processed_tweets:
                        # Already processed, scroll past it
                        page.evaluate("window.scrollBy(0, window.innerHeight)")
                        time.sleep(0.3)
                        scroll_count += 1
                        continue

                    # Mark as processed
                    processed_tweets.add(tweet_id)

                    # Check if this is your tweet
                    if author_handle == "0xJ_builder":
                        # This is your reply - delete it
                        processed += 1
                        print(f"[Your Reply #{processed}] Scroll #{scroll_count + 1}")
                        print(f"  Content: {content}...")
                        print("  🗑️  Deleting reply...")

                        try:
                            # Click the more button
                            more_btn = tweet.query_selector('[data-testid="caret"]')
                            if more_btn:
                                more_btn.click(force=True)
                                time.sleep(0.3)

                                # Find delete
                                delete_selectors = [
                                    '[data-testid="delete"]',
                                    'div[role="menuitem"]:has-text("Delete")',
                                    'div[role="menuitem"] div[dir="auto"]:has-text("Delete")',
                                ]

                                delete_clicked = False
                                for selector in delete_selectors:
                                    try:
                                        delete_btn = page.query_selector(selector)
                                        if delete_btn:
                                            delete_btn.click(force=True)
                                            time.sleep(0.3)
                                            delete_clicked = True
                                            print(f"    ✓ Delete clicked")
                                            break
                                    except:
                                        continue

                                if delete_clicked:
                                    # Confirm deletion
                                    confirm_selectors = [
                                        '[data-testid="confirmationSheetConfirm"]',
                                        'div[role="button"]:has-text("Delete")',
                                    ]

                                    confirmed = False
                                    for selector in confirm_selectors:
                                        try:
                                            confirm_btn = page.query_selector(selector)
                                            if confirm_btn:
                                                confirm_btn.click(force=True)
                                                time.sleep(0.5)
                                                print("    ✓✓ Deleted!")
                                                success += 1
                                                confirmed = True
                                                break
                                        except:
                                            continue

                                    if not confirmed:
                                        print("    ✗ Confirm not found")
                                        failed += 1
                                else:
                                    print("    ✗ Delete button not found")
                                    failed += 1
                            else:
                                print("    ✗ Caret button not found")
                                failed += 1

                        except Exception as e:
                            print(f"    ✗ Delete error: {e}")
                            failed += 1

                    else:
                        # Someone else's tweet - unlike/unretweet it
                        print(f"[ @{author_handle}'s tweet] Scroll #{scroll_count + 1}")
                        print(f"  Content: {content}...")

                        # Step 1: Unretweet if you retweeted it
                        unretweet_selectors = [
                            '[data-testid="unretweet"]',
                            'div[role="button"][aria-label*="Undo repost"]',
                            'div[role="button"][aria-label*="Unretweet"]',
                        ]

                        unretweet_btn = None
                        for selector in unretweet_selectors:
                            try:
                                unretweet_btn = tweet.query_selector(selector)
                                if unretweet_btn:
                                    break
                            except:
                                continue

                        if unretweet_btn:
                            print("  🔓 You retweeted this, unretweeting...")
                            try:
                                unretweet_btn.click(force=True)
                                time.sleep(0.3)
                                print("    ✓ Unretweet clicked")

                                # Confirm unretweet
                                confirm_selectors = [
                                    'span:has-text("Undo repost")',
                                    '[data-testid="confirmationSheetConfirm"]',
                                    'div[role="button"]:has-text("Undo repost")',
                                ]

                                confirmed = False
                                for selector in confirm_selectors:
                                    try:
                                        confirm_btn = page.query_selector(selector)
                                        if confirm_btn:
                                            confirm_btn.click(force=True)
                                            time.sleep(0.5)
                                            print("    ✓✓ Unretweeted!")
                                            unretweeted += 1
                                            confirmed = True
                                            break
                                    except:
                                        continue

                                if not confirmed:
                                    print("    ✗ Unretweet confirm failed")
                            except Exception as e:
                                print(f"    ✗ Unretweet error: {e}")

                        # Step 2: Unlike if you liked it
                        # ONLY find the unlike button (data-testid="unlike")
                        # DO NOT search for like button or we'll accidentally like tweets!
                        unlike_selectors = [
                            '[data-testid="unlike"]',
                            'div[role="button"][aria-label*="Unlike"]',
                        ]

                        unlike_btn = None
                        for selector in unlike_selectors:
                            try:
                                unlike_btn = tweet.query_selector(selector)
                                if unlike_btn:
                                    break
                            except:
                                continue

                        if unlike_btn:
                            print("  ❤️ You liked this tweet, unliking...")
                            try:
                                unlike_btn.click(force=True)
                                time.sleep(0.3)
                                print("    ✓ Unliked!")
                                unliked += 1
                            except Exception as e:
                                print(f"    ✗ Unlike error: {e}")

                        # Done with this other person's tweet
                        print(f"  ✓ Processed @{author_handle}'s tweet")
                        print()
                        skipped += 1
                        if skipped % 10 == 0:
                            print(f"[Progress] Processed {skipped} others' tweets...")

                    # Scroll by full viewport height to move past current tweet
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(0.5)
                    scroll_count += 1

            except Exception as e:
                print(f"✗ ERROR: {e}")
                failed += 1
                random_delay(0.1, 0.3)

        print("=" * 80)
        print("NON-STOP DELETION COMPLETE!")
        print("=" * 80)
        print()
        print(f"Your replies processed: {processed}")
        print(f"✓ Successfully deleted: {success}")
        print(f"✗ Failed: {failed}")
        print(f"❤️  Unliked: {unliked}")
        print(f"🔓 Unretweeted: {unretweeted}")
        print(f"⏭️  Skipped (others' tweets): {skipped}")
        print()

        if processed >= args.max:
            print(f"⚠️  Reached safety limit of {args.max} replies")
            print("   Run again to continue cleaning")
        else:
            print("✅ All your replies are clean! No more replies found.")
        print("=" * 80)
        print()

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(0)
