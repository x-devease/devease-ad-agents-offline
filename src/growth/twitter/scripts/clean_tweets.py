#!/usr/bin/env python3
"""
Non-stop tweet deletion - clean your entire profile.

This will:
1. Continuously scroll to load tweets
2. Process each tweet (unretweet retweets + delete original tweets)
3. Auto-recover if navigated away from profile
4. Continue until NO MORE TWEETS FOUND or safety limit reached

Usage:
    python clean_tweets.py --yes --max 10000

    --yes: Skip confirmation prompts
    --max: Safety limit (default 10000 tweets)
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--max", type=int, default=10000, help="Maximum tweets to process (safety limit)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("NON-STOP Tweet Deletion (Until profile is clean)")
    print("=" * 80)
    print(f"Safety limit: {args.max} tweets maximum")
    print()

    screenshots_dir = Path("logs/twitter/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        print("🔄 Connecting...")
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()
        print("✓ Connected")
        print()

        # Go to profile
        print("📍 Navigating to profile...")
        page.goto("https://x.com/0xJ_builder", timeout=15000)
        time.sleep(3)
        print(f"  ✓ On profile")
        print()

        # Function to ensure we're on profile page
        def ensure_on_profile():
            """Check if on profile, navigate back if not."""
            current_url = page.url

            # Check if navigated away from profile
            if '/status/' in current_url or current_url == 'https://x.com/home' or '/i/' in current_url:
                print(f"  ⚠️  Navigated away! Current: {current_url}")
                print("  🔙 Returning to profile...")
                page.goto("https://x.com/0xJ_builder", timeout=15000)
                time.sleep(3)
                print("  ✓ Back on profile")
                return True
            elif not current_url.startswith('https://x.com/0xJ_builder'):
                print(f"  ⚠️  Wrong page! Current: {current_url}")
                print("  🔙 Going to profile...")
                page.goto("https://x.com/0xJ_builder", timeout=15000)
                time.sleep(3)
                print("  ✓ Back on profile")
                return True

            return False

        print("=" * 80)
        print("Starting non-stop deletion...")
        print("Will continue until no more tweets found or safety limit reached")
        print("=" * 80)
        print()

        success = 0
        failed = 0
        processed = 0
        scroll_count = 0
        no_tweets_count = 0  # Count consecutive times no tweets found
        max_no_tweets = 3  # Stop after 3 consecutive scans with no tweets

        while processed < args.max and no_tweets_count < max_no_tweets:
            try:
                # Check if we're still on profile page
                if ensure_on_profile():
                    # Page was refreshed/navigated, reset tweets
                    time.sleep(2)
                    continue

                # Get current tweets
                tweets = page.query_selector_all('[data-testid="tweet"]')

                if len(tweets) == 0:
                    no_tweets_count += 1
                    print(f"  No tweets found (scan #{no_tweets_count}/{max_no_tweets}), scrolling to load more...")
                    page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
                    time.sleep(3)

                    if no_tweets_count >= max_no_tweets:
                        print("  ✓✓✓ No more tweets found after multiple scans - DONE!")
                        break

                    scroll_count += 1
                    continue

                # Reset no_tweets_count when we find tweets
                no_tweets_count = 0

                # Process first tweet
                tweet = tweets[0]
                processed += 1

                print(f"[Tweet #{processed}] Scroll #{scroll_count + 1}")

                # Get content
                text_elem = tweet.query_selector('[data-testid="tweetText"]')
                content = text_elem.inner_text()[:40] if text_elem else "(No text)"
                print(f"  Content: {content}...")

                # Scroll into view
                tweet.scroll_into_view_if_needed()
                time.sleep(1)

                # Check if retweet
                has_unretweet = tweet.query_selector('[data-testid="unretweet"]') is not None
                is_retweet = False

                if has_unretweet:
                    print("  🔓 This is a retweet, unretweeting...")
                    is_retweet = True

                    try:
                        unretweet_btn = tweet.query_selector('[data-testid="unretweet"]')
                        if unretweet_btn:
                            unretweet_btn.click(force=True)
                            time.sleep(2)
                            print("    ✓ Unretweet button clicked")

                            # Click confirm
                            confirm_selectors = [
                                'span:has-text("Undo repost")',
                                '[data-testid="confirmationSheetConfirm"]',
                                'div[role="button"]:has-text("Undo repost")',
                                'button:has-text("Undo repost")',
                            ]

                            confirmed = False
                            for selector in confirm_selectors:
                                try:
                                    confirm_btn = page.query_selector(selector)
                                    if confirm_btn:
                                        confirm_btn.click()
                                        time.sleep(3)
                                        print("    ✓✓ Unretweeted!")
                                        confirmed = True
                                        success += 1
                                        break
                                except:
                                    continue

                            if not confirmed:
                                print("    ✗ Confirm failed")
                                failed += 1

                    except Exception as e:
                        print(f"    ✗ Unretweet error: {e}")
                        failed += 1

                else:
                    # Not a retweet, try to delete
                    print("  🗑️  This is an original tweet, deleting...")

                    try:
                        more_btn = tweet.query_selector('[data-testid="caret"]')
                        if more_btn:
                            more_btn.click(force=True)
                            time.sleep(2)

                            # Find delete
                            delete_selectors = [
                                'div[role="menuitem"]:has-text("Delete")',
                                '[data-testid="delete"]',
                            ]

                            delete_clicked = False
                            for selector in delete_selectors:
                                delete_btn = page.query_selector(selector)
                                if delete_btn:
                                    delete_btn.click()
                                    time.sleep(2)
                                    delete_clicked = True
                                    break

                            if delete_clicked:
                                # Confirm
                                confirm = page.query_selector('[data-testid="confirmationSheetConfirm"]')
                                if confirm:
                                    confirm.click()
                                    time.sleep(3)
                                    print("    ✓✓ Deleted!")
                                    success += 1
                                else:
                                    print("    ✗ Confirm not found")
                                    failed += 1
                            else:
                                print("    ✗ Delete not found")
                                failed += 1

                    except Exception as e:
                        print(f"    ✗ Delete error: {e}")
                        failed += 1

                # Scroll to next
                print(f"  ⬇️  Scrolling to next tweet...")
                page.evaluate("window.scrollBy(0, 500)")
                time.sleep(2)

                # Periodically scroll more to load new tweets
                if processed % 5 == 0:
                    print(f"  📜 Loading more tweets...")
                    for _ in range(3):
                        page.evaluate("window.scrollBy(0, window.innerHeight)")
                        time.sleep(1)

                print()

            except Exception as e:
                print(f"✗ ERROR: {e}")
                failed += 1
                time.sleep(2)

        print("=" * 80)
        print("NON-STOP DELETION COMPLETE!")
        print("=" * 80)
        print()
        print(f"Total tweets processed: {processed}")
        print(f"✓ Successfully deleted/unretweeted: {success}")
        print(f"✗ Failed: {failed}")
        print()

        if processed >= args.max:
            print(f"⚠️  Reached safety limit of {args.max} tweets")
            print("   Run again to continue cleaning")
        else:
            print("✅ Profile is clean! No more tweets found.")
        print("=" * 80)
        print()

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(0)
