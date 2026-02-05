#!/usr/bin/env python3
"""
Clean Replies Page - Delete your tweets, unlike & unretweet others

This script:
1. Finds YOUR tweets and deletes them
2. Finds ALL unlike buttons on screen and clicks them
3. Finds ALL unretweet buttons on screen and clicks them
4. Scrolls down and repeats

Usage:
    python clean_replies.py --max 10000
"""

import sys
from pathlib import Path
import argparse
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9223)
    parser.add_argument("--max", type=int, default=10000)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CLEAN REPLIES PAGE")
    print("=" * 80)
    print("Will: Delete your tweets | Unlike others | Unretweet others")
    print()

    with sync_playwright() as p:
        print("🔄 Connecting...")
        browser = p.chromium.connect_over_cdp(f"http://localhost:{args.port}")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()
        print("✓ Connected")
        print()

        # Go to replies
        print("📍 Navigating to Replies...")
        page.goto("https://x.com/0xJ_builder/with_replies", timeout=15000)
        time.sleep(2)
        print("  ✓ On Replies")
        print()

        print("=" * 80)
        print("Processing...")
        print("=" * 80)
        print()

        deleted = 0
        unliked = 0
        unretweeted = 0
        failed = 0
        loops = 0
        max_loops = 500  # Safety limit

        while loops < max_loops:
            loops += 1
            print(f"[Loop #{loops}] Scanning...")

            try:
                # Check if still on Replies page
                current_url = page.url
                if '/status/' in current_url or current_url == 'https://x.com/home':
                    print("  ⚠️  Navigated away! Returning to Replies...")
                    page.goto("https://x.com/0xJ_builder/with_replies", timeout=15000)
                    time.sleep(2)
                    continue

                tweets = page.query_selector_all('[data-testid="tweet"]')

                if len(tweets) == 0:
                    print("  Loading more tweets...")
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(2)
                    continue

                # STEP 1: Delete YOUR tweets (process top to bottom)
                your_tweets_found = 0
                for i, tweet in enumerate(tweets):
                    try:
                        author_elem = tweet.query_selector('[data-testid="User-Name"] a')
                        if not author_elem:
                            continue

                        author_href = author_elem.get_attribute("href") or ""
                        author_handle = author_href.strip("/")

                        if author_handle == "0xJ_builder":
                            your_tweets_found += 1
                            print(f"  [#{i+1}] YOUR TWEET → Deleting...")
                            try:
                                # Get fresh reference
                                tweets_fresh = page.query_selector_all('[data-testid="tweet"]')
                                if len(tweets_fresh) > i:
                                    tweet = tweets_fresh[i]

                                more_btn = tweet.query_selector('[data-testid="caret"]')
                                if more_btn:
                                    more_btn.click(force=True)
                                    time.sleep(1.0)

                                    # Try delete selectors
                                    delete_selectors = [
                                        '[data-testid="delete"]',
                                        'div[role="menuitem"]:has-text("Delete")',
                                    ]

                                    delete_clicked = False
                                    for sel in delete_selectors:
                                        delete_btn = page.query_selector(sel)
                                        if delete_btn:
                                            delete_btn.click(force=True)
                                            time.sleep(0.8)
                                            delete_clicked = True
                                            break

                                    if delete_clicked:
                                        confirm_btn = page.query_selector('[data-testid="confirmationSheetConfirm"]')
                                        if confirm_btn:
                                            confirm_btn.click(force=True)
                                            time.sleep(0.6)
                                            print("    ✓✓ Deleted!")
                                            deleted += 1
                                        else:
                                            print("    ✗ No confirm")
                                            failed += 1
                                    else:
                                        print("    ✗ No delete button")
                                        failed += 1
                                else:
                                    print("    ✗ No caret")
                                    failed += 1
                            except Exception as e:
                                print(f"    ✗ Error: {str(e)[:40]}")
                                failed += 1
                                time.sleep(1)
                    except:
                        continue

                # STEP 2: Unlike ALL tweets on screen
                unlike_buttons = page.query_selector_all('[data-testid="unlike"]')
                if len(unlike_buttons) > 0:
                    print(f"  💔 Found {len(unlike_buttons)} tweets to unlike...")
                    for btn in unlike_buttons:
                        try:
                            btn.click(force=True)
                            time.sleep(0.5)
                            unliked += 1
                            print(f"    ✓ Unliked! (Total: {unliked})")
                        except:
                            continue

                # STEP 3: Unretweet ALL tweets on screen
                unretweet_buttons = page.query_selector_all('[data-testid="unretweet"]')
                if len(unretweet_buttons) > 0:
                    print(f"  🔄 Found {len(unretweet_buttons)} tweets to unretweet...")
                    for btn in unretweet_buttons:
                        try:
                            btn.click(force=True)
                            time.sleep(1.0)

                            confirm = page.query_selector('[data-testid="confirmationSheetConfirm"]')
                            if confirm:
                                confirm.click(force=True)
                                time.sleep(0.5)
                                unretweeted += 1
                                print(f"    ✓ Unretweeted! (Total: {unretweeted})")
                        except:
                            continue

                # If no actions taken and no your tweets, we might be at the end
                actions_this_loop = your_tweets_found + len(unlike_buttons) + len(unretweet_buttons)
                if actions_this_loop == 0:
                    print("  ⚪ No actions this loop, scrolling...")
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(2)

                    # Check if we've scrolled past many tweets with no actions
                    if loops > 10 and actions_this_loop == 0:
                        print("  ⚠️ No actions for several loops, checking if done...")
                        # Count consecutive no-action loops could be added here
                else:
                    print(f"  ✓ Actions: {your_tweets_found} your tweets, {len(unlike_buttons)} unlikes, {len(unretweet_buttons)} unretweets")

                # Small scroll between loops
                page.evaluate("window.scrollBy(0, 200)")
                time.sleep(0.3)

            except Exception as e:
                print(f"✗ ERROR: {e}")
                failed += 1
                time.sleep(1)

        print("=" * 80)
        print("COMPLETE!")
        print("=" * 80)
        print(f"✓ Deleted: {deleted}")
        print(f"✓ Unliked: {unliked}")
        print(f"✓ Unretweeted: {unretweeted}")
        print(f"✗ Failed: {failed}")
        print("=" * 80)
        print()

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(0)
