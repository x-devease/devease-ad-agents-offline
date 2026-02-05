#!/usr/bin/env python3
"""
Non-stop like removal - clean all your likes.

This will:
1. Continuously scroll through your likes
2. Unlike each tweet
3. Auto-recover if navigated away
4. Continue until NO MORE LIKES FOUND or safety limit reached

Usage:
    python clean_likes.py --port 9224 --max 10000

    --port: Chrome remote debugging port (default 9223)
    --max: Safety limit (default 10000 likes)
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
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--max", type=int, default=10000, help="Maximum likes to process (safety limit)")
    parser.add_argument("--port", type=int, default=9223, help="Chrome debug port (default: 9223 for testing)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("NON-STOP Like Removal (Until all likes are removed)")
    print("=" * 80)
    print(f"Safety limit: {args.max} likes maximum")
    print()

    with sync_playwright() as p:
        print("🔄 Connecting...")
        browser = p.chromium.connect_over_cdp(f"http://localhost:{args.port}")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()
        print("✓ Connected")
        print()

        # Go to likes page
        print("📍 Navigating to your likes page...")
        page.goto("https://x.com/0xJ_builder/likes", timeout=15000)
        time.sleep(3)
        print(f"  ✓ On likes page")
        print()

        # Function to ensure we're on likes page
        def ensure_on_likes_page():
            """Check if on likes page, navigate back if not."""
            current_url = page.url

            # Check if navigated away
            if '/status/' in current_url or current_url == 'https://x.com/home' or '/i/' in current_url:
                print(f"  ⚠️  Navigated away! Current: {current_url}")
                print("  🔙 Returning to likes page...")
                page.goto("https://x.com/0xJ_builder/likes", timeout=15000)
                time.sleep(3)
                print("  ✓ Back on likes page")
                return True
            elif not current_url.startswith('https://x.com/0xJ_builder/likes'):
                print(f"  ⚠️  Wrong page! Current: {current_url}")
                print("  🔙 Going to likes page...")
                page.goto("https://x.com/0xJ_builder/likes", timeout=15000)
                time.sleep(3)
                print("  ✓ Back on likes page")
                return True

            return False

        print("=" * 80)
        print("Starting non-stop like removal...")
        print("Will continue until no more likes found or safety limit reached")
        print("=" * 80)
        print()

        success = 0
        failed = 0
        processed = 0
        no_unlike_count = 0  # Count tweets without unlike button
        max_no_unlike = 50  # Stop after 50 consecutive tweets already unliked

        while processed < args.max and no_unlike_count < max_no_unlike:
            try:
                # Check if we're still on likes page
                if ensure_on_likes_page():
                    time.sleep(2)
                    continue

                # Get current tweets
                tweets = page.query_selector_all('[data-testid="tweet"]')

                if len(tweets) == 0:
                    print(f"  No tweets found, loading more...")
                    page.evaluate("window.scrollBy(0, window.innerHeight)")
                    time.sleep(2)
                    continue

                # Process FIRST tweet (top of page)
                tweet = tweets[0]
                text_elem = tweet.query_selector('[data-testid="tweetText"]')
                content = text_elem.inner_text()[:50] if text_elem else "(No text)"

                processed += 1
                print(f"[Tweet #{processed}] {content}...")

                # Check if liked
                has_unlike = tweet.query_selector('[data-testid="unlike"]') is not None

                if has_unlike:
                    print("  💔 Liked - unliking...")
                    no_unlike_count = 0  # Reset counter
                    try:
                        unlike_btn = tweet.query_selector('[data-testid="unlike"]')
                        if unlike_btn:
                            unlike_btn.click(force=True)
                            time.sleep(1)  # Wait for Twitter to process
                            print("    ✓ Unliked!")
                            success += 1
                        else:
                            print("    ✗ Button not found")
                            failed += 1
                    except Exception as e:
                        print(f"    ✗ Error: {e}")
                        failed += 1
                else:
                    print("  ⚪ Already unliked")
                    no_unlike_count += 1
                    print(f"  ({no_unlike_count}/{max_no_unlike} unliked)")

                # Scroll ONE tweet down
                page.evaluate("window.scrollBy(0, 100)")
                time.sleep(0.2)

                print()

            except Exception as e:
                print(f"✗ ERROR: {e}")
                failed += 1
                # No wait - immediate continue

        print("=" * 80)
        print("NON-STOP LIKE REMOVAL COMPLETE!")
        print("=" * 80)
        print()
        print(f"Total likes processed: {processed}")
        print(f"✓ Successfully unliked: {success}")
        print(f"✗ Failed: {failed}")
        print()

        if processed >= args.max:
            print(f"⚠️  Reached safety limit of {args.max} likes")
            print("   Run again to continue cleaning")
        else:
            print("✅ All likes removed! No more likes found.")
        print("=" * 80)
        print()

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(0)
