#!/usr/bin/env python3
"""
Test draft creation in existing browser (no prompts).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from playwright.sync_api import sync_playwright
import time
import random

print("=" * 80)
print("Create Draft in Existing Browser")
print("=" * 80)
print()

try:
    print("🔄 Connecting to Chrome on port 9222...")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        default_context = browser.contexts[0]

        # Find Twitter page
        twitter_page = None
        for page in default_context.pages:
            if 'twitter.com' in page.url or 'x.com' in page.url:
                twitter_page = page
                break

        if not twitter_page:
            print("Creating new Twitter tab...")
            twitter_page = default_context.new_page()
            twitter_page.goto("https://twitter.com")
            time.sleep(2)

        print(f"✓ Connected to: {twitter_page.url}")
        print(f"  Title: {twitter_page.title()}")

        # Look for tweet box
        print()
        print("🔍 Looking for tweet box...")

        tweet_box = twitter_page.wait_for_selector('[data-testid="tweetTextarea_0"]', timeout=5000)
        print("✓ Found tweet box!")

        # Click to focus
        tweet_box.click()
        time.sleep(0.5)

        # Type draft content
        draft_content = "Test draft from browser agent 🤖 - not posting!"
        print(f"📝 Typing draft: {draft_content}")

        # Type with human-like delays
        for char in draft_content:
            tweet_box.type(char)
            time.sleep(random.uniform(0.05, 0.12))

        print("✓ Draft typed into tweet box!")
        print()
        print("⚠️  Draft is ready in your browser!")
        print("   Check the Chrome window to see it.")
        print()
        print("   You can:")
        print("   - Review and edit the draft")
        print("   - Post it manually by clicking Tweet")
        print("   - Or just close the tab to discard")

        # Take screenshot
        screenshots_dir = Path("logs/twitter/screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = screenshots_dir / f"draft_{timestamp}.png"

        twitter_page.screenshot(path=str(screenshot_path), full_page=False)

        print()
        print(f"📸 Screenshot saved: {screenshot_path}")

        # Keep browser open for user to see
        print()
        print("⏳ Browser will stay open for 10 seconds...")
        print("   Go check your Chrome window!")
        time.sleep(10)

        browser.close()

    print()
    print("=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)
    print()
    print("The browser agent successfully:")
    print("✓ Connected to your existing Chrome")
    print("✓ Used your Twitter session")
    print("✓ Typed a draft into the tweet box")
    print("✓ Took a screenshot")
    print()
    print("Your draft is ready in Chrome - go post it if you like!")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
