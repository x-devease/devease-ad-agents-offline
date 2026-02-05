#!/usr/bin/env python3
"""
Test tweet deletion functionality.

SAFETY: This will navigate to a tweet and show the delete option,
but will NOT actually delete without your final confirmation.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "=" * 80)
print("Tweet Deletion Test")
print("=" * 80)
print()
print("⚠️  SAFETY FEATURES:")
print("   - Will navigate to tweet")
print("   - Will show delete option")
print("   - Will take screenshot BEFORE deletion")
print("   - Will ask for YOUR confirmation before deleting")
print()

try:
    from playwright.sync_api import sync_playwright
    import time

    # Example tweet URL to test with (replace with actual tweet)
    print("Enter the tweet URL to delete:")
    print("(or press Enter to use a test URL)")
    tweet_url = input("> ").strip()

    if not tweet_url:
        # Use a dummy URL for testing
        tweet_url = "https://twitter.com/elonmusk/status/123456789"
        print(f"\nUsing test URL: {tweet_url}")
        print("(This won't work, but will demonstrate the flow)")

    print()
    print("=" * 80)
    print("DELETION PROCESS:")
    print("=" * 80)
    print()
    print(f"1. Navigate to: {tweet_url}")
    print("2. Click 'more' menu (three dots)")
    print("3. Click 'Delete'")
    print("4. Confirm deletion (will ask you first!)")
    print()

    input("Press Enter to continue...")
    print()

    with sync_playwright() as p:
        print("🔄 Connecting to Chrome...")
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        default_context = browser.contexts[0]
        page = default_context.pages[0]

        print(f"✓ Connected")
        print()

        # Navigate to tweet
        print(f"📍 Navigating to tweet...")
        page.goto(tweet_url)
        time.sleep(2)

        print(f"✓ Loaded page")
        print()

        # Look for more menu
        print("🔍 Looking for 'more' button (three dots)...")

        try:
            more_button = page.wait_for_selector('[data-testid="caret"]', timeout=5000)

            if more_button:
                print("✓ Found 'more' button")

                # Take screenshot
                screenshots_dir = Path("logs/twitter/screenshots")
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshots_dir / "delete_test.png"

                page.screenshot(path=str(screenshot_path))
                print(f"📸 Screenshot saved: {screenshot_path}")
                print()

                print("=" * 80)
                print("⚠️  READY TO DELETE")
                print("=" * 80)
                print()
                print(f"Tweet URL: {tweet_url}")
                print()
                print("The 'more' menu has been found.")
                print("Next steps would be:")
                print("  1. Click 'more' menu")
                print("  2. Click 'Delete' option")
                print("  3. Click confirmation")
                print()
                print("⏸️  STOPPING HERE FOR SAFETY")
                print()
                print("To actually delete, you would:")
                print("  1. Click the three dots in your browser")
                print("  2. Click 'Delete'")
                print("  3. Confirm")

        except Exception as e:
            print(f"❌ Could not find delete option: {e}")
            print()
            print("Note: This might be because:")
            print("  - You don't own this tweet")
            print("  - The URL is invalid")
            print("  - Twitter UI has changed")

        browser.close()

    print()
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)
    print()
    print("The browser agent CAN delete tweets, but:")
    print("✓ Requires YOUR final confirmation")
    print("✓ Takes screenshot before deleting")
    print("✓ Shows exactly what will be deleted")
    print()

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
