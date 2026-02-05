"""
Test connecting to existing Chrome browser with user's session.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load keys silently
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


def test_connect_to_existing_browser():
    """Test connecting to existing Chrome browser."""
    print("\n" + "=" * 80)
    print("Connect to Existing Chrome Browser")
    print("=" * 80)
    print()
    print("INSTRUCTIONS:")
    print("1. Make sure Chrome is open with your Twitter session")
    print("2. Run Chrome with remote debugging enabled:")
    print("   /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome")
    print("   --remote-debugging-port=9222")
    print()
    print("Or if Chrome is already running:")
    print("   1. Open a new terminal")
    print("   2. Run: /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug")
    print()

    try:
        from playwright.sync_api import sync_playwright

        print("🔄 Connecting to Chrome on port 9222...")

        with sync_playwright() as p:
            # Connect to existing Chrome instance
            browser = p.chromium.connect_over_cdp("http://localhost:9222")

            print("✓ Connected to existing Chrome!")

            # Get existing context/page
            default_context = browser.contexts[0]
            page = default_context.pages[0]

            print(f"✓ Found {len(default_context.pages)} page(s)")
            print(f"  Current URL: {page.url}")
            print(f"  Page title: {page.title()}")

            # Navigate to Twitter
            print()
            print("🔄 Navigating to Twitter...")

            # Find or create Twitter page
            twitter_page = None
            for existing_page in default_context.pages:
                if 'twitter.com' in existing_page.url or 'x.com' in existing_page.url:
                    twitter_page = existing_page
                    print(f"✓ Found existing Twitter tab: {existing_page.url}")
                    break

            if not twitter_page:
                print("Creating new Twitter tab...")
                twitter_page = default_context.new_page()
                twitter_page.goto("https://twitter.com")
                print("✓ Navigated to Twitter")

            print()
            print("Current page info:")
            print(f"  URL: {twitter_page.url}")
            print(f"  Title: {twitter_page.title()}")
            print(f"  Cookies: {len(default_context.cookies())} cookies available")

            print()
            print("✅ Successfully connected to your Chrome session!")
            print("   The browser agent can now use your existing Twitter login.")

            # Don't close the browser, just disconnect
            browser.close()

        return True

    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure Chrome is running with --remote-debugging-port=9222")
        print("2. Check if port 9222 is available: lsof -i :9222")
        print("3. Try launching Chrome with:")
        print("   /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome")
        print("   --remote-debugging-port=9222")
        return False


def test_create_draft_in_browser():
    """Test creating a draft tweet in the browser (without posting)."""
    print("\n" + "=" * 80)
    print("Create Draft in Browser")
    print("=" * 80)

    try:
        from playwright.sync_api import sync_playwright
        import time

        print("🔄 Connecting to Chrome...")

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            default_context = browser.contexts[0]

            # Find Twitter page or create new one
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

            print(f"✓ On Twitter page: {twitter_page.url}")

            # Look for tweet box
            print()
            print("🔍 Looking for tweet box...")

            # Wait for tweet box
            try:
                tweet_box = twitter_page.wait_for_selector('[data-testid="tweetTextarea_0"]', timeout=5000)
                print("✓ Found tweet box!")

                # Click on it to focus
                tweet_box.click()
                time.sleep(0.5)

                # Type draft content (without posting)
                draft_content = "This is a test draft - not posting yet! 🧪"
                print(f"📝 Typing: {draft_content}")

                # Type with human-like delays
                import random
                for char in draft_content:
                    tweet_box.type(char)
                    time.sleep(random.uniform(0.05, 0.12))

                print("✓ Draft typed into tweet box!")
                print()
                print("⚠️  Draft is ready but NOT posted.")
                print("   You can:")
                print("   - Review it in the browser")
                print("   - Edit it manually")
                print("   - Post it yourself when ready")
                print("   - Or close the tab to discard")

                # Take screenshot
                screenshots_dir = Path("logs/twitter/screenshots")
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshots_dir / "draft_created.png"
                twitter_page.screenshot(path=str(screenshot_path))
                print(f"\n📸 Screenshot saved: {screenshot_path}")

            except Exception as e:
                print(f"❌ Could not find tweet box: {e}")
                print("Note: You might need to log in to Twitter first")
                return False

            browser.close()

        print("\n✅ Draft creation test complete!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests."""
    print("\n" + "=" * 80)
    print("Browser Agent - Existing Browser Connection")
    print("=" * 80)

    # First test: connection
    result1 = test_connect_to_existing_browser()

    if result1:
        # Second test: create draft
        input("\nPress Enter to test draft creation...")
        result2 = test_create_draft_in_browser()

        if result2:
            print("\n" + "=" * 80)
            print("🎉 All tests passed!")
            print("=" * 80)
            print("\nThe browser agent can:")
            print("✓ Connect to your existing Chrome browser")
            print("✓ Use your Twitter session (no login needed)")
            print("✓ Navigate to Twitter")
            print("✓ Type drafts into tweet box")
            print("✓ Take screenshots for verification")
            print("\nNext: Implement full posting workflow with human confirmation!")
            return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
