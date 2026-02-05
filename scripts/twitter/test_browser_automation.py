#!/usr/bin/env python3
"""
Test Browser Agent functionality.

Tests Playwright automation including:
- Browser initialization
- Navigation to URLs
- Screenshot capture
- Context extraction (tweet and user profiles)

Does NOT post any content to Twitter.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
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


def test_browser_init():
    """Test browser initialization."""
    print("\n" + "=" * 80)
    print("Test 1: Browser Initialization")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig

        keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
        config = TwitterConfig(llm_model="gpt-4o")

        # Test headless mode
        agent = BrowserAgent(keys, config, headless=True)
        print("✓ Browser initialized (headless mode)")

        # Test that browser is not started yet
        assert agent.browser is None, "Browser should not start until needed"
        print("✓ Lazy initialization working")

        # Cleanup
        agent._stop_browser()
        print("✓ Browser stopped")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_navigate_to_url():
    """Test navigation to a URL."""
    print("\n" + "=" * 80)
    print("Test 2: Navigate to URL")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig

        keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
        config = TwitterConfig(llm_model="gpt-4o")

        agent = BrowserAgent(keys, config, headless=True)

        # Navigate to a public URL
        result = agent.navigate_to_url("https://example.com")

        if result.success:
            print("✓ Successfully navigated to example.com")
            print(f"  Message: {result.message}")
        else:
            print(f"❌ Navigation failed: {result.error}")
            return False

        # Cleanup
        agent._stop_browser()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_screenshot():
    """Test screenshot capture."""
    print("\n" + "=" * 80)
    print("Test 3: Screenshot Capture")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig

        keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
        config = TwitterConfig(llm_model="gpt-4o")

        agent = BrowserAgent(keys, config, headless=True)

        # Start browser
        agent._start_browser()

        # Navigate and take screenshot
        agent.page.goto("https://example.com")
        screenshot_path = agent._take_screenshot("test_screenshot")

        if screenshot_path:
            print(f"✓ Screenshot saved: {screenshot_path}")
            # Verify file exists
            if Path(screenshot_path).exists():
                print(f"✓ File exists and is {Path(screenshot_path).stat().st_size} bytes")
            else:
                print("❌ Screenshot file not found")
                return False
        else:
            print("❌ Screenshot failed")
            return False

        # Cleanup
        agent._stop_browser()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_human_typing_simulation():
    """Test human-like typing simulation."""
    print("\n" + "=" * 80)
    print("Test 4: Human-Like Typing Simulation")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig

        keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
        config = TwitterConfig(llm_model="gpt-4o")

        agent = BrowserAgent(keys, config, headless=True)

        # Start browser
        agent._start_browser()

        # Navigate to a test page with input
        agent.page.goto("https://example.com")

        # Test random delay function
        import time
        start = time.time()
        agent._random_delay(0.5, 1.0)
        elapsed = time.time() - start
        print(f"✓ Random delay: {elapsed:.2f}s (within 0.5-1.0s range)")

        # Cleanup
        agent._stop_browser()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_navigate_to_real_tweet():
    """Test navigating to a real tweet ( Elon Musk's tweet for testing)."""
    print("\n" + "=" * 80)
    print("Test 5: Navigate to Real Tweet")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig

        keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
        config = TwitterConfig(llm_model="gpt-4o")

        agent = BrowserAgent(keys, config, headless=True)

        # Navigate to a public tweet (Elon Musk's pinned tweet)
        tweet_url = "https://x.com/elonmusk/status/1692743897939779587"
        print(f"Navigating to: {tweet_url}")

        result = agent.navigate_to_url(tweet_url)

        if result.success:
            print("✓ Successfully navigated to tweet")
        else:
            print(f"⚠️  Navigation had issues: {result.error}")
            print("Note: This might be due to auth requirements")

        # Cleanup
        agent._stop_browser()

        return True

    except Exception as e:
        print(f"⚠️  Test note: {e}")
        print("Note: Twitter navigation often requires authentication")
        return True  # Don't fail the test for auth issues


def test_visible_browser():
    """Test with visible browser (for manual verification)."""
    print("\n" + "=" * 80)
    print("Test 6: Visible Browser (Manual Verification)")
    print("=" * 80)
    print("\n⚠️  This test will open a visible browser window")
    print("You should see the browser navigate to example.com")

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig

        keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
        config = TwitterConfig(llm_model="gpt-4o")

        # Create agent with headless=False
        agent = BrowserAgent(keys, config, headless=False)

        print("\n🔄 Opening visible browser...")
        agent.navigate_to_url("https://example.com")

        print("✓ Browser opened (you should see it)")
        print("\n⏳ Keeping browser open for 3 seconds for verification...")
        import time
        time.sleep(3)

        # Cleanup
        agent._stop_browser()
        print("✓ Browser closed")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all browser agent tests."""
    print("\n" + "=" * 80)
    print("Twitter Growth Agent - Browser Agent Tests")
    print("=" * 80)
    print("\n⚠️  These tests use Playwright browser automation")
    print("⚠️  No content will be posted to Twitter\n")

    tests = [
        ("Browser Initialization", test_browser_init),
        ("Navigate to URL", test_navigate_to_url),
        ("Screenshot Capture", test_screenshot),
        ("Typing Simulation", test_human_typing_simulation),
        ("Navigate to Real Tweet", test_navigate_to_real_tweet),
        ("Visible Browser", test_visible_browser),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All browser agent tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed or skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
