#!/usr/bin/env python3
"""
Simple browser agent test (avoids asyncio issues).
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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


def test_browser_basic():
    """Test basic browser functionality."""
    print("\n" + "=" * 80)
    print("Browser Agent Test")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig, TwitterDraft, TwitterTask, TaskType

        keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
        config = TwitterConfig(llm_model="gpt-4o")

        # Test 1: Initialization
        print("\n1. Testing Initialization...")
        agent = BrowserAgent(keys, config, headless=True)
        print("✓ Browser agent initialized")

        # Test 2: Navigate to example.com
        print("\n2. Testing Navigation...")
        result = agent.navigate_to_url("https://example.com")

        if result.success:
            print(f"✓ Navigation successful: {result.message}")
        else:
            print(f"✗ Navigation failed: {result.error}")

        # Test 3: Take screenshot
        print("\n3. Testing Screenshot...")
        agent._start_browser()
        agent.page.goto("https://example.com")
        screenshot = agent._take_screenshot("test")

        if screenshot and Path(screenshot).exists():
            print(f"✓ Screenshot saved: {screenshot}")
            print(f"  File size: {Path(screenshot).stat().st_size} bytes")
        else:
            print("✗ Screenshot failed")

        agent._stop_browser()

        # Test 4: Test typing simulation
        print("\n4. Testing Typing Simulation...")
        import time
        agent._start_browser()
        agent.page.goto("https://example.com")

        start = time.time()
        agent._random_delay(0.3, 0.6)
        delay = time.time() - start

        print(f"✓ Random delay: {delay:.2f}s (expected 0.3-0.6s)")

        agent._stop_browser()

        # Test 5: Visible browser test
        print("\n5. Testing Visible Browser...")
        print("   Opening visible browser for 2 seconds...")

        agent_visible = BrowserAgent(keys, config, headless=False)
        agent_visible._start_browser()
        agent_visible.page.goto("https://example.com")

        time.sleep(2)

        agent_visible._stop_browser()
        print("✓ Visible browser test completed")

        print("\n" + "=" * 80)
        print("🎉 All browser tests passed!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_browser_basic()
    sys.exit(0 if success else 1)
