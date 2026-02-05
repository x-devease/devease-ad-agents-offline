#!/usr/bin/env python3
"""
Full workflow test: Generate drafts → Type into browser

No UI prompts - automatically uses first draft.
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

print("\n" + "=" * 80)
print("Full Workflow Test: Generate Drafts → Type into Browser")
print("=" * 80)
print()

try:
    from src.growth.twitter.agents.content_agent import ContentAgent
    from src.growth.twitter.core.types import TwitterKeys, TwitterConfig, TwitterTask, TaskType
    from playwright.sync_api import sync_playwright
    import time
    import random

    # Step 1: Generate drafts
    print("Step 1: Generating AI drafts...")
    print("-" * 80)

    keys = TwitterKeys(openai_api_key=api_key, openai_org_id=org_id)
    config = TwitterConfig(llm_model="gpt-4o")

    content_agent = ContentAgent(keys, config)

    task = TwitterTask(
        id="test_full_workflow",
        type=TaskType.POST,
        idea="分享今天 Judge Model 发现的一个离谱案例：某电商在 3 点全投了垃圾流量，浪费 $500。",
        style="犀利吐槽，硬核数据"
    )

    drafts = content_agent.generate_drafts(task)

    print(f"\n✓ Generated {len(drafts)} drafts:\n")

    for i, draft in enumerate(drafts, 1):
        print(f"Draft {i} ({draft.version}):")
        print(f"  {draft.content}")
        print(f"  Rationale: {draft.rationale}")
        print()

    # Step 2: Connect to browser
    print("Step 2: Connecting to your Chrome browser...")
    print("-" * 80)

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
            twitter_page.goto("https://twitter.com/home")
            time.sleep(2)
        else:
            print(f"✓ Found existing Twitter tab")

        # Step 3: Type draft into browser
        print()
        print("Step 3: Typing draft into browser...")
        print("-" * 80)

        # Use first draft
        selected_draft = drafts[0]
        print(f"Selected draft: {selected_draft.version}")
        print(f"Content: {selected_draft.content}")
        print()

        # Find and click tweet box
        tweet_box = twitter_page.wait_for_selector('[data-testid="tweetTextarea_0"]', timeout=5000)
        tweet_box.click()
        time.sleep(0.5)

        # Type with human-like simulation
        print("📝 Typing with human-like delays...")
        for char in selected_draft.content:
            tweet_box.type(char)
            time.sleep(random.uniform(0.05, 0.12))

        print("✓ Draft typed into tweet box!")
        print()

        # Take screenshot
        screenshots_dir = Path("logs/twitter/screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = screenshots_dir / f"full_workflow_{timestamp}.png"

        twitter_page.screenshot(path=str(screenshot_path))
        print(f"📸 Screenshot saved: {screenshot_path}")

        # Keep browser open
        print()
        print("⏳ Browser will stay open for 15 seconds...")
        print("   Check your Chrome window - the draft is ready!")
        print()
        print("   The draft is NOT posted yet.")
        print("   You can review it and post manually when ready.")

        time.sleep(15)

        browser.close()

    print()
    print("=" * 80)
    print("✅ Full Workflow Test Complete!")
    print("=" * 80)
    print()
    print("What happened:")
    print("✓ Step 1: Generated 3 AI drafts using GPT-4o")
    print("✓ Step 2: Connected to your Chrome browser")
    print("✓ Step 3: Typed the first draft into Twitter")
    print("✓ Step 4: Took screenshot for verification")
    print()
    print("The draft is ready in your browser - go post it if you like!")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
