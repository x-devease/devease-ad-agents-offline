#!/usr/bin/env python3
"""
Test OpenAI API integration with real credentials.
Keys are loaded but never displayed.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Suppress logging for cleaner output
import logging
logging.basicConfig(level=logging.ERROR)


def load_keys_silently():
    """Load keys without displaying them."""
    keys_path = Path.home() / ".devease" / "keys"

    if not keys_path.exists():
        print("❌ Keys file not found")
        return None

    env_vars = {}
    with open(keys_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()

    return {
        'openai_api_key': env_vars.get('OPENAI_API_KEY', ''),
        'openai_org_id': env_vars.get('OPENAI_ORG_ID', ''),
    }


def test_openai_api(keys):
    """Test OpenAI API with loaded keys."""
    print("\n" + "=" * 80)
    print("Testing OpenAI API Integration")
    print("=" * 80)

    try:
        from openai import OpenAI

        # Initialize client (keys not displayed)
        client = OpenAI(
            api_key=keys['openai_api_key'],
            organization=keys.get('openai_org_id')
        )

        print("✓ OpenAI client initialized")

        # Test API call
        print("\n🔄 Testing API call...")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Say 'API test successful' in exactly those words."
                }
            ],
            max_tokens=50
        )

        content = response.choices[0].message.content
        print(f"✓ API Response: {content}")

        return True

    except Exception as e:
        print(f"❌ API Test Failed: {str(e)}")
        return False


def test_content_generation(keys):
    """Test content generation with real Twitter task."""
    print("\n" + "=" * 80)
    print("Testing Content Generation")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.content_agent import ContentAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig, TwitterTask, TaskType

        # Create TwitterKeys (not displayed)
        twitter_keys = TwitterKeys(
            openai_api_key=keys['openai_api_key'],
            openai_org_id=keys.get('openai_org_id')
        )

        config = TwitterConfig(llm_model="gpt-4o")

        # Initialize agent
        agent = ContentAgent(twitter_keys, config)
        print("✓ ContentAgent initialized")

        # Create test task
        task = TwitterTask(
            id="test_api",
            type=TaskType.POST,
            idea="分享今天在广告投放中发现的一个有趣模式：提高ROAS的3个反直觉技巧",
            style="犀利吐槽，硬核数据"
        )

        print(f"\n🔄 Generating drafts for: {task.idea[:40]}...")

        # Generate drafts
        drafts = agent.generate_drafts(task)

        print(f"\n✓ Generated {len(drafts)} drafts:\n")

        for i, draft in enumerate(drafts, 1):
            print(f"{'─' * 80}")
            print(f"Draft {i} - {draft.version}")
            print(f"{'─' * 80}")
            print(f"Content: {draft.content}")
            print(f"\nRationale: {draft.rationale}")
            print(f"Tone: {draft.tone}")
            if draft.hashtags:
                print(f"Hashtags: {', '.join(draft.hashtags)}")
            print()

        return True

    except Exception as e:
        print(f"❌ Content Generation Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Twitter Growth Agent - API Test")
    print("=" * 80)
    print("\n⚠️  Keys loaded from ~/.devease/keys")
    print("   Keys will NOT be displayed in output\n")

    # Load keys
    keys = load_keys_silently()
    if not keys:
        return 1

    if not keys.get('openai_api_key'):
        print("❌ OPENAI_API_KEY not found in keys file")
        return 1

    print("✓ Keys loaded successfully")

    # Test 1: Basic API call
    test1 = test_openai_api(keys)

    # Test 2: Content generation
    test2 = test_content_generation(keys)

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    if test1:
        print("✓ OpenAI API: Working")
    else:
        print("❌ OpenAI API: Failed")

    if test2:
        print("✓ Content Generation: Working")
    else:
        print("❌ Content Generation: Failed")

    if test1 and test2:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
