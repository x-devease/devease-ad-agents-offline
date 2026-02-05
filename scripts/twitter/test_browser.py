#!/usr/bin/env python3
"""
Simple test script to verify Twitter Growth Agent components.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 80)
    print("Testing Imports...")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.content_agent import ContentAgent
        print("✓ ContentAgent imported")
    except Exception as e:
        print(f"✗ ContentAgent failed: {e}")
        return False

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        print("✓ BrowserAgent imported")
    except Exception as e:
        print(f"✗ BrowserAgent failed: {e}")
        return False

    try:
        from src.growth.twitter.agents.ui_agent import UIAgent
        print("✓ UIAgent imported")
    except Exception as e:
        print(f"✗ UIAgent failed: {e}")
        return False

    try:
        from src.growth.twitter.agents.orchestrator import TwitterOrchestrator
        print("✓ TwitterOrchestrator imported")
    except Exception as e:
        print(f"✗ TwitterOrchestrator failed: {e}")
        return False

    print("\nAll imports successful!")
    return True


def test_browser_init():
    """Test browser agent initialization."""
    print("\n" + "=" * 80)
    print("Testing Browser Agent Initialization...")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.browser_agent import BrowserAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig

        keys = TwitterKeys(
            openai_api_key="test_key",
            openai_org_id="test_org"
        )

        config = TwitterConfig(
            llm_model="gpt-4"
        )

        # Initialize browser agent (headless)
        agent = BrowserAgent(keys, config, headless=True)
        print("✓ BrowserAgent initialized successfully")
        print(f"  - Headless mode: {agent.headless}")
        print(f"  - Config model: {config.llm_model}")

        return True

    except Exception as e:
        print(f"✗ BrowserAgent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yaml_parser():
    """Test YAML task parser."""
    print("\n" + "=" * 80)
    print("Testing YAML Task Parser...")
    print("=" * 80)

    try:
        from src.growth.twitter.core.yaml_parser import YAMLTaskParser
        from src.growth.twitter.core.types import TwitterConfig

        config = TwitterConfig(
            llm_model="gpt-4",
            tasks_path=Path("config/twitter/tasks.yaml")
        )

        parser = YAMLTaskParser("config/twitter/tasks.yaml", config)
        tasks = parser.load_tasks()

        print(f"✓ Loaded {len(tasks)} tasks from YAML")

        for task in tasks:
            print(f"\n  Task {task.id}:")
            print(f"    Type: {task.type.value}")
            print(f"    Status: {task.status.value}")
            print(f"    Idea: {task.idea[:50]}...")

        return True

    except Exception as e:
        print(f"✗ YAML parser failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_content_agent_basic():
    """Test content agent basic functionality (without real API calls)."""
    print("\n" + "=" * 80)
    print("Testing Content Agent (Basic)...")
    print("=" * 80)

    try:
        from src.growth.twitter.agents.content_agent import ContentAgent
        from src.growth.twitter.core.types import TwitterKeys, TwitterConfig, TwitterTask, TaskType

        keys = TwitterKeys(
            openai_api_key="test_key",  # Won't make real API calls in this test
            openai_org_id="test_org"
        )

        config = TwitterConfig(
            llm_model="gpt-4"
        )

        agent = ContentAgent(keys, config)
        print("✓ ContentAgent initialized")

        # Test fallback drafts (no API call needed)
        task = TwitterTask(
            id="test_001",
            type=TaskType.POST,
            idea="Test idea",
            style="professional"
        )

        drafts = agent._generate_fallback_drafts(task)
        print(f"✓ Generated {len(drafts)} fallback drafts")

        for i, draft in enumerate(drafts, 1):
            print(f"  Draft {i}: {draft.content[:50]}...")

        return True

    except Exception as e:
        print(f"✗ ContentAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Twitter Growth Agent - Component Tests")
    print("=" * 80)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Browser Agent Init
    results.append(("Browser Agent Init", test_browser_init()))

    # Test 3: YAML Parser
    results.append(("YAML Parser", test_yaml_parser()))

    # Test 4: Content Agent Basic
    results.append(("Content Agent Basic", test_content_agent_basic()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
