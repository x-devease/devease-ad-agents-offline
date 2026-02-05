#!/usr/bin/env python3
"""
Demo script for Twitter Growth Agent.

This script demonstrates the Twitter Growth Agent functionality without
actually posting to Twitter (use --no-post flag).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def demo_content_generation():
    """Demo: Content generation only."""
    print("\n" + "=" * 80)
    print("DEMO: Content Generation")
    print("=" * 80)

    from src.growth.twitter.agents.content_agent import ContentAgent
    from src.growth.twitter.core.key_manager import KeyManager
    from src.growth.twitter.core.types import TwitterTask, TwitterConfig, TaskType

    # Load keys
    key_manager = KeyManager()
    keys = key_manager.load_keys()

    # Create config
    config = TwitterConfig(
        llm_model="gpt-4",
        tasks_path=Path("config/twitter/tasks.yaml"),
    )

    # Create content agent
    agent = ContentAgent(keys, config)

    # Create demo task
    task = TwitterTask(
        id="demo_001",
        type=TaskType.POST,
        idea="分享今天在广告投放中发现的一个有趣模式：提高ROAS的3个反直觉技巧",
        style="犀利吐槽，硬核数据"
    )

    print(f"\nTask: {task.idea}")
    print(f"Style: {task.style}\n")

    # Generate drafts
    drafts = agent.generate_drafts(task)

    print(f"\nGenerated {len(drafts)} drafts:")
    for i, draft in enumerate(drafts, 1):
        print(f"\n{'─' * 80}")
        print(f"Draft {i} - {draft.version}")
        print(f"{'─' * 80}")
        print(draft.content)
        print(f"\nRationale: {draft.rationale}")
        print(f"Tone: {draft.tone}")
        if draft.hashtags:
            print(f"Hashtags: {', '.join(draft.hashtags)}")


def demo_full_workflow():
    """Demo: Full workflow with user interaction (no actual posting)."""
    print("\n" + "=" * 80)
    print("DEMO: Full Workflow (Dry Run)")
    print("=" * 80)

    from src.growth.twitter.agents.orchestrator import TwitterOrchestrator
    from src.growth.twitter.core.key_manager import KeyManager
    from src.growth.twitter.core.types import TwitterConfig
    from src.growth.twitter.core.memory import MemorySystem
    from pathlib import Path

    # Load keys
    key_manager = KeyManager()
    keys = key_manager.load_keys()

    # Create config
    config = TwitterConfig(
        llm_model="gpt-4",
        tasks_path=Path("config/twitter/tasks.yaml"),
    )

    # Initialize memory system
    memory = MemorySystem()

    # Initialize orchestrator with headless=False for demo
    orchestrator = TwitterOrchestrator(
        keys=keys,
        config=config,
        memory=memory,
        headless=True  # Run in headless mode
    )

    print("\nNote: This is a dry run. No actual tweets will be posted.")
    print("To post for real, use: python run.py twitter run\n")


def main():
    """Main demo entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Twitter Growth Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --content-only      # Demo content generation only
  python demo.py --full-workflow     # Demo full workflow (dry run)
        """,
    )

    parser.add_argument(
        "--content-only",
        action="store_true",
        help="Demo content generation only",
    )

    parser.add_argument(
        "--full-workflow",
        action="store_true",
        help="Demo full workflow (dry run)",
    )

    args = parser.parse_args()

    if not any([args.content_only, args.full_workflow]):
        # Default: show both demos
        demo_content_generation()
        demo_full_workflow()
    elif args.content_only:
        demo_content_generation()
    elif args.full_workflow:
        demo_full_workflow()


if __name__ == "__main__":
    main()
