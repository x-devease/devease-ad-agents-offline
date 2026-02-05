"""
UI Agent for Twitter Growth Agent.

Provides CLI interface for human-in-the-loop workflow.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

from ..core.types import TwitterTask, TwitterDraft, TwitterKeys, TwitterConfig

logger = logging.getLogger(__name__)


@dataclass
class UserSelection:
    """Result of user selection."""
    action: str  # 'confirm', 'regenerate', 'edit', 'skip'
    selected_index: int
    edited_content: Optional[str] = None
    feedback: Optional[str] = None


class UIAgent:
    """
    Manage CLI interface for human-in-the-loop workflow.

    Responsibilities:
    - Present generated drafts to user
    - Get user selection from drafts
    - Request confirmation before posting
    - Enable draft editing
    - Collect feedback on drafts
    """

    def __init__(self, keys: TwitterKeys, config: TwitterConfig):
        """
        Initialize UI agent.

        Args:
            keys: TwitterKeys object
            config: TwitterConfig object
        """
        self.keys = keys
        self.config = config

    def present_drafts(self, task: TwitterTask, drafts: List[TwitterDraft]) -> UserSelection:
        """
        Present drafts to user and get selection.

        Args:
            task: TwitterTask being processed
            drafts: List of 3 generated drafts

        Returns:
            UserSelection with user's choice
        """
        print("\n" + "=" * 80)
        print(f"📝 Task {task.id}: {task.type.value}")
        print("=" * 80)
        print(f"Idea: {task.idea}")
        print(f"Style: {task.style}")
        print()

        # Display all drafts
        for i, draft in enumerate(drafts, 1):
            print(f"\n{'─' * 80}")
            print(f"📄 Draft {i} - {draft.version}")
            print(f"{'─' * 80}")
            print(f"Content: {draft.content}")
            print()
            print(f"Rationale: {draft.rationale}")
            print(f"Tone: {draft.tone}")
            if draft.hashtags:
                print(f"Hashtags: {', '.join(draft.hashtags)}")
            print(f"Character count: {draft.character_count or len(draft.content)}")
            print()

        # Get user selection
        while True:
            print("\n" + "─" * 80)
            print("Choose an option:")
            print("  1, 2, 3 - Select draft to post")
            print("  0 - Skip this task")
            print("  r - Regenerate all drafts")
            print("  e - Edit a draft before posting")
            print("─" * 80)

            choice = input("\nYour choice: ").strip().lower()

            if choice == '0':
                return UserSelection(
                    action='skip',
                    selected_index=-1
                )
            elif choice == 'r':
                return UserSelection(
                    action='regenerate',
                    selected_index=-1
                )
            elif choice == 'e':
                # Get draft index to edit
                draft_num = input("Which draft to edit? (1-3): ").strip()
                if draft_num in ['1', '2', '3']:
                    draft_index = int(draft_num) - 1
                    edited_content = self._edit_draft(drafts[draft_index])
                    if edited_content:
                        return UserSelection(
                            action='edit',
                            selected_index=draft_index,
                            edited_content=edited_content
                        )
                else:
                    print("❌ Invalid draft number")
            elif choice in ['1', '2', '3']:
                draft_index = int(choice) - 1

                # Show confirmation
                if self._confirm_selection(drafts[draft_index]):
                    return UserSelection(
                        action='confirm',
                        selected_index=draft_index
                    )
                else:
                    print("❌ Cancelled. Please select another option.")
            else:
                print("❌ Invalid choice. Please try again.")

    def _edit_draft(self, draft: TwitterDraft) -> Optional[str]:
        """
        Allow user to edit a draft.

        Args:
            draft: Draft to edit

        Returns:
            Edited content or None if cancelled
        """
        print("\n" + "─" * 80)
        print(f"✏️  Editing Draft {draft.version}")
        print("─" * 80)
        print("Current content:")
        print(draft.content)
        print("\n" + "─" * 80)
        print("Enter new content (press Enter twice to finish, or 'cancel' to cancel):")
        print("─" * 80)

        lines = []
        empty_line_count = 0

        while True:
            line = input()
            if line.lower() == 'cancel':
                return None
            elif line == '':
                empty_line_count += 1
                if empty_line_count >= 2:
                    break
            else:
                empty_line_count = 0
                lines.append(line)

        edited_content = '\n'.join(lines).strip()

        if edited_content:
            print(f"\n✅ Draft updated. New length: {len(edited_content)} characters")
            return edited_content
        else:
            print("\n❌ Empty content. Edit cancelled.")
            return None

    def _confirm_selection(self, draft: TwitterDraft) -> bool:
        """
        Get user confirmation before posting.

        Args:
            draft: Draft to confirm

        Returns:
            True if user confirms, False otherwise
        """
        print("\n" + "=" * 80)
        print("📤 Ready to Post")
        print("=" * 80)
        print(draft.content)
        print("=" * 80)
        print(f"Character count: {len(draft.content)}")
        print("=" * 80)

        while True:
            choice = input("\nPost this draft? (yes/no): ").strip().lower()
            if choice in ['yes', 'y', '']:
                return True
            elif choice in ['no', 'n']:
                return False
            else:
                print("Please enter 'yes' or 'no'")

    def request_regeneration_feedback(self, rejected_draft: TwitterDraft) -> Optional[str]:
        """
        Request feedback on why draft was rejected.

        Args:
            rejected_draft: Draft that was rejected

        Returns:
            Feedback string or None
        """
        print("\n" + "─" * 80)
        print("💬 Optional: Provide feedback for regeneration")
        print("─" * 80)
        print("Why didn't this draft work? (Enter to skip)")
        print(f"Rejected draft: {rejected_draft.content[:100]}...")
        print("─" * 80)

        feedback = input().strip()
        return feedback if feedback else None

    def display_task_summary(self, task: TwitterTask, result: dict):
        """
        Display summary after task completion.

        Args:
            task: Completed task
            result: Result dict with status and metadata
        """
        print("\n" + "=" * 80)
        print("✅ Task Completed")
        print("=" * 80)
        print(f"Task ID: {task.id}")
        print(f"Type: {task.type.value}")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Idea: {task.idea}")

        if result.get('tweet_url'):
            print(f"Tweet URL: {result['tweet_url']}")

        if result.get('error'):
            print(f"Error: {result['error']}")

        print("=" * 80)

    def display_batch_summary(self, tasks: List[TwitterTask], results: List[dict]):
        """
        Display summary after batch processing.

        Args:
            tasks: List of processed tasks
            results: List of result dicts
        """
        print("\n" + "=" * 80)
        print("📊 Batch Processing Summary")
        print("=" * 80)

        total = len(tasks)
        successful = sum(1 for r in results if r.get('status') == 'completed')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        skipped = sum(1 for r in results if r.get('status') == 'skipped')

        print(f"Total tasks: {total}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        print("=" * 80)

        # Show failed tasks
        if failed > 0:
            print("\nFailed Tasks:")
            for task, result in zip(tasks, results):
                if result.get('status') == 'failed':
                    print(f"  - {task.id}: {result.get('error', 'Unknown error')}")

        print()

    def display_progress(self, current: int, total: int, task_id: str):
        """
        Display progress during batch processing.

        Args:
            current: Current task number
            total: Total number of tasks
            task_id: Current task ID
        """
        percentage = (current / total) * 100
        print(f"\n⏳ Progress: [{current}/{total}] {percentage:.0f}% - Task {task_id}")

    def confirm_batch_start(self, tasks: List[TwitterTask]) -> bool:
        """
        Confirm before starting batch processing.

        Args:
            tasks: List of tasks to process

        Returns:
            True if user confirms, False otherwise
        """
        print("\n" + "=" * 80)
        print("🚀 Ready to Process Tasks")
        print("=" * 80)
        print(f"Total tasks: {len(tasks)}")
        print()

        # Group by type
        from collections import Counter
        task_types = Counter(task.type.value for task in tasks)
        for task_type, count in task_types.most_common():
            print(f"  {task_type}: {count}")

        print("=" * 80)

        choice = input("\nStart processing? (yes/no): ").strip().lower()
        return choice in ['yes', 'y', '']

    def handle_error(self, error: Exception, task: TwitterTask) -> str:
        """
        Handle error and get user input on how to proceed.

        Args:
            error: Exception that occurred
            task: Task being processed

        Returns:
            User's choice: 'retry', 'skip', 'abort'
        """
        print("\n" + "=" * 80)
        print("❌ Error Occurred")
        print("=" * 80)
        print(f"Task: {task.id}")
        print(f"Error: {str(error)}")
        print("=" * 80)
        print("How would you like to proceed?")
        print("  1 - Retry this task")
        print("  2 - Skip this task and continue")
        print("  3 - Abort batch processing")
        print("=" * 80)

        while True:
            choice = input("\nYour choice: ").strip()
            if choice == '1':
                return 'retry'
            elif choice == '2':
                return 'skip'
            elif choice == '3':
                return 'abort'
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

    def collect_post_feedback(self, task: TwitterTask, draft: TwitterDraft) -> Optional[str]:
        """
        Collect optional feedback after posting.

        Args:
            task: Completed task
            draft: Draft that was posted

        Returns:
            Feedback string or None
        """
        print("\n" + "─" * 80)
        print("💬 Optional Feedback")
        print("─" * 80)
        print("Any thoughts on this post? (Enter to skip)")
        print("This helps improve future content generation.")
        print("─" * 80)

        feedback = input().strip()
        return feedback if feedback else None
