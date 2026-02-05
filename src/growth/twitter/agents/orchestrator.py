"""
Orchestrator for Twitter Growth Agent.

Coordinates all agents and manages the main workflow.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from ..core.types import TwitterTask, TwitterDraft, TwitterKeys, TwitterConfig, TaskStatus
from ..core.yaml_parser import YAMLTaskParser
from ..core.memory import MemorySystem
from .content_agent import ContentAgent
from .browser_agent import BrowserAgent
from .ui_agent import UIAgent

logger = logging.getLogger(__name__)


class TwitterOrchestrator:
    """
    Main orchestrator for Twitter Growth Agent.

    Responsibilities:
    - Coordinate all agents (Content, Browser, UI)
    - Manage task queue from tasks.yaml
    - Execute end-to-end workflow
    - Handle errors and retries
    - Update task status in YAML
    - Record performance data
    """

    def __init__(
        self,
        keys: TwitterKeys,
        config: TwitterConfig,
        memory: Optional[MemorySystem] = None,
        headless: bool = True
    ):
        """
        Initialize orchestrator.

        Args:
            keys: TwitterKeys object
            config: TwitterConfig object
            memory: Optional MemorySystem for learning
            headless: Whether to run browser in headless mode
        """
        self.keys = keys
        self.config = config
        self.memory = memory

        # Initialize agents
        self.content_agent = ContentAgent(keys, config, memory)
        self.browser_agent = BrowserAgent(keys, config, headless=headless)
        self.ui_agent = UIAgent(keys, config)

        # Task management
        self.task_parser = YAMLTaskParser(str(config.tasks_path), config)

        logger.info("Twitter Orchestrator initialized")

    def run_single_task(self, task: TwitterTask) -> Dict[str, Any]:
        """
        Execute a single task from start to finish.

        Args:
            task: TwitterTask to execute

        Returns:
            Result dict with status, tweet_url, error (if any)
        """
        logger.info(f"Starting task {task.id}: {task.type.value}")
        self.ui_agent.display_progress(1, 1, task.id)

        try:
            # Step 1: Generate drafts
            logger.info("Generating drafts...")
            context = self._build_context(task)
            drafts = self.content_agent.generate_drafts(task, context)

            if not drafts:
                raise Exception("Failed to generate drafts")

            # Step 2: Present drafts to user
            selection = self.ui_agent.present_drafts(task, drafts)

            # Handle user actions
            if selection.action == 'skip':
                return self._skip_task(task, "User skipped")

            elif selection.action == 'regenerate':
                # Get feedback and regenerate
                feedback = self.ui_agent.request_regeneration_feedback(
                    drafts[selection.selected_index]
                )
                new_draft = self.content_agent.regenerate_draft(
                    task,
                    selection.selected_index,
                    feedback
                )

                # Update drafts with regenerated version
                drafts = [new_draft]

                # Present regenerated draft
                selection = self.ui_agent.present_drafts(task, drafts)

                if selection.action == 'skip':
                    return self._skip_task(task, "User skipped after regeneration")

            # Step 3: Handle editing
            if selection.action == 'edit' and selection.edited_content:
                draft = drafts[selection.selected_index]
                draft.content = selection.edited_content
                draft.version = f"{draft.version}_edited"

            # Step 4: Get final draft
            selected_draft = drafts[selection.selected_index]

            # Step 5: Post to Twitter
            logger.info(f"Posting {task.type.value}...")
            result = self._post_to_twitter(task, selected_draft)

            if not result.success:
                raise Exception(result.error or "Failed to post")

            # Step 6: Record success
            self._complete_task(task, selected_draft, result.tweet_url)
            self._record_feedback(task, selected_draft, 'confirmed')

            # Step 7: Collect optional feedback
            feedback = self.ui_agent.collect_post_feedback(task, selected_draft)
            if feedback:
                self.content_agent.record_feedback(
                    selected_draft,
                    'confirmed',
                    feedback
                )

            return {
                'status': 'completed',
                'tweet_url': result.tweet_url,
                'draft_version': selected_draft.version
            }

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            return self._fail_task(task, str(e))

    def run_batch(self, max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Execute all pending tasks in batch.

        Args:
            max_tasks: Maximum number of tasks to process (None = all)

        Returns:
            List of result dicts
        """
        # Load pending tasks
        tasks = self.task_parser.load_tasks()

        if max_tasks:
            tasks = tasks[:max_tasks]

        if not tasks:
            logger.info("No pending tasks to process")
            return []

        # Confirm batch start
        if not self.ui_agent.confirm_batch_start(tasks):
            logger.info("Batch processing cancelled by user")
            return []

        results = []
        for i, task in enumerate(tasks, 1):
            self.ui_agent.display_progress(i, len(tasks), task.id)

            # Execute task
            result = self.run_single_task(task)
            results.append(result)

            # Handle user choice after error
            if result.get('status') == 'failed':
                choice = self.ui_agent.handle_error(
                    Exception(result.get('error', 'Unknown error')),
                    task
                )

                if choice == 'retry':
                    # Retry task
                    logger.info(f"Retrying task {task.id}")
                    result = self.run_single_task(task)
                    results.append(result)

                elif choice == 'abort':
                    # Abort batch
                    logger.info("Batch processing aborted by user")
                    break

                # else: skip and continue

        # Display summary
        self.ui_agent.display_batch_summary(tasks, results)

        return results

    def _build_context(self, task: TwitterTask) -> Optional[Dict[str, Any]]:
        """
        Build context for content generation.

        Args:
            task: TwitterTask to generate context for

        Returns:
            Context dict or None
        """
        try:
            if task.type == task.type.REPLY_TWEET and task.target_url:
                # Extract tweet context
                return self.browser_agent.get_tweet_context(task.target_url)

            elif task.type == task.type.REPLY_DM and task.handle:
                # Extract user context
                return self.browser_agent.get_user_context(task.handle)

            else:
                # No context needed for POST
                return None

        except Exception as e:
            logger.warning(f"Failed to build context: {e}")
            return None

    def _post_to_twitter(self, task: TwitterTask, draft: TwitterDraft):
        """
        Post draft to Twitter based on task type.

        Args:
            task: TwitterTask with type and metadata
            draft: TwitterDraft to post

        Returns:
            BrowserActionResult
        """
        if task.type == task.type.POST:
            return self.browser_agent.post_tweet(draft)

        elif task.type == task.type.REPLY_TWEET and task.target_url:
            return self.browser_agent.reply_to_tweet(draft, task.target_url)

        elif task.type == task.type.REPLY_DM and task.handle:
            return self.browser_agent.send_dm(draft, task.handle)

        else:
            raise Exception(f"Invalid task type or missing metadata: {task.type}")

    def _complete_task(self, task: TwitterTask, draft: TwitterDraft, tweet_url: str):
        """
        Mark task as completed in YAML.

        Args:
            task: Completed task
            draft: Draft that was posted
            tweet_url: URL of posted tweet
        """
        task.status = TaskStatus.COMPLETED
        task.selected_draft_index = 0
        task.completed_at = datetime.now()
        task.tweet_url = tweet_url

        self.task_parser.update_task(task)
        logger.info(f"Task {task.id} marked as completed")

        # Record to memory if available
        if self.memory:
            self.memory.record_post(task, draft, tweet_url)

        # Display summary
        self.ui_agent.display_task_summary(task, {
            'status': 'completed',
            'tweet_url': tweet_url
        })

    def _skip_task(self, task: TwitterTask, reason: str) -> Dict[str, Any]:
        """
        Mark task as skipped.

        Args:
            task: Task to skip
            reason: Reason for skipping

        Returns:
            Result dict
        """
        task.status = TaskStatus.SKIPPED
        task.error_message = reason
        self.task_parser.update_task(task)

        logger.info(f"Task {task.id} skipped: {reason}")

        return {
            'status': 'skipped',
            'reason': reason
        }

    def _fail_task(self, task: TwitterTask, error: str) -> Dict[str, Any]:
        """
        Mark task as failed.

        Args:
            task: Failed task
            error: Error message

        Returns:
            Result dict
        """
        task.status = TaskStatus.FAILED
        task.error_message = error
        self.task_parser.update_task(task)

        # Display summary
        self.ui_agent.display_task_summary(task, {
            'status': 'failed',
            'error': error
        })

        return {
            'status': 'failed',
            'error': error
        }

    def _record_feedback(self, task: TwitterTask, draft: TwitterDraft, action: str, user_edits: Optional[str] = None):
        """
        Record user feedback for learning.

        Args:
            task: Task that was processed
            draft: Draft that was acted on
            action: User action ('confirmed', 'regenerated', 'edited', 'skipped')
            user_edits: What user changed (if edited)
        """
        self.content_agent.record_feedback(draft, action, user_edits)

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze performance across all posted content.

        Returns:
            Performance metrics and insights
        """
        if not self.memory:
            logger.warning("Memory system not available for performance analysis")
            return {}

        logger.info("Analyzing performance...")

        # Load all tasks
        all_tasks = self.task_parser.load_tasks(include_completed=True)

        # Analyze patterns
        insights = self.memory.analyze_patterns(all_tasks)

        return insights

    def generate_report(self) -> str:
        """
        Generate performance report.

        Returns:
            Report file path
        """
        insights = self.analyze_performance()

        # Create report directory
        reports_dir = Path("logs/twitter/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"performance_report_{timestamp}.txt"

        # Write report
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Twitter Growth Agent - Performance Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            if insights:
                f.write("📊 Performance Insights\n")
                f.write("-" * 80 + "\n")

                for key, value in insights.items():
                    f.write(f"\n{key}:\n")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    elif isinstance(value, list):
                        for item in value:
                            f.write(f"  - {item}\n")
                    else:
                        f.write(f"  {value}\n")
            else:
                f.write("No performance data available yet.\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Performance report saved: {report_path}")

        return str(report_path)

    def interactive_mode(self):
        """
        Run in interactive mode for continuous task processing.

        Monitors tasks.yaml for new tasks and processes them as they appear.
        """
        logger.info("Starting interactive mode...")
        print("\n" + "=" * 80)
        print("🤖 Twitter Growth Agent - Interactive Mode")
        print("=" * 80)
        print("Monitoring tasks.yaml for new tasks...")
        print("Press Ctrl+C to exit")
        print("=" * 80 + "\n")

        try:
            import time

            while True:
                # Load pending tasks
                pending_tasks = self.task_parser.load_tasks()

                if pending_tasks:
                    logger.info(f"Found {len(pending_tasks)} pending task(s)")

                    # Process all pending tasks
                    self.run_batch()

                    # Brief pause before checking again
                    print("\n⏸️  Waiting for new tasks...")
                else:
                    # No tasks, wait before checking again
                    time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n👋 Exiting interactive mode...")
            logger.info("Interactive mode terminated by user")
