#!/usr/bin/env python3
"""
Task Consumer Example - How to use generated tasks with your Agent system

This script demonstrates how to consume the tasks.yaml file
and execute outreach actions.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class TaskConsumer:
    """Consume and execute tasks from tasks.yaml."""

    def __init__(self, tasks_path: str = "./data/tasks.yaml"):
        self.tasks_path = tasks_path
        self.tasks = []

    def load_tasks(self) -> List[Dict]:
        """Load tasks from YAML file."""
        try:
            with open(self.tasks_path, 'r') as f:
                data = yaml.safe_load(f)

            self.tasks = data.get('tasks', [])
            print(f"Loaded {len(self.tasks)} tasks from {self.tasks_path}")
            return self.tasks

        except FileNotFoundError:
            print(f"[ERROR] Tasks file not found: {self.tasks_path}")
            return []
        except yaml.YAMLError as e:
            print(f"[ERROR] Failed to parse YAML: {e}")
            return []

    def filter_tasks(
        self,
        task_type: str = None,
        priority: str = None,
        min_score: float = None
    ) -> List[Dict]:
        """
        Filter tasks by criteria.

        Args:
            task_type: Filter by task type (TWITTER_REPLY, EMAIL_OUTREACH, etc.)
            priority: Filter by priority (high, medium, low)
            min_score: Minimum score threshold

        Returns:
            Filtered list of tasks
        """
        filtered = self.tasks

        if task_type:
            filtered = [t for t in filtered if t.get('type') == task_type]

        if priority:
            filtered = [t for t in filtered if t.get('priority') == priority]

        if min_score:
            filtered = [t for t in filtered if t.get('score', 0) >= min_score]

        return filtered

    def execute_task(self, task: Dict, dry_run: bool = True):
        """
        Execute a single task.

        Args:
            task: Task dict
            dry_run: If True, print only (don't actually send)
        """
        task_type = task.get('type')
        target = task.get('target', {})
        contact = task.get('contact', {})
        context = task.get('context', '')
        talking_points = task.get('talking_points', [])

        print(f"\n{'='*60}")
        print(f"Executing Task: {task.get('id')}")
        print(f"Type: {task_type}")
        print(f"Priority: {task.get('priority')} (Score: {task.get('score')})")
        print(f"{'='*60}\n")

        print(f"Target: {target.get('brand_name')}")
        print(f"Domain: {target.get('domain')}")
        print(f"Custom Domain: {target.get('custom_domain')}")
        print()

        if contact.get('name'):
            print(f"Contact: {contact.get('name')}")
        if contact.get('email'):
            print(f"Email: {contact.get('email')}")
        if contact.get('twitter'):
            print(f"Twitter: @{contact.get('twitter')}")
        if contact.get('linkedin'):
            print(f"LinkedIn: {contact.get('linkedin')}")

        print(f"\nContext: {context}")
        print(f"\nTalking Points:")
        for point in talking_points:
            print(f"  • {point}")

        if dry_run:
            print(f"\n[DRY RUN] Would send {task_type} to {target.get('brand_name')}")
        else:
            print(f"\n[TODO] Implement actual {task_type} execution here")
            # This is where you would integrate with your Agent system
            # For example:
            # - Send email via SendGrid/Mailgun
            # - Post tweet via Twitter API
            # - Send LinkedIn message via LinkedIn API

    def execute_all(
        self,
        task_type: str = None,
        priority: str = None,
        min_score: float = None,
        dry_run: bool = True
    ):
        """
        Execute filtered tasks.

        Args:
            task_type: Filter by task type
            priority: Filter by priority
            min_score: Minimum score threshold
            dry_run: If True, print only
        """
        self.load_tasks()

        tasks_to_execute = self.filter_tasks(task_type, priority, min_score)

        print(f"\nExecuting {len(tasks_to_execute)} tasks...")

        for task in tasks_to_execute:
            self.execute_task(task, dry_run=dry_run)

            if not dry_run:
                # Mark task as completed
                self._mark_task_completed(task.get('id'))

    def _mark_task_completed(self, task_id: str):
        """
        Mark a task as completed (you'd persist this to a database).

        Args:
            task_id: Task ID to mark complete
        """
        # In production, you'd update a database or tracking system
        print(f"\n[INFO] Marked task {task_id} as completed")

    def print_summary(self):
        """Print a summary of loaded tasks."""
        if not self.tasks:
            self.load_tasks()

        print(f"\n{'='*60}")
        print(f"Task Summary")
        print(f"{'='*60}\n")

        print(f"Total Tasks: {len(self.tasks)}")

        # Count by type
        by_type = {}
        for task in self.tasks:
            task_type = task.get('type', 'unknown')
            by_type[task_type] = by_type.get(task_type, 0) + 1

        print(f"\nBy Type:")
        for task_type, count in by_type.items():
            print(f"  {task_type}: {count}")

        # Count by priority
        by_priority = {}
        for task in self.tasks:
            priority = task.get('priority', 'unknown')
            by_priority[priority] = by_priority.get(priority, 0) + 1

        print(f"\nBy Priority:")
        for priority, count in by_priority.items():
            print(f"  {priority}: {count}")

        # Score distribution
        scores = [t.get('score', 0) for t in self.tasks]
        if scores:
            print(f"\nScore Distribution:")
            print(f"  Min: {min(scores):.1f}")
            print(f"  Max: {max(scores):.1f}")
            print(f"  Avg: {sum(scores)/len(scores):.1f}")

        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Consume and execute tasks from the DTC pipeline"
    )
    parser.add_argument(
        '--tasks',
        type=str,
        default='./data/tasks.yaml',
        help='Path to tasks.yaml file (default: ./data/tasks.yaml)'
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['TWITTER_REPLY', 'EMAIL_OUTREACH', 'LINKEDIN_MESSAGE'],
        help='Filter by task type'
    )
    parser.add_argument(
        '--priority',
        type=str,
        choices=['high', 'medium', 'low'],
        help='Filter by priority'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        help='Minimum score threshold'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute tasks (default: dry run)'
    )

    args = parser.parse_args()

    consumer = TaskConsumer(tasks_path=args.tasks)

    # Load and display summary
    consumer.load_tasks()
    consumer.print_summary()

    # Execute tasks
    consumer.execute_all(
        task_type=args.type,
        priority=args.priority,
        min_score=args.min_score,
        dry_run=not args.execute
    )


if __name__ == "__main__":
    main()
