"""
YAML task parser for Twitter Growth Agent.

Loads, parses, and manages tasks from tasks.yaml file.
"""

import yaml
from pathlib import Path
from typing import List, Callable, Optional, Dict, Any
from datetime import datetime
import logging

from .types import TwitterTask, TaskStatus, TaskType, TwitterConfig

logger = logging.getLogger(__name__)


class YAMLTaskParser:
    """
    Parse and manage Twitter tasks from YAML file.

    Responsibilities:
    - Load and parse tasks.yaml
    - Validate task structure
    - Track task status
    - Persist status updates
    - Monitor file for changes
    """

    def __init__(self, yaml_path: str, config: TwitterConfig):
        """
        Initialize YAML task parser.

        Args:
            yaml_path: Path to tasks.yaml file
            config: TwitterConfig object
        """
        self.yaml_path = Path(yaml_path)
        self.config = config
        self._tasks: List[TwitterTask] = []

    def load_tasks(self) -> List[TwitterTask]:
        """
        Load all tasks from YAML, filter by PENDING status.

        Returns:
            List of TwitterTask objects
        """
        if not self.yaml_path.exists():
            logger.error(f"Tasks file not found: {self.yaml_path}")
            return []

        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        self._tasks = []
        tasks_data = data.get('tasks', [])

        for idx, task_dict in enumerate(tasks_data):
            try:
                task = self.parse_task_dict(task_dict, idx)
                # Only return PENDING tasks for processing
                if task.status == TaskStatus.PENDING:
                    self._tasks.append(task)
            except Exception as e:
                logger.error(f"Error parsing task at index {idx}: {e}")

        logger.info(f"Loaded {len(self._tasks)} pending tasks from {self.yaml_path}")
        return self._tasks

    def parse_task_dict(self, task_dict: Dict[str, Any], idx: int) -> TwitterTask:
        """
        Convert YAML dict to TwitterTask dataclass.

        Args:
            task_dict: Task dictionary from YAML
            idx: Task index for ID generation

        Returns:
            TwitterTask object
        """
        # Parse task type
        type_str = task_dict.get('type', 'POST')
        try:
            task_type = TaskType[type_str] if isinstance(type_str, str) else type_str
        except KeyError:
            logger.warning(f"Unknown task type: {type_str}, defaulting to POST")
            task_type = TaskType.POST

        # Parse status
        status_str = task_dict.get('status', 'PENDING')
        try:
            task_status = TaskStatus[status_str] if isinstance(status_str, str) else status_str
        except KeyError:
            logger.warning(f"Unknown status: {status_str}, defaulting to PENDING")
            task_status = TaskStatus.PENDING

        return TwitterTask(
            id=task_dict.get('id', f"task_{idx}"),
            type=task_type,
            idea=task_dict['idea'],
            style=task_dict.get('style', 'professional'),
            target_url=task_dict.get('target_url'),
            handle=task_dict.get('handle'),
            status=task_status,
        )

    def update_task_status(self, task_id: str, status: TaskStatus):
        """
        Update status in memory and persist to YAML.

        Args:
            task_id: Task identifier
            status: New task status
        """
        # Update in-memory task
        for task in self._tasks:
            if task.id == task_id:
                task.status = status
                break

        # Persist to YAML
        self._persist_status(task_id, status)

    def mark_completed(self, task_id: str):
        """
        Mark task as COMPLETED in YAML file.

        Args:
            task_id: Task identifier
        """
        self.update_task_status(task_id, TaskStatus.COMPLETED)
        logger.info(f"Marked task {task_id} as COMPLETED")

    def _persist_status(self, task_id: str, status: TaskStatus):
        """
        Persist status update to YAML file.

        Args:
            task_id: Task identifier
            status: New status to persist
        """
        try:
            # Load current YAML
            with open(self.yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # Update status
            for task_dict in data.get('tasks', []):
                if task_dict.get('id') == task_id:
                    task_dict['status'] = status.value
                    break

            # Write back to YAML
            with open(self.yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            logger.debug(f"Persisted status {status.value} for task {task_id}")

        except Exception as e:
            logger.error(f"Failed to persist status for task {task_id}: {e}")

    def get_task(self, task_id: str) -> Optional[TwitterTask]:
        """
        Get a specific task by ID.

        Args:
            task_id: Task identifier

        Returns:
            TwitterTask if found, None otherwise
        """
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    def watch_for_changes(self, callback: Callable):
        """
        Monitor file for changes and trigger callback.

        Args:
            callback: Function to call when file changes
        """
        import time
        last_modified = self.yaml_path.stat().st_mtime if self.yaml_path.exists() else 0

        try:
            while True:
                if self.yaml_path.exists():
                    current_modified = self.yaml_path.stat().st_mtime
                    if current_modified != last_modified:
                        logger.info(f"Tasks file changed: {self.yaml_path}")
                        callback()
                        last_modified = current_modified
                time.sleep(self.config.poll_interval)
        except KeyboardInterrupt:
            logger.info("File watching stopped")

    def create_sample_tasks(self):
        """
        Create a sample tasks.yaml file if it doesn't exist.

        This is useful for first-time setup.
        """
        if self.yaml_path.exists():
            logger.info(f"Tasks file already exists: {self.yaml_path}")
            return

        sample_data = {
            'tasks': [
                {
                    'id': 'task_0',
                    'type': 'POST',
                    'idea': 'Share today\'s Judge Model discovery: Found a $500 waste case',
                    'style': '犀利吐槽，硬核数据',
                    'status': 'PENDING'
                },
                {
                    'id': 'task_1',
                    'type': 'REPLY_TWEET',
                    'target_url': 'https://x.com/elonmusk/status/123456',
                    'idea': 'Mention our automated optimization approach',
                    'style': 'Professional but direct',
                    'status': 'PENDING'
                }
            ],
            'global_settings': {
                'human_confirmation': True,
                'anti_bot_delay': [2, 8]
            }
        }

        # Create parent directory if needed
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.yaml_path, 'w') as f:
            yaml.dump(sample_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created sample tasks file: {self.yaml_path}")
