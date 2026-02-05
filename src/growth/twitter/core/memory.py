"""
Memory system for Twitter Growth Agent.

Stores performance metrics and analytics data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .types import PerformanceMetrics, TwitterDraft, TwitterTask

logger = logging.getLogger(__name__)


class MemorySystem:
    """
    Store and retrieve performance data.

    Responsibilities:
    - Store posted content with metrics
    - Track performance over time
    - Analyze patterns
    - Provide feedback for content generation
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize memory system.

        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path or "~/devease/data/twitter").expanduser()
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.data_path / "metrics.json"
        self.feedback_file = self.data_path / "feedback_history.json"

        # Load existing data
        self.metrics: List[PerformanceMetrics] = self._load_metrics()
        self.feedback_history: List[Dict[str, Any]] = self._load_feedback()

    def _load_metrics(self) -> List[PerformanceMetrics]:
        """Load metrics from JSON file."""
        if not self.metrics_file.exists():
            return []

        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)

            metrics = []
            for item in data:
                metrics.append(PerformanceMetrics(
                    task_id=item['task_id'],
                    posted_at=datetime.fromisoformat(item['posted_at']),
                    content=item['content'],
                    tweet_url=item['tweet_url'],
                    likes=item.get('likes', 0),
                    retweets=item.get('retweets', 0),
                    replies=item.get('replies', 0),
                    impressions=item.get('impressions', 0),
                    engagement_rate=item.get('engagement_rate', 0.0),
                ))
            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return []

    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Load feedback history from JSON file."""
        if not self.feedback_file.exists():
            return []

        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load feedback history: {e}")
            return []

    def record_post(
        self,
        task: TwitterTask,
        draft: TwitterDraft,
        tweet_url: str
    ):
        """
        Record posted content with metadata.

        Args:
            task: The task that was posted
            draft: The draft that was posted
            tweet_url: URL of the posted tweet
        """
        metric = PerformanceMetrics(
            task_id=task.id,
            posted_at=datetime.now(),
            content=draft.content,
            tweet_url=tweet_url,
        )

        self.metrics.append(metric)
        self._persist_metrics()
        logger.info(f"Recorded post for task {task.id}: {tweet_url}")

    def _persist_metrics(self):
        """Persist metrics to JSON file."""
        try:
            data = [m.to_dict() for m in self.metrics]
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")

    def fetch_metrics(self, tweet_url: str) -> Optional[PerformanceMetrics]:
        """
        Fetch engagement metrics for a tweet.

        Args:
            tweet_url: URL of the tweet

        Returns:
            PerformanceMetrics object, or None if not found
        """
        for metric in self.metrics:
            if metric.tweet_url == tweet_url:
                return metric
        return None

    def analyze_patterns(self, tasks: List[TwitterTask]) -> Dict[str, Any]:
        """
        Analyze what content styles perform best.

        Args:
            tasks: List of tasks to analyze

        Returns:
            Dictionary with analysis results
        """
        if not self.metrics:
            return {"message": "No metrics available for analysis"}

        # Group by style
        style_performance: Dict[str, List[float]] = {}
        for metric in self.metrics:
            # Find associated task to get style
            task = next((t for t in tasks if t.id == metric.task_id), None)
            if task:
                if task.style not in style_performance:
                    style_performance[task.style] = []
                style_performance[task.style].append(metric.engagement_rate)

        # Calculate averages
        averages = {
            style: sum(rates) / len(rates)
            for style, rates in style_performance.items()
        }

        # Find best performing style
        if averages:
            best_style = max(averages, key=averages.get)
            return {
                "best_style": best_style,
                "style_averages": averages,
                "total_posts": len(self.metrics),
            }

        return {"message": "Insufficient data for analysis"}

    def record_feedback(self, draft: TwitterDraft, action: str, user_edits: Optional[str] = None):
        """
        Record user feedback on drafts.

        Args:
            draft: The draft that was acted on
            action: What user did ('confirmed', 'regenerated', 'edited', 'skipped')
            user_edits: What the user changed (if edited)
        """
        feedback_entry = {
            "draft_features": {
                "tone": draft.tone,
                "character_count": draft.character_count,
                "version": draft.version,
                "has_hashtags": len(draft.hashtags) > 0,
            },
            "action": action,
            "user_edits": user_edits,
            "timestamp": datetime.now().isoformat(),
        }

        self.feedback_history.append(feedback_entry)
        self._persist_feedback()

    def _persist_feedback(self):
        """Persist feedback history to JSON file."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist feedback: {e}")

    def get_feedback_summary(self) -> str:
        """
        Generate insights from feedback history.

        Returns:
            Summary string of feedback patterns
        """
        if not self.feedback_history:
            return "No feedback data available yet."

        # Analyze patterns
        confirmed = sum(1 for f in self.feedback_history if f['action'] == 'confirmed')
        regenerated = sum(1 for f in self.feedback_history if f['action'] == 'regenerated')
        edited = sum(1 for f in self.feedback_history if f['action'] == 'edited')
        total = len(self.feedback_history)

        summary = f"""
Feedback Summary ({total} total actions):
- Confirmed: {confirmed} ({confirmed/total*100:.1f}%)
- Regenerated: {regenerated} ({regenerated/total*100:.1f}%)
- Edited: {edited} ({edited/total*100:.1f}%)

Top patterns will be analyzed after {min(50, total)} more data points.
        """.strip()

        return summary

    def get_feedback_for_generation(self) -> str:
        """
        Generate insights to inform future content generation.

        Returns:
            Insights string for LLM prompt
        """
        if len(self.feedback_history) < 10:
            return ""

        # Analyze what gets approved
        confirmed = [f for f in self.feedback_history if f['action'] == 'confirmed']
        if not confirmed:
            return ""

        # Find patterns in confirmed drafts
        avg_char_count = sum(f['draft_features']['character_count'] for f in confirmed) / len(confirmed)

        insights = f"""
Content Generation Insights:
- Average character count of approved drafts: {avg_char_count:.0f}
- Based on {len(confirmed)} confirmed drafts

Recommendation: Aim for content around {avg_char_count:.0f} characters.
        """.strip()

        return insights
