"""
Core data models for Twitter Growth Agent.

Defines all dataclasses used throughout the agent system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class TaskType(Enum):
    """Types of Twitter tasks."""
    POST = "POST"
    REPLY_TWEET = "REPLY_TWEET"
    REPLY_DM = "REPLY_DM"


class TaskStatus(Enum):
    """Status of a Twitter task in the workflow."""
    PENDING = "PENDING"
    DRAFTING = "DRAFTING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    CONFIRMED = "CONFIRMED"
    POSTED = "POSTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class TwitterTask:
    """
    Represents a single Twitter task from tasks.yaml.

    Attributes:
        id: Unique task identifier
        type: Type of task (POST, REPLY_TWEET, REPLY_DM)
        idea: Content idea/description
        style: Desired writing style
        target_url: URL for REPLY_TWEET tasks
        handle: User handle for REPLY_DM tasks
        status: Current task status
        created_at: Task creation timestamp
        selected_draft_index: Which draft was selected
        error_message: Error details if failed
    """
    id: str
    type: TaskType
    idea: str
    style: str
    target_url: Optional[str] = None
    handle: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    selected_draft_index: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for YAML serialization."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, TaskType) else self.type,
            "idea": self.idea,
            "style": self.style,
            "target_url": self.target_url,
            "handle": self.handle,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "created_at": self.created_at.isoformat(),
            "selected_draft_index": self.selected_draft_index,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], idx: int) -> "TwitterTask":
        """Create TwitterTask from YAML dictionary."""
        return cls(
            id=data.get("id", f"task_{idx}"),
            type=TaskType[data["type"]] if isinstance(data.get("type"), str) else data["type"],
            idea=data["idea"],
            style=data.get("style", "professional"),
            target_url=data.get("target_url"),
            handle=data.get("handle"),
            status=TaskStatus[data.get("status", "PENDING")] if isinstance(data.get("status"), str) else data.get("status", TaskStatus.PENDING),
        )


@dataclass
class TwitterDraft:
    """
    Represents a generated content draft.

    Attributes:
        content: The actual tweet/DM content
        rationale: Explanation of why this approach works
        tone: Detected tone (professional, casual, witty, etc.)
        hashtags: List of hashtags included
        character_count: Content length
        version: Draft version identifier (e.g., "spicy", "balanced")
    """
    content: str
    rationale: str
    tone: str = "professional"
    hashtags: List[str] = field(default_factory=list)
    character_count: int = 0
    version: str = "default"

    def __post_init__(self):
        """Calculate character count after initialization."""
        self.character_count = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert draft to dictionary."""
        return {
            "content": self.content,
            "rationale": self.rationale,
            "tone": self.tone,
            "hashtags": self.hashtags,
            "character_count": self.character_count,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TwitterDraft":
        """Create TwitterDraft from dictionary."""
        return cls(
            content=data["content"],
            rationale=data.get("rationale", ""),
            tone=data.get("tone", "professional"),
            hashtags=data.get("hashtags", []),
            version=data.get("version", "default"),
        )


@dataclass
class TwitterConfig:
    """
    Configuration settings for the Twitter agent.

    Attributes:
        human_confirmation: Require user approval before posting
        anti_bot_delay: Min and max seconds for delays between actions
        num_drafts: Number of drafts to generate per task
        llm_model: OpenAI model to use
        max_retries: Maximum retry attempts for failed operations
        enable_analytics: Track performance metrics
    """
    human_confirmation: bool = True
    anti_bot_delay: tuple = (2, 8)
    num_drafts: int = 3
    llm_model: str = "gpt-4o"
    max_retries: int = 3
    enable_analytics: bool = True

    # Browser settings
    browser_headless: bool = False
    browser_slow_mo: int = 100
    browser_timeout: int = 30000
    user_data_dir: Optional[str] = None

    # UI settings
    show_rationale: bool = True
    enable_editing: bool = True

    # Anti-detection
    typing_delay_min: int = 50
    typing_delay_max: int = 150
    random_pauses: bool = True

    # Monitoring
    poll_interval: int = 10
    watch_file_changes: bool = True
    max_concurrent_tasks: int = 1


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a posted tweet.

    Attributes:
        task_id: Associated task ID
        posted_at: When the content was posted
        content: The posted content
        tweet_url: URL to the posted tweet
        likes: Number of likes
        retweets: Number of retweets
        replies: Number of replies
        impressions: Number of impressions
        engagement_rate: Calculated engagement rate
    """
    task_id: str
    posted_at: datetime
    content: str
    tweet_url: str
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0

    def calculate_engagement_rate(self) -> float:
        """Calculate engagement rate from existing metrics."""
        if self.impressions == 0:
            return 0.0
        total_engagements = self.likes + self.retweets + self.replies
        self.engagement_rate = (total_engagements / self.impressions) * 100
        return self.engagement_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "task_id": self.task_id,
            "posted_at": self.posted_at.isoformat(),
            "content": self.content,
            "tweet_url": self.tweet_url,
            "likes": self.likes,
            "retweets": self.retweets,
            "replies": self.replies,
            "impressions": self.impressions,
            "engagement_rate": self.engagement_rate,
        }


@dataclass
class TwitterKeys:
    """
    API keys and credentials for Twitter and OpenAI.

    Attributes:
        openai_api_key: OpenAI API key (required)
        openai_org_id: OpenAI organization ID
        twitter_api_key: Twitter API key (optional)
        twitter_api_secret: Twitter API secret (optional)
        twitter_access_token: Twitter access token (optional)
        twitter_access_secret: Twitter access secret (optional)
        twitter_bearer_token: Twitter bearer token (optional)
        twitter_cookies_path: Path to browser cookies
        devease_data_path: Path to DevEase data
    """
    openai_api_key: str
    openai_org_id: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    twitter_cookies_path: Optional[str] = None
    devease_data_path: Optional[str] = None


@dataclass
class ContextData:
    """
    Context information for reply generation.

    Attributes:
        tweet_text: The target tweet content
        author_handle: Tweet author's handle
        author_bio: Author's bio
        likes: Tweet like count
        retweets: Tweet retweet count
        top_replies: Top replies to the tweet
        conversation_thread: Previous tweets in thread
    """
    tweet_text: str = ""
    author_handle: str = ""
    author_bio: str = ""
    likes: int = 0
    retweets: int = 0
    top_replies: List[str] = field(default_factory=list)
    conversation_thread: List[str] = field(default_factory=list)
