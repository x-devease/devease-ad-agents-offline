"""
Content Agent for Twitter Growth Agent.

Generates tweet/DM drafts using OpenAI GPT-4.
"""

import json
import logging
from typing import List, Optional
from pathlib import Path

from ..core.types import TwitterTask, TwitterDraft, TwitterConfig, TwitterKeys
from ..core.memory import MemorySystem

logger = logging.getLogger(__name__)


class ContentAgent:
    """
    Generate Twitter content using OpenAI GPT-4.

    Responsibilities:
    - Use OpenAI GPT-4 to generate 3 draft versions per task
    - Apply "non-robotic" writing style
    - Include context awareness for replies
    - Validate character limits
    - Support few-shot learning from golden examples
    """

    def __init__(self, keys: TwitterKeys, config: TwitterConfig, memory: Optional[MemorySystem] = None):
        """
        Initialize content agent.

        Args:
            keys: TwitterKeys object with OpenAI credentials
            config: TwitterConfig object
            memory: Optional MemorySystem for learning
        """
        self.keys = keys
        self.config = config
        self.memory = memory
        self.client = None
        self._init_openai()

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.keys.openai_api_key,
                organization=self.keys.openai_org_id
            )
            logger.info(f"Initialized OpenAI client with model: {self.config.llm_model}")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate_drafts(self, task: TwitterTask, context: Optional[str] = None) -> List[TwitterDraft]:
        """
        Generate 3 draft versions for a task.

        Args:
            task: TwitterTask to generate content for
            context: Optional context for replies

        Returns:
            List of 3 TwitterDraft objects
        """
        try:
            # Load golden examples for few-shot learning
            golden_examples = self._load_golden_examples()

            # Build prompt
            prompt = self._build_prompt(task, context, golden_examples)

            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=1000
            )

            # Parse response
            content = response.choices[0].message.content
            drafts = self._parse_llm_response(content)

            logger.info(f"Generated {len(drafts)} drafts for task {task.id}")
            return drafts

        except Exception as e:
            logger.error(f"Failed to generate drafts for task {task.id}: {e}")
            # Return fallback drafts
            return self._generate_fallback_drafts(task)

    def _get_system_prompt(self) -> str:
        """
        Get core persona system prompt.

        Returns:
            System prompt string
        """
        return """You are **Xin**, DevEase's founder.

**Personality:** Minimalist, technical fundamentalist, deeply disgusted by mediocrity and waste in the ad industry.

**Speaking Style:**
- **FORBIDDEN VOCABULARY:** Absolutely NEVER use: "Transforming", "Unlocking", "Comprehensive", "Empowering", "Leveraging", "Revolutionary"
- **Sentence Structure:** Prefer short, punchy sentences. Don't avoid colloquialisms and slang. If something is cool, just say "Wild" or "Huge"; if something sucks, call it out directly.
- **Strategy:** Always lead with "data" or "logic", never vague sentiments.

**Writing Principles:**
- Be specific, not abstract
- Use numbers and concrete examples
- Avoid corporate buzzwords at all costs
- Sound like a real human having a conversation, not a marketing department"""

    def _build_prompt(self, task: TwitterTask, context: Optional[str], golden_examples: dict) -> str:
        """
        Build prompt with system prompt, task, style, and context.

        Args:
            task: TwitterTask object
            context: Optional context string
            golden_examples: Few-shot learning examples

        Returns:
            Complete prompt string
        """
        # Load appropriate prompt template
        template_path = self._get_template_path(task.type)
        template = self._load_template(template_path)

        # Build context data
        context_data = {
            "idea": task.idea,
            "style": task.style,
            "internal_data": context or "(No internal data available)",
        }

        # Add reply-specific context
        if task.type == task.type.REPLY_TWEET and context:
            context_data.update({
                "tweet_text": context.get("tweet_text", ""),
                "author_handle": context.get("author_handle", ""),
                "likes": context.get("likes", 0),
                "retweets": context.get("retweets", 0),
                "top_replies": context.get("top_replies", []),
            })
        elif task.type == task.type.REPLY_DM:
            context_data.update({
                "handle": task.handle,
                "bio": context.get("bio", "") if context else "",
                "last_5_tweets": context.get("last_5_tweets", []) if context else [],
            })

        # Format template
        prompt = template.format(**context_data)

        # Add golden examples if available
        if golden_examples and golden_examples.get("examples"):
            prompt = f"{golden_examples['system_prompt']}\n\n{prompt}"

        # Add memory insights if available
        if self.memory and len(self.memory.feedback_history) >= 10:
            insights = self.memory.get_feedback_for_generation()
            if insights:
                prompt = f"{prompt}\n\n**Content Generation Insights:**\n{insights}"

        return prompt

    def _get_template_path(self, task_type) -> Path:
        """Get path to appropriate prompt template."""
        base_path = Path("config/twitter/prompts")

        if task_type == task_type.POST:
            return base_path / "post_template.txt"
        elif task_type == task_type.REPLY_TWEET:
            return base_path / "reply_template.txt"
        elif task_type == task_type.REPLY_DM:
            return base_path / "dm_template.txt"
        else:
            return base_path / "post_template.txt"

    def _load_template(self, template_path: Path) -> str:
        """Load prompt template from file."""
        try:
            with open(template_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Template not found: {template_path}, using default")
            return self._get_default_template()

    def _get_default_template(self) -> str:
        """Get default prompt template."""
        return """Generate Twitter content based on:

**Idea:** {idea}
**Style:** {style}

Requirements:
- Be authentic and conversational
- Use specific examples and numbers
- No corporate buzzwords
- Keep it concise and engaging

Return JSON format with content and rationale."""

    def _load_golden_examples(self) -> dict:
        """Load golden examples for few-shot learning."""
        try:
            examples_path = Path("config/twitter/golden_examples.json")
            if examples_path.exists():
                with open(examples_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load golden examples: {e}")
        return {}

    def _parse_llm_response(self, response: str) -> List[TwitterDraft]:
        """
        Parse LLM response into TwitterDraft objects.

        Args:
            response: LLM response string

        Returns:
            List of TwitterDraft objects
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)

            if "drafts" in data:
                # Multiple drafts format
                return [
                    TwitterDraft(
                        content=d["content"],
                        rationale=d.get("rationale", ""),
                        tone=d.get("tone", "professional"),
                        hashtags=d.get("hashtags", []),
                        version=d.get("version", "default"),
                    )
                    for d in data["drafts"]
                ]
            else:
                # Single draft format
                return [
                    TwitterDraft(
                        content=data.get("content", response),
                        rationale=data.get("rationale", ""),
                        tone=data.get("tone", "professional"),
                        hashtags=data.get("hashtags", []),
                        version="default",
                    )
                ]

        except json.JSONDecodeError:
            # Fallback: treat entire response as content
            logger.warning("Failed to parse LLM response as JSON, using raw content")
            return [
                TwitterDraft(
                    content=response.strip(),
                    rationale="Generated by LLM",
                    tone="unknown",
                )
            ]

    def regenerate_draft(self, task: TwitterTask, rejected_index: int, context: Optional[str] = None) -> TwitterDraft:
        """
        Regenerate a single draft with feedback.

        Args:
            task: TwitterTask object
            rejected_index: Index of rejected draft
            context: Optional context string

        Returns:
            New TwitterDraft object
        """
        try:
            # Build regeneration prompt
            prompt = f"""The previous draft wasn't quite right. Please regenerate a new version.

**Original Idea:** {task.idea}
**Style:** {task.style}
**Context:** {context or "N/A"}

Make it different from the previous attempt while maintaining the core idea."""

            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.9,  # Higher temperature for variety
                max_tokens=500
            )

            content = response.choices[0].message.content
            return TwitterDraft(
                content=content.strip(),
                rationale="Regenerated draft",
                tone="professional",
                version=f"regenerated_{rejected_index}",
            )

        except Exception as e:
            logger.error(f"Failed to regenerate draft: {e}")
            return self._generate_fallback_draft(task)

    def _generate_fallback_drafts(self, task: TwitterTask) -> List[TwitterDraft]:
        """Generate fallback drafts when LLM fails."""
        return [
            TwitterDraft(
                content=f"[Draft 1] {task.idea}",
                rationale="Fallback draft - LLM unavailable",
                tone="neutral",
                version="fallback_1",
            ),
            TwitterDraft(
                content=f"[Draft 2] {task.idea}",
                rationale="Fallback draft - LLM unavailable",
                tone="neutral",
                version="fallback_2",
            ),
            TwitterDraft(
                content=f"[Draft 3] {task.idea}",
                rationale="Fallback draft - LLM unavailable",
                tone="neutral",
                version="fallback_3",
            ),
        ]

    def _generate_fallback_draft(self, task: TwitterTask) -> TwitterDraft:
        """Generate a single fallback draft."""
        return TwitterDraft(
            content=f"[Fallback] {task.idea}",
            rationale="Fallback draft - LLM unavailable",
            tone="neutral",
            version="fallback",
        )

    def record_feedback(self, draft: TwitterDraft, action: str, user_edits: Optional[str] = None):
        """
        Record user feedback on draft for learning.

        Args:
            draft: The draft that was acted on
            action: What user did ('confirmed', 'regenerated', 'edited', 'skipped')
            user_edits: What the user changed (if edited)
        """
        if self.memory:
            self.memory.record_feedback(draft, action, user_edits)
            logger.debug(f"Recorded feedback: {action} for draft version {draft.version}")
