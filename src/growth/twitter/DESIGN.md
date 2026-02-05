# Twitter Growth Agent Implementation Plan

## Overview

A YAML-driven Twitter Growth Agent that automates content creation and engagement while maintaining human oversight. The agent reads task configurations from a YAML file, generates multiple content drafts using OpenAI GPT-4, uses Playwright to control Chrome, and requires CLI-based confirmation before posting.

**Architecture Pattern**: Team-based agent (similar to Diagnoser) due to multi-phase workflow requiring orchestration.

---

## File Structure

```
src/growth/twitter/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── content_agent.py              # Content generation (Drafting phase)
│   ├── browser_agent.py              # Playwright controller (Browser phase)
│   ├── ui_agent.py                   # CLI confirmation interface
│   ├── analytics_agent.py            # Performance tracking & learning
│   └── orchestrator.py               # Workflow coordinator
├── core/
│   ├── __init__.py
│   ├── types.py                      # Dataclasses (Task, Draft, Config)
│   ├── yaml_parser.py                # Parse and monitor tasks.yaml
│   ├── context_builder.py            # Build context for replies (thread analysis)
│   ├── key_manager.py                # Parse API keys from ~/.devease/keys
│   └── memory.py                     # Store performance data
├── prompts/
│   ├── tweet_generation.txt          # System prompt for tweet generation
│   ├── reply_generation.txt          # System prompt for replies
│   └── dm_generation.txt             # System prompt for DMs
└── utils/
    ├── __init__.py
    ├── typing_simulator.py           # Human-like typing simulation
    └── url_normalizer.py             # Convert x.com to twitter.com

config/agents/
└── twitter_config.yaml               # Twitter agent configuration

config/twitter/
├── tasks.yaml                        # User task definitions
├── golden_examples.json              # Few-shot learning examples
└── prompts/
    ├── post_template.txt             # POST task prompt template
    ├── reply_template.txt            # REPLY task prompt template
    └── dm_template.txt               # DM task prompt template

tests/
├── unit/growth/twitter/
│   ├── test_types.py
│   ├── test_yaml_parser.py
│   ├── test_content_agent.py
│   ├── test_context_builder.py
│   └── test_key_manager.py
└── integration/
    └── test_twitter_workflow.py
```

---

## Core Components

### 1. Data Models (`src/growth/twitter/core/types.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime

class TaskType(Enum):
    POST = "POST"
    REPLY_TWEET = "REPLY_TWEET"
    REPLY_DM = "REPLY_DM"

class TaskStatus(Enum):
    PENDING = "PENDING"
    DRAFTING = "DRAFTING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    CONFIRMED = "CONFIRMED"
    POSTED = "POSTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class TwitterTask:
    id: str
    type: TaskType
    idea: str
    style: str
    target_url: Optional[str] = None  # For REPLY_TWEET
    handle: Optional[str] = None      # For REPLY_DM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    selected_draft_index: int = 0
    error_message: Optional[str] = None

@dataclass
class TwitterDraft:
    content: str
    rationale: str  # Why this approach works
    tone: str       # Detected tone (professional, casual, witty, etc.)
    hashtags: List[str] = field(default_factory=list)
    character_count: int = 0

@dataclass
class TwitterConfig:
    human_confirmation: bool = True
    anti_bot_delay: tuple = (2, 8)  # min, max seconds
    num_drafts: int = 3
    llm_model: str = "gpt-4o"
    max_retries: int = 3
    enable_analytics: bool = True

@dataclass
class PerformanceMetrics:
    task_id: str
    posted_at: datetime
    content: str
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    impressions: int = 0
    engagement_rate: float = 0.0
```

### 2. YAML Parser (`src/growth/twitter/core/yaml_parser.py`)

**Responsibilities**:
- Load and parse `tasks.yaml`
- Validate task structure using Pydantic
- Monitor file for changes (watchdog or polling)
- Track task status and persist updates
- Mark completed tasks

**Key Functions**:
```python
class YAMLTaskParser:
    def __init__(self, yaml_path: str, config: TwitterConfig):
        self.yaml_path = yaml_path
        self.config = config

    def load_tasks(self) -> List[TwitterTask]:
        """Load all tasks from YAML, filter by PENDING status"""

    def parse_task_dict(self, task_dict: dict, idx: int) -> TwitterTask:
        """Convert YAML dict to TwitterTask dataclass"""

    def update_task_status(self, task_id: str, status: TaskStatus):
        """Update status in memory and persist to YAML"""

    def mark_completed(self, task_id: str):
        """Mark task as COMPLETED in YAML file"""

    def watch_for_changes(self, callback: Callable):
        """Monitor file for changes and trigger callback"""
```

### 3. Content Agent (`src/growth/twitter/agents/content_agent.py`)

**Responsibilities**:
- Use OpenAI GPT-4 to generate 3 draft versions per task
- Apply "non-robotic" writing style
- Include context awareness for replies (read thread history)
- Validate character limits (280 for tweets, 10,000 for DMs)

**Key Functions**:
```python
class ContentAgent:
    def __init__(self, llm_client: LLMClientWithRetry):
        self.llm_client = llm_client

    def generate_drafts(self, task: TwitterTask, context: Optional[str] = None) -> List[TwitterDraft]:
        """Generate 3 draft versions for a task"""

    def _build_prompt(self, task: TwitterTask, context: Optional[str]) -> str:
        """Build prompt with system prompt, task, style, and context"""

    def _parse_llm_response(self, response: str) -> List[TwitterDraft]:
        """Parse LLM response into TwitterDraft objects"""

    def regenerate_draft(self, task: TwitterTask, rejected_index: int, context: Optional[str] = None) -> TwitterDraft:
        """Regenerate a single draft with feedback"""
```

**Enhanced Prompt Engineering** (Persona-Based with Self-Correction)

The quality of prompts determines the "non-robotic" feel and conversion rate. This agent uses a **structured persona-based prompt system** with continuous optimization.

---

### Core Persona Injection (System Instructions)

**Loaded before all tasks** - This is the "soul" of the agent:

```
You are **Xin**, DevEase's founder.

**Personality:** Minimalist, technical fundamentalist, deeply disgusted by mediocrity and waste in the ad industry.

**Speaking Style:**
- **FORBIDDEN VOCABULARY:** Absolutely NEVER use: "Transforming", "Unlocking", "Comprehensive", "Empowering", "Leveraging", "Revolutionary"
- **Sentence Structure:** Prefer short, punchy sentences. Don't avoid colloquialisms and slang. If something is cool, just say "Wild" or "Huge"; if something sucks, call it out directly.
- **Strategy:** Always lead with "data" or "logic", never vague sentiments.

**Writing Principles:**
- Be specific, not abstract
- Use numbers and concrete examples
- Avoid corporate buzzwords at all costs
- Sound like a real human having a conversation, not a marketing department
```

---

### Task-Specific Prompt Templates

#### 🟢 Template A: Original Post (POST)

**File:** `prompts/post_template.txt`

```
Generate 3 different draft versions based on the following:

**Idea:** {yaml.idea}
**Style Context:** {yaml.style}
**Data Support (if available):** {internal_data.audit_log}

**Requirements:**
1. **Version 1 (Spicy):** Directly call out industry pain points, with some sarcasm. Be provocative.
2. **Version 2 (Hardcore):** Pure technical/data-driven. Show off Judge Model's capabilities.
3. **Version 3 (Observation):** Start with an interesting industry observation, naturally lead to DevEase.

**Constraints:**
- Each draft must be under 280 characters
- Include relevant hashtags (max 2-3)
- No corporate speak or buzzwords
- Sound authentic and conversational

**Output Format:**
Return JSON with this structure:
```json
{
  "drafts": [
    {
      "version": "spicy",
      "content": "...",
      "rationale": "Why this approach works",
      "tone": "detected_tone",
      "hashtags": ["tag1", "tag2"],
      "character_count": 123
    },
    ...
  ]
}
```
```

#### 🔵 Template B: Context-Aware Reply (REPLY_TWEET)

**File:** `prompts/reply_template.txt`

```
You're replying to a tweet. Your goal: "intercept" traffic from influential accounts, demonstrate intelligence, don't be annoying.

**Target Tweet Content:**
{scraped.tweet_text}
Author: {scraped.author_handle}
Tweet Engagement: {scraped.likes} likes, {scraped.retweets} retweets

**Conversation Context:**
{scraped.top_replies}

**My Viewpoint:**
{yaml.idea}

**Task:**
Write a reply draft that:

1. **DO NOT parrot what they said** - add value, don't repeat
2. **Match the energy:**
   - If they're complaining: Offer a technical solution
   - If they're bragging: Provide a different logical perspective
   - If they're asking questions: Give a direct, helpful answer
3. **Tone:** Like two senior engineers chatting in the same office - casual but sharp
4. **NO CTA** unless the topic is a perfect fit - no "check out my product" spam

**Constraints:**
- Under 280 characters
- No hashtags (they feel spammy in replies)
- Sound like you're already part of the conversation

**Output Format:**
```json
{
  "content": "...",
  "rationale": "Why this reply adds value to the conversation",
  "approach": "technical_solution|different_perspective|helpful_answer"
}
```
```

#### 🟡 Template C: Direct Message (REPLY_DM)

**File:** `prompts/dm_template.txt`

```
You're sending a DM to convert a potential partner. Goal: Quick conversion, build trust, don't waste their time.

**Target User:**
Handle: {user.handle}
Bio: {user.bio}
Recent Activity: {user.last_5_tweets}

**My Task:**
{yaml.idea}

**Task:**
Write a concise DM that:

1. **First sentence:** Directly respond to their need or reference their recent tweet (show you did your homework)
2. **Second sentence:** Drop our value proposition (e.g., "Partner 50% revenue share", "0-maintenance plugin", etc.)
3. **No fluff:** If they're an Agency Boss, be direct - no pleasantries needed

**Tone Guidelines:**
- For technical founders: Be precise, use data
- For agency bosses: Be direct, focus on revenue
- For marketers: Focus on efficiency and results

**Constraints:**
- Under 500 characters
- No emojis unless it feels natural
- One clear CTA or question to start conversation

**Output Format:**
```json
{
  "content": "...",
  "rationale": "Why this approach works for this specific user type",
  "personalization_elements": ["referenced their tweet about X", "noticed they're Y type"]
}
```
```

---

### Self-Correction Loop (Prompt Optimization)

**The agent learns from your actions:**

```python
class PromptOptimizer:
    """
    Tracks user acceptance/rejection patterns to improve future prompts.
    """

    def record_feedback(self, draft: TwitterDraft, action: str, user_edits: Optional[str]):
        """
        Record what happened with each draft:
        - action: 'confirmed' | 'regenerated' | 'edited' | 'skipped'
        - user_edits: What the user changed (if edited)
        """
        self.feedback_history.append({
            'draft_features': self._extract_features(draft),
            'action': action,
            'user_edits': user_edits,
            'task_metadata': {
                'type': draft.task.type,
                'style': draft.task.style,
                'tone': draft.tone
            }
        })

    def get_feedback_summary(self) -> str:
        """
        Generate insights to inject into next LLM call.
        """
        # Analyze patterns: What gets approved? What gets edited?
        # Example output: "Users consistently add more data points. Drafts with technical terminology have 80% higher acceptance."
```

**How it works:**
1. User clicks **[Send]** → Agent records: "This type of content works"
2. User clicks **[Regenerate]** → Agent analyzes: "What was wrong with this draft?"
3. User **[Edits]** draft → Agent records: "User prefers this phrasing/structure"
4. Every 10 tasks → Agent updates system prompt with learnings

---

### Few-Shot Learning Strategy

**Golden Examples File:** `config/twitter/golden_examples.json`

```json
{
  "system_prompt": "Study these examples of Xin's highest-performing tweets and internalize the style:",
  "examples": [
    {
      "input": {
        "idea": "分享今天 Judge Model 发现的一个离谱案例：某电商在 3 点全投了垃圾流量，浪费 $500。",
        "style": "犀利吐槽，硬核数据"
      },
      "output": {
        "content": "Found a $500 disaster today. E-commerce site dumped 100% budget into garbage traffic at 3 AM. Judge Model caught it instantly.\n\nThis is why manual monitoring = throwing money away.",
        "rationale": "Short, specific, numbers-first, clear problem/solution",
        "performance": {
          "likes": 847,
          "retweets": 123,
          "replies": 45
        }
      }
    },
    {
      "input": {
        "idea": "针对 AI 效率话题，提我们的自编码组织架构",
        "style": "幽默，像老朋友聊天"
      },
      "output": {
        "content": "We achieved 0-maintenance plugin architecture.\n\nNot because we're geniuses. Because we were too lazy to do manual updates.\n\nLaziness > Motivation",
        "rationale": "Self-deprecating humor, counterintuitive take, memorable",
        "performance": {
          "likes": 1203,
          "retweets": 234,
          "replies": 89
        }
      }
    }
    // Add 3-5 more examples...
  ]
}
```

**Integration in Content Agent:**

```python
class ContentAgent:
    def generate_drafts(self, task: TwitterTask, context: Optional[str] = None) -> List[TwitterDraft]:
        # Load golden examples
        examples = self._load_golden_examples()

        # Build prompt with few-shot learning
        prompt = self._build_prompt(
            task=task,
            context=context,
            examples=examples  # ← Inject golden examples here
        )

        response = self.llm_client.call(prompt)
        return self._parse_llm_response(response)
```

**Why this works:**
- Forces the model to pattern-match against proven high-performing content
- More effective than abstract style instructions
- Continuously improvable: Add new examples as you discover what works

---

### Enhanced YAML with Prompt Template Selection

**Advanced `tasks.yaml` with template override:**

```yaml
tasks:
  # Default: Uses template based on task type
  - type: POST
    idea: "Share today's Judge Model discovery"
    style: "犀利吐槽，硬核数据"
    status: PENDING

  # Override: Specify exact prompt template
  - type: REPLY_TWEET
    target_url: "https://x.com/elonmusk/status/123456"
    idea: "Show him what real ROI audit looks like"
    prompt_template: "spicy_v2"  # ← Use most aggressive template
    status: PENDING

  # Override: Provide custom system prompt
  - type: POST
    idea: "Announce new feature launch"
    style: "专业"
    custom_system_prompt: |
      You are announcing a major feature. Be exciting but not hype-y.
      Focus on user benefits, not technical details.
    status: PENDING
```

**Prompt Template Registry** (`config/agents/twitter_config.yaml`):

```yaml
# Twitter Growth Agent Configuration

agent:
  name: "twitter_growth_agent"
  version: "1.0.0"

# Prompt Templates
prompt_templates:
  default:
    post: "prompts/post_template.txt"
    reply: "prompts/reply_template.txt"
    dm: "prompts/dm_template.txt"

  custom_templates:
    spicy_v1:
      system: "You are particularly bold today. Take strong positions."
      temperature: 0.9

    spicy_v2:
      system: "You are feeling provocative. Challenge conventional wisdom."
      temperature: 1.0
      max_tokens: 280

    balanced:
      system: "Be measured and analytical. Let data speak."
      temperature: 0.7

    conservative:
      system: "Be cautious and professional. Avoid controversy."
      temperature: 0.5

# LLM Settings
llm:
  model: "gpt-4o"
  temperature: 0.8
  max_tokens: 500
  num_drafts: 3
  use_few_shot: true
  # Golden examples are now defined inline in config/tasks.yaml

# Learning Settings
learning:
  enable_self_correction: true
  min_feedback_samples: 10  # Start learning after 10 data points
  prompt_update_interval: 50  # Update system prompt every 50 tasks
```

### 4. Context Builder (`src/growth/twitter/core/context_builder.py`)

**Responsibilities**:
- For REPLY_TWEET: Scrape the tweet and its thread using Playwright
- Extract key points, tone, and previous replies
- For REPLY_DM: Extract previous DM conversation
- Build context string for LLM

**Key Functions**:
```python
class ContextBuilder:
    def __init__(self, browser_agent):
        self.browser_agent = browser_agent

    def build_reply_context(self, target_url: str) -> str:
        """Scrape tweet and build context string"""

    def build_dm_context(self, handle: str) -> str:
        """Extract DM conversation history"""
```

### 5. Browser Agent (`src/growth/twitter/agents/browser_agent.py`)

**Responsibilities**:
- Use Playwright to control Chrome (assume user is already logged in)
- Navigate to target URLs or user profiles
- Locate and interact with input elements (tweet box, reply box, DM box)
- Simulate human-like typing (random delays, typos corrected)
- Handle timeouts and element not found errors

**Key Functions**:
```python
class BrowserAgent:
    def __init__(self, config: TwitterConfig):
        self.config = config
        self.playwright = None
        self.browser = None
        self.page = None

    async def start(self):
        """Launch Chrome and connect to existing session or start new"""

    async def navigate_to_tweet(self, url: str):
        """Navigate to specific tweet URL"""

    async def navigate_to_profile(self, handle: str):
        """Navigate to user profile for DM"""

    async def fill_tweet_box(self, content: str):
        """Locate tweet box and type content with human-like speed"""

    async def fill_reply_box(self, content: str):
        """Locate reply box for specific tweet"""

    async def fill_dm_box(self, handle: str, content: str):
        """Locate DM input for specific user"""

    async def click_send(self):
        """Click the send/post button"""

    async def get_page_content(self) -> str:
        """Get page HTML for context extraction"""

    async def screenshot(self, path: str):
        """Take screenshot for debugging"""

    async def close(self):
        """Close browser"""
```

**Element Selectors** (Twitter DOM is dynamic, may need adjustment):
- Tweet box: `div[contenteditable="true"][aria-label*="Post text"]`
- Reply box: `div[contenteditable="true"][aria-label*="Reply text"]`
- Send button: `div[role="button"] css:div >> text="Post"`

### 6. UI Agent (`src/growth/twitter/agents/ui_agent.py`)

**Responsibilities**:
- Present drafts to user in CLI
- Allow user to select, edit, or regenerate drafts
- Get final confirmation before posting
- Display progress and status updates

**Key Functions**:
```python
class UIAgent:
    def present_drafts(self, task: TwitterTask, drafts: List[TwitterDraft]) -> int:
        """Display drafts and get user selection"""

    def request_confirmation(self, task: TwitterTask, draft: TwitterDraft) -> bool:
        """Show final preview and get confirmation"""

    def request_edit(self, draft: TwitterDraft) -> str:
        """Allow user to edit draft in terminal"""

    def display_progress(self, message: str):
        """Show progress update"""

    def display_error(self, error: str):
        """Show error message"""

    def display_success(self, message: str):
        """Show success message"""
```

**CLI Interface Design**:
```
📝 Task 1/3: POST tweet
Idea: "分享今天 Judge Model 发现的一个离谱案例..."

Draft Options:
[1] 🔥 Bold: "Found a $500 waste disaster today. An e-commerce site dumped 100% budget into garbage traffic at 3 AM. Judge Model caught it. This is why automated monitoring isn't optional."
[2] ⚖️ Balanced: "Case study from today: $500 saved. Our Judge Model detected an anomaly where an advertiser's entire budget shifted to low-quality traffic during off-hours. The fix? Automated alerts + instant budget reallocation."
[3] 🎯 Conservative: "Interesting anomaly detected by our Judge Model today: an advertiser accidentally bid on irrelevant traffic segments during off-hours, wasting $500. Automated monitoring saved the day."

Select draft [1-3] or [r]egenerate specific draft > 1

Selected Draft 1:
"Found a $500 waste disaster today. An e-commerce site dumped 100% budget into garbage traffic at 3 AM. Judge Model caught it. This is why automated monitoring isn't optional."

Actions:
[1] ✅ Confirm & Post
[2] ✍️ Edit Draft
[3] 🔄 Regenerate
[4] ❌ Skip Task

Choice > 1

⏳ Posting to Twitter...
✅ Posted successfully! Tweet URL: https://x.com/your_handle/status/123456789
```

### 7. Analytics Agent (`src/growth/twitter/agents/analytics_agent.py`)

**Responsibilities**:
- Track performance of posted content (likes, retweets, replies)
- Store metrics in memory system
- Analyze patterns (what styles work best)
- Provide feedback for future content generation

**Key Functions**:
```python
class AnalyticsAgent:
    def __init__(self, memory_system: MemorySystem):
        self.memory_system = memory_system

    def record_post(self, task: TwitterTask, draft: TwitterDraft, tweet_url: str):
        """Record posted content with metadata"""

    def fetch_metrics(self, tweet_url: str) -> PerformanceMetrics:
        """Fetch engagement metrics using Playwright or Twitter API"""

    def analyze_patterns(self, tasks: List[TwitterTask]) -> Dict[str, Any]:
        """Analyze what content styles perform best"""

    def get_feedback_for_generation(self) -> str:
        """Generate insights to inform future content generation"""
```

### 8. Orchestrator (`src/growth/twitter/agents/orchestrator.py`)

**Responsibilities**:
- Coordinate all agents in the workflow
- Manage task queue and state transitions
- Handle errors and retries
- Persist task status

**Workflow**:
```python
class TwitterOrchestrator:
    def __init__(self, config_path: str):
        self.yaml_parser = YAMLTaskParser(config_path)
        self.content_agent = ContentAgent(llm_client)
        self.browser_agent = BrowserAgent(config)
        self.ui_agent = UIAgent()
        self.analytics_agent = AnalyticsAgent(memory)
        self.running = False

    async def run(self):
        """Main workflow loop"""
        while self.running:
            tasks = self.yaml_parser.load_tasks()

            for task in tasks:
                if task.status != TaskStatus.PENDING:
                    continue

                await self.process_task(task)

            await asyncio.sleep(10)  # Poll for new tasks

    async def process_task(self, task: TwitterTask):
        """Process a single task through all phases"""
        try:
            # Phase 1: Drafting
            task.status = TaskStatus.DRAFTING
            context = await self.build_context(task)
            drafts = self.content_agent.generate_drafts(task, context)
            task.status = TaskStatus.READY_FOR_REVIEW

            # Phase 2: Review & Selection
            selected_index = self.ui_agent.present_drafts(task, drafts)
            task.selected_draft_index = selected_index

            while True:
                selected_draft = drafts[selected_index]

                # Phase 3: Final Confirmation
                confirmed = self.ui_agent.request_confirmation(task, selected_draft)
                if not confirmed:
                    action = self.ui_agent.request_action()
                    if action == "regenerate":
                        drafts[selected_index] = self.content_agent.regenerate_draft(task, selected_index, context)
                        continue
                    elif action == "edit":
                        drafts[selected_index].content = self.ui_agent.request_edit(selected_draft)
                        continue
                    elif action == "skip":
                        return

                break

            # Phase 4: Browser Loading & Posting
            await self.browser_agent.start()
            if task.type == TaskType.POST:
                await self.browser_agent.fill_tweet_box(selected_draft.content)
            elif task.type == TaskType.REPLY_TWEET:
                await self.browser_agent.navigate_to_tweet(task.target_url)
                await self.browser_agent.fill_reply_box(selected_draft.content)
            elif task.type == TaskType.REPLY_DM:
                await self.browser_agent.navigate_to_profile(task.handle)
                await self.browser_agent.fill_dm_box(task.handle, selected_draft.content)

            task.status = TaskStatus.CONFIRMED
            await self.browser_agent.click_send()
            task.status = TaskStatus.POSTED

            # Phase 5: Analytics
            tweet_url = await self.browser_agent.get_current_url()
            self.analytics_agent.record_post(task, selected_draft, tweet_url)

            # Phase 6: Complete
            self.yaml_parser.mark_completed(task.id)
            self.ui_agent.display_success(f"Posted! {tweet_url}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self.ui_agent.display_error(f"Failed: {e}")
            self.yaml_parser.update_task_status(task.id, TaskStatus.FAILED)

    async def build_context(self, task: TwitterTask) -> Optional[str]:
        """Build context for replies"""
        if task.type == TaskType.REPLY_TWEET:
            return self.context_builder.build_reply_context(task.target_url)
        elif task.type == TaskType.REPLY_DM:
            return self.context_builder.build_dm_context(task.handle)
        return None

    def stop(self):
        """Stop the orchestrator"""
        self.running = False
```

---

## Dependencies to Add

**Add to `requirements.txt`**:
```txt
playwright>=1.40.0
watchdog>=4.0.0
questionary>=2.0.0  # For rich CLI prompts
python-dotenv>=1.0.0  # For environment variable loading
```

**Install Playwright browsers**:
```bash
playwright install chromium
```

---

## Key Management & Authentication

### API Keys from `~/.devease/keys`

The agent should parse API keys from a centralized keys file to avoid hardcoding sensitive credentials.

**Keys File Format** (`~/.devease/keys`):
```ini
# Twitter Growth Agent Keys
# Format: KEY_NAME=value

# OpenAI API (for content generation) - REQUIRED
OPENAI_API_KEY=sk-...
OPENAI_ORG_ID=org-...

# Twitter API (optional, for analytics/fallback)
TWITTER_API_KEY=...
TWITTER_API_SECRET=...
TWITTER_ACCESS_TOKEN=...
TWITTER_ACCESS_SECRET=...
TWITTER_BEARER_TOKEN=...

# Twitter Cookies (for browser automation)
# Store browser session cookies after first login
TWITTER_COOKIES_PATH=~/.devease/twitter_cookies.json

# DevEase Internal (for golden examples, audit logs)
DEVEASE_DATA_PATH=/path/to/devease/data
```

**Important**: No template file is stored in the repository. The code generates helpful instructions when the keys file is missing.

**Key Parser Implementation** (`src/growth/twitter/core/key_manager.py`):

```python
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class TwitterKeys:
    openai_api_key: str
    openai_org_id: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    twitter_cookies_path: Optional[str] = None
    devease_data_path: Optional[str] = None

class KeyManager:
    """
    Parse and manage API keys from ~/.devease/keys file.
    Provides helpful instructions if file is missing.
    """

    DEFAULT_KEYS_PATH = Path.home() / ".devease" / "keys"

    def __init__(self, keys_path: Optional[Path] = None):
        self.keys_path = keys_path or self.DEFAULT_KEYS_PATH
        self._keys: Optional[TwitterKeys] = None

    def load_keys(self) -> TwitterKeys:
        """Load and parse keys from file."""
        if not self.keys_path.exists():
            self._print_setup_instructions()
            raise FileNotFoundError(
                f"\nKeys file not found at {self.keys_path}\n"
                f"Please create it with the instructions above."
            )

        env_vars = {}
        with open(self.keys_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

        # Expand ~ in paths
        cookies_path = env_vars.get('TWITTER_COOKIES_PATH', '~/.devease/twitter_cookies.json')
        data_path = env_vars.get('DEVEASE_DATA_PATH', '~/devease/data')

        self._keys = TwitterKeys(
            openai_api_key=env_vars.get('OPENAI_API_KEY', ''),
            openai_org_id=env_vars.get('OPENAI_ORG_ID'),
            twitter_api_key=env_vars.get('TWITTER_API_KEY'),
            twitter_api_secret=env_vars.get('TWITTER_API_SECRET'),
            twitter_access_token=env_vars.get('TWITTER_ACCESS_TOKEN'),
            twitter_access_secret=env_vars.get('TWITTER_ACCESS_SECRET'),
            twitter_bearer_token=env_vars.get('TWITTER_BEARER_TOKEN'),
            twitter_cookies_path=os.path.expanduser(cookies_path),
            devease_data_path=os.path.expanduser(data_path)
        )

        return self._keys

    def _print_setup_instructions(self):
        """Print helpful instructions for creating the keys file."""
        print("\n" + "="*70)
        print("Twitter Growth Agent - Keys Setup")
        print("="*70)
        print(f"\nKeys file not found: {self.keys_path}")
        print("\nStep 1: Create the directory")
        print(f"  mkdir -p {self.keys_path.parent}")
        print("\nStep 2: Create the keys file")
        print(f"  cat > {self.keys_path} << 'EOF'")
        print("# Twitter Growth Agent Keys")
        print("# Format: KEY_NAME=value")
        print("")
        print("# REQUIRED: OpenAI API (for content generation)")
        print("OPENAI_API_KEY=sk-your-openai-api-key-here")
        print("OPENAI_ORG_ID=org-your-org-id-here  # Optional")
        print("")
        print("# OPTIONAL: Twitter API (for analytics/metrics)")
        print("TWITTER_API_KEY=your-twitter-api-key-here")
        print("TWITTER_API_SECRET=your-twitter-api-secret-here")
        print("TWITTER_ACCESS_TOKEN=your-twitter-access-token-here")
        print("TWITTER_ACCESS_SECRET=your-twitter-access-secret-here")
        print("TWITTER_BEARER_TOKEN=your-twitter-bearer-token-here")
        print("")
        print("# OPTIONAL: Browser session cookies")
        print("TWITTER_COOKIES_PATH=~/.devease/twitter_cookies.json")
        print("")
        print("# OPTIONAL: DevEase data paths")
        print("DEVEASE_DATA_PATH=~/devease/data")
        print("EOF")
        print("\nStep 3: Set restrictive permissions")
        print(f"  chmod 600 {self.keys_path}")
        print("\nStep 4: Edit the file and add your actual API keys")
        print(f"  nano {self.keys_path}")
        print("\n" + "="*70)

    @property
    def keys(self) -> TwitterKeys:
        """Get loaded keys (lazy load)."""
        if self._keys is None:
            self.load_keys()
        return self._keys

    def validate_required_keys(self) -> bool:
        """Check that required keys are present."""
        keys = self.keys
        if not keys.openai_api_key or keys.openai_api_key.startswith('sk-'):
            raise ValueError(
                f"OPENAI_API_KEY is required but not properly set in {self.keys_path}\n"
                f"Current value: {keys.openai_api_key[:20]}..." if keys.openai_api_key else "empty"
            )
        return True
```

**Usage in Orchestrator**:

```python
class TwitterOrchestrator:
    def __init__(self, config_path: str):
        # Load API keys
        self.key_manager = KeyManager()
        self.keys = self.key_manager.load_keys()
        self.key_manager.validate_required_keys()

        # Initialize LLM client with keys
        self.llm_client = OpenAI(
            api_key=self.keys.openai_api_key,
            organization=self.keys.openai_org_id
        )

        # Initialize browser agent with cookies path
        self.browser_agent = BrowserAgent(
            config=config,
            cookies_path=self.keys.twitter_cookies_path
        )
```

**Environment Variable Fallback**:

The `KeyManager` should also support environment variables as fallback:

```python
def load_keys(self) -> TwitterKeys:
    """Load keys from file, with env var fallback."""
    # ... parse file ...

    # Fallback to environment variables if not in file
    return TwitterKeys(
        openai_api_key=env_vars.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY', ''),
        openai_org_id=env_vars.get('OPENAI_ORG_ID') or os.getenv('OPENAI_ORG_ID'),
        # ... other keys with env fallback ...
    )
```

**Security Notes**:

1. **File Permissions**: Keys file should have restricted permissions:
   ```bash
   chmod 600 ~/.devease/keys
   ```

2. **Git Ignore**: Add to `.gitignore`:
   ```
   .devease/
   *.cookies.json
   ~/.devease/
   ```

3. **No Template in Repo**: Template files are NOT stored in the repository. The code provides helpful instructions when keys are missing. This prevents accidental commits of keys.

4. **Validation**: Agent should fail fast with clear error if keys are missing or invalid:
   ```
   Error: OPENAI_API_KEY is required but not properly set in ~/.devease/keys
   Current value: sk-your-openai...
   ```

---

## Configuration Files

### 1. Twitter Agent Config (`config/agents/twitter_config.yaml`)

```yaml
# Twitter Growth Agent Configuration

agent:
  name: "twitter_growth_agent"
  version: "1.0.0"

# LLM Settings
llm:
  model: "gpt-4o"
  temperature: 0.8
  max_tokens: 500
  num_drafts: 3

# Browser Settings
browser:
  headless: false  # Show browser for debugging
  slow_mo: 100  # Slow down actions by 100ms
  timeout: 30000
  user_data_dir: null  # Use default Chrome profile or specify path

# Human Interaction
ui:
  human_confirmation: true
  show_rationale: true
  enable_editing: true

# Anti-Detection
anti_bot:
  typing_delay_min: 50  # ms per character
  typing_delay_max: 150
  random_pauses: true
  action_delay_min: 2  # seconds
  action_delay_max: 8

# Analytics
analytics:
  enabled: true
  check_metrics_after: 3600  # seconds (1 hour)
  metrics_history_file: "data/twitter/metrics.json"

# Task Monitoring
monitoring:
  poll_interval: 10  # seconds
  watch_file_changes: true
  max_concurrent_tasks: 1
```

### 2. Tasks YAML Template (`config/twitter/tasks.yaml`)

```yaml
# Twitter Growth Tasks
# Add new tasks to the list. Agent will process PENDING tasks.

tasks:
  - type: POST
    idea: "分享今天 Judge Model 发现的一个离谱案例：某电商在 3 点全投了垃圾流量，浪费 $500。"
    style: "犀利吐槽，硬核数据，最后带一个钩子"
    status: PENDING

  - type: REPLY_TWEET
    target_url: "https://x.com/elonmusk/status/123456"
    idea: "针对他说的 AI 效率，提一下我们的自编码组织已经实现了 0 人力维护插件。"
    style: "幽默，像老朋友聊天，不要像推销"
    status: PENDING

  - type: REPLY_DM
    handle: "ad_agency_boss"
    idea: "回复他关于插件定价的问题，说我们可以给内测期 Partner 50% 分成。"
    style: "专业，诚恳"
    status: PENDING

global_settings:
  human_confirmation: true
  anti_bot_delay: [2, 8]
```

---

## CLI Integration

**Add to `run.py`**:

```python
# Twitter Growth Agent commands
twitter_parser = subparsers.add_parser('twitter', help='Twitter Growth Agent')
twitter_subparsers = twitter_parser.add_subparsers(dest='twitter_command')

# Start command
start_parser = twitter_subparsers.add_parser('start', help='Start Twitter agent')
start_parser.add_argument('--config', type=str, default='config/agents/twitter_config.yaml',
                          help='Path to agent config')
start_parser.add_argument('--tasks', type=str, default='config/twitter/tasks.yaml',
                          help='Path to tasks YAML')

# Test command
test_parser = twitter_subparsers.add_parser('test', help='Test browser connection')
test_parser.add_argument('--url', type=str, help='Test navigate to URL')

# Status command
status_parser = twitter_subparsers.add_parser('status', help='Show task status')
status_parser.add_argument('--tasks', type=str, default='config/twitter/tasks.yaml',
                           help='Path to tasks YAML')
```

---

## Implementation Steps

### Phase 1: Foundation (Core Types & Config)
1. Create directory structure under `src/growth/twitter/`
2. Implement `core/types.py` with all dataclasses
3. Implement `core/yaml_parser.py` for loading and parsing tasks
4. Create `config/agents/twitter_config.yaml`
5. Create sample `config/twitter/tasks.yaml`
6. Write unit tests for types and YAML parser

### Phase 2: Content Generation
7. Implement `agents/content_agent.py` with OpenAI integration
8. Create `prompts/tweet_generation.txt`, `reply_generation.txt`, `dm_generation.txt`
9. Integrate with existing `LLMClientWithRetry` from `src/utils/llm_retry.py`
10. Write unit tests for content generation (mock LLM responses)

### Phase 3: Browser Automation
11. Implement `agents/browser_agent.py` with Playwright
12. Implement `utils/typing_simulator.py` for human-like typing
13. Implement `utils/url_normalizer.py` for URL handling
14. Test browser navigation and element location
15. Write integration tests for browser actions

### Phase 4: Context Building
16. Implement `core/context_builder.py` for thread analysis
17. Add logic to scrape tweet content and replies
18. Add logic to extract DM conversation history
19. Test context extraction with real tweets

### Phase 5: CLI Interface
20. Implement `agents/ui_agent.py` with rich CLI prompts
21. Use `questionary` library for interactive menus
22. Add draft selection, editing, and confirmation flows
23. Test CLI UX manually

### Phase 6: Orchestration
24. Implement `agents/orchestrator.py` with full workflow
25. Add error handling and retry logic
26. Integrate all agents in the main loop
27. Add file watching for tasks.yaml changes

### Phase 7: Analytics & Learning
28. Implement `agents/analytics_agent.py`
29. Implement `core/memory.py` for storing metrics
30. Add metric fetching logic (scraping or API)
31. Add pattern analysis

### Phase 8: Integration & Polish
32. Add CLI commands to `run.py`
33. Add end-to-end integration tests
34. Write documentation
35. Test complete workflow with real tasks

---

## Testing Strategy

### Unit Tests
- `test_types.py`: Test dataclass creation and validation
- `test_yaml_parser.py`: Test YAML loading, parsing, status updates
- `test_content_agent.py`: Mock LLM, test prompt building, response parsing
- `test_context_builder.py`: Test context string generation

### Integration Tests
- `test_twitter_workflow.py`: Test full workflow with fake browser
- Use Playwright's `browser_context` to test without real Twitter
- Mock LLM responses for deterministic testing

### Manual Testing Checklist
- [ ] Load tasks from YAML
- [ ] Generate drafts for POST task
- [ ] Generate drafts for REPLY_TWEET with context
- [ ] Generate drafts for REPLY_DM with context
- [ ] CLI displays drafts correctly
- [ ] User can select, edit, regenerate drafts
- [ ] Browser navigates to correct URLs
- [ ] Browser fills content with human-like typing
- [ ] Click send button works
- [ ] Task status updates correctly
- [ ] Completed tasks marked in YAML
- [ ] Analytics records posts
- [ ] Error handling works (timeout, element not found)

---

## Key Implementation Notes

### 1. Playwright Setup
Use persistent context to keep user logged in:
```python
context = browser.new_context(
    storage_state="path/to/cookies.json",  # Save cookies after first login
    viewport={"width": 1280, "height": 720},
    user_agent="Mozilla/5.0 ..."
)
```

### 2. Twitter DOM Challenges
- Twitter uses React with dynamic classes
- Use `aria-label` and `role` attributes for stable selectors
- Consider using Twitter API as fallback if DOM changes break automation

### 3. Human-Like Typing
```python
import random
import asyncio

async def human_type(page, selector, text):
    await page.click(selector)
    for char in text:
        await page.keyboard.type(char)
        await asyncio.sleep(random.uniform(0.05, 0.15))  # 50-150ms per char
```

### 4. Error Recovery
- If element not found: Wait up to timeout, then retry
- If posting fails: Save draft to file for manual posting
- If browser crashes: Restart and continue with next task

### 5. Rate Limiting
- Respect Twitter's rate limits (posts: 300/day, DMs: 500/day)
- Add delays between tasks
- Log all actions for debugging

---

## Critical Files to Modify/Create

**New Files** (all under `src/growth/twitter/`):
- `core/types.py` - Data models
- `core/yaml_parser.py` - YAML loading
- `core/context_builder.py` - Context extraction
- `core/key_manager.py` - Parse API keys from ~/.devease/keys
- `agents/content_agent.py` - LLM content generation
- `agents/browser_agent.py` - Playwright automation
- `agents/ui_agent.py` - CLI interface
- `agents/analytics_agent.py` - Performance tracking
- `agents/orchestrator.py` - Main workflow

**Existing Files to Modify**:
- `run.py` - Add Twitter CLI commands
- `requirements.txt` - Add Playwright, watchdog, questionary, python-dotenv
- `.gitignore` - Add .devease/ and *.cookies.json

**Config Files**:
- `config/agents/twitter_config.yaml` - Agent configuration
- `config/twitter/tasks.yaml` - User task definitions
- `config/twitter/golden_examples.json` - Few-shot learning examples
- `config/twitter/prompts/post_template.txt` - POST prompt template
- `config/twitter/prompts/reply_template.txt` - REPLY prompt template
- `config/twitter/prompts/dm_template.txt` - DM prompt template

**User Files** (created by users, never in repo):
- `~/.devease/keys` - API keys (created by user with code-generated instructions)
- `~/.devease/twitter_cookies.json` - Browser session cookies (auto-generated after first login)

---

## Verification Steps

After implementation:

0. **Setup Keys** (first time only):
   ```bash
   # The code will provide instructions when keys file is missing
   python run.py twitter start
   # Follow the printed instructions to create ~/.devease/keys
   ```

1. **Test Key Management**:
   ```bash
   python -m src.growth.twitter.core.key_manager
   # Should successfully load and validate keys
   # Or print helpful instructions if file is missing
   ```

2. **Test YAML Loading**:
   ```bash
   python run.py twitter status --tasks config/twitter/tasks.yaml
   ```

3. **Test Browser Connection**:
   ```bash
   python run.py twitter test --url https://x.com/elonmusk/status/123456
   ```

4. **Run Full Workflow**:
   ```bash
   python run.py twitter start --config config/agents/twitter_config.yaml --tasks config/twitter/tasks.yaml
   ```

5. **Verify Posting**:
   - Check that tweets appear on Twitter
   - Check that tasks are marked COMPLETED in YAML
   - Check that analytics are recorded

6. **Test Error Scenarios**:
   - Missing ~/.devease/keys file (should print instructions)
   - Invalid API key in keys file
   - Invalid URL in target_url
   - LLM timeout
   - Browser element not found
   - User rejects all drafts

---

---

## Golden Examples Collection (Xin's Voice)

**To initialize the few-shot learning system, here are 5 example tweets that embody the Xin persona. Add these to `config/twitter/golden_examples.json`:**

### Example 1: The "Spicy Data" Tweet
```json
{
  "input": {
    "idea": "发现某客户的 Judge Model 在凌晨3点检测到异常流量，浪费了$500预算",
    "style": "犀利吐槽，硬核数据"
  },
  "output": {
    "content": "Found a $500 disaster today.\n\nE-commerce site dumped 100% budget into garbage traffic at 3 AM. Judge Model caught it instantly.\n\nThis is why manual monitoring = throwing money away.",
    "rationale": "Short, specific, numbers-first, clear problem/solution, no fluff",
    "performance": {
      "likes": 847,
      "retweets": 123,
      "replies": 45
    }
  }
}
```

### Example 2: The "Counterintuitive" Tweet
```json
{
  "input": {
    "idea": "宣传我们的0维护插件架构，是因为懒而不是聪明",
    "style": "幽默，自嘲"
  },
  "output": {
    "content": "We achieved 0-maintenance plugin architecture.\n\nNot because we're geniuses.\n\nBecause we were too lazy to do manual updates.\n\nLaziness > Motivation",
    "rationale": "Self-deprecating humor, counterintuitive take, memorable framing",
    "performance": {
      "likes": 1203,
      "retweets": 234,
      "replies": 89
    }
  }
}
```

### Example 3: The "Technical Roast" Tweet
```json
{
  "input": {
    "idea": "吐槽广告行业滥用AI生成垃圾创意",
    "style": "犀利吐槽"
  },
  "output": {
    "content": "Everyone's using AI to generate more ads.\n\nMeanwhile we're using AI to DELETE bad ads.\n\nOne of these approaches makes money.\n\nGuess which one?",
    "rationale": "Provocative contrast, implied superiority, clear value proposition without naming features",
    "performance": {
      "likes": 2341,
      "retweets": 567,
      "replies": 234
    }
  }
}
```

### Example 4: The "Observation" Tweet
```json
{
  "input": {
    "idea": "观察到大部分广告主都在优化创意，但没有人优化投放逻辑",
    "style": "观察，温和引导"
  },
  "output": {
    "content": "Interesting observation:\n\n90% of ad tech companies focus on \"better creatives.\"\n\nAlmost nobody focuses on \"don't show bad creatives in the first place.\"\n\nSecond one is 100x more valuable.",
    "rationale": "Starts with observation, builds to insight, data-backed claim, implies DevEase's value",
    "performance": {
      "likes": 1567,
      "retweets": 289,
      "replies": 123
    }
  }
}
```

### Example 5: The "Direct Reply" Example
```json
{
  "input": {
    "idea": "回复一个抱怨广告投放效率低的VP",
    "style": "专业但直接",
    "context": {
      "target_tweet": "We're spending $50k/mo on ads and can't tell what's working. Agency reports are useless.",
      "author": "VP Marketing at e-commerce brand"
    }
  },
  "output": {
    "content": "The problem isn't measurement.\n\nThe problem is you're running ads that should never have been launched.\n\nFix the upstream质量问题, reporting becomes irrelevant.\n\nHappy to audit your account for free if you want to see what I mean.",
    "rationale": "Directly addresses pain point, reframes problem, offers specific value (free audit), no hard sell",
    "performance": {
      "likes": 234,
      "retweets": 45,
      "replies": 23,
      "conversion": "Yes - became a partner"
    }
  }
}
```

**Key Patterns Extracted from These Examples:**

1. **Forbid corporate vocabulary** - No "transforming", "empowering", "leveraging"
2. **Start strong** - First sentence must hook immediately
3. **Data > Adjectives** - "$500" > "expensive", "3 AM" > "late at night"
4. **Short paragraphs** - 1-2 sentences per line, easy to scan
5. **Counterintuitive framing** - "Laziness > Motivation" (unexpected but memorable)
6. **No explicit CTAs in posts** - Let the value speak for itself
7. **Replies add value** - Don't parrot, offer a different perspective
8. **Self-deprecation works** - "Not because we're geniuses" builds trust

**Template for Adding New Examples:**

When you find a tweet that performs exceptionally well (or you write one manually that you love), add it to `golden_examples.json` with:
- Original input (idea + style)
- Final output (content)
- Rationale (why it worked)
- Performance metrics (if available)

Over time, the agent will learn your voice and generate content that feels indistinguishable from your own writing.

---

## Future Enhancements

1. **Twitter API Integration**: Add option to use API instead of browser for posting
2. **Scheduler**: Add time-based scheduling for posts
3. **Multi-Account Support**: Support multiple Twitter accounts
4. **A/B Testing**: Post variations and track which performs better
5. **Image Support**: Add image upload capability
6. **Thread Support**: Generate and post multi-tweet threads
7. **Web Dashboard**: Add web UI for review instead of CLI
8. **Voice Cloning**: Use more advanced few-shot learning to capture even more nuanced personality traits
