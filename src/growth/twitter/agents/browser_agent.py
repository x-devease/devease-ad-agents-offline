"""
Browser Agent for Twitter Growth Agent.

Automates Twitter interactions using Playwright.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from ..core.types import TwitterTask, TwitterDraft, TwitterKeys, TwitterConfig

logger = logging.getLogger(__name__)


@dataclass
class BrowserActionResult:
    """Result of a browser action."""
    success: bool
    message: str
    screenshot_path: Optional[str] = None
    tweet_url: Optional[str] = None
    error: Optional[str] = None


class BrowserAgent:
    """
    Automate Twitter interactions using Playwright.

    Responsibilities:
    - Navigate to Twitter URLs (tweets, profiles)
    - Fill tweet/reply/DM boxes with human-like typing
    - Click send buttons
    - Anti-detection (random delays, typing simulation)
    - Screenshot capture for debugging
    """

    def __init__(self, keys: TwitterKeys, config: TwitterConfig, headless: bool = True):
        """
        Initialize browser agent.

        Args:
            keys: TwitterKeys object with session cookies
            config: TwitterConfig object
            headless: Whether to run browser in headless mode
        """
        self.keys = keys
        self.config = config
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None
        self._init_playwright()

    def _init_playwright(self):
        """Initialize Playwright browser."""
        try:
            from playwright.sync_api import sync_playwright
            self.sync_playwright = sync_playwright
            logger.info("Playwright initialized successfully")
        except ImportError:
            logger.error("Playwright package not installed. Run: pip install playwright")
            raise

    def _start_browser(self):
        """Start browser and authenticate with session cookies."""
        if self.browser is not None:
            return  # Already started

        playwright = self.sync_playwright().start()

        # Launch browser
        self.browser = playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        )

        # Create context with realistic user agent
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/Los_Angeles',
        )

        # Add session cookies if available
        if self.keys.twitter_cookies_path:
            import json
            cookies_path = Path(self.keys.twitter_cookies_path)
            if cookies_path.exists():
                with open(cookies_path, 'r') as f:
                    cookies = json.load(f)
                    self.context.add_cookies(cookies)
                    logger.info(f"Loaded {len(cookies)} session cookies")

        # Create page
        self.page = self.context.new_page()
        self.page.set_default_timeout(30000)  # 30 seconds

        logger.info("Browser started successfully")

    def _stop_browser(self):
        """Stop browser and cleanup."""
        if self.page:
            self.page.close()
            self.page = None
        if self.context:
            self.context.close()
            self.context = None
        if self.browser:
            self.browser.close()
            self.browser = None
        logger.info("Browser stopped")

    def _human_like_type(self, element, text: str, min_delay: float = 0.05, max_delay: float = 0.15):
        """
        Type text with human-like delays between characters.

        Args:
            element: Playwright element handle
            text: Text to type
            min_delay: Minimum delay between characters (seconds)
            max_delay: Maximum delay between characters (seconds)
        """
        import random
        import time

        element.click()  # Focus the element

        for char in text:
            element.type(char)
            # Random delay to simulate human typing
            time.sleep(random.uniform(min_delay, max_delay))

    def _random_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """
        Add random delay to simulate human behavior.

        Args:
            min_seconds: Minimum delay
            max_seconds: Maximum delay
        """
        import random
        import time
        time.sleep(random.uniform(min_seconds, max_seconds))

    def _take_screenshot(self, name: str) -> Optional[str]:
        """
        Take screenshot for debugging.

        Args:
            name: Screenshot name

        Returns:
            Path to screenshot file
        """
        try:
            screenshots_dir = Path("logs/twitter/screenshots")
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            timestamp = asyncio.get_event_loop().time()
            screenshot_path = screenshots_dir / f"{name}_{int(timestamp)}.png"

            self.page.screenshot(path=str(screenshot_path))
            logger.info(f"Screenshot saved: {screenshot_path}")

            return str(screenshot_path)
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    def post_tweet(self, draft: TwitterDraft) -> BrowserActionResult:
        """
        Post a tweet.

        Args:
            draft: TwitterDraft to post

        Returns:
            BrowserActionResult with tweet URL or error
        """
        try:
            self._start_browser()

            # Navigate to Twitter home
            logger.info("Navigating to Twitter home")
            self.page.goto("https://twitter.com/home", wait_until="networkidle")

            # Wait for tweet box to be visible
            self._random_delay(2, 4)
            tweet_box = self.page.wait_for_selector('[data-testid="tweetTextarea_0"]', timeout=10000)

            # Type content with human-like typing
            logger.info(f"Typing tweet: {draft.content[:50]}...")
            self._human_like_type(tweet_box, draft.content)

            # Take screenshot before posting
            screenshot = self._take_screenshot("pre_tweet")

            # Click tweet button
            self._random_delay(1, 2)
            tweet_button = self.page.wait_for_selector('[data-testid="tweetButton"]', timeout=5000)
            tweet_button.click()

            # Wait for tweet to be posted
            self._random_delay(2, 3)

            # Get tweet URL from URL bar
            tweet_url = self.page.url
            logger.info(f"Tweet posted successfully: {tweet_url}")

            # Take screenshot after posting
            self._take_screenshot("post_tweet")

            return BrowserActionResult(
                success=True,
                message="Tweet posted successfully",
                screenshot_path=screenshot,
                tweet_url=tweet_url
            )

        except Exception as e:
            logger.error(f"Failed to post tweet: {e}")
            screenshot = self._take_screenshot("error_tweet")

            return BrowserActionResult(
                success=False,
                message="Failed to post tweet",
                screenshot_path=screenshot,
                error=str(e)
            )

        finally:
            self._stop_browser()

    def reply_to_tweet(self, draft: TwitterDraft, target_url: str) -> BrowserActionResult:
        """
        Reply to a tweet.

        Args:
            draft: TwitterDraft with reply content
            target_url: URL of tweet to reply to

        Returns:
            BrowserActionResult with tweet URL or error
        """
        try:
            self._start_browser()

            # Navigate to tweet
            logger.info(f"Navigating to tweet: {target_url}")
            self.page.goto(target_url, wait_until="networkidle")

            # Wait for reply box to be visible
            self._random_delay(2, 4)
            reply_box = self.page.wait_for_selector('[data-testid="reply"]', timeout=10000)
            reply_box.click()

            # Wait for text area to appear
            self._random_delay(0.5, 1)
            text_area = self.page.wait_for_selector('[data-testid="tweetTextarea_0"]', timeout=5000)

            # Type reply with human-like typing
            logger.info(f"Typing reply: {draft.content[:50]}...")
            self._human_like_type(text_area, draft.content)

            # Take screenshot before posting
            screenshot = self._take_screenshot("pre_reply")

            # Click reply button
            self._random_delay(1, 2)
            reply_button = self.page.wait_for_selector('[data-testid="tweetButton"]', timeout=5000)
            reply_button.click()

            # Wait for reply to be posted
            self._random_delay(2, 3)

            logger.info("Reply posted successfully")

            # Take screenshot after posting
            self._take_screenshot("post_reply")

            return BrowserActionResult(
                success=True,
                message="Reply posted successfully",
                screenshot_path=screenshot,
                tweet_url=target_url  # Reply URL is the same as target tweet
            )

        except Exception as e:
            logger.error(f"Failed to post reply: {e}")
            screenshot = self._take_screenshot("error_reply")

            return BrowserActionResult(
                success=False,
                message="Failed to post reply",
                screenshot_path=screenshot,
                error=str(e)
            )

        finally:
            self._stop_browser()

    def send_dm(self, draft: TwitterDraft, handle: str) -> BrowserActionResult:
        """
        Send a direct message.

        Args:
            draft: TwitterDraft with DM content
            handle: User handle to send DM to (without @)

        Returns:
            BrowserActionResult with success status
        """
        try:
            self._start_browser()

            # Navigate to messages
            logger.info("Navigating to Twitter messages")
            self.page.goto("https://twitter.com/messages", wait_until="networkidle")

            # Click new message button
            self._random_delay(2, 3)
            new_message_button = self.page.wait_for_selector('a[href="/messages/compose"]', timeout=10000)
            new_message_button.click()

            # Wait for search box and type handle
            self._random_delay(1, 2)
            search_box = self.page.wait_for_selector('[data-testid="searchInput"]', timeout=10000)
            self._human_like_type(search_box, handle)

            # Wait for user to appear in search results and click
            self._random_delay(1, 2)
            user_result = self.page.wait_for_selector(f'text={handle}', timeout=5000)
            user_result.click()

            # Wait for message box and type DM
            self._random_delay(1, 2)
            message_box = self.page.wait_for_selector('[data-testid="dmComposerTextInput"]', timeout=10000)

            logger.info(f"Typing DM: {draft.content[:50]}...")
            self._human_like_type(message_box, draft.content)

            # Take screenshot before sending
            screenshot = self._take_screenshot("pre_dm")

            # Click send button
            self._random_delay(1, 2)
            send_button = self.page.wait_for_selector('[data-testid="dmComposerSendButton"]', timeout=5000)
            send_button.click()

            # Wait for DM to be sent
            self._random_delay(2, 3)

            logger.info(f"DM sent successfully to {handle}")

            # Take screenshot after sending
            self._take_screenshot("post_dm")

            return BrowserActionResult(
                success=True,
                message=f"DM sent to {handle}",
                screenshot_path=screenshot
            )

        except Exception as e:
            logger.error(f"Failed to send DM: {e}")
            screenshot = self._take_screenshot("error_dm")

            return BrowserActionResult(
                success=False,
                message="Failed to send DM",
                screenshot_path=screenshot,
                error=str(e)
            )

        finally:
            self._stop_browser()

    def navigate_to_url(self, url: str) -> BrowserActionResult:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to

        Returns:
            BrowserActionResult with success status
        """
        try:
            self._start_browser()

            logger.info(f"Navigating to: {url}")
            self.page.goto(url, wait_until="networkidle")

            self._random_delay(2, 3)

            return BrowserActionResult(
                success=True,
                message=f"Navigated to {url}"
            )

        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            return BrowserActionResult(
                success=False,
                message=f"Failed to navigate to {url}",
                error=str(e)
            )

    def get_tweet_context(self, tweet_url: str) -> Dict[str, Any]:
        """
        Extract context from a tweet for reply generation.

        Args:
            tweet_url: URL of tweet to analyze

        Returns:
            Dict with tweet text, author, engagement metrics, top replies
        """
        try:
            self._start_browser()

            logger.info(f"Extracting context from: {tweet_url}")
            self.page.goto(tweet_url, wait_until="networkidle")

            self._random_delay(2, 3)

            # Extract tweet text
            tweet_text_elem = self.page.query_selector('[data-testid="tweetText"]')
            tweet_text = tweet_text_elem.inner_text() if tweet_text_elem else ""

            # Extract author handle
            author_elem = self.page.query_selector('[data-testid="User-Name"] a')
            author_handle = author_elem.get_attribute("href").strip("/") if author_elem else ""

            # Extract engagement metrics
            likes_elem = self.page.query_selector('[data-testid="like"] span')
            likes = int(likes_elem.inner_text().replace(",", "")) if likes_elem else 0

            retweets_elem = self.page.query_selector('[data-testid="retweet"] span')
            retweets = int(retweets_elem.inner_text().replace(",", "")) if retweets_elem else 0

            # Extract top 3 replies
            reply_elems = self.page.query_selector_all('[data-testid="tweet"]')[1:4]  # Skip original tweet
            top_replies = []
            for reply_elem in reply_elems:
                reply_text_elem = reply_elem.query_selector('[data-testid="tweetText"]')
                if reply_text_elem:
                    top_replies.append(reply_text_elem.inner_text())

            context = {
                "tweet_text": tweet_text,
                "author_handle": author_handle,
                "likes": likes,
                "retweets": retweets,
                "top_replies": top_replies
            }

            logger.info(f"Extracted context: {len(tweet_text)} chars, {likes} likes, {len(top_replies)} replies")

            return context

        except Exception as e:
            logger.error(f"Failed to extract context: {e}")
            return {}

        finally:
            self._stop_browser()

    def get_user_context(self, handle: str) -> Dict[str, Any]:
        """
        Extract context from a user profile for DM generation.

        Args:
            handle: User handle (without @)

        Returns:
            Dict with bio and recent tweets
        """
        try:
            self._start_browser()

            profile_url = f"https://twitter.com/{handle}"
            logger.info(f"Extracting user context from: {profile_url}")
            self.page.goto(profile_url, wait_until="networkidle")

            self._random_delay(2, 3)

            # Extract bio
            bio_elem = self.page.query_selector('[data-testid="UserDescription"]')
            bio = bio_elem.inner_text() if bio_elem else ""

            # Extract last 5 tweets
            tweet_elems = self.page.query_selector_all('[data-testid="tweet"]')[:5]
            last_5_tweets = []
            for tweet_elem in tweet_elems:
                tweet_text_elem = tweet_elem.query_selector('[data-testid="tweetText"]')
                if tweet_text_elem:
                    last_5_tweets.append(tweet_text_elem.inner_text())

            context = {
                "bio": bio,
                "last_5_tweets": last_5_tweets
            }

            logger.info(f"Extracted user context: {len(bio)} chars bio, {len(last_5_tweets)} tweets")

            return context

        except Exception as e:
            logger.error(f"Failed to extract user context: {e}")
            return {}

        finally:
            self._stop_browser()
