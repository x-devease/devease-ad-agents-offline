#!/usr/bin/env python3
"""
DM Agency Coaches - Create DM drafts for coaches found by find_coaches.py

This script:
1. Loads coaches from find_coaches.py JSON output
2. For each coach, navigates to their profile
3. Opens DM conversation
4. Personalizes the message template
5. Types the message in the DM (DOES NOT SEND)
6. Leaves for human to review and send manually

Safety features:
- NEVER sends any messages - only types drafts
- Shows progress for each coach
- Logs all actions
- Anti-bot delays and human-like behavior

Usage:
    python dm_agency_coaches.py --max-coaches 5
    python dm_agency_coaches.py --coaches-data agency_coaches.json
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(line_buffering=True)


# ============================================================================
# Utility Functions
# ============================================================================

def random_delay(min_seconds: float = 1.0, max_seconds: float = 3.0):
    """Add random delay to simulate human behavior."""
    time.sleep(random.uniform(min_seconds, max_seconds))


def human_like_scroll(page, direction: str = "down", intensity: float = 1.0):
    """Scroll with human-like variation."""
    for _ in range(random.randint(2, 4)):
        distance = random.randint(200, 500) * intensity
        if direction == "up":
            distance = -distance
        page.evaluate(f"window.scrollBy(0, {distance})")
        time.sleep(random.uniform(0.1, 0.4))


def load_coaches(json_path: str) -> List[Dict]:
    """Load coaches from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['coaches']


def load_message_template(template_path: str) -> str:
    """Load DM message template from tasks.yaml."""
    import yaml

    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    # Find the dm_agency_coaches task
    for task in config.get('tasks', []):
        if task.get('id') == 'dm_agency_coaches':
            return task.get('draft', '')

    raise ValueError("dm_agency_coaches task not found in tasks.yaml")


def personalize_message(template: str, coach: Dict) -> str:
    """Personalize message with coach's name and details."""
    name = coach.get('name', coach.get('handle', 'there'))

    # Replace [Name] placeholder
    message = template.replace('[Name]', name)

    return message


def print_dm_preview(coach: Dict, message: str):
    """Print DM preview for human review."""
    print("\n" + "=" * 80)
    print(f"DM PREVIEW: @{coach['handle']}")
    print("=" * 80)
    print(f"\nCoach: {coach['name']}")
    print(f"Handle: @{coach['handle']}")
    print(f"Followers: {coach['follower_count']:,}")
    print(f"Validation: {coach['validation_score']} confidence")
    print(f"\nBio: {coach['bio'][:100]}...")
    print(f"\n📩 MESSAGE TO DRAFT:")
    print("-" * 80)
    print(message)
    print("-" * 80)


# ============================================================================
# DM Drafting Functions
# ============================================================================

def open_dm_conversation(handle: str, page) -> bool:
    """
    Navigate to chat page and start new conversation with coach.

    Args:
        handle: Twitter handle (without @)
        page: Playwright page object

    Returns:
        True if DM conversation opened successfully, False otherwise
    """
    try:
        # 1. Go to chat page
        print("  📍 Going to chat page...")
        page.goto("https://x.com/i/chat", timeout=30000)

        # Wait for page to fully load - wait for DM container
        print("  ⏳ Waiting for page to load...")
        try:
            page.wait_for_selector('div[data-testid="dm-container"]', timeout=10000)
            print("  ✓ Page loaded")
        except:
            print("  ⚠️  Page load timeout, continuing...")

        random_delay(2, 3)
        print("  ✓ On chat page")

        # 2. Find and click "New chat" button
        print("  📍 Looking for New chat button...")

        new_chat_selectors = [
            'button[data-testid="dm-new-chat-button"]',  # Correct selector!
            'div[data-testid="dm-new-chat-button"]',  # Fallback
            'a[href*="/messages/compose"]',  # Compose link
            'div[role="button"]:has-text("New chat")',  # Text-based
        ]

        new_chat_clicked = False
        for i, selector in enumerate(new_chat_selectors, 1):
            try:
                new_chat_btn = page.query_selector(selector)
                if new_chat_btn:
                    new_chat_btn.click(force=True)
                    random_delay(2, 3)
                    new_chat_clicked = True
                    print("  ✓ New chat button clicked")
                    break
            except Exception as e:
                continue

        if not new_chat_clicked:
            print("  ✗ New chat button not found")
            return False

        # 3. Search for the coach's handle
        print(f"  🔍 Searching for @{handle}...")

        search_input_selectors = [
            'div[data-testid="dm-search-bar"] input',  # DM search input
            'input[data-testid="SearchBox_Search_Input"]',
            'input[placeholder*="Search"]',
            'input[aria-label*="Search"]',
            'div[contenteditable="true"][data-testid="dm-search-bar"]',  # Editable div in search
        ]

        search_done = False
        for selector in search_input_selectors:
            try:
                search_input = page.query_selector(selector)
                if search_input:
                    search_input.click(force=True)
                    random_delay(0.5, 1)
                    search_input.fill(f"@{handle}")
                    random_delay(2, 3)
                    search_done = True
                    print(f"  ✓ Searched for @{handle}")
                    break
            except:
                continue

        if not search_done:
            print("  ✗ Search input not found")
            return False

        # 4. Click the coach from search results
        print(f"  📍 Selecting @{handle} from search results...")

        # Wait for search results
        random_delay(2, 3)

        user_result_selectors = [
            f'div[role="option"]:has-text("@{handle}")',
            f'div[data-testid="TypeaheadUser"]:has-text("@{handle}")',
            f'a[href="/{handle}"]',
        ]

        user_clicked = False
        for selector in user_result_selectors:
            try:
                user_elem = page.query_selector(selector)
                if user_elem:
                    user_elem.click(force=True)
                    random_delay(2, 3)
                    user_clicked = True
                    print(f"  ✓ Selected @{handle}")
                    break
            except:
                continue

        if not user_clicked:
            # Try Enter key
            try:
                page.keyboard.press("Enter")
                random_delay(2, 3)
                user_clicked = True
                print(f"  ✓ Selected @{handle} (Enter)")
            except:
                pass

        if not user_clicked:
            print("  ✗ Failed to select user")
            return False

        # Check if in conversation
        final_url = page.url
        print(f"  📍 URL: {final_url}")

        if '/i/chat' in final_url:
            print("  ✓ DM conversation ready")
            return True
        else:
            print("  ⚠️  May not be in DM conversation")
            return True  # Continue anyway, might work

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def draft_dm_message(message: str, page) -> bool:
    """
    Type message in the open DM conversation (DOES NOT SEND).

    Args:
        message: Message text to draft
        page: Playwright page object (already in DM page)

    Returns:
        True if message drafted successfully, False otherwise
    """
    try:
        # Wait for DM input field to load
        print(f"  📍 Current URL: {page.url}")
        random_delay(2, 3)

        # Try multiple selectors for the message input - DM SPECIFIC ONLY
        input_selectors = [
            'div[data-testid="dmComposerTextInput"]',  # Primary DM composer
            'div[contenteditable="true"][data-testid="dmComposerTextInput"]',  # With contenteditable
            'div[data-testid="dmComposer"] div[contenteditable="true"]',  # Inside DM composer
            'div[aria-label*="Message"][contenteditable="true"]',  # DM aria-label
            'div[aria-label*="Direct Message"][contenteditable="true"]',  # DM aria-label alt
            'div[data-testid="conversation"] div[contenteditable="true"]',  # In conversation
            'div[data-testid="DmComposer"] div[contenteditable="true"]',  # Alternative DM composer
            'div[role="textbox"][contenteditable="true"]',  # Generic textbox in DM
            'textarea[data-testid="tweetTextarea"]',  # Sometimes DMs use textarea
        ]

        input_found = False
        for i, selector in enumerate(input_selectors, 1):
            try:
                print(f"  🔄 Trying selector {i}/{len(input_selectors)}: {selector[:60]}...")
                input_elem = page.query_selector(selector)
                if input_elem:
                    print(f"  ✓ Found input element with selector {i}")
                    # Click input to focus
                    input_elem.click(force=True)
                    random_delay(0.5, 1)

                    # Type message with human-like pacing
                    input_elem.fill(message)
                    random_delay(1, 2)
                    input_found = True
                    print(f"  ✓ Message typed successfully")
                    break
            except Exception as e:
                print(f"  ✗ Selector {i} failed: {e}")
                continue

        if not input_found:
            print("  ✗ DM input not found")
            print("  💡 Debug info: Check if chat is fully loaded")
            return False

        # Message typed successfully (DOES NOT SEND)
        print("  ✓✓ Message drafted (NOT sent - leaving for human to review)")
        return True

    except Exception as e:
        print(f"  ✗ Error drafting message: {e}")
        return False


def draft_dm_for_coach(coach: Dict, message: str, page) -> bool:
    """
    Complete workflow to draft DM for a coach (DOES NOT SEND).

    Args:
        coach: Coach data dictionary
        message: Personalized message
        page: Playwright page object

    Returns:
        True if DM drafted successfully, False on error
    """
    handle = coach['handle']

    try:
        # Open DM conversation
        if not open_dm_conversation(handle, page):
            return False

        # Draft message (don't send)
        if not draft_dm_message(message, page):
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error in DM workflow: {e}")
        return False


def log_action(log_file: Path, action: str, coach: Dict, message: str):
    """Log action to file."""
    timestamp = datetime.now().isoformat()

    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "coach_handle": coach['handle'],
        "coach_name": coach['name'],
        "message": message,
    }

    # Append to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create DM drafts for agency coaches (NEVER sends)"
    )

    parser.add_argument(
        "--coaches-data",
        type=str,
        default="src/growth/twitter/data/coaches/agency_coaches.json",
        help="Path to coaches JSON file from find_coaches.py"
    )
    parser.add_argument(
        "--message-template",
        type=str,
        default="config/twitter/tasks.yaml",
        help="Path to tasks.yaml with dm_agency_coaches template"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9223,
        help="Chrome remote debugging port"
    )
    parser.add_argument(
        "--max-coaches",
        type=int,
        default=None,
        help="Maximum number of coaches to draft"
    )
    parser.add_argument(
        "--min-validation",
        type=str,
        choices=["high", "medium", "low"],
        default="medium",
        help="Minimum validation score for coaches"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/twitter/dm_coaches.log",
        help="Log file for actions"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("DM AGENCY COACHES - DRAFT MODE ONLY")
    print("=" * 80)
    print("⚠️  This script will NEVER send messages - only type drafts")
    print(f"Coaches data: {args.coaches_data}")
    print(f"Message template: {args.message_template}")
    print(f"Min validation: {args.min_validation}")
    print()

    # Load coaches
    print("📂 Loading coaches...")
    coaches = load_coaches(args.coaches_data)
    print(f"✓ Loaded {len(coaches)} coaches")

    # Filter by validation score
    validation_order = {"high": 3, "medium": 2, "low": 1}
    min_score = validation_order[args.min_validation]
    coaches = [c for c in coaches if validation_order.get(c.get('validation_score', 'unknown'), 0) >= min_score]
    print(f"✓ Filtered to {len(coaches)} coaches (min validation: {args.min_validation})")

    # Limit coaches if specified
    if args.max_coaches:
        coaches = coaches[:args.max_coaches]
        print(f"✓ Limited to {len(coaches)} coaches")

    print()

    # Load message template
    print("📝 Loading message template...")
    message_template = load_message_template(args.message_template)
    print(f"✓ Template loaded")
    print()

    # Setup log file
    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Connect to browser
    print("🔄 Connecting to browser...")
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://localhost:{args.port}")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()
        print("✓ Connected")
        print()

        # Stats
        drafted = 0
        failed = 0

        try:
            for i, coach in enumerate(coaches, 1):
                print(f"\n[{i}/{len(coaches)}] Processing @{coach['handle']}...")

                # Personalize message
                message = personalize_message(message_template, coach)

                # Show preview
                print_dm_preview(coach, message)

                # Draft the DM
                print("\n  ✅ Creating draft...")
                success = draft_dm_for_coach(coach, message, page)

                if success:
                    drafted += 1
                    log_action(log_file, "drafted", coach, message)
                    print(f"  ✓ DM drafted for @{coach['handle']}")
                else:
                    failed += 1
                    log_action(log_file, "failed", coach, message)
                    print(f"  ✗ Failed to draft DM for @{coach['handle']}")
                    print(f"  ⏭️  Skipping to next coach...")

                # Anti-bot delay between DMs (10-20 seconds)
                if i < len(coaches):
                    delay = random.uniform(10, 20)
                    print(f"\n  ⏸️  Waiting {delay:.1f}s before next DM...")
                    time.sleep(delay)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")

        finally:
            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            print(f"Total coaches: {len(coaches)}")
            print(f"✓ DMs drafted: {drafted}")
            print(f"✗ Failed: {failed}")
            print(f"\n⚠️  All DMs left as drafts - YOU must review and send manually")
            print(f"\nLog file: {log_file}")
            print("=" * 80)
            print()

            browser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(0)
