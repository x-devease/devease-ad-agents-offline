#!/usr/bin/env python3
"""
Debug script - Find the correct selector for New chat button
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from playwright.sync_api import sync_playwright

print("🔍 Debug: Finding New chat button selector...")
print()

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://localhost:9224")
    context = browser.contexts[0]
    page = context.pages[0] if context.pages else context.new_page()

    # Go to chat page
    print("📍 Going to chat page...")
    page.goto("https://x.com/i/chat", timeout=15000)

    import time
    time.sleep(3)

    print(f"📍 URL after page load: {page.url}")

    # Debug: Show all buttons on the page
    print("\n🔍 All buttons with data-testid on page:")
    all_buttons = page.query_selector_all('button[data-testid]')
    print(f"Found {len(all_buttons)} buttons")
    for i, btn in enumerate(all_buttons[:20], 1):
        try:
            testid = btn.get_attribute("data-testid") or ""
            aria_label = btn.get_attribute("aria-label") or ""
            text = btn.inner_text()[:30]
            print(f"  [{i}] data-testid='{testid}' aria-label='{aria_label}' text='{text}'")
        except:
            pass
    print()

    # Click New chat button
    print("📍 Clicking New chat button...")
    try:
        new_chat_btn = page.query_selector('button[data-testid="dm-new-chat-button"]')
        if new_chat_btn:
            new_chat_btn.click()
            time.sleep(3)
            print(f"✓ New chat clicked, URL: {page.url}")
        else:
            print("✗ New chat button not found")
    except Exception as e:
        print(f"✗ Could not click New chat: {e}")

    # Try clicking the empty conversation new chat button if it exists
    time.sleep(1)
    try:
        empty_new_chat_btn = page.query_selector('button[data-testid="dm-empty-conversation-new-chat-button"]')
        if empty_new_chat_btn:
            print("  ✓ Found dm-empty-conversation-new-chat-button, clicking...")
            empty_new_chat_btn.click(force=True)
            time.sleep(2)
            print("  ✓ Empty conversation new chat clicked")
        else:
            print("  ⚠️  dm-empty-conversation-new-chat-button not found")
    except Exception as e:
        print(f"  ⚠️  Error clicking empty conversation button: {e}")

    # Debug: Show what elements appear after clicking New chat
    print("\n  🔍 Elements immediately after New chat click:")
    try:
        all_after_click = page.query_selector_all('*[data-testid*="dm"], *[data-testid*="chat"], *[data-testid*="composer"], *[data-testid*="search"], *[role="dialog"], *[role="modal"]')
        for i, elem in enumerate(all_after_click[:30], 1):
            try:
                testid = elem.get_attribute("data-testid") or ""
                role = elem.get_attribute("role") or ""
                if testid or role in ["dialog", "modal"]:
                    tag = elem.evaluate("el => el.tagName")
                    print(f"    [{i}] <{tag}> data-testid='{testid}' role='{role}'")
            except:
                pass
    except Exception as e:
        print(f"  Error showing elements: {e}")

    # Check conversation panel
    print("\n  🔍 Checking dm-conversation-panel contents:")
    try:
        conv_panel = page.query_selector('div[data-testid="dm-conversation-panel"]')
        if conv_panel:
            print("    ✓ Found dm-conversation-panel")
            panel_children = conv_panel.query_selector_all('*')
            print(f"    Panel has {len(panel_children)} child elements")

            # Look for inputs within the conversation panel
            print("\n  🔍 Looking for inputs within conversation panel:")
            panel_inputs = conv_panel.query_selector_all('input, div[contenteditable="true"], textarea')
            print(f"    Found {len(panel_inputs)} input elements in panel")
            for i, inp in enumerate(panel_inputs[:10], 1):
                try:
                    inp_tag = inp.evaluate("el => el.tagName")
                    inp_testid = inp.get_attribute("data-testid") or ""
                    inp_placeholder = inp.get_attribute("placeholder") or ""
                    inp_aria = inp.get_attribute("aria-label") or ""
                    print(f"      [{i}] <{inp_tag}> placeholder='{inp_placeholder}' data-testid='{inp_testid}' aria-label='{inp_aria}'")
                except:
                    pass

            # Show first 10 elements in panel
            print("\n  🔍 First 10 elements in panel:")
            for i, child in enumerate(panel_children[:10], 1):
                try:
                    child_testid = child.get_attribute("data-testid") or ""
                    child_role = child.get_attribute("role") or ""
                    child_tag = child.evaluate("el => el.tagName")
                    if child_testid or child_role:
                        print(f"      [{i}] <{child_tag}> data-testid='{child_testid}' role='{child_role}'")
                except:
                    pass
        else:
            print("    ✗ dm-conversation-panel not found")
    except Exception as e:
        print(f"  Error checking panel: {e}")
    print()

    # Search for a test user
    print("🔍 Searching for test user...")
    try:
        # Try multiple selectors for search input
        search_selectors = [
            'div[data-testid="dm-search-bar"] input',
            'input[placeholder*="Search"]',
            'input[aria-label*="Search"]',
            'div[contenteditable="true"][data-testid="dm-search-bar"]',
        ]

        search_found = False
        for i, selector in enumerate(search_selectors, 1):
            print(f"  🔄 Trying search selector {i}/{len(search_selectors)}...")
            search_input = page.query_selector(selector)
            if search_input:
                print(f"  ✓ Found search input with selector {i}")
                search_input.click(force=True)
                time.sleep(0.5)
                search_input.fill("@elonmusk")
                time.sleep(2)
                print(f"  ✓ Filled search with @elonmusk")
                search_found = True
                break

        if search_found:
            # Wait for search results to appear
            time.sleep(2)

            # Look for search results/typeahead
            print("  🔍 Looking for search results...")
            result_selectors = [
                'div[role="option"]',
                'div[data-testid="TypeaheadUser"]',
                'a[href="/elonmusk"]',
            ]

            result_found = False
            for selector in result_selectors:
                results = page.query_selector_all(selector)
                if results:
                    print(f"  ✓ Found {len(results)} results with selector: {selector}")
                    # Try clicking the first result
                    try:
                        results[0].click(force=True)
                        time.sleep(3)
                        print(f"  ✓ Clicked first result")
                        result_found = True
                        break
                    except:
                        pass

            if not result_found:
                # Fallback to Enter
                print("  ⚠️  No results clicked, trying Enter...")
                page.keyboard.press("Enter")
                time.sleep(3)
        else:
            print("  ✗ Search input not found")

    except Exception as e:
        print(f"  ✗ Search failed: {e}")

    time.sleep(2)

    # Show current URL
    print(f"\n📍 Current URL: {page.url}\n")

    # Find all buttons
    print("\n✓ Looking for all clickable elements on the page...")
    print()

    # Try to find any clickable elements
    buttons = page.query_selector_all('div, a, button, svg')
    links = page.query_selector_all('a[href]')

    print(f"Found {len(buttons)} div/a/svg elements")
    print(f"Found {len(links)} links")
    print()

    print("=== LINKS ===")
    for i, link in enumerate(links[:20], 1):  # First 20 links
        try:
            href = link.get_attribute("href") or ""
            text = link.inner_text()[:30]
            if href and ('message' in href.lower() or 'chat' in href.lower() or 'compose' in href.lower()):
                print(f"[{i}] href: '{href}'")
                print(f"    Text: '{text}'")
        except:
            pass

    print()
    print("=== ELEMENTS WITH NEW/CHAT/MESSAGE/COMPOSER/INPUT IN ARIA-LABEL ===")
    all_elements = page.query_selector_all('*[aria-label], *[data-testid], *[contenteditable="true"]')
    for i, elem in enumerate(all_elements[:50], 1):  # First 50
        try:
            aria_label = elem.get_attribute("aria-label") or ""
            data_testid = elem.get_attribute("data-testid") or ""
            contenteditable = elem.get_attribute("contenteditable") or ""

            if aria_label or data_testid or contenteditable == "true":
                if any(keyword in aria_label.lower() or keyword in data_testid.lower()
                       for keyword in ['new', 'chat', 'message', 'compose', 'dm', 'composer', 'input', 'text']):
                    tag = elem.evaluate("el => el.tagName")
                    print(f"[{i}] <{tag}>")
                    if aria_label:
                        print(f"    aria-label: '{aria_label}'")
                    if data_testid:
                        print(f"    data-testid: '{data_testid}'")
                    if contenteditable == "true":
                        print(f"    contenteditable: true")
        except:
            pass

    print()
    print("=== ALL CONTENTEDITABLE ELEMENTS (Message Inputs) ===")
    editable_elements = page.query_selector_all('*[contenteditable="true"]')
    print(f"Found {len(editable_elements)} contenteditable elements")
    for i, elem in enumerate(editable_elements[:20], 1):
        try:
            tag = elem.evaluate("el => el.tagName")
            data_testid = elem.get_attribute("data-testid") or ""
            aria_label = elem.get_attribute("aria-label") or ""
            placeholder = elem.get_attribute("placeholder") or ""
            print(f"\n[{i}] <{tag}>")
            if data_testid:
                print(f"    data-testid: '{data_testid}'")
            if aria_label:
                print(f"    aria-label: '{aria_label}'")
            if placeholder:
                print(f"    placeholder: '{placeholder}'")
        except:
            pass

    browser.close()
