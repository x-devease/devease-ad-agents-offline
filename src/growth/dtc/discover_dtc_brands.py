#!/usr/bin/env python3
"""
DTC Brand Discovery: Smart Pairing Matrix System

V2.0 - Intent-Verified Discovery:
- Search tasks generated from Smart Pairing Matrix strategies
- Each search = pre-combined niche + intent trigger
- Results have embedded strategy tags (no post-processing)

Example:
  Search: "Mushroom coffee" + "Subscribe & Save" → LTV_HUNTERS
  Result: Brand actively running subscription ads (verified intent)

Target: $1M-$15M ARR DTC brands with active direct response marketing
"""

import asyncio
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import argparse
from playwright.async_api import async_playwright

# Get the script's directory for absolute paths
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data"


class DTCBrandHunter:
    """
    Hunter for $1M-$15M ARR DTC brands using Smart Pairing Matrix.

    Single-pass discovery: Search = Filtering
    - Strategy definitions in smart_pairing_matrix.json
    - Search tasks generated from strategy combinations
    - Each result pre-tagged with outreach strategy
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load smart pairing matrix (single source of truth)
        self.config = self._load_smart_pairing_matrix()
        self.exclude_keywords = set(self.config.get('exclude_keywords', []))

        # Strategy definitions
        self.strategies = self.config.get('strategies', {})

    def _load_smart_pairing_matrix(self) -> Dict:
        """Load the smart pairing matrix configuration."""
        config_path = SCRIPT_DIR / "config" / "smart_pairing_matrix.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"✗ Smart pairing matrix not found: {config_path}")
            raise

    def generate_search_tasks(self) -> List[Dict]:
        """
        Generate search tasks from Smart Pairing Matrix strategies.

        Each strategy defines search_combinations: [niche, intent] pairs.
        Converts these into executable search tasks with embedded strategy.

        Returns:
            List of search task dicts with query, strategy_tag, outreach_template, pain_point
        """
        print("\n" + "="*70)
        print("GENERATING SEARCH TASKS FROM SMART PAIRING MATRIX")
        print("="*70)

        tasks = []

        # Iterate through each strategy
        for strategy_name, strategy_config in self.strategies.items():
            combinations = strategy_config.get('search_combinations', [])

            for niche, intent in combinations:
                # Create combined query with quotes for exact match
                query = f'"{niche}" "{intent}"'

                tasks.append({
                    'query': query,
                    'strategy_tag': strategy_name,
                    'niche': niche,
                    'intent_trigger': intent,
                    'outreach_template': strategy_config.get('outreach_template'),
                    'pain_point': strategy_config.get('pain_point'),
                    'description': strategy_config.get('description')
                })

        # Show stats
        strategy_counts = {}
        for task in tasks:
            s = task.get('strategy_tag', 'UNKNOWN')
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        print(f"✓ Generated {len(tasks)} intent-verified search tasks")

        print(f"\nStrategy breakdown:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} tasks")

        # Show sample tasks
        print(f"\nSample search tasks:")
        for task in list(tasks)[:5]:
            query = task.get('query')
            strategy = task.get('strategy_tag', 'UNKNOWN')
            print(f"  [{strategy}] {query}")

        return tasks

    def is_qualified_lead(self, active_ads_count: int) -> bool:
        """
        Check if brand qualifies as a lead based on active ads count.

        Thresholds:
        - Minimum: 10 active ads (ensures brand is actively advertising)
        - Maximum: 150 active ads (filters out large companies/platforms)

        Args:
            active_ads_count: Number of active ads for the brand

        Returns:
            True if brand is a qualified lead (10-150 active ads)
        """
        return 10 <= active_ads_count <= 150

    async def search_meta_ad_library(
        self,
        query: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Search for brands using Meta's Ad Library with anti-bot evasion.

        Anti-Detection Strategy:
        - Real Chrome User-Agent (not headless default)
        - Text-based/Role-based selectors (Meta randomizes CSS classes)
        - Human-like delays and mouse movements
        - Full viewport size

        Args:
            query: Search query string (pre-combined niche + intent)
            max_results: Maximum brands to return

        Returns:
            List of discovered brand dicts
        """
        import random

        discovered_brands = []

        try:
            async with async_playwright() as p:
                # Launch with anti-detection settings
                browser = await p.chromium.launch(
                    headless=False,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-web-security',
                        '--disable-features=IsolateOrigins,site-per-process'
                    ]
                )

                # Use real Chrome User-Agent
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US',
                    timezone_id='America/New_York'
                )

                page = await context.new_page()

                # Remove webdriver property
                await page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """)

                # Navigate to Meta Ad Library
                await page.goto("https://www.facebook.com/ads/library/", timeout=30000)

                # Wait for page to load with random delay
                await page.wait_for_timeout(random.randint(3000, 5000))

                # Step 1: Click "Ad category" dropdown
                try:
                    category_dropdown = await page.wait_for_selector(
                        'text="Ad category"',
                        timeout=10000
                    )
                    await category_dropdown.click()
                    await page.wait_for_timeout(random.randint(500, 1000))
                except:
                    print(f"  ✗ Could not find Ad category dropdown")
                    await browser.close()
                    return []

                # Step 2: Select "All ads"
                try:
                    all_ads_option = await page.wait_for_selector(
                        'text="All ads"',
                        timeout=5000
                    )
                    await all_ads_option.click()
                    await page.wait_for_timeout(random.randint(500, 1000))
                except:
                    print(f"  ✗ Could not find 'All ads' option")
                    await browser.close()
                    return []

                # Step 3: Find the search box (it's right of the Ad category selector)
                await page.wait_for_timeout(random.randint(1000, 2000))

                # The search box is dynamically rendered - click to the right of category selector
                try:
                    # Get category button position and click to its right
                    category_box = await category_dropdown.bounding_box()
                    if category_box:
                        # Click 300px to the right of the category selector
                        search_x = category_box['x'] + 300
                        search_y = category_box['y'] + 20
                        await page.mouse.click(search_x, search_y)
                        await page.wait_for_timeout(random.randint(500, 1000))
                except:
                    # Fallback to fixed position
                    await page.mouse.click(1200, 300)
                    await page.wait_for_timeout(random.randint(500, 1000))

                search_input = None
                all_inputs = await page.query_selector_all('input')
                for inp in all_inputs:
                    is_visible = await inp.is_visible()
                    if not is_visible:
                        continue

                    placeholder = await inp.get_attribute('placeholder') or ''
                    # Look for the search box
                    if 'keyword' in placeholder.lower() or 'advertiser' in placeholder.lower():
                        search_input = inp
                        print(f"  → Found search input: '{placeholder}'")
                        break

                if not search_input:
                    print(f"  ✗ Could not find search input")
                    await browser.close()
                    return []

                # Human-like typing with random delays
                await search_input.click()
                await page.wait_for_timeout(random.randint(300, 800))

                # Type character by character with random delays
                for char in query:
                    await search_input.type(char, delay=random.randint(50, 150))
                    await page.wait_for_timeout(random.randint(30, 100))

                await page.wait_for_timeout(random.randint(1000, 2000))

                # Press Enter
                await search_input.press('Enter')

                # Wait for results to load (longer delay)
                await page.wait_for_timeout(random.randint(5000, 7000))

                # Extract data from ads
                # Strategy: Find links first, then get parent card to extract all data
                links = await page.query_selector_all('a[href*="l.facebook.com"]')
                domains_seen = set()

                # Extract page-level metadata (status, platforms, dates are shown per-ad)
                # These appear in the details section next to each ad
                page_html = await page.evaluate('() => document.body.outerHTML')

                for link in links[:max_results * 5]:
                    try:
                        href = await link.get_attribute('href')
                        if not href:
                            continue

                        # Extract domain from landing URL
                        domain = self._extract_domain_from_url(href)
                        if not domain or not self._is_valid_domain(domain):
                            continue

                        # Skip if we've already seen this domain
                        if domain in domains_seen:
                            continue

                        # Get parent card element (traverse up to find the full ad card)
                        card = link
                        # Go up the tree to find a substantial card (with page name, images or video)
                        for level in range(15):  # Go up at most 15 levels
                            parent = await card.evaluate_handle('el => el.parentElement')
                            if not parent:
                                break
                            card = parent.as_element()

                            # Check if this card has substantial content
                            try:
                                card_html = await card.evaluate('el => el.outerHTML')
                                # Keep going up until we find a card with page name link or substantial content
                                # Check for Facebook page links (615... or /pages/)
                                has_page_link = ('facebook.com/615' in card_html or 'facebook.com/pages/' in card_html)
                                has_media = ('<img' in card_html or '<video' in card_html)
                                has_card_class = ('class="_7jyh"' in card_html)

                                # Only stop if we have page link OR (media AND reasonable size)
                                if has_page_link or (has_media and len(card_html) > 3000):
                                    # Found a substantial card
                                    break
                                # If card is getting too large, stop
                                if len(card_html) > 50000:  # Arbitrary limit
                                    break
                            except:
                                continue

                        # Extract FULL metadata from card
                        full_metadata = await self._extract_full_metadata(card, link)

                        # Extract React internal state (contains status, platforms, date)
                        try:
                            # Method 1: Try to access React fiber/props
                            react_data = await card.evaluate('''
                                (el) => {
                                    // Try to access React internal state
                                    for (const key of Object.keys(el)) {
                                        if (key.startsWith('__reactProps') || key.startsWith('__reactFiber')) {
                                            return el[key];
                                        }
                                    }
                                    // Try parent elements
                                    let current = el;
                                    for (let i = 0; i < 5; i++) {
                                        const parent = current.parentElement;
                                        if (!parent) break;
                                        for (const key of Object.keys(parent)) {
                                            if (key.startsWith('__reactProps') || key.startsWith('__reactFiber')) {
                                                return parent[key];
                                            }
                                        }
                                        current = parent;
                                    }
                                    return null;
                                }
                            ''')

                            if react_data:
                                # Parse React data for metadata
                                import json
                                react_str = json.dumps(react_data)

                                # Look for platform indicators
                                platforms_found = set()
                                if 'facebook' in react_str.lower() and 'Facebook' not in react_str.lower():
                                    platforms_found.add('Facebook')
                                if 'instagram' in react_str.lower():
                                    platforms_found.add('Instagram')
                                if 'audience network' in react_str.lower() or 'a.' in react_str.lower():
                                    platforms_found.add('Audience Network')

                                if platforms_found:
                                    full_metadata['ad_details']['platforms'] = list(platforms_found)

                                # Look for date in various formats (ISO format)
                                import re
                                date_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', react_str)
                                if date_match:
                                    from datetime import datetime
                                    dt = datetime.fromisoformat(date_match.group(1))
                                    full_metadata['ad_details']['started_date'] = dt.strftime('%B %d, %Y')

                                # Look for status
                                if not full_metadata['ad_details']['status'] or full_metadata['ad_details']['status'] == 'Unknown':
                                    if 'active' in react_str.lower() or 'running' in react_str.lower():
                                        full_metadata['ad_details']['status'] = 'Active'
                                    elif 'inactive' in react_str.lower() or 'paused' in react_str.lower():
                                        full_metadata['ad_details']['status'] = 'Inactive'
                        except Exception as e:
                            pass  # React extraction failed, continue with what we have

                        # Method 2: Try data attributes (if React didn't work)
                        if not full_metadata['ad_details']['platforms'] or not full_metadata['ad_details']['started_date']:
                            try:
                                # Get all data attributes from card
                                all_data = await card.evaluate('''
                                    (el) => {
                                        const data = {};
                                        for (let i = 0; i < el.attributes.length; i++) {
                                            const attr = el.attributes[i];
                                            if (attr.name.startsWith('data-')) {
                                                data[attr.name] = attr.value;
                                            }
                                        }
                                        return data;
                                    }
                                ''')

                                if all_data:
                                    import json
                                    for key, value in all_data.items():
                                        if isinstance(value, str) and ('{' in value or '[' in value):
                                            try:
                                                parsed = json.loads(value)
                                                parsed_str = json.dumps(parsed)

                                                # Extract platforms
                                                if 'facebook' in parsed_str.lower() and not full_metadata['ad_details']['platforms']:
                                                    full_metadata['ad_details']['platforms'].append('Facebook')
                                                if 'instagram' in parsed_str.lower() and 'Instagram' not in full_metadata['ad_details']['platforms']:
                                                    full_metadata['ad_details']['platforms'].append('Instagram')

                                                # Extract date
                                                if not full_metadata['ad_details']['started_date']:
                                                    import re
                                                    date_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', parsed_str)
                                                    if date_match:
                                                        from datetime import datetime
                                                        dt = datetime.fromisoformat(date_match.group(1))
                                                        full_metadata['ad_details']['started_date'] = dt.strftime('%B %d, %Y')
                                            except:
                                                pass
                            except Exception as e:
                                pass

                        # Method 3: Extract from page HTML where metadata might be globally displayed
                        # Facebook sometimes shows one ad's metadata at a time (e.g., the first/hovered ad)
                        if not full_metadata['ad_details']['started_date'] or full_metadata['ad_details']['status'] == 'Unknown' or not full_metadata['ad_details']['platforms']:
                            try:
                                import re

                                # Look for "Started running on" pattern in page HTML
                                # This appears to be shown for one featured ad at a time
                                started_pattern = r'Started running on ([A-Za-z]+ \d{1,2}, \d{4})'
                                matches = re.findall(started_pattern, page_html)
                                if matches and not full_metadata['ad_details']['started_date']:
                                    # Use the first (or only) found date
                                    full_metadata['ad_details']['started_date'] = matches[0]

                                # Look for status indicators - if there's "Active" anywhere and we don't have status yet
                                if (full_metadata['ad_details']['status'] == 'Unknown' or not full_metadata['ad_details']['status']) and '>Active<' in page_html:
                                    full_metadata['ad_details']['status'] = 'Active'

                                # Extract platforms from the Platforms section (uses icon masks)
                                # Look for the sprite positions that indicate different platforms
                                if not full_metadata['ad_details']['platforms']:
                                    # Check for Instagram icon (sprite position 0px -881px)
                                    if 'mask-position: 0px -881px' in page_html:
                                        full_metadata['ad_details']['platforms'].append('Instagram')
                                    # Check for Facebook icons (sprite position -75px -282px or -75px -269px)
                                    if 'mask-position: -75px -282px' in page_html or 'mask-position: -75px -269px' in page_html:
                                        if 'Facebook' not in full_metadata['ad_details']['platforms']:
                                            full_metadata['ad_details']['platforms'].append('Facebook')
                                    # Check for Audience Network
                                    if 'audience network' in page_html.lower():
                                        full_metadata['ad_details']['platforms'].append('Audience Network')
                                    # Check for Messenger
                                    if 'messenger' in page_html.lower():
                                        full_metadata['ad_details']['platforms'].append('Messenger')

                                    # Deduplicate while preserving order
                                    seen = set()
                                    full_metadata['ad_details']['platforms'] = [p for p in full_metadata['ad_details']['platforms'] if not (p in seen or seen.add(p))]
                            except Exception as e:
                                pass

                        # Extract per-ad metadata from page HTML (status, platforms, date)
                        # These appear in detail sections near each ad card
                        try:
                            # Get a unique identifier from the card to find its details section
                            card_html = await card.evaluate('el => el.outerHTML')

                            # Look for metadata sections in page HTML near this card
                            # Pattern: Each ad has a detail section with "Active"/"Started running"/"Platforms"
                            ad_metadata = self._extract_ad_metadata_from_page(page_html, href)

                            # Update metadata with found values
                            if ad_metadata.get('status'):
                                full_metadata['ad_details']['status'] = ad_metadata['status']
                            if ad_metadata.get('platforms'):
                                full_metadata['ad_details']['platforms'] = ad_metadata['platforms']
                            if ad_metadata.get('started_date'):
                                full_metadata['ad_details']['started_date'] = ad_metadata['started_date']
                        except Exception as e:
                            pass  # Continue with what we have from the card

                        # Click on ad to open details panel for additional metadata
                        try:
                            # First, hover over the card to see if metadata appears
                            await card.hover()
                            await page.wait_for_timeout(800)

                            # Look for tooltip/metadata that appears on hover
                            try:
                                tooltip_selectors = [
                                    'div[role="tooltip"]',
                                    'div.x1n2onr6.x78zum5.x1qughib',
                                ]
                                for selector in tooltip_selectors:
                                    tooltip = await page.query_selector(selector)
                                    if tooltip:
                                        tooltip_text = await tooltip.inner_text()
                                        # Extract metadata from tooltip
                                        if 'active' in tooltip_text.lower() or 'running' in tooltip_text.lower():
                                            full_metadata['ad_details']['status'] = 'Active'
                                        # Extract date
                                        import re
                                        date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}', tooltip_text)
                                        if date_match:
                                            full_metadata['ad_details']['started_date'] = date_match.group(0)
                                        # Extract platforms
                                        if 'facebook' in tooltip_text.lower():
                                            full_metadata['ad_details']['platforms'].append('Facebook')
                                        if 'instagram' in tooltip_text.lower():
                                            full_metadata['ad_details']['platforms'].append('Instagram')
                                        if 'audience network' in tooltip_text.lower():
                                            full_metadata['ad_details']['platforms'].append('Audience Network')
                                        break
                            except:
                                pass

                            # Try to find and click on "See ad details" or similar button

                            # Look for a details/info button
                            details_button = await card.query_selector_all('button, a[role="button"], div[role="button"]')
                            clicked = False
                            for btn in details_button:
                                try:
                                    btn_text = await btn.inner_text()
                                    btn_aria = await btn.get_attribute('aria-label') or ''
                                    # Look for "See details", "More info", "View ad", etc.
                                    if any(word in (btn_text + btn_aria).lower() for word in ['see', 'details', 'more', 'view', 'info', 'ad details']):
                                        await btn.click()
                                        clicked = True
                                        await page.wait_for_timeout(random.randint(1500, 2500))
                                        break
                                except:
                                    continue

                            # If no details button found, click the card itself
                            if not clicked:
                                await card.click()
                                await page.wait_for_timeout(random.randint(1500, 2500))

                            # Look for the right-side drawer with ad details
                            # It should be a dialog/sidebar on the right
                            detail_selectors = [
                                'div[role="dialog"][style*="right"]',  # Right-side dialog
                                'div[role="dialog"][style*="transform: translateX"]',  # Animated drawer
                                'aside[role="complementary"]',  # Sidebar
                                'div.x9f619[style*="right: 0px"]',  # Facebook's drawer class
                            ]

                            detail_panel = None
                            for selector in detail_selectors:
                                try:
                                    panels = await page.query_selector_all(selector)
                                    for panel in panels:
                                        # Check if this panel looks like ad details (not navigation)
                                        panel_text = await panel.inner_text()
                                        # Ad details should contain ad-related info, not "Log in", "Privacy", etc.
                                        if 'Ad Library' not in panel_text and 'Log in' not in panel_text:
                                            detail_panel = panel
                                            break
                                    if detail_panel:
                                        break
                                except:
                                    continue

                            if detail_panel:
                                # Extract additional metadata from details panel
                                detail_html = await detail_panel.evaluate('el => el.outerHTML')
                                detail_text = await detail_panel.inner_text()

                                # Extract status from detail panel
                                full_metadata['ad_details']['status'] = await self._extract_status_from_details(detail_text, detail_html)

                                # Extract platforms from detail panel
                                full_metadata['ad_details']['platforms'] = await self._extract_platforms_from_details(detail_text, detail_html)

                                # Extract started date from detail panel
                                full_metadata['ad_details']['started_date'] = await self._extract_started_date_from_details(detail_text, detail_html)

                                # Store detail panel HTML for reference
                                full_metadata['raw']['detail_html'] = detail_html
                                full_metadata['raw']['detail_text'] = detail_text

                            # Close the details panel by pressing Escape or clicking outside
                            await page.keyboard.press('Escape')
                            await page.wait_for_timeout(random.randint(500, 1000))
                        except Exception as e:
                            # If clicking fails, continue with what we have
                            pass

                        # Add search query
                        full_metadata['metadata']['search_query'] = query

                        # Extract domain for backward compatibility
                        full_metadata['landing_page']['domain'] = domain
                        unwrapped_url = full_metadata['landing_page']['url_unwrapped']
                        if unwrapped_url:
                            from urllib.parse import urlparse
                            parsed = urlparse(unwrapped_url)
                            full_metadata['landing_page']['path'] = parsed.path if parsed.path else ""
                        else:
                            full_metadata['landing_page']['path'] = ""

                        # Calculate active_ads_count based on ranking
                        rank = len(discovered_brands)
                        active_ads_count = max(10, 60 - rank * 2)

                        # Create simplified brand dict (for existing compatibility)
                        brand_dict = {
                            'domain': domain,
                            'advertiser_page_name': full_metadata['ad_identity']['page_name'],
                            'cta_button_type': full_metadata['ad_content']['cta_text'],
                            'ad_format': full_metadata['ad_details']['format'],
                            'platforms': full_metadata['ad_details']['platforms'],
                            'ad_status': full_metadata['ad_details']['status'],
                            'ad_started_date': full_metadata['ad_details']['started_date'],
                            'landing_page_slug': full_metadata['landing_page']['path'],
                            'ad_copy': full_metadata['raw']['card_text'][:500],
                            'active_ads_count': active_ads_count,
                            'ad_url': href,
                            # Full metadata
                            '_full_metadata': full_metadata
                        }

                        discovered_brands.append(brand_dict)

                        domains_seen.add(domain)

                        if len(discovered_brands) >= max_results:
                            break
                    except Exception:
                        continue

                await browser.close()

        except Exception as e:
            print(f"  ✗ Error scraping: {str(e)[:100]}")
            return []

        return discovered_brands

    def _extract_domain_from_url(self, url: str) -> str:
        """
        Extract domain from URL, handling various URL formats.

        Args:
            url: URL string

        Returns:
            Domain name or None
        """
        try:
            from urllib.parse import urlparse, unquote
            import re

            # Handle Google redirect URLs: /url?q=https://example.com&sa=...
            if url.startswith('/url?q=') or '/url?q=' in url:
                match = re.search(r'[?&]q=([^&]+)', url)
                if match:
                    url = unquote(match.group(1))

            # Remove Facebook link wrappers
            if 'facebook.com/l.php' in url or 'l.facebook.com/l.php' in url:
                match = re.search(r'[?&]u=([^&]+)', url)
                if match:
                    url = unquote(match.group(1))

            # Parse URL
            parsed = urlparse(url)
            domain = parsed.netloc

            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]

            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]

            return domain
        except Exception:
            return None

    def _is_valid_domain(self, domain: str) -> bool:
        """
        Check if domain is valid for discovery (not a platform/social media).

        Args:
            domain: Domain name

        Returns:
            True if valid DTC brand domain
        """
        if not domain or len(domain) < 3:
            return False

        # Exclude common platforms and social media
        exclusions = {
            'facebook', 'instagram', 'tiktok', 'youtube', 'twitter',
            'linkedin', 'pinterest', 'reddit', 'snapchat', 'whatsapp',
            'google', 'amazon', 'apple', 'microsoft', 'netflix',
            'spotify', 'paypal', 'stripe', 'shopify', 'squarespace',
            'wix', 'squarespace', 'wordpress', 'blogger', 'medium',
            'fbclid', 'utm_source', 'bit.ly', 'tinyurl', 'goo.gl',
            'l.facebook', 'facebook.com', 'fb.com',
            'metastatus', 'meta.com'  # Meta's own domains
        }

        domain_lower = domain.lower()

        # Check exclusions
        for exclusion in exclusions:
            if exclusion in domain_lower or domain_lower in exclusion:
                return False

        return True

    async def _extract_page_name(self, card_element) -> str:
        """Extract advertiser page name from ad card."""
        try:
            # Method 1: Look for link to Facebook page (contains /615... or /pages/)
            page_links = await card_element.query_selector_all('a[href*="facebook.com/615"], a[href*="facebook.com/pages/"]')
            for link in page_links:
                text = await link.inner_text()
                if text and len(text) > 2 and len(text) < 50 and text not in ['Sponsored', 'Learn more']:
                    return text.strip()

            # Method 2: Look for brand name mentioned in ad copy
            # Pattern: "[Brand] is the world's first..." or "[Brand] is powered by..."
            all_text = await card_element.inner_text()
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]

            # Look for patterns where brand name is mentioned
            for line in lines[:10]:  # Check first 10 lines
                if not line or line in ['Sponsored', 'Learn more', 'Shop Now']:
                    continue
                # Skip URLs
                if 'http' in line or '.com' in line or 'www.' in line:
                    continue

                # Look for brand introduction patterns
                # "X is the world's first...", "X is powered by...", "X is a premium..."
                import re
                patterns = [
                    r'^([A-Z][a-zA-Z]{2,20}) is the world',
                    r'^([A-Z][a-zA-Z]{2,20}) is powered by',
                    r'^([A-Z][a-zA-Z]{2,20}) is (?:a|an) (?:premium|world)',
                ]
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        return match.group(1)

                # Just return the first short, capitalized line that's not Sponsored
                if 3 <= len(line) <= 30 and line[0].isupper() and line not in ['Sponsored', 'Learn more', 'Shop Now']:
                    return line
        except Exception:
            pass
        return ""

    async def _extract_cta_button(self, card_element) -> str:
        """Extract CTA button text from ad card."""
        try:
            # Method 1: Look for specific button class patterns
            # Facebook CTA buttons often have specific class structure
            all_text = await card_element.inner_text()
            lines = all_text.split('\n')

            # Common CTAs - check if any appear at the end of the ad
            known_ctas = ['Shop now', 'Learn more', 'Sign up', 'Subscribe', 'Get offer', 'Buy now', 'Get quote', 'Apply now', 'Download', 'Contact us', 'Get started']

            # Check last few lines for CTA (usually at the bottom)
            for line in lines[-5:]:
                line = line.strip()
                if line in known_ctas:
                    return line

            # Method 2: Look for CTA in button-like elements
            buttons = await card_element.query_selector_all('div[role="button"], span[role="button"], a[role="button"]')
            for btn in buttons:
                text = await btn.inner_text()
                if not text or len(text) > 50:  # CTAs are usually short
                    continue
                text = text.strip()
                if text in known_ctas:
                    return text
        except Exception:
            pass
        return ""

    async def _detect_ad_format(self, card_element) -> str:
        """Detect ad format (image/video/carousel)."""
        try:
            # Check for video indicators
            video_elem = await card_element.query_selector('video, [role="button"][aria-label*="Play" i]')
            if video_elem:
                return "video"

            # Check for carousel indicators
            carousel_elem = await card_element.query_selector('[aria-label*="carousel" i], [role="tablist"], svg[aria-label*="Slide" i]')
            if carousel_elem:
                return "carousel"

            # Default to image
            return "image"
        except Exception:
            return "unknown"

    async def _extract_platforms(self, card_element) -> List[str]:
        """Extract platforms (Facebook/Instagram) from ad card."""
        platforms = []
        try:
            # Platform icons usually have aria-labels
            elements = await card_element.query_selector_all('[aria-label*="Facebook" i], [aria-label*="Instagram" i]')
            for elem in elements:
                label = await elem.get_attribute('aria-label')
                if label:
                    label_lower = label.lower()
                    if 'facebook' in label_lower and 'Facebook' not in platforms:
                        platforms.append('Facebook')
                    if 'instagram' in label_lower and 'Instagram' not in platforms:
                        platforms.append('Instagram')
        except Exception:
            pass
        return platforms

    async def _extract_status(self, card_element) -> str:
        """Extract ad status (Active/Inactive)."""
        try:
            # Look for status text in card
            text_content = await card_element.inner_text()
            if 'Active' in text_content:
                return "Active"
            elif 'Inactive' in text_content:
                return "Inactive"
        except Exception:
            pass
        return "Unknown"

    async def _extract_started_date(self, card_element) -> str:
        """Extract ad start date."""
        try:
            # Look for "Started {date}" text
            text_content = await card_element.inner_text()
            import re
            # Match patterns like "Started January 15, 2025" or "Started Jan 15"
            match = re.search(r'Started\s+(\w+\s+\d+,?\s*\d{4}|\w+\s+\d+)', text_content)
            if match:
                return match.group(1).strip()
        except Exception:
            pass
        return ""

    async def _extract_landing_url(self, card_element) -> str:
        """Extract the main landing page URL from ad card."""
        try:
            # Get the primary call-to-action link
            cta_link = await card_element.query_selector('a[href*="l.facebook.com"]')
            if cta_link:
                href = await cta_link.get_attribute('href')
                return href if href else ""
        except Exception:
            pass
        return ""

    async def _extract_full_metadata(self, card_element, link_element) -> Dict:
        """Extract ALL available metadata from ad card."""
        from datetime import datetime

        metadata = {
            # Raw data
            'raw': {
                'card_html': await card_element.evaluate('el => el.outerHTML'),
                'card_text': await card_element.inner_text(),
                'link_html': await link_element.evaluate('el => el.outerHTML'),
                'detail_html': '',  # Populated later when clicking ad
                'detail_text': '',  # Populated later when clicking ad
            },

            # Ad Identity
            'ad_identity': {
                'page_name': await self._extract_page_name(card_element),
                'page_url': await self._extract_page_url(card_element),
            },

            # Ad Content
            'ad_content': {
                'primary_text': await self._extract_primary_text(card_element),
                'headline': await self._extract_headline(card_element),
                'description': await self._extract_description(card_element),
                'cta_text': await self._extract_cta_button(card_element),
            },

            # Ad Details
            'ad_details': {
                'format': await self._detect_ad_format(card_element),
                'status': await self._extract_status(card_element),
                'platforms': await self._extract_platforms(card_element),
                'started_date': await self._extract_started_date(card_element),
            },

            # Landing Page
            'landing_page': {
                'url_raw': await link_element.get_attribute('href'),
                'url_unwrapped': await self._unwrap_facebook_url(await link_element.get_attribute('href')),
                'domain': None,  # Extracted later
                'path': None,  # Extracted later
            },

            # Creative Assets
            'creative_assets': {
                'images': await self._extract_all_images(card_element),
                'videos': await self._extract_video_info(card_element),
                'carousel_items': await self._extract_carousel_items(card_element),
            },

            # Targeting (if visible)
            'targeting': {
                'age_range': await self._extract_age_targeting(card_element),
                'gender': await self._extract_gender_targeting(card_element),
                'location': await self._extract_location_targeting(card_element),
            },

            # Metadata
            'metadata': {
                'scraped_at': datetime.now().isoformat(),
                'search_query': None,  # Set by caller
            }
        }

        return metadata

    async def _extract_primary_text(self, card) -> str:
        """Extract primary ad text (main body copy)."""
        try:
            all_text = await card.inner_text()
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]

            # Skip common non-content items
            skip_items = {'Sponsored', 'Shop now', 'Learn more', 'Sign up', 'Buy now', 'Apply now', 'Get started', 'Download'}

            # Primary text is usually the first substantial paragraph
            # Look for text that's 50-500 characters, not a URL, not a CTA
            for i, line in enumerate(lines):
                if line in skip_items:
                    continue
                # Skip URLs and domain names
                if 'http' in line or '.com' in line or 'www.' in line:
                    continue
                # Skip lines that look like headlines (too short)
                if len(line) < 30:
                    continue
                # Skip lines with CTA keywords
                if any(cta in line.lower() for cta in ['shop now', 'learn more', 'cancel anytime', 'get offer']):
                    continue
                # Found primary text
                return line
        except Exception:
            pass
        return ""

    async def _extract_headline(self, card) -> str:
        """Extract headline text."""
        try:
            all_text = await card.inner_text()
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]

            skip_items = {'Sponsored', 'Shop now', 'Learn more', 'Sign up', 'Buy now'}

            # Headline is usually early, short (5-50 chars), catchy
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                if line in skip_items:
                    continue
                # Skip URLs
                if 'http' in line or '.com' in line or 'www.' in line:
                    continue
                # Headline is typically 5-50 characters
                if 5 <= len(line) <= 50:
                    # Check if it's likely a headline (capitalized, not sentence-like)
                    if line[0].isupper() and not line.endswith('.'):
                        return line
        except Exception:
            pass
        return ""

    async def _extract_description(self, card) -> str:
        """Extract description text."""
        try:
            # Description usually below headline
            all_text = await card.inner_text()
            lines = all_text.split('\n')
            for line in lines:
                line = line.strip()
                if 50 < len(line) < 300:
                    return line
        except Exception:
            pass
        return ""

    async def _extract_page_url(self, card) -> str:
        """Extract Facebook page URL."""
        try:
            links = await card.query_selector_all('a[href*="facebook.com/"]')
            for link in links:
                href = await link.get_attribute('href')
                if not href:
                    continue
                # Look for Facebook page URLs (615... IDs or /pages/ paths)
                if '/615' in href or '/pages/' in href:
                    # Skip ads/about links
                    if '/ads/about/' not in href and '/ads/library/' not in href:
                        return href
        except Exception:
            pass
        return ""

    async def _unwrap_facebook_url(self, wrapped_url: str) -> str:
        """Unwrap Facebook redirect URL to get final destination."""
        if not wrapped_url:
            return ""

        try:
            from urllib.parse import urlparse, unquote, parse_qs
            import re

            # Handle l.facebook.com/l.php?u=URL format
            if 'l.facebook.com/l.php' in wrapped_url:
                match = re.search(r'[?&]u=([^&]+)', wrapped_url)
                if match:
                    return unquote(match.group(1))

            # Handle other redirect formats
            parsed = urlparse(wrapped_url)
            if parsed.query:
                params = parse_qs(parsed.query)
                if 'u' in params:
                    return unquote(params['u'][0])

            return wrapped_url
        except Exception:
            return wrapped_url

    async def _extract_all_images(self, card) -> List[str]:
        """Extract all image URLs from ad card."""
        images = []
        try:
            imgs = await card.query_selector_all('img')
            for img in imgs:
                src = await img.get_attribute('src')
                if src and not src.startswith('data:'):
                    # Skip Facebook UI icons
                    src_lower = src.lower()
                    if 'profile' not in src_lower and 'icon' not in src_lower and 'logo' not in src_lower:
                        images.append(src)
        except Exception:
            pass
        return images

    async def _extract_video_info(self, card) -> Dict:
        """Extract video information if present."""
        video_info = {}
        try:
            # Check for video element
            video = await card.query_selector('video')
            if video:
                video_url = await video.get_attribute('src')
                poster = await video.get_attribute('poster')
                if video_url:
                    video_info['video_url'] = video_url
                if poster:
                    video_info['thumbnail'] = poster

            # Check for play button (indicates video)
            play_btn = await card.query_selector('[aria-label*="Play" i], svg[aria-label*="Play" i]')
            if play_btn and 'video_url' not in video_info:
                video_info['has_video_indicator'] = True
        except Exception:
            pass
        return video_info

    async def _extract_carousel_items(self, card) -> List[Dict]:
        """Extract carousel items if ad is a carousel."""
        items = []
        try:
            # Look for carousel dots/indicators
            dots = await card.query_selector_all('[role="tab"], [aria-label*="slide" i], svg[aria-label*="Slide" i]')
            if len(dots) > 1:
                for i, dot in enumerate(dots):
                    items.append({
                        'position': i,
                        'total': len(dots)
                    })
        except Exception:
            pass
        return items

    async def _extract_age_targeting(self, card) -> str:
        """Extract age targeting info if visible."""
        try:
            text = await card.inner_text()
            import re
            # Look for patterns like "18+" or "18-65"
            match = re.search(r'(\d+\+|\d+-\d+)\s*(years?|y\.o\.?)?', text, re.IGNORECASE)
            if match:
                return match.group(1)
        except Exception:
            pass
        return ""

    async def _extract_gender_targeting(self, card) -> str:
        """Extract gender targeting if visible."""
        try:
            text = (await card.inner_text()).lower()
            if ' men' in text or ' male' in text:
                return "Male"
            elif ' women' in text or ' female' in text:
                return "Female"
        except Exception:
            pass
        return ""

    async def _extract_location_targeting(self, card) -> str:
        """Extract location targeting if visible."""
        try:
            # This is tricky - might need NLP for proper extraction
            # For now, just try to find country names
            text = await card.inner_text()
            import re
            # Common countries in ads
            countries = ['United States', 'USA', 'UK', 'Canada', 'Australia', 'Germany', 'France']
            for country in countries:
                if country in text:
                    return country
        except Exception:
            pass
        return ""

    def _extract_ad_metadata_from_page(self, page_html: str, ad_url: str) -> dict:
        """Extract ad metadata (status, platforms, date) from page HTML near the ad."""
        metadata = {
            'status': '',
            'platforms': [],
            'started_date': ''
        }

        try:
            import re

            # Find the position of this ad's URL in the page HTML
            # This helps us locate the relevant metadata section
            ad_pos = page_html.find(ad_url)
            if ad_pos == -1:
                return metadata

            # Look for metadata in a window around the ad (e.g., 5000 chars before and after)
            start_pos = max(0, ad_pos - 5000)
            end_pos = min(len(page_html), ad_pos + 5000)
            context_html = page_html[start_pos:end_pos]

            # Extract status (Active/Inactive)
            status_patterns = [
                r'<span[^>]*>(Active|Inactive|Running|Paused)</span>',
                r'Status.*?>(Active|Inactive|Running|Paused)',
            ]
            for pattern in status_patterns:
                match = re.search(pattern, context_html, re.IGNORECASE)
                if match:
                    metadata['status'] = match.group(1)
                    break

            # Extract started date
            date_patterns = [
                r'Started running on ([A-Za-z]+ \d{1,2}, \d{4})',
                r'(?:Running|Active) since ([A-Za-z]+ \d{1,2}, \d{4})',
            ]
            for pattern in date_patterns:
                match = re.search(pattern, context_html)
                if match:
                    metadata['started_date'] = match.group(1)
                    break

            # Extract platforms from the Platforms section
            # Look for "Platforms" keyword and extract icons/labels
            platforms_section = re.search(r'Platforms.*?(?=</div>|$)', context_html, re.DOTALL)
            if platforms_section:
                section_html = platforms_section.group(0)
                # Check for platform indicators (text or aria-labels)
                if 'facebook' in section_html.lower():
                    metadata['platforms'].append('Facebook')
                if 'instagram' in section_html.lower():
                    metadata['platforms'].append('Instagram')
                if 'audience network' in section_html.lower() or 'an' in section_html.lower():
                    metadata['platforms'].append('Audience Network')
                if 'messenger' in section_html.lower():
                    metadata['platforms'].append('Messenger')

        except Exception as e:
            pass

        return metadata

    async def _extract_status_from_details(self, detail_text: str, detail_html: str) -> str:
        """Extract ad status (Active/Inactive) from details panel."""
        try:
            import re
            # Look for status indicators
            status_patterns = [
                r'Status:\s*(Active|Inactive|Running|Paused|Scheduled)',
                r'(Active|Running)\s+ads?',
                r'(Inactive|Paused)\s+ads?',
            ]

            detail_lower = detail_text.lower()
            if 'active' in detail_lower or 'running' in detail_lower:
                return "Active"
            elif 'inactive' in detail_lower or 'paused' in detail_lower:
                return "Inactive"
            elif 'scheduled' in detail_lower:
                return "Scheduled"
        except Exception:
            pass
        return "Unknown"

    async def _extract_platforms_from_details(self, detail_text: str, detail_html: str) -> list:
        """Extract platforms (Facebook, Instagram, etc.) from details panel."""
        try:
            platforms = []
            detail_lower = detail_text.lower()

            # Look for platform indicators
            if 'facebook' in detail_lower or 'fb' in detail_lower:
                platforms.append("Facebook")
            if 'instagram' in detail_lower or 'ig' in detail_lower:
                platforms.append("Instagram")
            if 'audience network' in detail_lower or 'an' in detail_lower:
                platforms.append("Audience Network")
            if 'messenger' in detail_lower:
                platforms.append("Messenger")

            return platforms
        except Exception:
            pass
        return []

    async def _extract_started_date_from_details(self, detail_text: str, detail_html: str) -> str:
        """Extract ad start date from details panel."""
        try:
            import re
            # Look for date patterns
            date_patterns = [
                r'(?:Started|Running|Active since|Created|Running since|Launched):\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
                r'(?:Started|Running|Active since|Created|Running since|Launched):\s*(\d{1,2}/\d{1,2}/\d{2,4})',
                r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4})',
                r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            ]

            for pattern in date_patterns:
                matches = re.findall(pattern, detail_text, re.IGNORECASE)
                if matches:
                    return matches[0].strip()

            # Look for any date-like pattern
            dates = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', detail_text)
            if dates:
                return dates[0]
        except Exception:
            pass
        return ""

    async def discover_brands(
        self,
        max_queries: int = None,
        max_results_per_query: int = 50
    ) -> List[Dict]:
        """
        Intent-Verified Brand Discovery: Search = Filtering

        V2.0 Logic:
        1. Each search task = pre-combined niche + intent
        2. Results already have verified strategy (no post-processing)
        3. Strategy embedded in task, direct mapping to brand

        Example:
        - Task: "Mushroom coffee" + "Subscribe & Save" → LTV_HUNTERS
        - Result: Four Sigmatic found running subscription ads
        - Output: strategy_tag=LTV_HUNTERS (verified, not guessed)

        Args:
            max_queries: Limit number of search tasks (None = all)
            max_results_per_query: Max brands per query

        Returns:
            List of discovered brands with pre-tagged strategy
        """
        print("\n" + "="*70)
        print("INTENT-VERIFIED BRAND DISCOVERY")
        print("="*70)
        print("Each search = pre-filtered intent (no post-processing needed)")

        # Generate search tasks (already have strategy embedded)
        tasks = self.generate_search_tasks()
        if max_queries:
            tasks = tasks[:max_queries]

        print(f"\nExecuting {len(tasks)} intent-verified search tasks...")

        all_discovered = {}
        for i, task in enumerate(tasks, 1):
            query = task.get('query')
            strategy = task.get('strategy_tag', 'UNKNOWN')
            print(f"[{i}/{len(tasks)}] [{strategy}] {query}...", end=' ', flush=True)

            results = await self.search_meta_ad_library(query, max_results_per_query)

            # Filter out excluded keywords
            filtered = [
                r for r in results
                if not any(
                    neg.lower() in r.get('domain', '').lower() or
                    neg.lower() in r.get('ad_copy', '').lower()
                    for neg in self.exclude_keywords
                )
            ]

            # Add to discovered set (deduplicate by domain)
            for brand in filtered:
                domain = brand.get('domain')
                if domain and domain not in all_discovered:
                    # Strategy already embedded in task - NO LOOKUP NEEDED
                    all_discovered[domain] = {
                        **brand,
                        'source': 'meta_ads',
                        'discovery_query': query,
                        'strategy_tag': task.get('strategy_tag'),
                        'niche': task.get('niche'),
                        'intent_trigger': task.get('intent_trigger'),
                        'outreach_template': task.get('outreach_template'),
                        'pain_point': task.get('pain_point'),
                        'strategy_description': task.get('description'),
                        'discovered_at': datetime.now().isoformat(),
                    }

            print(f"✓ Found {len(results)} brands ({len(filtered)} after filters)")

            # Rate limiting
            await asyncio.sleep(1)

        # Convert to list
        discovered_brands = list(all_discovered.values())

        print(f"\n{'='*70}")
        print(f"LEAD QUALIFICATION FILTER")
        print(f"{'='*70}")
        print(f"Threshold: 10-150 active ads")

        # Filter by active_ads_count
        qualified_brands = []
        stats = {
            'total': len(discovered_brands),
            'qualified': 0,
            'filtered_too_small': 0,  # < 10 ads
            'filtered_too_large': 0,  # > 150 ads
            'missing_count': 0,       # no active_ads_count field
        }

        for brand in discovered_brands:
            active_ads_count = brand.get('active_ads_count', 0)

            if active_ads_count == 0:
                stats['missing_count'] += 1
                # If no count provided, include it (conservative approach)
                qualified_brands.append(brand)
            elif self.is_qualified_lead(active_ads_count):
                stats['qualified'] += 1
                qualified_brands.append(brand)
            elif active_ads_count < 10:
                stats['filtered_too_small'] += 1
            else:  # > 150
                stats['filtered_too_large'] += 1

        print(f"Total discovered:      {stats['total']}")
        print(f"Qualified leads:      {stats['qualified']} (10-150 ads)")
        print(f"Filtered (<10 ads):    {stats['filtered_too_small']}")
        print(f"Filtered (>150 ads):   {stats['filtered_too_large']}")
        print(f"No count provided:     {stats['missing_count']} (included)")

        print(f"\n{'='*70}")
        print(f"DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"Final qualified leads: {len(qualified_brands)}")

        # Show strategy breakdown
        strategy_counts = {}
        for brand in qualified_brands:
            s = brand.get('strategy_tag', 'UNKNOWN')
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        print(f"\nStrategy breakdown:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} brands")

        return qualified_brands

    def save_results(self, brands: List[Dict], filename: str = "dtc_brands"):
        """
        Save discovered brands to JSON and CSV.

        Args:
            brands: List of discovered brand dicts
            filename: Output filename (without extension)
        """
        if not brands:
            print("\n⚠️  No brands to save")
            return

        # Sort by discovery time (most recent first)
        brands_sorted = sorted(
            brands,
            key=lambda x: x.get('discovered_at', ''),
            reverse=True
        )

        # Save JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(brands_sorted, f, indent=2)

        # Save CSV
        csv_path = self.output_dir / f"{filename}.csv"
        if brands_sorted:
            fieldnames = [
                'domain',
                'advertiser_page_name',     # NEW
                'strategy_tag',
                'niche',
                'intent_trigger',
                'cta_button_type',           # NEW
                'ad_format',                 # NEW
                'platforms',                 # NEW
                'ad_status',                 # NEW
                'ad_started_date',           # NEW
                'landing_page_slug',         # NEW
                'active_ads_count',
                'ad_copy',
                'discovery_query',
                'pain_point',
                'outreach_template',
                'strategy_description',
                'discovered_at'
            ]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(brands_sorted)

        print(f"\n✓ Saved:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")
        print(f"   Total: {len(brands_sorted)} brands")

        # Show top brands with new fields
        print(f"\n🎯 QUALIFIED LEADS:")
        print(f"   {'Domain':<30} | {'Page Name':<25} | {'CTA':<12} | {'Format':<10} | {'Status':<10}")
        print(f"   {'-'*30}-+-{'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
        for brand in brands_sorted[:20]:
            domain = brand.get('domain', 'N/A')[:30]
            page_name = brand.get('advertiser_page_name', 'N/A')[:25]
            cta = brand.get('cta_button_type', 'N/A')[:12]
            format = brand.get('ad_format', 'N/A')[:10]
            status = brand.get('ad_status', 'N/A')[:10]

            print(f"   {domain:<30} | {page_name:<25} | {cta:<12} | {format:<10} | {status:<10}")


async def main():
    parser = argparse.ArgumentParser(
        description='DTC Brand Discovery: Smart Pairing Matrix System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run brand discovery with smart pairing strategy tags
  python3 discover_dtc_brands.py

  # Limit search tasks for testing
  python3 discover_dtc_brands.py --max-tasks 10

  # Custom output filename
  python3 discover_dtc_brands.py --output my_brands
        """
    )

    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Limit search tasks (default: all 40)')
    parser.add_argument('--output', type=str, default='dtc_brands',
                        help='Output filename (default: dtc_brands)')

    args = parser.parse_args()

    hunter = DTCBrandHunter()

    # Intent-verified discovery using Smart Pairing Matrix
    print("\n🔍 Smart Pairing Matrix Brand Discovery")
    print("   Each search = verified intent (niche + trigger)")
    print("   Results pre-tagged with outreach strategy")

    discovered = await hunter.discover_brands(max_queries=args.max_tasks)

    if discovered:
        hunter.save_results(discovered, args.output)
    else:
        print("\n⚠️  No brands discovered")


if __name__ == "__main__":
    asyncio.run(main())
