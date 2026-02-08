#!/usr/bin/env python3
"""
Module 3: fetch_meta_intel.py - Meta Ad Library Intelligence

This module uses Playwright to scrape the Meta (Facebook) Ad Library
to detect merchants with active advertising campaigns.

This is the CORE intelligence gathering module that identifies
merchants in "scaling mode" or "creative anxiety" phase.

Input: data/store_profiles.json
Output: data/ad_intent.json
"""

import json
import asyncio
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from playwright.async_api import async_playwright, Page, Browser
from tqdm import tqdm
import time


class MetaAdLibraryScraper:
    """Scrape Meta Ad Library for advertising intelligence."""

    def __init__(
        self,
        input_path: str = "./data/store_profiles.json",
        output_path: str = "./data/ad_intent.json",
        headless: bool = True,
        timeout: int = 30000
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.headless = headless
        self.timeout = timeout
        self.base_url = "https://www.facebook.com/ads/library/"

    async def _setup_browser(self) -> Browser:
        """
        Setup Playwright browser with stealth configuration.

        Returns:
            Browser instance
        """
        playwright = await async_playwright().start()

        # Launch browser with anti-detection settings
        browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process'
            ]
        )

        return browser

    async def _search_ad_library(
        self,
        page: Page,
        domain: str,
        retry_count: int = 0
    ) -> Optional[Dict]:
        """
        Search Meta Ad Library for ads from a specific domain.

        Args:
            page: Playwright page instance
            domain: Store domain to search for
            retry_count: Current retry attempt

        Returns:
            Ad intelligence dict or None if failed
        """
        search_url = f"{self.base_url}?q={domain}&ad_type=all&active_status=all"

        try:
            # Navigate to ad library
            await page.goto(search_url, wait_until='networkidle', timeout=self.timeout)

            # Wait for page to load
            await asyncio.sleep(3)

            # Check if we hit a login wall or captcha
            if await page.locator('input[type="email"]').count() > 0:
                print(f"[WARNING] Hit login wall for {domain}")
                return None

            # Try to find ad count results
            # Meta changes selectors frequently, so we'll try multiple approaches

            # Method 1: Look for result count text
            selectors_to_try = [
                'xpath://*[contains(@aria-label, "results") or contains(text(), "results")]',
                'xpath://*[contains(text(), "Result")]',
                '[data-pagelet="root"]',
                'div[role="feed"]'
            ]

            ad_data = await self._extract_ad_data(page)

            if ad_data:
                return {
                    'domain': domain,
                    'scraped_at': datetime.now().isoformat(),
                    **ad_data
                }

            # If we couldn't find ads, return minimal data
            return {
                'domain': domain,
                'scraped_at': datetime.now().isoformat(),
                'ad_count': 0,
                'active_ads': False,
                'ads_by_status': {}
            }

        except Exception as e:
            if retry_count < 2:
                await asyncio.sleep(5)
                return await self._search_ad_library(page, domain, retry_count + 1)
            print(f"[ERROR] Failed to scrape {domain}: {e}")
            return None

    async def _extract_ad_data(self, page: Page) -> Optional[Dict]:
        """
        Extract ad data from the loaded page.

        Args:
            page: Playwright page instance

        Returns:
            Dict with ad intelligence
        """
        try:
            # Look for any ad cards or result elements
            ad_cards = await page.locator('div[role="article"], [data-ad], [data-ads]').all()

            if not ad_cards:
                # Try alternate selectors
                ad_cards = await page.locator('div[class*="ad"], div[class*="Ad"]').all()

            ad_count = len(ad_cards)

            if ad_count == 0:
                return {
                    'ad_count': 0,
                    'active_ads': False,
                    'ads_by_status': {}
                }

            # Extract additional data from ads
            active_ads = 0
            inactive_ads = 0
            ad_types = {'image': 0, 'video': 0, 'carousel': 0}

            for card in ad_cards[:20]:  # Limit to first 20 for speed
                try:
                    text = await card.inner_text()

                    # Check for activity indicators
                    if any(keyword in text.lower() for keyword in ['active', 'running', 'now']):
                        active_ads += 1
                    else:
                        inactive_ads += 1

                    # Detect ad types
                    if 'video' in text.lower():
                        ad_types['video'] += 1
                    elif 'carousel' in text.lower():
                        ad_types['carousel'] += 1
                    else:
                        ad_types['image'] += 1

                except:
                    continue

            # Check for recent activity (ads started in last 7 days)
            is_recently_active = await self._check_recent_activity(page)

            return {
                'ad_count': ad_count,
                'active_ads': ad_count > 0,
                'estimated_active_ads': active_ads,
                'ads_by_status': {
                    'active': active_ads,
                    'inactive': inactive_ads
                },
                'ad_types_detected': ad_types,
                'recent_activity': is_recently_active,
                'scaling_indicators': self._calculate_scaling_signals(active_ads, ad_count)
            }

        except Exception as e:
            print(f"[ERROR] Failed to extract ad data: {e}")
            return None

    async def _check_recent_activity(self, page: Page) -> Dict:
        """
        Check if there are ads started in the last 7 days.

        Args:
            page: Playwright page instance

        Returns:
            Dict with recent activity indicators
        """
        try:
            # Look for date-related text
            page_text = await page.inner_text('body')

            # Check for recent time indicators
            recent_keywords = ['day ago', 'days ago', 'week ago', 'running', 'active now']

            has_recent = any(keyword in page_text.lower() for keyword in recent_keywords)

            return {
                'has_recent_ads': has_recent,
                'checked_at': datetime.now().isoformat()
            }

        except:
            return {
                'has_recent_ads': False,
                'checked_at': datetime.now().isoformat()
            }

    def _calculate_scaling_signals(self, active_ads: int, total_ads: int) -> Dict:
        """
        Calculate signals that indicate a merchant is in scaling mode.

        Args:
            active_ads: Number of active ads
            total_ads: Total ad count

        Returns:
            Dict with scaling signals
        """
        # High active ad count suggests scaling
        is_scaling = active_ads >= 5

        # High ratio of active to total ads suggests recent campaign push
        if total_ads > 0:
            active_ratio = active_ads / total_ads
        else:
            active_ratio = 0

        is_aggressive = active_ratio > 0.5 and total_ads >= 3

        return {
            'is_scaling': is_scaling,
            'is_aggressive': is_aggressive,
            'active_ratio': round(active_ratio, 2),
            'confidence': 'high' if is_scaling else 'medium' if active_ads > 0 else 'low'
        }

    def load_store_profiles(self) -> List[Dict]:
        """
        Load store profiles from JSON file.

        Returns:
            List of store profiles
        """
        try:
            with open(self.input_path, 'r') as f:
                profiles = json.load(f)

            print(f"Loaded {len(profiles)} store profiles from {self.input_path}")
            return profiles

        except FileNotFoundError:
            print(f"Error: Input file {self.input_path} not found")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.input_path}")
            return []

    async def scrape_ad_intel(self) -> List[Dict]:
        """
        Scrape ad intelligence for all stores.

        Returns:
            List of ad intelligence results
        """
        profiles = self.load_store_profiles()

        if not profiles:
            return []

        domains = [p['domain'] for p in profiles]

        results = []

        browser = await self._setup_browser()

        try:
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            # Add stealth scripts
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            page = await context.new_page()

            print(f"\nScraping Meta Ad Library for {len(domains)} stores...")

            for domain in tqdm(domains, desc="Scraping ads"):
                try:
                    ad_data = await self._search_ad_library(page, domain)

                    if ad_data:
                        results.append(ad_data)

                    # Random delay to avoid rate limiting
                    await asyncio.sleep(2 + (hash(domain) % 3))

                except Exception as e:
                    print(f"\n[ERROR] Error scraping {domain}: {e}")
                    continue

        finally:
            await browser.close()

        print(f"\nSuccessfully scraped ad data for {len(results)} stores")

        return results

    def save_ad_intel(self, ad_intel: List[Dict]):
        """
        Save ad intelligence to JSON file.

        Args:
            ad_intel: List of ad intelligence results
        """
        import os
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, 'w') as f:
            json.dump(ad_intel, f, indent=2)

        print(f"Saved ad intelligence to {self.output_path}")

    async def run(self) -> int:
        """
        Main scraping workflow.

        Returns:
            Number of successfully scraped stores
        """
        print(f"\n{'='*60}")
        print(f"Meta Ad Library Intelligence - {datetime.now().isoformat()}")
        print(f"{'='*60}\n")

        ad_intel = await self.scrape_ad_intel()

        if ad_intel:
            self.save_ad_intel(ad_intel)

            # Print summary
            print(f"\n{'='*60}")
            print(f"Ad Intelligence Summary")
            print(f"{'='*60}")

            stores_with_ads = sum(1 for a in ad_intel if a.get('ad_count', 0) > 0)
            scaling_stores = sum(1 for a in ad_intel if a.get('scaling_indicators', {}).get('is_scaling'))
            aggressive_stores = sum(1 for a in ad_intel if a.get('scaling_indicators', {}).get('is_aggressive'))

            print(f"Total stores scraped: {len(ad_intel)}")
            print(f"Stores with active ads: {stores_with_ads}")
            print(f"Scaling stores (high ad count): {scaling_stores}")
            print(f"Aggressive stores (recent push): {aggressive_stores}")
            print(f"{'='*60}\n")

        return len(ad_intel)


async def main_async(args):
    scraper = MetaAdLibraryScraper(
        input_path=args.input,
        output_path=args.output,
        headless=not args.headful,
        timeout=args.timeout
    )

    await scraper.run()


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Meta Ad Library for advertising intelligence"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='./data/store_profiles.json',
        help='Input JSON file path (default: ./data/store_profiles.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/ad_intent.json',
        help='Output JSON file path (default: ./data/ad_intent.json)'
    )
    parser.add_argument(
        '--headful',
        action='store_true',
        help='Run with visible browser (default: headless)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30000,
        help='Page load timeout in ms (default: 30000)'
    )

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
