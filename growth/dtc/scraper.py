"""Shopify scraper with rate limiting and proxy rotation."""

import asyncio
import random
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from loguru import logger
from playwright.async_api import async_playwright, Browser, Page

from models import ShopifyStore, ProxyConfig


class RateLimiter:
    """Simple rate limiter with random delays."""

    def __init__(self, min_delay: float = 2.0, max_delay: float = 5.0):
        self.min_delay = min_delay
        self.max_delay = max_delay

    async def wait(self):
        """Wait for random delay."""
        delay = random.uniform(self.min_delay, self.max_delay)
        logger.debug(f"Rate limiter waiting {delay:.2f}s")
        await asyncio.sleep(delay)


class ProxyRotator:
    """Manage proxy rotation."""

    def __init__(self, proxies: List[str] = None):
        self.proxies = proxies or []
        self.current_idx = 0

    def get_proxy(self) -> Optional[str]:
        """Get next proxy in rotation."""
        if not self.proxies:
            return None
        proxy = self.proxies[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.proxies)
        return proxy


class ShopifyScraper:
    """Scrape Shopify stores with products.json."""

    def __init__(
        self,
        proxies: List[str] = None,
        min_delay: float = 2.0,
        max_delay: float = 5.0
    ):
        self.rate_limiter = RateLimiter(min_delay, max_delay)
        self.proxy_rotator = ProxyRotator(proxies)
        self.client = httpx.AsyncClient(timeout=30.0)

    async def is_shopify_store(self, domain: str) -> bool:
        """Check if domain is a Shopify store."""
        try:
            # Check for products.json endpoint
            url = f"https://{domain}/products.json"
            proxy = self.proxy_rotator.get_proxy()

            response = await self.client.get(
                url,
                proxies=proxy,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            is_shopify = response.status_code == 200
            logger.info(f"{domain}: Shopify={is_shopify}")
            return is_shopify

        except Exception as e:
            logger.warning(f"{domain}: Error checking Shopify: {e}")
            return False

        finally:
            await self.rate_limiter.wait()

    async def scrape_store(self, domain: str) -> Optional[ShopifyStore]:
        """Scrape store data from products.json."""
        try:
            url = f"https://{domain}/products.json"
            proxy = self.proxy_rotator.get_proxy()

            response = await self.client.get(
                url,
                proxies=proxy,
                params={"limit": 250},
                headers={"User-Agent": "Mozilla/5.0"}
            )

            if response.status_code != 200:
                logger.warning(f"{domain}: Failed to fetch products.json")
                return None

            data = response.json()
            products = data.get("products", [])

            if not products:
                logger.info(f"{domain}: No products found")
                return ShopifyStore(domain=domain, is_shopify=True, product_count=0)

            # Calculate launch velocity
            now = datetime.now()
            velocity_7d = 0
            velocity_30d = 0
            total_price = 0
            price_count = 0

            for product in products:
                created_str = product.get("created_at")
                if created_str:
                    created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    days_old = (now - created).days

                    if days_old <= 7:
                        velocity_7d += 1
                    if days_old <= 30:
                        velocity_30d += 1

                # Extract pricing
                for variant in product.get("variants", []):
                    price = variant.get("price")
                    if price:
                        try:
                            total_price += float(price)
                            price_count += 1
                        except ValueError:
                            pass

            avg_price = total_price / price_count if price_count > 0 else 0.0

            store = ShopifyStore(
                domain=domain,
                is_shopify=True,
                product_count=len(products),
                products=[{"id": p.get("id"), "title": p.get("title")} for p in products[:50]],
                launch_velocity_7d=velocity_7d,
                launch_velocity_30d=velocity_30d,
                avg_price=round(avg_price, 2)
            )

            logger.success(
                f"{domain}: {len(products)} products, "
                f"{velocity_7d}/{velocity_30d} new (7d/30d), "
                f"${avg_price:.2f} avg"
            )

            return store

        except Exception as e:
            logger.error(f"{domain}: Scraping error: {e}")
            return None

        finally:
            await self.rate_limiter.wait()

    async def scrape_domains(self, domains: List[str]) -> List[ShopifyStore]:
        """Scrape multiple domains concurrently."""
        logger.info(f"Scraping {len(domains)} domains...")

        tasks = [self.scrape_store(d) for d in domains]
        results = await asyncio.gather(*tasks)

        stores = [r for r in results if r is not None]
        logger.success(f"Successfully scraped {len(stores)} stores")

        return stores

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class MetaAdScraper:
    """Scrape Meta Ad Library with stealth."""

    def __init__(
        self,
        proxies: List[str] = None,
        min_delay: float = 3.0,
        max_delay: float = 7.0,
        headless: bool = True
    ):
        self.rate_limiter = RateLimiter(min_delay, max_delay)
        self.proxies = proxies
        self.headless = headless

    async def _create_browser(self) -> Browser:
        """Create stealth browser."""
        playwright = await async_playwright().start()

        browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage"
            ]
        )

        return browser

    async def scrape_ads(self, domain: str) -> Optional[dict]:
        """Scrape ad data for a domain."""
        browser = None
        try:
            browser = await self._create_browser()
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            )

            # Stealth script
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            page = await context.new_page()

            url = f"https://www.facebook.com/ads/library/?q={domain}&ad_type=all"
            await page.goto(url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(3)

            # Check for login wall
            if await page.locator('input[type="email"]').count() > 0:
                logger.warning(f"{domain}: Hit login wall")
                return None

            # Try to extract ad data
            ad_cards = await page.locator('div[role="article"]').all()
            ad_count = len(ad_cards)

            # Check for date indicators in page text
            page_text = await page.inner_text("body")
            has_recent = any(
                kw in page_text.lower()
                for kw in ["active now", "running", "day ago"]
            )

            result = {
                "domain": domain,
                "ad_count": ad_count,
                "active_ads": ad_count > 0,
                "has_recent_activity": has_recent
            }

            logger.success(f"{domain}: {ad_count} ads found, recent={has_recent}")

            return result

        except Exception as e:
            logger.error(f"{domain}: Ad scraping error: {e}")
            return None

        finally:
            if browser:
                await browser.close()
            await self.rate_limiter.wait()

    async def scrape_domains(self, domains: List[str]) -> List[dict]:
        """Scrape ad data for multiple domains."""
        logger.info(f"Scraping Meta ads for {len(domains)} domains...")

        results = []
        for domain in domains:
            result = await self.scrape_ads(domain)
            if result:
                results.append(result)

        logger.success(f"Scraped ads for {len(results)} domains")
        return results
