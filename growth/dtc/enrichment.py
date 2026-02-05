"""Lead enrichment via Apollo.io API."""

import os
import asyncio
from typing import Optional, List
from datetime import datetime

import httpx
from loguru import logger
from dotenv import load_dotenv

from models import ContactInfo

load_dotenv()


class ApolloClient:
    """Apollo.io API client with rate limiting."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("APOLLO_API_KEY")
        if not self.api_key:
            logger.warning("No Apollo API key found")

        self.base_url = "https://api.apollo.io/v1"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": self.api_key or ""
            },
            timeout=30.0
        )

    async def find_founder(
        self,
        domain: str,
        titles: List[str] = None
    ) -> Optional[ContactInfo]:
        """Find founder/CEO contact for domain."""
        if not self.api_key:
            logger.warning("Apollo: No API key, skipping")
            return None

        if titles is None:
            titles = ["Founder", "CEO", "Owner", "CMO"]

        try:
            payload = {
                "q_keywords": domain,
                "page": 1,
                "per_page": 10
            }

            response = await self.client.post(
                "/mixed_people/search",
                json=payload
            )

            if response.status_code != 200:
                logger.warning(f"Apollo: API error {response.status_code}")
                return None

            data = response.json()
            people = data.get("people", [])

            if not people:
                logger.info(f"Apollo: No people found for {domain}")
                return None

            # Find first matching title
            for person in people:
                title = (person.get("title") or "").lower()
                if any(t.lower() in title for t in titles):
                    contact = ContactInfo(
                        name=person.get("name"),
                        email=person.get("email"),
                        title=person.get("title"),
                        linkedin_url=person.get("linkedin_url"),
                        source="apollo"
                    )

                    logger.success(
                        f"Apollo: Found {person.get('title')} for {domain}: "
                        f"{person.get('name')}"
                    )

                    return contact

            # Fallback to first person
            person = people[0]
            contact = ContactInfo(
                name=person.get("name"),
                email=person.get("email"),
                title=person.get("title"),
                linkedin_url=person.get("linkedin_url"),
                source="apollo"
            )

            logger.info(f"Apollo: Using first result for {domain}: {person.get('name')}")
            return contact

        except Exception as e:
            logger.error(f"Apollo: Error for {domain}: {e}")
            return None

        finally:
            await asyncio.sleep(2)  # Rate limit

    async def enrich_domains(
        self,
        domains: List[str],
        titles: List[str] = None
    ) -> dict[str, ContactInfo]:
        """Enrich multiple domains."""
        logger.info(f"Enriching {len(domains)} domains via Apollo...")

        results = {}
        for domain in domains:
            contact = await self.find_founder(domain, titles)
            if contact:
                results[domain] = contact

        logger.success(f"Enriched {len(results)} domains")
        return results

    async def close(self):
        """Close client."""
        await self.client.aclose()


class TwitterFinder:
    """Find Twitter handles from various sources."""

    async def find_handle(self, domain: str, contact: ContactInfo) -> Optional[str]:
        """Try to find Twitter handle."""

        # Method 1: Check LinkedIn URL for Twitter links
        if contact.linkedin_url:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        str(contact.linkedin_url),
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=10.0
                    )

                    if "twitter.com" in response.text:
                        # Simple extraction - you'd want to scrape properly
                        import re
                        twitter_match = re.search(
                            r'twitter\.com/(\w+)',
                            response.text
                        )
                        if twitter_match:
                            handle = twitter_match.group(1)
                            logger.info(f"Found Twitter: @{handle}")
                            return handle

            except Exception as e:
                logger.debug(f"LinkedIn scrape failed: {e}")

        # Method 2: Guess from domain name
        store_name = domain.replace(".myshopify.com", "").replace(".com", "")
        guessed_handle = store_name.replace("-", "")

        logger.debug(f"Guessing Twitter handle: @{guessed_handle}")
        return guessed_handle

    async def verify_handle(self, handle: str) -> bool:
        """Check if Twitter handle exists."""
        try:
            url = f"https://twitter.com/{handle}"
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10.0
                )
                return response.status_code == 200 and "doesn't exist" not in response.text

        except Exception as e:
            logger.debug(f"Twitter verification failed for @{handle}: {e}")
            return False
