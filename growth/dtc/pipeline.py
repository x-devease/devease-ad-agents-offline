"""Main pipeline orchestrator with loss aversion strategy."""

import asyncio
from pathlib import Path
from typing import List
from datetime import datetime

from loguru import logger
from dotenv import load_dotenv

from models import Lead, ShopifyStore, MetaAdData, ContactInfo
from scraper import ShopifyScraper, MetaAdScraper
from enrichment import ApolloClient, TwitterFinder
from scoring import IntentScorer
from orchestrate_tasks import TaskOrchestrator, TwitterOrchestrator

load_dotenv()

# Configure logger
logger.add(
    "logs/pipeline_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)


class LeadGenPipeline:
    """Orchestrate lead generation pipeline with loss aversion."""

    def __init__(
        self,
        domains: List[str] = None,
        proxies: List[str] = None,
        apollo_key: str = None
    ):
        self.domains = domains or []
        self.proxies = proxies or []

        # Initialize components
        self.shopify_scraper = ShopifyScraper(proxies=proxies)
        self.meta_scraper = MetaAdScraper(proxies=proxies)
        self.apollo_client = ApolloClient(api_key=apollo_key)
        self.twitter_finder = TwitterFinder()
        self.scorer = IntentScorer()
        self.task_orchestrator = TaskOrchestrator()
        self.twitter_orchestrator = TwitterOrchestrator()

    async def run(self, top_n: int = 20):
        """Run full pipeline."""

        logger.info("🚀 Starting Loss Aversion Pipeline")
        logger.info(f"Processing {len(self.domains)} domains")

        # Step 1: Scrape Shopify stores
        logger.info("Step 1: Scraping Shopify stores...")
        stores = await self.shopify_scraper.scrape_domains(self.domains)

        if not stores:
            logger.warning("No stores found, stopping pipeline")
            return

        # Step 2: Scrape Meta ads
        logger.info("Step 2: Scraping Meta Ad Library...")
        store_domains = [s.domain for s in stores]
        ad_data_list = await self.meta_scraper.scrape_domains(store_domains)

        # Step 3: Build leads
        logger.info("Step 3: Building lead profiles...")
        leads = []

        for store in stores:
            # Find matching ad data
            meta_data = next(
                (m for m in ad_data_list if m["domain"] == store.domain),
                None
            )

            if not meta_data:
                continue

            meta = MetaAdData(**meta_data)

            lead = Lead(
                domain=store.domain,
                store=store,
                meta_ads=meta
            )

            leads.append(lead)

        logger.success(f"Built {len(leads)} lead profiles")

        # Step 4: Enrich contacts
        logger.info("Step 4: Enriching contacts via Apollo...")
        contacts = await self.apollo_client.enrich_domains(
            [l.domain for l in leads]
        )

        for lead in leads:
            if lead.domain in contacts:
                lead.contact = contacts[lead.domain]

                # Try to find Twitter
                handle = await self.twitter_finder.find_handle(
                    lead.domain,
                    lead.contact
                )
                if handle:
                    lead.contact.twitter_handle = handle

        logger.success(f"Enriched {len(contacts)} leads with contacts")

        # Step 5: Score and rank
        logger.info("Step 5: Scoring and ranking leads...")
        ranked_leads = self.scorer.rank_leads(leads)

        # Step 6: Generate loss aversion tasks
        logger.info(f"Step 6: Generating loss aversion tasks (top {top_n})...")
        tasks = self.task_orchestrator.generate_tasks(ranked_leads, top_n=top_n)

        # Step 7: Generate Twitter threads for public proofing
        logger.info("Step 7: Generating Twitter threads for public proofing...")
        threads = []
        for lead in ranked_leads[:10]:  # Top 10 for Twitter
            thread = self.twitter_orchestrator.generate_thread(lead)
            if thread:
                threads.append({
                    "domain": lead.domain,
                    "thread": thread
                })

        logger.success(f"Generated {len(threads)} Twitter threads")

        # Save tasks
        self.task_orchestrator.save_tasks_yaml(tasks)

        # Save threads
        self._save_threads(threads)

        # Cleanup
        await self.shopify_scraper.close()
        await self.apollo_client.close()

        logger.success(f"✅ Pipeline complete!")
        logger.success(f"   {len(tasks)} loss aversion tasks generated")
        logger.success(f"   {len(threads)} Twitter threads ready")
        logger.info(f"   Total addressable loss: ${sum(t.estimated_loss for t in tasks):,.0f}/mo")
        logger.info("   Run dashboard: streamlit run dashboard.py")

        return ranked_leads, tasks, threads

    def _save_threads(self, threads: List[dict]):
        """Save Twitter threads to file."""
        import json

        output = Path("data/twitter_threads.json")
        output.parent.mkdir(exist_ok=True)

        with open(output, 'w') as f:
            json.dump(threads, f, indent=2)

        logger.info(f"Saved {len(threads)} Twitter threads to {output}")


async def main():
    """Run pipeline with sample data."""

    # Sample domains
    domains = [
        "brand.myshopify.com",
        "store.myshopify.com",
        # Add your domains here
    ]

    pipeline = LeadGenPipeline(domains=domains)

    leads, tasks, threads = await pipeline.run(top_n=10)

    # Print summary
    logger.info("\n🎯 Pipeline Summary")
    logger.info(f"Leads processed: {len(leads)}")
    logger.info(f"Tasks generated: {len(tasks)}")
    logger.info(f"Twitter threads: {len(threads)}")
    logger.info(f"Total addressable loss: ${sum(t.estimated_loss for t in tasks):,.0f}/mo")


if __name__ == "__main__":
    asyncio.run(main())
