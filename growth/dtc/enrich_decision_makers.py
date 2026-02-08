#!/usr/bin/env python3
"""
Module 4: enrich_decision_makers.py - Decision Maker Contact Enrichment

This module uses APIs (Apollo.io, Hunter.io) to find contact information
for key decision makers at target stores.

Input: data/ad_intent.json
Output: data/final_targets.json
"""

import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DecisionMakerEnricher:
    """Enrich store data with decision maker contact information."""

    def __init__(
        self,
        input_path: str = "./data/ad_intent.json",
        store_profiles_path: str = "./data/store_profiles.json",
        output_path: str = "./data/final_targets.json"
    ):
        self.input_path = input_path
        self.store_profiles_path = store_profiles_path
        self.output_path = output_path

        # Load API keys from environment
        self.apollo_api_key = os.getenv('APOLLO_API_KEY')
        self.hunter_api_key = os.getenv('HUNTER_API_KEY')

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def _extract_domain_name(self, myshopify_domain: str) -> str:
        """
        Extract the store name from myshopify.com domain.

        Args:
            myshopify_domain: Full myshopify.com domain

        Returns:
            Store name/brand name
        """
        return myshopify_domain.replace('.myshopify.com', '')

    def _guess_custom_domain(self, store_name: str) -> str:
        """
        Try to guess the custom domain for a Shopify store.

        Args:
            store_name: Store name from myshopify.com

        Returns:
            Guessed custom domain
        """
        # Try .com first
        return f"{store_name}.com"

    def _query_apollo_api(self, domain: str, titles: List[str]) -> Optional[Dict]:
        """
        Query Apollo.io API for people at a domain.

        Args:
            domain: Company domain
            titles: List of job titles to search for

        Returns:
            Person data or None
        """
        if not self.apollo_api_key:
            print("[WARNING] No Apollo API key found")
            return None

        url = "https://api.apollo.io/v1/mixed_people/search"

        headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': self.apollo_api_key
        }

        # Build query for specific titles
        title_queries = []
        for title in titles:
            title_queries.append({
                "type": "and",
                "predicates": [
                    {
                        "type": "title_profile",
                        "value": title
                    }
                ]
            })

        payload = {
            "q_keywords": domain,
            "page": 1,
            "per_page": 10
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('people') and len(data['people']) > 0:
                    # Return first matching person
                    person = data['people'][0]

                    return {
                        'name': person.get('name'),
                        'title': person.get('title'),
                        'email': person.get('email'),
                        'linkedin_url': person.get('linkedin_url'),
                        'twitter_handle': self._extract_twitter_handle(person.get('linkedin_url', '')),
                        'source': 'apollo'
                    }

        except Exception as e:
            print(f"[ERROR] Apollo API error for {domain}: {e}")

        return None

    def _query_hunter_api(self, domain: str) -> Optional[Dict]:
        """
        Query Hunter.io API for email addresses at a domain.

        Args:
            domain: Company domain

        Returns:
            Email data or None
        """
        if not self.hunter_api_key:
            return None

        url = f"https://api.hunter.io/v2/domain-search"
        params = {
            'domain': domain,
            'api_key': self.hunter_api_key,
            'limit': 10
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data.get('data', {}).get('emails'):
                    # Look for decision maker emails
                    emails = data['data']['emails']

                    # Prioritize founder, CEO, CMO, owner
                    priority_keywords = ['founder', 'ceo', 'cmo', 'owner', 'director', 'partner']

                    for email in emails:
                        if any(kw in email.get('value', '').lower() for kw in priority_keywords):
                            return {
                                'email': email.get('value'),
                                'first_name': email.get('first_name'),
                                'last_name': email.get('last_name'),
                                'confidence': email.get('confidence'),
                                'source': 'hunter'
                            }

                    # If no priority emails, return first available
                    if emails:
                        email = emails[0]
                        return {
                            'email': email.get('value'),
                            'first_name': email.get('first_name'),
                            'last_name': email.get('last_name'),
                            'confidence': email.get('confidence'),
                            'source': 'hunter'
                        }

        except Exception as e:
            print(f"[ERROR] Hunter API error for {domain}: {e}")

        return None

    def _extract_twitter_handle(self, linkedin_url: str) -> Optional[str]:
        """
        Attempt to extract or guess Twitter handle from various signals.

        Args:
            linkedin_url: LinkedIn profile URL

        Returns:
            Twitter handle or None
        """
        # This is a placeholder - in real implementation, you might:
        # 1. Scrape the LinkedIn profile for Twitter links
        # 2. Use a Twitter API to search for the company
        # 3. Use a third-party enrichment service

        return None

    def _infer_store_name_brand(self, store_name: str) -> str:
        """
        Convert store name to brand name format.

        Args:
            store_name: Raw store name

        Returns:
            Formatted brand name
        """
        # Remove common prefixes/suffixes
        store_name = store_name.lower()

        # Remove -shop, -store, -official, etc.
        for suffix in ['-shop', '-store', '-official', '-online', '-us', '-uk']:
            store_name = store_name.replace(suffix, '')

        # Capitalize words
        return ' '.join(word.capitalize() for word in store_name.split('-'))

    def enrich_single_target(self, ad_data: Dict, store_profiles: List[Dict]) -> Dict:
        """
        Enrich a single target with contact information.

        Args:
            ad_data: Ad intelligence data
            store_profiles: List of all store profiles for lookup

        Returns:
            Enriched target data
        """
        domain = ad_data['domain']
        store_name = self._extract_domain_name(domain)

        # Find corresponding store profile
        store_profile = None
        for profile in store_profiles:
            if profile['domain'] == domain:
                store_profile = profile
                break

        # Try custom domain
        custom_domain = self._guess_custom_domain(store_name)

        # Titles to search for
        titles = ['Founder', 'CEO', 'Owner', 'CMO', 'Marketing Director', 'E-commerce Manager']

        # Try to find decision maker via Apollo
        person = self._query_apollo_api(custom_domain, titles)

        # Fallback to Hunter for email only
        if not person:
            person = self._query_hunter_api(custom_domain)

        # Build enriched target
        enriched = {
            'domain': domain,
            'store_name': store_name,
            'custom_domain': custom_domain,
            'brand_name': self._infer_store_name_brand(store_name),
            'ad_intelligence': {
                'ad_count': ad_data.get('ad_count', 0),
                'active_ads': ad_data.get('active_ads', False),
                'scaling_indicators': ad_data.get('scaling_indicators', {}),
                'recent_activity': ad_data.get('recent_activity', {})
            },
            'decision_maker': person if person else None,
            'store_metrics': store_profile if store_profile else {},
            'enriched_at': datetime.now().isoformat()
        }

        return enriched

    def load_data(self) -> tuple:
        """
        Load ad intelligence and store profile data.

        Returns:
            Tuple of (ad_intel_list, store_profiles_list)
        """
        # Load ad intelligence
        try:
            with open(self.input_path, 'r') as f:
                ad_intel = json.load(f)
            print(f"Loaded {len(ad_intel)} ad intelligence records")
        except FileNotFoundError:
            print(f"[WARNING] {self.input_path} not found")
            ad_intel = []

        # Load store profiles
        try:
            with open(self.store_profiles_path, 'r') as f:
                store_profiles = json.load(f)
            print(f"Loaded {len(store_profiles)} store profiles")
        except FileNotFoundError:
            print(f"[WARNING] {self.store_profiles_path} not found")
            store_profiles = []

        return ad_intel, store_profiles

    def enrich_targets(self) -> List[Dict]:
        """
        Enrich all targets with contact information.

        Returns:
            List of enriched targets
        """
        ad_intel, store_profiles = self.load_data()

        if not ad_intel:
            print("[ERROR] No ad intelligence data to enrich")
            return []

        print(f"\nEnriching {len(ad_intel)} targets with contact info...")

        enriched_targets = []

        # Use ThreadPoolExecutor for parallel enrichment
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.enrich_single_target, ad_data, store_profiles): ad_data
                for ad_data in ad_intel
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Enriching"):
                try:
                    enriched = future.result()
                    enriched_targets.append(enriched)
                except Exception as e:
                    print(f"\n[ERROR] Failed to enrich target: {e}")

        print(f"\nSuccessfully enriched {len(enriched_targets)} targets")

        return enriched_targets

    def save_targets(self, targets: List[Dict]):
        """
        Save enriched targets to JSON file.

        Args:
            targets: List of enriched targets
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, 'w') as f:
            json.dump(targets, f, indent=2)

        print(f"Saved {len(targets)} enriched targets to {self.output_path}")

    def run(self) -> int:
        """
        Main enrichment workflow.

        Returns:
            Number of successfully enriched targets
        """
        print(f"\n{'='*60}")
        print(f"Decision Maker Enrichment - {datetime.now().isoformat()}")
        print(f"{'='*60}\n")

        if not self.apollo_api_key and not self.hunter_api_key:
            print("[WARNING] No API keys found. Contact enrichment will be limited.")

        targets = self.enrich_targets()

        if targets:
            self.save_targets(targets)

            # Print summary
            print(f"\n{'='*60}")
            print(f"Enrichment Summary")
            print(f"{'='*60}")

            with_contacts = sum(1 for t in targets if t.get('decision_maker'))
            with_ads = sum(1 for t in targets if t.get('ad_intelligence', {}).get('ad_count', 0) > 0)
            scaling = sum(1 for t in targets if t.get('ad_intelligence', {}).get('scaling_indicators', {}).get('is_scaling'))

            print(f"Total targets enriched: {len(targets)}")
            print(f"Targets with contact info: {with_contacts}")
            print(f"Targets with active ads: {with_ads}")
            print(f"Scaling targets: {scaling}")
            print(f"{'='*60}\n")

        return len(targets)


def main():
    parser = argparse.ArgumentParser(
        description="Enrich targets with decision maker contact information"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='./data/ad_intent.json',
        help='Input ad intelligence JSON path (default: ./data/ad_intent.json)'
    )
    parser.add_argument(
        '--store-profiles',
        type=str,
        default='./data/store_profiles.json',
        help='Store profiles JSON path (default: ./data/store_profiles.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/final_targets.json',
        help='Output enriched targets JSON path (default: ./data/final_targets.json)'
    )

    args = parser.parse_args()

    enricher = DecisionMakerEnricher(
        input_path=args.input,
        store_profiles_path=args.store_profiles,
        output_path=args.output
    )

    enricher.run()


if __name__ == "__main__":
    main()
