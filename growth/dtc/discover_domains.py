#!/usr/bin/env python3
"""
Module 1: discover_domains.py - Real-time Shopify Domain Discovery

This module monitors Certificate Transparency (CT) logs via crt.sh
to discover newly registered .myshopify.com domains.

Input: None (monitors public CT logs)
Output: data/new_domains.csv
"""

import csv
import re
import time
import argparse
from datetime import datetime, timedelta
from typing import Set, List
import requests
from urllib.parse import quote

class ShopifyDomainDiscoverer:
    """Discover new Shopify stores via Certificate Transparency logs."""

    def __init__(self, lookback_days: int = 1, output_path: str = "./data/new_domains.csv"):
        self.lookback_days = lookback_days
        self.output_path = output_path
        self.ct_url = "https://crt.sh/"
        self.seen_domains: Set[str] = set()
        self.processed_domains: Set[str] = set()

        # Load previously processed domains to avoid duplicates
        self._load_processed_domains()

    def _load_processed_domains(self):
        """Load list of already processed domains."""
        try:
            with open("./data/processed_domains.txt", "r") as f:
                self.processed_domains = set(line.strip() for line in f)
            print(f"Loaded {len(self.processed_domains)} previously processed domains")
        except FileNotFoundError:
            print("No previous processed domains file found")

    def _save_processed_domains(self):
        """Save processed domains to avoid re-processing."""
        with open("./data/processed_domains.txt", "w") as f:
            for domain in self.processed_domains:
                f.write(f"{domain}\n")

    def _is_valid_shopify_domain(self, domain: str) -> bool:
        """
        Validate that a domain is a Shopify store.

        Returns True if domain matches *.myshopify.com pattern
        """
        pattern = r'^(?:[a-zA-Z0-9-]+)\.myshopify\.com$'
        return re.match(pattern, domain) is not None

    def _extract_store_name(self, domain: str) -> str:
        """Extract the store name from myshopify.com domain."""
        match = re.match(r'^([a-zA-Z0-9-]+)\.myshopify\.com$', domain)
        return match.group(1) if match else domain

    def query_ct_logs(self, retry_count: int = 0) -> List[str]:
        """
        Query crt.sh for newly issued Shopify certificates.

        Args:
            retry_count: Current retry attempt number

        Returns:
            List of unique Shopify domains
        """
        domains = set()

        try:
            # Search for certificates issued in the last N days
            query = '%.myshopify.com'
            params = {
                'q': query,
                'output': 'json',
                'exclude': 'expired',  # Only active certificates
                'deduplicate': 'Y'  # Remove duplicate entries
            }

            print(f"Querying crt.sh for Shopify domains...")
            response = requests.get(
                self.ct_url,
                params=params,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )

            if response.status_code == 200:
                results = response.json()

                for cert in results:
                    name_value = cert.get('name_value', '')
                    # Split by newlines and process each domain
                    for line in name_value.split('\n'):
                        domain = line.strip()
                        if domain and self._is_valid_shopify_domain(domain):
                            # Convert to lowercase for consistency
                            domain = domain.lower()
                            if domain not in self.processed_domains:
                                domains.add(domain)

                print(f"Found {len(domains)} new Shopify domains")
            else:
                print(f"Error querying crt.sh: Status {response.status_code}")
                if retry_count < 3:
                    time.sleep(5)
                    return self.query_ct_logs(retry_count + 1)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if retry_count < 3:
                time.sleep(10)
                return self.query_ct_logs(retry_count + 1)
        except Exception as e:
            print(f"Unexpected error: {e}")

        return list(domains)

    def filter_new_domains(self, domains: List[str]) -> List[str]:
        """
        Filter domains to only include new ones.

        Args:
            domains: List of discovered domains

        Returns:
            Filtered list of truly new domains
        """
        new_domains = []

        for domain in domains:
            if domain not in self.seen_domains and domain not in self.processed_domains:
                new_domains.append(domain)
                self.seen_domains.add(domain)
                self.processed_domains.add(domain)

        return new_domains

    def save_to_csv(self, domains: List[str]):
        """
        Save discovered domains to CSV file.

        Args:
            domains: List of domains to save
        """
        if not domains:
            print("No new domains to save")
            return

        # Ensure data directory exists
        import os
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Append mode to add to existing file
        file_exists = os.path.exists(self.output_path)

        with open(self.output_path, 'a' if file_exists else 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow(['domain', 'store_name', 'discovered_at'])

            # Write domains
            timestamp = datetime.now().isoformat()
            for domain in domains:
                store_name = self._extract_store_name(domain)
                writer.writerow([domain, store_name, timestamp])

        print(f"Saved {len(domains)} domains to {self.output_path}")

        # Also update processed domains file
        self._save_processed_domains()

    def discover(self) -> int:
        """
        Main discovery workflow.

        Returns:
            Number of new domains discovered
        """
        print(f"\n{'='*60}")
        print(f"Shopify Domain Discovery - {datetime.now().isoformat()}")
        print(f"{'='*60}\n")

        # Query CT logs
        all_domains = self.query_ct_logs()

        if not all_domains:
            print("No domains found from CT logs")
            return 0

        # Filter to only new domains
        new_domains = self.filter_new_domains(all_domains)

        if not new_domains:
            print("No new domains found (all previously processed)")
            return 0

        # Save to CSV
        self.save_to_csv(new_domains)

        print(f"\nDiscovery complete! Found {len(new_domains)} new domains")
        print(f"Total seen this session: {len(self.seen_domains)}")

        return len(new_domains)


def main():
    parser = argparse.ArgumentParser(
        description="Discover new Shopify stores via Certificate Transparency logs"
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=1,
        help='Days to look back for new certificates (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/new_domains.csv',
        help='Output CSV file path (default: ./data/new_domains.csv)'
    )

    args = parser.parse_args()

    discoverer = ShopifyDomainDiscoverer(
        lookback_days=args.lookback_days,
        output_path=args.output
    )

    discoverer.discover()


if __name__ == "__main__":
    main()
