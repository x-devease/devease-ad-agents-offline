#!/usr/bin/env python3
"""
Module 2: profile_shopify.py - Store Profiling & SKU Analysis

This module analyzes Shopify stores via their public JSON API to determine
merchant potential based on SKU count, pricing, and launch velocity.

Input: data/new_domains.csv
Output: data/store_profiles.json
"""

import csv
import json
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class ShopifyProfiler:
    """Profile Shopify stores to assess their potential."""

    def __init__(
        self,
        input_path: str = "./data/new_domains.csv",
        output_path: str = "./data/store_profiles.json",
        max_workers: int = 10,
        request_timeout: int = 10
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def _fetch_products(self, domain: str, retry_count: int = 0) -> Optional[Dict]:
        """
        Fetch products from a Shopify store's public JSON endpoint.

        Args:
            domain: The store's domain
            retry_count: Current retry attempt

        Returns:
            Product data dict or None if failed
        """
        url = f"https://{domain}/products.json"

        try:
            response = self.session.get(
                url,
                params={'limit': 250},
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                # Store not found or doesn't have products
                return None
            elif response.status_code == 429:
                # Rate limited - wait and retry
                if retry_count < 3:
                    time.sleep(5 * (retry_count + 1))
                    return self._fetch_products(domain, retry_count + 1)
                return None
            else:
                return None

        except requests.exceptions.RequestException:
            if retry_count < 2:
                time.sleep(2)
                return self._fetch_products(domain, retry_count + 1)
            return None

    def _calculate_launch_velocity(self, products: List[Dict], days: int = 30) -> int:
        """
        Calculate how many products were launched in the last N days.

        Args:
            products: List of product dicts
            days: Number of days to look back

        Returns:
            Count of recently launched products
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_count = 0

        for product in products:
            created_at = product.get('created_at')
            if created_at:
                try:
                    product_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if product_date > cutoff_date:
                        recent_count += 1
                except (ValueError, AttributeError):
                    continue

        return recent_count

    def _calculate_pricing_metrics(self, products: List[Dict]) -> Dict[str, float]:
        """
        Calculate pricing statistics from product list.

        Args:
            products: List of product dicts

        Returns:
            Dict with min, max, avg prices
        """
        prices = []

        for product in products:
            variants = product.get('variants', [])
            for variant in variants:
                price = variant.get('price')
                if price:
                    try:
                        prices.append(float(price))
                    except (ValueError, TypeError):
                        continue

        if not prices:
            return {'min_price': 0, 'max_price': 0, 'avg_price': 0}

        return {
            'min_price': round(min(prices), 2),
            'max_price': round(max(prices), 2),
            'avg_price': round(sum(prices) / len(prices), 2)
        }

    def profile_store(self, domain: str) -> Optional[Dict]:
        """
        Profile a single Shopify store.

        Args:
            domain: Store domain to profile

        Returns:
            Store profile dict or None if profiling failed
        """
        products_data = self._fetch_products(domain)

        if not products_data or 'products' not in products_data:
            return None

        products = products_data['products']
        sku_count = len(products)

        # Skip stores with very few products
        if sku_count < 5:
            return None

        # Calculate metrics
        launch_velocity_7d = self._calculate_launch_velocity(products, days=7)
        launch_velocity_30d = self._calculate_launch_velocity(products, days=30)
        pricing = self._calculate_pricing_metrics(products)

        # Extract store name from domain
        store_name = domain.replace('.myshopify.com', '')

        return {
            'domain': domain,
            'store_name': store_name,
            'sku_count': sku_count,
            'launch_velocity_7d': launch_velocity_7d,
            'launch_velocity_30d': launch_velocity_30d,
            'pricing': pricing,
            'profiled_at': datetime.now().isoformat(),
            'estimated_annual_revenue': self._estimate_revenue(sku_count, pricing['avg_price'])
        }

    def _estimate_revenue(self, sku_count: int, avg_price: float) -> Dict[str, float]:
        """
        Rough revenue estimation based on SKU count and pricing.

        This is a heuristic - actual revenue varies greatly by niche, marketing, etc.

        Args:
            sku_count: Number of products
            avg_price: Average product price

        Returns:
            Dict with revenue estimates
        """
        # Estimate 2-5 sales per month per SKU on average
        estimated_monthly_sales_low = sku_count * 2 * avg_price
        estimated_monthly_sales_high = sku_count * 5 * avg_price

        return {
            'monthly_low': round(estimated_monthly_sales_low, 2),
            'monthly_high': round(estimated_monthly_sales_high, 2),
            'yearly_low': round(estimated_monthly_sales_low * 12, 2),
            'yearly_high': round(estimated_monthly_sales_high * 12, 2)
        }

    def load_domains(self) -> List[str]:
        """
        Load domains from CSV file.

        Returns:
            List of domains
        """
        domains = []

        try:
            with open(self.input_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    domain = row.get('domain')
                    if domain:
                        domains.append(domain)

            print(f"Loaded {len(domains)} domains from {self.input_path}")
        except FileNotFoundError:
            print(f"Error: Input file {self.input_path} not found")
            return []

        return domains

    def profile_stores(self, min_sku_count: int = 10) -> List[Dict]:
        """
        Profile multiple stores in parallel.

        Args:
            min_sku_count: Minimum SKU count to include in results

        Returns:
            List of store profiles
        """
        domains = self.load_domains()

        if not domains:
            return []

        profiles = []

        print(f"\nProfiling {len(domains)} stores (using {self.max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.profile_store, domain): domain for domain in domains}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Profiling"):
                profile = future.result()
                if profile and profile['sku_count'] >= min_sku_count:
                    profiles.append(profile)

                # Small delay to be respectful
                time.sleep(0.1)

        print(f"\nSuccessfully profiled {len(profiles)} stores")

        return profiles

    def save_profiles(self, profiles: List[Dict]):
        """
        Save store profiles to JSON file.

        Args:
            profiles: List of store profiles
        """
        import os
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, 'w') as f:
            json.dump(profiles, f, indent=2)

        print(f"Saved {len(profiles)} profiles to {self.output_path}")

    def run(self, min_sku_count: int = 10) -> int:
        """
        Main profiling workflow.

        Args:
            min_sku_count: Minimum SKU count threshold

        Returns:
            Number of successfully profiled stores
        """
        print(f"\n{'='*60}")
        print(f"Shopify Store Profiling - {datetime.now().isoformat()}")
        print(f"{'='*60}\n")

        profiles = self.profile_stores(min_sku_count=min_sku_count)

        if profiles:
            self.save_profiles(profiles)

            # Print summary statistics
            print(f"\n{'='*60}")
            print(f"Profiling Summary")
            print(f"{'='*60}")

            total_skus = sum(p['sku_count'] for p in profiles)
            avg_skus = total_skus / len(profiles)
            high_velocity = sum(1 for p in profiles if p['launch_velocity_30d'] > 10)

            print(f"Total stores profiled: {len(profiles)}")
            print(f"Total SKUs across all stores: {total_skus}")
            print(f"Average SKUs per store: {avg_skus:.1f}")
            print(f"High-velocity stores (>10 new products/30d): {high_velocity}")
            print(f"{'='*60}\n")

        return len(profiles)


def main():
    parser = argparse.ArgumentParser(
        description="Profile Shopify stores via public JSON API"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='./data/new_domains.csv',
        help='Input CSV file path (default: ./data/new_domains.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/store_profiles.json',
        help='Output JSON file path (default: ./data/store_profiles.json)'
    )
    parser.add_argument(
        '--min-sku',
        type=int,
        default=10,
        help='Minimum SKU count threshold (default: 10)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )

    args = parser.parse_args()

    profiler = ShopifyProfiler(
        input_path=args.input,
        output_path=args.output,
        max_workers=args.workers
    )

    profiler.run(min_sku_count=args.min_sku)


if __name__ == "__main__":
    main()
