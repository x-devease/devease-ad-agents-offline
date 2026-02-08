#!/usr/bin/env python3
import json

with open('data/dtc_brands.json', 'r') as f:
    data = json.load(f)

print('🎯 Sample Discovered Brands (15 diverse examples):\n')
print(f"{'Domain':<30} | {'Page Name':<25} | {'Status':<10} | {'Platforms':<30} | {'Started':<15}")
print(f"{'-'*30}-+-{'-'*25}-+-{'-'*10}-+-{'-'*30}-+-{'-'*15}")

# Get diverse sample
import random
random.seed(42)
sample = random.sample(data, min(15, len(data)))

for b in sample:
    domain = b.get('domain', 'N/A')[:28]
    page_name = b.get('advertiser_page_name', 'N/A')[:23]
    status = b.get('ad_status', 'N/A')[:8]
    platforms = str(b.get('platforms', []))[:28]
    started = b.get('ad_started_date', 'N/A')[:13]
    print(f"{domain:<30} | {page_name:<25} | {status:<10} | {platforms:<30} | {started:<15}")

print(f"\n... and {len(data) - 15} more brands!")
