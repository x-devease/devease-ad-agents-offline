#!/usr/bin/env python3
import json

with open('data/dtc_brands.json', 'r') as f:
    data = json.load(f)

print(f'Total brands: {len(data)}')

# Check metadata quality
status_count = sum(1 for b in data if b.get('ad_status') and b['ad_status'] != 'Unknown')
platforms_count = sum(1 for b in data if b.get('platforms') and len(b.get('platforms', [])) > 0)
date_count = sum(1 for b in data if b.get('ad_started_date'))

print(f'Ad Status populated: {status_count}/{len(data)} ({100*status_count//len(data)}%)')
print(f'Platforms populated: {platforms_count}/{len(data)} ({100*platforms_count//len(data)}%)')
print(f'Started Date populated: {date_count}/{len(data)} ({100*date_count//len(data)}%)')

# Sample brand
if data:
    sample = data[0]
    print(f'\nSample brand: {sample.get("domain", "unknown")}')
    print(f'  Status: {sample.get("ad_status", "N/A")}')
    print(f'  Platforms: {sample.get("platforms", [])}')
    print(f'  Started: {sample.get("ad_started_date", "N/A")}')
    print(f'  Format: {sample.get("ad_format", "N/A")}')
