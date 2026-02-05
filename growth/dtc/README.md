# 🎯 Modular Lead Gen System for DTC Brands

A lean, modular system to discover, score, and outreach to high-intent DTC brands. Uses Pydantic for validation, Loguru for logging, and includes a human-in-the-loop dashboard.

---

## 🏗️ Architecture

```
Domains → Shopify Scraper → Meta Ads Monitor → Enrichment → Scorer → Dashboard
                    ↓              ↓                ↓           ↓
              Products.json    Ad Library      Apollo.io    Intent Score
              Launch Velocity  Ad Count        Founder     Priority Rank
```

---

## 📦 Modules

### 1. **models.py** - Pydantic Data Models
- `ShopifyStore` - Store profile with products, velocity, pricing
- `MetaAdData` - Ad intelligence with counts, dates
- `ContactInfo` - Enriched contact from Apollo
- `Lead` - Complete lead profile
- `OutreachTask` - Task ready for execution

### 2. **scraper.py** - Scraping Services
- `ShopifyScraper` - Async scraper for products.json
- `MetaAdScraper` - Stealth Playwright scraper for Ad Library
- Built-in rate limiting (random delays)
- Proxy rotation support

### 3. **enrichment.py** - Contact Discovery
- `ApolloClient` - Find Founder/CEO contacts
- `TwitterFinder` - Discover Twitter handles
- Rate-limited API calls

### 4. **scoring.py** - Intent Scoring
- Multi-factor scoring (SKU, velocity, ads, recency)
- Configurable weights
- Filter by threshold

### 5. **dashboard.py** - Human-in-the-Loop
- Streamlit UI for task review
- Approve/reject tasks
- Execute approved tasks
- Analytics view

### 6. **pipeline.py** - Orchestrator
- Tie everything together
- Run end-to-end pipeline

---

## 🚀 Quick Start

### Install

```bash
cd /Users/anthony/coding/devease-ad-agents-offline/growth/dtc

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt

# Install Playwright
playwright install chromium

# Setup
mkdir -p logs data
cp .env.example .env
```

### Configure

```bash
# Edit .env
APOLLO_API_KEY=your_key_here

# Add proxies (optional)
# export HTTP_PROXY="http://user:pass@host:port"
```

### Run Pipeline

```python
# Edit pipeline.py with your domains
domains = [
    "brand.myshopify.com",
    "another.myshopify.com",
]

# Run
python pipeline.py
```

### Launch Dashboard

```bash
streamlit run dashboard.py
```

---

## 📊 Usage Examples

### Scrape a single store

```python
import asyncio
from scraper import ShopifyScraper

async def main():
    scraper = ShopifyScraper()
    store = await scraper.scrape_store("brand.myshopify.com")
    print(f"Found {store.product_count} products")
    await scraper.close()

asyncio.run(main())
```

### Score a lead

```python
from scoring import IntentScorer
from models import Lead

scorer = IntentScorer()
lead = Lead(domain="brand.com", store=store_data, meta_ads=ad_data)
score = scorer.score_lead(lead)
print(f"Intent score: {score}")
```

### Run with proxies

```python
proxies = [
    "http://user:pass@proxy1.com:8080",
    "http://user:pass@proxy2.com:8080",
]

pipeline = LeadGenPipeline(
    domains=domains,
    proxies=proxies
)
await pipeline.run()
```

---

## ⚙️ Configuration

### Scoring Weights

```python
scorer = IntentScorer(
    weights={
        "sku_count": 0.25,
        "launch_velocity": 0.30,
        "ad_count": 0.25,
        "ad_recency": 0.20
    },
    min_sku=10,
    min_ads=3
)
```

### Rate Limits

```python
scraper = ShopifyScraper(
    proxies=proxies,
    min_delay=2.0,  # seconds
    max_delay=5.0
)
```

---

## 📁 Data Flow

```
Input: domains.txt
  ↓
1. Scrape products.json → store profile
2. Scrape Ad Library → ad intel
3. Enrich Apollo.io → contact info
4. Score → rank by intent
5. Generate tasks → tasks.yaml
  ↓
Dashboard → human review → execute
```

---

## 🔒 Features

- ✅ Pydantic validation on all data
- ✅ Loguru logging with rotation
- ✅ Random delays (rate limiting)
- ✅ Proxy rotation
- ✅ Stealth scraping (Playwright)
- ✅ Async/await throughout
- ✅ Human-in-the-loop
- ✅ No boilerplate - lean code

---

## 📝 Output Files

```
data/tasks.yaml      # Generated tasks
logs/pipeline_*.log # Execution logs
```

---

## 🎯 Scoring Formula

```
score = (sku_score × 0.25)
      + (velocity_score × 0.30)
      + (ad_score × 0.25)
      + (recency_score × 0.20)
      + (contact_bonus × 1.0)
```

Max score: 100

---

## 🚦 Troubleshooting

### No products found
- Store might not be Shopify
- products.json endpoint disabled
- Rate limited - increase delays

### Meta scraper fails
- Login wall detected
- Run with `headless=False`
- Use proxy rotation

### Apollo returns empty
- Check API key
- Domain not in database
- Rate limited

---

## 🔄 Workflow

```
00:00 - Run pipeline on overnight domains
08:00 - Open dashboard over coffee
08:05 - Review pending tasks
08:15 - Approve high-value targets
08:20 - Execute tasks
08:30 - Done
```

---

## 📞 API Keys Needed

- **Apollo.io** - Contact enrichment
- **Proxy** (optional) - Scale operations

---

Built lean. No fluff. 🎯
