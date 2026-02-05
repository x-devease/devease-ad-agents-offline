# 🚀 DTC Loss Aversion System - Quick Start

## 📋 What This Does

1. **Discovers** Shopify stores with negative signals
2. **Quantifies** how much money they're losing daily
3. **Generates** diagnostic reports with instant value
4. **Creates** loss aversion outreach (no "book a call")
5. **Builds** Twitter threads for public proofing

---

## ⚡ Quick Run

```bash
cd /Users/anthony/coding/devease-ad-agents-offline/growth/dtc

# Install deps
pip install -r requirements.txt
playwright install chromium

# Set API key (optional - for contact enrichment)
echo "APOLLO_API_KEY=your_key_here" > .env

# Run pipeline
python pipeline.py

# Launch dashboard
streamlit run dashboard.py
```

---

## 📊 Input → Output

**INPUT:**
```python
domains = ["brand1.myshopify.com", "brand2.myshopify.com"]
```

**OUTPUT:**
- `data/tasks.yaml` - Outreach tasks with loss aversion hooks
- `data/reports/*.md` - 20-page diagnostic reports
- `data/twitter_threads.json` - Public proofing threads

---

## 🎯 Example Outreach

### Hook (Triggers Loss Aversion)
```
Not trying to alarm you, but you're leaking $6,000/month on Brand Name 
and the fix takes 20 minutes.
```

### Instant Value (Diagnostic)
```
⚠️ Low SKU + High Ads

Burning cash on 8+ active ads but only 12 products to convert to

💸 Estimated loss: $400/day
🔧 Fix: Expand catalog to 20+ SKUs per ad
```

### CTA (No Meeting Request)
```
Reply 'send it' and I'll email you the 20-page PDF.
Free. No catch. Just fix the bleeding.
```

---

## 📡 Negative Signals Detected

| Signal | Condition | Severity | Est. Loss |
|--------|-----------|----------|-----------|
| Low SKU + High Ads | <15 products, >5 ads | HIGH | $50/day per ad |
| Stale Creative | >10 ads, <3 new products | MEDIUM | $80/day per ad |
| Zero Velocity | Active ads, 0 launches | CRITICAL | $200/day |
| Pricing Mismatch | >$100 avg, <20 SKUs | MEDIUM | $5/day per SKU |

---

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestrator |
| `signals.py` | Negative signal detection |
| `orchestrate_tasks.py` | Loss aversion task generator |
| `diagnostic.py` | Report + Twitter thread generator |
| `dashboard.py` | Human review UI |

---

## 📈 Scoring Formula

```
score = (sku_score × 0.25)          # 0-25 points
      + (velocity_score × 0.30)     # 0-35 points  
      + (ad_score × 0.25)           # 0-30 points
      + (recency_score × 0.20)      # 0-10 points
      + (contact_bonus × 1.0)       # +5 points

Max: 100
```

Higher score = Higher intent lead

---

## 🎭 Psychology Used

### Loss Aversion
- Frame as money "leaking" not "potential savings"
- Use daily loss numbers (feels urgent)
- Show quick fix time ("20 minutes")

### Instant Value
- Full diagnostic attached (not teaser)
- No signup required
- No "book a call" (2026 behavior)

### Social Proof
- "Seen this pattern 50+ times"
- "Brands that fix it first win"

---

## 📝 Dashboard Workflow

1. Run `pipeline.py` to generate tasks
2. Open `dashboard.py` to review
3. Approve/reject each task
4. Execute approved tasks
5. Monitor response rates

---

