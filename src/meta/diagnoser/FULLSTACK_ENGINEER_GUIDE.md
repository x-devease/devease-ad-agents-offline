# Diagnoser - Fullstack Engineer Integration Guide

## Overview

The Diagnoser is a **machine learning system** that automatically detects ad performance issues:
- **Creative Fatigue**: Ads shown too many times to same audience
- **Dark Hours**: Time slots with wasted spend
- **Response Latency**: Slow reaction to performance drops

## Current Status: 🟡 Development Preview

### ✅ What's Ready

1. **Core Detection Algorithms** - All 3 detectors implemented and functional
2. **Evaluation Framework** - Can measure precision/recall/F1 on test data
3. **Code Organization** - Clean structure, well-documented
4. **Performance**:
   - ✅ LatencyDetector: P=95%, R=86%, F1=90% (PRODUCTION READY)
   - ❌ FatigueDetector: P=100%, R=59%, F1=74% (needs optimization)
   - ❌ DarkHoursDetector: P=95%, R=63%, F1=75% (needs optimization)

### ❌ What's Missing for Production

1. **Performance Targets Not Met** - 2/3 detectors below 80% recall
2. **No REST API** - Currently Python scripts, need HTTP endpoint
3. **No Production Infrastructure** - Need deployment, monitoring, scaling
4. **No Customer Documentation** - Need user guides, API docs
5. **Limited Error Handling** - Need retries, circuit breakers, graceful failures

---

## Architecture

```
┌─────────────────┐
│   Frontend      │  (React dashboard - TO BE BUILT)
│   Dashboard     │
└────────┬────────┘
         │ HTTP API
         ▼
┌─────────────────┐
│  REST API       │  (FastAPI/Flask - TO BE BUILT)
│  Endpoint       │
│  - /detect      │
│  - /evaluate    │
│  - /optimize    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│       Diagnoser Core            │
├─────────────────────────────────┤
│  detectors/                     │
│  ├── fatigue_detector.py        │  ✅ READY (needs optimization)
│  ├── dark_hours_detector.py     │  ✅ READY (needs optimization)
│  └── latency_detector.py        │  ✅ PRODUCTION READY
├─────────────────────────────────┤
│  evaluator/                     │
│  ├── evaluator.py               │  ✅ READY
│  ├── backtest.py                │  ✅ READY
│  └── metrics.py                 │  ✅ READY
├─────────────────────────────────┤
│  agents/ (optional)             │
│  └── Auto-optimization system   │  🟡 EXPERIMENTAL
└─────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Data Sources   │
│  - Ad Platforms │  (Facebook Ads API, Google Ads)
│  - Database     │  (PostgreSQL - TO BE SETUP)
└─────────────────┘
```

---

## Integration Options

### Option 1: Quick Start (Python Scripts)

**Best for**: Internal testing, proof-of-concept

```python
# Run detection on local data
from src.meta.diagnoser.detectors import FatigueDetector
import pandas as pd

# Load data
data = pd.read_csv('ad_performance.csv')

# Run detector
detector = FatigueDetector()
issues = detector.detect(data, entity_id="ad_123")

# Issues detected
for issue in issues:
    print(f"{issue.severity}: {issue.title}")
    print(f"  {issue.description}")
```

**Pros**: Fast, works today
**Cons**: No API, no scaling, no user interface

---

### Option 2: REST API (FastAPI)

**Best for**: Web dashboard, mobile apps, external integrations

**File: `src/meta/diagnoser/api/main.py`** (TO BE BUILT)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.meta.diagnoser.detectors import FatigueDetector, DarkHoursDetector, LatencyDetector

app = FastAPI(title="Diagnoser API")

class DetectRequest(BaseModel):
    entity_id: str
    detector_type: str  # "fatigue", "dark_hours", "latency"
    data: List[Dict]  # Daily/hourly performance data

class IssueResponse(BaseModel):
    issues: List[Dict]
    metrics: Dict

@app.post("/api/v1/detect", response_model=IssueResponse)
async def detect_issues(request: DetectRequest):
    """Run detection on ad performance data"""
    try:
        detector = get_detector(request.detector_type)
        data = pd.DataFrame(request.data)
        issues = detector.detect(data, request.entity_id)

        return IssueResponse(
            issues=[issue.dict() for issue in issues],
            metrics={"detector": request.detector_type}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}
```

**Deployment** (TO BE SETUP):
```bash
# Build Docker image
docker build -t diagnoser-api .

# Run with docker-compose
docker-compose up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

**Pros**: Standard API, scales well, easy integration
**Cons**: Need to build API layer, setup infrastructure

---

### Option 3: Background Job System

**Best for**: Batch processing, scheduled analysis

**Architecture**:
```python
# Celery task (TO BE BUILT)
from celery import Celery
from src.meta.diagnoser.detectors import all_detectors

app = Celery('diagnoser', broker='redis://localhost')

@app.task
def analyze_ad_account(account_id: str):
    """Run all detectors on account"""
    data = fetch_account_data(account_id)

    all_issues = []
    for detector_class in all_detectors:
        detector = detector_class()
        issues = detector.detect(data, account_id)
        all_issues.extend(issues)

    # Save results to database
    save_issues(account_id, all_issues)

    # Send notification if critical issues found
    if any(i.severity == "CRITICAL" for i in all_issues):
        send_alert(account_id, all_issues)

# Schedule to run daily
from celery.schedules import crontab
app.conf.beat_schedule = {
    'analyze-all-accounts': {
        'task': 'analyze_ad_account',
        'schedule': crontab(hour=2),  # Run at 2 AM
    },
}
```

**Pros**: Async processing, can handle large accounts
**Cons**: Complex infrastructure (Redis, Celery workers)

---

## Data Requirements

### Input Data Format

**For FatigueDetector** (Ad-level daily data):
```python
{
    "date_start": "2024-01-01",
    "spend": 100.0,
    "impressions": 10000,
    "reach": 5000,
    "conversions": 10  # or "actions" JSON
}
```

**For DarkHoursDetector** (AdSet-level hourly + daily):
```python
# Hourly data (last 24h)
{
    "date_start": "2024-01-01 14:00:00",
    "hour": 14,
    "spend": 50.0,
    "purchase_roas": 2.5
}

# Daily data (30 days)
{
    "date_start": "2024-01-01",
    "spend": 500.0,
    "purchase_roas": 2.8
}
```

**For LatencyDetector** (AdSet-level daily):
```python
{
    "date_start": "2024-01-01",
    "spend": 100.0,
    "purchase_roas": 1.5
}
```

### Output Format

All detectors return standardized `Issue` objects:

```python
{
    "id": "fatigue_ad_123",
    "category": "FATIGUE",  # or "PERFORMANCE", "CONFIGURATION"
    "severity": "HIGH",      # LOW, MEDIUM, HIGH, CRITICAL
    "title": "Creative Fatigue: Ad performance declining (Health: 45/100)",
    "description": "Detailed explanation of what's happening...",
    "affected_entities": ["ad_123"],
    "metrics": {
        "health_score": 45.0,
        "fatigue_freq": 3.2,
        "cpa_increase_pct": 35.0,
        "action_recommendation": "Consider pausing or reducing budget"
    }
}
```

---

## Performance Optimization

### Current Issues

1. **FatigueDetector**: Recall = 59% (target: 80%)
   - Problem: Missing fatigue issues
   - Solution: Lower thresholds, but needs validation

2. **DarkHoursDetector**: Recall = 63% (target: 80%)
   - Problem: Missing underperforming time slots
   - Solution: Relax CVR thresholds, needs testing

### Optimization Workflow

```bash
# 1. Run evaluation
cd src/meta/diagnoser/scripts
PYTHONPATH=$(pwd) python3 evaluate_diagnosers.py

# 2. If not satisfied, run optimization
PYTHONPATH=$(pwd) python3 optimize_diagnosers.py

# 3. Re-evaluate
PYTHONPATH=$(pwd) python3 evaluate_diagnosers.py

# 4. Repeat until targets met (P≥70%, R≥80%, F1≥75%)
```

**Note**: Auto-optimization via agents exists but is experimental.

---

## Deployment Checklist

### Phase 1: Development (Current)
- ✅ Core detectors implemented
- ✅ Evaluation framework working
- ✅ Local testing scripts available

### Phase 2: Staging (Next)
- [ ] Build REST API (FastAPI/Flask)
- [ ] Add authentication (JWT/OAuth)
- [ ] Setup database (PostgreSQL)
- [ ] Add request/response validation
- [ ] Implement error handling
- [ ] Add logging (structured JSON)
- [ ] Add metrics (Prometheus)

### Phase 3: Production
- [ ] Optimize detector performance (reach 80% recall)
- [ ] Load testing (1000+ requests/sec)
- [ ] Security audit
- [ ] Customer documentation
- [ ] User dashboard
- [ ] Alerting system
- [ ] Backup/restore procedures

---

## Questions for Fullstack Engineer

1. **API Framework Preference**: FastAPI (recommended) or Flask?
2. **Deployment Target**: AWS/GCP/Azure? Docker/Kubernetes?
3. **Database**: PostgreSQL, MongoDB, or serverless?
4. **Authentication**: JWT, OAuth, or API keys?
5. **Frontend**: React dashboard needed? Or just API?
6. **Data Ingestion**: Real-time streaming or batch processing?
7. **Scale**: How many accounts? How much data per day?
8. **Timeline**: When is production launch needed?

---

## Next Steps

1. **Review This Guide** - Understand current state
2. **Choose Integration Option** - Quick start vs full API
3. **Define Requirements** - Answer questions above
4. **Prototype API** - Build simple FastAPI wrapper
5. **Test Integration** - Run detectors with sample data
6. **Plan Infrastructure** - Design production architecture
7. **Iterate** - Optimize detectors, add features

---

## Contact

For questions about:
- **Detection algorithms**: Review code in `src/meta/diagnoser/detectors/`
- **Evaluation system**: Review `src/meta/diagnoser/evaluator/`
- **Optimization**: See `src/meta/diagnoser/scripts/README.md`

**Status**: Development preview - not production ready yet!
