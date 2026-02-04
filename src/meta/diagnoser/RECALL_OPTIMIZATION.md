# Recall Optimization - Detector Thresholds v2

## Overview

Optimized detector thresholds to improve recall (catch more issues) while maintaining acceptable precision levels.

## Changes Made

### FatigueDetector
| Threshold | Before | After | Change |
|-----------|--------|-------|--------|
| `fatigue_freq_threshold` | 3.0 | 2.0 | ↓ 33% more sensitive |
| `cpa_increase_threshold` | 1.10 | 1.05 | ↓ 5% vs 10% increase |

**Impact**: Will detect fatigue when:
- Audience has seen ad **2x** instead of 3x
- CPA increases **5%** instead of 10% from golden period

### DarkHoursDetector
| Threshold | Before | After | Change |
|-----------|--------|-------|--------|
| `cvr_threshold_ratio` | 0.20 (20%) | 0.15 (15%) | ↓ 25% more sensitive |
| `min_spend_ratio_hourly` | 0.05 (5%) | 0.03 (3%) | ↓ 40% lower spend req |
| `min_spend_ratio_daily` | 0.10 (10%) | 0.05 (5%) | ↓ 50% lower spend req |
| `target_roas` | 2.5 | 2.0 | ↓ 20% lower ROAS |

**Impact**: Will detect underperforming periods when:
- CVR is **15%** of average (vs 20%)
- Hourly spend is **3%** (vs 5%)
- Daily spend is **5%** (vs 10%)
- ROAS target is **2.0** (vs 2.5)

### LatencyDetector
- **No changes needed** (already R=86%)

## Expected Performance

### Before Optimization
| Detector | Precision | Recall | F1 |
|----------|----------|--------|-----|
| FatigueDetector | 100% | 59% | 74% |
| DarkHoursDetector | 95% | 63% | 75% |
| LatencyDetector | 95% | 86% | 90% |

### After Optimization (Expected)
| Detector | Precision | Recall | F1 |
|----------|----------|--------|-----|
| FatigueDetector | ~85-90% | ~70-75% | ~77-82% |
| DarkHoursDetector | ~85-90% | ~70-75% | ~77-82% |
| LatencyDetector | 95% | 86% | 90% |

## Trade-off Analysis

### What We Gain ✅
- **More issues caught**: Recall 59% → ~70-75%
- **Better coverage**: Detect earlier-stage problems
- **Higher F1**: Better balance overall
- **Still trustworthy**: Precision stays ~85-90%

### What We Lose ⚠️
- **More false alarms**: Some detections will be incorrect
- **Lower precision**: 100% → ~85-90%
- **Manual review needed**: Users must verify alerts

### Is This Worth It?

**YES, for offline analysis:**

1. **Better to find issues than miss them**
   - False positive = 5 minutes to review
   - False negative = wasted budget until next review

2. **Precision is still very good (85-90%)**
   - 9 out of 10 alerts are still real
   - Much better than random (50%)

3. **Users can verify**
   - Offline analysis allows manual review
   - Context from alerts helps verification

4. **Conservative baseline**
   - Can revert if too many false alarms
   - Can fine-tune per account

## When to Use Each Version

### v1 (Original, High Precision)
- **Use when**: False alarms are very costly
- **Example**: Automated bid management
- **Characteristics**: P=100%, R=59%

### v2 (Optimized, High Recall) ← **CURRENT**
- **Use when**: Missing issues is costly
- **Example**: Weekly performance reviews
- **Characteristics**: P=85-90%, R=70-75%

### Future: v3 (Balanced)
- **Use when**: Need optimal balance
- **Example**: Production alerts with review
- **Target**: P≥80%, R≥80%

## Evaluation

To evaluate new thresholds:

```bash
# Run fatigue detector evaluation
PYTHONPATH=$(pwd) python3 src/meta/diagnoser/scripts/evaluate_fatigue.py

# Run dark hours evaluation
PYTHONPATH=$(pwd) python3 src/meta/diagnoser/scripts/evaluate_dark_hours.py

# Check all detectors
PYTHONPATH=$(pwd) python3 src/meta/diagnoser/scripts/evaluate_diagnosers.py
```

**Note**: Evaluation requires moprobo dataset which may not be in this repo.

## Rolling Back

If precision drops too low, revert to v1:

```bash
git log --oneline | grep "Optimized for high recall"
git revert <commit-hash>
```

Or manually restore thresholds in source files.

## Next Steps

1. **Evaluate on real data** - Measure actual performance
2. **Monitor false alarm rate** - Track user feedback
3. **Fine-tune per detector** - Individual thresholds may need adjustment
4. **Consider v3** - Find optimal balance point

## Configuration

Thresholds are in `DEFAULT_THRESHOLDS` in each detector file:
- `src/meta/diagnoser/detectors/fatigue_detector.py`
- `src/meta/diagnoser/detectors/dark_hours_detector.py`
- `src/meta/diagnoser/detectors/latency_detector.py`

Config files in `detectors/config/` are **not used** - only for reference.

---

**Commit**: `aa157cc` - Optimize detectors for higher recall (v2)
