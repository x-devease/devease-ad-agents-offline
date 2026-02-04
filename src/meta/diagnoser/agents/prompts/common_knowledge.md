# Diagnoser System - Shared Knowledge

**Version**: 2.0.0
**Last Updated**: 2025-02-04
**Purpose**: Shared knowledge referenced by all agents (PM, Coder, Reviewer, Judge, Memory)

This file contains common knowledge to eliminate redundancy across agent prompts. When you see references to this file, the content is loaded at runtime.

---

## Detector Architecture

### FatigueDetector
**Purpose**: Detect creative fatigue via cumulative frequency analysis
**Key Thresholds** (from `{THRESHOLD_SNAPSHOT}`):
- `cpa_increase_threshold`: {THRESHOLD:FatigueDetector.cpa_increase_threshold} (default: 1.10)
- `window_size_days`: {THRESHOLD:FatigueDetector.window_size_days} (default: 23)
- `consecutive_days`: {THRESHOLD:FatigueDetector.consecutive_days} (default: 1)
- `min_golden_days`: {THRESHOLD:FatigueDetector.min_golden_days} (default: 1)

**Rolling Window**: 23 days
**Detection Method**: Golden period identification → Cumulative frequency tracking → Fatigue confirmation
**File**: `src/meta/diagnoser/detectors/fatigue_detector.py`

**Current Performance**: Precision: 100%, Recall: 54.1%, F1: 70.21% (as of 2025-02-04)

### LatencyDetector
**Purpose**: Detect response delays to performance drops
**Key Thresholds** (from `{THRESHOLD_SNAPSHOT}`):
- `roas_threshold`: {THRESHOLD:LatencyDetector.roas_threshold} (default: 1.0)
- `rolling_window_days`: {THRESHOLD:LatencyDetector.rolling_window_days} (default: 3)
- `min_daily_spend`: {THRESHOLD:LatencyDetector.min_daily_spend} (default: 50)
- `min_drop_ratio`: {THRESHOLD:LatencyDetector.min_drop_ratio} (default: 0.2)

**Rolling Window**: 3 days
**Key Metric**: Responsiveness score (0-100, higher=better)
**File**: `src/meta/diagnoser/detectors/latency_detector.py`

**Current Performance**: Precision: 95.00%, Recall: 85.59%, F1: 90.05% (as of 2025-02-04)

### DarkHoursDetector
**Purpose**: Detect underperforming time slots
**Two Modes**:
1. **Hourly Analysis**: Dead hours detection (hour-level granularity)
2. **Weekly Analysis**: Weak days detection (day-of-week granularity)

**Key Thresholds** (from `{THRESHOLD_SNAPSHOT}`):
- `target_roas`: {THRESHOLD:DarkHoursDetector.target_roas} (default: 2.5)
- `cvr_threshold_ratio`: {THRESHOLD:DarkHoursDetector.cvr_threshold_ratio} (default: 0.2)
- `min_spend_ratio_hourly`: {THRESHOLD:DarkHoursDetector.min_spend_ratio_hourly} (default: 0.05)
- `min_spend_ratio_daily`: {THRESHOLD:DarkHoursDetector.min_spend_ratio_daily} (default: 0.1)
- `min_days`: {THRESHOLD:DarkHoursDetector.min_days} (default: 21)

**File**: `src/meta/diagnoser/detectors/dark_hours_detector.py`

**Current Performance**:
- Hourly: Precision: 94.51%, Recall: 62.77%, F1: 75.44% (as of 2025-02-04)

---

## Evaluation System

### Metrics Definition

**Precision** = TP / (TP + FP)
- Measures: How many detected issues are real issues
- Interpretation: Avoids false alarms
- Business Impact: High precision = user trust, less wasted effort

**Recall** = TP / (TP + FN)
- Measures: How many real issues we catch
- Interpretation: Catches all problems
- Business Impact: High recall = catches more wasted spend

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Measures: Harmonic mean of precision and recall
- Interpretation: Balanced performance
- Business Impact: Optimal tradeoff between catching issues and avoiding false alarms

### Validation Method

**Sliding Window Backtest**:
- Window size: 10 windows
- Prevents: Lookahead bias
- Labels: Zero-cost generation from historical data
- Method: Time series cross-validation

### Performance Targets (from `{CONFIG:performance_targets}`)

- **Minimum Precision**: {CONFIG:performance_targets.min_precision} (90%)
- **Minimum Recall**: {CONFIG:performance_targets.min_recall} (60%)
- **Minimum F1-Score**: {CONFIG:performance_targets.min_f1} (70%)
- **Minimum Improvement**: {CONFIG:performance_targets.min_improvement} (3%)

---

## Critical Anti-Patterns

### ❌ Lookahead Bias

**Definition**: Using future data to make past predictions.
**Detection**: Code uses `data[i+1]` or `data[i:window+i+1]` when predicting at index `i`.
**Prevention**: Always use rolling windows `[i-window:i]` (exclusive of current).
**Impact**: Overestimates performance by 2-3x.

**Example of WRONG code**:
```python
# ❌ WRONG - Uses future data
for i in range(len(data)):
    future_window = data[i:i+window_size+1]  # Includes future
    if is_drop(future_window):
        predict_drop_at(i)
```

**Example of CORRECT code**:
```python
# ✅ CORRECT - Only uses past data
for i in range(window_size, len(data)):
    past_window = data[i-window_size:i]  # Only past
    if is_drop(past_window):
        predict_drop_at(i)
```

### ❌ Hardcoded Test Data

**Definition**: Writing code specifically for test cases.
**Detection**: Magic numbers that match test dataset exactly.
**Prevention**: Reviewer validates against production data ranges.
**Impact**: Code fails in production.

**Red Flags**:
- Constants like `if freq == 3.2:` (too specific)
- Comments like `# Matches test case #3`
- Thresholds that don't match production distributions

### ❌ Threshold Inconsistency

**Definition**: Label generator uses different thresholds than detector.
**Detection**: Compare `label_generator.py` thresholds with detector defaults.
**Prevention**: Runtime injection from `{THRESHOLD_SNAPSHOT}`.
**Impact**: Evaluation metrics are unreliable.

**Example of WRONG code**:
```python
# ❌ WRONG - Hardcoded in label_generator.py
if cpa_increase > 1.2:  # Detector uses 1.10!
    label_as_fatigue()
```

**Example of CORRECT code**:
```python
# ✅ CORRECT - Uses detector's actual thresholds
from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector
threshold = FatigueDetector.DEFAULT_THRESHOLDS["cpa_increase_threshold"]
if cpa_increase > threshold:
    label_as_fatigue()
```

---

## Parameter Sensitivity Analysis

### FatigueDetector Sensitivity

| Parameter | Sensitivity | Direction | Precision Impact | Recall Impact |
|-----------|------------|----------|------------------|---------------|
| `cpa_increase_threshold` | **HIGH** | ↓ (lower) | -2% to -5% | +10% to +20% |
| `window_size_days` | MEDIUM | ↑ (larger) | -1% to -3% | +2% to +5% |
| `min_golden_days` | LOW | ↑ (more) | 0% to -1% | +1% to +3% |
| `consecutive_days` | MEDIUM | ↑ (more) | +2% to +5% | -5% to -10% |

**Key Insight**: `cpa_increase_threshold` is the primary tuning knob. Lower it to catch more fatigue (higher recall), but expect some precision loss.

### LatencyDetector Sensitivity

| Parameter | Sensitivity | Direction | Precision Impact | Recall Impact |
|-----------|------------|----------|------------------|---------------|
| `roas_threshold` | **HIGH** | ↓ (lower) | -3% to -8% | +15% to +25% |
| `rolling_window_days` | LOW | ↑ (larger) | -1% to -2% | +1% to +3% |
| `min_drop_ratio` | MEDIUM | ↓ (lower) | -2% to -4% | +5% to +10% |

**Key Insight**: `roas_threshold` is most sensitive. Lower it to catch more drops, but false alarms increase.

### DarkHoursDetector Sensitivity

| Parameter | Sensitivity | Direction | Precision Impact | Recall Impact |
|-----------|------------|----------|------------------|---------------|
| `target_roas` | **HIGH** | ↓ (lower) | -5% to -10% | +10% to +20% |
| `cvr_threshold_ratio` | MEDIUM | ↓ (lower) | -2% to -5% | +3% to +8% |

**Key Insight**: `target_roas` is the main control. Set based on business requirements (breakeven ROAS).

---

## File Locations

All paths are configurable via `{CONFIG:paths}`:

- **Detectors**: `{CONFIG:paths.detectors}`
- **Scripts**: `{CONFIG:paths.scripts}`
- **Evaluation Reports**: `{CONFIG:paths.reports}`
- **Memory Storage**: `{CONFIG:paths.memory_storage}`
- **Prompts**: `{CONFIG:paths.prompts}`
- **Schemas**: `{CONFIG:paths.schemas}`

---

## Business Context

**Currency**: {CONFIG:business.currency} (USD)
**Average Monthly Spend per Ad**: {CONFIG:business.avg_monthly_spend_per_ad}
**Daily Waste per Fatigued Ad**: {CONFIG:business.daily_waste_per_fatigue_ad}
**Loss per False Positive**: {CONFIG:business.loss_per_false_positive}

**Business Impact Formula**:
```
Waste = daily_waste_per_fatigue_ad × days_fatigued
Savings = Waste - (loss_per_false_positive × false_positives)
```

---

## Decision Frameworks

### When to Optimize for Recall

**Optimize for recall (catch more issues) when**:
- Detector has precision > 95% (headroom for improvement)
- Business cost of missing issues > cost of false alarms
- Example: FatigueDetector (P=100%, R=54%) → Lower threshold to improve recall

### When to Optimize for Precision

**Optimize for precision (avoid false alarms) when**:
- Detector has precision < 90% (too many false alarms)
- Business cost of false alarms > cost of missing issues
- User trust is low due to false positives

### When to Accept Current Performance

**Accept current performance when**:
- Precision ≥ 90% AND recall ≥ 60%
- Improvement potential < 3% (diminishing returns)
- Focus should shift to other detectors

---

## Error Handling

### Agent Failure Escalation

1. **PM Agent fails** → Use last known good spec (if < 7 days old), else skip iteration
2. **Coder Agent fails** → Skip iteration, log error, do not modify code
3. **Reviewer Agent fails** → Default to REJECT (conservative), log error
4. **Judge Agent fails** → Default to REJECT (conservative), log error
5. **Memory Agent fails** → Proceed without historical context, add warning

### Retry Logic

- **Transient failures** (rate limits, timeout): Retry with exponential backoff (1s, 2s, 4s)
- **Persistent failures**: Escalate to human after 3 attempts
- **Max retries per agent call**: 3

### Fallback Strategies

- **If threshold_snapshot.json missing**: Use hardcoded values, log warning
- **If config YAML missing**: Use default values, log error
- **If prompt file missing**: Raise FileNotFoundError (fatal)
- **If LLM API unavailable**: Fall back to mock mode (if enabled)

---

## Version History

### v2.0.0 (2025-02-04) - Major Modernization

**Changes**:
- Critical fixes: Threshold synchronization, timestamps, error handling
- Parameterization: All hardcoded values moved to agent_config.yaml
- Shared knowledge: This file created (500+ lines of redundancy eliminated)
- Schema validation: All outputs now validated against JSON schemas
- Prompt optimization: Target reduction from 7,611 to ~5,000 lines
- Advanced features: Prepared for semantic search, telemetry (disabled by default)

**Compatibility**: Requires Detector v1.5+, Evaluation System v2.0+

### v1.5.0 (2025-01-28)

- Added Memory Agent integration
- Updated FatigueDetector thresholds (1.2 → 1.15 → 1.10)

### v1.0.0 (2025-01-15)

- Initial prompt system
