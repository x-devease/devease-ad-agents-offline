# Ad Miner Final Design: Beat History

**Goal:** Create a reliable system to identify creative patterns that beat historical performance.

**Date:** 2026-01-27
**Status:** Ready for Final Review
**Focus:** Statistical pattern mining for ROAS lift

---

## Executive Summary

### Problem
Current ad recommender produces generic recommendations without:
- Concrete ROAS numbers (e.g., "2.8x higher ROAS" vs "Opportunity Size: 100.00")
- Context awareness (conversion vs traffic campaigns need different creatives)
- Statistical significance (are patterns real or noise?)
- Specific feature values to implement

### Solution
**Ad Miner v2.0**: Context-aware pattern mining with statistical validation

**Core Promise:** Identify creative features that statistically correlate with higher ROAS in specific contexts, providing concrete lift numbers.

**Target:** +20% ROAS lift from implementing mined patterns vs historical average.

---

## System Architecture

```
Input CSV (with schema validation)
    ↓
[Data Loader] → Validates & Loads
    ↓
[Segment Filter] → Split by (customer, product, branch, goal)
    ↓
[Sample Size Check] → Are there 200+ creatives?
    ↓ Yes                    ↓ No
[Level 1 Mining]      [Fallback to Level 2]
    ↓                        ↓
[PatternMiner] ← Core Algorithm
    ↓
Top Quartile vs Bottom Quartile Analysis
    ↓
Statistical Testing (Chi-square, p-values)
    ↓
ROAS Lift Calculations
    ↓
Pattern Ranking (Priority 0-10)
    ↓
Output JSON (validated) + MD (generated)
```

---

## Core Algorithm: PatternMiner

### 1. Quartile Analysis

**Input:** Segment data (e.g., "moprobo, Power Station, US, conversion")

```python
# Sort all creatives by ROAS descending
sorted_creatives = df.sort_values('roas', ascending=False)

# Split into quartiles
n = len(sorted_creatives)
top_quartile = sorted_creatives[:n//4]    # Top 25% performers
bottom_quartile = sorted_creatives[-n//4:] # Bottom 25% performers
```

**Key Metrics:**
- `avg_roas_top`: Average ROAS in top quartile
- `avg_roas_bottom`: Average ROAS in bottom quartile
- `roas_range`: Top ROAS / Bottom ROAS (e.g., 4.65x = top quartile performs 4.65x better)

### 2. Feature Prevalence Analysis

For each feature (e.g., `product_position`):

```python
# Count how many top creatives have each value
top_prevalence = {
    'bottom-right': 0.67,  # 67% of top quartile
    'center': 0.21,
    'top-left': 0.12,
}

# Count how many bottom creatives have each value
bottom_prevalence = {
    'bottom-right': 0.12,  # Only 12% of bottom quartile
    'center': 0.45,
    'top-left': 0.43,
}
```

### 3. ROAS Lift Calculation

```python
# For feature value "bottom-right" in product_position:
roas_lift_multiple = avg_roas_top_with_value / avg_roas_bottom_without_value
# Example: 4.5 / 1.6 = 2.8x

roas_lift_pct = (roas_lift_multiple - 1) * 100
# Example: (2.8 - 1) * 100 = 180% lift
```

**Concrete Example:**
> "For conversion campaigns with Power Station in US, products positioned bottom-right show **2.8x higher ROAS** (180% lift). Present in 67% of top performers vs 12% in bottom quartile."

### 4. Statistical Significance Testing

**Chi-square test:** Is this pattern real or random noise?

```python
from scipy.stats import chi2_contingency

# Contingency table
#                Top Quartile    Bottom Quartile
# bottom-right       57               10
# other values       28               75

chi2, p_value, _, _ = chi2_contingency(contingency_table)

# p_value < 0.05 = statistically significant
```

**Confidence Levels:**
- `high`: p < 0.01 (99% confident it's real)
- `medium`: 0.01 ≤ p < 0.05 (95% confident)
- `low`: p ≥ 0.05 (not significant, but worth noting)

### 5. Pattern Ranking (Priority Score)

Rank patterns by **actionable impact**:

```python
priority_score = (
    (roas_lift_multiple * 2.0) +      # Higher lift = more important
    (prevalence_lift * 1.5) +         # Bigger prevalence gap = more important
    (confidence_multiplier) +         # High confidence = +2, Medium = +1, Low = 0
    (sample_size_bonus)               # Larger sample = more reliable
)

# Normalize to 0-10 scale
priority_score = min(10.0, priority_score)
```

**Example:**
- Pattern: "product_position = bottom-right"
- ROAS lift: 2.8x → 5.6 points
- Prevalence lift: 0.55 (67% - 12%) → 0.825 points
- Confidence: high → 2.0 points
- **Total: 8.4/10** (top priority!)

---

## Granularity Hierarchy (Automatic Fallback)

### 4-Level System

**Level 1: Most Specific** (Target 200+ samples)
```
(customer, product, branch, campaign_goal)
Example: ("moprobo", "Power Station", "US", "conversion")
→ "For Power Station conversion campaigns in US..."
```

**Level 2: Product-specific** (Target 100+ samples)
```
(customer, product, campaign_goal)
Example: ("moprobo", "Power Station", "conversion")
→ "For Power Station conversion campaigns (all regions)..."
```

**Level 3: Goal-specific** (Target 50+ samples)
```
(customer, campaign_goal)
Example: ("moprobo", "conversion")
→ "For all conversion campaigns (all products)..."
```

**Level 4: Generic Fallback** (Minimum 30 samples)
```
(customer)
Example: ("moprobo")
→ "General patterns for moprobo..."
```

### Fallback Logic

```python
def get_patterns(customer, product, branch, goal):
    # Try Level 1
    if sample_size(customer, product, branch, goal) >= 200:
        return mine_level_1(customer, product, branch, goal)

    # Try Level 2
    if sample_size(customer, product, goal) >= 100:
        return mine_level_2(customer, product, goal)

    # Try Level 3
    if sample_size(customer, goal) >= 50:
        return mine_level_3(customer, goal)

    # Fallback to Level 4
    return mine_level_4(customer)
```

---

## Input Schema (CSV)

**File:** `src/meta/ad/miner/schemas/input_schema.yaml`

**Key Columns:**

### Identifiers
- `creative_id` - Unique ID (required, unique)
- `filename` - Image filename (required)

### Performance
- `roas` - Return on ad spend (required, ≥ 0)
- `spend` - Amount spent
- `impressions` - Number of impressions
- `clicks` - Number of clicks

### Context (Segmentation Keys)
- `campaign_goal` - "awareness" | "conversion" | "traffic" | "lead_generation" | "app_installs" | "engagement" | "sales" | "unknown"
- `product` - Product name (e.g., "Power Station")
- `branch` - "US" | "EU" | "UK" | "APAC" | "LATAM" | "Global" | "unknown"

### Creative Features (32+ features)
**Visual:**
- `direction`, `lighting_style`, `lighting_type`, `mood_lighting`
- `primary_colors`, `color_balance`, `temperature`, `color_saturation`, `color_vibrancy`
- `product_position`, `product_placement`, `product_visibility`, `visual_prominence`
- `human_elements`, `product_context`, `context_richness`, `background_content_type`
- `relationship_depiction`, `visual_flow`, `composition_style`, `depth_layers`
- `contrast_level`, `background_tone_contrast`, `local_contrast`
- `image_style`, `visual_complexity`, `product_angle`, `product_presentation`, `framing`
- `architectural_elements_presence`, `primary_focal_point`

**Human Subjects:**
- `person_count`, `person_relationship_type`, `person_gender`, `person_age_group`, `person_activity`

**Text & CTA:**
- `text_elements`, `cta_visuals`, `problem_solution_narrative`, `emotional_tone`, `activity_level`

---

## Output Schema (JSON v2.0)

**File:** `src/meta/ad/miner/schemas/output_schema.yaml`

**Structure:**

```json
{
  "metadata": {
    "schema_version": "2.0",
    "customer": "moprobo",
    "product": "Power Station",
    "branch": "US",
    "campaign_goal": "conversion",
    "granularity_level": 1,
    "sample_size": 342,
    "min_threshold": 200,
    "analysis_date": "2026-01-27",
    "fallback_used": false,
    "data_quality": {
      "completeness_score": 0.95,
      "avg_roas": 2.34,
      "top_quartile_roas": 4.56,
      "bottom_quartile_roas": 0.98,
      "roas_range": 4.65
    }
  },

  "patterns": [
    {
      "feature": "product_position",
      "value": "bottom-right",
      "pattern_type": "DO_CONVERSION",
      "confidence": "high",
      "roas_lift_multiple": 2.8,
      "roas_lift_pct": 180.0,
      "top_quartile_prevalence": 0.67,
      "bottom_quartile_prevalence": 0.12,
      "prevalence_lift": 0.55,
      "goal_specific": true,
      "product_specific": true,
      "branch_specific": false,
      "reason": "For conversion campaigns with Power Station in US, products positioned bottom-right show 2.8x higher ROAS. Present in 67% of top performers vs 12% in bottom quartile.",
      "maps_to_template": "product_position",
      "priority_score": 9.5,
      "statistical_significance": {
        "chi_square_stat": 45.23,
        "p_value": 0.00001,
        "significant": true
      }
    }
    // ... more patterns ranked by priority_score
  ],

  "anti_patterns": [
    {
      "feature": "product_position",
      "avoid_value": "top-left",
      "pattern_type": "DON'T",
      "confidence": "high",
      "roas_penalty_multiple": 0.6,
      "roas_penalty_pct": -40.0,
      "bottom_quartile_prevalence": 0.65,
      "top_quartile_prevalence": 0.15,
      "reason": "Used in 65% of worst performers, 40% lower ROAS than average",
      "maps_to_template": "product_position"
    }
    // ... more anti-patterns
  ],

  "low_priority_insights": [
    {
      "feature": "contrast_level",
      "value": "high",
      "confidence": "low",
      "roas_lift_multiple": 1.05,
      "roas_lift_pct": 5.0,
      "reason": "Slight positive trend (5% lift), but not statistically significant (p=0.15)"
    }
    // ... more minor trends
  ]
}
```

---

## Path Structure

**Base:** `config/ad/miner/`

### Mined Patterns (Hierarchical)
```
mined_patterns/
├── {customer}/
│   ├── {product}/
│   │   ├── {branch}/
│   │   │   ├── {goal}/
│   │   │   │   ├── patterns.json      ← Level 1 output
│   │   │   │   ├── patterns.md        ← Generated MD
│   │   │   │   └── index.json         ← Segment metadata
│   │   │   └── {goal}/
│   │   │       └── ...
│   │   ├── {goal}/                    ← Level 2 (no branch)
│   │   │   └── patterns.json
│   │   └── ...
│   ├── {goal}/                        ← Level 3 (no product)
│   │   └── patterns.json
│   └── patterns.json                  ← Level 4 (customer only)
```

### Input
```
input/
├── {customer}/
│   ├── creative_features.csv
│   └── validation_report.json
```

### Cache
```
cache/
├── {customer}_features_cache.pkl
└── {customer}_checkpoint.ckpt
```

### Results
```
results/
└── {customer}/
    ├── statistical_tests.json
    ├── feature_importance.png
    └── quartile_comparison.png
```

---

## Query API (with Automatic Fallback)

**File:** `src/meta/ad/miner/api.py` (proposed)

```python
class AdMinerAPI:
    """Query patterns with automatic fallback across granularity levels."""

    def get_patterns(
        self,
        customer: str,
        product: str = "all",
        branch: str = "all",
        goal: str = "all",
        min_priority: float = 0.0,
    ) -> dict:
        """
        Get patterns for a segment with automatic fallback.

        Strategy:
        1. Try Level 1 (product+branch+goal) - if 200+ samples
        2. Fallback to Level 2 (product+goal) - if 100+ samples
        3. Fallback to Level 3 (goal) - if 50+ samples
        4. Fallback to Level 4 (customer) - always available

        Returns:
            dict with:
                - patterns: List of high-priority DOs
                - anti_patterns: List of DON'Ts
                - granularity_level: Which level was returned
                - fallback_used: Whether fallback occurred
                - metadata: Data quality info
        """
        # Implementation...
```

---

## Implementation Status

### ✅ Completed (Phase 1-2)

**Schemas & Validation:**
- ✅ `schemas/input_schema.yaml` - Complete CSV schema (32+ features)
- ✅ `schemas/output_schema.yaml` - Complete JSON v2.0 schema
- ✅ `validation/input_validator.py` - CSV validation
- ✅ `validation/output_validator.py` - JSON validation

**Data Management:**
- ✅ `utils/paths.py` - MinerPaths class (4-level hierarchy)
- ✅ `data/loader.py` - CreativeFeaturesLoader (segment filtering, data quality)
- ✅ `io/patterns_io.py` - JSON I/O + Markdown generator

### 🚧 Pending (Phase 3-4)

**Core Algorithm:**
- 🚧 `PatternMiner` class - Quartile analysis, statistical testing, ROAS calculations
- 🚧 Statistical testing module - Chi-square, p-values, confidence intervals
- 🚧 Pattern ranking - Priority score calculation
- 🚧 Anti-pattern detection - Bottom quartile patterns to avoid

**Segmentation:**
- 🚧 Multi-level mining logic - Automatic fallback system
- 🚧 Sample size thresholds - Level detection
- 🚧 Segment caching - Store/retrieve mined patterns

**API:**
- 🚧 Query API - Get patterns with fallback
- 🚧 Pattern integration - Feed to ad generator

---

## Design Decisions

### 1. Why "Pattern" instead of "Recommendation"?
**Answer:** "Pattern" describes what we do: discover statistical patterns in data. "Recommendation" suggests advice, but we provide data-driven insights.

### 2. Why JSON instead of just Markdown?
**Answer:**
- JSON = Machine-readable, structured, validated
- Markdown = Human-readable, generated from JSON
- Best of both worlds: API integration + readability

### 3. Why 4-level granularity?
**Answer:** Balance between specificity and sample size
- Level 1 = Most actionable (context-specific)
- Fallback ensures we always have patterns
- Sample size thresholds prevent false positives

### 4. Why quartile analysis vs mean/median?
**Answer:**
- Quartiles show the gap between best and worst
- Clearer signal: "top creatives do X, bottom don't"
- More actionable than "average is X"

### 5. Why chi-square test?
**Answer:**
- Tests: "Is feature distribution different between top/bottom?"
- Standard for categorical feature analysis
- P-value gives confidence level

---

## Success Criteria

### Quantitative
- ✅ Input schema validated (100% coverage of features)
- ✅ Output schema validated (100% coverage of fields)
- 🎯 95%+ of patterns have p < 0.05 (statistically significant)
- 🎯 Average ROAS lift in patterns > 1.5x (50% improvement)
- 🎯 Query latency < 100ms (cached segments)

### Qualitative
- ✅ Clear, structured schemas
- ✅ Hierarchical path organization
- 🎯 Actionable patterns with concrete ROAS numbers
- 🎀 Easy to integrate with ad generator
- 🎀 Clear documentation

---

## Next Steps (For Final Review)

1. **Review this design** - Does this meet the "beat history" goal?
2. **Approve implementation plan** - Ready to build PatternMiner?
3. **Define testing strategy** - How do we validate it works?
4. **Plan rollout** - Migration path from current system?

---

## Appendix: Example Output

### Input Request
```python
api.get_patterns(
    customer="moprobo",
    product="Power Station",
    branch="US",
    goal="conversion"
)
```

### Output (JSON)
```json
{
  "patterns": [
    {
      "feature": "product_position",
      "value": "bottom-right",
      "confidence": "high",
      "roas_lift_multiple": 2.8,
      "priority_score": 9.5,
      "reason": "For conversion campaigns with Power Station in US, products positioned bottom-right show 2.8x higher ROAS (180% lift). Present in 67% of top performers vs 12% in bottom quartile. Statistically significant (p<0.0001)."
    },
    {
      "feature": "lighting_style",
      "value": "studio",
      "confidence": "high",
      "roas_lift_multiple": 2.1,
      "priority_score": 8.2,
      "reason": "Studio lighting shows 2.1x higher ROAS for Power Station conversion campaigns. Present in 72% of top quartile."
    }
  ],
  "anti_patterns": [
    {
      "feature": "product_position",
      "avoid_value": "top-left",
      "confidence": "high",
      "roas_penalty_multiple": 0.6,
      "reason": "Products positioned top-left show 40% lower ROAS. Used in 65% of worst performers."
    }
  ],
  "metadata": {
    "granularity_level": 1,
    "sample_size": 342,
    "avg_roas": 2.34,
    "top_quartile_roas": 4.56,
    "bottom_quartile_roas": 0.98
  }
}
```

### Generated Markdown
```markdown
# Mined Creative Patterns

**Customer:** moprobo
**Product:** Power Station
**Branch:** US
**Campaign Goal:** conversion
**Analysis Date:** 2026-01-27

## Data Quality

- **Sample Size:** 342
- **Completeness Score:** 0.95
- **Average ROAS:** 2.34
- **Top Quartile ROAS:** 4.56
- **Bottom Quartile ROAS:** 0.98

## Positive Patterns (DOs)

Found 12 positive patterns ranked by priority:

### 1. product_position = bottom-right

- **Confidence:** high
- **ROAS Lift:** 2.8x (180.0% increase)
- **Top Quartile Prevalence:** 67.0%
- **Priority Score:** 9.5/10
- **Reason:** For conversion campaigns with Power Station in US, products positioned bottom-right show 2.8x higher ROAS (180% lift). Present in 67% of top performers vs 12% in bottom quartile. Statistically significant (p<0.0001).

### 2. lighting_style = studio

- **Confidence:** high
- **ROAS Lift:** 2.1x (110.0% increase)
- **Top Quartile Prevalence:** 72.0%
- **Priority Score:** 8.2/10
- **Reason:** Studio lighting shows 2.1x higher ROAS for Power Station conversion campaigns. Present in 72% of top quartile.

## Negative Patterns (DON'Ts)

Found 5 negative patterns to avoid:

### 1. Avoid: product_position = top-left

- **Confidence:** high
- **ROAS Penalty:** 0.6x (-40.0% decrease)
- **Bottom Quartile Prevalence:** 65.0%
- **Reason:** Products positioned top-left show 40% lower ROAS. Used in 65% of worst performers.

---

*Generated by Ad Miner v2.0*
*2026-01-27*
```

---

**End of Design Document**
