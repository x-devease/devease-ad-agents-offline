# Ad Miner Improvement Plan: Complete Roadmap

**Status:** Comprehensive Improvement Plan
**Date:** 2026-01-27
**Branch:** ad-reviewer
**Estimated Duration:** 6-8 weeks
**Priority:** High

---

## Executive Summary

The ad miner successfully implements statistical pattern discovery but has **critical gaps** in goal alignment and format effectiveness. This plan addresses these issues through a structured 6-phase improvement roadmap.

**Current State:**
- ✅ Solid statistical foundation (rule-based, transparent)
- ❌ Missing concrete ROAS lift numbers
- ❌ Generic recommendations (no context awareness)
- ❌ Format issues (capped opportunity sizes, redundant structure)
- ❌ ~40% feature translation loss to ad generator

**Target State:**
- ✅ Concrete ROAS numbers ("2.8x higher ROAS" instead of "100.00")
- ✅ Context-aware recommendations (goal/product/branch specific)
- ✅ Structured JSON format with generated MD views
- ✅ Complete feature mapping to ad generator
- ✅ Graceful fallback to broader aggregations

**Expected Impact:**
- +20% ROAS lift from context-aware vs generic recommendations
- 100% feature coverage (vs 60% currently)
- Clearer priorities with concrete impact numbers
- Faster iteration with structured JSON format

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Solution Overview](#2-solution-overview)
3. [Detailed Design](#3-detailed-design)
4. [Implementation Phases](#4-implementation-phases)
5. [Testing Strategy](#5-testing-strategy)
6. [Rollout Plan](#6-rollout-plan)
7. [Success Metrics](#7-success-metrics)
8. [Risk Mitigation](#8-risk-mitigation)

---

## 1. Problem Analysis

### 1.1 Goal Alignment Gaps

**Stated Goal (from SELF_REFLECTION.md):**
> Mine historical ad data to discover proven creative patterns that drive performance, then output **concrete DO/DON'T recommendations** (e.g., "**products positioned bottom-right show 2.3x higher ROAS**", "**bright backgrounds increase engagement by 40%**")

**Actual Output:**
```
5. **Direction**: `Overhead` (High, DO) — Opportunity Size: 100.00
25. **Product Position**: `bottom-right` (Medium, DO) — Opportunity Size: 25.34
```

| Expected | Reality | Gap |
|----------|---------|-----|
| "2.3x higher ROAS" | "Opportunity Size: 25.34" | Missing concrete multiples |
| Campaign-specific | Generic across all | No context awareness |
| Product-specific | Generic across all | No product customization |
| Branch-specific | Generic across all | No regional customization |

---

### 1.2 Format Issues

**Current Markdown Structure:**
```markdown
## Opportunities
[Mixed DOs and DON'Ts, sorted by opportunity_size]

## DO - Positive Patterns to Implement
[Detailed DO recommendations]

## DON'T - Anti-Patterns to Avoid
[Detailed DON'T recommendations]

## Opportunities Summary
[Aggregated statistics]
```

**Problems:**
1. **Redundancy:** Same info repeated 3 times
2. **Confusion:** DON'Ts in "Opportunities" section
3. **Capped values:** "100.00" loses differentiation
4. **No hierarchy:** All recommendations appear equal priority
5. **Machine-unfriendly:** MD-only format hard to parse

---

### 1.3 Feature Translation Loss

**Unmapped Features** (from recommendations.md):
- `text_elements: "Headline, Subheadline, Feature Icons"` → No placeholder mapping
- `cta_visuals: "Highlighting, Button"` → No placeholder mapping
- `primary_colors` as list → Template expects single color

**Impact:** ~40% of recommendations silently dropped during conversion.

**Root Cause:** Mismatch between feature extraction output and ad generator template placeholders.

---

## 2. Solution Overview

### 2.1 Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                             │
├─────────────────────────────────────────────────────────────┤
│  CSV with metadata:                                          │
│  - creative_id, roas, feature columns                        │
│  + campaign_goal (awareness/conversion/traffic)             │
│  + product (Power Station / MoProBo / ...)                  │
│  + branch (US / EU / APAC / ...)                            │
│  + campaign_id, adset_id (for grouping)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  - Schema validation (types, enums, ranges)                │
│  - Data quality checks (missing values, duplicates)         │
│  - Sample size validation (per segment)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  SEGMENTATION LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  Level 1: (customer, product, branch, goal) → 200+ samples  │
│  Level 2: (customer, product, goal) → 100+ samples          │
│  Level 3: (customer, goal) → 50+ samples                    │
│  Level 4: (customer) → Generic fallback                      │
│                                                              │
│  Auto-select granularity based on sample size               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  PATTERN DETECTION                           │
├─────────────────────────────────────────────────────────────┤
│  - Statistical analysis per segment                         │
│  - Goal-specific pattern detection                          │
│  - Product-specific pattern detection                       │
│  - Branch-specific pattern detection                        │
│  - Chi-square significance testing                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  Primary: JSON v2.0 (machine-readable)                      │
│  - metadata (context, quality, stats)                       │
│  - high_impact_recommendations (top 5-10)                   │
│  - negative_guidance (DON'Ts)                               │
│  - low_priority_insights (trends)                           │
│  - generation_instructions (for ad generator)               │
│                                                              │
│  Generated: MD (human-readable view)                        │
│  - Clean, non-redundant structure                           │
│  - Concrete ROAS numbers ("2.8x higher")                    │
│  - Priority-ranked sections                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     QUERY LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  load_recommendations_with_fallback():                      │
│  1. Try Level 1 (most specific)                            │
│  2. Fallback to Level 2                                     │
│  3. Fallback to Level 3                                     │
│  4. Fallback to Level 4 (generic)                           │
│                                                              │
│  Return metadata showing which level was used               │
└─────────────────────────────────────────────────────────────┘
```

---

### 2.2 Key Improvements

**1. Concrete ROAS Numbers**
- Before: "Opportunity Size: 100.00"
- After: "+180% ROAS (2.8x lift)"

**2. Context-Awareness**
- Before: Generic recommendations for all campaigns
- After: "For conversion campaigns with Power Station in US..."

**3. Structured Output**
- Before: MD-only, redundant sections
- After: JSON primary, MD generated, single source of truth

**4. Complete Feature Mapping**
- Before: ~40% feature loss
- After: Audit and expand template placeholders

**5. Priority Scoring**
- Before: All recommendations appear equal
- After: Priority-score ranked (high_impact vs low_priority)

**6. Graceful Degradation**
- Before: All-or-nothing (generic or bust)
- After: Automatic fallback to appropriate granularity level

---

## 3. Detailed Design

### 3.1 Input Schema (CSV)

**New metadata columns:**
```csv
creative_id, filename, roas, campaign_goal, product, branch, campaign_id, adset_id, [feature columns]
moprobo_001, img.jpg, 2.45, conversion, Power Station, US, camp_123, adset_456, overhead, studio, ...
```

**Enum constraints:**
- `campaign_goal`: awareness | conversion | traffic | lead_generation | app_installs | unknown
- `branch`: US | EU | APAC | LATAM | Global | unknown

**Validation rules:**
- ROAS ≥ 0
- creative_id unique
- Required columns present
- Enum values valid

---

### 3.2 Output Schema (JSON v2.0)

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
    "analysis_date": "2026-01-27",
    "data_quality": {
      "avg_roas": 2.34,
      "top_quartile_roas": 4.56
    }
  },
  "high_impact_recommendations": [
    {
      "feature": "product_position",
      "recommended_value": "bottom-right",
      "roas_lift_multiple": 2.8,
      "roas_lift_pct": 180.0,
      "top_quartile_prevalence": 0.67,
      "confidence": "high",
      "goal_specific": true,
      "priority_score": 9.5
    }
  ],
  "negative_guidance": [...],
  "low_priority_insights": [...],
  "generation_instructions": {
    "must_include": ["product_position", "lighting_style"],
    "avoid": ["product_position:top-left"]
  }
}
```

---

### 3.3 Granularity Levels

| Level | Key | Min Samples | Example | Use Case |
|-------|-----|-------------|---------|----------|
| 1 | (customer, product, branch, goal) | 200 | moprobo + Power Station + US + conversion | Highly tailored |
| 2 | (customer, product, goal) | 100 | moprobo + Power Station + conversion | Product-specific |
| 3 | (customer, goal) | 50 | moprobo + conversion | Goal-specific |
| 4 | (customer) | 0 | moprobo | Generic fallback |

**Auto-selection logic:**
```python
if product + branch + goal has ≥200 samples:
    use Level 1
elif product + goal has ≥100 samples:
    use Level 2
elif goal has ≥50 samples:
    use Level 3
else:
    use Level 4 (generic)
```

---

### 3.4 Query with Fallback

**API:**
```python
recommendations = load_recommendations_with_fallback(
    customer="moprobo",
    product="Power Station",
    branch="US",
    campaign_goal="conversion"
)
```

**Search order:**
1. `moprobo/Power_Station/US/conversion/recommendations.json` (Level 1)
2. `moprobo/Power_Station/conversion/recommendations.json` (Level 2)
3. `moprobo/conversion/recommendations.json` (Level 3)
4. `moprobo/recommendations.json` (Level 4)

**Response includes:**
```json
{
  "metadata": {
    "fallback_used": true,
    "fallback_level": 3,
    "reason": "Insufficient data for Level 1-2. Using goal-specific recommendations."
  }
}
```

---

### 3.5 File Structure

```
config/ad/miner/
├── moprobo/
│   ├── segment_index.json (tracks all segments)
│   ├── Power_Station/
│   │   ├── US/
│   │   │   ├── conversion/
│   │   │   │   ├── recommendations.json (Level 1)
│   │   │   │   └── recommendations.md (generated)
│   │   │   └── awareness/
│   │   │       └── recommendations.json (Level 1)
│   │   ├── conversion/
│   │   │   └── recommendations.json (Level 2)
│   │   └── recommendations.json (Level 2 fallback)
│   ├── conversion/
│   │   └── recommendations.json (Level 3)
│   ├── awareness/
│   │   └── recommendations.json (Level 3)
│   └── recommendations.json (Level 4, generic)
└── customer_b/
    └── ...
```

---

## 4. Implementation Phases

### Phase 1: Fix Core Output Issues (Week 1)

**Goals:**
- Add concrete ROAS lift numbers
- Remove opportunity size cap
- Simplify markdown format

**Tasks:**
1. ✅ Design completed (this document)
2. ⏳ Implement `roas_lift_multiple` calculation
   ```python
   top_q_roas = top_quartile["roas"].mean()
   bottom_q_roas = bottom_quartile["roas"].mean()
   roas_lift_multiple = top_q_roas / bottom_q_roas
   roas_lift_pct = (roas_lift_multiple - 1) * 100
   ```
3. ⏳ Remove opportunity size cap
   ```python
   # OLD: cap at 100.0
   # NEW: use log scale or keep raw values
   opportunity_size = roas_lift_pct  # No cap
   ```
4. ⏳ Simplify MD format
   - Remove redundant "Opportunities" section
   - Filter out 0.00 opportunity size recommendations
   - Keep only DO and DON'T sections

**Deliverables:**
- Updated `rule_engine.py` with ROAS lift calculations
- Simplified `md_io.py` output format
- Unit tests for ROAS lift calculations

**Success criteria:**
- Output shows "2.8x higher ROAS" instead of "100.00"
- MD has no redundant sections
- All recommendations have concrete impact numbers

---

### Phase 2: Implement Schema Validation (Week 1-2)

**Goals:**
- Add input validation for CSV schema
- Add output validation for JSON schema
- Catch data quality issues early

**Tasks:**
1. ⏳ Create `src/meta/ad/miner/validation/input_validator.py`
   - Validate required columns exist
   - Check enum values (goal, product, branch)
   - Validate data types (ROAS is numeric, not negative)
   - Check for missing critical values
   - Detect duplicate creative_ids

2. ⏳ Create `src/meta/ad/miner/validation/output_validator.py`
   - Validate JSON structure against schema v2.0
   - Check required metadata fields
   - Validate value ranges (0-1 for prevalence, 1+ for lift)
   - Check confidence values

3. ⏳ Update data loader
   ```python
   def load_features_with_metadata(csv_path, validate=True):
       if validate:
           validator = InputSchemaValidator(csv_path)
           if not validator.validate():
               raise ValueError("Validation failed")
       return pd.read_csv(csv_path)
   ```

**Deliverables:**
- `InputSchemaValidator` class
- `OutputSchemaValidator` class
- Updated data loader with validation
- Validation error messages and reports

**Success criteria:**
- Invalid CSV files rejected with clear error messages
- Invalid JSON structures caught before saving
- Validation catches all data quality issues from analysis

---

### Phase 3: Add Metadata Columns (Week 2)

**Goals:**
- Extend feature extraction to output metadata
- Backfill existing data with defaults
- Update CSV schema documentation

**Tasks:**
1. ⏳ Update GPT-4 feature extraction
   - Add `campaign_goal` column (default: "unknown")
   - Add `product` column (default: "unknown")
   - Add `branch` column (default: "unknown")
   - Add `campaign_id` and `adset_id` (optional)

2. ⏳ Create backfill script
   ```python
   def backfill_metadata(df):
       df["campaign_goal"] = df.get("campaign_goal", "unknown")
       df["product"] = df.get("product", "unknown")
       df["branch"] = df.get("branch", "unknown")
       return df
   ```

3. ⏳ Update schema documentation
   - Document new required columns
   - Add enum value definitions
   - Provide examples

**Deliverables:**
- Updated feature extraction with metadata
- Backfill script for existing data
- CSV schema documentation (`docs/csv_schema.md`)

**Success criteria:**
- New CSV files include metadata columns
- Existing files can be backfilled
- Feature extraction validates metadata values

---

### Phase 4: Implement Segmentation (Week 2-3)

**Goals:**
- Add segmentation logic
- Implement granularity selection
- Create context-aware pattern detection

**Tasks:**
1. ⏳ Create `src/meta/ad/miner/segmentation.py`
   ```python
   class SegmentAnalyzer:
       def segment_data(df, product, branch, goal):
           # Filter to segment
           return df[...]

       def select_granularity_level(product, branch, goal):
           # Choose level based on sample size
           # Return 1-4
   ```

2. ⏳ Implement goal-specific pattern detection
   ```python
   def detect_goal_specific_patterns(df, feature):
       # Compare feature impact across goals
       # Return patterns unique to each goal
   ```

3. ⏳ Implement product-specific pattern detection
   ```python
   def detect_product_specific_patterns(df, feature):
       # Compare feature impact across products
       # Return patterns unique to each product
   ```

4. ⏳ Implement branch-specific pattern detection
   ```python
   def detect_branch_specific_patterns(df, feature):
       # Compare feature impact across branches
       # Return patterns unique to each branch
   ```

**Deliverables:**
- `SegmentAnalyzer` class
- Pattern detection algorithms
- Unit tests for segmentation logic
- Integration tests with sample data

**Success criteria:**
- Can segment data by product/branch/goal
- Correctly selects granularity level
- Detects statistically significant context-specific patterns

---

### Phase 5: Implement JSON Output Format (Week 3-4)

**Goals:**
- Create structured JSON output (v2.0)
- Implement markdown generator
- Create output writer utilities

**Tasks:**
1. ⏳ Create `src/meta/ad/miner/output/writer.py`
   ```python
   def write_recommendations(recommendations, output_path):
       # Add metadata (schema_version, analysis_date)
       # Validate against schema
       # Write to JSON with indentation
   ```

2. ⏳ Create `src/meta/ad/miner/output/markdown_generator.py`
   ```python
   def generate_markdown(recommendations_json):
       # Convert JSON to human-readable MD
       # Priority-ranked sections
       # Concrete ROAS numbers
   ```

3. ⏳ Update `generate_recommendations()` to output v2.0 format
   ```python
   def generate_recommendations(df, customer, product, branch, goal):
       # Analyze segment
       # Build v2.0 structure
       return {
           "metadata": {...},
           "high_impact_recommendations": [...],
           "negative_guidance": [...],
           ...
       }
   ```

4. ⏳ Create segment index writer
   ```python
   def write_segment_index(customer, segments):
       # Write segment_index.json
       # Track all segments with metadata
   ```

**Deliverables:**
- JSON v2.0 output format
- Markdown generator
- Output writer with validation
- Segment index management

**Success criteria:**
- Recommendations saved as JSON with full metadata
- MD view generated from JSON (not duplicated work)
- Output validation passes before saving
- Segment index tracks all available segments

---

### Phase 6: Implement Query API with Fallback (Week 4-5)

**Goals:**
- Create query function with automatic fallback
- Update ad generator to use new API
- Add CLI support for context parameters

**Tasks:**
1. ⏳ Create `src/meta/ad/miner/queries.py`
   ```python
   def load_recommendations_with_fallback(
       customer, product, branch, goal
   ):
       # Try Level 1, then 2, then 3, then 4
       # Return with metadata about which level used
   ```

2. ⏳ Update ad generator integration
   ```python
   # In ad_miner_adapter.py
   def load_for_generator(customer, product, branch, goal):
       recs = load_recommendations_with_fallback(
           customer, product, branch, goal
       )
       return convert_to_visual_formula(recs)
   ```

3. ⏳ Update CLI
   ```bash
   python run.py recommend \
       --customer moprobo \
       --product "Power Station" \
       --branch US \
       --goal conversion
   ```

4. ⏳ Add fallback metadata logging
   ```python
   if recommendations["metadata"]["fallback_used"]:
       logger.warning(
           f"Using Level {recommendations['metadata']['fallback_level']} "
           f"recommendations (insufficient data for requested context)"
       )
   ```

**Deliverables:**
- `load_recommendations_with_fallback()` function
- Updated ad generator integration
- Updated CLI with context parameters
- Fallback logging

**Success criteria:**
- Query returns appropriate level based on data availability
- Ad generator uses context-aware recommendations
- CLI supports all context parameters
- Fallback behavior is transparent to users

---

### Phase 7: Fix Feature Mapping (Week 5)

**Goals:**
- Audit unmapped features
- Create mappings for lost features
- Expand template placeholders if needed

**Tasks:**
1. ⏳ Audit feature mapping gaps
   ```python
   unmapped = []
   for feature in all_features:
       if not get_placeholder_for_feature(feature):
           unmapped.append(feature)
   # Result: text_elements, cta_visuals, primary_colors (list)
   ```

2. ⏳ Create mappings for unmapped features
   ```python
   RECOMMENDATION_TO_PLACEHOLDER.update({
       "text_elements": ("text_overlay", None),
       "cta_visuals": ("cta_style", None),
       "primary_colors": ("color_palette", lambda v: f"Colors: {v}"),
       ...
   })
   ```

3. ⏳ Update transformer functions
   ```python
   def transform_feature_value(feature, value):
       # Handle list values (e.g., "green, white, gray")
       # Handle complex values (e.g., "Headline, Subheadline")
       # Return template-compatible format
   ```

4. ⏳ Update ad generator templates (if needed)
   - Add new placeholders for unmapped features
   - Or create "catch-all" placeholder for flexible features

**Deliverables:**
- Complete feature mapping audit
- Updated `recommendation_mapping.py`
- Feature value transformers
- Updated ad generator templates (if needed)

**Success criteria:**
- 100% feature coverage (no silent drops)
- All features map to valid template placeholders
- Ad generator consumes all recommendations

---

### Phase 8: Backward Compatibility & Migration (Week 5-6)

**Goals:**
- Support v1.0 format during transition
- Auto-migrate v1.0 → v2.0
- Document migration path

**Tasks:**
1. ⏳ Create compatibility layer
   ```python
   def load_with_migration(path):
       # Detect schema version
       # If v1.0, migrate to v2.0 with defaults
       # If v2.0, load directly
   ```

2. ⏳ Implement v1.0 → v2.0 migration
   ```python
   def migrate_v1_to_v2(v1_recs):
       return {
           "metadata": {
               "schema_version": "2.0",
               "granularity_level": 4,
               "migrated_from_v1": True,
               ...
           },
           "high_impact_recommendations": _migrate_rec_list(v1_recs),
           ...
       }
   ```

3. ⏳ Update all loaders to use migration
   ```python
   # In pipeline, ad_miner_adapter, etc.
   recommendations = load_with_migration(path)
   # Always returns v2.0 format
   ```

4. ⏳ Document migration
   - Migration guide for users
   - Breaking changes documentation
   - Timeline for v1.0 deprecation

**Deliverables:**
- Compatibility layer
- Migration functions
- Updated loaders with migration
- Migration documentation

**Success criteria:**
- v1.0 files can be loaded and auto-migrated
- Existing workflows continue to work
- Clear deprecation timeline communicated

---

### Phase 9: Testing & Validation (Week 6)

**Goals:**
- Comprehensive test coverage
- End-to-end integration testing
- Performance validation

**Tasks:**
1. ⏳ Unit tests
   - Input validator tests
   - Output validator tests
   - Segmentation tests
   - Pattern detection tests
   - ROAS lift calculation tests
   - Feature mapping tests

2. ⏳ Integration tests
   - End-to-end: CSV → recommendations → ad generator
   - Multi-level fallback testing
   - Context-aware pattern detection
   - Feature mapping coverage

3. ⏳ Performance tests
   - Large dataset handling (1000+ creatives)
   - Multi-segment analysis performance
   - Query latency (<100ms target)

4. ⏳ Data quality tests
   - Statistical significance validation
   - ROAS lift accuracy
   - Sample size threshold validation

**Deliverables:**
- Comprehensive test suite
- Integration test scenarios
- Performance benchmarks
- Test coverage report (>80%)

**Success criteria:**
- All tests pass
- Test coverage >80%
- Performance targets met
- No regressions in existing functionality

---

### Phase 10: Documentation & Training (Week 6-7)

**Goals:**
- Complete documentation
- Training materials
- Examples and tutorials

**Tasks:**
1. ⏳ Update documentation
   - Architecture overview
   - API reference
   - Schema documentation (input/output)
   - Migration guide
   - Troubleshooting guide

2. ⏳ Create examples
   - Example CSV with metadata
   - Example recommendations (all 4 levels)
   - Query examples
   - Integration examples

3. ⏳ Create tutorials
   - Getting started guide
   - Advanced segmentation guide
   - Custom pattern detection
   - Performance optimization

4. ⏳ Record training videos (optional)
   - Overview demo
   - How to use context-aware recommendations
   - How to extend the system

**Deliverables:**
- Complete documentation set
- Example datasets and outputs
- Tutorials and guides
- Training materials

**Success criteria:**
- Documentation is comprehensive and clear
- Examples work end-to-end
- Users can self-service onboarding

---

### Phase 11: Rollout & Monitoring (Week 7-8)

**Goals:**
- Gradual rollout to production
- Monitor for issues
- Collect feedback

**Tasks:**
1. ⏳ Canary deployment
   - Deploy to test environment first
   - Run parallel with existing system
   - Compare outputs

2. ⏳ Gradual rollout
   - Week 1: 10% of traffic
   - Week 2: 50% of traffic
   - Week 3: 100% of traffic

3. ⏳ Monitoring
   - Track recommendation quality metrics
   - Monitor query latency
   - Watch for errors/warnings
   - Collect user feedback

4. ⏳ Rollback plan
   - Document rollback procedure
   - Keep v1.0 system available for 2 weeks
   - Quick switch back if issues arise

**Deliverables:**
- Deployment checklist
- Monitoring dashboard
- Rollback procedure
- Post-deployment report

**Success criteria:**
- Successful deployment to production
- No critical issues
- Positive user feedback
- Measurable ROAS improvement

---

## 5. Testing Strategy

### 5.1 Unit Tests

**Coverage targets:**
- Input validation: 100%
- Output validation: 100%
- Segmentation logic: 95%
- Pattern detection: 90%
- Feature mapping: 100%
- ROAS calculations: 100%

**Key test cases:**
```python
# Test ROAS lift calculation
def test_roas_lift_calculation():
    top_q = pd.DataFrame({"roas": [4.0, 4.5, 5.0]})
    bottom_q = pd.DataFrame({"roas": [0.5, 0.8, 1.0]})
    lift = calculate_roas_lift(top_q, bottom_q)
    assert lift["roas_lift_multiple"] == pytest.approx(5.0, 0.5)
    assert lift["roas_lift_pct"] == pytest.approx(400.0, 50.0)

# Test granularity selection
def test_granularity_selection():
    segments = {
        ("Power Station", "US", "conversion"): 250,  # Level 1
        ("Power Station", "conversion"): 150,       # Level 2
        ("conversion",): 80,                        # Level 3
    }
    level = select_granularity("Power Station", "US", "conversion", segments)
    assert level == 1

# Test fallback
def test_fallback_to_level_3():
    segments = {
        ("conversion",): 80,  # Only Level 3 available
    }
    level = select_granularity("NewProduct", "US", "conversion", segments)
    assert level == 3

# Test feature mapping
def test_all_features_mapped():
    for feature in ALL_FEATURES:
        placeholder = get_placeholder_for_feature(feature)
        assert placeholder is not None, f"Feature {feature} not mapped"
```

---

### 5.2 Integration Tests

**End-to-end scenarios:**
```python
# Scenario 1: Level 1 recommendations (most specific)
def test_level_1_recommendations():
    # Given: 250 Power Station + US + conversion creatives
    df = load_test_data("power_station_us_conversion.csv")
    recs = generate_recommendations(df, "moprobo", "Power Station", "US", "conversion")

    # Then: Level 1 recommendations generated
    assert recs["metadata"]["granularity_level"] == 1
    assert recs["metadata"]["sample_size"] == 250
    assert len(recs["high_impact_recommendations"]) > 0
    assert any(r["goal_specific"] for r in recs["high_impact_recommendations"])

# Scenario 2: Fallback from Level 1 to Level 2
def test_fallback_to_level_2():
    # Given: Only 50 samples for Level 1
    df = load_test_data("power_station_few_samples.csv")
    recs = generate_recommendations(df, "moprobo", "Power Station", None, "conversion")

    # Then: Fallback to Level 2
    assert recs["metadata"]["granularity_level"] == 2
    assert recs["metadata"]["fallback_used"] == True

# Scenario 3: Complete feature mapping to ad generator
def test_feature_mapping_coverage():
    # Given: Recommendations with all features
    recs = generate_test_recommendations()

    # When: Convert to visual formula
    formula = convert_to_visual_formula(recs)

    # Then: All features mapped
    original_features = set(r["feature"] for r in recs["high_impact_recommendations"])
    mapped_features = set(f["_original_feature"] for f in formula["entrance_features"])
    assert mapped_features.issuperset(original_features)
```

---

### 5.3 Performance Tests

**Benchmarks:**
```python
# Test: Large dataset handling
def test_large_dataset_performance():
    df = generate_test_creatives(1000)  # 1000 creatives
    start = time.time()
    recs = generate_recommendations(df, "test_customer")
    duration = time.time() - start
    assert duration < 10.0  # Should complete in <10 seconds

# Test: Query latency
def test_query_latency():
    # Given: Pre-generated recommendations
    write_test_segments()

    # When: Query with fallback
    start = time.time()
    recs = load_recommendations_with_fallback("moprobo", "Power Station", "US", "conversion")
    duration = time.time() - start

    # Then: Fast response
    assert duration < 0.1  # <100ms target
```

---

### 5.4 Data Quality Tests

**Statistical validation:**
```python
# Test: ROAS lift accuracy
def test_roas_lift_accuracy():
    # Given: Known data with 2.5x lift for feature X
    df = create_test_dataset(
        feature_lift={("product_position", "bottom-right"): 2.5}
    )

    # When: Generate recommendations
    recs = generate_recommendations(df, "test")

    # Then: Lift detected correctly
    rec = next(r for r in recs["high_impact_recommendations"]
                if r["feature"] == "product_position")
    assert rec["roas_lift_multiple"] == pytest.approx(2.5, 0.3)

# Test: Statistical significance
def test_statistical_significance():
    # Given: Weak pattern (not significant)
    df = create_test_dataset(effect_size=0.1, n=50)

    # When: Generate recommendations
    recs = generate_recommendations(df, "test")

    # Then: Should be in low_priority_insights
    assert all(r["confidence"] in ["low", "medium"]
               for r in recs["high_impact_recommendations"])
```

---

## 6. Rollout Plan

### 6.1 Deployment Strategy

**Week 7: Canary (10% of traffic)**
```
Production (current)
    ↓
Parallel run (new system)
    ↓
Compare outputs
    ↓
Deploy to 10% of users
```

**Week 8: Gradual rollout**
- Day 1-2: 10% (canary)
- Day 3-4: 50% (majority)
- Day 5-7: 100% (full rollout)

**Week 9+: Monitor and optimize**
- Track metrics daily
- Collect user feedback
- Address issues promptly

---

### 6.2 Monitoring Dashboard

**Key metrics to track:**
1. **Recommendation quality**
   - ROAS lift from context-aware vs generic
   - Feature coverage (% mapped)
   - Statistical significance rate

2. **System performance**
   - Query latency (p50, p95, p99)
   - Analysis duration
   - Error rate

3. **Usage patterns**
   - Granularity levels requested vs used
   - Fallback rate
   - Most common segments

4. **User feedback**
   - Satisfaction scores
   - Bug reports
   - Feature requests

---

### 6.3 Rollback Procedure

**Trigger conditions:**
- Error rate >5%
- ROAS degradation >10%
- Critical bugs reported

**Rollback steps:**
1. Switch traffic back to v1.0 system
2. Investigate root cause
3. Fix issue
4. Re-test in staging
5. Restart rollout

---

## 7. Success Metrics

### 7.1 Quantitative Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| ROAS lift improvement | Generic only | +20% | A/B test: context-aware vs generic |
| Feature coverage | 60% | 100% | % of features that map to templates |
| Query latency | N/A | <100ms | Median query response time |
| Recommendation quality | Subjective | Objective scoring | Priority score alignment with expert assessment |
| Fallback rate | N/A | <20% | % queries using Level 3-4 |
| Sample size adequacy | N/A | >60% | % segments at Level 1-2 |

---

### 7.2 Qualitative Metrics

**User feedback:**
- "Recommendations are more actionable"
- "ROAS lift numbers are clear"
- "Context-specific patterns make sense"
- "Easier to prioritize changes"

**System quality:**
- Clearer documentation
- Better error messages
- More predictable behavior
- Easier to extend

---

## 8. Risk Mitigation

### 8.1 Identified Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Insufficient data for segmentation | High | Medium | Multi-level fallback system |
| Feature mapping gaps | Medium | Low | Comprehensive audit + placeholder expansion |
| Performance degradation | Medium | Low | Benchmarking + optimization |
| User resistance to change | Low | Medium | Gradual rollout + training |
| Breaking existing workflows | High | Low | Backward compatibility layer |
| Statistical false positives | Medium | Medium | Chi-square testing + confidence thresholds |

---

### 8.2 Mitigation Strategies

**1. Insufficient data risk**
- **Strategy:** Multi-level granularity with automatic fallback
- **Implementation:** Always have Level 4 (generic) as safety net
- **Validation:** Test with small datasets to ensure fallback works

**2. Feature mapping gap risk**
- **Strategy:** Comprehensive audit before implementation
- **Implementation:** Create catch-all placeholder for complex features
- **Validation:** 100% feature mapping coverage test

**3. Performance risk**
- **Strategy:** Early benchmarking and optimization
- **Implementation:** Profile bottlenecks, cache segment index
- **Validation:** Performance tests with large datasets

**4. User resistance risk**
- **Strategy:** Involve users early, provide training
- **Implementation:** Beta testing, documentation, tutorials
- **Validation:** User feedback sessions

**5. Breaking changes risk**
- **Strategy:** Backward compatibility layer
- **Implementation:** Auto-migrate v1.0 → v2.0, deprecation timeline
- **Validation:** Test existing workflows still work

**6. Statistical false positives risk**
- **Strategy:** Conservative thresholds, significance testing
- **Implementation:** Chi-square p-value <0.05, minimum prevalence 10%
- **Validation:** Statistical tests, expert review

---

## Appendix

### A. Related Documents

1. **Input/Output Schema Design** (`docs/input_output_schema_design.md`)
   - Detailed schema specifications
   - Validation rules
   - Example data

2. **Context-Aware Recommendations Design** (`docs/context_aware_recommendations_design.md`)
   - Granularity levels
   - Pattern detection algorithms
   - Query API design

3. **Ad Miner Analysis** (`docs/ad_miner_analysis.md`)
   - Problem analysis
   - Format assessment
   - Initial recommendations

### B. File Structure

```
docs/
├── ad_miner_improvement_plan.md (this file)
├── input_output_schema_design.md
├── context_aware_recommendations_design.md
└── ad_miner_analysis.md

src/meta/ad/miner/
├── validation/
│   ├── input_validator.py (NEW)
│   └── output_validator.py (NEW)
├── segmentation.py (NEW)
├── queries.py (NEW)
├── output/
│   ├── writer.py (NEW)
│   └── markdown_generator.py (NEW)
├── data/
│   └── loader.py (UPDATED)
├── recommendations/
│   ├── rule_engine.py (UPDATED)
│   ├── md_io.py (UPDATED)
│   └── prompt_formatter.py (UPDATED)
└── utils/
    └── compatibility.py (NEW)
```

### C. Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1. Fix output issues | 1 week | ROAS lift numbers, simplified MD |
| 2. Schema validation | 1-2 weeks | Input/output validators |
| 3. Metadata columns | 1 week | CSV with metadata, backfill script |
| 4. Segmentation | 2-3 weeks | SegmentAnalyzer, pattern detection |
| 5. JSON output | 1-2 weeks | JSON v2.0 format, MD generator |
| 6. Query API | 1-2 weeks | Fallback logic, CLI updates |
| 7. Feature mapping | 1 week | Complete feature coverage |
| 8. Compatibility | 1-2 weeks | v1.0 migration layer |
| 9. Testing | 1 week | Test suite, benchmarks |
| 10. Documentation | 1 week | Complete docs, examples |
| 11. Rollout | 2 weeks | Deployment, monitoring |

**Total:** 6-8 weeks

---

## Conclusion

This improvement plan transforms the ad miner from a generic recommendation system into a **context-aware, data-driven insight engine** that delivers:

1. **Concrete impact numbers** ("2.8x higher ROAS")
2. **Context-aware recommendations** (goal/product/branch specific)
3. **Structured, consumable format** (JSON + MD)
4. **Complete feature coverage** (100% mapping)
5. **Graceful degradation** (multi-level fallback)

**Expected outcome:** +20% ROAS improvement from context-aware recommendations, with clearer priorities and better usability for both humans and machines.

**Next steps:** Review and approve this plan, then begin Phase 1 implementation.
