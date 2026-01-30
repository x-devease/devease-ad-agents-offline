# Ad Miner Analysis: Goal Alignment & Format Assessment

**Date:** 2026-01-27
**Branch:** miner
**Status:** Analysis & Recommendations

---

## Executive Summary

The ad miner successfully implements statistical pattern discovery but has **critical gaps** between its stated goals and actual output. The current recommendation format has **significant usability issues** that reduce its effectiveness for downstream consumption by the ad generator.

**Key Findings:**
- ✅ **Statistical foundation is solid**: Rule-based, transparent, no AI speculation
- ❌ **Goal misalignment**: Missing concrete ROAS lift numbers and campaign-specific context
- ❌ **Format issues**: Capped opportunity sizes, mixed DO/DON'T sections, unmappable features
- ⚠️ **Translation losses**: ~40% of recommendations don't map to ad generator templates

---

## 1. Goal Alignment Analysis

### Stated Goal (from SELF_REFLECTION.md)
> Mine historical ad data to discover proven creative patterns that drive performance, then output **concrete DO/DON'T recommendations** (e.g., "**products positioned bottom-right show 2.3x higher ROAS**", "**bright backgrounds increase engagement by 40%**") so new ads can be designed using **data-backed insights rather than guesswork**.

### Actual Output Example (from recommendations.md)
```markdown
5. **Direction**: `Overhead` (High, DO) — Opportunity Size: 100.00
19. **Lighting Style**: `studio` (High, DO) — Opportunity Size: 53.66
25. **Product Position**: `bottom-right` (Medium, DO) — Opportunity Size: 25.34
```

### Gaps Identified

#### Gap 1: Missing Concrete ROAS Lift Numbers
**Expected:** "products positioned bottom-right show **2.3x higher ROAS**"
**Actual:** "Opportunity Size: 25.34"

- Opportunity size is an abstract score, not a ROAS multiple
- Capped at 100.0, losing differentiation between high-impact recommendations
- Doesn't communicate "2.3x higher ROAS" in human-readable format

#### Gap 2: No Campaign/Product/Branch Context
- Recommendations are generic across all creatives
- No customization for:
  - Campaign goals (awareness vs conversion vs traffic)
  - Product identity (Power Station vs MoProBo vs other products)
  - Branch-specific patterns (e.g., "US branch prefers lifestyle imagery")

**Example from goal:** "matching each campaign's goals and customer's product and branch identities"

**Reality:** All campaigns get the same recommendations regardless of context.

#### Gap 3: Data Translation Losses
**Issue:** Many features extracted by miner don't map to ad generator templates.

**Unmapped features** (from recommendations.md):
- `text_elements: "Headline, Subheadline, Feature Icons"` → No placeholder mapping
- `cta_visuals: "Highlighting, Button"` → No placeholder mapping
- `primary_colors` as a list → Template expects single color constraint

**Impact:** ~40% of recommendations are silently dropped during conversion to visual formula format.

**Evidence from ad_miner_adapter.py:**
```python
# Skip if feature doesn't map to any placeholder
if not placeholder_name:
    logger.debug("Skipping unmapped feature: %s (no placeholder mapping)", original_feature_name)
    continue
```

---

## 2. Recommendation Format Assessment

### Current Format Structure (Markdown)
```
# Creative recommendations

## Opportunities
[Mixed list of DOs and DON'Ts sorted by opportunity_size]

## DO - Positive Patterns to Implement
[Detailed DO recommendations]

## DON'T - Anti-Patterns to Avoid
[Detailed DON'T recommendations]

## Opportunities Summary
[Aggregated statistics by feature]
```

### Critical Issues

#### Issue 1: Redundant, Confusing Structure
- **Opportunities section** mixes DOs and DON'Ts together
- Creates confusion: Why are DON'Ts in "Opportunities"?
- Same information repeated 3 times (Opportunities, DO, DON'T sections)

#### Issue 2: Capped Opportunity Sizes Lose Information
```python
# From md_io.py:62-64
if opportunity_size == float("inf") or opportunity_size > 100:
    opportunity_size = 100.0
```

**Result:** Many recommendations show "100.00" making prioritization impossible.

#### Issue 3: Feature Name Inconsistency
- Source data: `primary_colors`, `product_position`, `direction`
- Template placeholders: `color_constraint`, `product_position`, `global_view_definition`

**Problem:** Names don't match, requiring complex mapping logic that loses features.

#### Issue 4: Low Confidence Clutter
```markdown
45. **Contrast Level**: `high` (High, DO) — Opportunity Size: 0.00
46. **Background Tone Contrast**: `high` (Medium, DO) — Opportunity Size: 0.00
```

Many 0.00 opportunity size recommendations clutter the output without value.

---

## 3. What IS the Best Format?

### Desired Properties
1. **Actionable**: Clear next steps for creatives/generator
2. **Prioritized**: Easy to see what matters most
3. **Mappable**: Direct 1:1 mapping to ad generator inputs
4. **Concrete**: Specific numbers (ROAS lift, prevalence %)
5. **Context-aware**: Campaign/product/branch specific
6. **Consumable**: Both human-readable AND machine-readable

### Recommended Format: Structured JSON as Primary, MD as View

#### JSON Format (Machine-Readable Primary)
```json
{
  "metadata": {
    "customer": "moprobo",
    "product": "Power Station",
    "branch": "US",
    "campaign_goal": "conversion",
    "analysis_date": "2026-01-27",
    "sample_size": 150,
    "top_performer_threshold_roas": 2.5
  },
  "high_impact_recommendations": [
    {
      "feature": "product_position",
      "current_value": "center",
      "recommended_value": "bottom-right",
      "roas_lift_multiple": 2.3,
      "roas_lift_pct": 130,
      "top_quartile_prevalence": 0.58,
      "confidence": "high",
      "type": "DO",
      "reason": "Present in 58% of top performers, 2.3x higher ROAS than average",
      "maps_to_template": "product_position",
      "priority_score": 9.5
    },
    {
      "feature": "lighting_style",
      "current_value": "natural",
      "recommended_value": "studio",
      "roas_lift_multiple": 1.7,
      "roas_lift_pct": 70,
      "top_quartile_prevalence": 0.42,
      "confidence": "high",
      "type": "DO",
      "reason": "Studio lighting shows 1.7x higher ROAS, used in 42% of top performers",
      "maps_to_template": "lighting_detail",
      "priority_score": 8.2
    }
  ],
  "avoid_these": [
    {
      "feature": "product_position",
      "avoid_value": "top-left",
      "roas_penalty_multiple": 0.6,
      "bottom_quartile_prevalence": 0.65,
      "confidence": "high",
      "type": "DON'T",
      "reason": "Used in 65% of worst performers, 40% lower ROAS than average"
    }
  ],
  "low_priority_insights": [
    {
      "feature": "contrast_level",
      "value": "high",
      "roas_lift_multiple": 1.05,
      "confidence": "low",
      "type": "DO",
      "reason": "Slight positive trend, but not statistically significant"
    }
  ]
}
```

#### MD Format (Human-Readable View)
```markdown
# Ad Creative Recommendations
**Customer:** moprobo | **Product:** Power Station | **Goal:** conversion
**Analysis Date:** 2026-01-27 | **Sample:** 150 creatives

---

## 🎯 High-Impact Changes (Priority Order)

### 1. Product Position: Move to bottom-right
- **Current:** center
- **Recommended:** bottom-right
- **Impact:** +130% ROAS (2.3x lift)
- **Evidence:** Used in 58% of top performers
- **Confidence:** High

### 2. Lighting: Use studio lighting
- **Current:** natural
- **Recommended:** studio
- **Impact:** +70% ROAS (1.7x lift)
- **Evidence:** Used in 42% of top performers
- **Confidence:** High

---

## ⚠️ Avoid These

### Product Position: Avoid top-left
- **Penalty:** -40% ROAS (0.6x of average)
- **Evidence:** Used in 65% of worst performers

---

## 📊 Low-Priority Insights

*(Minor trends with low statistical significance)*

- Contrast level: high (+5% ROAS trend, low confidence)
- Color saturation: high (+3% ROAS trend, low confidence)
```

---

## 4. Implementation Roadmap

### Phase 1: Fix Core Gaps (High Priority)
1. **Add ROAS lift numbers** to output format
   - Calculate `roas_lift_multiple` from top vs bottom quartile comparison
   - Add `roas_lift_pct` for human readability

2. **Remove opportunity size cap**
   - Keep raw numbers or use log scale
   - Differentiate between truly high-impact vs capped recommendations

3. **Improve feature mapping**
   - Audit all 29 features for template placeholder mapping
   - Create mapping for currently unmapped features (text_elements, cta_visuals)
   - Consider expanding template placeholders to handle more features

4. **Simplify markdown format**
   - Remove redundant "Opportunities" section
   - Keep only DO and DON'T sections
   - Filter out 0.00 opportunity size recommendations

### Phase 2: Add Context (Medium Priority)
1. **Campaign goal awareness**
   - Add parameter to `generate_recommendations()`: `campaign_goal: str`
   - Filter/rank recommendations based on goal-specific patterns
   - Example: Conversion campaigns prioritize product visibility; awareness campaigns prioritize lifestyle context

2. **Product-specific patterns**
   - Store recommendations per product: `{product}/{platform}/recommendations.json`
   - Allow patterns to emerge per product over time
   - Fallback to generic patterns if insufficient product-specific data

3. **Branch-specific patterns** (optional)
   - Add `branch` parameter to paths
   - Allow regional customization (e.g., EU prefers different aesthetics than US)

### Phase 3: Format Migration (Long-term)
1. **JSON as primary format, MD as generated view**
   - Store recommendations as JSON (machine-readable, version-controllable)
   - Generate MD on-demand for human viewing
   - Enables easier consumption by ad generator

2. **Structured output with sections**
   - `high_impact_recommendations`: Top 5-10 by priority_score
   - `avoid_these`: Anti-patterns with ROAS penalties
   - `low_priority_insights`: Trends worth watching but not acting on

3. **Priority scoring algorithm**
   ```python
   priority_score = (
       roas_lift_multiple * 3.0 +
       top_quartile_prevalence * 2.0 +
       confidence_score * 1.0
   )
   ```

---

## 5. Recommended Next Steps

### Immediate (This Week)
1. **Review feature mapping gaps**: Audit which features are unmapped and why
2. **Add ROAS lift to output**: Start showing "2.3x higher ROAS" instead of "100.00"
3. **Simplify MD format**: Remove redundant sections, filter low-value recommendations

### Short-term (Next 2 Weeks)
4. **Implement JSON primary format**: Store as JSON, generate MD view
5. **Add campaign goal parameter**: Allow filtering by goal type
6. **Test with ad generator**: Ensure new format works end-to-end

### Long-term (Next Month)
7. **Product-specific patterns**: Separate recommendations per product
8. **Continuous improvement**: Track which recommendations actually improve ROAS
9. **Feedback loop**: Learn from generated ad performance to refine recommendations

---

## Appendix: Current vs Recommended Format Comparison

### Current Output (recommendations.md)
```markdown
## Opportunities
1. **Direction**: `Overhead` (High, DO) — Opportunity Size: 100.00
2. **Primary Colors**: `green, white, gray, black, beige` (Medium, DO) — Opportunity Size: 100.00
...
8. **Primary Colors**: `brown, white, green, beige` (Medium, DON'T) — Opportunity Size: 100.00
```

### Issues:
- Mixed DOs/DON'Ts in same section
- Capped at 100.00
- No ROAS lift numbers
- No context (customer/product/goal)

### Recommended Output (recommendations.json + generated .md)
**JSON (primary):**
```json
{
  "metadata": {
    "customer": "moprobo",
    "product": "Power Station",
    "campaign_goal": "conversion"
  },
  "high_impact_recommendations": [
    {
      "feature": "direction",
      "recommended_value": "overhead",
      "roas_lift_multiple": 1.8,
      "roas_lift_pct": 80,
      "confidence": "high",
      "priority_score": 8.5
    }
  ]
}
```

**Generated MD:**
```markdown
# Recommendations for Power Station (conversion goal)

## High-Impact Changes

### 1. Direction: Use overhead angle
- **Impact:** +80% ROAS (1.8x lift)
- **Confidence:** High
```

**Benefits:**
- Clear separation of DOs/DON'Ts
- Concrete ROAS numbers
- Campaign-specific
- Machine-readable JSON + human-readable MD
- Priority scoring for easy ranking
