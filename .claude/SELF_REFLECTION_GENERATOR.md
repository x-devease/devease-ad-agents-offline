# Claude Self-Reflection Framework - Adset Generator

## Purpose
Ensure any changes made to the audience configuration generator align with the core goals and constraints.

---

## Quick Reference (Claude Cheat Sheet)

### DO
- **Validate recommendations beat or match historical performance**
- Generate audience configuration strategies (regions, ages, creative formats, audience types)
- Use rules-based, transparent logic (KISS principle)
- Calculate headroom before recommending scale-up or new audiences
- Output recommendations with confidence + evidence
- Segment by geography, audience type, creative format
- Maintain priority scoring (CRITICAL > HIGH > MEDIUM > LOW)
- Use conservative estimates for opportunity values
- Respect platform-specific targeting capabilities
- Use `# pylint: disable` or `# type: ignore` ONLY when absolutely necessary

### DON'T
- Implement black-box ML for strategy generation
- Add budget allocation controls (that's in allocator module)
- Recommend configurations without performance validation
- Over-engineer for theoretical perfection
- Ignore platform targeting constraints (Meta vs Google vs TikTok)
- Use synthetic data for evaluation
- Ignore saturation warnings
- Make aggressive recommendations that risk production
- **Claim improvements without historical baseline comparison**
- **Use `# pylint: disable` instead of fixing the issue**
- **Ignore pylint warnings without proper justification**

---

## Repo Goals

### 1. Primary Goal
**Generate strategies for creating adset audience configurations (regions, ages, creative formats and other audience types) given each platform's targeting settings.**

### 2. Actual Objective (CRITICAL)
**Analyze customer data and deliver audience configurations that can beat or match historical performance.**

- Focus on practical improvement over theoretical perfection
- Compare performance against historical baseline
- **Every recommendation must answer: "Will this perform better than what we've seen?"**
- Deliver working configurations, not optimal ones

### 3. Performance Validation Requirement
**All recommendations must be validated against historical performance data.**

- **New audience configurations**: Must show similar segments performed well historically
- **Scale-up recommendations**: Must prove current performance beats baseline
- **Mistake detection**: Must show current config underperforms vs historical averages
- **Calibrate predictions**: Cap at historical 95th percentile (mean + 2*std)
- **No look-ahead bias**: Use only data available at decision time

### 4. Configuration Strategy
**Recommend audience configurations across multiple dimensions:**

- **Geography**: Which countries/regions to target (based on historical performance by geo)
- **Age Ranges**: Optimal age targeting (e.g., 18-38, 25-45, 35-55)
- **Audience Type**: Lookalike, Interest, Broad (based on historical ROAS by type)
- **Creative Format**: Video, Image, UGC (based on historical segment performance)
- **Combinations**: Which segments (geo × audience × creative) perform best

### 5. Platform-Aware Design
**Each platform has different targeting capabilities - recommendations must respect these.**

- **Meta Ads**: Age ranges, LAL percentages, interest targeting, geo targeting
- **Google Ads**: Demographics, in-market audiences, affinity audiences
- **TikTok Ads**: Age ranges, interests, behaviors
- Never recommend targeting options not available on the platform

### 6. Headroom Validation
**All scale-up or new audience recommendations must be validated against headroom limits.**

- Safe headroom: budget up to frequency 2.5 (optimal)
- Max headroom: budget up to frequency 4.0 (diminishing returns)
- Never recommend launching new audiences without market capacity
- Prevent recommending over-saturated audience configurations

### 7. Evidence-Based Recommendations
**Every recommendation includes supporting evidence with historical context.**

- Current values (age_min, age_max, countries, custom_audiences_count)
- Historical baseline performance for comparison
- Thresholds used for detection
- Estimated improvements with rationale
- Segment performance data
- Notes on uncertainty or assumptions

### 8. Confidence Scoring
**All recommendations include confidence: HIGH/MEDIUM/LOW.**

- HIGH: Clear violation of threshold OR strong historical evidence
- MEDIUM: Estimate based on typical patterns OR limited historical data
- LOW: Insufficient data OR high uncertainty OR no historical baseline
- Confidence must be justified in evidence

### 9. Conservative Approach
**Value reliability over aggressive optimization.**

- Use conservative estimates for opportunity values
- Prefer "test" over "commit" (e.g., A/B test ranges)
- Avoid over-claiming improvement potential
- Protect customer budget from bad recommendations
- **If uncertain, recommend small test vs full rollout**

### 10. Rules-Based (KISS Principle)
**This is a rules-based system, not an ML model.**

- No training data required
- No model overfitting concerns
- No hyperparameter tuning
- Fully interpretable logic
- Changes are immediate (no retraining needed)

---

## Performance Validation Rules

### For New Audience Configurations
```python
# ✅ CORRECT
# Validate against historical segment performance
if recommending_new_config:
    # Find similar historical segments
    historical_roas = get_similar_segments_performance(
        geography=geo,
        audience_type=audience_type,
        creative_format=creative
    )

    if historical_roas > baseline_roas * 1.2:
        recommend(config, evidence={
            'historical_roas_similar_segments': historical_roas,
            'baseline_roas': baseline_roas,
            'expected_to_beat_history': True
        })
    else:
        do_not_recommend(config, reason="No historical evidence this will beat baseline")

# ❌ WRONG
# Recommend without historical validation
if segment_looks_good:
    recommend(config)  # No proof it will beat history
```

### For Scale-Up Recommendations
```python
# ✅ CORRECT
# Prove current performance beats history
if current_roas > historical_baseline_roas:
    if headroom > 0:
        recommend_scale(evidence={
            'current_roas': current_roas,
            'historical_baseline': historical_baseline_roas,
            'beats_history_by': current_roas / historical_baseline_roas
        })

# ❌ WRONG
# Scale without beating history
if headroom > 0:
    recommend_scale()  # May be scaling a mediocre performer
```

### For Mistake Detection
```python
# ✅ CORRECT
# Show underperformance vs historical average
if current_roas < historical_segment_average * 0.7:
    flag_issue(evidence={
        'current_roas': current_roas,
        'segment_historical_avg': historical_segment_average,
        'underperforming_by_pct': (1 - current_roas / historical_segment_average) * 100
    })

# ❌ WRONG
# Flag issue without context
if roas < 1.0:
    flag_issue()  # May be normal for this segment
```

---

## Recommendation Types

| Type | Priority | Description | Action | Performance Validation Required |
|------|----------|-------------|--------|-------------------------------- |
| **launch_new** | HIGH | High-potential untested audience configuration | Create new adset with suggested config | Similar segments beat historical baseline by >20% |
| **scale_up** | HIGH | Underfunded winning configuration | Increase budget (up to headroom limit) | Current ROAS > historical baseline × 1.5 |
| **wasting** | CRITICAL | Low ROAS + high spend | PAUSE immediately | ROAS < historical segment average × 0.5 |
| **too_broad** | MEDIUM | Age range too wide for optimization | Test narrower segments | Narrower segments historically outperform |
| **missing_lal** | MEDIUM | No LALs + low ROAS | Create LAL 1% from best customers | LALs historically 2-3x ROAS vs interest |
| **oversaturated** | MEDIUM | Frequency > 4.0 + low ROAS | Reduce budget to target freq 2.5 | Frequency-ROAS curve shows diminishing returns |
| **optimize_or_pause** | HIGH | Underperforming with low confidence | Review or pause | Underperforms historical baseline significantly |

---

## Configuration Dimensions (with Historical Validation)

### 1. Geography
- **Input**: `adset_targeting_countries` (e.g., "['US']", "['US', 'CA']")
- **Output**: Recommended countries/regions to test
- **Validation**:
  - Analyze historical ROAS by country
  - Recommend high-performing regions for expansion
  - Flag underperforming regions to pause
  - **Rule**: Only recommend countries where historical ROAS > baseline

### 2. Age Targeting
- **Input**: `adset_targeting_age_min`, `adset_targeting_age_max`
- **Output**: Recommended age ranges to test
- **Validation**:
  - Age range > 30 years → too_broad (test narrower segments)
  - Use 20-year ranges for testing (e.g., 18-38, 25-45)
  - A/B test multiple ranges before committing
  - **Rule**: Only recommend age ranges with historical precedent

### 3. Audience Type
- **Input**: `adset_targeting_custom_audiences_count`, age_range
- **Output**: Lookalike vs Interest vs Broad
- **Validation**:
  - Historical ROAS by audience type
  - LALs typically 2-3x vs interest targeting
  - Broad requires high historical ROAS to justify
  - **Rule**: Recommend LAL if historical LAL ROAS > current Interest ROAS × 1.5

### 4. Creative Format
- **Input**: `video_30_sec_watched_actions`, `video_p100_watched_actions`
- **Output**: Video vs Image vs UGC
- **Validation**:
  - Segment by creative format to find winners
  - Historical ROAS by format
  - **Rule**: Recommend format with highest historical ROAS for segment

---

## Pre-Change Reflection Checklist

Before making any code change to the generator, Claude must verify:

### Goal Alignment
- [ ] Does this support audience configuration strategy generation?
- [ ] **Does this validate against historical performance?**
- [ ] **Does this prove recommendations will beat or match history?**
- [ ] Does this respect platform-specific targeting capabilities?
- [ ] Does this include segmentation analysis?
- [ ] Does this use conservative, rules-based logic?
- [ ] Does this validate against headroom limits?
- [ ] Does this include confidence + evidence?
- [ ] Does this suggest testing over committing?

### Anti-Goal Check
- [ ] Does NOT introduce black-box ML for strategy?
- [ ] Does NOT add budget allocation features (that's in allocator module)?
- [ ] **Does NOT recommend without historical validation?**
- [ ] **Does NOT ignore historical baseline comparison?**
- [ ] Does NOT ignore platform constraints?
- [ ] Does NOT over-claim opportunity values?
- [ ] Does NOT skip headroom validation?
- [ ] Does NOT recommend unavailable targeting options?
- [ ] Does NOT use synthetic data for evaluation?

### Change Scope
- [ ] Is this the minimum change needed?
- [ ] Preserves existing behavior where appropriate?
- [ ] Maintains backward compatibility?
- [ ] All tests pass?

### Reliability Check
- [ ] Is this safe for production audiences?
- [ ] **Does this recommendation beat historical performance?**
- [ ] What happens if this recommendation is wrong?
- [ ] Are estimates conservative or aggressive?
- [ ] Is there a rollback path?
- [ ] Does this respect platform API limits?

### Code Pattern Check
- [ ] No hard-coded customer/platform names?
- [ ] No hard-coded file paths?
- [ ] No `# pylint: disable` or `# type: ignore` suppressions?
- [ ] Uses path abstraction helpers?

---

## Decision Path Examples

### Example 1: ML-Based Strategy Generation
**Proposed change**: "Use reinforcement learning to learn optimal audience configs"

**Reflection**:
- ❌ Violates rules-based, transparent approach
- ❌ Black-box, not interpretable
- ❌ Requires extensive training
- **Decision**: DECLINE. Use simple rules with historical analysis.

### Example 2: Recommendation Without Historical Validation
**Proposed change**: "Recommend launching US + 25-45 + Lookalike audience"

**Reflection**:
- ❌ No historical performance data for this combination
- ❌ No proof it will beat baseline
- **Decision**: DECLINE. Must show similar segments performed well.

### Example 3: Historical Baseline Comparison
**Proposed change**: "Add historical average ROAS by segment to evidence"

**Reflection**:
- ✅ Enables performance validation
- ✅ Shows if recommendation beats history
- **Decision**: PROCEED. Essential for goal.

### Example 4: Calibrating Predictions to History
**Proposed change**: "Cap predictions at historical 95th percentile"

**Reflection**:
- ✅ Prevents unrealistic claims
- ✅ Aligns with "beat history, not perfection" goal
- **Decision**: PROCEED. Already in code.

### Example 5: Segment-Based Recommendations with History
**Proposed change**: "Analyze geo × audience × creative segments, recommend top performers"

**Reflection**:
- ✅ Uses historical data to find winners
- ✅ Recommendations based on actual performance
- **Decision**: PROCEED. Core to the repo's purpose.

### Example 6: Ignoring Platform Constraints
**Proposed change**: "Recommend LAL 1% for all platforms"

**Reflection**:
- ❌ Not all platforms support LAL
- ❌ Google uses in-market/affinity, not LAL
- **Decision**: DECLINE. Make platform-aware.

### Example 7: Headroom for New Audiences
**Proposed change**: "Check market headroom before recommending new audience launch"

**Reflection**:
- ✅ Prevents over-saturation
- ✅ Validates market capacity
- **Decision**: PROCEED. Essential for reliability.

### Example 8: Performance Threshold for New Audiences
**Proposed change**: "Only recommend new audience if similar segments beat baseline by 20%"

**Reflection**:
- ✅ Ensures recommendations beat history
- ✅ Conservative threshold
- **Decision**: PROCEED. Aligns with performance goal.

### Example 9: Budget Allocation Feature
**Proposed change**: "Add automatic budget redistribution across audiences"

**Reflection**:
- ❌ Outside scope (that's in allocator module)
- ❌ Not about configuration strategy
- **Decision**: DECLINE. That's for allocator module.

### Example 10: Single Configuration vs Testing
**Proposed change**: "Recommend specific age range vs A/B test multiple ranges"

**Reflection**:
- ❌ Single range assumes knowledge
- ✅ Testing is more conservative
- ✅ Testing validates against actual performance
- **Decision**: DECLINE single recommendation. Use A/B test approach.

---

## Red Flags (Stop and Reconsider)

### 🚩 Performance Violations (CRITICAL)
1. **Recommending configurations without historical validation**
2. **Ignoring historical baseline comparison**
3. **Claiming improvements without segment evidence**
4. **Scaling underperforming audiences (ROAS < baseline)**
5. **Launching new audiences without proving similar segments work**

### 🚩 Design Violations
6. Adding ML models for configuration strategy
7. Adding budget allocation features (that's in allocator module)
8. Ignoring platform-specific targeting capabilities
9. Recommending unavailable targeting options
10. Breaking configuration generation focus

### 🚩 Reliability Violations
11. Recommending new audiences without headroom check
12. Aggressive opportunity estimates (2x+, 3x+)
13. Ignoring saturation warnings
14. Missing confidence scores
15. Missing evidence dictionaries
16. Over-claiming without segment data

### 🚩 Code Quality Violations
17. Over-engineering simple rules
18. Hard-coding platform-specific logic without abstraction
19. Adding unnecessary dependencies
20. Breaking backward compatibility
21. Skipping tests for new configuration rules
22. Using `# pylint: disable` instead of fixing the underlying issue
23. Ignoring pylint warnings without justification
24. Adding arbitrary thresholds (e.g., `--fail-under`) just to make CI pass at a low score
25. Lowering quality standards instead of improving code coverage
26. Adding feature branches to CI workflow triggers (CI runs on main/PRs only)
27. Modifying README structure/style when updating automatically (keep consistent format)

---

## Key File Locations

### Generator Core Files
| Component | File | Purpose |
|-----------|------|---------|
| **Core** | `src/adset/generator/core/recommender.py` | Base recommender class |
| **Detection** | `src/adset/generator/detection/mistake_detector.py` | Detect issues in configs |
| **Sizing** | `src/adset/generator/analyzers/opportunity_sizer.py` | Calculate opportunity size |
| **Shopify** | `src/adset/generator/analyzers/shopify_analyzer.py` | Shopify revenue analysis |
| **Generation** | `src/adset/generator/generation/audience_recommender.py` | Generate recommendations |
| **Aggregator** | `src/adset/generator/generation/audience_aggregator.py` | Aggregate recommendations |
| **Compatibility** | `src/adset/generator/generation/creative_compatibility.py` | Creative x audience |
| **Segmentation** | `src/adset/generator/segmentation/segmenter.py` | Segment analysis |
| **Constraints** | `src/adset/generator/analyzers/advantage_constraints.py` | Competitive advantages |

---

## Core Principle Summary

**"Every recommendation must answer: Will this perform better than what we've seen historically?"**

If the answer is "I don't know" or "maybe," then the recommendation should be framed as a test, not a commitment.
