# Claude Self-Reflection Framework

## Purpose
Ensure any changes made by Claude align with the core goals and constraints of this comprehensive adset management system.

---

## Quick Reference (Claude Cheat Sheet)

### DO
- **Allocator**: Use rules-based budget allocation with Bayesian optimization
- **Generator**: Generate audience configurations that beat/match historical performance
- Use `config/{customer}/{platform}.yaml` for params
- Use `datasets/{customer}/{platform}/` for data
- Output actions as YAML with confidence + evidence
- Require high confidence before param updates
- Run all CI workflows before committing
- Use `customer_paths.py` for path access
- Evaluate on real data only
- Use `# pylint: disable` or `# type: ignore` ONLY when absolutely necessary

### DON'T
- Hard-code paths or parameters
- Share params across customers/platforms
- Update params on low-confidence data
- Use synthetic data for evaluation claims
- Risk real money on unproven features
- Assume platform is "meta" only
- Break path abstraction
- Create PR-specific documentation (no TODO.md, VALIDATION.md per PR)
- Use `# pylint: disable` or `# type: ignore` to suppress warnings when fixing is possible
- Leave temporary scripts in working directory when pushing

---

## System Architecture

This repository combines TWO distinct adset management capabilities:

### 1. Adset Allocator (`src/adset/allocator/`)
**Purpose:** Budget allocation - distributes budget across adsets based on performance

**Primary Goal:** Allocate budgets to maximize ROAS while maintaining stability

**Key Features:**
- 42+ decision rules for budget adjustments
- Bayesian optimization of 60+ parameters
- Monthly budget tracking with rollover
- Shopify ROAS integration for validation
- Safety rules (freeze low performers, cap budget)

### 2. Adset Generator (`src/adset/generator/`)
**Purpose:** Audience configuration - generates audience strategies and configurations

**Primary Goal:** Generate audience configurations (regions, ages, audience types) that beat or match historical performance

**Key Features:**
- Audience configuration strategies (geo, age, creative format)
- Mistake detection (identify suboptimal configs)
- Opportunity sizing (headroom calculation)
- Creative x audience compatibility analysis
- Historical performance validation

---

## File Location Rules

| Type | Location | Example |
|------|----------|---------|
| Configs | `config/{customer}/{platform}/` | `config/moprobo/meta/` |
| Raw Data | `datasets/{customer}/{platform}/raw/` | `datasets/moprobo/meta/raw/` |
| Features | `datasets/{customer}/{platform}/features/` | `datasets/moprobo/meta/features/` |
| Results | `results/{customer}/{platform}/` | `results/moprobo/meta/` |

**Rule**: All data access must use `src/utils/customer_paths.py` helper functions. Never hard-code paths.

---

## Repo Goals

### Allocator Goals
1. **Primary**: Allocate budgets across adsets to maximize ROAS
2. **Actual Objective**: Beat or match historical performance, not achieve perfection
3. **Method**: Rules-based system with Bayesian-optimized parameters
4. **Validation**: Time-series cross-validation to prevent overfitting
5. **Safety**: Freeze low performers, cap budgets, track monthly spend

### Generator Goals
1. **Primary**: Generate strategies for creating adset audience configurations (regions, ages, creative formats, audience types)
2. **Actual Objective**: Deliver audience configurations that beat or match historical performance
3. **Method**: Rules-based, transparent logic with historical validation
4. **Validation**: All recommendations must validate against historical baseline
5. **Evidence**: Every recommendation includes confidence + historical evidence

---

## Performance Validation Rules

### Allocator Validation
```python
# ✅ CORRECT
# Use time-series cross-validation
if recommending_budget_adjustment:
    historical_performance = get_historical_baseline(adset)
    simulated_performance = simulate_with_params(
        params,
        time_series_cv=True  # Prevent look-ahead bias
    )

    if simulated_performance > historical_performance:
        recommend(params, evidence={
            'historical_roas': historical_performance,
            'simulated_roas': simulated_performance,
            'expected_improvement': (simulated_performance / historical_performance) - 1
        })
```

### Generator Validation
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
```

---

## Common Mistakes to Avoid

### 🚩 Allocator Violations
1. **Ignoring monthly budget caps** - Track cumulative spend
2. **Aggressive scaling** - Use conservative factors (0.95 multiplier)
3. **Not freezing low performers** - Safety first
4. **Look-ahead bias in tuning** - Use time-series CV
5. **Overfitting parameters** - Validate on holdout periods

### 🚩 Generator Violations
1. **Recommending without historical validation** - Must prove it beats history
2. **Ignoring platform constraints** - Meta vs Google vs TikTok targeting differences
3. **Missing headroom checks** - Don't oversaturate
4. **Aggressive opportunity estimates** - Use conservative 95th percentile
5. **Single recommendations vs testing** - Recommend A/B tests

### 🚩 Shared Violations (Both)
6. **Hard-coding paths** - Use path abstraction
7. **Ignoring CI workflows** - All tests must pass
8. **Suppressing warnings unnecessarily** - Fix root cause
9. **Synthetic data for evaluation** - Use real customer data
10. **Breaking backward compatibility** - Preserve existing behavior

---

## Pre-Change Reflection Checklist

Before making any code change, Claude must verify:

### Goal Alignment
- [ ] Does this support allocator OR generator goals?
- [ ] **Does this validate against historical performance?**
- [ ] **Does this prove recommendations will beat or match history?**
- [ ] Does this use conservative, rules-based logic?
- [ ] Does this respect platform constraints?
- [ ] Does this include confidence + evidence?

### Code Quality
- [ ] All tests pass?
- [ ] No unnecessary `# pylint: disable`?
- [ ] No unnecessary `# type: ignore`?
- [ ] CI workflows pass?
- [ ] Coverage maintained or improved?
- [ ] Path abstraction preserved?

### Safety
- [ ] Is this safe for production budgets?
- [ ] What happens if this recommendation is wrong?
- [ ] Are estimates conservative or aggressive?
- [ ] Is there a rollback path?
- [ ] Does this respect platform API limits?

---

## Key File Locations

### Allocator Files
| Component | File | Purpose |
|-----------|------|---------|
| **Engine** | `src/adset/allocator/engine.py` | Main allocator interface |
| **Rules** | `src/adset/lib/decision_rules.py` | 42+ decision rules |
| **Safety** | `src/adset/lib/safety_rules.py` | Freeze/cap logic |
| **Models** | `src/adset/lib/models.py` | Data models |
| **Tuning** | `src/optimizer/lib/bayesian_tuner.py` | Bayesian optimization |

### Generator Files
| Component | File | Purpose |
|-----------|------|---------|
| **Core** | `src/adset/generator/core/recommender.py` | Base recommender |
| **Detection** | `src/adset/generator/detection/mistake_detector.py` | Mistake detection |
| **Sizing** | `src/adset/generator/analyzers/opportunity_sizer.py` | Headroom calculation |
| **Generation** | `src/adset/generator/generation/audience_recommender.py` | Generate recommendations |
| **Segmentation** | `src/adset/generator/segmentation/segmenter.py` | Segment analysis |

### Shared Files
| Component | File | Purpose |
|-----------|------|---------|
| **Config** | `src/config/manager.py` | Config loading |
| **Paths** | `src/utils/customer_paths.py` | Path abstraction |
| **Logging** | `src/utils/logger_config.py` | Logging setup |
| **Shopify** | `src/integrations/shopify/` | Shopify integration |

---

## Core Principle Summary

### Allocator
**"Allocate budget to maximize ROAS, not to achieve perfection"**
- Beat or match historical performance
- Use conservative safety factors
- Validate with time-series cross-validation

### Generator
**"Every recommendation must answer: Will this perform better than what we've seen historically?"**
- If answer is "I don't know" → frame as test, not commitment
- Validate all recommendations against historical baseline
- Use conservative estimates (95th percentile cap)

---

## Red Flags (Stop and Reconsider)

### 🚩 Performance Violations (CRITICAL)
1. **Allocator**: Simulating without time-series CV (look-ahead bias)
2. **Allocator**: Aggressive scaling without safety checks
3. **Generator**: Recommending without historical validation
4. **Generator**: Ignoring historical baseline comparison
5. **Both**: Claiming improvements without evidence

### 🚩 Design Violations
6. **Allocator**: Adding black-box ML for allocation
7. **Generator**: Adding black-box ML for configuration
8. **Both**: Ignoring platform-specific capabilities
9. **Both**: Hard-coding customer/platform logic
10. **Both**: Breaking allocator/generator separation

### 🚩 Code Quality Violations
11. **Both**: Using `# pylint: disable` to suppress fixable warnings
12. **Both**: Using `# type: ignore` to suppress fixable type errors
13. **Both**: Hard-coding paths instead of using abstraction
14. **Both**: Leaving temporary scripts in repo
15. **Both**: Adding feature branches to CI triggers
16. **Both**: Lowering coverage standards instead of improving
17. **Both**: Modifying README structure/style (keep consistent)
18. **Both**: Risking real money on unvalidated features

---

## Decision Path Examples

### Example 1: ML-Based Allocation
**Proposed**: "Use reinforcement learning for budget allocation"

**Reflection**:
- ❌ Black-box, not interpretable
- ❌ Violates rules-based approach
- **Decision**: DECLINE. Use Bayesian-tuned rules.

### Example 2: ML-Based Configuration
**Proposed**: "Use reinforcement learning to learn optimal audience configs"

**Reflection**:
- ❌ Violates rules-based, transparent approach
- ❌ Black-box, not interpretable
- **Decision**: DECLINE. Use simple rules with historical analysis.

### Example 3: Recommendation Without Historical Validation
**Proposed**: "Recommend launching US + 25-45 + Lookalike audience"

**Reflection**:
- ❌ No historical performance data for this combination
- ❌ No proof it will beat baseline
- **Decision**: DECLINE. Must show similar segments performed well.

### Example 4: Hard-coded Platform
**Proposed**: "Add meta-specific logic to allocator"

**Reflection**:
- ❌ Not all platforms are Meta
- ❌ Should abstract platform differences
- **Decision**: DECLINE. Make platform-agnostic.

### Example 5: Pylint Disable
**Proposed**: "Add `# pylint: disable=unused-argument`"

**Reflection**:
- ❌ Can fix by renaming variable or using `_`
- ❌ Should fix underlying issue
- **Decision**: DECLINE. Fix the code instead.

---

## Commit Message Guidelines

### Allocator Features
```
feat: allocator - brief description

Details of the budget allocation change.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### Generator Features
```
feat: generator - brief description

Details of the audience configuration change.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

### Shared Components
```
feat: shared - brief description

Details of the change affecting both allocator and generator.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```
