# Claude Self-Reflection Framework

## Purpose
Ensure any changes made align with the core goals and constraints of the **budget allocator**, **audience configuration generator**, **ad recommender** (creative scorer), and **ad generator** (creative image generation).

---

## Quick Reference (Claude Cheat Sheet)

### DO
- Use `config/{customer}/{platform}/` for params
- Use `datasets/{customer}/{platform}/` for data
- Output actions as YAML with confidence + evidence
- Require high confidence before param updates
- Run all CI workflows before committing
- Use `customer_paths.py` for path access (allocator/adset); use `recommender` utils and `generator` `Paths` for ad modules
- Evaluate on real data only
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
- **Use statistical patterns (lift analysis) for creative recommendations**
- **Hard-code pattern thresholds (top_pct=0.25, bottom_pct=0.25) to avoid data leakage**
- **Apply conservative impact estimates (50% of lift) for creative recommendations**
- **Require minimum thresholds: lift >= 1.5, prevalence >= 10% for patterns**
- **Use chi-square tests for statistical significance in creative patterns**

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
- Implement black-box ML for strategy generation
- Add budget allocation controls in generator (that's in allocator module)
- Add audience configuration controls in allocator (that's in generator module)
- Recommend configurations without performance validation
- Over-engineer for theoretical perfection
- Ignore platform targeting constraints (Meta vs Google vs TikTok)
- Ignore saturation warnings
- Make aggressive recommendations that risk production
- **Claim improvements without historical baseline comparison**
- **Use `# pylint: disable` instead of fixing the issue**
- **Ignore pylint warnings without proper justification**
- **Use ML models for creative pattern detection (statistics only)**
- **Tune creative pattern parameters on test/eval data (data leakage)**
- **Claim causation from correlation in creative recommendations**
- **Overpromise on creative impact estimates (use 50% conservative factor)**
- **Skip statistical significance testing for creative patterns**
- **Recommend creative patterns with low prevalence (< 10%)**
- **Use creative patterns with lift < 1.5x**
- **Ignore feature recommendations from creative scorer in generator**
- **Skip feature validation after image generation**

---

## File Location Rules

| Type | Location | Example |
|------|----------|---------|
| Configs | `config/{customer}/{platform}/` | `config/moprobo/meta/` |
| Raw Data | `datasets/{customer}/{platform}/raw/` | `datasets/moprobo/meta/raw/` |
| Features | `datasets/{customer}/{platform}/features/` | `datasets/moprobo/meta/features/` |
| Results | `results/{customer}/{platform}/` | `results/moprobo/meta/` |
| GPT-4 Configs (ad recommender) | `config/ad/recommender/gpt4/` | `config/ad/recommender/gpt4/features.yaml` |
| Creative Features | `src/ad/recommender/features/` | `src/ad/recommender/features/extract.py` |
| Creative Recommendations | `src/ad/recommender/recommendations/` | `src/ad/recommender/recommendations/rule_engine.py` |
| Recommender Output | `config/ad/recommender/{customer}/{platform}/{date}/` | `.../moprobo/meta/2026-01-26/recommendations.md` |
| Generator Config | `config/ad/generator/{customer}/{platform}/` | `config/ad/generator/moprobo/taboola/generation_config.yaml` |
| Generator Templates | `config/ad/generator/templates/{customer}/{platform}/` | `config/ad/generator/templates/moprobo/taboola/` |
| Generator Prompts | `config/ad/generator/prompts/{customer}/{platform}/{date}/{type}/` | `.../moprobo/taboola/2026-01-23/structured/` |
| Generator Output | `config/ad/generator/generated/{customer}/{platform}/{date}/{model}/` | `.../moprobo/taboola/2026-01-23/nano-banana-pro/` |

**Rule**: Allocator/adset generator use `src/utils/customer_paths.py`. Ad recommender uses `src/ad/recommender/utils/paths.py` and `config_manager.py`. Ad generator uses `src/ad/generator/core/paths.py`. Never hard-code paths.

---

## Code Patterns to Follow

### Path Access
```python
# ✅ CORRECT
from src.utils.customer_paths import get_raw_data_dir
raw_dir = get_raw_data_dir(customer="moprobo", platform="meta")

# ❌ WRONG
raw_dir = Path("datasets/moprobo/meta/raw")
```

### Config Loading
```python
# ✅ CORRECT
from src.config.manager import ConfigManager
config = ConfigManager.load(customer="moprobo", platform="meta")

# ❌ WRONG
config = yaml.safe_load("config/moprobo/meta.yaml")
```

### Param Updates
```python
# ✅ CORRECT
if confidence > threshold and has_sufficient_data:
    update_params(new_params)

# ❌ WRONG
update_params(new_params)  # No checks
```

---

## Repo Goals

### 1. Primary Goals

**Budget Allocator (`src/adset/allocator/`)**
- The goal is NOT to deliver the perfect budget allocation solution
- Given a predefined monthly budget, allocate it among a list of pregenerated adsets
- Scope: Daily budget allocation at adset level
- Outputs: Daily budget allocation per adset

**Audience Configuration Generator (`src/adset/generator/`)**
- Generate strategies for creating adset audience configurations (regions, ages, creative formats and other audience types) given each platform's targeting settings
- Recommend audience configurations across multiple dimensions (geography, age ranges, audience types, creative formats)
- Outputs: Audience configuration recommendations with confidence and evidence

### 2. Actual Objective (CRITICAL for Both)
**Analyze customer data for each platform and deliver a solution that can beat history (or at least match).**

- Focus on practical improvement over theoretical perfection
- Compare performance against historical baseline
- **Every recommendation must answer: "Will this perform better than what we've seen?"**
- Deliver working solutions, not optimal ones

### 3. Auto-Parameter Selection
**Each platform and customer's parameters are auto-decided by the algorithm.**

- The system automatically determines optimal parameters per customer/platform
- No manual parameter tuning required
- Data-driven decision making

### 4. Config Separation
**Each platform and customer's parameters are separate through config.**

- Structure: `config/{customer}/{platform}/rules.yaml`
- Example: `config/moprobo/meta/rules.yaml`, `config/ecoflow/google/rules.yaml`
- Platform-specific isolation
- No cross-contamination between customers/platforms

### 5. Code Quality Checks
**All CI workflows must pass before committing changes.**

- Run all CI checks (lint, test, etc.) before committing
- Fix all failures
- Maintain code quality standards

#### Pre-Push CI/CD Checklist
Before pushing code, verify:

- [ ] **No test failures in CI pipeline** - All unit tests must pass
- [ ] **No lint failures** - pylint, mypy, and other linters must report 0 errors
- [ ] **Coverage maintained at or above baseline** - Check `.coverage.baseline` file; coverage must not drop (e.g., from 49% to 47%)
- [ ] **No increase in unjustified test skips** - CI skips should only be used for real environment limitations, not to hide failures
- [ ] **No workarounds to bypass CI checks** - Fix root causes instead of using temporary workarounds
- [ ] **Type checking passes** - mypy and other type checkers must report no errors

### 6. Daily Data Updates
**Algorithm and parameters must update when new daily data arrives.**

- System should adapt to changing data patterns
- Auto-reoptimize parameters on daily data updates
- No manual intervention required

### 7. Reliable Parameter Updates
**When adding new daily data, update params only if high confidence and enough data support.**

- Require statistical significance before param changes
- Minimum data thresholds for updates
- Conservative: keep old params if uncertainty is high
- Protect against noise in daily data

### 8. Configurable Objectives
**Optimization objectives (ROAS, CTR, CVR, CPA, etc.) are configurable through config.**

- Objectives defined per customer/platform in config
- Support single or multi-objective optimization
- Easy to add/change objectives without code changes

### 9. Campaign-Level Configurability
**Each campaign is configurable through config.**

- Campaign-specific overrides and settings
- Granular control at campaign level
- Supports heterogeneous campaign strategies

### 10. Reliability Over Aggression
**Value reliability in optimization since this is real money and no A/B tests now.**

- Conservative approach preferred over aggressive optimization
- No experimental features in production
- Protect customer budget over testing hypotheses
- If it breaks production, it's not worth it

### 11. Daily Parameter Re-optimization
**Rerun parameter optimizations when data is updated daily.**

- Automated pipeline for daily re-tuning
- Parameters adapt to new data patterns
- No stale parameters from old data
- Apply reliability constraints (see #7)

### 12. Realistic Evaluation
**Evaluation must be realistic and reliable when deciding params.**

- Use real customer data, not synthetic
- Proper backtesting (no look-ahead bias)
  - Train on past data, test on future data
  - Never use future information to make past decisions
  - Time-series cross-validation with temporal splits
- Statistical significance testing
- Multiple scenarios, not single-seed results

### 13. Unified Action Output Format
**Output real actions with details, confidence scores, and evidences in unified YAML format.**

- Structure: `.yaml` file with:
  - Action: What to do (scale up/down, freeze, launch_new, etc.)
  - Details: Action specifics (amount, target, config, etc.)
  - Confidence score: Numeric confidence (0-1) or HIGH/MEDIUM/LOW
  - Evidence: Supporting data/reasoning
- Standardized format for all customers/platforms
- Machine-readable + human-readable

---

## Budget Allocator Architecture

### Core Components (`src/adset/allocator/`)
**Engine** (`engine.py`): Main allocator interface
- Orchestrates budget allocation across adsets
- Applies decision rules in priority order
- Returns final budgets with decision paths

**Rules** (`rules.py`): Decision rule orchestration
- Calls appropriate rules from `src/adset/lib/`
- Returns budget adjustments with reasons

### Supporting Library (`src/adset/lib/`)
**models.py**: Data models for budget allocation
- `BudgetAllocationMetrics`: Adset performance metrics
- `BudgetAdjustmentParams`: Parameters for decision rules
- Property-based access for type safety

**decision_rules.py**: 42+ decision rules for budget adjustments
- Safety checks: Freeze low performers, cap budgets
- Performance tiers: Excellent, high, medium, low performers
- Lifecycle rules: Cold start, learning phase, established
- Time-based: Weekend boosts, Monday recovery, Q4 seasonal
- Trend-based: Rising, falling, stable trends
- Volume-based: Spend, impressions, clicks thresholds

**safety_rules.py**: Safety checks
- Freeze on low ROAS
- Budget cap enforcement
- Monthly budget tracking

**decision_rules_helpers.py**: Helper functions
- Gradient adjustment
- Trend scaling
- Health scoring
- Relative performance

### Decision Rules Framework

#### Priority Order (Highest to Lowest)
1. **Safety checks** - Freeze underperforming adsets immediately
2. **Excellent performers** - Aggressive increases for top performers
3. **High performers** - Moderate increases for strong performers
4. **Medium performers** - Maintenance budgets
5. **Declining performers** - Decreases for poor performance
6. **Low performers** - Aggressive decreases or freeze

#### Rule Output Format
Each rule returns:
```python
(
    adjustment_factor,  # e.g., 1.20 = +20% increase
    decision_path      # e.g., "excellent_roas_rising_trend"
)
```

#### Key Decision Factors
- **roas_7d**: Meta's 7-day rolling ROAS (primary signal)
- **roas_trend**: Trend in ROAS over time
- **health_score**: Overall adset health (0-1)
- **efficiency**: Revenue per impression
- **spend**: Current spend level
- **impressions**, **clicks**: Volume metrics
- **days_active**: Age of adset
- **shopify_roas**: Actual revenue ROAS (supplemental validation)

---

## Audience Configuration Generator Architecture

### Core Components (`src/adset/generator/`)
**Core** (`core/recommender.py`): Base recommender classes
- ConfigurableRecommender, ROASRecommender, MetricRecommender
- RecommendationStrategy patterns
- Create recommenders with custom configurations

**Detection** (`detection/mistake_detector.py`): Detect issues in configs
- Identifies suboptimal audience configurations
- Detects human mistakes in setup
- Flags wasting, too_broad, missing_lal, oversaturated

**Analyzers** (`analyzers/`):
- `opportunity_sizer.py`: Calculate opportunity size (frequency, ROAS, budget)
- `shopify_analyzer.py`: Shopify revenue analysis
- `advantage_constraints.py`: Competitive advantages and constraints

**Generation** (`generation/`):
- `audience_recommender.py`: Generate audience-level recommendations with historical calibration
- `audience_aggregator.py`: Aggregate audience-level recommendations
- `creative_compatibility.py`: Creative x audience compatibility

**Segmentation** (`segmentation/segmenter.py`): Segment by geo, audience, creative

### Performance Validation Requirement (CRITICAL)

**All recommendations must be validated against historical performance data.**

- **New audience configurations**: Must show similar segments performed well historically
- **Scale-up recommendations**: Must prove current performance beats baseline
- **Mistake detection**: Must show current config underperforms vs historical averages
- **Calibrate predictions**: Cap at historical 95th percentile (mean + 2*std)
- **No look-ahead bias**: Use only data available at decision time

### Recommendation Types

| Type | Priority | Description | Action | Performance Validation Required |
|------|----------|-------------|--------|-------------------------------- |
| **launch_new** | HIGH | High-potential untested audience configuration | Create new adset with suggested config | Similar segments beat historical baseline by >20% |
| **scale_up** | HIGH | Underfunded winning configuration | Increase budget (up to headroom limit) | Current ROAS > historical baseline × 1.5 |
| **wasting** | CRITICAL | Low ROAS + high spend | PAUSE immediately | ROAS < historical segment average × 0.5 |
| **too_broad** | MEDIUM | Age range too wide for optimization | Test narrower segments | Narrower segments historically outperform |
| **missing_lal** | MEDIUM | No LALs + low ROAS | Create LAL 1% from best customers | LALs historically 2-3x ROAS vs interest |
| **oversaturated** | MEDIUM | Frequency > 4.0 + low ROAS | Reduce budget to target freq 2.5 | Frequency-ROAS curve shows diminishing returns |
| **optimize_or_pause** | HIGH | Underperforming with low confidence | Review or pause | Underperforms historical baseline significantly |

### Headroom Validation

**All scale-up or new audience recommendations must be validated against headroom limits.**

- Safe headroom: budget up to frequency 2.5 (optimal)
- Max headroom: budget up to frequency 4.0 (diminishing returns)
- Never recommend launching new audiences without market capacity
- Prevent recommending over-saturated audience configurations

### Platform-Aware Design

**Each platform has different targeting capabilities - recommendations must respect these.**

- **Meta Ads**: Age ranges, LAL percentages, interest targeting, geo targeting
- **Google Ads**: Demographics, in-market audiences, affinity audiences
- **TikTok Ads**: Age ranges, interests, behaviors
- Never recommend targeting options not available on the platform

### Configuration Dimensions (with Historical Validation)

#### 1. Geography
- **Input**: `adset_targeting_countries` (e.g., "['US']", "['US', 'CA']")
- **Output**: Recommended countries/regions to test
- **Validation**:
  - Analyze historical ROAS by country
  - Recommend high-performing regions for expansion
  - Flag underperforming regions to pause
  - **Rule**: Only recommend countries where historical ROAS > baseline

#### 2. Age Targeting
- **Input**: `adset_targeting_age_min`, `adset_targeting_age_max`
- **Output**: Recommended age ranges to test
- **Validation**:
  - Age range > 30 years → too_broad (test narrower segments)
  - Use 20-year ranges for testing (e.g., 18-38, 25-45)
  - A/B test multiple ranges before committing
  - **Rule**: Only recommend age ranges with historical precedent

#### 3. Audience Type
- **Input**: `adset_targeting_custom_audiences_count`, age_range
- **Output**: Lookalike vs Interest vs Broad
- **Validation**:
  - Historical ROAS by audience type
  - LALs typically 2-3x vs interest targeting
  - Broad requires high historical ROAS to justify
  - **Rule**: Recommend LAL if historical LAL ROAS > current Interest ROAS × 1.5

#### 4. Creative Format
- **Input**: `video_30_sec_watched_actions`, `video_p100_watched_actions`
- **Output**: Video vs Image vs UGC
- **Validation**:
  - Segment by creative format to find winners
  - Historical ROAS by format
  - **Rule**: Recommend format with highest historical ROAS for segment

### Rules-Based (KISS Principle)

**This is a rules-based system, not an ML model.**

- No training data required
- No model overfitting concerns
- No hyperparameter tuning
- Fully interpretable logic
- Changes are immediate (no retraining needed)

---

## Common Mistakes to Avoid

| Mistake | Why It's Wrong | Correct Approach |
|---------|---------------|------------------|
| Hard-coding customer names | Breaks multi-customer support | Use `--customer` parameter |
| Assuming platform is "meta" | Breaks multi-platform support | Support all platforms |
| Updating params without confidence checks | Could update on noise/outliers | Add thresholds |
| Using synthetic data for claims | Not realistic or reliable | Use real data only |
| Breaking path abstraction | Fragile, customer-specific | Use `customer_paths.py` (allocator/adset) or ad path helpers |
| Sharing params across customers | Breaks config isolation | Keep separate configs |
| Claiming "optimal" solutions | Goal is beating history, not perfection | Focus on practical improvement |
| Over-engineering for perfection | Adds complexity without benefit | Keep it simple |
| Skipping CI checks | Lowers code quality | All workflows must pass |
| Outputting without confidence/evidence | Not actionable or auditable | Unified YAML format |
| Using `# pylint: disable` or `# type: ignore` | Suppresses warnings, hides real issues | Fix the actual problem |
| Ignoring monthly budget caps | Can overspend budget | Track cumulative spend |
| Aggressive scaling factors | Risks real money | Use conservative (0.95) multiplier |
| Not freezing low performers | Wastes budget on poor performers | Safety first |
| Recommending configurations without historical validation | No proof it will beat baseline | Validate against similar segments |
| Adding ML models for strategy | Black-box, not interpretable | Use rules-based approach |
| Adding budget allocation in generator | Wrong module (that's allocator) | Keep modules separated |
| Adding audience config in allocator | Wrong module (that's generator) | Keep modules separated |
| Ignoring platform targeting constraints | Recommends unavailable options | Platform-aware design |
| Recommending without headroom check | Risks over-saturation | Calculate headroom first |

---

## Pre-Change Reflection Checklist

Before making any code change, Claude must verify:

### Goal Alignment
- [ ] Does this change support beating/matching historical performance?
- [ ] Does this preserve per-customer/per-platform config separation?
- [ ] Does this maintain auto-parameter selection capability?
- [ ] Does this support daily data updates?
- [ ] Does this output actions in unified YAML format with confidence+evidence?
- [ ] Does this require high confidence before param updates?
- [ ] Does this respect monthly budget caps? (allocator)
- [ ] **Does this validate against historical performance?** (generator)
- [ ] **Does this prove recommendations will beat or match history?** (generator)
- [ ] Does this respect platform-specific targeting capabilities? (generator)
- [ ] Does this include segmentation analysis? (generator)
- [ ] Does this use conservative, rules-based logic? (generator)
- [ ] Does this validate against headroom limits? (generator)
- [ ] Does this suggest testing over committing? (generator)
- [ ] **Does this support statistical pattern detection (not ML)?** (creative recommender)
- [ ] **Does this maintain hard-coded thresholds (no data leakage)?** (creative recommender)
- [ ] **Does this use conservative impact estimates (50% factor)?** (creative recommender)
- [ ] **Does this require statistical significance (chi-square)?** (creative recommender)
- [ ] **Does this filter by lift >= 1.5 and prevalence >= 10%?** (creative recommender)
- [ ] **Does this load recommendations from creative scorer?** (creative generator)
- [ ] **Does this convert features to optimized prompts?** (creative generator)
- [ ] **Does this validate generated features?** (creative generator)

### Anti-Goal Check
- [ ] Does NOT claim to deliver "perfect" allocation?
- [ ] Does NOT hard-code parameters across customers/platforms?
- [ ] Does NOT break config isolation?
- [ ] Does NOT over-engineer for theoretical optimality?
- [ ] Does NOT ignore real-money consequences?
- [ ] Does NOT require A/B testing to validate?
- [ ] Does NOT update params on low-confidence/noisy data?
- [ ] Does NOT use synthetic data for evaluation claims?
- [ ] Does NOT ignore monthly budget constraints?
- [ ] Does NOT introduce black-box ML for strategy? (generator)
- [ ] Does NOT add budget allocation features in generator? (generator)
- [ ] Does NOT add audience configuration in allocator? (allocator)
- [ ] **Does NOT recommend without historical validation?** (generator)
- [ ] **Does NOT ignore historical baseline comparison?** (generator)
- [ ] Does NOT ignore platform constraints? (generator)
- [ ] Does NOT over-claim opportunity values? (generator)
- [ ] Does NOT skip headroom validation? (generator)
- [ ] **Does NOT use ML models for creative pattern detection?** (creative recommender)
- [ ] **Does NOT tune creative pattern parameters on test data?** (creative recommender)
- [ ] **Does NOT claim causation from correlation?** (creative recommender)
- [ ] **Does NOT overpromise creative impact (uses 50% factor)?** (creative recommender)
- [ ] **Does NOT skip statistical significance testing?** (creative recommender)
- [ ] **Does NOT recommend low-prevalence patterns (< 10%)?** (creative recommender)
- [ ] **Does NOT recommend low-lift patterns (< 1.5x)?** (creative recommender)
- [ ] **Does NOT ignore feature recommendations from creative scorer?** (creative generator)
- [ ] **Does NOT skip feature validation after generation?** (creative generator)

### Change Scope
- [ ] Is this the minimum change needed?
- [ ] Preserves existing behavior where appropriate
- [ ] Maintains backward compatibility
- [ ] **All CI workflows pass**
  - [ ] No test failures in CI pipeline
  - [ ] No lint failures (pylint, mypy, etc.)
  - [ ] **Coverage maintained at or above baseline** (check `.coverage.baseline`)
  - [ ] No increase in unjustified test skips
  - [ ] No workarounds used to bypass CI checks

### Reliability Check
- [ ] Is this safe for production with real money?
- [ ] Does this preserve customer budget?
- [ ] What happens if this goes wrong?
- [ ] Is there a rollback path?
- [ ] Evaluation uses real data with proper testing?
- [ ] **Does this recommendation beat historical performance?** (generator)
- [ ] Are estimates conservative or aggressive? (generator)
- [ ] Does this respect platform API limits? (generator)

### Code Pattern Check
- [ ] Uses `customer_paths.py` (allocator/adset) or ad path helpers (`recommender` utils, `generator` `Paths`) as appropriate?
- [ ] Uses `ConfigManager` (allocator/adset) or `config_manager`/config paths (ad modules) for config loading?
- [ ] No hard-coded customer/platform names?
- [ ] No hard-coded file paths?
- [ ] No `# pylint: disable` or `# type: ignore` suppressions?

---

## Decision Path Examples

### Example 1: Adding Hard-Coded Parameters
**Proposed change**: "Add global threshold X for all customers"

**Reflection**:
- ❌ Violates auto-parameter selection
- ❌ Breaks per-customer config separation
- **Decision**: DECLINE. Use config-based approach instead.

### Example 2: Over-Engineering for Perfection
**Proposed change**: "Implement complex optimization for theoretical maximum ROAS"

**Reflection**:
- ❌ Goal is beating history, not perfection
- ❌ Adds complexity without clear benefit
- **Decision**: DECLINE. Focus on practical improvements.

### Example 3: Cross-Customer Parameter Sharing
**Proposed change**: "Share learned parameters across customers"

**Reflection**:
- ❌ Breaks config isolation
- ❌ Violates per-customer separation principle
- **Decision**: DECLINE. Keep customers separate.

### Example 4: Data-Driven Auto-Tuning
**Proposed change**: "Automatically determine optimal thresholds from customer data"

**Reflection**:
- ✅ Supports auto-parameter selection
- ✅ Maintains per-customer separation
- **Decision**: PROCEED. Aligns with core goals.

### Example 5: Aggressive New Algorithm
**Proposed change**: "Add aggressive optimization that could 2x ROAS but risks 50% loss"

**Reflection**:
- ❌ Real money at stake, no A/B testing
- ❌ Violates reliability principle
- **Decision**: DECLINE. Not worth the risk.

### Example 6: Changing Output Format
**Proposed change**: "Output actions in JSON instead of YAML"

**Reflection**:
- ❌ Breaks unified YAML format requirement
- ❌ Requires updating all consumers
- **Decision**: DECLINE unless compelling reason. Stick to YAML.

### Example 7: Adding New Objective
**Proposed change**: "Add CPA as optimization objective"

**Reflection**:
- ✅ Objectives should be configurable
- ✅ Add to config, not hard-code
- **Decision**: PROCEED. Make it config-driven.

### Example 8: Updating Params on Noisy Data
**Proposed change**: "Immediately update params when new daily data arrives"

**Reflection**:
- ❌ Must check confidence and data support
- ❌ Could update on noise/outliers
- **Decision**: DECLINE pure auto-update. Add confidence threshold.

### Example 9: Synthetic Data Evaluation
**Proposed change**: "Prove this works with synthetic data simulation"

**Reflection**:
- ❌ Not realistic or reliable
- ❌ Must use real customer data
- **Decision**: DECLINE. Use real data for validation.

### Example 10: Hard-Coding Platform
**Proposed change**: "Set platform='meta' in this function"

**Reflection**:
- ❌ Assumes single platform
- ❌ Breaks multi-platform support
- **Decision**: DECLINE. Pass platform as parameter.

### Example 11: Suppressing Lint Warnings
**Proposed change**: "Add `# pylint: disable=no-member` to make tests pass"

**Reflection**:
- ❌ Suppresses warnings instead of fixing issues
- ❌ Hides real problems (removed attributes don't exist)
- ❌ Creates technical debt
- **Decision**: DECLINE. Fix the actual code or remove unused tests.

### Example 12: Ignoring Monthly Budget
**Proposed change**: "Allow allocation to exceed monthly budget for high performers"

**Reflection**:
- ❌ Violates monthly budget cap requirement
- ❌ Risks overspending customer budget
- ❌ Breaks trust with customer
- **Decision**: DECLINE. Must track cumulative spend and respect caps.

### Example 13: ML-Based Strategy Generation (Generator)
**Proposed change**: "Use reinforcement learning to learn optimal audience configs"

**Reflection**:
- ❌ Violates rules-based, transparent approach
- ❌ Black-box, not interpretable
- ❌ Requires extensive training
- **Decision**: DECLINE. Use simple rules with historical analysis.

### Example 14: Recommendation Without Historical Validation (Generator)
**Proposed change**: "Recommend launching US + 25-45 + Lookalike audience"

**Reflection**:
- ❌ No historical performance data for this combination
- ❌ No proof it will beat baseline
- **Decision**: DECLINE. Must show similar segments performed well.

### Example 15: Historical Baseline Comparison (Generator)
**Proposed change**: "Add historical average ROAS by segment to evidence"

**Reflection**:
- ✅ Enables performance validation
- ✅ Shows if recommendation beats history
- **Decision**: PROCEED. Essential for goal.

### Example 16: Calibrating Predictions to History (Generator)
**Proposed change**: "Cap predictions at historical 95th percentile"

**Reflection**:
- ✅ Prevents unrealistic claims
- ✅ Aligns with "beat history, not perfection" goal
- **Decision**: PROCEED. Already in code.

### Example 17: Segment-Based Recommendations with History (Generator)
**Proposed change**: "Analyze geo × audience × creative segments, recommend top performers"

**Reflection**:
- ✅ Uses historical data to find winners
- ✅ Recommendations based on actual performance
- **Decision**: PROCEED. Core to the repo's purpose.

### Example 18: Ignoring Platform Constraints (Generator)
**Proposed change**: "Recommend LAL 1% for all platforms"

**Reflection**:
- ❌ Not all platforms support LAL
- ❌ Google uses in-market/affinity, not LAL
- **Decision**: DECLINE. Make platform-aware.

### Example 19: Headroom for New Audiences (Generator)
**Proposed change**: "Check market headroom before recommending new audience launch"

**Reflection**:
- ✅ Prevents over-saturation
- ✅ Validates market capacity
- **Decision**: PROCEED. Essential for reliability.

### Example 20: Performance Threshold for New Audiences (Generator)
**Proposed change**: "Only recommend new audience if similar segments beat baseline by 20%"

**Reflection**:
- ✅ Ensures recommendations beat history
- ✅ Conservative threshold
- **Decision**: PROCEED. Aligns with performance goal.

### Example 21: Budget Allocation Feature in Generator (Generator)
**Proposed change**: "Add automatic budget redistribution across audiences"

**Reflection**:
- ❌ Outside scope (that's in allocator module)
- ❌ Not about configuration strategy
- **Decision**: DECLINE. That's for allocator module.

### Example 22: Single Configuration vs Testing (Generator)
**Proposed change**: "Recommend specific age range vs A/B test multiple ranges"

**Reflection**:
- ❌ Single range assumes knowledge
- ✅ Testing is more conservative
- ✅ Testing validates against actual performance
- **Decision**: DECLINE single recommendation. Use A/B test approach.

---

## Red Flags (Stop and Reconsider)

### 🚩 Config Violations
1. Hard-coding parameters that should be auto-determined
2. Sharing configuration between customers/platforms
3. Breaking config directory structure (`config/{customer}/{platform}/rules.yaml`)
4. Hard-coding objectives (ROAS, CTR, etc.) in code

### 🚩 Reliability Violations
5. Claiming "optimal" or "perfect" solutions
6. Implementing aggressive experiments with real money
7. Updating params without confidence/data support checks
8. Using synthetic data for evaluation claims
9. Ignoring statistical significance in evaluation
10. Skipping realistic backtesting evaluation

### 🚩 Monthly Budget Violations (Allocator)
11. Ignoring monthly budget cap
12. Not tracking cumulative spend
13. Allowing overspend on high performers
14. Front-loading spend early in month
15. Missing state persistence across daily runs

### 🚩 Performance Violations (Generator - CRITICAL)
16. **Recommending configurations without historical validation**
17. **Ignoring historical baseline comparison**
18. **Claiming improvements without segment evidence**
19. **Scaling underperforming audiences (ROAS < baseline)**
20. **Launching new audiences without proving similar segments work**

### 🚩 Design Violations
21. Adding ML models for configuration strategy (generator)
22. Adding budget allocation features in generator (wrong module)
23. Adding audience configuration in allocator (wrong module)
24. Ignoring platform-specific targeting capabilities (generator)
25. Recommending unavailable targeting options (generator)
26. Breaking module separation (allocator vs generator)

### 🚩 Headroom Violations (Generator)
27. Recommending new audiences without headroom check
28. Ignoring saturation warnings
29. Aggressive opportunity estimates (2x+, 3x+)

### 🚩 Code Quality Violations
30. Over-engineering for theoretical vs practical improvement
31. Hard-coding file paths (use `customer_paths.py` for allocator/adset; `recommender` utils / `generator` `Paths` for ad modules)
32. Hard-coding customer/platform names
33. Skipping CI workflow checks
34. Breaking unified YAML output format
35. Creating PR-specific documentation (TODO.md, VALIDATION.md, etc.)
36. Using `# pylint: disable` or `# type: ignore` to suppress warnings
37. Adding back `--fail-under` threshold to pylint (allows CI to pass with low scores)
38. Automatically updating README without preserving existing style/formatting
39. Pushing structural changes without running tests first (moved files, reorganized code, etc.)
40. Leaving temporary scripts in working directory when pushing (cleanup scripts, test files, etc.)
41. Missing confidence scores (generator)
42. Missing evidence dictionaries (generator)
43. Over-claiming without segment data (generator)

### 🚩 CI/CD Violations (CRITICAL)
44. **Coverage regression** - Code coverage drops below baseline (e.g., from 49% to 47%)
45. **Test failures in CI** - Any unit tests failing in CI pipeline
46. **Lint failures in CI** - pylint or other linting tools reporting errors
47. **Skipping CI checks to bypass failures** - Using workarounds to pass CI instead of fixing root causes
48. **Increasing test skip count without justification** - Adding CI skips to hide test failures rather than fixing them
49. **Type checking failures** - mypy or other type checkers reporting errors that are ignored

### 🚩 Design Violations (Both)
44. Ignore historical baseline comparison
45. Break daily re-optimization pipeline
46. Output actions without confidence scores
47. Output actions without supporting evidence
48. Use non-YAML format for actions

---

## Key File Locations

### Allocator Core Files
| Component | File | Purpose |
|-----------|------|---------|
| **Engine** | `src/adset/allocator/engine.py` | Main allocator interface |
| **Rules** | `src/adset/allocator/rules.py` | Rule orchestration |
| **Decision Rules** | `src/adset/lib/decision_rules.py` | 42+ decision rules |
| **Safety Rules** | `src/adset/lib/safety_rules.py` | Freeze/cap logic |
| **Models** | `src/adset/lib/models.py` | Data models |
| **Helpers** | `src/adset/lib/decision_rules_helpers.py` | Rule helper functions |
| **Tuning** | `src/optimizer/lib/bayesian_tuner.py` | Bayesian optimization |
| **State** | `src/budget/state_manager.py` | Monthly state persistence |
| **Tracking** | `src/budget/monthly_tracker.py` | Budget tracking logic |

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

### Ad Recommender Core Files (`src/ad/recommender/`)
| Component | File | Purpose |
|-----------|------|---------|
| **Extract** | `src/ad/recommender/features/extract.py` | Feature extraction and ROAS integration |
| **GPT-4 Extractor** | `src/ad/recommender/features/extractors/gpt4_feature_extractor.py` | GPT-4 Vision API extractor |
| **Transformer** | `src/ad/recommender/features/transformers/gpt4_feature_transformer.py` | Transform GPT-4 responses to features |
| **Interactions** | `src/ad/recommender/features/interactions.py` | Feature interactions |
| **Lib** | `src/ad/recommender/features/lib/` | Loaders, mergers, parsers, synthetic data |
| **Rule Engine** | `src/ad/recommender/recommendations/rule_engine.py` | Statistical pattern-based recommendation engine |
| **Prompt Formatter** | `src/ad/recommender/recommendations/prompt_formatter.py` | Format recommendations as prompts |
| **Evidence** | `src/ad/recommender/recommendations/evidence_builder.py` | Build evidence for recommendations |
| **Formatters** | `src/ad/recommender/recommendations/formatters.py` | Output formatting |
| **Config** | `src/ad/recommender/utils/config_manager.py` | Config loading (gpt4 features/prompts) |
| **Paths** | `src/ad/recommender/utils/paths.py` | Data dir, features CSV resolution |
| **Statistics** | `src/ad/recommender/utils/statistics.py` | Chi-square and statistical tests |
| **Predictor** | `src/ad/recommender/predictor.py` | Prediction utilities |

### Ad Generator Core Files (`src/ad/generator/`)
| Component | File | Purpose |
|-----------|------|---------|
| **Paths** | `src/ad/generator/core/paths.py` | Customer/platform/date path management |
| **Scorer Loader** | `src/ad/generator/core/scorer_recommendations_loader.py` | Load creative scorer recommendations |
| **Prompts** | `src/ad/generator/core/prompts/` | Converters, feature loader, recommendations loader, variants |
| **Generation** | `src/ad/generator/core/generation/generator.py` | Image generation via FAL.ai |
| **Prompt Converter** | `src/ad/generator/core/generation/prompt_converter.py` | Nano Banana prompt conversion |
| **Orchestrator** | `src/ad/generator/orchestrator/prompt_builder.py` | Build prompts from features |
| **Feature Mapper** | `src/ad/generator/orchestrator/feature_mapper.py` | Map features to prompt elements |
| **Template Engine** | `src/ad/generator/orchestrator/template_engine.py` | Template-based generation |
| **Pipeline** | `src/ad/generator/pipeline/pipeline.py` | End-to-end generation pipeline |
| **Feature Validator** | `src/ad/generator/pipeline/feature_validator.py` | Validate generated features |
| **Recommendation Loader** | `src/ad/generator/pipeline/recommendation_loader.py` | Load recommendations for pipeline |

### Shared Files
| Component | File | Purpose |
|-----------|------|---------|
| **Config** | `src/config/manager.py` | Config loading |
| **Paths** | `src/utils/customer_paths.py` | Path abstraction |
| **Logging** | `src/utils/logger_config.py` | Logging setup |
| **Shopify** | `src/integrations/shopify/` | Shopify ROAS validation |
| **Allocator Workflow** | `src/workflows/allocation_workflow.py` | Allocation workflow |
| **Generator CLI** | `src/cli/commands/rules.py` | Rules pipeline |
| **Generator CLI** | `src/cli/commands/auto_params.py` | Auto-calc parameters |
| **Ad Recommender/Generator CLI** | `run.py` | extract-features, recommend, prompt, generate, run |

---

## Core Principle Summary

### Budget Allocator
**"Allocate budget to maximize ROAS, not to achieve perfection"**

- Beat or match historical performance
- Use conservative safety factors (0.95 multiplier)
- Validate with time-series cross-validation
- Track monthly spend and respect caps
- Freeze low performers before scaling winners
- If uncertain, maintain current budget

### Audience Configuration Generator
**"Every recommendation must answer: Will this perform better than what we've seen historically?"**

If the answer is "I don't know" or "maybe," then the recommendation should be framed as a test, not a commitment.

- Validate all recommendations against historical performance
- Use rules-based, transparent logic (KISS principle)
- Calculate headroom before recommending scale-up or new audiences
- Output recommendations with confidence + evidence
- Segment by geography, audience type, creative format
- Maintain priority scoring (CRITICAL > HIGH > MEDIUM > LOW)
- Use conservative estimates for opportunity values
- Respect platform-specific targeting capabilities
- **If uncertain, recommend small test vs full rollout**

### Ad Recommender (Creative Scorer)
**"NO AI. NO SPECULATION. JUST STATISTICS."**

- Statistical pattern detection only (no ML)
- Hard-coded thresholds (top/bottom 25%, lift ≥ 1.5, prevalence ≥ 10%)
- Conservative impact estimates (50% of lift)
- Chi-square significance required
- Use `config_manager` and `paths` (recommender utils) for config/data

### Ad Generator (Creative Image Generation)
**"Feature-driven prompts, validated output."**

- Load recommendations from creative scorer; convert to prompts
- Use `Paths` (generator `core/paths.py`) for all config/output paths
- Validate generated images match requested features
- FAL.ai (Nano Banana) for image generation
- Customer/platform/date isolation

---

## Monthly Budget Tracking (Allocator)

### State File Location
`results/{customer}/{platform}/monthly_state_YYYY-MM.json`

### State Structure
```json
{
  "metadata": {
    "customer": "moprobo",
    "platform": "meta",
    "month": "2026-01",
    "last_updated": "2026-01-23T10:30:00"
  },
  "budget": {
    "monthly_budget_cap": 10000.0,
    "source": "config"
  },
  "tracking": {
    "total_spent": 7450.50,
    "total_allocated": 7500.00,
    "remaining_budget": 2550.00,
    "days_active": 23,
    "days_in_month": 31
  },
  "execution_history": [
    {
      "date": "2026-01-01",
      "daily_budget": 322.58,
      "allocated": 320.00,
      "spent": 315.50,
      "num_adsets": 45
    }
  ]
}
```

### Daily Budget Calculation
```python
daily_budget = (monthly_cap - total_spent) / remaining_days * 0.95
```

**Conservative multiplier (0.95)** prevents:
- Front-loading spend early in month
- Over-spending due to under-spend prediction
- Running out of budget before month end

---

## Creative Recommender Architecture (`src/ad/recommender/`)

**Purpose:** Statistical pattern-based creative optimization - analyzes image features to generate ROAS improvement recommendations

### Core Components

**Features** (`features/`):
- `extract.py`: Feature extraction and ROAS integration
- `extractors/gpt4_feature_extractor.py`: GPT-4 Vision API extractor
- `transformers/gpt4_feature_transformer.py`: Transform GPT-4 responses to features
- `interactions.py`: Feature interactions
- `lib/`: Loaders, mergers, parsers, synthetic data

**Recommendations** (`recommendations/`):
- `rule_engine.py`: Statistical pattern-based recommendation engine
- `prompt_formatter.py`: Format recommendations as prompts for creative generation
- `formatters.py`, `enhanced_output.py`, `output_structure.py`: Output formatting
- `evidence_builder.py`: Build evidence for recommendations

**Utils** (`utils/`):
- `config_manager.py`: Config loading for `config/ad/recommender/gpt4/` (features, prompts)
- `paths.py`: Data dir, features CSV resolution (`CREATIVE_SCORER_DATA_DIR`, `CREATIVE_SCORER_FEATURES_CSV`)
- `statistics.py`: Chi-square and statistical tests
- `api_keys.py`, `config_loader.py`, `constants.py`, `platform_normalizer.py`, `model_persistence.py`: Support utilities

### Philosophy: "NO AI. NO SPECULATION. JUST STATISTICS."

- No ML models - only statistical pattern detection
- Transparent lift calculations and prevalence percentages
- Conservative impact estimates (50% factor)
- Fact-based recommendations from actual data

### Pattern Detection Rules

- **Top/Bottom Split**: Hard-coded 25% top, 25% bottom (no tuning to avoid leakage)
- **Lift Threshold**: Minimum 1.5x lift required
- **Prevalence Threshold**: Minimum 10% prevalence in top performers
- **Statistical Significance**: Chi-square test required (p-value < 0.05)
- **Sample Size**: Minimum 10% of top/bottom performers must have the feature

### Recommendation Generation

- Compare creative features against discovered patterns
- Identify gaps (missing high-performing features)
- Calculate potential impact (conservative 50% of lift)
- Generate DOs: "Add feature X" recommendations
- Generate DON'Ts: "Avoid feature X" recommendations (anti-patterns)
- Sort by potential impact (highest first)

### Output Format

- Markdown file with DOs and DON'Ts
- Each recommendation includes:
  - Feature name and value
  - Current vs recommended value
  - Confidence level (high/medium/low)
  - Potential impact (ROAS improvement)
  - Evidence (pattern statistics)
- Grouped by opportunity size

### Data Leakage Prevention

- **Hard-code pattern thresholds**: No tuning on test data
- **All data for pattern discovery**: No train/test split needed
- **Conservative estimates**: 50% factor prevents overpromising
- **Statistical significance**: Chi-square tests prevent false patterns

### Feature Extraction Pipeline

1. Extract features using GPT-4 Vision API
2. Load ad performance data (ROAS)
3. Merge features with ROAS data
4. Identify top 25% and bottom 25% performers
5. Calculate lift for each feature value
6. Filter patterns (lift >= 1.5, prevalence >= 10%, significant)
7. Generate recommendations

### Configuration

- **GPT-4 Config**: `config/ad/recommender/gpt4/features.yaml` (feature definitions)
- **GPT-4 Prompts**: `config/ad/recommender/gpt4/prompts.yaml` (prompt templates)
- **Output**: `config/ad/recommender/{customer}/{platform}/{date}/recommendations.md` (markdown format, DOs/DON'Ts)

### File Location Rules (Ad Recommender)

| Type | Location | Example |
|------|----------|---------|
| Configs | `config/ad/recommender/gpt4/` | `config/ad/recommender/gpt4/features.yaml` |
| Creative Features | `src/ad/recommender/features/` | `src/ad/recommender/features/extract.py` |
| Recommendations | `src/ad/recommender/recommendations/` | `src/ad/recommender/recommendations/rule_engine.py` |
| Output | `config/ad/recommender/{customer}/{platform}/{date}/` | `.../moprobo/meta/2026-01-26/recommendations.md` |

**Rule**: Use `src/ad/recommender/utils/config_manager.py` for config; `src/ad/recommender/utils/paths.py` for data/features. Never hard-code paths.

### Code Patterns to Follow

#### Pattern Detection
```python
# ✅ CORRECT
# Hard-code thresholds to avoid data leakage
top_pct = 0.25      # Top 25% performers
bottom_pct = 0.25   # Bottom 25% performers
lift_threshold = 1.5
prevalence_threshold = 0.10

# Calculate lift
lift = high_pct / low_pct if low_pct > 0 else float("inf")

# Require statistical significance
chi2_result = chi_square_test(contingency)
if lift >= lift_threshold and high_pct >= prevalence_threshold and chi2_result["is_significant"]:
    # Pattern is valid
    pass

# ❌ WRONG
# Tuning thresholds on test data (data leakage)
for top_pct in [0.1, 0.2, 0.25, 0.3]:
    score = evaluate_on_test_data(top_pct)  # ❌ LEAKAGE!
```

#### Impact Calculation
```python
# ✅ CORRECT
# Conservative 50% factor
potential_impact = current_roas * (lift - 1) * 0.5

# ❌ WRONG
# Overpromising (no conservative factor)
potential_impact = current_roas * (lift - 1)  # Too optimistic
```

### Common Mistakes to Avoid (Creative Recommender)

| Mistake | Why It's Wrong | Correct Approach |
|---------|---------------|------------------|
| Tuning top_pct on test data | Data leakage, lookahead bias | Hard-code 0.25 (25%) |
| No conservative factor | Overpromising impact | Use 50% factor: `impact = roas * (lift - 1) * 0.5` |
| Using ML models | Black-box, speculation | Use statistical patterns only |
| Skipping significance tests | False patterns from noise | Require chi-square p-value < 0.05 |
| Low prevalence patterns | Not representative | Require >= 10% prevalence |
| Low lift patterns | Weak signal | Require >= 1.5x lift |
| Hard-coding paths | Breaks multi-customer support | Use path helpers |
| Sharing configs | Breaks isolation | Separate per customer/platform |
| Claiming causation | Correlation ≠ causation | Use conservative estimates |
| No minimum sample size | Unreliable patterns | Require 10% of top/bottom performers |

### Pre-Change Reflection Checklist (Creative Recommender)

Before making any code change to creative recommender, Claude must verify:

#### Goal Alignment
- [ ] Does this support statistical pattern detection (not ML)?
- [ ] Does this maintain hard-coded thresholds (no data leakage)?
- [ ] Does this use conservative impact estimates (50% factor)?
- [ ] Does this require statistical significance (chi-square)?
- [ ] Does this filter by lift >= 1.5 and prevalence >= 10%?
- [ ] Does this output actionable DOs and DON'Ts?
- [ ] Does this use GPT-4 Vision API correctly?
- [ ] Does this merge features with ROAS data properly?
- [ ] Does this use `config_manager` and `paths` (recommender utils) for config/data?

#### Anti-Goal Check
- [ ] Does NOT use ML models for pattern detection?
- [ ] Does NOT tune parameters on test data?
- [ ] Does NOT claim causation from correlation?
- [ ] Does NOT overpromise impact (uses 50% factor)?
- [ ] Does NOT skip statistical significance testing?
- [ ] Does NOT recommend low-prevalence patterns (< 10%)?
- [ ] Does NOT recommend low-lift patterns (< 1.5x)?
- [ ] Does NOT hard-code paths or parameters?
- [ ] Does NOT bypass `config_manager` / `paths` for config or data?

### Key Principles (Creative Recommender)

1. **Statistics Over ML**: Use statistical patterns, not ML models
2. **Transparency**: Show exact lift values, prevalence percentages
3. **Conservative**: 50% factor on impact estimates
4. **Significance**: Require statistical tests (chi-square)
5. **No Leakage**: Hard-code thresholds, no tuning on test data
6. **Actionable**: Clear DOs and DON'Ts with evidence
7. **Fact-Based**: Recommendations from actual data, not speculation

---

## Creative Generator Architecture (`src/ad/generator/`)

**Purpose:** Image generation system that converts feature recommendations into optimized prompts and generates images via FAL.ai

### Core Components

**Core** (`core/`):
- `paths.py`: Customer/platform/date path management (`config/ad/generator/`, `config/ad/recommender/recommendations/`)
- `scorer_recommendations_loader.py`: Load creative scorer recommendations
- `prompts/`: Feature-to-prompt conversion
  - `converter.py`, `converter_simple.py`, `converter_advanced.py`, `converter_enhanced.py`: Prompt converters
  - `feature_loader.py`, `recommendations_loader.py`: Load feature/recommendation data
  - `variants.py`: Feature variants and combinations
  - `feature_descriptions.py`, `feature_value_descriptions.py`, `feature_validation.py`, `output_formatter.py`: Support
- `generation/`: Image generation via FAL.ai
  - `generator.py`: Main image generator
  - `prompt_converter.py`: Nano Banana prompt conversion
  - `watermark.py`, `text_overlay.py`: Watermark and overlay
  - `constants.py`, `markets.py`: Generation config

**Orchestrator** (`orchestrator/`):
- `prompt_builder.py`: Build prompts from features
- `feature_mapper.py`, `feature_registry.py`: Map features to prompt elements
- `template_engine.py`: Template-based generation
- `scene_config.py`, `defaults.py`: Scene and default config

**Pipeline** (`pipeline/`):
- `pipeline.py`: End-to-end generation pipeline
- `recommendation_loader.py`: Load recommendations for pipeline
- `product_context.py`, `prompt_templates.py`: Product context and templates
- `feature_reproduction.py`: Feature reproduction logic
- `feature_validator.py`: Validate generated features

**Constants / Utils**: `constants/`, `utils/` (e.g. `optional_imports.py`)

### Workflow

1. **Load Recommendations**: Load feature recommendations from creative scorer
2. **Convert to Prompts**: Convert feature recommendations to image generation prompts
3. **Generate Images**: Generate images using FAL.ai (Nano Banana models)
4. **Validate Features**: Validate that generated images match requested features

### Configuration

- **Config**: `config/ad/generator/{customer}/{platform}/generation_config.yaml`
- **Templates**: `config/ad/generator/templates/{customer}/{platform}/`
- **Prompts**: `config/ad/generator/prompts/{customer}/{platform}/{date}/{type}/` (e.g. `structured`, `nano`, `variants`)
- **Generated**: `config/ad/generator/generated/{customer}/{platform}/{date}/{model}/` (e.g. `nano-banana-pro`)
- **Recommendations** (input): `config/ad/recommender/{customer}/{platform}/{date}/recommendations.md`

### File Location Rules (Ad Generator)

| Type | Location | Example |
|------|----------|---------|
| Config | `config/ad/generator/{customer}/{platform}/` | `config/ad/generator/moprobo/taboola/generation_config.yaml` |
| Templates | `config/ad/generator/templates/{customer}/{platform}/` | `config/ad/generator/templates/moprobo/taboola/` |
| Prompts | `config/ad/generator/prompts/{customer}/{platform}/{date}/{type}/` | `.../moprobo/taboola/2026-01-23/structured/` |
| Generated | `config/ad/generator/generated/{customer}/{platform}/{date}/{model}/` | `.../nano-banana-pro/` |
| Recommendations | `config/ad/recommender/{customer}/{platform}/{date}/` | `.../moprobo/meta/2026-01-26/recommendations.md` |

**Rule**: Use `src/ad/generator/core/paths.py` (`Paths`) for all config/output paths. Never hard-code paths.

### Key Principles

1. **Feature-Driven**: Generate images based on feature recommendations from creative scorer
2. **Prompt Optimization**: Convert features to optimized prompts for Nano Banana models
3. **Validation**: Validate that generated images match requested features
4. **Customer/Platform Isolation**: Separate configs per customer/platform
5. **Template-Based**: Use templates for consistent generation
6. **FAL.ai Integration**: Use FAL.ai for image generation (Nano Banana models)

### Common Mistakes to Avoid (Creative Generator)

| Mistake | Why It's Wrong | Correct Approach |
|---------|---------------|------------------|
| Hard-coding paths | Breaks multi-customer support | Use path helpers |
| Ignoring feature recommendations | Not using data-driven approach | Load from creative scorer |
| Skipping validation | Generated images may not match features | Validate features after generation |
| Sharing configs | Breaks isolation | Separate per customer/platform |
| Not using templates | Inconsistent generation | Use template engine |

### Pre-Change Reflection Checklist (Creative Generator)

Before making any code change to creative generator, Claude must verify:

#### Goal Alignment
- [ ] Does this load recommendations from creative scorer (`config/ad/recommender/recommendations/...`)?
- [ ] Does this convert features to optimized prompts?
- [ ] Does this use FAL.ai for image generation?
- [ ] Does this validate generated features?
- [ ] Does this use `Paths` (`core/paths.py`) for customer/platform/date paths?
- [ ] Does this use templates for consistency?

#### Anti-Goal Check
- [ ] Does NOT hard-code paths or parameters?
- [ ] Does NOT share configs across customers/platforms?
- [ ] Does NOT skip feature validation?
- [ ] Does NOT ignore recommendations from creative scorer?
- [ ] Does NOT bypass `Paths` for config or output paths?
