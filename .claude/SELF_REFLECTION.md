# Claude Self-Reflection Framework

## Purpose
Ensure any changes made by Claude align with the core goals and constraints of the budget allocation system.

---

## Quick Reference (Claude Cheat Sheet)

### DO
- Use `config/{customer}/{platform}/` for params
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

## File Location Rules

| Type | Location | Example |
|------|----------|---------|
| Configs | `config/{customer}/{platform}/` | `config/moprobo/meta/` |
| Raw Data | `datasets/{customer}/{platform}/raw/` | `datasets/moprobo/meta/raw/` |
| Features | `datasets/{customer}/{platform}/features/` | `datasets/moprobo/meta/features/` |
| Results | `results/{customer}/{platform}/` | `results/moprobo/meta/` |

**Rule**: All data access must use `src/utils/customer_paths.py` helper functions. Never hard-code paths.

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

### 1. Primary Goal
**The goal of this repo is NOT to deliver the perfect budget allocation solution.**

### 2. Problem Statement
**Given a predefined monthly budget, allocate it among a list of pregenerated adsets.**

- **Scope**: Daily budget allocation at adset level
- **Inputs**:
  - Monthly budget cap (predefined)
  - Pregenerated adsets (existing campaigns)
  - Adset performance features (ROAS, CTR, spend, impressions, etc.)
  - Optional: Shopify data (revenue, conversion data)
  - Optional: Lookalike audience data
- **Outputs**: Daily budget allocation per adset
- **Requirements**:
  - **Initialization**: Set initial budgets when new adsets are created
  - **Daily Updates**: Adjust budgets daily based on performance
  - **Constraints**: Stay within monthly budget cap

### 3. Actual Objective
**Analyze customer data for each platform and deliver a solution that can beat history (or at least match).**

- Focus on practical improvement over theoretical perfection
- Compare performance against historical baseline
- Deliver working solutions, not optimal ones

### 4. Auto-Parameter Selection
**Each platform and customer's parameters are auto-decided by the algorithm.**

- The system automatically determines optimal parameters per customer/platform
- No manual parameter tuning required
- Data-driven decision making

### 5. Config Separation
**Each platform and customer's parameters are separate through config.**

- Structure: `config/{customer}/{platform}/rules.yaml`
- Example: `config/moprobo/meta/rules.yaml`, `config/ecoflow/google/rules.yaml`
- Platform-specific isolation
- No cross-contamination between customers/platforms

### 6. Code Quality Checks
**All CI workflows must pass before committing changes.**

- Run all CI checks (lint, test, etc.) before committing
- Fix all failures
- Maintain code quality standards

### 7. Daily Data Updates
**Algorithm and parameters must update when new daily data arrives.**

- System should adapt to changing data patterns
- Auto-reoptimize parameters on daily data updates
- No manual intervention required

### 8. Reliable Parameter Updates
**When adding new daily data, update params only if high confidence and enough data support.**

- Require statistical significance before param changes
- Minimum data thresholds for updates
- Conservative: keep old params if uncertainty is high
- Protect against noise in daily data

### 9. Configurable Objectives
**Optimization objectives (ROAS, CTR, CVR, CPA, etc.) are configurable through config.**

- Objectives defined per customer/platform in config
- Support single or multi-objective optimization
- Easy to add/change objectives without code changes

### 10. Campaign-Level Configurability
**Each campaign is configurable through config.**

- Campaign-specific overrides and settings
- Granular control at campaign level
- Supports heterogeneous campaign strategies

### 11. Reliability Over Aggression
**Value reliability in optimization since this is real money and no A/B tests now.**

- Conservative approach preferred over aggressive optimization
- No experimental features in production
- Protect customer budget over testing hypotheses
- If it breaks production, it's not worth it

### 12. Daily Parameter Re-optimization
**Rerun parameter optimizations when data is updated daily.**

- Automated pipeline for daily re-tuning
- Parameters adapt to new data patterns
- No stale parameters from old data
- Apply reliability constraints (see #8)

### 13. Realistic Evaluation
**Evaluation must be realistic and reliable when deciding params.**

- Use real customer data, not synthetic
- Proper backtesting (no look-ahead bias)
  - Train on past data, test on future data
  - Never use future information to make past decisions
  - Time-series cross-validation with temporal splits
- Statistical significance testing
- Multiple scenarios, not single-seed results

### 14. Unified Action Output Format
**Output real actions with details, confidence scores, and evidences in unified YAML format.**

- Structure: `.yaml` file with:
  - Action: What to do (scale up/down, freeze, etc.)
  - Details: Action specifics (amount, target, etc.)
  - Confidence score: Numeric confidence (0-1)
  - Evidence: Supporting data/reasoning
- Standardized format for all customers/platforms
- Machine-readable + human-readable

---

## Allocator Architecture

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

---

## Decision Rules Framework

### Priority Order (Highest to Lowest)
1. **Safety checks** - Freeze underperforming adsets immediately
2. **Excellent performers** - Aggressive increases for top performers
3. **High performers** - Moderate increases for strong performers
4. **Medium performers** - Maintenance budgets
5. **Declining performers** - Decreases for poor performance
6. **Low performers** - Aggressive decreases or freeze

### Rule Output Format
Each rule returns:
```python
(
    adjustment_factor,  # e.g., 1.20 = +20% increase
    decision_path      # e.g., "excellent_roas_rising_trend"
)
```

### Key Decision Factors
- **roas_7d**: Meta's 7-day rolling ROAS (primary signal)
- **roas_trend**: Trend in ROAS over time
- **health_score**: Overall adset health (0-1)
- **efficiency**: Revenue per impression
- **spend**: Current spend level
- **impressions**, **clicks**: Volume metrics
- **days_active**: Age of adset
- **shopify_roas**: Actual revenue ROAS (supplemental validation)

---

## Common Mistakes to Avoid

| Mistake | Why It's Wrong | Correct Approach |
|---------|---------------|------------------|
| Hard-coding customer names | Breaks multi-customer support | Use `--customer` parameter |
| Assuming platform is "meta" | Breaks multi-platform support | Support all platforms |
| Updating params without confidence checks | Could update on noise/outliers | Add thresholds |
| Using synthetic data for claims | Not realistic or reliable | Use real data only |
| Breaking path abstraction | Fragile, customer-specific | Use `customer_paths.py` |
| Sharing params across customers | Breaks config isolation | Keep separate configs |
| Claiming "optimal" solutions | Goal is beating history, not perfection | Focus on practical improvement |
| Over-engineering for perfection | Adds complexity without benefit | Keep it simple |
| Skipping CI checks | Lowers code quality | All workflows must pass |
| Outputting without confidence/evidence | Not actionable or auditable | Unified YAML format |
| Using `# pylint: disable` or `# type: ignore` | Suppresses warnings, hides real issues | Fix the actual problem |
| Ignoring monthly budget caps | Can overspend budget | Track cumulative spend |
| Aggressive scaling factors | Risks real money | Use conservative (0.95) multiplier |
| Not freezing low performers | Wastes budget on poor performers | Safety first |

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
- [ ] Does this respect monthly budget caps?

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

### Change Scope
- [ ] Is this the minimum change needed?
- [ ] Preserves existing behavior where appropriate
- [ ] Maintains backward compatibility
- [ ] All CI workflows pass

### Reliability Check
- [ ] Is this safe for production with real money?
- [ ] Does this preserve customer budget?
- [ ] What happens if this goes wrong?
- [ ] Is there a rollback path?
- [ ] Evaluation uses real data with proper testing?

### Code Pattern Check
- [ ] Uses `customer_paths.py` for data access?
- [ ] Uses `ConfigManager` for config loading?
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

### 🚩 Monthly Budget Violations
11. Ignoring monthly budget cap
12. Not tracking cumulative spend
13. Allowing overspend on high performers
14. Front-loading spend early in month
15. Missing state persistence across daily runs

### 🚩 Code Quality Violations
16. Over-engineering for theoretical vs practical improvement
17. Hard-coding file paths (use `customer_paths.py`)
18. Hard-coding customer/platform names
19. Skipping CI workflow checks
20. Breaking unified YAML output format
21. Creating PR-specific documentation (TODO.md, VALIDATION.md, etc.)
22. Using `# pylint: disable` or `# type: ignore` to suppress warnings
23. Adding back `--fail-under` threshold to pylint (allows CI to pass with low scores)
24. Automatically updating README without preserving existing style/formatting
25. Pushing structural changes without running tests first (moved files, reorganized code, etc.)
26. Leaving temporary scripts in working directory when pushing (cleanup scripts, test files, etc.)

### 🚩 Design Violations
27. Ignore historical baseline comparison
28. Break daily re-optimization pipeline
29. Output actions without confidence scores
30. Output actions without supporting evidence
31. Use non-YAML format for actions
32. Breaking allocator/generator separation (generator is for audience config, not budget allocation)

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
| **Tracking** | `src/budget/state_manager.py` | Monthly state persistence |
| **Tracking** | `src/budget/monthly_tracker.py` | Budget tracking logic |

### Shared Files
| Component | File | Purpose |
|-----------|------|---------|
| **Config** | `src/config/manager.py` | Config loading |
| **Paths** | `src/utils/customer_paths.py` | Path abstraction |
| **Logging** | `src/utils/logger_config.py` | Logging setup |
| **Shopify** | `src/integrations/shopify/` | Shopify ROAS validation |
| **Workflow** | `src/workflows/allocation_workflow.py` | Allocation workflow |

---

## Core Principle Summary

**"Allocate budget to maximize ROAS, not to achieve perfection"**

- Beat or match historical performance
- Use conservative safety factors (0.95 multiplier)
- Validate with time-series cross-validation
- Track monthly spend and respect caps
- Freeze low performers before scaling winners
- If uncertain, maintain current budget

---

## Monthly Budget Tracking

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
