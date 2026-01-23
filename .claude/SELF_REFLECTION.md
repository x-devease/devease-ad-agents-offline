# Claude Self-Reflection Framework

## Purpose
Ensure any changes made by Claude align with the core goals and constraints of this repo.

---

## Quick Reference (Claude Cheat Sheet)

### DO
- Use `config/{customer}/{platform}.yaml` for params
- Use `datasets/{customer}/{platform}/` for data
- Output actions as YAML with confidence + evidence
- Require high confidence before param updates
- Run all CI workflows before committing
- Use `customer_paths.py` for path access
- Evaluate on real data only

### DON'T
- Hard-code paths or parameters
- Share params across customers/platforms
- Update params on low-confidence data
- Use synthetic data for evaluation claims
- Risk real money on unproven features
- Assume platform is "meta" only
- Break path abstraction
- Create PR-specific documentation (no TODO.md, VALIDATION.md per PR)
- Use `# pylint: disable` or `# type: ignore` to suppress warnings
- Leave temporary scripts in working directory when pushing

---

## File Location Rules

| Type | Location | Example |
|------|----------|---------|
| Configs | `config/{customer}/{platform}.yaml` | `config/moprobo/meta.yaml` |
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

- Structure: `config/{customer}/{platform}.yaml`
- Example: `config/moprobo/meta.yaml`, `config/ecoflow/google.yaml`
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

### Anti-Goal Check
- [ ] Does NOT claim to deliver "perfect" allocation?
- [ ] Does NOT hard-code parameters across customers/platforms?
- [ ] Does NOT break config isolation?
- [ ] Does NOT over-engineer for theoretical optimality?
- [ ] Does NOT ignore real-money consequences?
- [ ] Does NOT require A/B testing to validate?
- [ ] Does NOT update params on low-confidence/noisy data?
- [ ] Does NOT use synthetic data for evaluation claims?

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

---

## Red Flags (Stop and Reconsider)

### 🚩 Config Violations
1. Hard-coding parameters that should be auto-determined
2. Sharing configuration between customers/platforms
3. Breaking config directory structure (`config/{customer}/{platform}.yaml`)
4. Hard-coding objectives (ROAS, CTR, etc.) in code

### 🚩 Reliability Violations
5. Claiming "optimal" or "perfect" solutions
6. Implementing aggressive experiments with real money
7. Updating params without confidence/data support checks
8. Using synthetic data for evaluation claims
9. Ignoring statistical significance in evaluation
10. Skipping realistic/backtesting evaluation

### 🚩 Code Quality Violations
11. Over-engineering for theoretical vs practical improvement
12. Hard-coding file paths (use `customer_paths.py`)
13. Hard-coding customer/platform names
14. Skipping CI workflow checks
15. Breaking unified YAML output format
16. Creating PR-specific documentation (TODO.md, VALIDATION.md, etc.)
17. Using `# pylint: disable` or `# type: ignore` to suppress warnings
18. Adding back `--fail-under` threshold to pylint (allows CI to pass with low scores)
19. Automatically updating README without preserving existing style/formatting
20. Pushing structural changes without running tests first (moved files, reorganized code, etc.)
21. Leaving temporary scripts in working directory when pushing (cleanup scripts, test files, etc.)

### 🚩 Design Violations
18. Ignore historical baseline comparison
19. Break daily re-optimization pipeline
20. Output actions without confidence scores
21. Output actions without supporting evidence
22. Use non-YAML format for actions
