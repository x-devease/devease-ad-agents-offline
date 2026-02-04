# Ad Miner Agents Team - Self-Evolving Pattern Mining System

## Architecture Overview

A multi-agent system for autonomous ad creative pattern mining, recommendation generation, and continuous improvement.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AD MINER AGENTS TEAM                         │
│                      Self-Evolving System                       │
└─────────────────────────────────────────────────────────────────┘

                   ┌──────────────────┐
                   │   Orchestrator   │  ← System Coordinator
                   └────────┬─────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
  ┌──────────┐       ┌──────────┐        ┌──────────┐
  │    PM    │       │  Coder   │        │ Reviewer │
  │  Agent   │──────▶│  Agent   │───────▶│  Agent   │
  └────┬─────┘       └────┬─────┘        └────┬─────┘
       │                  │                   │
       │                  │                   │
       ▼                  ▼                   ▼
  ┌──────────┐       ┌──────────┐        ┌──────────┐
  │  Memory  │       │  Judge   │        │  Monitor │
  │  Agent   │◀──────│  Agent   │───────▶│  Agent   │
  └──────────┘       └──────────┘        └──────────┘
```

## Agent Specifications

### 1. PM Agent (需求与实验规划者)

**Role:** Pattern Mining Strategist & Experiment Planner

**Objective:** Maximize mining ROI by discovering high-impact ad creative patterns while controlling evolution risk.

**Responsibilities:**
- Convert performance insights into specific mining experiment specs
- Set mining parameters (quantile thresholds, feature sets, psychology types)
- Define experiment boundaries (e.g., "only optimize psychology extraction, don't touch core mining")

**Context Window:**
- Current mining performance metrics
- Memory Agent's historical experiments
- Latest ad creative performance data

**Toolset:**
- `query_mining_performance()` - Get current pattern lift scores
- `set_mining_parameters()` - Update quantile, confidence thresholds
- `create_experiment_spec()` - Generate structured experiment plan
- `search_memory()` - Retrieve similar past experiments

**System Prompt Components:**
```yaml
role: "Pattern Mining Strategist"
expertise:
  - "Statistical analysis and A/B testing"
  - "Feature engineering for ad creatives"
  - "Psychology-based marketing optimization"
constraints:
  - "Cannot modify core data loaders without approval"
  - "Must validate any parameter changes with Memory Agent"
  - "Experiments must be backward compatible"
output_format: "JSON"
```

---

### 2. Coder Agent (逻辑与 Prompt 工程师)

**Role:** Feature Extraction & Mining Logic Engineer

**Objective:** Implement PM Agent specs to improve pattern discovery accuracy and psychology classification.

**Responsibilities:**
- Modify feature extraction logic (GPT-4 prompts, visual feature parsers)
- Update psychology classification rules
- Implement new pattern mining algorithms
- Optimize data processing pipelines

**Context Window:**
- Current codebase (miner, extractors, transformers)
- PM Agent's experiment spec
- Memory Agent's relevant historical code changes

**Toolset:**
- `read_file(path)` - Read source code
- `write_file(path, content)` - Write source code
- `run_tests()` - Execute test suite
- `lint_code()` - Static code analysis
- `create_pr()` - Create pull request with diff

**System Prompt Components:**
```yaml
role: "Feature Engineering & Mining Logic Specialist"
expertise:
  - "Computer vision and visual feature extraction"
  - "Natural language processing for ad creative analysis"
  - "Statistical pattern mining and association rule learning"
constraints:
  - "FORBIDDEN: Hard-coding test cases (if id == 'x': return 'y')"
  - "FORBIDDEN: Modifying data validation logic without approval"
  - "MUST: Maintain backward compatibility with existing features"
output_format: "JSON with code diffs"
```

**Key Capabilities:**
- **Feature Extraction:** Enhance GPT-4 prompts for better visual feature extraction
- **Psychology Classification:** Update psychology keyword mappings and rules
- **Pattern Discovery:** Implement new mining algorithms for feature combinations
- **Performance Optimization:** Improve processing speed and memory usage

---

### 3. Reviewer Agent (逻辑警察与合规官)

**Role:** Code Quality & Compliance Guardian

**Objective:** Maintain code quality, system safety, and prevent regressions.

**Responsibilities:**
- Static code audit and semantic analysis
- Architecture compliance (ensure changes follow design patterns)
- Data privacy checks (prevent PII leaks)
- Bias and fairness validation

**Context Window:**
- Coder Agent's PR with full diff
- Current codebase architecture
- Security and compliance guidelines

**Toolset:**
- `static_analysis(diff)` - Run linters and type checkers
- `architecture_check(diff)` - Verify design pattern compliance
- `security_scan(diff)` - Check for security vulnerabilities
- `test_coverage_check()` - Ensure tests cover new code
- `approve_pr(pr_id)` or `reject_pr(pr_id, reason)` - PR decision

**System Prompt Components:**
```yaml
role: "Code Quality & Compliance Auditor"
expertise:
  - "Software architecture and design patterns"
  - "Data privacy and security best practices"
  - "Statistical validity and bias detection"
constraints:
  - "MUST reject: Any hardcoded test data"
  - "MUST reject: Changes that reduce test coverage"
  - "MUST reject: Modifications to core data validators without review"
output_format: "JSON with approval decision and rationale"
```

**Review Criteria:**
- **Architecture:** Does the change follow SOLID principles?
- **Performance:** Will this improve or degrade processing speed?
- **Security:** Are there any PII leaks or vulnerabilities?
- **Maintainability:** Is the code readable and well-documented?
- **Test Coverage:** Are there tests for the new logic?

---

### 4. Judge Agent (结果衡量与压力测试)

**Role:** Performance Evaluator & Reality Checker

**Objective:** objectively evaluate mining quality and break Coder Agent's illusions.

**Responsibilities:**
- Run backtests on golden set of high-performing ads
- Test on real-time bad cases (ads that should perform but don't)
- Compare pattern recommendations against actual ROAS/CTR
- Calculate lift scores and regression detection

**Context Window:**
- Historical mining results (baseline)
- Current experiment results
- Real-world performance metrics (CTR, ROAS, engagement)

**Toolset:**
- `run_backtest(branch_name, golden_set)` - Execute backtest suite
- `query_real_ctr(pattern_category)` - Get actual business metrics
- `calculate_lift_score(baseline, experiment)` - Compute improvement
- `detect_regressions(experiment_results)` - Find performance drops
- `generate_audit_report()` - Produce detailed evaluation

**System Prompt Components:**
```yaml
role: "Adversarial Quality Assurance Evaluator"
expertise:
  - "Statistical hypothesis testing"
  - "A/B testing and experimental design"
  - "Business metrics analysis (CTR, ROAS, conversion)"
constraints:
  - "MUST test on unseen data (no train-test leakage)"
  - "MUST verify no regression on existing high-performing patterns"
  - "MUST require statistical significance (p < 0.05)"
output_format: "JSON with lift score and pass/fail decision"
```

**Evaluation Logic:**
```python
IF lift_score > 10% AND complexity_increase < 20% AND no_regressions:
    RETURN {"decision": "PASS", "lift": lift_score, "reason": "..."}

ELSE IF new_error_rate > 5% OR regression_detected:
    RETURN {"decision": "FAIL", "reason": "Regression detected in ..."}

ELSE IF lift_score < 5%:
    RETURN {"decision": "FAIL", "reason": "Insufficient improvement"}
```

**Golden Set:**
- Top 100 performing ads across all customers
- Edge cases: vertical-specific patterns (gaming, e-commerce, etc.)
- Seasonal variations (holiday vs non-holiday)

**Bad Cases Tracking:**
- Ads predicted to perform well but didn't (false positives)
- High-performing ads that patterns missed (false negatives)

---

### 5. Memory Agent (组织记忆与知识库)

**Role:** Organizational Memory & Knowledge Base

**Objective:** Prevent organizational amnesia and accelerate evolution through historical experience.

**Responsibilities:**
- Record every experiment's input-process-output
- Index successful and failed patterns
- Retrieve relevant historical experiments for PM and Coder
- Issue warnings when current path matches historical failures

**Context Window:**
- All historical experiments
- Code change history with outcomes
- Performance metrics over time

**Toolset:**
- `store_experiment(experiment_id, data)` - Archive experiment results
- `search_similar(query, top_k)` - Retrieve semantically similar experiments
- `check_failure_pattern(current_spec)` - Detect repeating failure modes
- `get_successful_patterns(domain)` - Retrieve winning approaches
- `index_code_change(diff, outcome)` - Store code evolution

**System Prompt Components:**
```yaml
role: "Knowledge Management & Historical Experience Librarian"
expertise:
  - "Vector database and semantic search"
  - "Pattern recognition in experimental outcomes"
  - "Knowledge graph construction"
constraints:
  - "MUST maintain causal links (code change → performance impact)"
  - "MUST warn when current experiment matches 2+ past failures"
  - "MUST prioritize recent experiments (last 6 months)"
output_format: "JSON with retrieved experiments and warnings"
```

**Memory Structure:**
```yaml
experiment_record:
  id: "exp_20250203_001"
  timestamp: "2025-02-03T12:00:00Z"
  spec:
    pm_agent: "Improve psychology classification"
    parameters: {quantile: 0.80, confidence: 0.70}
  code_changes:
    file: "psychology_classifier.py"
    diff: "..."
  results:
    lift_score: 15.2
    regression: false
    judge_decision: "PASS"
  lessons_learned:
    - "Using GPT-4 for psychology extraction improved accuracy by 20%"
    - "Rule-based fallback needed for VLM failures"
```

---

### 6. Monitor Agent (NEW - 系统监控与异常检测)

**Role:** System Health Monitor & Anomaly Detector

**Objective:** Continuously monitor mining pipeline health and detect anomalies in real-time.

**Responsibilities:**
- Monitor processing times, error rates, and resource usage
- Detect data quality issues (missing features, outliers)
- Alert on performance degradation
- Track model drift over time

**Context Window:**
- Real-time pipeline metrics
- Historical baseline metrics
- System health logs

**Toolset:**
- `get_pipeline_metrics()` - Fetch current performance stats
- `detect_anomalies(metrics)` - Identify statistical anomalies
- `check_data_quality()` - Validate input data
- `send_alert(severity, message)` - Trigger notifications
- `generate_health_report()` - Produce system status

**System Prompt Components:**
```yaml
role: "System Health & Anomaly Detection Specialist"
expertise:
  - "Statistical process control"
  - "Anomaly detection algorithms"
  - "Performance monitoring and alerting"
constraints:
  - "MUST alert on error rate > 5%"
  - "MUST detect data drift (feature distribution changes)"
  - "MUST track model performance degradation over time"
output_format: "JSON with health status and alerts"
```

---

## Evolution Loop (核心演进闭环)

### Phase 1: Observation (观测层)
```
Monitor Agent detects anomaly:
  "Psychology classification accuracy dropped 15% for gaming ads"
      │
      ▼
Judge Agent confirms:
  "Baseline: 82% accuracy → Current: 67% accuracy (p < 0.01)"
```

### Phase 2: Cognition (认知层)
```
Memory Agent retrieves history:
  "Found 3 similar experiments:
   - exp_20241101: Added gaming-specific keywords → +22% lift
   - exp_20241015: Fine-tuned GPT-4 prompt → -5% lift (overfitting)
   - exp_20240920: Used ensemble approach → +18% lift"

PM Agent creates spec:
  {
    "objective": "Improve gaming psychology classification",
    "approach": "Add gaming-specific visual keywords",
    "boundary": "Only modify psychology_classifier.py",
    "success_criteria": {"accuracy_improvement": ">10%"}
  }
```

### Phase 3: Production (生产层)
```
Coder Agent implements:
  File: psychology_classifier.py
  Changes:
    + PSYCHOLOGY_KEYWORDS["Social_Proof"]["visual"].extend([
    +   "gameplay", "multiplayer", "esports", "streaming", "leaderboard"
    + ])

Reviewer Agent validates:
  ✓ Architecture: Follows existing pattern
  ✓ Security: No PII leaks
  ✓ Tests: Added unit tests for new keywords
  ✓ Performance: No significant slowdown
  → APPROVED
```

### Phase 4: Validation (验证层)
```
Judge Agent runs backtest:
  Golden Set: 1000 gaming ads
  Baseline: 67% accuracy
  Experiment: 79% accuracy
  Lift: +18% (p < 0.001)

  Regression Check:
  - E-commerce ads: 82% → 81% (no significant regression)
  - Lifestyle ads: 78% → 79% (no significant regression)

  → PASS
```

### Phase 5: Landing (落地层)
```
Memory Agent archives:
  exp_20250203_gaming_psychology:
    approach: "Added gaming-specific visual keywords"
    lift_score: 18.0
    regression: false
    lessons:
      - "Domain-specific keywords significantly improve accuracy"
      - "No cross-domain performance impact"

  Auto-merge PR → Deploy to production
```

---

## Adversarial Mechanisms (对抗机制)

### Coder vs Judge
- **Coder's Goal:** Maximize lift score by any means
- **Judge's Goal:** Find flaws, overfitting, and regressions
- **Dynamic:** Coder must create robust solutions, not clever hacks

### Reviewer vs Coder
- **Reviewer's Goal:** Maintain code quality and prevent technical debt
- **Coder's Goal:** Ship features quickly
- **Dynamic:** Forces clean, maintainable code

### Memory vs PM
- **Memory's Goal:** Prevent repeating mistakes
- **PM's Goal:** Try new approaches
- **Dynamic:** Balances exploration with exploitation

---

## Knowledge Leverage (经验杠杆)

**Without Memory:**
- Random experimentation
- Repeating failures
- Slow convergence

**With Memory:**
- "Last time we added gaming keywords, lift was +22%"
- "Warning: This approach failed 3 times in the past"
- Accelerated learning through historical data

---

## Implementation Stack

### Agent Framework
- **LangGraph** for agent orchestration
- **LangChain** for tool integration
- **Vector DB** (ChromaDB/Pinecone) for Memory Agent

### State Management
- **Redis** for agent state and coordination
- **PostgreSQL** for experiment records
- **S3** for code artifacts and models

### Tools Integration
- **Git** for PR management
- **GitHub Actions** for CI/CD
- **Pytest** for testing
- **Prometheus** for monitoring

---

## File Structure

```
src/meta/ad/miner/agents/
├── README.md                              # This file
├── __init__.py
├── orchestrator.py                        # Main agent coordinator
│
├── pm_agent/                              # PM Agent
│   ├── __init__.py
│   ├── agent.py                           # PM Agent implementation
│   ├── prompts.py                         # System prompts
│   └── tools.py                           # Query tools
│
├── coder_agent/                           # Coder Agent
│   ├── __init__.py
│   ├── agent.py                           # Coder implementation
│   ├── prompts.py                         # Coding prompts
│   └── tools.py                           # File operations, Git, tests
│
├── reviewer_agent/                        # Reviewer Agent
│   ├── __init__.py
│   ├── agent.py                           # Reviewer implementation
│   ├── prompts.py                         # Review prompts
│   └── tools.py                           # Linting, architecture checks
│
├── judge_agent/                           # Judge Agent
│   ├── __init__.py
│   ├── agent.py                           # Judge implementation
│   ├── prompts.py                         # Evaluation prompts
│   └── tools.py                           # Backtesting, metrics
│
├── memory_agent/                          # Memory Agent
│   ├── __init__.py
│   ├── agent.py                           # Memory implementation
│   ├── prompts.py                         # Retrieval prompts
│   ├── vector_store.py                    # Vector DB wrapper
│   └── tools.py                           # Store, search, index
│
├── monitor_agent/                         # Monitor Agent
│   ├── __init__.py
│   ├── agent.py                           # Monitor implementation
│   ├── prompts.py                         # Alerting prompts
│   └── tools.py                           # Metrics, anomaly detection
│
├── shared/                                # Shared utilities
│   ├── __init__.py
│   ├── state.py                           # Agent state management
│   ├── messaging.py                       # Inter-agent communication
│   └── utils.py                           # Common utilities
│
└── config/                                # Agent configurations
    ├── agents_config.yaml                 # Global agent settings
    ├── pm_config.yaml
    ├── coder_config.yaml
    ├── reviewer_config.yaml
    ├── judge_config.yaml
    ├── memory_config.yaml
    └── monitor_config.yaml
```

---

## Quick Start

### Initialize Agents
```python
from src.meta.ad.miner.agents import Orchestrator

# Initialize orchestrator with all agents
orchestrator = Orchestrator(
    pm_agent_config="config/pm_config.yaml",
    coder_agent_config="config/coder_config.yaml",
    reviewer_agent_config="config/reviewer_config.yaml",
    judge_agent_config="config/judge_config.yaml",
    memory_agent_config="config/memory_config.yaml",
    monitor_agent_config="config/monitor_config.yaml",
)

# Start autonomous evolution loop
orchestrator.run_evolution_loop(
    objective="improve_psychology_classification",
    domain="gaming_ads",
    max_iterations=10
)
```

### Manual Agent Invocation
```python
# PM Agent creates experiment spec
pm_agent = PMAgent()
spec = pm_agent.create_experiment_spec(
    objective="Improve gaming psychology accuracy",
    constraints={"only_modify": ["psychology_classifier.py"]}
)

# Coder Agent implements
coder_agent = CoderAgent()
pr = coder_agent.implement_spec(spec)

# Reviewer Agent validates
reviewer_agent = ReviewerAgent()
decision = reviewer_agent.review_pr(pr.id)

if decision.approved:
    # Judge Agent evaluates
    judge_agent = JudgeAgent()
    results = judge_agent.evaluate_experiment(pr.id)

    # Memory Agent archives
    memory_agent = MemoryAgent()
    memory_agent.archive_experiment(spec, pr, results)
```

---

## Success Metrics

- **Pattern Lift:** Average lift of recommended patterns vs baseline
- **Psychology Accuracy:** Classification accuracy for psychology types
- **Processing Speed:** Time to mine patterns from N ads
- **Experiment Success Rate:** % of experiments that pass Judge Agent
- **Regression Rate:** % of experiments that cause performance drops
- **Knowledge Reuse:** % of experiments that leverage Memory Agent insights

---

## Next Steps

1. **Implement Orchestrator** - Create main agent coordinator
2. **Develop Toolsets** - Build tool integrations for each agent
3. **Setup Memory System** - Configure vector DB and indexing
4. **Create Test Suite** - Golden set for Judge Agent
5. **Deploy Monitor Agent** - Real-time system health tracking
6. **Run Pilot Evolution** - First autonomous improvement cycle

---

**Goal:** Create a self-improving ad mining system that continuously discovers high-impact creative patterns without human intervention.
