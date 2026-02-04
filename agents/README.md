# Ad Generator Development Team - AI Agents

Self-evolving code system for the ad/generator module. This team of AI agents works together to continuously improve the codebase through automated experiments.

## Architecture Overview

The development team implements a closed-loop evolution process:

```
┌─────────────────────────────────────────────────────────────────┐
│                     🔄 Continuous Evolution Loop                │
└─────────────────────────────────────────────────────────────────┘

  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
  │  Judge  │────▶│    PM   │────▶│  Coder  │────▶│Reviewer │
  │  Agent  │     │  Agent  │     │  Agent  │     │  Agent  │
  └─────────┘     └─────────┘     └─────────┘     └─────────┘
       │               │                              │
       │               ▼                              ▼
       │          ┌─────────┐                   ┌─────────┐
       └─────────▶│ Memory  │◀──────────────────│ Judge   │
                  │  Agent  │                   |  Agent  │
                  └─────────┘                   └─────────┘
```

## Team Members

### 1. PM Agent (Product Manager)
**Role:** Requirements and Experiment Planning

**Responsibilities:**
- Translates Judge Agent findings into experiment specs
- Sets logical boundaries for changes (Prompt vs Logic vs Config)
- Maximizes experiment ROI while controlling risk
- Retrieves historical context from Memory Agent

**Key Features:**
- Component-specific risk tolerances
- Scope-based constraints (PROMPT_ONLY, LOGIC_ONLY, etc.)
- Historical failure awareness

### 2. Coder Agent (Implementation Engineer)
**Role:** Code Implementation

**Responsibilities:**
- Implements experiment specs from PM Agent
- Modifies Python/Prompt/SQL code
- Creates Pull Requests with documentation
- Avoids hardcoding and test overfitting

**Key Features:**
- Code pattern analysis
- Forbidden pattern detection
- Quality assessment

### 3. Reviewer Agent (Quality & Compliance Officer)
**Role:** Code Review and Compliance

**Responsibilities:**
- Static code analysis and security checks
- Architecture compliance verification
- CI/CD pipeline validation
- Prompt leakage prevention
- Data bias detection

**Key Features:**
- Architecture pattern enforcement
- Security vulnerability scanning
- Compliance checking

### 4. Judge Agent (Quality Evaluator)
**Role:** Quality Evaluation and Testing

**Responsibilities:**
- Runs automated backtests
- Evaluates performance metrics (CTR, ROAS, Lift Score)
- Tests against Golden Set and real Bad Cases
- Detects regressions and side effects
- Makes merge/no-merge decisions

**Key Features:**
- Component-specific metrics
- Lift score calculation
- Regression detection

### 5. Memory Agent (Knowledge Base)
**Role:** Organizational Memory

**Responsibilities:**
- Records all experiment inputs, processes, and results
- Retrieves relevant historical experiments
- Detects repeated failure patterns
- Provides context and learnings

**Key Features:**
- Experiment record storage
- Pattern detection
- Historical context retrieval

### 6. Orchestrator (Team Coordinator)
**Role:** Workflow Management

**Responsibilities:**
- Coordinates all agents
- Manages experiment lifecycle
- Enforces closed-loop process
- Handles agent communication

**Key Features:**
- Multiple operation modes (continuous, single, supervised)
- Callback system for human oversight
- State tracking

## Usage

### Basic Setup

```python
from pathlib import Path
from agents import create_orchestrator, OrchestratorMode, JudgeFindings, Component, ExperimentPriority

# Create orchestrator
orchestrator = create_orchestrator(
    repo_path=Path("."),
    mode=OrchestratorMode.SUPERVISED,  # Requires human approval
    memory_db_path=Path("data/agents/memory.json"),
)
```

### Running a Single Experiment

```python
# Create findings from issues
findings = JudgeFindings(
    issue_type="performance_drop",
    component=Component.AD_GENERATOR,
    severity="high",
    description="Prompt quality score decreased by 15%",
    evidence={"current_score": 0.70, "baseline_score": 0.85},
    suggested_priority=ExperimentPriority.HIGH,
)

# Run experiment
result = await orchestrator.run_experiment_from_findings(findings)

print(f"Experiment: {result.experiment_id}")
print(f"Success: {result.success}")
print(f"Lift: {result.lift_score:.2%}")
print(f"Approved: {result.approved}")
```

### Running Continuous Mode

```python
# Run multiple experiments automatically
results = await orchestrator.run_continuous(
    max_experiments=10,
    min_lift_threshold=0.01,  # 1% minimum lift
)

for result in results:
    print(f"{result.experiment_id}: {result.lift_score:.2%}")
```

### Supervised Mode with Callbacks

```python
def on_spec_created(data):
    spec = data["spec"]
    print(f"Spec created: {spec.title}")
    response = input("Approve? (y/n): ")
    return response.lower() == 'y'

orchestrator.register_callback("spec_created", on_spec_created)
```

## Experiment Workflow

1. **Detection:** Judge Agent detects issues → creates findings
2. **Planning:** PM Agent receives findings → queries Memory → creates spec
3. **Implementation:** Coder Agent implements spec → creates PR
4. **Review:** Reviewer Agent reviews PR → approves/rejects
5. **Evaluation:** Judge Agent evaluates → makes merge decision
6. **Learning:** Memory Agent records results → feeds back to step 2

## Component-Specific Behavior

Each agent is tailored to the specific components of the ad/generator system:

- **AD_MINER:** Feature extraction, ROAS prediction
- **AD_GENERATOR:** Prompt generation, image quality
- **ADSET_ALLOCATOR:** Budget allocation, safety rules
- **ADSET_GENERATOR:** Audience configuration
- **NANO_BANANA_PRO:** Prompt enhancement, quality verification
- **SHARED_UTILS:** Common utilities (high risk)
- **FRAMEWORK:** Agent framework (highest risk)

## Data Storage

Memory is stored in `data/agents/memory.json` as a JSON array of experiment records:

```json
[
  {
    "experiment_id": "exp-abc123",
    "spec_id": "spec-def456",
    "pr_id": "pr-ghi789",
    "component": "ad_generator",
    "outcome": "success",
    "lift_score": 0.12,
    "lessons_learned": [...],
    "tags": ["approved", "high_lift"]
  }
]
```

## Adversarial Dynamic

The system implements an adversarial relationship:
- **Coder Agent** tries to maximize lift and pass tests
- **Judge Agent** tries to find issues and detect regressions
- **Reviewer Agent** acts as the neutral arbiter

This dynamic ensures genuine improvement rather than gaming the system.

## Configuration

Agents can be configured with YAML files:

```yaml
# config/agents/dev_team_config.yaml
pm_agent:
  risk_tolerances:
    ad_generator:
      max_prompt_changes: 10
      requires_visual_review: true

coder_agent:
  forbidden_patterns:
    - test_specific_id
    - hardcoded_threshold

reviewer_agent:
  quality_thresholds:
    min_coverage: 0.80
    max_complexity: 10
```

## License

Copyright (c) 2026 Ad System Development Team
