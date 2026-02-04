"""
Team Agents - Template for Creating Multi-Agent Development Teams

This package provides a template and guide for creating multi-agent development
teams for any algorithm. It is NOT the actual implementation - those live in
their respective algorithm directories (e.g., src/meta/ad/generator/agents/).

## Purpose

The team agents pattern enables closed-loop code evolution through specialized
AI agents working together:

    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │  Judge  │─────▶│    PM   │─────▶│  Coder  │
    └─────────┘      └─────────┘      └─────────┘
         ▲                                    │
         │                                    ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Memory  │◀────│Reviewer │◀─────│    PR   │
    └─────────┘      └─────────┘      └─────────┘

## Workflow

1. **Judge Agent** detects issues → creates findings
2. **PM Agent** + Memory → creates experiment spec
3. **Coder Agent** → implements changes → creates PR
4. **Reviewer Agent** → approves/rejects PR
5. **Judge Agent** → evaluates results → makes decision
6. **Memory Agent** → records learnings
7. Loop back to step 1

## Creating Your Own Team

Copy this template structure to your algorithm directory:

    src/meta/YOUR_ALGORITHM/agents/
    ├── __init__.py           # Export all agents
    ├── pm_agent.py           # Product Manager
    ├── coder_agent.py        # Implementation engineer
    ├── reviewer_agent.py     # Quality/compliance reviewer
    ├── judge_agent.py        # Testing & evaluation
    ├── memory_agent.py       # Knowledge base
    └── orchestrator.py       # Workflow coordinator

## Example: Ad Generator Team

See `src/meta/ad/generator/agents/` for a complete working example.

## Key Design Patterns

### 1. Dataclass Types
All agents use dataclasses for clear, type-safe interfaces:

    @dataclass
    class ExperimentSpec:
        spec_id: str
        title: str
        component: Component
        ...

### 2. Enum-Based Categories
Use enums for fixed sets of values:

    class Component(Enum):
        AD_MINER = "ad_miner"
        AD_GENERATOR = "ad_generator"
        ...

### 3. Factory Functions
Provide factory functions for easy instantiation:

    def create_pm_agent(
        memory_client: Optional[MemoryClient] = None,
        **kwargs
    ) -> PMAgent:
        return PMAgent(memory_client=memory_client, **kwargs)

### 4. Relative Imports
Within the agents package, use relative imports:

    from .pm_agent import ExperimentSpec
    from .coder_agent import PullRequest

### 5. Quality Gates
Each agent has verification and validation:

    class CoderAgent:
        def implement_spec(self, spec: ExperimentSpec):
            # Generate changes
            # Check quality
            # Verify constraints
            # Return result

## Agent Responsibilities

### PM Agent
- Analyze findings and historical context
- Create experiment specifications
- Define scope and constraints
- Set success criteria

### Coder Agent
- Implement experiment specifications
- Generate code changes
- Ensure code quality
- Run tests and linting

### Reviewer Agent
- Review pull requests
- Check architecture compliance
- Verify security constraints
- Validate against spec

### Judge Agent
- Run backtests
- Evaluate performance
- Make merge decisions
- Generate new findings

### Memory Agent
- Store all experiment records
- Provide historical context
- Track patterns and learnings
- Enable retrieval by similarity

### Orchestrator
- Coordinate all agents
- Manage workflow state
- Handle callbacks
- Track experiment progress

## Common Pitfalls

### DON'T: Hard-code algorithm logic
    # Bad
    def create_spec(self):
        return ExperimentSpec(
            component="ad_generator",  # Hard-coded!
            ...
        )

### DO: Make it configurable
    # Good
    def create_spec(self, component: Component):
        return ExperimentSpec(
            component=component,
            ...
        )

### DON'T: Use absolute imports
    # Bad
    from meta.ad.generator.agents.pm_agent import ExperimentSpec

### DO: Use relative imports
    # Good
    from .pm_agent import ExperimentSpec

### DON'T: Skip type hints
    # Bad
    def create_spec(self, findings):
        ...

### DO: Use type hints
    # Good
    def create_spec(self, findings: JudgeFindings) -> ExperimentSpec:
        ...

## Testing

Each agent should have unit tests:

    tests/unit/meta/YOUR_ALGORITHM/agents/
    ├── test_pm_agent.py
    ├── test_coder_agent.py
    ├── test_reviewer_agent.py
    ├── test_judge_agent.py
    ├── test_memory_agent.py
    └── test_orchestrator.py

## Further Reading

See the complete guide: src/agents/creator.md

Author: Devease Dev Team
Date: 2026-02-04
"""

# This package is a template - no actual implementations here
# Import your algorithm-specific team from its canonical location:
# from meta.YOUR_ALGORITHM.agents import create_orchestrator

__all__ = []
