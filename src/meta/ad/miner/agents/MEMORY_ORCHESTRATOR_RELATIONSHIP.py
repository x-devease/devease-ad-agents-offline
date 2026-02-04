"""
Memory Agent x Orchestrator: Correct Relationship Design

Memory Agent should be an INDEPENDENT SERVICE that all agents access directly,
NOT a component owned by Orchestrator.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ARCHITECTURE PATTERN 1: Memory as Independent Service (✅ RECOMMENDED)
# =============================================================================

class MemoryAgent:
    """
    Independent Knowledge Service.

    ALL agents (PM, Coder, Reviewer, Judge, Monitor) access Memory directly.
    Orchestrator does NOT own or control Memory.
    """

    _instance = None  # Singleton pattern

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.experiments = {}
        self.embeddings = {}
        self._initialized = True

        logger.info("Memory Agent: Independent service initialized")

    def store(self, experiment_id: str, data: Dict[str, Any]):
        """Store experiment - accessible by ALL agents."""
        self.experiments[experiment_id] = data
        logger.info(f"Memory: Stored {experiment_id}")

    def search(self, query: str) -> List[Dict]:
        """Search experiments - accessible by ALL agents."""
        # Search logic here
        results = [
            exp for exp_id, exp in self.experiments.items()
            if query.lower() in exp.get("objective", "").lower()
        ]
        return results


class AgentOrchestrator:
    """
    System Coordinator - does NOT own Memory.

    Orchestrator coordinates the FLOW between agents,
    but Memory is accessed directly by each agent.
    """

    def __init__(self, memory_service: MemoryAgent):
        """
        Initialize with external Memory service.

        Args:
            memory_service: Independent Memory Agent instance
        """
        # Memory is NOT owned by Orchestrator
        self.memory = memory_service  # ✅ Reference to external service

        # Initialize agents with direct Memory access
        self.pm_agent = PMAgent(memory=self.memory)  # ✅ Direct access
        self.coder_agent = CoderAgent(memory=self.memory)  # ✅ Direct access
        self.reviewer_agent = ReviewerAgent(memory=self.memory)  # ✅ Direct access
        self.judge_agent = JudgeAgent(memory=self.memory)  # ✅ Direct access
        self.monitor_agent = MonitorAgent(memory=self.memory)  # ✅ Direct access

        logger.info("Orchestrator: Initialized with external Memory service")

    def run_evolution_cycle(self, objective: str):
        """Coordinate flow between agents."""

        # Phase 1: Observation
        anomaly = self.monitor_agent.detect_anomaly()

        # Phase 2: Cognition
        # PM Agent directly queries Memory (not through Orchestrator)
        spec = self.pm_agent.create_spec(
            objective=objective,
            context=anomaly,
            # PM Agent internally calls: self.memory.search(...)
        )

        # Phase 3: Production
        # Coder Agent directly queries Memory for past code changes
        pr = self.coder_agent.implement(spec)
        # Coder Agent internally calls: self.memory.search_similar_code(...)

        # Phase 4: Validation
        # Reviewer Agent directly queries Memory for past failures
        decision = self.reviewer_agent.review(pr)
        # Reviewer Agent internally calls: self.memory.check_failure_patterns(...)

        # Judge Agent directly queries Memory for baselines
        results = self.judge_agent.evaluate(pr)
        # Judge Agent internally calls: self.memory.get_baselines(...)

        # Phase 5: Landing
        # All agents can directly update Memory
        self.memory.store_experiment(
            experiment_id=spec.id,
            data={
                "spec": spec,
                "pr": pr,
                "results": results,
            }
        )

        return results


# =============================================================================
# ARCHITECTURE PATTERN 2: Dependency Injection (✅ ALSO GOOD)
# =============================================================================

class PMAgent:
    """PM Agent with direct Memory access."""

    def __init__(self, memory: MemoryAgent):
        """
        Initialize with Memory dependency.

        Args:
            memory: Independent Memory service (injected)
        """
        self.memory = memory  # ✅ Direct reference, not through Orchestrator
        logger.info("PM Agent: Initialized with direct Memory access")

    def create_spec(self, objective: str, context: Dict):
        """Create experiment spec."""

        # ✅ Directly query Memory (not through Orchestrator)
        similar = self.memory.search(f"{objective}")

        # Check for failure patterns
        warnings = self.memory.check_failure_patterns(objective)

        return {
            "objective": objective,
            "approach": self._select_approach(similar, warnings),
            "historical_context": similar,
        }


class CoderAgent:
    """Coder Agent with direct Memory access."""

    def __init__(self, memory: MemoryAgent):
        self.memory = memory
        logger.info("Coder Agent: Initialized with direct Memory access")

    def implement(self, spec: Dict):
        """Implement experiment spec."""

        # ✅ Directly query Memory for similar past code
        similar_code = self.memory.search_similar_code(spec["approach"])

        # Learn from past implementations
        lessons = self.memory.get_implementation_lessons(spec["objective"])

        # Generate code based on historical success
        code = self._generate_code(spec, similar_code, lessons)

        return {"pr_id": "pr_123", "code": code}


class ReviewerAgent:
    """Reviewer Agent with direct Memory access."""

    def __init__(self, memory: MemoryAgent):
        self.memory = memory
        logger.info("Reviewer Agent: Initialized with direct Memory access")

    def review(self, pr: Dict):
        """Review pull request."""

        # ✅ Directly query Memory for past failures with similar changes
        failure_patterns = self.memory.check_failure_patterns(pr["code"])

        # Check architectural compliance
        compliance = self.memory.get_architecture_rules()

        return {
            "decision": "APPROVE" if not failure_patterns else "REJECT",
            "warnings": failure_patterns,
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Demonstrate correct Memory x Orchestrator relationship."""

    print("\n" + "="*80)
    print("MEMORY AGENT × ORCHESTRATOR: CORRECT ARCHITECTURE")
    print("="*80)

    # Step 1: Initialize Memory as independent service (singleton)
    memory = MemoryAgent()  # ✅ Independent service
    print("\n✅ Step 1: Memory Agent initialized as independent service")

    # Step 2: Initialize Orchestrator WITH Memory (not owning it)
    orchestrator = AgentOrchestrator(memory_service=memory)
    print("✅ Step 2: Orchestrator initialized with Memory reference")

    # Step 3: All agents have direct Memory access
    print("\n✅ Step 3: All agents have direct Memory access:")
    print("   - PM Agent can query Memory directly")
    print("   - Coder Agent can query Memory directly")
    print("   - Reviewer Agent can query Memory directly")
    print("   - Judge Agent can query Memory directly")
    print("   - Monitor Agent can query Memory directly")

    # Step 4: Run evolution cycle
    print("\n✅ Step 4: Running evolution cycle...")
    results = orchestrator.run_evolution_cycle(
        objective="improve_psychology_classification"
    )

    print(f"\n✅ Complete! Results: {results}")

    # Step 5: Memory persists beyond Orchestrator lifecycle
    print("\n✅ Step 5: Memory persists beyond Orchestrator:")
    print(f"   Total experiments in Memory: {len(memory.experiments)}")
    print("   Orchestrator can be destroyed, Memory remains")

    print("\n" + "="*80)
    print("KEY INSIGHT: Memory is a SERVICE, not a COMPONENT")
    print("="*80)


# =============================================================================
# COMPARISON: WRONG vs RIGHT
# =============================================================================

COMPARISON = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE COMPARISON                                  │
└─────────────────────────────────────────────────────────────────────────────┘

❌ WRONG: Orchestrator OWNS Memory
──────────────────────────────────────────────────────────────────────────────
   Orchestrator
   ├── Memory Agent  ← Created by Orchestrator
   ├── PM Agent      → Must go through Orchestrator to access Memory
   ├── Coder Agent   → Must go through Orchestrator to access Memory
   └── Judge Agent   → Must go through Orchestrator to access Memory

   Problems:
   • Tight coupling
   • Orchestrator becomes bottleneck
   • Memory not accessible outside Orchestrator context
   • Violates single responsibility principle

✅ RIGHT: Memory is INDEPENDENT SERVICE
──────────────────────────────────────────────────────────────────────────────
   Memory Agent  ← Independent service (singleton)
   ↑    ↑    ↑    ↑    ↑
   │    │    │    │    └───────────────────┐
   │    │    │    └─────────────────┐     │
   │    │    └──────────────┐     │     │
   │    └───────────┐       │     │     │
PM     Coder      Reviewer  Judge  Monitor  Orchestrator (coordinator)

   Benefits:
   • Loose coupling
   • All agents access Memory directly
   • Memory persists beyond any agent lifecycle
   • Follows microservice patterns
   • Easy to test (mock Memory)

✅ KEY RELATIONSHIP:
──────────────────────────────────────────────────────────────────────────────
   Memory Agent = Knowledge Service (like a database)
   Orchestrator = Workflow Coordinator (like a conductor)

   Memory does NOT depend on Orchestrator
   Orchestrator REFERENCES Memory (but doesn't own it)
   All agents can access Memory independently

🎯 ANALOGY:
──────────────────────────────────────────────────────────────────────────────
   Memory Agent  = Library (everyone can read/write books)
   Orchestrator  = Librarian (coordinates who uses which room)

   You don't need the librarian to read a book!
   You can walk into the library directly.
"""

if __name__ == "__main__":
    print(COMPARISON)
    main()
