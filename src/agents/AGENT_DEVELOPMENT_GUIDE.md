# Agent Development Guide

**Comprehensive guide for building algorithm-specific agents**

Learn how to create AI agents by studying patterns from three production systems:
- **ad/miner agents** - Generic framework with adapter pattern
- **diagnoser agents** - Multi-agent optimization system
- **ad/generator agents** - Autonomous code evolution team

---

## Table of Contents

1. [Introduction to Agent Architectures](#chapter-1-introduction)
2. [Core Foundations](#chapter-2-core-foundations)
3. [Architecture Pattern 1: Pipeline Agent](#chapter-3-pipeline-agent)
4. [Architecture Pattern 2: Adapter-Based Framework](#chapter-4-adapter-framework)
5. [Architecture Pattern 3: Multi-Agent Orchestration](#chapter-5-multi-agent-orchestration)
6. [Specialization Patterns](#chapter-6-specialization)
7. [Quality Assurance Patterns](#chapter-7-quality-assurance)
8. [Learning and Memory](#chapter-8-learning-memory)
9. [Real-World Examples](#chapter-9-real-world-examples)
10. [Best Practices and Anti-Patterns](#chapter-10-best-practices)
11. [Quick Reference](#chapter-11-quick-reference)

---

## Chapter 1: Introduction {#chapter-1-introduction}

### What is an Agent?

An **agent** is an autonomous system that:
- Receives inputs (data, prompts, specifications)
- Processes using domain-specific algorithms
- Produces structured outputs with quality assurance
- Learns from past interactions (optional)

### Three Architectural Patterns

#### Pattern 1: Pipeline Agent (Nano-style)

```
Input → Parse → Analyze → Transform → Verify → Output
```

**Best for:** Single-domain transformations with clear sequential steps

**Example:** PromptEnhancementAgent in `ad/miner/src/agents/nano/`

**Characteristics:**
- Linear, easy-to-follow pipeline
- Deep domain expertise
- Rich type system
- Quality verification built-in

---

#### Pattern 2: Adapter-Based Framework (ad/miner-style)

```
Generic Core (BaseAgent)
    ↓
Domain Adapter (BaseAdapter)
    ↓
Generic Components (Memory, Examples, Reflexion)
```

**Best for:** Multi-domain systems requiring reusability

**Example:** BaseAgent framework in `ad/miner/src/agents/framework/`

**Characteristics:**
- Domain-agnostic core
- Pluggable adapters
- Research-backed features (Reflexion, Memory, Examples)
- Composable components

---

#### Pattern 3: Multi-Agent Orchestration (Diagnoser/Dev Team-style)

```
Judge → PM → Memory → Coder → Reviewer → Judge (loop)
        ↓
    Orchestrator coordinates all
```

**Best for:** Complex workflows requiring specialized roles

**Example:**
- Diagnoser orchestrator in `src/meta/diagnoser/agents/orchestrator.py`
- Dev team in `agents/orchestrator.py`

**Characteristics:**
- Role-based agents (PM, Coder, Reviewer, Judge, Memory)
- Closed-loop evolution
- Comprehensive quality gates
- Risk-aware constraints

---

### Choosing the Right Architecture

| Your Scenario | Recommended Pattern |
|---------------|---------------------|
| Single domain, linear transformation | Pipeline Agent |
| Multiple similar domains, need reusability | Adapter Framework |
| Complex workflow with specialized roles | Multi-Agent System |
| Need continuous autonomous optimization | Multi-Agent System |
| Quick prototype, single use case | Pipeline Agent |

---

## Chapter 2: Core Foundations {#chapter-2-core-foundations}

All agent architectures share these foundational patterns.

### Pattern 1: Dataclass-Based Type System

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

@dataclass
class AgentInput:
    """Structured input with clear typing."""

    # Required fields
    core_input: str

    # Optional context
    optional_context: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AgentOutput:
    """Structured output with observability."""

    # Core result
    result: str

    # Quality metrics
    confidence: float = 0.0  # 0.0 to 1.0
    processing_time_ms: int = 0

    # What was applied
    techniques_used: List[str] = field(default_factory=list)

    # Tracking
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
```

**Benefits:**
- Strong typing with automatic `__init__`
- Easy serialization with `to_dict()` method
- Clear field documentation
- Default values for optional fields

---

### Pattern 2: Enum-Based Categorization

```python
from enum import Enum

class AgentStatus(Enum):
    """Agent states for workflow tracking."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"

class IssueSeverity(Enum):
    """Issue severity for prioritization."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class OutputCategory(Enum):
    """Output categorization."""
    PRODUCT_PHOTOGRAPHY = "product_photography"
    LIFESTYLE_AD = "lifestyle_advertisement"
    INFOGRAPHIC = "infographic"
    TECHNICAL = "technical_diagram"
```

**Benefits:**
- Type-safe categorization
- IDE autocomplete support
- Clear state tracking
- Prevents invalid values

---

### Pattern 3: Factory Functions

```python
from pathlib import Path

def create_agent(
    config_path: Optional[Path] = None,
    **kwargs
) -> YourAgentClass:
    """
    Convenient factory with sensible defaults.

    Args:
        config_path: Path to configuration file
        **kwargs: Additional parameters

    Returns:
        Configured agent instance
    """
    config = AgentConfig(config_path)
    return YourAgentClass(config=config, **kwargs)
```

**Benefits:**
- Hide initialization complexity
- Provide sensible defaults
- Easy to extend
- Consistent API

---

### Pattern 4: Configuration-Driven Design

```python
class AgentConfig:
    """Externalized configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """Load from file or use defaults."""
        if self.config_path and self.config_path.exists():
            import yaml
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_defaults()

    def _get_defaults(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "quality_threshold": 0.7,
            "enable_logging": True,
            "max_iterations": 3,
        }
```

**Benefits:**
- Flexibility without code changes
- Environment-specific settings
- Easy to test with different configs
- No need to recompile

---

### Pattern 5: Logging and Observability

```python
import logging
import time

logger = logging.getLogger(__name__)

class ObservableAgent:
    """Agent with comprehensive logging."""

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process with full observability."""
        logger.info(f"Processing: {agent_input.core_input[:50]}...")
        start_time = time.time()

        try:
            # Do the actual work
            result = self._do_process(agent_input)

            # Calculate metrics
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Complete: {duration_ms}ms, "
                f"confidence={result.confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            raise
```

**Benefits:**
- Debugging support
- Performance tracking
- Error tracing
- Production monitoring

---

### Pattern 6: State Tracking

```python
@dataclass
class AgentState:
    """Track agent state for orchestration."""

    status: AgentStatus = AgentStatus.IDLE
    current_step: str = ""
    step_history: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/transmission."""
        return {
            "status": self.status.value,
            "current_step": self.current_step,
            "step_history": self.step_history,
            "metrics": self.metrics,
        }
```

**Benefits:**
- Orchestration support
- Progress tracking
- Recovery after failure
- Audit trail

---

## Chapter 3: Architecture Pattern 1 - Pipeline Agent {#chapter-3-pipeline-agent}

**Best for:** Single-domain transformations with clear sequential steps

### Architecture Diagram

```
                    ┌─────────────────┐
                    │   Input Data    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  1. Parse       │
                    │  (categorize)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  2. Enrich      │
                    │  (add context)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  3. Transform   │
                    │  (core logic)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  4. Verify      │
                    │  (quality check)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  5. Format      │
                    │  (output)       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Output Data   │
                    └─────────────────┘
```

### Base Class Structure

```python
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

class PipelineAgent(ABC):
    """
    Base class for pipeline-oriented agents.

    Implement the stages by overriding protected methods.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Optional stage validators
        self._stage_validators: List[Callable] = []

    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Main pipeline - executes all stages in sequence.
        """
        start_time = time.time()
        self.logger.info("Starting pipeline")

        try:
            # Stage 1: Parse
            category, intent = self._parse(input_data)
            self.logger.debug(f"Parsed: {category}, {intent}")

            # Stage 2: Enrich
            enriched = self._enrich(input_data, category, intent)
            self.logger.debug("Enriched context")

            # Stage 3: Transform
            transformed = self._transform(enriched, category, intent)
            self.logger.debug("Transformed")

            # Stage 4: Verify
            quality_check = self._verify(transformed, enriched)
            self.logger.debug(f"Quality: {quality_check.confidence:.2f}")

            # Stage 5: Format
            output = self._format(transformed, quality_check)

            # Calculate processing time
            duration_ms = int((time.time() - start_time) * 1000)
            output.processing_time_ms = duration_ms

            self.logger.info(f"Pipeline complete: {duration_ms}ms")
            return output

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    # ======================================================================
    # Abstract methods - implement these in your agent
    # ======================================================================

    @abstractmethod
    def _parse(self, input_data: AgentInput) -> tuple:
        """
        Parse and categorize input.

        Returns:
            (category, intent) tuple
        """
        pass

    @abstractmethod
    def _enrich(
        self,
        input_data: AgentInput,
        category: str,
        intent: str
    ) -> AgentInput:
        """Add domain-specific context."""
        pass

    @abstractmethod
    def _transform(
        self,
        enriched_input: AgentInput,
        category: str,
        intent: str
    ) -> Any:
        """Core transformation logic."""
        pass

    @abstractmethod
    def _verify(self, transformed: Any, context: AgentInput) -> QualityCheck:
        """Verify output quality."""
        pass

    @abstractmethod
    def _format(self, transformed: Any, quality: QualityCheck) -> AgentOutput:
        """Format final output."""
        pass
```

### Stage Implementation Pattern

```python
class MyDetectorAgent(PipelineAgent):
    """Example: Detect issues in time series data."""

    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "window_size": 7,
        "threshold": 2.5,
        "min_periods": 3,
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if config and config.config.get("thresholds"):
            self.thresholds.update(config.config["thresholds"])

    def _parse(self, input_data: AgentInput) -> tuple:
        """Parse input to extract entity and data."""
        # Assume input_data.core_input is JSON string
        import json
        data = json.loads(input_data.core_input)

        entity_id = data.get("entity_id", "unknown")
        category = "time_series_analysis"
        intent = "anomaly_detection"

        # Store for later stages
        input_data.metadata["entity_id"] = entity_id
        input_data.metadata["time_series_data"] = data.get("data", [])

        return category, intent

    def _enrich(
        self,
        input_data: AgentInput,
        category: str,
        intent: str
    ) -> AgentInput:
        """Add statistical context."""
        data = input_data.metadata["time_series_data"]

        # Calculate statistics
        import statistics
        input_data.metadata["mean"] = statistics.mean(data)
        input_data.metadata["stdev"] = statistics.stdev(data) if len(data) > 1 else 0

        return input_data

    def _transform(
        self,
        enriched_input: AgentInput,
        category: str,
        intent: str
    ) -> List[Dict]:
        """Detect anomalies using z-score."""
        data = enriched_input.metadata["time_series_data"]
        mean = enriched_input.metadata["mean"]
        stdev = enriched_input.metadata["stdev"]

        anomalies = []
        threshold = self.thresholds["threshold"]

        for i, value in enumerate(data):
            if stdev > 0:
                z_score = abs((value - mean) / stdev)
                if z_score > threshold:
                    anomalies.append({
                        "index": i,
                        "value": value,
                        "z_score": z_score,
                    })

        return anomalies

    def _verify(self, transformed: List[Dict], context: AgentInput) -> QualityCheck:
        """Verify detection quality."""
        issues = []

        # Check 1: Did we find too many anomalies?
        if len(transformed) > len(context.metadata["time_series_data"]) * 0.5:
            issues.append("Too many anomalies detected - threshold may be too low")

        # Check 2: Minimum data quality
        if len(context.metadata["time_series_data"]) < self.thresholds["min_periods"]:
            issues.append("Insufficient data for reliable detection")

        # Calculate confidence
        if not issues:
            confidence = 0.9
        elif len(issues) == 1:
            confidence = 0.6
        else:
            confidence = 0.3

        return QualityCheck(
            passes=len(issues) == 0,
            confidence=confidence,
            issues=issues
        )

    def _format(self, transformed: List[Dict], quality: QualityCheck) -> AgentOutput:
        """Format output."""
        import json

        output = AgentOutput(
            result=json.dumps({
                "anomalies": transformed,
                "quality": {
                    "passes": quality.passes,
                    "confidence": quality.confidence,
                    "issues": quality.issues,
                }
            }),
            confidence=quality.confidence,
            techniques_used=["z_score_detection"]
        )

        return output
```

### Quality Verification Integration

```python
@dataclass
class QualityCheck:
    """Result of quality verification."""

    passes: bool
    confidence: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)

    def add_issue(self, issue: str):
        """Add an issue."""
        self.issues.append(issue)
        self.passes = False
```

**See template:** `templates/pipeline_agent.py`

---

## Chapter 4: Architecture Pattern 2 - Adapter-Based Framework {#chapter-4-adapter-framework}

**Best for:** Multi-domain systems requiring reusability

### Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                  BaseAgent (Generic)                │
│  ┌──────────────────────────────────────────────┐  │
│  │  Generic Orchestration Pipeline              │  │
│  │  1. Parse input      (via adapter)           │  │
│  │  2. Enrich context   (via adapter)           │  │
│  │  3. Generate output (via adapter)           │  │
│  │  4. Reflexion loop   (generic + adapter)     │  │
│  │  5. Quality verify   (generic)               │  │
│  │  6. Store memory     (generic)               │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  Generic Components:                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Memory    │  │  Examples   │  │ Reflexion   │  │
│  │   System    │  │   Manager   │  │   Engine    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                         │
                         │ implements
                         ▼
         ┌───────────────────────────────┐
         │     BaseAdapter (ABC)         │
         │  ┌─────────────────────────┐  │
         │  │ Abstract Methods:       │  │
         │  │ - parse_input()         │  │
         │  │ - enrich_context()      │  │
         │  │ - generate_output()     │  │
         │  │ - refine()              │  │
         │  └─────────────────────────┘  │
         └───────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ DomainA │    │ DomainB │    │ DomainC │
   │ Adapter │    │ Adapter │    │ Adapter │
   └─────────┘    └─────────┘    └─────────┘
```

### BaseAgent Generic Orchestration

```python
class BaseAgent:
    """
    Generic agent that works with any domain via adapters.

    The agent provides:
    - Generic orchestration pipeline
    - Learning components (memory, examples, reflexion)
    - Quality verification
    - Observability

    Domain-specific logic is provided by the adapter.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        config: Optional[FrameworkConfig] = None
    ):
        """
        Initialize with domain-specific adapter.

        Args:
            adapter: Domain adapter implementing BaseAdapter
            config: Framework configuration
        """
        self.adapter = adapter
        self.config = config or FrameworkConfig()

        # Initialize generic components
        self.example_manager = ExampleManager(
            examples_db_path=self.config.examples_db_path
        )
        self.reflexion_engine = ReflexionEngine(
            max_iterations=self.config.max_reflexion_iterations,
            quality_threshold=self.config.quality_threshold,
        )
        self.memory = MemorySystem(
            memory_db_path=self.config.memory_db_path,
            max_entries=self.config.memory_max_entries,
        )
        self.quality_verifier = QualityVerifier(
            threshold=self.config.quality_threshold
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"BaseAgent initialized with {adapter.domain} adapter")

    def process(self, agent_input: AgentInput) -> AgentOutput:
        """
        Main processing pipeline (generic, works for all domains).

        Pipeline:
        1. Parse input (via adapter)
        2. Retrieve examples (generic)
        3. Enrich context (via adapter)
        4. Generate output (via adapter)
        5. Reflexion loop (generic + adapter)
        6. Quality verify (generic)
        7. Store in memory (generic)
        """
        start_time = time.time()

        self.logger.info("=" * 70)
        self.logger.info(f"PROCESSING PIPELINE ({self.adapter.domain.upper()})")
        self.logger.info("=" * 70)

        # Step 1: Parse input (via adapter)
        self.logger.info("Step 1: Parsing input...")
        category, intent = self.adapter.parse_input(agent_input.core_input)
        self.logger.info(f"  → Category: {category}, Intent: {intent}")

        # Step 2: Retrieve examples (generic)
        if self.config.enable_examples:
            self.logger.info("Step 2: Retrieving examples...")
            examples = self.example_manager.retrieve_relevant(
                agent_input.core_input,
                self.adapter.domain,
                k=3,
            )
            self.logger.info(f"  → Retrieved {len(examples)} examples")
        else:
            examples = []

        # Step 3: Enrich context (via adapter)
        self.logger.info("Step 3: Enriching context...")
        enriched_input = self.adapter.enrich_context(agent_input)
        self.logger.info("  → Context enriched")

        # Step 4: Generate output (via adapter)
        self.logger.info("Step 4: Generating output...")
        base_output = self.adapter.generate_output(
            enriched_input, category, intent, examples
        )
        self.logger.info(f"  → Base output generated ({len(base_output)} chars)")

        # Step 5: Reflexion loop (generic + adapter)
        if self.config.enable_reflexion:
            self.logger.info("Step 5: Reflexion loop...")
            refined_output, critique_history = self.reflexion_engine.refine(
                base_output, enriched_input, self.adapter
            )
            self.logger.info(f"  → Refined through {len(critique_history)} iterations")
            final_output = refined_output
        else:
            final_output = base_output
            critique_history = []

        # Step 6: Quality verification
        self.logger.info("Step 6: Verifying quality...")
        quality_check = self.quality_verifier.verify(
            final_output, enriched_input, self.adapter, examples
        )
        self.logger.info(
            f"  → Quality: {quality_check.confidence:.2f} "
            f"(passes: {quality_check.passes})"
        )

        # Step 7: Store in memory (generic)
        if self.config.enable_memory and quality_check.confidence >= 0.7:
            self.logger.info("Step 7: Storing in memory...")
            memory_entry = MemoryEntry(
                input_prompt=agent_input.core_input,
                output_prompt=final_output,
                domain=self.adapter.domain,
                detected_category=category,
                detected_intent=intent,
                confidence=quality_check.confidence,
                techniques_used=self.adapter._extract_techniques(final_output)
                if hasattr(self.adapter, "_extract_techniques")
                else [],
            )
            self.memory.add_entry(memory_entry)
            self.logger.info(f"  → Stored in memory (entry_id: {memory_entry.entry_id})")

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Create output
        output = AgentOutput(
            result=final_output,
            confidence=quality_check.confidence,
            processing_time_ms=processing_time_ms,
            techniques_used=self.adapter._extract_techniques(final_output)
            if hasattr(self.adapter, "_extract_techniques")
            else [],
        )

        self.logger.info("=" * 70)
        self.logger.info(f"PIPELINE COMPLETE ({processing_time_ms:.0f}ms)")
        self.logger.info("=" * 70)

        return output
```

### BaseAdapter Interface

```python
class BaseAdapter(ABC):
    """
    Abstract base class for domain-specific adapters.

    Each domain provides an adapter that implements domain-specific logic
    while using the generic framework for orchestration.
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier (e.g., 'nano', 'dalle', 'stable_diffusion')."""
        pass

    @abstractmethod
    def parse_input(self, generic_input: str) -> Tuple[str, str]:
        """
        Parse input into category and intent.

        Args:
            generic_input: Raw input string

        Returns:
            (category, intent) tuple
        """
        pass

    @abstractmethod
    def enrich_context(self, agent_input: AgentInput) -> AgentInput:
        """
        Enrich agent input with domain-specific context.

        Args:
            agent_input: Input to enrich

        Returns:
            Enriched input
        """
        pass

    @abstractmethod
    def generate_output(
        self,
        enriched_input: AgentInput,
        category: str,
        intent: str,
        examples: List[Any]
    ) -> str:
        """
        Generate domain-specific output.

        Args:
            enriched_input: Enriched input
            category: Parsed category
            intent: Parsed intent
            examples: Retrieved similar examples

        Returns:
            Generated output string
        """
        pass

    def refine_output(
        self,
        output: str,
        critique: str,
        agent_input: AgentInput
    ) -> str:
        """
        Refine output based on critique (for reflexion).

        Default implementation returns output unchanged.
        Override to implement refinement logic.
        """
        return output

    def _extract_techniques(self, output: str) -> List[str]:
        """
        Extract techniques applied (for memory tracking).

        Default implementation returns empty list.
        Override to track domain-specific techniques.
        """
        return []
```

### Domain Adapter Implementation Example

```python
class MyDomainAdapter(BaseAdapter):
    """Example adapter for a specific domain."""

    @property
    def domain(self) -> str:
        return "my_domain"

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Domain-specific components
        self.parser = MyDomainParser()
        self.context_enricher = MyDomainContextEnricher()
        self.generator = MyDomainGenerator()

    def parse_input(self, generic_input: str) -> Tuple[str, str]:
        """Parse input using domain-specific logic."""
        return self.parser.parse(generic_input)

    def enrich_context(self, agent_input: AgentInput) -> AgentInput:
        """Add domain-specific context."""
        return self.context_enricher.enrich(agent_input)

    def generate_output(
        self,
        enriched_input: AgentInput,
        category: str,
        intent: str,
        examples: List[Any]
    ) -> str:
        """Generate output using domain-specific logic."""
        return self.generator.generate(
            enriched_input, category, intent, examples
        )

    def refine_output(
        self,
        output: str,
        critique: str,
        agent_input: AgentInput
    ) -> str:
        """Refine based on critique."""
        # Domain-specific refinement logic
        refined = self.generator.refine(output, critique)
        return refined

    def _extract_techniques(self, output: str) -> List[str]:
        """Extract techniques from output."""
        return self.generator.extract_techniques(output)
```

### Generic Components

**Memory System:**
```python
class MemorySystem:
    """Organizational memory for learning from past interactions."""

    def __init__(self, memory_db_path: str, max_entries: int = 1000):
        self.memory_db_path = Path(memory_db_path)
        self.max_entries = max_entries
        self.memories: List[MemoryEntry] = self._load_memories()

    def add_entry(self, entry: MemoryEntry):
        """Store a memory entry."""
        self.memories.append(entry)

        # LRU eviction
        if len(self.memories) > self.max_entries:
            self.memories = self.memories[-self.max_entries:]

        self._save_memories()

    def find_similar(
        self,
        query: str,
        domain: str,
        k: int = 3
    ) -> List[MemoryEntry]:
        """Find similar past interactions."""
        # Filter by domain
        domain_memories = [
            m for m in self.memories if m.domain == domain
        ]

        # Simple similarity (could use embeddings)
        scores = []
        for memory in domain_memories:
            score = self._similarity(query, memory.input_prompt)
            scores.append((memory, score))

        # Return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scores[:k]]
```

**Reflexion Engine:**
```python
class ReflexionEngine:
    """
    Self-refinement through critique-refine iterations.

    Research shows this provides +20-30% quality improvement.
    """

    def __init__(
        self,
        max_iterations: int = 2,
        quality_threshold: float = 0.7
    ):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    def refine(
        self,
        output: str,
        agent_input: AgentInput,
        adapter: BaseAdapter
    ) -> Tuple[str, List[Dict]]:
        """
        Refine output through critique-refine iterations.
        """
        current_output = output
        critique_history = []

        for iteration in range(self.max_iterations):
            # Generate critique
            critique = self._generate_critique(
                current_output, agent_input
            )
            critique_history.append({
                "iteration": iteration,
                "critique": critique,
            })

            # Refine based on critique
            current_output = adapter.refine_output(
                current_output, critique, agent_input
            )

            # Check if good enough
            quality = self._assess_quality(current_output)
            if quality >= self.quality_threshold:
                break

        return current_output, critique_history
```

**See templates:**
- `templates/base_agent.py` - Generic framework
- `templates/base_adapter.py` - Adapter interface
- `templates/memory_system.py` - Memory implementation

---

## Chapter 5: Architecture Pattern 3 - Multi-Agent Orchestration {#chapter-5-multi-agent-orchestration}

**Best for:** Complex workflows requiring specialized roles

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                     Orchestrator                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │          Workflow State Management                 │  │
│  │  - Track current step                              │  │
│  │  - Handle failures                                 │  │
│  │  - Coordinate handoffs                             │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│    PM Agent  │  │ Coder Agent  │  │ Reviewer     │
│              │  │              │  │   Agent      │
│ - Analyze    │  │ - Implement  │  │ - Review     │
│ - Plan       │  │ - Create PR  │  │ - Security   │
│ - Spec       │  │              │  │ - Architecture│
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Memory Agent │  │  Judge Agent │  │  (Loop)      │
│              │  │              │  │              │
│ - History    │  │ - Evaluate   │  │              │
│ - Patterns   │  │ - Backtest   │  │              │
│ - Context    │  │ - Metrics    │  │              │
└──────────────┘  └──────┬───────┘  └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   Decision   │
                  │              │
                  │ - MERGE/     │
                  │   REJECT     │
                  └──────────────┘
```

### Orchestrator Base Class

```python
from enum import Enum
from typing import Callable, Dict, Any, Optional, List

class WorkflowStatus(Enum):
    """Status of the workflow."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class WorkflowState:
    """Current state of the workflow."""
    status: WorkflowStatus = WorkflowStatus.IDLE
    current_step: str = ""
    step_history: List[str] = field(default_factory=list)
    experiments_completed: int = 0
    experiments_succeeded: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/transmission."""
        return {
            "status": self.status.value,
            "current_step": self.current_step,
            "step_history": self.step_history,
            "experiments_completed": self.experiments_completed,
            "experiments_succeeded": self.experiments_succeeded,
        }

class Orchestrator:
    """
    Coordinates multiple specialized agents.

    Workflow:
    1. Receive findings/issues
    2. PM Agent creates experiment spec (with Memory context)
    3. Coder Agent implements spec
    4. Reviewer Agent reviews implementation
    5. Judge Agent evaluates results
    6. Memory Agent records learnings
    7. Loop back to step 1
    """

    def __init__(
        self,
        repo_path: Path,
        agents: Dict[str, Any],
        mode: str = "supervised"
    ):
        """
        Initialize orchestrator with agents.

        Args:
            repo_path: Path to repository
            agents: Dictionary of agent instances
                    {"pm": pm_agent, "coder": coder_agent, ...}
            mode: "supervised" (human-in-loop) or "auto" (fully automatic)
        """
        self.repo_path = Path(repo_path)
        self.agents = agents
        self.mode = mode

        self.state = WorkflowState()
        self.callbacks: Dict[str, Callable] = {}

        self.logger = logging.getLogger(self.__class__.__name__)

    def register_callback(self, event: str, callback: Callable):
        """
        Register callback for supervised mode events.

        Events:
        - "spec_created": PM Agent creates spec
        - "pr_created": Coder Agent creates PR
        - "review_complete": Reviewer Agent completes review
        - "decision_made": Judge Agent makes decision
        - "experiment_complete": Experiment is complete

        Callback signature: callback(data: Dict) -> bool
        Return True to continue, False to pause.
        """
        self.callbacks[event] = callback
        self.logger.info(f"Registered callback for event: {event}")

    async def run_experiment(
        self,
        findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a complete experiment from findings.

        Args:
            findings: Issues/requirements to address

        Returns:
            Experiment result
        """
        experiment_id = f"exp-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self.logger.info("=" * 70)
        self.logger.info(f"STARTING EXPERIMENT: {experiment_id}")
        self.logger.info("=" * 70)

        self.state.status = WorkflowStatus.RUNNING
        result = {
            "experiment_id": experiment_id,
            "success": False,
        }

        try:
            # Step 1: PM Agent creates spec
            self.logger.info("Step 1: PM Agent creating spec...")
            self.state.current_step = "create_spec"
            self.state.step_history.append("pm_create_spec")

            spec = self.agents["pm"].create_spec(
                findings=findings,
                memory_context=self.agents["memory"].query(findings)
            )
            result["spec_id"] = spec["spec_id"]

            # Callback for supervised mode
            if not await self._check_callback("spec_created", {
                "spec": spec, "findings": findings
            }):
                self.logger.info("Paused by user at spec creation")
                self.state.status = WorkflowStatus.PAUSED
                return result

            # Step 2: Coder Agent implements
            self.logger.info("Step 2: Coder Agent implementing...")
            self.state.current_step = "implement"
            self.state.step_history.append("coder_implement")

            implementation = self.agents["coder"].implement(spec)

            if not implementation["success"]:
                self.logger.error(f"Coder failed: {implementation['errors']}")
                result["success"] = False
                self.state.status = WorkflowStatus.FAILED
                return result

            result["pr_id"] = implementation["pr_id"]

            if not await self._check_callback("pr_created", {
                "implementation": implementation, "spec": spec
            }):
                self.logger.info("Paused by user at PR creation")
                self.state.status = WorkflowStatus.PAUSED
                return result

            # Step 3: Reviewer Agent reviews
            self.logger.info("Step 3: Reviewer Agent reviewing...")
            self.state.current_step = "review"
            self.state.step_history.append("reviewer_review")

            review = self.agents["reviewer"].review(
                implementation, spec
            )

            if review["status"] in ["needs_changes", "rejected"]:
                self.logger.warning(f"Review failed: {review['status']}")
                result["success"] = False
                self.state.status = WorkflowStatus.FAILED

                # Still record in memory
                self.agents["memory"].record(
                    spec, implementation, review, None
                )
                return result

            # Step 4: Judge Agent evaluates
            self.logger.info("Step 4: Judge Agent evaluating...")
            self.state.current_step = "evaluate"
            self.state.step_history.append("judge_evaluate")

            decision = self.agents["judge"].evaluate(
                implementation, spec
            )

            result["approved"] = decision["approve"]
            result["lift_score"] = decision.get("lift_score", 0.0)

            # Step 5: Record in Memory
            self.logger.info("Step 5: Recording in Memory...")
            self.agents["memory"].record(
                spec, implementation, review, decision
            )

            # Update state
            result["success"] = decision["approve"]

            if decision["approve"]:
                self.state.experiments_completed += 1
                self.state.experiments_succeeded += 1
                self.state.status = WorkflowStatus.COMPLETED
            else:
                self.state.experiments_completed += 1
                self.state.status = WorkflowStatus.FAILED

            self.logger.info("=" * 70)
            self.logger.info(f"EXPERIMENT COMPLETE: {experiment_id}")
            self.logger.info(f"  Success: {result['success']}")
            self.logger.info("=" * 70)

            return result

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            result["success"] = False
            self.state.status = WorkflowStatus.FAILED
            return result

    async def _check_callback(
        self,
        event: str,
        data: Dict[str, Any]
    ) -> bool:
        """Check if callback allows continuation."""
        if self.mode == "supervised" and event in self.callbacks:
            try:
                callback = self.callbacks[event]
                should_continue = await self._async_wrap(callback, data)
                if not should_continue:
                    self.logger.warning(f"Callback '{event}' requested pause")
                return should_continue
            except Exception as e:
                self.logger.error(f"Callback '{event}' failed: {e}")
                return True
        return True

    async def _async_wrap(
        self,
        callback: Callable,
        data: Dict[str, Any]
    ) -> bool:
        """Wrap sync/async callback."""
        import asyncio

        if asyncio.iscoroutinefunction(callback):
            return await callback(data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, callback, data)
```

### Role-Based Agent Interfaces

**PM Agent:**
```python
class PMAgent:
    """Product Manager - creates experiment specs."""

    def create_spec(
        self,
        findings: Dict[str, Any],
        memory_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create experiment spec from findings.

        Args:
            findings: Issues to address
            memory_context: Historical context from Memory

        Returns:
            Experiment specification
        """
        spec = {
            "spec_id": self._generate_id(),
            "title": findings["description"],
            "component": findings["component"],
            "scope": self._determine_scope(findings, memory_context),
            "constraints": self._get_constraints(findings),
            "success_criteria": findings.get("success_criteria", []),
        }
        return spec
```

**Coder Agent:**
```python
class CoderAgent:
    """Coder - implements specs."""

    def implement(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement experiment spec.

        Args:
            spec: Experiment specification

        Returns:
            Implementation result with PR
        """
        changes = self._generate_changes(spec)
        pr = self._create_pr(spec, changes)

        return {
            "success": True,
            "pr_id": pr["id"],
            "changes": changes,
        }
```

**Reviewer Agent:**
```python
class ReviewerAgent:
    """Reviewer - reviews implementations."""

    def review(
        self,
        implementation: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Review implementation.

        Args:
            implementation: Implementation from Coder
            spec: Original spec

        Returns:
            Review result
        """
        # Architecture check
        arch_check = self._check_architecture(implementation)

        # Security check
        security_check = self._check_security(implementation)

        # CI validation
        ci_check = self._validate_ci(implementation)

        overall_pass = (
            arch_check["passes"] and
            security_check["passes"] and
            ci_check["passes"]
        )

        return {
            "status": "approved" if overall_pass else "needs_changes",
            "passes": overall_pass,
            "issues": (
                arch_check["issues"] +
                security_check["issues"] +
                ci_check["issues"]
            ),
        }
```

**Judge Agent:**
```python
class JudgeAgent:
    """Judge - evaluates and makes decisions."""

    def evaluate(
        self,
        implementation: Dict[str, Any],
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate implementation and make merge decision.

        Args:
            implementation: Implementation from Coder
            spec: Original spec

        Returns:
            Merge decision
        """
        # Run backtests
        backtest_result = self._run_backtests(implementation)

        # Calculate lift
        lift_score = self._calculate_lift(
            backtest_result["metrics_before"],
            backtest_result["metrics_after"]
        )

        # Check for regressions
        regressions = self._detect_regressions(backtest_result)

        # Make decision
        approve = (
            backtest_result["all_tests_pass"] and
            lift_score > 0.01 and
            len(regressions) == 0
        )

        return {
            "approve": approve,
            "lift_score": lift_score,
            "regressions": regressions,
            "backtest_result": backtest_result,
        }
```

**Memory Agent:**
```python
class MemoryAgent:
    """Memory - records and retrieves experiments."""

    def record(
        self,
        spec: Dict,
        implementation: Dict,
        review: Dict,
        decision: Optional[Dict]
    ):
        """Record experiment in memory."""
        record = {
            "spec_id": spec["spec_id"],
            "pr_id": implementation["pr_id"],
            "outcome": "success" if decision and decision["approve"] else "failure",
            "lift_score": decision.get("lift_score", 0.0) if decision else 0.0,
            "lessons_learned": self._extract_lessons(spec, review, decision),
        }

        self._save_record(record)

    def query(self, findings: Dict) -> Dict:
        """Query memory for relevant history."""
        similar = self._find_similar_experiments(
            component=findings["component"],
            issue_type=findings.get("issue_type"),
        )

        failures = self._find_similar_failures(
            component=findings["component"],
            scope=findings.get("scope"),
        )

        return {
            "similar_experiments": similar,
            "similar_failures": failures,
            "warnings": self._generate_warnings(similar, failures),
        }
```

**See template:** `templates/orchestrator.py`

---

## Chapter 6: Specialization Patterns {#chapter-6-specialization}

How to make agents algorithm-specific.

### Pattern 1: Algorithm Detection

```python
class AlgorithmDetector:
    """Detects which algorithm to apply."""

    ALGORITHM_PATTERNS = {
        "time_series": ["trend", "seasonal", "forecast", "anomaly"],
        "classification": ["classify", "categorize", "label"],
        "regression": ["predict", "estimate", "forecast_value"],
        "clustering": ["group", "cluster", "segment"],
    }

    def detect(self, input_text: str) -> str:
        """Detect algorithm type from input."""
        input_lower = input_text.lower()

        scores = {}
        for algo, patterns in self.ALGORITHM_PATTERNS.items():
            score = sum(
                1 for pattern in patterns
                if pattern in input_lower
            )
            if score > 0:
                scores[algo] = score

        if not scores:
            return "default"

        return max(scores, key=scores.get)
```

### Pattern 2: Domain Types

```python
class DomainTypes:
    """Algorithm-specific type definitions."""

    # For time series algorithms
    @dataclass
    class TimeSeriesConfig:
        window_size: int = 7
        threshold: float = 2.5
        min_periods: int = 3

    # For classification algorithms
    @dataclass
    class ClassificationConfig:
        algorithm: str = "random_forest"
        min_samples: int = 10
        confidence_threshold: float = 0.7

    # For clustering algorithms
    @dataclass
    class ClusteringConfig:
        n_clusters: int = 3
        algorithm: str = "kmeans"
        distance_metric: str = "euclidean"
```

### Pattern 3: Threshold Management

```python
class ThresholdManager:
    """Manages tunable parameters for algorithms."""

    DEFAULT_THRESHOLDS = {
        "sensitivity": 0.8,
        "specificity": 0.9,
        "precision": 0.85,
    }

    def __init__(self, custom_thresholds: Optional[Dict] = None):
        self.thresholds = self.DEFAULT_THRESHOLDS.copy()
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    def get(self, key: str) -> float:
        """Get threshold value."""
        return self.thresholds.get(key, 0.5)

    def set(self, key: str, value: float):
        """Set threshold value."""
        self.thresholds[key] = value

    def validate(self) -> List[str]:
        """Validate threshold values."""
        issues = []

        for key, value in self.thresholds.items():
            if not 0 <= value <= 1:
                issues.append(f"{key} must be between 0 and 1, got {value}")

        return issues
```

### Pattern 4: Technique System

```python
class Technique:
    """A transformation technique."""

    def __init__(self, name: str, apply_fn: Callable):
        self.name = name
        self.apply_fn = apply_fn

class TechniqueRegistry:
    """Registry of composable techniques."""

    def __init__(self):
        self.techniques: Dict[str, Technique] = {}

    def register(self, technique: Technique):
        """Register a technique."""
        self.techniques[technique.name] = technique

    def apply(
        self,
        data: Any,
        technique_names: List[str]
    ) -> Any:
        """Apply techniques in sequence."""
        result = data

        for name in technique_names:
            if name in self.techniques:
                result = self.techniques[name].apply_fn(result)

        return result

# Example usage
registry = TechniqueRegistry()

registry.register(Technique(
    name="normalize",
    apply_fn=lambda x: (x - x.mean()) / x.std()
))

registry.register(Technique(
    name="smooth",
    apply_fn=lambda x: x.rolling(window=3).mean()
))

# Apply techniques
result = registry.apply(data, ["normalize", "smooth"])
```

### Pattern 5: Quality Metrics

```python
class AlgorithmMetrics:
    """Algorithm-specific quality metrics."""

    @staticmethod
    def precision_score(y_true, y_pred) -> float:
        """Calculate precision."""
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    @staticmethod
    def recall_score(y_true, y_pred) -> float:
        """Calculate recall."""
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred) -> float:
        """Calculate F1 score."""
        precision = AlgorithmMetrics.precision_score(y_true, y_pred)
        recall = AlgorithmMetrics.recall_score(y_true, y_pred)
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

    @staticmethod
    def calculate_lift(
        baseline_metric: float,
        new_metric: float
    ) -> float:
        """Calculate lift percentage."""
        if baseline_metric == 0:
            return 0.0
        return (new_metric - baseline_metric) / baseline_metric
```

---

## Chapter 7: Quality Assurance Patterns {#chapter-7-quality-assurance}

### Multi-Dimensional Verification

```python
class QualityVerifier:
    """Multi-dimensional quality verification."""

    def verify(
        self,
        output: str,
        context: Dict[str, Any]
    ) -> QualityCheck:
        """Verify output across multiple dimensions."""
        issues = []
        scores = {}

        # Dimension 1: Completeness
        completeness = self._check_completeness(output, context)
        scores["completeness"] = completeness
        if completeness < 0.7:
            issues.append("Output lacks required elements")

        # Dimension 2: Specificity
        specificity = self._check_specificity(output)
        scores["specificity"] = specificity
        if specificity < 0.6:
            issues.append("Output is too vague")

        # Dimension 3: Consistency
        consistency = self._check_consistency(output, context)
        scores["consistency"] = consistency
        if consistency < 0.7:
            issues.append("Output inconsistent with context")

        # Overall confidence
        confidence = sum(scores.values()) / len(scores)
        passes = len(issues) == 0

        return QualityCheck(
            passes=passes,
            confidence=confidence,
            issues=issues,
            **scores
        )
```

### Guardrails and Constraints

```python
class Guardrails:
    """Prevent invalid outputs."""

    FORBIDDEN_PATTERNS = [
        r"TODO:",  # Incomplete implementation
        r"FIXME:",  # Known issues
        r"XXX:",  # Placeholder
    ]

    REQUIRED_PATTERNS = [
        r"def ",  # Function definitions
        r"return",  # Return statements
    ]

    def validate(self, code: str) -> List[str]:
        """Check code against guardrails."""
        issues = []

        # Check forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                issues.append(f"Forbidden pattern detected: {pattern}")

        # Check required patterns
        for pattern in self.REQUIRED_PATTERNS:
            if not re.search(pattern, code):
                issues.append(f"Required pattern missing: {pattern}")

        return issues
```

### Overfitting Prevention

```python
class OverfittingDetector:
    """Detect overfitting to test data."""

    # Patterns that indicate test overfitting
    OVERFITTING_PATTERNS = [
        r"if\s+id\s*==\s*['\"]test_",  # Hardcoded test IDs
        r"if\s+name\s*==\s*['\"]example",  # Hardcoded test names
        r"#\s*HACK.*for.*test",  # Test-specific hacks
        r"#\s*TODO.*remove.*hardcode",  # Temporary hardcoding
    ]

    def check(self, code: str, file_path: str) -> List[str]:
        """Check for overfitting patterns."""
        issues = []

        for pattern in self.OVERFITTING_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append(
                    f"Line {line_num}: Potential overfitting - {match.group()}"
                )

        return issues
```

### Regression Detection

```python
class RegressionDetector:
    """Detect performance regression."""

    def detect(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        threshold: float = 0.05
    ) -> List[str]:
        """
        Detect regression in metrics.

        Args:
            metrics_before: Baseline metrics
            metrics_after: New metrics
            threshold: Max allowed regression (5% by default)

        Returns:
            List of regression descriptions
        """
        regressions = []

        for metric_name, baseline_value in metrics_before.items():
            if metric_name not in metrics_after:
                continue

            new_value = metrics_after[metric_name]

            # For metrics where higher is better
            if metric_name in ["accuracy", "precision", "recall", "f1"]:
                if new_value < baseline_value * (1 - threshold):
                    regressions.append(
                        f"{metric_name}: {baseline_value:.3f} → {new_value:.3f} "
                        f"({((new_value - baseline_value) / baseline_value):.1%})"
                    )

            # For metrics where lower is better
            elif metric_name in ["error_rate", "latency", "cost"]:
                if new_value > baseline_value * (1 + threshold):
                    regressions.append(
                        f"{metric_name}: {baseline_value:.3f} → {new_value:.3f} "
                        f"({((new_value - baseline_value) / baseline_value):.1%})"
                    )

        return regressions
```

**See template:** `templates/quality_verifier.py`

---

## Chapter 8: Learning and Memory {#chapter-8-learning-memory}

### Memory System Implementation

```python
@dataclass
class MemoryEntry:
    """A memory entry."""
    input_prompt: str
    output_prompt: str
    domain: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    entry_id: str = ""

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.entry_id:
            content = f"{self.input_prompt}:{self.output_prompt}:{self.timestamp.isoformat()}"
            self.entry_id = hashlib.md5(content.encode()).hexdigest()[:12]

class MemorySystem:
    """Organizational memory."""

    def __init__(self, memory_db_path: str, max_entries: int = 1000):
        self.memory_db_path = Path(memory_db_path)
        self.max_entries = max_entries
        self.memories: List[MemoryEntry] = self._load_memories()

    def add_entry(self, entry: MemoryEntry):
        """Store entry."""
        self.memories.append(entry)

        # LRU eviction
        if len(self.memories) > self.max_entries:
            self.memories = self.memories[-self.max_entries:]

        self._save_memories()

    def find_similar(
        self,
        query: str,
        domain: str,
        k: int = 3
    ) -> List[MemoryEntry]:
        """Find similar entries."""
        domain_memories = [
            m for m in self.memories if m.domain == domain
        ]

        if not domain_memories:
            return []

        # Calculate similarity scores
        scores = []
        for memory in domain_memories:
            score = self._similarity(query, memory.input_prompt)
            scores.append((memory, score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scores[:k]]

    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity (simple word overlap)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0
```

### Reflexion (Self-Critique) Pattern

```python
class ReflexionEngine:
    """
    Self-refinement through critique-refine iterations.

    Research: "Reflexion: Language Agents with Verbal Reinforcement Learning"
    Shows +20-30% quality improvement.
    """

    def __init__(
        self,
        max_iterations: int = 2,
        quality_threshold: float = 0.7
    ):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.logger = logging.getLogger(__name__)

    def refine(
        self,
        output: str,
        agent_input: AgentInput,
        adapter: BaseAdapter
    ) -> Tuple[str, List[Dict]]:
        """
        Refine through critique-refine iterations.
        """
        current_output = output
        critique_history = []

        for iteration in range(self.max_iterations):
            self.logger.info(f"Reflexion iteration {iteration + 1}/{self.max_iterations}")

            # Step 1: Generate critique
            critique = self._generate_critique(current_output, agent_input)
            critique_history.append({
                "iteration": iteration,
                "critique": critique,
            })

            # Step 2: Refine based on critique
            current_output = adapter.refine_output(
                current_output, critique, agent_input
            )

            # Step 3: Check if good enough
            quality = self._assess_quality(current_output)
            self.logger.info(f"  Quality: {quality:.2f}")

            if quality >= self.quality_threshold:
                self.logger.info("  Quality threshold met, stopping refinement")
                break

        return current_output, critique_history

    def _generate_critique(self, output: str, agent_input: AgentInput) -> str:
        """Generate critique of output."""
        critiques = []

        # Check 1: Length
        if len(output) < 50:
            critiques.append("Output is too short")

        # Check 2: Contains input
        if agent_input.core_input.lower() not in output.lower():
            critiques.append("Output doesn't reference the input")

        # Check 3: Structure
        if not any(c in output for c in ['.!?']):
            critiques.append("Output lacks proper sentence structure")

        return "; ".join(critiques) if critiques else "No issues found"

    def _assess_quality(self, output: str) -> float:
        """Assess output quality."""
        score = 0.5  # Base score

        # Length bonus
        if 50 <= len(output) <= 500:
            score += 0.2

        # Structure bonus
        if any(c in output for c in ['.!?']):
            score += 0.2

        # Vocabulary bonus
        word_count = len(set(output.split()))
        if word_count >= 10:
            score += 0.1

        return min(1.0, score)
```

### Historical Context Retrieval

```python
class ContextRetriever:
    """Retrieve relevant historical context."""

    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system

    def get_context(
        self,
        query: str,
        domain: str,
        context_type: str = "similar"
    ) -> Dict[str, Any]:
        """
        Get historical context for a query.

        Args:
            query: Current query/input
            domain: Domain to search in
            context_type: Type of context ("similar", "failures", "successes")
        """
        if context_type == "similar":
            memories = self.memory.find_similar(query, domain, k=5)

            return {
                "similar_cases": [
                    {
                        "input": m.input_prompt,
                        "output": m.output_prompt,
                        "confidence": m.confidence,
                    }
                    for m in memories
                ],
                "count": len(memories),
            }

        elif context_type == "failures":
            all_memories = self.memory.memories
            failures = [
                m for m in all_memories
                if m.domain == domain and m.confidence < 0.5
            ][:5]

            return {
                "failure_cases": [
                    {
                        "input": m.input_prompt,
                        "output": m.output_prompt,
                        "confidence": m.confidence,
                    }
                    for m in failures
                ],
                "count": len(failures),
            }

        elif context_type == "successes":
            all_memories = self.memory.memories
            successes = [
                m for m in all_memories
                if m.domain == domain and m.confidence >= 0.8
            ][:5]

            return {
                "success_cases": [
                    {
                        "input": m.input_prompt,
                        "output": m.output_prompt,
                        "confidence": m.confidence,
                    }
                    for m in successes
                ],
                "count": len(successes),
            }
```

---

## Chapter 9: Real-World Examples {#chapter-9-real-world-examples}

### Example 1: FatigueDetector (from diagnoser)

**Algorithm:** Rolling window analysis for creative fatigue detection

**File:** `src/meta/diagnoser/detectors/fatigue_detector.py`

```python
class FatigueDetector(BaseDetector):
    """
    Detects creative fatigue using rolling window analysis.

    Algorithm:
    1. For each day t, only use data from day [t-window_size : t-1]
    2. Calculate cumulative frequency within the window
    3. Identify golden period in the window
    4. Check if current day shows fatigue signals
    5. Report fatigue only if consecutive days detected
    """

    DEFAULT_THRESHOLDS = {
        "window_size_days": 23,
        "golden_min_freq": 1.0,
        "golden_max_freq": 2.5,
        "fatigue_freq_threshold": 3.0,
        "cpa_increase_threshold": 1.15,
        "consecutive_days": 1,
        "min_golden_days": 1,
    }

    def detect(self, data: pd.DataFrame, entity_id: str) -> List[Issue]:
        """Detect fatigue issues."""
        issues = []

        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)

        # Apply rolling window analysis
        fatigue_days = self._analyze_fatigue_rolling(data, entity_id)

        # Create issues for consecutive fatigue days
        for day_info in fatigue_days:
            issues.append(Issue(
                entity_id=entity_id,
                issue_type="creative_fatigue",
                severity=day_info["severity"],
                date=day_info["date"],
                details=day_info,
            ))

        return issues

    def _analyze_fatigue_rolling(self, data: pd.DataFrame, entity_id: str) -> List[Dict]:
        """Rolling window analysis to avoid lookahead bias."""
        window_size = self.thresholds["window_size_days"]
        min_days = window_size + 1

        if len(data) < min_days:
            return []

        fatigue_days = []
        consecutive_count = 0

        for i in range(min_days, len(data)):
            # ✅ ONLY use historical data (days [i-window_size : i-1])
            window = data.iloc[i - window_size : i].copy()

            # Calculate cumulative frequency within the window
            window = self._calculate_cumulative_frequency(window)

            # Find golden period
            golden_period = self._find_golden_period(window)

            # Check current day for fatigue
            current = data.iloc[i]
            is_fatigued = self._check_fatigue_conditions(
                current, golden_period, window
            )

            if is_fatigued:
                consecutive_count += 1
                if consecutive_count >= self.thresholds["consecutive_days"]:
                    fatigue_days.append({
                        "date": current["date"],
                        "severity": self._calculate_severity(current, golden_period),
                        "frequency": current.get("cumulative_frequency", 0),
                        "cpa": current.get("cpa", 0),
                    })
            else:
                consecutive_count = 0

        return fatigue_days
```

**Key patterns:**
- `DEFAULT_THRESHOLDS` for tunable parameters
- Rolling window to avoid lookahead bias
- Scoring system (0-100)
- Consecutive day detection

---

### Example 2: PromptEnhancementAgent (from ad/miner)

**Algorithm:** Multi-stage text transformation

**File:** `src/agents/nano/core/agent.py`

```python
class PromptEnhancementAgent:
    """
    Enhances prompts using multi-stage pipeline.

    Pipeline stages:
    1. Parse: Detect category and intent
    2. Enrich: Add product/brand context
    3. Think: Generate strategy (ThinkingBlock)
    4. Build: Create natural language prompt
    5. Apply: Apply Nano Banana Pro techniques
    6. Verify: Quality check
    """

    def enhance(self, agent_input: AgentInput) -> AgentOutput:
        """Main enhancement pipeline."""
        start_time = time.time()

        # Stage 1: Parse
        category, intent = self.parser.parse(agent_input.generic_prompt)

        # Stage 2: Enrich
        enriched = self.context_enrichment.enrich(agent_input)

        # Stage 3: Think
        thinking = self.thinking_engine.generate_thinking(
            enriched, category, intent
        )

        # Stage 4: Build
        base_prompt = self.nl_builder.build(enriched, thinking)

        # Stage 5: Apply techniques
        enhanced, techniques = self.technique_orchestrator.apply(
            base_prompt, thinking
        )

        # Stage 6: Verify
        quality = self.quality_verifier.verify(enhanced, enriched)

        duration_ms = int((time.time() - start_time) * 1000)

        return AgentOutput(
            enhanced_prompt=enhanced,
            techniques_used=techniques,
            confidence=quality.confidence,
            processing_time_ms=duration_ms,
        )
```

**Key patterns:**
- Linear pipeline with clear stages
- Thinking block for strategy
- Technique orchestration
- Quality verification

---

### Example 3: CodeEvolutionTeam (from ad/generator)

**Algorithm:** Multi-agent optimization loop

**File:** `agents/orchestrator.py`

```python
class CodeEvolutionOrchestrator:
    """
    Orchestrates multi-agent team for code evolution.

    Agents:
    - PM: Plans experiments
    - Coder: Implements changes
    - Reviewer: Reviews code
    - Judge: Evaluates performance
    - Memory: Learns from history
    """

    def run_optimization_cycle(
        self,
        detector: str,
        target_metrics: Dict
    ) -> Dict:
        """
        Run complete optimization cycle.
        """
        # Phase 1: PM creates spec with memory context
        memory_context = self.memory.query(
            query_type="SIMILAR_EXPERIMENTS",
            detector=detector,
            tags=target_metrics.get("tags", [])
        )
        spec = self.pm.create_spec(target_metrics, memory_context)

        # Phase 2: Coder implements
        implementation = self.coder.implement(spec)

        # Phase 3: Reviewer reviews
        review = self.reviewer.review(implementation, spec)

        if review["status"] == "rejected":
            return {"success": False, "reason": "Review failed"}

        # Phase 4: Judge evaluates
        evaluation = self.judge.evaluate(implementation, spec)

        # Phase 5: Record in memory
        self.memory.record(spec, implementation, review, evaluation)

        # Phase 6: Rollback if failed
        if evaluation["decision"] == "FAIL":
            self._rollback_changes(implementation)
            return {"success": False, "reason": "Evaluation failed"}

        return {"success": True, "lift": evaluation["lift_score"]}
```

**Key patterns:**
- Multi-agent coordination
- Memory-driven planning
- Automatic rollback on failure
- Lift score optimization

---

## Chapter 10: Best Practices and Anti-Patterns {#chapter-10-best-practices}

### DO: Clear Separation of Concerns

✅ **Good:**
```python
class Agent:
    def __init__(self):
        self.parser = Parser()      # Separate parser
        self.transformer = Transformer()  # Separate transformer
        self.verifier = Verifier()  # Separate verifier
```

❌ **Bad:**
```python
class Agent:
    def process(self, input):
        # Everything in one method
        parsed = self._parse(input)
        transformed = self._transform(parsed)
        verified = self._verify(transformed)
        # Can't test or reuse individual components
```

### DO: Type-Safe Interfaces

✅ **Good:**
```python
@dataclass
class AgentInput:
    core_input: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentOutput:
    result: str
    confidence: float
```

❌ **Bad:**
```python
def process(input):  # What type is input?
    return result  # What type is result?
```

### DO: Comprehensive Logging

✅ **Good:**
```python
logger.info(f"Processing: {input_data[:50]}...")
logger.debug(f"Parsed: {category}, {intent}")
logger.info(f"Complete: {duration_ms}ms, confidence={confidence:.2f}")
```

❌ **Bad:**
```python
# No logging - impossible to debug in production
```

### DON'T: Hard-Code Domain Logic

✅ **Good:**
```python
class AgentConfig:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.thresholds = self.config.get("thresholds", DEFAULT_THRESHOLDS)
```

❌ **Bad:**
```python
class Agent:
    # Can't change without modifying code
    THRESHOLD = 2.5
    WINDOW_SIZE = 7
```

### DON'T: Skip Quality Gates

✅ **Good:**
```python
def process(input):
    output = self._generate(input)
    quality = self._verify(output)
    if not quality.passes:
        raise QualityError(quality.issues)
    return output
```

❌ **Bad:**
```python
def process(input):
    output = self._generate(input)
    # No quality check - could return garbage
    return output
```

### DON'T: Ignore Observability

✅ **Good:**
```python
@dataclass
class AgentState:
    status: AgentStatus
    current_step: str
    step_history: List[str]
    metrics: Dict[str, float]
```

❌ **Bad:**
```python
# No state tracking - can't monitor progress
```

---

## Chapter 11: Quick Reference {#chapter-11-quick-reference}

### Architecture Decision Checklist

```
1. Single domain, linear transformation?
   YES → Use Pipeline Agent (Chapter 3)
   NO  → Continue to 2

2. Multiple similar domains, need reusability?
   YES → Use Adapter Framework (Chapter 4)
   NO  → Continue to 3

3. Complex workflow, specialized roles?
   YES → Use Multi-Agent Orchestration (Chapter 5)
   NO  → Reconsider requirements
```

### Template Code Snippets

**Create a Pipeline Agent:**
```python
from templates.pipeline_agent import PipelineAgent

class MyAgent(PipelineAgent):
    def _parse(self, input_data):
        # Parse and categorize
        return category, intent

    def _enrich(self, input_data, category, intent):
        # Add context
        return enriched_input

    def _transform(self, enriched_input, category, intent):
        # Core logic
        return result

    def _verify(self, result, context):
        # Quality check
        return QualityCheck(...)

    def _format(self, result, quality):
        # Format output
        return AgentOutput(...)
```

**Create an Adapter:**
```python
from templates.base_adapter import BaseAdapter

class MyAdapter(BaseAdapter):
    @property
    def domain(self):
        return "my_domain"

    def parse_input(self, generic_input):
        # Parse logic
        return category, intent

    def enrich_context(self, agent_input):
        # Enrichment logic
        return enriched_input

    def generate_output(self, enriched_input, category, intent, examples):
        # Generation logic
        return output
```

### Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Import errors | Use absolute imports from package root |
| Config not loading | Check file path, use `Path(config_path).exists()` |
| Type errors | Use `@dataclass` with type hints |
| Memory bloat | Implement LRU eviction in memory system |
| Test failures | Mock external dependencies |
| Performance issues | Add caching, profile bottlenecks |
| State corruption | Use immutable dataclasses, copy on modify |

---

## Appendix: File References

### Template Files

All templates are in `agents/templates/`:

- `pipeline_agent.py` - Pipeline architecture template
- `base_agent.py` - Generic framework template
- `base_adapter.py` - Adapter interface template
- `orchestrator.py` - Multi-agent template
- `memory_system.py` - Memory system template
- `quality_verifier.py` - Quality verification template

### Example Implementations

All examples are in `agents/examples/`:

- `simple_detector.py` - Simple pipeline agent example
- `multi_domain_adapter.py` - Adapter-based example
- `optimization_team.py` - Multi-agent example

### Production Examples

**ad/miner:**
- `/ad/miner/src/agents/framework/core/base_agent.py`
- `/ad/miner/src/agents/framework/adapters/base.py`
- `/ad/miner/src/agents/nano/core/agent.py`

**diagnoser:**
- `/ad/generator/src/meta/diagnoser/agents/orchestrator.py`
- `/ad/generator/src/meta/diagnoser/core/issue_detector.py`
- `/ad/generator/src/meta/diagnoser/detectors/fatigue_detector.py`

**ad/generator:**
- `/ad/generator/agents/orchestrator.py`
- `/ad/generator/agents/pm_agent.py`
- `/ad/generator/agents/coder_agent.py`
- `/ad/generator/agents/reviewer_agent.py`
- `/ad/generator/agents/judge_agent.py`
- `/ad/generator/agents/memory_agent.py`

---

**End of Guide**

For questions or contributions, refer to the production examples and templates provided.
