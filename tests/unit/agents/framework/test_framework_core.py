"""
Unit tests for Framework Core Components.
"""

import pytest
import sys
import os
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from src.agents.framework.core.types import (
    FrameworkConfig,
    GroundingExample,
    MemoryEntry,
    QualityCheck,
)
from src.agents.framework.core.examples import ExampleManager
from src.agents.framework.core.reflexion import ReflexionEngine, CritiqueResult
from src.agents.framework.core.memory import MemorySystem
from src.agents.framework.core.quality import QualityVerifier


class TestFrameworkTypes:
    """Test framework data types."""

    def test_framework_config_defaults(self):
        """Test FrameworkConfig default values."""
        config = FrameworkConfig()
        assert config.examples_db_path == "data/agents/examples.json"
        assert config.enable_reflexion is True
        assert config.enable_memory is True
        assert config.max_reflexion_iterations == 2
        assert config.quality_threshold == 0.7

    def test_grounding_example_auto_id(self):
        """Test GroundingExample generates ID automatically."""
        example = GroundingExample(
            input_prompt="test",
            output_prompt="enhanced",
            domain="nano",
            category="ultra_simple",
            intent="product_photography",
        )
        assert example.example_id != ""
        assert len(example.example_id) == 12

    def test_memory_entry_auto_id(self):
        """Test MemoryEntry generates ID automatically."""
        entry = MemoryEntry(
            input_prompt="test",
            enhanced_prompt="enhanced",
            domain="nano",
            detected_category="ultra_simple",
            detected_intent="product_photography",
            confidence=0.8,
            techniques_used=["text_rendering"],
        )
        assert entry.entry_id != ""
        assert len(entry.entry_id) == 12

    def test_quality_check_to_dict(self):
        """Test QualityCheck serialization."""
        check = QualityCheck(
            passes=True,
            confidence=0.85,
            issues=[],
            specificity_score=0.8,
            pattern_consistency=0.9,
            natural_language_score=0.85,
            completeness_score=0.8,
        )
        data = check.to_dict()
        assert data["passes"] is True
        assert data["confidence"] == 0.85
        assert "specificity_score" in data


class TestExampleManager:
    """Test ExampleManager component."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary file for tests
        self.temp_db = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        )
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

        self.manager = ExampleManager(examples_db_path=self.temp_db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)

    def test_initialization_creates_db(self):
        """Test that initialization creates database file."""
        assert os.path.exists(self.temp_db_path)

    def test_add_example(self):
        """Test adding an example."""
        example = GroundingExample(
            input_prompt="test input",
            output_prompt="enhanced output",
            domain="nano",
            category="ultra_simple",
            intent="product_photography",
        )
        self.manager.add_example(example)

        stats = self.manager.get_stats()
        assert stats["total_examples"] == 1
        assert "nano" in stats["domains"]

    def test_retrieve_relevant_examples(self):
        """Test retrieving relevant examples."""
        # Add test examples
        examples = [
            GroundingExample(
                input_prompt="Create an ad for our mop",
                output_prompt="Enhanced mop ad",
                domain="nano",
                category="ultra_simple",
                intent="product_photography",
            ),
            GroundingExample(
                input_prompt="Generate art for headphones",
                output_prompt="Enhanced headphone art",
                domain="nano",
                category="specific_request",
                intent="product_photography",
            ),
        ]

        for ex in examples:
            self.manager.add_example(ex)

        # Retrieve similar example
        results = self.manager.retrieve_relevant(
            "Create an ad for a product", domain="nano", k=1
        )

        assert len(results) > 0
        assert results[0].domain == "nano"

    def test_get_stats(self):
        """Test getting statistics."""
        stats = self.manager.get_stats()
        assert "total_examples" in stats
        assert "domains" in stats
        assert "examples_per_domain" in stats


class TestReflexionEngine:
    """Test ReflexionEngine component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ReflexionEngine(max_iterations=2, quality_threshold=0.7)

    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.max_iterations == 2
        assert self.engine.quality_threshold == 0.7

    def test_refine_returns_tuple(self):
        """Test that refine returns tuple of (prompt, critique_history)."""
        from src.agents.framework.adapters.base import BaseAdapter, AdapterConfig
        from src.agents.nano.core.types import AgentInput

        # Create a mock adapter
        class MockAdapter(BaseAdapter):
            @property
            def domain(self):
                return "test"

            def parse_input(self, prompt):
                return ("ultra_simple", "product_photography")

            def enrich_context(self, agent_input):
                return agent_input

            def generate_thinking(self, agent_input, category, intent, examples):
                return "Test thinking"

            def build_prompt(self, agent_input, category, intent):
                return "Test prompt"

            def apply_techniques(self, prompt, thinking):
                return prompt

            def refine_prompt(self, prompt, critique, agent_input):
                return prompt + " [Refined]"

        adapter = MockAdapter(AdapterConfig(domain="test"))
        agent_input = AgentInput(generic_prompt="Test")

        refined, history = self.engine.refine("Test prompt", agent_input, adapter)

        assert isinstance(refined, str)
        assert isinstance(history, list)


class TestMemorySystem:
    """Test MemorySystem component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        )
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

        self.memory = MemorySystem(memory_db_path=self.temp_db_path, max_entries=100)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)

    def test_initialization_creates_db(self):
        """Test that initialization creates database file."""
        assert os.path.exists(self.temp_db_path)

    def test_add_entry(self):
        """Test adding a memory entry."""
        entry = MemoryEntry(
            input_prompt="test input",
            enhanced_prompt="enhanced output",
            domain="nano",
            detected_category="ultra_simple",
            detected_intent="product_photography",
            confidence=0.8,
            techniques_used=["text_rendering"],
        )
        self.memory.add_entry(entry)

        stats = self.memory.get_stats()
        assert stats["total_entries"] == 1
        assert "nano" in stats["domains"]

    def test_get_stats(self):
        """Test getting statistics."""
        stats = self.memory.get_stats()
        assert "total_entries" in stats
        assert "max_entries" in stats
        assert "domains" in stats

    def test_clear_domain(self):
        """Test clearing memories for a domain."""
        entry = MemoryEntry(
            input_prompt="test",
            enhanced_prompt="enhanced",
            domain="nano",
            detected_category="ultra_simple",
            detected_intent="product_photography",
            confidence=0.8,
            techniques_used=[],
        )
        self.memory.add_entry(entry)
        self.memory.clear_domain("nano")

        stats = self.memory.get_stats()
        assert stats["total_entries"] == 0

    def test_lru_eviction(self):
        """Test that LRU eviction works when max_entries is exceeded."""
        # Create memory system with small max
        small_memory = MemorySystem(
            memory_db_path=self.temp_db_path, max_entries=3
        )

        # Add 5 entries
        for i in range(5):
            entry = MemoryEntry(
                input_prompt=f"test input {i}",
                enhanced_prompt=f"enhanced output {i}",
                domain="nano",
                detected_category="ultra_simple",
                detected_intent="product_photography",
                confidence=0.8,
                techniques_used=[],
            )
            small_memory.add_entry(entry)

        # Should only have 3 entries (max)
        stats = small_memory.get_stats()
        assert stats["total_entries"] == 3
        assert stats["max_entries"] == 3

    def test_add_feedback(self):
        """Test adding feedback to a memory entry."""
        entry = MemoryEntry(
            input_prompt="test input",
            enhanced_prompt="enhanced output",
            domain="nano",
            detected_category="ultra_simple",
            detected_intent="product_photography",
            confidence=0.8,
            techniques_used=[],
        )
        self.memory.add_entry(entry)

        # Add feedback
        self.memory.add_feedback(entry.entry_id, "This was helpful")

        # Reload and check feedback
        new_memory = MemorySystem(
            memory_db_path=self.temp_db_path, max_entries=100
        )
        assert len(new_memory.memories) == 1
        assert new_memory.memories[0].user_feedback == "This was helpful"

    def test_find_similar_no_memories(self):
        """Test find_similar when no memories exist."""
        from src.agents.framework.adapters.base import BaseAdapter, AdapterConfig

        class MockAdapter(BaseAdapter):
            @property
            def domain(self):
                return "test"

            def parse_input(self, generic_prompt: str):
                return "test_category", "test_intent"

            def enrich_context(self, agent_input):
                return agent_input

            def generate_thinking(self, agent_input, category, intent, examples):
                return "thinking"

            def build_prompt(self, agent_input, category, intent):
                return "prompt"

            def apply_techniques(self, prompt, thinking):
                return prompt

            def refine_prompt(self, prompt, critique, agent_input):
                return prompt

            def validate_domain_specific(self, prompt):
                return []

            def compute_similarity(self, prompt1: str, prompt2: str) -> float:
                return 0.5

        adapter = MockAdapter(AdapterConfig(domain="test"))
        similar = self.memory.find_similar("test prompt", "nano", adapter, k=3)

        assert similar == []

    def test_find_similar_with_memories(self):
        """Test find_similar with existing memories."""
        from src.agents.framework.adapters.base import BaseAdapter, AdapterConfig

        # Add some memories
        for i in range(3):
            entry = MemoryEntry(
                input_prompt=f"test input {i}",
                enhanced_prompt=f"enhanced output {i}",
                domain="nano",
                detected_category="ultra_simple",
                detected_intent="product_photography",
                confidence=0.8,  # High enough to be included
                techniques_used=[],
            )
            self.memory.add_entry(entry)

        # Add low confidence entry (should be filtered out)
        low_entry = MemoryEntry(
            input_prompt="low confidence input",
            enhanced_prompt="low confidence output",
            domain="nano",
            detected_category="ultra_simple",
            detected_intent="product_photography",
            confidence=0.5,  # Too low
            techniques_used=[],
        )
        self.memory.add_entry(low_entry)

        class MockAdapter(BaseAdapter):
            def __init__(self, compute_sim_func):
                self.compute_sim_func = compute_sim_func
                self._domain = "test"

            @property
            def domain(self):
                return self._domain

            def parse_input(self, generic_prompt: str):
                return "test_category", "test_intent"

            def enrich_context(self, agent_input):
                return agent_input

            def generate_thinking(self, agent_input, category, intent, examples):
                return "thinking"

            def build_prompt(self, agent_input, category, intent):
                return "prompt"

            def apply_techniques(self, prompt, thinking):
                return prompt

            def refine_prompt(self, prompt, critique, agent_input):
                return prompt

            def validate_domain_specific(self, prompt):
                return []

            def compute_similarity(self, prompt1: str, prompt2: str) -> float:
                return self.compute_sim_func(prompt1, prompt2)

        # Mock similarity function
        def mock_similarity(p1, p2):
            return 0.9 if "0" in p2 else 0.5

        adapter = MockAdapter(mock_similarity)
        similar = self.memory.find_similar("test", "nano", adapter, k=2)

        # Should return 2 memories (k=2), filtered by confidence >= 0.7
        assert len(similar) == 2


class TestQualityVerifier:
    """Test QualityVerifier component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = QualityVerifier(threshold=0.7)

    def test_initialization(self):
        """Test verifier initialization."""
        assert self.verifier.threshold == 0.7

    def test_verify_returns_quality_check(self):
        """Test that verify returns QualityCheck object."""
        from src.agents.nano.core.types import AgentInput
        from src.agents.framework.adapters.base import BaseAdapter, AdapterConfig

        # Create a mock adapter
        class MockAdapter(BaseAdapter):
            @property
            def domain(self):
                return "test"

            def parse_input(self, prompt):
                return ("ultra_simple", "product_photography")

            def enrich_context(self, agent_input):
                return agent_input

            def generate_thinking(self, agent_input, category, intent, examples):
                return "Test thinking"

            def build_prompt(self, agent_input, category, intent):
                return "Test prompt"

            def apply_techniques(self, prompt, thinking):
                return prompt

            def refine_prompt(self, prompt, critique, agent_input):
                return prompt

        adapter = MockAdapter(AdapterConfig(domain="test"))
        agent_input = AgentInput(generic_prompt="Create an ad for our mop")

        check = self.verifier.verify(
            "A professional product photograph showing our 360° rotating mop with spray mechanism",
            agent_input,
            adapter,
        )

        assert isinstance(check, QualityCheck)
        assert isinstance(check.passes, bool)
        assert isinstance(check.confidence, float)
        assert 0 <= check.confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
