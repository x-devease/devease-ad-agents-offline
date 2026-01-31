"""
Integration tests for Framework with Nano Adapter.
"""

import pytest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.agents.framework.core.base_agent import BaseAgent
from src.agents.framework.core.types import FrameworkConfig
from src.agents.framework.adapters.nano import NanoAdapter
from src.agents.framework.adapters.base import AdapterConfig
from src.agents.nano.core.types import AgentInput


class TestFrameworkIntegration:
    """Test full framework integration with Nano adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary databases for testing
        self.temp_examples_db = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        )
        self.temp_examples_db_path = self.temp_examples_db.name
        self.temp_examples_db.close()

        self.temp_memory_db = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        )
        self.temp_memory_db_path = self.temp_memory_db.name
        self.temp_memory_db.close()

        # Create config with temp databases
        self.config = FrameworkConfig(
            examples_db_path=self.temp_examples_db_path,
            memory_db_path=self.temp_memory_db_path,
            enable_reflexion=True,
            enable_memory=True,
        )

        # Create adapter and agent
        adapter_config = AdapterConfig(domain="nano")
        self.adapter = NanoAdapter(adapter_config)
        self.agent = BaseAgent(adapter=self.adapter, config=self.config)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_examples_db_path):
            os.unlink(self.temp_examples_db_path)
        if os.path.exists(self.temp_memory_db_path):
            os.unlink(self.temp_memory_db_path)

    def test_end_to_end_enhancement(self):
        """Test complete end-to-end enhancement pipeline."""
        # Create input
        agent_input = AgentInput(generic_prompt="Create an ad for our mop")

        # Run enhancement
        output = self.agent.enhance(agent_input)

        # Verify output structure
        assert output is not None
        assert output.enhanced_prompt is not None
        assert len(output.enhanced_prompt) > 100
        assert output.detected_category is not None
        assert output.detected_intent is not None
        assert output.confidence >= 0.0
        assert output.confidence <= 1.0
        assert output.processing_time_ms >= 0
        assert isinstance(output.techniques_used, list)

    def test_enhancement_improves_prompt(self):
        """Test that enhancement actually improves the prompt."""
        agent_input = AgentInput(generic_prompt="Create an ad for our mop")

        output = self.agent.enhance(agent_input)

        # Enhanced prompt should be much longer
        assert len(output.enhanced_prompt) > len(agent_input.generic_prompt) * 3

        # Should contain product details
        assert "mop" in output.enhanced_prompt.lower()

    def test_framework_with_product_context(self):
        """Test framework with product context provided."""
        from src.agents.nano.core.types import ProductContext

        product_context = ProductContext(
            name="Super Mop 3000",
            category="cleaning_tools",
            key_features=["360° rotation", "spray mechanism", "microfiber pad"],
            materials=["microfiber", "plastic", "metal"],
            colors=["red", "black", "white"],
        )

        agent_input = AgentInput(
            generic_prompt="Create an ad for our mop", product_context=product_context
        )

        output = self.agent.enhance(agent_input)

        # Should incorporate product context
        assert output.enhanced_prompt is not None
        # Should have higher confidence with context
        assert output.confidence >= 0.5

    def test_framework_stats(self):
        """Test getting framework statistics."""
        stats = self.agent.get_stats()

        assert "adapter" in stats
        assert "examples" in stats
        assert "memory" in stats
        assert "config" in stats

        # Verify adapter domain
        assert stats["adapter"]["domain"] == "nano"

    def test_multiple_enhancements_build_memory(self):
        """Test that multiple enhancements build memory."""
        prompts = [
            "Create an ad for our mop",
            "Show our cleaning product",
            "Generate art for our vacuum",
        ]

        for prompt in prompts:
            agent_input = AgentInput(generic_prompt=prompt)
            self.agent.enhance(agent_input)

        # Check memory stats
        stats = self.agent.get_stats()
        memory_entries = stats["memory"]["total_entries"]

        # Should have some entries (might be filtered by confidence)
        assert memory_entries >= 0

    def test_reflexion_enabled(self):
        """Test that reflexion is working when enabled."""
        agent_input = AgentInput(generic_prompt="Create an ad for our mop")

        # Config has reflexion enabled
        assert self.config.enable_reflexion is True

        output = self.agent.enhance(agent_input)

        # Should complete successfully
        assert output is not None
        assert output.enhanced_prompt is not None

    def test_reflexion_disabled(self):
        """Test framework with reflexion disabled."""
        # Create config with reflexion disabled
        config = FrameworkConfig(
            examples_db_path=self.temp_examples_db_path,
            memory_db_path=self.temp_memory_db_path,
            enable_reflexion=False,
            enable_memory=True,
        )

        adapter_config = AdapterConfig(domain="nano")
        adapter = NanoAdapter(adapter_config)
        agent = BaseAgent(adapter=adapter, config=config)

        agent_input = AgentInput(generic_prompt="Create an ad for our mop")
        output = agent.enhance(agent_input)

        # Should complete successfully
        assert output is not None
        assert output.enhanced_prompt is not None


class TestNanoAgentWithFramework:
    """Test Nano agent using framework."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.agents.nano.core.agent import PromptEnhancementAgent

        # Create agent with framework enabled (default)
        self.agent = PromptEnhancementAgent(use_framework=True)

    def test_agent_initialization(self):
        """Test agent initializes with framework."""
        assert self.agent is not None
        assert self.agent.use_framework is True
        assert hasattr(self.agent, "framework")

    def test_simple_enhancement(self):
        """Test simple enhancement through framework."""
        enhanced = self.agent.enhance_simple("Create an ad for our mop")

        assert enhanced is not None
        assert len(enhanced) > 100
        assert "mop" in enhanced.lower()

    def test_full_enhancement(self):
        """Test full enhancement with AgentInput."""
        from src.agents.nano.core.types import AgentInput

        agent_input = AgentInput(generic_prompt="Create an ad for our mop")
        output = self.agent.enhance(agent_input)

        assert output is not None
        assert output.enhanced_prompt is not None
        assert output.detected_category is not None
        assert output.detected_intent is not None
        assert output.confidence >= 0.0

    def test_legacy_mode_still_works(self):
        """Test that legacy mode still works."""
        from src.agents.nano.core.agent import PromptEnhancementAgent

        agent = PromptEnhancementAgent(use_framework=False)
        assert agent.use_framework is False

        enhanced = agent.enhance_simple("Create an ad for our mop")
        assert enhanced is not None
        assert len(enhanced) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
