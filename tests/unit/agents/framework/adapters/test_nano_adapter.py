"""
Unit tests for Nano Adapter.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from src.agents.framework.adapters.nano import NanoAdapter
from src.agents.framework.adapters.base import AdapterConfig
from src.agents.nano.core.types import AgentInput


class TestNanoAdapter:
    """Test NanoAdapter implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        config = AdapterConfig(domain="nano")
        self.adapter = NanoAdapter(config)

    def test_initialization(self):
        """Test adapter initialization."""
        assert self.adapter.domain == "nano"

    def test_parse_input_ultra_simple(self):
        """Test parsing an ultra-simple prompt."""
        category, intent = self.adapter.parse_input("Create an ad for our mop")

        assert category == "ultra_simple" or category in [
            "ultra_simple",
            "basic_direction",
        ]
        assert intent in ["product_photography", "lifestyle_advertisement"]

    def test_enrich_context(self):
        """Test context enrichment."""
        agent_input = AgentInput(generic_prompt="Create an ad for our mop")
        enriched = self.adapter.enrich_context(agent_input)

        # Should return same object (or modified version)
        assert isinstance(enriched, AgentInput)
        assert enriched.generic_prompt == "Create an ad for our mop"

    def test_generate_thinking(self):
        """Test thinking generation."""
        agent_input = AgentInput(generic_prompt="Create an ad for our mop")
        category = "ultra_simple"
        intent = "product_photography"

        thinking = self.adapter.generate_thinking(agent_input, category, intent, [])

        assert isinstance(thinking, str)
        assert len(thinking) > 0

    def test_build_prompt(self):
        """Test prompt building."""
        agent_input = AgentInput(generic_prompt="Create an ad for our mop")
        category = "ultra_simple"
        intent = "product_photography"

        prompt = self.adapter.build_prompt(agent_input, category, intent)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "mop" in prompt.lower()

    def test_apply_techniques(self):
        """Test technique application."""
        prompt = "A product photograph of our mop"
        thinking = "We should use text rendering and character consistency techniques"

        enhanced = self.adapter.apply_techniques(prompt, thinking)

        assert isinstance(enhanced, str)
        assert len(enhanced) >= len(prompt)

    def test_refine_prompt(self):
        """Test prompt refinement."""
        prompt = "A product photograph"
        critique = "The prompt is too short and lacks specific details"

        refined = self.adapter.refine_prompt(prompt, critique, AgentInput(generic_prompt="test"))

        assert isinstance(refined, str)
        assert len(refined) >= len(prompt)

    def test_validate_domain_specific_no_techniques(self):
        """Test domain validation with no techniques."""
        issues = self.adapter.validate_domain_specific("A simple prompt")

        # Should have issues since no Nano techniques are present
        assert len(issues) > 0

    def test_validate_domain_specific_with_techniques(self):
        """Test domain validation with techniques."""
        prompt = "A product photograph with text rendering and character consistency"
        issues = self.adapter.validate_domain_specific(prompt)

        # Should have fewer or no issues
        assert isinstance(issues, list)

    def test_compute_similarity(self):
        """Test similarity computation."""
        prompt1 = "Create an ad for our mop with 360 rotation"
        prompt2 = "Generate an image showing our mop with spray mechanism"

        similarity = self.adapter.compute_similarity(prompt1, prompt2)

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        # Should have some similarity due to "mop"
        assert similarity > 0

    def test_extract_techniques_from_thinking(self):
        """Test extracting techniques from thinking block."""
        thinking = "We should use text rendering and physics understanding"
        techniques = self.adapter._extract_techniques_from_thinking(thinking)

        assert isinstance(techniques, list)
        assert "text rendering" in techniques


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
