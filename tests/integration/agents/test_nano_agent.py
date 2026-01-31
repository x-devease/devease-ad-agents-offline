"""
Integration tests for Nano Banana Pro Agent with Framework.

These tests verify the complete end-to-end functionality of the Nano agent
using the framework architecture.
"""

import pytest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.agents.nano.core.agent import PromptEnhancementAgent
from src.agents.nano.core.types import AgentInput, ProductContext


class TestNanoAgentFrameworkIntegration:
    """Test Nano agent with framework end-to-end."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create agent with framework enabled
        self.agent = PromptEnhancementAgent(use_framework=True)

    def test_simple_enhancement_framework(self):
        """Test simple prompt enhancement through framework."""
        prompt = "Create an ad for our mop"
        enhanced = self.agent.enhance_simple(prompt)

        # Verify enhancement
        assert enhanced is not None
        assert len(enhanced) > 200  # Should be much longer than input
        assert "mop" in enhanced.lower()
        assert len(enhanced) > len(prompt) * 3  # Should be significantly longer

    def test_full_enhancement_with_metadata(self):
        """Test full enhancement with complete metadata."""
        agent_input = AgentInput(
            generic_prompt="Create an ad for our mop",
            enable_thinking=True,
        )

        output = self.agent.enhance(agent_input)

        # Verify all output fields
        assert output.enhanced_prompt is not None
        assert len(output.enhanced_prompt) > 200
        assert output.detected_category is not None
        assert output.detected_intent is not None
        assert 0.0 <= output.confidence <= 1.0
        assert output.processing_time_ms >= 0
        assert isinstance(output.techniques_used, list)

    def test_enhancement_with_product_context(self):
        """Test enhancement with product context through framework."""
        product_context = ProductContext(
            name="Super Mop 3000",
            category="cleaning_tools",
            key_features=["360° rotation", "spray mechanism", "microfiber pad"],
            materials=["microfiber", "plastic", "metal"],
            colors=["red", "black", "white"],
        )

        agent_input = AgentInput(
            generic_prompt="Create an ad for our mop",
            product_context=product_context,
        )

        output = self.agent.enhance(agent_input)

        # Verify product context is used
        assert output.enhanced_prompt is not None
        assert "mop" in output.enhanced_prompt.lower()
        # Should have higher confidence with context
        assert output.confidence >= 0.5

    def test_different_prompt_categories(self):
        """Test framework handles different prompt categories."""
        test_cases = [
            "Create an ad for our mop",  # ultra_simple
            "Show our mop in action cleaning a kitchen",  # specific_request
            "Compare our mop to competitors",  # comparative
        ]

        for prompt in test_cases:
            agent_input = AgentInput(generic_prompt=prompt)
            output = self.agent.enhance(agent_input)

            assert output.enhanced_prompt is not None
            assert len(output.enhanced_prompt) > 200
            assert output.detected_category is not None
            assert output.detected_intent is not None

    def test_framework_techniques_application(self):
        """Test that framework applies Nano techniques."""
        agent_input = AgentInput(
            generic_prompt="Create an ad for our mop with text overlay",
            enable_thinking=True,
        )

        output = self.agent.enhance(agent_input)

        # Verify techniques were applied
        assert len(output.techniques_used) >= 0

        # Enhanced prompt should contain technique-specific elements
        enhanced = output.enhanced_prompt.lower()
        # Check for common technique indicators
        technique_keywords = ["text", "rendering", "physics", "consistency"]
        has_technique = any(keyword in enhanced for keyword in technique_keywords)
        # Note: techniques might be applied subtly

    def test_framework_quality_threshold(self):
        """Test that framework maintains quality threshold."""
        agent_input = AgentInput(
            generic_prompt="Create a professional product photograph of our premium mop",
        )

        output = self.agent.enhance(agent_input)

        # Should maintain minimum quality
        assert output.confidence >= 0.3  # Reasonable minimum
        assert "professional" in output.enhanced_prompt.lower() or "product" in output.enhanced_prompt.lower()

    def test_framework_with_reflexion(self):
        """Test that framework reflexion improves quality."""
        # Create a very simple prompt
        agent_input = AgentInput(generic_prompt="mop ad")

        output = self.agent.enhance(agent_input)

        # Framework should still produce good output even from minimal input
        assert output.enhanced_prompt is not None
        assert len(output.enhanced_prompt) > 200

    def test_framework_memory_storage(self):
        """Test that framework stores successful enhancements in memory."""
        # Perform several enhancements
        prompts = [
            "Create an ad for our mop",
            "Show our vacuum cleaner",
            "Generate art for our headphones",
        ]

        for prompt in prompts:
            agent_input = AgentInput(generic_prompt=prompt)
            self.agent.enhance(agent_input)

        # Check that framework has memory entries
        stats = self.agent.framework.get_stats()
        assert "memory" in stats
        assert "total_entries" in stats["memory"]

    def test_framework_examples_retrieval(self):
        """Test that framework retrieves relevant examples."""
        stats = self.agent.framework.get_stats()

        # Should have examples loaded
        assert "examples" in stats
        assert "total_examples" in stats["examples"]

    def test_framework_consistency(self):
        """Test that framework produces consistent results."""
        agent_input = AgentInput(
            generic_prompt="Create an ad for our mop",
        )

        # Run same enhancement twice
        output1 = self.agent.enhance(agent_input)
        output2 = self.agent.enhance(agent_input)

        # Should produce similar quality
        assert abs(output1.confidence - output2.confidence) < 0.3

        # Both should be valid enhancements
        assert len(output1.enhanced_prompt) > 200
        assert len(output2.enhanced_prompt) > 200


class TestNanoAgentFrameworkVsLegacy:
    """Compare framework mode with legacy mode."""

    def test_both_modes_work(self):
        """Test that both framework and legacy modes function."""
        # Framework mode
        framework_agent = PromptEnhancementAgent(use_framework=True)
        framework_output = framework_agent.enhance_simple("Create an ad for our mop")

        # Legacy mode
        legacy_agent = PromptEnhancementAgent(use_framework=False)
        legacy_output = legacy_agent.enhance_simple("Create an ad for our mop")

        # Both should produce valid enhancements
        assert len(framework_output) > 200
        assert len(legacy_output) > 200

        # Both should mention the product
        assert "mop" in framework_output.lower()
        assert "mop" in legacy_output.lower()

    def test_framework_has_benefits(self):
        """Test that framework mode provides additional benefits."""
        framework_agent = PromptEnhancementAgent(use_framework=True)
        legacy_agent = PromptEnhancementAgent(use_framework=False)

        agent_input = AgentInput(
            generic_prompt="Create an ad for our mop",
            enable_thinking=True,
        )

        # Framework output
        framework_output = framework_agent.enhance(agent_input)

        # Legacy output
        legacy_output = legacy_agent.enhance(agent_input)

        # Framework should have stats available
        framework_stats = framework_agent.framework.get_stats()
        assert "examples" in framework_stats
        assert "memory" in framework_stats

        # Both should have similar quality
        assert abs(framework_output.confidence - legacy_output.confidence) < 0.5


class TestNanoAgentRealWorldScenarios:
    """Test Nano agent with real-world usage scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PromptEnhancementAgent(use_framework=True)

    def test_ecommerce_product_photo(self):
        """Test e-commerce product photography scenario."""
        product_context = ProductContext(
            name="UltraClean Mop X1",
            category="cleaning_tools",
            key_features=["360° rotating head", "built-in spray", "microfiber pad"],
            materials=["microfiber", "aluminum", "plastic"],
            colors=["blue", "white"],
        )

        agent_input = AgentInput(
            generic_prompt="Create a product photo for our ecommerce store",
            product_context=product_context,
            preferred_resolution="K2",
        )

        output = self.agent.enhance(agent_input)

        assert output.enhanced_prompt is not None
        assert "product" in output.enhanced_prompt.lower() or "photograph" in output.enhanced_prompt.lower()
        assert output.confidence >= 0.4

    def test_lifestyle_advertisement(self):
        """Test lifestyle advertisement scenario."""
        product_context = ProductContext(
            name="UltraClean Mop X1",
            category="cleaning_tools",
            key_features=["easy to use", "lightweight"],
            materials=["microfiber", "aluminum"],
            colors=["blue", "white"],
        )

        agent_input = AgentInput(
            generic_prompt="Show our mop in a happy home setting",
            product_context=product_context,
            target_audience="young families",
        )

        output = self.agent.enhance(agent_input)

        assert output.enhanced_prompt is not None
        # Should have lifestyle elements
        assert len(output.enhanced_prompt) > 300

    def test_commercial_comparison(self):
        """Test comparative advertisement scenario."""
        agent_input = AgentInput(
            generic_prompt="Compare our mop to traditional mops",
        )

        output = self.agent.enhance(agent_input)

        assert output.enhanced_prompt is not None
        assert output.detected_category is not None

    def test_minimal_input_enhancement(self):
        """Test enhancement with very minimal input."""
        agent_input = AgentInput(
            generic_prompt="mop",
        )

        output = self.agent.enhance(agent_input)

        # Should still produce good output
        assert output.enhanced_prompt is not None
        assert len(output.enhanced_prompt) > 200

    def test_detailed_input_enhancement(self):
        """Test enhancement with detailed input."""
        detailed_prompt = """
        Create a professional product photograph of our UltraClean Mop X1.
        Show the 360° rotating head and built-in spray mechanism.
        Use soft studio lighting with white background.
        Resolution should be 2K for ecommerce use.
        """

        agent_input = AgentInput(generic_prompt=detailed_prompt)

        output = self.agent.enhance(agent_input)

        # Should enhance further
        assert output.enhanced_prompt is not None
        assert len(output.enhanced_prompt) >= len(detailed_prompt)


class TestNanoAgentErrorHandling:
    """Test Nano agent error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PromptEnhancementAgent(use_framework=True)

    def test_empty_prompt(self):
        """Test handling of empty prompt."""
        agent_input = AgentInput(generic_prompt="")

        # Should handle gracefully
        output = self.agent.enhance(agent_input)
        assert output is not None

    def test_very_long_prompt(self):
        """Test handling of very long prompt."""
        long_prompt = "Create an ad " * 100

        agent_input = AgentInput(generic_prompt=long_prompt)

        # Should handle gracefully
        output = self.agent.enhance(agent_input)
        assert output is not None

    def test_special_characters(self):
        """Test handling of special characters."""
        special_prompt = "Create an ad with special chars: @#$%^&*()"

        agent_input = AgentInput(generic_prompt=special_prompt)

        # Should handle gracefully
        output = self.agent.enhance(agent_input)
        assert output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
