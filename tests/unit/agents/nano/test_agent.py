"""
Unit tests for Nano Banana Pro Prompt Enhancement Agent.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

from src.agents.nano.core.types import (
    AgentInput,
    PromptCategory,
    PromptIntent,
    Resolution,
)
from src.agents.nano import PromptEnhancementAgent


class TestPromptEnhancementAgent:
    """Test the main PromptEnhancementAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PromptEnhancementAgent()

    def test_agent_initialization(self):
        """Test that agent initializes correctly."""
        assert self.agent is not None
        assert hasattr(self.agent, 'enhance')

    def test_enhance_ultra_simple_prompt(self):
        """Test enhancing an ultra-simple prompt."""
        input_prompt = AgentInput(
            generic_prompt="Create an ad for our mop",
        )

        output = self.agent.enhance(input_prompt)

        # Verify output structure
        assert output is not None
        assert output.enhanced_prompt is not None
        assert len(output.enhanced_prompt) > 100  # Should be much longer than input

        # Verify metadata
        assert output.detected_category is not None
        assert output.detected_intent is not None
        assert output.confidence >= 0.0
        assert output.confidence <= 1.0
        assert output.processing_time_ms >= 0

        # Verify techniques were applied
        assert len(output.techniques_used) > 0

    def test_enhance_with_product_context(self):
        """Test enhancing with product context provided."""
        from src.agents.nano.core.types import ProductContext

        product_context = ProductContext(
            name="test mop",
            category="cleaning_tools",
            key_features=["360° rotation", "spray mechanism"],
            materials=["microfiber", "plastic"],
            colors=["red", "black"],
        )

        input_prompt = AgentInput(
            generic_prompt="Create an ad for our mop",
            product_context=product_context,
        )

        output = self.agent.enhance(input_prompt)

        # Should have higher confidence with product context
        assert output.confidence >= 0.5

    def test_enhance_with_thinking_enabled(self):
        """Test enhancing with thinking block enabled."""
        input_prompt = AgentInput(
            generic_prompt="Create an ad for our mop",
            enable_thinking=True,
        )

        output = self.agent.enhance(input_prompt)

        # Should have thinking block
        assert output.thinking_block is not None
        assert output.thinking_block.analysis != ""
        assert len(output.thinking_block.techniques) > 0

    def test_enhance_with_thinking_disabled(self):
        """Test enhancing with thinking block disabled."""
        input_prompt = AgentInput(
            generic_prompt="Create an ad for our mop",
            enable_thinking=False,
        )

        output = self.agent.enhance(input_prompt)

        # Should still have thinking_block object but not in prompt
        # (The formatter should not include it in enhanced_prompt)
        assert output.thinking_block is not None

    def test_simple_interface(self):
        """Test the simple enhancement interface."""
        enhanced = self.agent.enhance_simple("Create an ad for our mop")

        assert enhanced is not None
        assert len(enhanced) > 100

    def test_convenience_function(self):
        """Test the module-level convenience function."""
        from src.agents.nano import enhance_prompt

        enhanced = enhance_prompt("Create an ad for our mop")

        assert enhanced is not None
        assert len(enhanced) > 100


class TestInputParser:
    """Test the InputParser component."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.agents.nano.parsers.input_parser import InputParser
        self.parser = InputParser()

    def test_parse_ultra_simple(self):
        """Test parsing an ultra-simple prompt."""
        category, intent = self.parser.parse("Create an ad for our mop")

        assert category == PromptCategory.ULTRA_SIMPLE
        assert intent in [PromptIntent.LIFESTYLE_ADVERTISEMENT, PromptIntent.PRODUCT_PHOTOGRAPHY]

    def test_parse_comparative(self):
        """Test parsing a comparative prompt."""
        category, intent = self.parser.parse("Compare our mop to competitors")

        assert category == PromptCategory.COMPARATIVE
        assert intent == PromptIntent.COMPARATIVE_INFOGRAPHIC

    def test_parse_sequential(self):
        """Test parsing a sequential prompt."""
        category, intent = self.parser.parse("Show a story of cleaning")

        assert category == PromptCategory.SEQUENTIAL
        assert intent == PromptIntent.STORYBOARD_SEQUENCE

    def test_identify_missing_elements(self):
        """Test identifying missing elements."""
        missing = self.parser.identify_missing_elements("Create an ad")

        # Should identify several missing elements
        assert len(missing) > 0


class TestQualityVerifier:
    """Test the QualityVerifier component."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.agents.nano.core.quality_verifier import QualityVerifier
        self.verifier = QualityVerifier()

    def test_verify_short_prompt(self):
        """Test verification of a short prompt."""
        passes, confidence, issues = self.verifier.verify(
            prompt="Create an ad",
            techniques_applied=[],
            has_thinking=False,
        )

        # Short prompt should fail
        assert not passes
        assert confidence < 0.5
        assert len(issues) > 0

    def test_verify_good_prompt(self):
        """Test verification of a good prompt."""
        good_prompt = """
        A professional product photograph showing the mop in detail.
        The mop displays 360° rotation and spray mechanism.
        Materials include microfiber and plastic with accurate color representation.
        Lighting is soft, diffused natural light.
        For e-commerce customers.
        Resolution: 2K. Style: Professional photography.
        """

        passes, confidence, issues = self.verifier.verify(
            prompt=good_prompt,
            techniques_applied=["natural_language", "character_consistency"],
            has_thinking=True,
        )

        # Good prompt should pass
        assert passes
        assert confidence >= 0.6


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
