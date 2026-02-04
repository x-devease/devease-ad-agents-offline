"""Unit tests for InputParser component."""

import pytest
from src.agents.nano.parsers.input_parser import InputParser
from src.agents.nano.core.types import PromptCategory, PromptIntent


class TestInputParserMissingElements:
    """Test InputParser missing element identification."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = InputParser()

    def test_identify_missing_elements_empty_prompt(self):
        """Test missing elements detection with empty prompt."""
        missing = self.parser.identify_missing_elements("")
        assert len(missing) > 0
        assert "lighting description" in missing
        assert "visual style" in missing
        assert "emotion/mood" in missing
        assert "resolution" in missing

    def test_identify_missing_elements_complete_prompt(self):
        """Test missing elements with detailed prompt."""
        detailed = (
            "Create a 4k high resolution photo with natural lighting, "
            "happy emotional mood, conversational style"
        )
        missing = self.parser.identify_missing_elements(detailed)
        # Should have fewer missing elements
        assert len(missing) < 5

    def test_identify_missing_elements_partial_prompt(self):
        """Test missing elements with partial information."""
        partial = "A photo with bright lighting and high resolution"
        missing = self.parser.identify_missing_elements(partial)
        # Still missing some elements
        assert len(missing) > 0
        assert "emotion/mood" in missing

    def test_extract_keywords_simple(self):
        """Test keyword extraction from simple prompt."""
        keywords = self.parser.extract_keywords("Create a photo of our mop product")
        assert len(keywords) > 0
        # Check that product or mop is in the keywords (convert generator to list)
        keywords_list = list(keywords)
        assert any(kw in " ".join(keywords_list) for kw in ["mop", "product"])

    def test_extract_keywords_empty(self):
        """Test keyword extraction from empty prompt."""
        keywords = self.parser.extract_keywords("")
        assert isinstance(keywords, list)


class TestInputParserEdgeCases:
    """Test InputParser edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = InputParser()

    def test_very_long_prompt(self):
        """Test handling of very long prompts."""
        long_prompt = "Create a photo " + "with many details " * 100
        category, intent = self.parser.parse(long_prompt)
        assert category in PromptCategory
        assert intent in PromptIntent

    def test_special_characters(self):
        """Test handling of special characters."""
        special_prompt = "Create a photo with @#$%^&*() special chars!"
        category, intent = self.parser.parse(special_prompt)
        assert category in PromptCategory
        assert intent in PromptIntent

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        unicode_prompt = "Create a photo with emojis 🎨📸 and 中文"
        category, intent = self.parser.parse(unicode_prompt)
        assert category in PromptCategory
        assert intent in PromptIntent

    def test_multiple_newlines(self):
        """Test handling of multiple newlines."""
        multiline = "Create a photo\n\n\nwith\n\n\nnewlines"
        category, intent = self.parser.parse(multiline)
        assert category in PromptCategory
        assert intent in PromptIntent
