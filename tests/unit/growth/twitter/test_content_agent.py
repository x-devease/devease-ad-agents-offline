"""
Unit tests for ContentAgent.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.growth.twitter.agents.content_agent import ContentAgent
from src.growth.twitter.core.types import TwitterTask, TwitterDraft, TwitterKeys, TwitterConfig, TaskType


@pytest.fixture
def mock_keys():
    """Mock TwitterKeys."""
    return TwitterKeys(
        openai_api_key="test_key",
        openai_org_id="test_org"
    )


@pytest.fixture
def mock_config():
    """Mock TwitterConfig."""
    return TwitterConfig(
        llm_model="gpt-4",
        tasks_path=Path("tasks.yaml")
    )


@pytest.fixture
def mock_task():
    """Mock TwitterTask."""
    return TwitterTask(
        id="test_001",
        type=TaskType.POST,
        idea="Test idea about ad optimization",
        style="professional"
    )


class TestContentAgentInit:
    """Test ContentAgent initialization."""

    @patch('openai.OpenAI')
    def test_init_default(self, mock_openai, mock_keys, mock_config):
        """Test initialization with defaults."""
        agent = ContentAgent(mock_keys, mock_config)

        assert agent.keys == mock_keys
        assert agent.config == mock_config
        assert agent.memory is None
        assert agent.client is not None

    @patch('openai.OpenAI')
    def test_init_with_memory(self, mock_openai, mock_keys, mock_config):
        """Test initialization with memory system."""
        mock_memory = Mock()
        agent = ContentAgent(mock_keys, mock_config, memory=mock_memory)

        assert agent.memory == mock_memory


class TestGenerateDrafts:
    """Test draft generation."""

    @patch('openai.OpenAI')
    def test_generate_drafts_fallback(self, mock_openai_class, mock_keys, mock_config, mock_task):
        """Test fallback drafts on LLM failure."""
        mock_client = mock_openai_class.return_value
        mock_client.chat.completions.create.side_effect = Exception("LLM error")

        agent = ContentAgent(mock_keys, mock_config)
        drafts = agent.generate_drafts(mock_task)

        assert len(drafts) == 3
        assert all("Fallback" in d.rationale for d in drafts)
        assert all(d.version.startswith("fallback_") for d in drafts)


class TestRegenerateDraft:
    """Test draft regeneration."""

    @patch('openai.OpenAI')
    def test_regenerate_draft(self, mock_openai_class, mock_keys, mock_config, mock_task):
        """Test regenerating a single draft."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Regenerated content"

        mock_client = mock_openai_class.return_value
        mock_client.chat.completions.create.return_value = mock_response

        agent = ContentAgent(mock_keys, mock_config)
        draft = agent.regenerate_draft(mock_task, 0)

        assert draft.content == "Regenerated content"
        assert draft.version == "regenerated_0"


class TestRecordFeedback:
    """Test feedback recording."""

    @patch('openai.OpenAI')
    def test_record_feedback_with_memory(self, mock_openai, mock_keys, mock_config):
        """Test recording feedback with memory system."""
        mock_memory = Mock()
        agent = ContentAgent(mock_keys, mock_config, memory=mock_memory)

        draft = TwitterDraft(
            content="Test content",
            rationale="Test rationale"
        )

        agent.record_feedback(draft, 'confirmed', 'Great tweet!')

        mock_memory.record_feedback.assert_called_once_with(
            draft, 'confirmed', 'Great tweet!'
        )

    @patch('openai.OpenAI')
    def test_record_feedback_without_memory(self, mock_openai, mock_keys, mock_config):
        """Test recording feedback without memory system."""
        agent = ContentAgent(mock_keys, mock_config)

        draft = TwitterDraft(
            content="Test content",
            rationale="Test rationale"
        )

        # Should not raise error
        agent.record_feedback(draft, 'confirmed')
