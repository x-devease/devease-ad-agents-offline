"""
Unit tests for UIAgent.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from io import StringIO

from src.growth.twitter.agents.ui_agent import UIAgent, UserSelection
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


@pytest.fixture
def mock_drafts():
    """Mock list of TwitterDraft."""
    return [
        TwitterDraft(
            content="Draft 1 content",
            rationale="Rationale 1",
            version="spicy",
            tone="provocative"
        ),
        TwitterDraft(
            content="Draft 2 content",
            rationale="Rationale 2",
            version="hardcore",
            tone="technical"
        ),
        TwitterDraft(
            content="Draft 3 content",
            rationale="Rationale 3",
            version="observation",
            tone="thoughtful"
        )
    ]


class TestUIAgentInit:
    """Test UIAgent initialization."""

    def test_init(self, mock_keys, mock_config):
        """Test initialization."""
        agent = UIAgent(mock_keys, mock_config)

        assert agent.keys == mock_keys
        assert agent.config == mock_config


class TestConfirmSelection:
    """Test confirmation dialog."""

    @patch('builtins.input', return_value='yes')
    def test_confirm_selection_yes(self, mock_input, mock_keys, mock_config):
        """Test confirmation with 'yes'."""
        agent = UIAgent(mock_keys, mock_config)
        draft = TwitterDraft(content="Test content", rationale="Test")

        result = agent._confirm_selection(draft)

        assert result is True

    @patch('builtins.input', return_value='no')
    def test_confirm_selection_no(self, mock_input, mock_keys, mock_config):
        """Test confirmation with 'no'."""
        agent = UIAgent(mock_keys, mock_config)
        draft = TwitterDraft(content="Test content", rationale="Test")

        result = agent._confirm_selection(draft)

        assert result is False


class TestCollectPostFeedback:
    """Test post-feedback collection."""

    @patch('builtins.input', return_value='Great tweet!')
    def test_collect_feedback_with_input(self, mock_input, mock_keys, mock_config):
        """Test collecting feedback with user input."""
        agent = UIAgent(mock_keys, mock_config)
        task = Mock()
        draft = Mock()

        result = agent.collect_post_feedback(task, draft)

        assert result == "Great tweet!"

    @patch('builtins.input', return_value='')
    def test_collect_feedback_empty(self, mock_input, mock_keys, mock_config):
        """Test collecting feedback with empty input."""
        agent = UIAgent(mock_keys, mock_config)
        task = Mock()
        draft = Mock()

        result = agent.collect_post_feedback(task, draft)

        assert result is None


class TestHandleError:
    """Test error handling."""

    @patch('builtins.input', return_value='1')
    def test_handle_error_retry(self, mock_input, mock_keys, mock_config, mock_task):
        """Test error handling with retry."""
        agent = UIAgent(mock_keys, mock_config)
        error = Exception("Test error")

        result = agent.handle_error(error, mock_task)

        assert result == 'retry'

    @patch('builtins.input', return_value='2')
    def test_handle_error_skip(self, mock_input, mock_keys, mock_config, mock_task):
        """Test error handling with skip."""
        agent = UIAgent(mock_keys, mock_config)
        error = Exception("Test error")

        result = agent.handle_error(error, mock_task)

        assert result == 'skip'

    @patch('builtins.input', return_value='3')
    def test_handle_error_abort(self, mock_input, mock_keys, mock_config, mock_task):
        """Test error handling with abort."""
        agent = UIAgent(mock_keys, mock_config)
        error = Exception("Test error")

        result = agent.handle_error(error, mock_task)

        assert result == 'abort'


class TestConfirmBatchStart:
    """Test batch start confirmation."""

    @patch('builtins.input', return_value='yes')
    def test_confirm_batch_start(self, mock_input, mock_keys, mock_config, mock_task):
        """Test batch start confirmation."""
        agent = UIAgent(mock_keys, mock_config)
        tasks = [mock_task]

        result = agent.confirm_batch_start(tasks)

        assert result is True
