"""
Unit tests for semantic search module.
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.meta.diagnoser.agents.semantic_search import SemanticSearch, SearchResult


@pytest.fixture
def sample_experiments():
    """Sample experiment records for testing."""
    return [
        {
            "experiment_id": "exp_1",
            "detector": "FatigueDetector",
            "spec": {
                "title": "Lower CPA threshold for better recall",
                "changes": [{
                    "parameter": "cpa_increase_threshold",
                    "from": 1.2,
                    "to": 1.15
                }]
            },
            "outcome": "SUCCESS",
            "evaluation": {
                "lift": {"f1_score": "+4.7%"}
            },
            "lessons_learned": [
                "Lower threshold improves recall",
                "Monitor FP growth"
            ]
        },
        {
            "experiment_id": "exp_2",
            "detector": "LatencyDetector",
            "spec": {
                "title": "Increase rolling window for stability",
                "changes": [{
                    "parameter": "rolling_window_days",
                    "from": 2,
                    "to": 3
                }]
            },
            "outcome": "SUCCESS",
            "evaluation": {
                "lift": {"f1_score": "+3.2%"}
            },
            "lessons_learned": [
                "Larger window improves stability"
            ]
        },
    ]


class TestSemanticSearchInit:
    """Test SemanticSearch initialization."""

    def test_init_disabled_by_default(self):
        """Test initialization is disabled by default."""
        search = SemanticSearch()

        assert search.enabled is False
        assert search.model is None
        assert search._initialized is False

    def test_init_enabled_flag(self):
        """Test initialization with enabled flag."""
        # Note: This won't actually initialize the model without sentence-transformers installed
        search = SemanticSearch(enabled=True)

        # The enabled flag should be set, but initialization may fail if sentence-transformers not installed
        assert search.enabled is True
        # If sentence-transformers is not installed, enabled will be set to False during _initialize_model

    def test_init_custom_model_name(self):
        """Test initialization with custom model name."""
        search = SemanticSearch(model_name="custom-model", enabled=False)

        assert search.model_name == "custom-model"
        assert search.enabled is False

    def test_init_empty_state(self):
        """Test initialization starts with empty state."""
        search = SemanticSearch(enabled=False)

        assert search.experiments == []
        assert isinstance(search.embeddings, dict)


class TestExperimentToText:
    """Test _experiment_to_text method."""

    def test_experiment_to_text_includes_title(self):
        """Test text conversion includes experiment title."""
        search = SemanticSearch(enabled=False)
        exp = {
            "title": "Test experiment title"
        }

        text = search._experiment_to_text(exp)

        assert "Test experiment title" in text

    def test_experiment_to_text_includes_changes(self):
        """Test text conversion includes parameter changes."""
        search = SemanticSearch(enabled=False)
        exp = {
            "changes": [{
                "parameter": "test_param",
                "reason": "test reason"
            }]
        }

        text = search._experiment_to_text(exp)

        assert "test_param" in text

    def test_experiment_to_text_handles_missing_fields(self):
        """Test text conversion handles missing optional fields."""
        search = SemanticSearch(enabled=False)
        exp = {
            "experiment_id": "exp_1"
        }

        # Should not crash
        text = search._experiment_to_text(exp)
        assert isinstance(text, str)


class TestIndexExperiments:
    """Test index_experiments method."""

    def test_index_when_disabled_returns_early(self, sample_experiments):
        """Test indexing does nothing when disabled."""
        search = SemanticSearch(enabled=False)

        search.index_experiments(sample_experiments)

        # Should remain empty since disabled
        assert search.experiments == []

    def test_index_sets_experiments(self):
        """Test indexing stores experiments."""
        search = SemanticSearch(enabled=False)
        # Manually set enabled to test the experiment storage logic
        search.enabled = True
        search._initialized = True
        # Create a mock model
        class MockModel:
            def encode(self, texts, **kwargs):
                import numpy as np
                return np.array([[0.1, 0.2]] * len(texts))

        search.model = MockModel()

        experiments = [{"experiment_id": "exp_1"}]
        search.index_experiments(experiments)

        assert len(search.experiments) == 1


class TestSearch:
    """Test search method."""

    def test_search_when_disabled_returns_empty(self):
        """Test search returns empty when disabled."""
        search = SemanticSearch(enabled=False)

        results = search.search("test query")

        assert results == []

    def test_search_respects_top_k(self):
        """Test search respects top_k parameter."""
        search = SemanticSearch(enabled=False)
        search.enabled = True
        search._initialized = True

        # Mock embeddings and experiments
        search.experiments = [
            {"experiment_id": "exp_1"},
            {"experiment_id": "exp_2"}
        ]
        import numpy as np
        search.embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Create a mock model
        class MockModel:
            def encode(self, texts, **kwargs):
                return np.array([[0.15, 0.25]])

        search.model = MockModel()

        results = search.search("test", top_k=1)

        # Should return at most 1 result
        assert len(results) <= 1


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test SearchResult can be created."""
        result = SearchResult(
            experiment={"id": "exp_1"},
            similarity=0.95,
            experiment_id="exp_1"
        )

        assert result.experiment == {"id": "exp_1"}
        assert result.similarity == 0.95
        assert result.experiment_id == "exp_1"


class TestSemanticSearchDisabled:
    """Test semantic search behavior when disabled (default)."""

    def test_all_operations_graceful_when_disabled(self, sample_experiments):
        """Test all operations are graceful when disabled."""
        search = SemanticSearch(enabled=False)

        # Should not crash
        search.index_experiments(sample_experiments)
        results = search.search("query")
        assert results == []

    def test_initialization_state(self):
        """Test proper initialization state when disabled."""
        search = SemanticSearch(enabled=False)

        assert search.enabled is False
        assert search._initialized is False
        assert search.model is None
        assert search.experiments == []
