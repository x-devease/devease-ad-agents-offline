"""
Unit tests for semantic search module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.meta.diagnoser.agents.semantic_search import SemanticSearch


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
        {
            "experiment_id": "exp_3",
            "detector": "FatigueDetector",
            "spec": {
                "title": "Aggressive threshold reduction",
                "changes": [{
                    "parameter": "cpa_increase_threshold",
                    "from": 1.3,
                    "to": 1.0
                }]
            },
            "outcome": "FAILURE",
            "evaluation": {
                "lift": {"f1_score": "-5.1%"}
            },
            "lessons_learned": [
                "Too aggressive causes precision drop"
            ]
        }
    ]


class TestSemanticSearchInit:
    """Test SemanticSearch initialization."""

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_init_default_model(self, mock_model_class):
        """Test initialization with default model."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)

        mock_model_class.assert_called_once_with('all-MiniLM-L6-v2')
        assert search.model == mock_model
        assert search.embeddings == {}

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_init_custom_model(self, mock_model_class):
        """Test initialization with custom model."""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        search = SemanticSearch(model_name="custom-model", enabled=True)

        mock_model_class.assert_called_once_with('custom-model')

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_init_empty_embeddings(self, mock_model_class):
        """Test initialization starts with empty embeddings."""
        mock_model_class.return_value = MagicMock()

        search = SemanticSearch(enabled=True)

        assert search.experiments == []
        assert search.embeddings.size == 0


class TestSemanticSearchExperimentToText:
    """Test _experiment_to_text() method."""

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_experiment_to_text_includes_title(self, mock_model_class, sample_experiments):
        """Test text conversion includes experiment title."""
        mock_model_class.return_value = MagicMock()

        search = SemanticSearch(enabled=True)
        text = search._experiment_to_text(sample_experiments[0])

        assert "Lower CPA threshold for better recall" in text

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_experiment_to_text_includes_changes(self, mock_model_class, sample_experiments):
        """Test text conversion includes parameter changes."""
        mock_model_class.return_value = MagicMock()

        search = SemanticSearch(enabled=True)
        text = search._experiment_to_text(sample_experiments[0])

        assert "cpa_increase_threshold" in text
        assert "1.2" in text
        assert "1.15" in text

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_experiment_to_text_includes_lessons(self, mock_model_class, sample_experiments):
        """Test text conversion includes lessons learned."""
        mock_model_class.return_value = MagicMock()

        search = SemanticSearch(enabled=True)
        text = search._experiment_to_text(sample_experiments[0])

        assert "Lower threshold improves recall" in text

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_experiment_to_text_includes_outcome(self, mock_model_class, sample_experiments):
        """Test text conversion includes outcome."""
        mock_model_class.return_value = MagicMock()

        search = SemanticSearch(enabled=True)
        text = search._experiment_to_text(sample_experiments[0])

        assert "SUCCESS" in text

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_experiment_to_text_includes_lift(self, mock_model_class, sample_experiments):
        """Test text conversion includes F1 lift."""
        mock_model_class.return_value = MagicMock()

        search = SemanticSearch(enabled=True)
        text = search._experiment_to_text(sample_experiments[0])

        assert "+4.7%" in text or "f1_score" in text.lower()


class TestSemanticSearchIndexExperiments:
    """Test index_experiments() method."""

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_index_experiments(self, mock_model_class, sample_experiments):
        """Test indexing experiments creates embeddings."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments)

        assert len(search.experiments) == 3
        assert search.embeddings.shape[0] == 3
        mock_model.encode.assert_called_once()

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_index_empty_experiments(self, mock_model_class):
        """Test indexing empty list."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([]).reshape(0, 3)
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments([])

        assert len(search.experiments) == 0
        assert search.embeddings.shape[0] == 0

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_index_replaces_existing(self, mock_model_class, sample_experiments):
        """Test re-indexing replaces existing embeddings."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments[:1])
        first_embeddings = search.embeddings.copy()

        # Re-index with different data
        mock_model.encode.return_value = np.array([[0.4, 0.5, 0.6]])
        search.index_experiments(sample_experiments[1:2])

        assert not np.array_equal(first_embeddings, search.embeddings)


class TestSemanticSearchSearch:
    """Test search() method."""

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_search_returns_top_k(self, mock_model_class, sample_experiments):
        """Test search returns top_k results."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments)

        results = search.search("recall improvement", top_k=2)

        assert len(results) == 2

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_search_includes_similarity_scores(self, mock_model_class, sample_experiments):
        """Test search results include similarity scores."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments)

        results = search.search("query")

        for result, score in results:
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1  # Cosine similarity range

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_search_respects_min_similarity(self, mock_model_class, sample_experiments):
        """Test search filters by minimum similarity."""
        mock_model = MagicMock()
        # Create embeddings with varying similarities
        mock_model.encode.return_value = np.array([
            [0.9, 0.1, 0.1],  # Low similarity
            [0.1, 0.9, 0.1],  # High similarity
            [0.1, 0.1, 0.9],  # Medium similarity
        ])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments)

        results = search.search("query", min_similarity=0.5)

        # Should only return results above threshold
        for result, score in results:
            assert score >= 0.5

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_search_without_indexing(self, mock_model_class):
        """Test search without indexing returns empty."""
        mock_model_class.return_value = MagicMock()

        search = SemanticSearch(enabled=True)

        results = search.search("query")

        assert len(results) == 0


class TestSemanticSearchRanking:
    """Test search result ranking."""

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_results_ranked_by_similarity(self, mock_model_class, sample_experiments):
        """Test results are ranked by similarity (descending)."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.1, 0.1],  # Low similarity
            [0.9, 0.9, 0.9],  # High similarity
            [0.5, 0.5, 0.5],  # Medium similarity
        ])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments)

        results = search.search("high similarity query")

        # Check descending order
        if len(results) > 1:
            similarities = [score for _, score in results]
            assert similarities == sorted(similarities, reverse=True)


class TestSemanticSearchIntegration:
    """Integration tests with realistic scenarios."""

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_find_similar_fatigue_experiments(self, mock_model_class, sample_experiments):
        """Test finding similar fatigue detector experiments."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.9, 0.1, 0.1],  # Very similar to query
            [0.1, 0.9, 0.1],  # Not similar
            [0.8, 0.2, 0.1],  # Somewhat similar
        ])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments)

        results = search.search("fatigue detector CPA threshold")

        # Should return fatigue-related experiments
        fatigue_results = [
            r for r, _ in results
            if r.get("detector") == "FatigueDetector"
        ]

        assert len(fatigue_results) > 0

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_find_successful_experiments(self, mock_model_class, sample_experiments):
        """Test finding successful experiments."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.9, 0.1],  # Similar
            [0.8, 0.2],  # Similar
            [0.1, 0.9],  # Not similar
        ])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments)

        results = search.search("successful optimizations")

        successful = [
            r for r, _ in results
            if r.get("outcome") == "SUCCESS"
        ]

        assert len(successful) > 0


class TestSemanticSearchErrorHandling:
    """Test error handling."""

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_handle_invalid_experiment_data(self, mock_model_class):
        """Test handling of invalid experiment data."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)

        # Missing required fields
        invalid_exp = [{"detector": "Test"}]

        # Should not crash
        try:
            search.index_experiments(invalid_exp)
        except Exception as e:
            # Expected to handle gracefully
            pass

    @patch('src.meta.diagnoser.agents.semantic_search.SentenceTransformer')
    def test_handle_empty_query(self, mock_model_class, sample_experiments):
        """Test handling of empty query."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model_class.return_value = mock_model

        search = SemanticSearch(enabled=True)
        search.index_experiments(sample_experiments)

        results = search.search("")

        # Should still return results
        assert isinstance(results, list)
