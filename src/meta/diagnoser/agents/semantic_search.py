"""
Semantic Search for Experiment Memory.

Provides vector embedding-based semantic search for experiment specs.
Enable via agent_config.yaml → advanced_features.semantic_search_enabled = true
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from semantic search."""
    experiment: Dict[str, Any]
    similarity: float
    experiment_id: str


class SemanticSearch:
    """
    Semantic search for experiment specs using vector embeddings.

    This feature is experimental and disabled by default.
    Enable via: agent_config.yaml → advanced_features.semantic_search_enabled = true

    Uses sentence-transformers for creating embeddings and cosine similarity for search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        enabled: bool = False
    ):
        """
        Initialize semantic search.

        Args:
            model_name: Name of sentence-transformers model
            enabled: Whether semantic search is enabled
        """
        self.enabled = enabled
        self.model_name = model_name
        self.model = None
        self.embeddings = {}
        self.experiments = []
        self._initialized = False

        if enabled:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info(f"Semantic search initialized with model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.enabled = False

    def index_experiments(self, experiments: List[Dict[str, Any]]):
        """
        Build embedding index from experiment history.

        Args:
            experiments: List of experiment dictionaries to index

        Example:
            >>> search = SemanticSearch(enabled=True)
            >>> search.index_experiments(experiments)
        """
        if not self.enabled or not self._initialized:
            logger.warning("Semantic search not enabled or not initialized")
            return

        self.experiments = experiments

        # Convert experiments to text for embedding
        texts = [self._experiment_to_text(exp) for exp in experiments]

        # Generate embeddings
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        logger.info(f"Indexed {len(experiments)} experiments for semantic search")

    def _experiment_to_text(self, exp: Dict[str, Any]) -> str:
        """
        Convert experiment to searchable text.

        Args:
            exp: Experiment dictionary

        Returns:
            Text representation for embedding
        """
        parts = []

        # Title and scope
        title = exp.get("title", "")
        scope = exp.get("scope", "")
        if title:
            parts.append(title)
        if scope:
            parts.append(f"Scope: {scope}")

        # Changes
        changes = exp.get("changes", [])
        if changes:
            for change in changes[:3]:  # Limit to first 3 changes
                param = change.get("parameter", "")
                reason = change.get("reason", "")
                if param and reason:
                    parts.append(f"Modify {param}: {reason}")

        # Constraints
        constraints = exp.get("constraints", [])
        if constraints:
            parts.extend(constraints[:2])  # Limit to first 2 constraints

        # Detector
        detector = exp.get("detector", "")
        if detector:
            parts.append(f"Detector: {detector}")

        return " ".join(parts)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[SearchResult]:
        """
        Find semantically similar experiments.

        Args:
            query: Search query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of SearchResult objects, sorted by similarity

        Example:
            >>> results = search.search("improve fatigue recall", top_k=3)
            >>> for r in results:
            ...     print(f"{r.similarity:.2f}: {r.experiment['title']}")
        """
        if not self.enabled or not self._initialized:
            logger.warning("Semantic search not enabled or not initialized")
            return []

        if len(self.experiments) == 0:
            logger.warning("No experiments indexed")
            return []

        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]

        # Calculate cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Filter by minimum similarity and create results
        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= min_similarity:
                exp = self.experiments[idx]
                exp_id = exp.get("id", f"exp_{idx}")
                results.append(SearchResult(
                    experiment=exp,
                    similarity=float(sim),
                    experiment_id=exp_id
                ))

        logger.info(f"Semantic search for '{query}' found {len(results)} results")
        return results

    def _cosine_similarity(
        self,
        vec1,
        vec2
    ) -> List[float]:
        """
        Calculate cosine similarity between vectors.

        Args:
            vec1: Query vector
            vec2: Matrix of document vectors

        Returns:
            List of similarity scores
        """
        import numpy as np

        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1) + 1e-8)[:, np.newaxis]

        # Calculate dot product
        similarities = np.dot(vec2_norm, vec1_norm)

        return similarities

    def find_similar_failed_experiments(
        self,
        experiment: Dict[str, Any],
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        Find similar experiments that failed.

        Args:
            experiment: Query experiment
            top_k: Number of results

        Returns:
            List of similar failed experiments
        """
        # Filter for failed experiments only
        failed_exps = [
            exp for exp in self.experiments
            if exp.get("outcome") == "REJECTED"
        ]

        if not failed_exps:
            return []

        # Create temporary index with only failed experiments
        original_experiments = self.experiments.copy()
        original_embeddings = self.embeddings.copy()

        self.experiments = failed_exps
        self.index_experiments(failed_exps)

        # Search
        query_text = self._experiment_to_text(experiment)
        results = self.search(query_text, top_k=top_k)

        # Restore original index
        self.experiments = original_experiments
        self.embeddings = original_embeddings

        return results


class MockSemanticSearch(SemanticSearch):
    """
    Mock implementation for testing without sentence-transformers.

    Uses keyword matching instead of embeddings.
    """

    def _initialize_model(self):
        """Initialize mock search (no model needed)."""
        self._initialized = True
        logger.info("Mock semantic search initialized (keyword-based)")

    def index_experiments(self, experiments: List[Dict[str, Any]]):
        """Index experiments for keyword search."""
        self.experiments = experiments
        logger.info(f"Mock indexed {len(experiments)} experiments")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Keyword-based search.

        Returns experiments matching query keywords.
        """
        query_lower = query.lower()
        keywords = set(query_lower.split())

        results = []
        for exp in self.experiments:
            exp_text = self._experiment_to_text(exp).lower()

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in exp_text)
            if matches > 0:
                # Simple similarity score based on keyword matches
                similarity = min(matches / len(keywords), 1.0)
                if similarity >= min_similarity:
                    exp_id = exp.get("id", "unknown")
                    results.append(SearchResult(
                        experiment=exp,
                        similarity=similarity,
                        experiment_id=exp_id
                    ))

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]
