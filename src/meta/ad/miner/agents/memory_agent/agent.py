"""
Memory Agent - Organizational Memory & Knowledge Base

Objective: Prevent organizational amnesia and accelerate evolution
through historical experience.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """Complete record of an experiment."""
    id: str
    timestamp: str
    spec: Dict[str, Any]  # PM Agent's spec
    code_changes: Dict[str, Any]  # Coder Agent's changes
    review_decision: Dict[str, Any]  # Reviewer Agent's decision
    results: Dict[str, Any]  # Judge Agent's evaluation
    lessons_learned: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # Vector embedding


class MemoryAgent:
    """
    Organizational Memory & Knowledge Base Agent.

    Records and retrieves every experiment's input-process-output,
    indexes successful and failed patterns, and warns about repeating failures.
    """

    def __init__(
        self,
        vector_store_client=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Memory Agent.

        Args:
            vector_store_client: Vector database client (e.g., ChromaDB, Pinecone)
            config: Agent configuration
        """
        self.vector_store = vector_store_client
        self.config = config or {}

        # In-memory storage (replace with vector DB in production)
        self.experiments: Dict[str, ExperimentRecord] = {}
        self.embeddings: Dict[str, List[float]] = {}

        # Semantic search cache
        self.search_cache: Dict[str, List[Dict]] = {}

        logger.info("Memory Agent initialized")

    def store_experiment(
        self,
        experiment_id: str,
        spec: Dict[str, Any],
        code_changes: Dict[str, Any],
        review_decision: Dict[str, Any],
        results: Dict[str, Any],
    ) -> ExperimentRecord:
        """
        Archive a complete experiment record.

        Args:
            experiment_id: Unique experiment identifier
            spec: PM Agent's experiment specification
            code_changes: Coder Agent's code modifications
            review_decision: Reviewer Agent's approval/rejection
            results: Judge Agent's evaluation results

        Returns:
            ExperimentRecord: Archived experiment record
        """
        timestamp = datetime.now().isoformat()

        # Extract lessons learned
        lessons_learned = self._extract_lessons(
            spec,
            code_changes,
            review_decision,
            results,
        )

        # Generate tags for indexing
        tags = self._generate_tags(
            spec,
            results,
        )

        # Create embedding for semantic search
        embedding = self._create_embedding(
            spec,
            code_changes,
            results,
        )

        record = ExperimentRecord(
            id=experiment_id,
            timestamp=timestamp,
            spec=spec,
            code_changes=code_changes,
            review_decision=review_decision,
            results=results,
            lessons_learned=lessons_learned,
            tags=tags,
            embedding=embedding,
        )

        # Store in memory
        self.experiments[experiment_id] = record
        if embedding:
            self.embeddings[experiment_id] = embedding

        logger.info(
            f"Stored experiment {experiment_id}: "
            f"{len(lessons_learned)} lessons, {len(tags)} tags"
        )

        return record

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar experiments.

        Args:
            query: Search query (objective, domain, etc.)
            top_k: Number of results to return
            filters: Optional filters (e.g., {"judge_decision": "PASS"})

        Returns:
            List of similar experiments with metadata
        """
        cache_key = f"{query}_{top_k}_{str(filters)}"

        # Check cache
        if cache_key in self.search_cache:
            logger.debug(f"Cache hit for query: {query}")
            return self.search_cache[cache_key]

        # Create query embedding
        query_embedding = self._create_embedding_from_text(query)

        # Semantic search using cosine similarity
        results = []
        for exp_id, record in self.experiments.items():
            # Apply filters
            if filters:
                if not self._matches_filters(record, filters):
                    continue

            # Calculate similarity
            if record.embedding and query_embedding:
                similarity = self._cosine_similarity(
                    query_embedding,
                    record.embedding,
                )
                results.append({
                    "experiment_id": exp_id,
                    "similarity": similarity,
                    "spec": record.spec,
                    "results": record.results,
                    "lessons_learned": record.lessons_learned,
                    "tags": record.tags,
                })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Return top_k
        top_results = results[:top_k]

        # Cache results
        self.search_cache[cache_key] = top_results

        logger.info(
            f"Search for '{query}' returned {len(top_results)} results "
            f"(from {len(self.experiments)} total experiments)"
        )

        return top_results

    def check_failure_pattern(
        self,
        current_spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect if current experiment matches historical failure patterns.

        Args:
            current_spec: Current experiment specification

        Returns:
            List of warnings about matching failure patterns
        """
        warnings = []

        # Extract key features from current spec
        current_objective = current_spec.get("objective", "")
        current_approach = current_spec.get("approach", "")
        current_domain = current_spec.get("domain", "")

        # Check against failed experiments
        for exp_id, record in self.experiments.items():
            if record.results.get("judge_decision") != "FAIL":
                continue  # Only check failures

            # Check for similar objectives
            spec_objective = record.spec.get("objective", "")
            if current_objective.lower() in spec_objective.lower():
                warnings.append({
                    "type": "similar_objective_failure",
                    "experiment_id": exp_id,
                    "failure_reason": record.results.get("failure_reason", "Unknown"),
                    "lessons": record.lessons_learned,
                    "timestamp": record.timestamp,
                })

            # Check for similar approaches
            spec_approach = record.spec.get("approach", "")
            if current_approach and current_approach.lower() in spec_approach.lower():
                warnings.append({
                    "type": "similar_approach_failure",
                    "experiment_id": exp_id,
                    "failure_reason": record.results.get("failure_reason", "Unknown"),
                    "lessons": record.lessons_learned,
                    "timestamp": record.timestamp,
                })

            # Check for domain-specific failures
            if current_domain:
                spec_domain = record.spec.get("domain", "")
                if current_domain.lower() == spec_domain.lower():
                    warnings.append({
                        "type": "domain_specific_failure",
                        "experiment_id": exp_id,
                        "failure_reason": record.results.get("failure_reason", "Unknown"),
                        "lessons": record.lessons_learned,
                        "timestamp": record.timestamp,
                    })

        # Group by failure type
        if warnings:
            logger.warning(
                f"Detected {len(warnings)} matching failure patterns "
                f"for current spec"
            )

        return warnings

    def get_successful_patterns(
        self,
        domain: Optional[str] = None,
        objective: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve winning approaches from successful experiments.

        Args:
            domain: Filter by domain
            objective: Filter by objective
            top_k: Maximum number of patterns to return

        Returns:
            List of successful patterns with approaches
        """
        successful = []

        for exp_id, record in self.experiments.items():
            # Filter by success
            if record.results.get("judge_decision") != "PASS":
                continue

            # Filter by domain
            if domain:
                spec_domain = record.spec.get("domain", "")
                if domain.lower() != spec_domain.lower():
                    continue

            # Filter by objective
            if objective:
                spec_objective = record.spec.get("objective", "")
                if objective.lower() not in spec_objective.lower():
                    continue

            # Extract successful pattern
            pattern = {
                "experiment_id": exp_id,
                "approach": record.spec.get("approach", ""),
                "lift_score": record.results.get("lift_score", 0),
                "parameters": record.spec.get("parameters", {}),
                "lessons_learned": record.lessons_learned,
                "timestamp": record.timestamp,
            }

            successful.append(pattern)

        # Sort by lift score
        successful.sort(key=lambda x: x["lift_score"], reverse=True)

        return successful[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total = len(self.experiments)
        successful = sum(
            1 for r in self.experiments.values()
            if r.results.get("judge_decision") == "PASS"
        )
        failed = total - successful

        # Tag distribution
        tag_counts = {}
        for record in self.experiments.values():
            for tag in record.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Objective distribution
        objective_counts = {}
        for record in self.experiments.values():
            objective = record.spec.get("objective", "unknown")
            objective_counts[objective] = objective_counts.get(objective, 0) + 1

        return {
            "total_experiments": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "objectives": objective_counts,
        }

    def _extract_lessons(
        self,
        spec: Dict[str, Any],
        code_changes: Dict[str, Any],
        review_decision: Dict[str, Any],
        results: Dict[str, Any],
    ) -> List[str]:
        """Extract lessons learned from experiment."""
        lessons = []

        # From results
        if results.get("judge_decision") == "PASS":
            lift_score = results.get("lift_score", 0)
            lessons.append(f"Achieved {lift_score}% lift with approach: {spec.get('approach', '')}")
        else:
            failure_reason = results.get("failure_reason", "")
            lessons.append(f"Failed due to: {failure_reason}")

        # From code changes
        if code_changes.get("files_modified"):
            lessons.append(f"Modified {len(code_changes['files_modified'])} files")

        # From review
        if review_decision.get("warnings"):
            lessons.append(f"Reviewer warnings: {len(review_decision['warnings'])}")

        return lessons

    def _generate_tags(
        self,
        spec: Dict[str, Any],
        results: Dict[str, Any],
    ) -> List[str]:
        """Generate tags for indexing."""
        tags = []

        # Objective tags
        objective = spec.get("objective", "")
        if "psychology" in objective.lower():
            tags.append("psychology")
        if "pattern" in objective.lower():
            tags.append("pattern_mining")
        if "feature" in objective.lower():
            tags.append("feature_extraction")
        if "performance" in objective.lower():
            tags.append("optimization")

        # Domain tags
        domain = spec.get("domain", "")
        if domain:
            tags.append(f"domain:{domain}")

        # Result tags
        if results.get("judge_decision") == "PASS":
            tags.append("successful")
        else:
            tags.append("failed")

        lift_score = results.get("lift_score", 0)
        if lift_score > 20:
            tags.append("high_impact")
        elif lift_score > 10:
            tags.append("medium_impact")
        elif lift_score > 0:
            tags.append("low_impact")

        return tags

    def _create_embedding(
        self,
        spec: Dict[str, Any],
        code_changes: Dict[str, Any],
        results: Dict[str, Any],
    ) -> List[float]:
        """Create vector embedding for semantic search."""
        # Combine key text fields
        text = f"{spec.get('objective', '')} {spec.get('approach', '')} {spec.get('domain', '')}"

        # Simple hash-based embedding (replace with real embeddings in production)
        hash_obj = hashlib.sha256(text.encode())
        hash_hex = hash_obj.hexdigest()

        # Convert to float vector (256 dimensions)
        embedding = []
        for i in range(0, len(hash_hex), 2):
            byte_val = int(hash_hex[i:i+2], 16)
            normalized = byte_val / 255.0
            embedding.append(normalized)

        return embedding

    def _create_embedding_from_text(self, text: str) -> List[float]:
        """Create embedding from query text."""
        hash_obj = hashlib.sha256(text.encode())
        hash_hex = hash_obj.hexdigest()

        embedding = []
        for i in range(0, len(hash_hex), 2):
            byte_val = int(hash_hex[i:i+2], 16)
            normalized = byte_val / 255.0
            embedding.append(normalized)

        return embedding

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    def _matches_filters(
        self,
        record: ExperimentRecord,
        filters: Dict[str, Any],
    ) -> bool:
        """Check if record matches filters."""
        for key, value in filters.items():
            # Check spec
            if key in record.spec:
                if record.spec[key] != value:
                    return False

            # Check results
            if key in record.results:
                if record.results[key] != value:
                    return False

            # Check tags
            if key == "tag":
                if value not in record.tags:
                    return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        stats = self.get_statistics()
        return {
            "total_experiments": stats["total_experiments"],
            "success_rate": stats["success_rate"],
            "cache_size": len(self.search_cache),
        }
