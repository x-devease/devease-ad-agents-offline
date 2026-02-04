"""
Example Manager for few-shot learning.

Manages grounding examples for prompt enhancement.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from src.agents.framework.core.types import GroundingExample


logger = logging.getLogger(__name__)


class ExampleManager:
    """
    Manages grounding examples for prompt enhancement.

    Examples provide few-shot grounding to show the model what good outputs look like.
    """

    def __init__(self, examples_db_path: str = "data/agents/examples.json"):
        """
        Initialize ExampleManager.

        Args:
            examples_db_path: Path to examples JSON database
        """
        self.examples_db_path = examples_db_path
        self.examples: List[GroundingExample] = []
        self.domain_index: Dict[str, List[str]] = {}  # {domain: [example_ids]}
        self.category_index: Dict[str, List[str]] = {}  # {category: [example_ids]}

        self._load_examples()

    def _load_examples(self):
        """Load examples from database."""
        examples_path = Path(self.examples_db_path)

        if not examples_path.exists():
            logger.info(f"Examples database not found at {self.examples_db_path}")
            logger.info("Creating empty examples database")
            self._save_examples()
            return

        try:
            with open(examples_path, "r") as f:
                data = json.load(f)

            self.examples = [
                GroundingExample(**example_data) for example_data in data.get("examples", [])
            ]

            self._rebuild_indexes()

            logger.info(f"Loaded {len(self.examples)} examples from {self.examples_db_path}")

        except Exception as e:
            logger.error(f"Failed to load examples: {e}")
            self.examples = []

    def _save_examples(self):
        """Save examples to database."""
        examples_path = Path(self.examples_db_path)
        examples_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "examples": [
                {
                    "input_prompt": ex.input_prompt,
                    "output_prompt": ex.output_prompt,
                    "domain": ex.domain,
                    "category": ex.category,
                    "intent": ex.intent,
                    "metadata": ex.metadata,
                    "created_at": ex.created_at,
                    "example_id": ex.example_id,
                }
                for ex in self.examples
            ]
        }

        with open(examples_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self.examples)} examples to {self.examples_db_path}")

    def _rebuild_indexes(self):
        """Rebuild domain and category indexes."""
        self.domain_index = {}
        self.category_index = {}

        for example in self.examples:
            # Domain index
            if example.domain not in self.domain_index:
                self.domain_index[example.domain] = []
            self.domain_index[example.domain].append(example.example_id)

            # Category index
            if example.category not in self.category_index:
                self.category_index[example.category] = []
            self.category_index[example.category].append(example.example_id)

    def add_example(self, example: GroundingExample):
        """
        Add a new example to the database.

        Args:
            example: Example to add
        """
        self.examples.append(example)

        # Update indexes
        if example.domain not in self.domain_index:
            self.domain_index[example.domain] = []
        self.domain_index[example.domain].append(example.example_id)

        if example.category not in self.category_index:
            self.category_index[example.category] = []
        self.category_index[example.category].append(example.example_id)

        self._save_examples()
        logger.info(f"Added example {example.example_id} for domain {example.domain}")

    def retrieve_relevant(
        self, input_prompt: str, domain: str, k: int = 3
    ) -> List[GroundingExample]:
        """
        Retrieve top-k most relevant examples for a given input and domain.

        Args:
            input_prompt: User's input prompt
            domain: Domain identifier
            k: Number of examples to retrieve (default: 3, optimal per Google research)

        Returns:
            List of relevant examples sorted by relevance
        """
        # Filter by domain
        domain_example_ids = self.domain_index.get(domain, [])
        domain_examples = [
            ex for ex in self.examples if ex.example_id in domain_example_ids
        ]

        if not domain_examples:
            logger.warning(f"No examples found for domain: {domain}")
            return []

        # Calculate similarity scores
        scored_examples = []
        for example in domain_examples:
            score = self._compute_similarity(input_prompt, example.input_prompt)
            scored_examples.append((example, score))

        # Sort by score (descending) and take top-k
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        top_examples = [ex for ex, score in scored_examples[:k]]

        logger.debug(
            f"Retrieved {len(top_examples)} examples for domain {domain} (k={k})"
        )

        return top_examples

    def _compute_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Compute similarity between two prompts.

        Uses keyword overlap as a simple similarity metric.
        Can be overridden in subclasses for more sophisticated similarity.

        Args:
            prompt1: First prompt
            prompt2: Second prompt

        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the examples database.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_examples": len(self.examples),
            "domains": list(self.domain_index.keys()),
            "examples_per_domain": {
                domain: len(ids) for domain, ids in self.domain_index.items()
            },
            "categories": list(self.category_index.keys()),
            "examples_per_category": {
                category: len(ids) for category, ids in self.category_index.items()
            },
        }
