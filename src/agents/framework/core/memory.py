"""
Memory System for learning from past interactions.

Enables the framework to learn and improve over time.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

from src.agents.framework.core.types import MemoryEntry
from src.agents.framework.adapters.base import BaseAdapter


logger = logging.getLogger(__name__)


class MemorySystem:
    """
    Memory system for storing and retrieving past prompt enhancements.

    Memory enables the framework to:
    - Learn from past successful enhancements
    - Avoid repeating mistakes
    - Improve over time with usage

    Research shows memory systems provide +10-20% quality improvement.
    """

    def __init__(self, memory_db_path: str = "data/agents/memory.json", max_entries: int = 1000):
        """
        Initialize MemorySystem.

        Args:
            memory_db_path: Path to memory JSON database
            max_entries: Maximum number of entries to keep (LRU eviction)
        """
        self.memory_db_path = memory_db_path
        self.max_entries = max_entries
        self.memories: List[MemoryEntry] = []
        self.domain_index: Dict[str, List[str]] = {}

        self._load_memories()

    def _load_memories(self):
        """Load memories from database."""
        memory_path = Path(self.memory_db_path)

        if not memory_path.exists():
            logger.info(f"Memory database not found at {self.memory_db_path}")
            logger.info("Creating empty memory database")
            self._save_memories()
            return

        try:
            with open(memory_path, "r") as f:
                data = json.load(f)

            self.memories = [
                MemoryEntry(**entry_data) for entry_data in data.get("memories", [])
            ]

            self._rebuild_indexes()

            logger.info(f"Loaded {len(self.memories)} memories from {self.memory_db_path}")

        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            self.memories = []

    def _save_memories(self):
        """Save memories to database."""
        memory_path = Path(self.memory_db_path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "memories": [
                {
                    "input_prompt": mem.input_prompt,
                    "enhanced_prompt": mem.enhanced_prompt,
                    "domain": mem.domain,
                    "detected_category": mem.detected_category,
                    "detected_intent": mem.detected_intent,
                    "confidence": mem.confidence,
                    "techniques_used": mem.techniques_used,
                    "user_feedback": mem.user_feedback,
                    "created_at": mem.created_at,
                    "entry_id": mem.entry_id,
                }
                for mem in self.memories
            ]
        }

        with open(memory_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self.memories)} memories to {self.memory_db_path}")

    def _rebuild_indexes(self):
        """Rebuild domain index."""
        self.domain_index = {}

        for memory in self.memories:
            if memory.domain not in self.domain_index:
                self.domain_index[memory.domain] = []
            self.domain_index[memory.domain].append(memory.entry_id)

    def add_entry(self, entry: MemoryEntry):
        """
        Add a new memory entry.

        Args:
            entry: Memory entry to add
        """
        self.memories.append(entry)

        # Update index
        if entry.domain not in self.domain_index:
            self.domain_index[entry.domain] = []
        self.domain_index[entry.domain].append(entry.entry_id)

        # Enforce max entries (LRU eviction)
        if len(self.memories) > self.max_entries:
            # Remove oldest entries
            excess = len(self.memories) - self.max_entries
            removed = self.memories[:excess]

            for entry in removed:
                # Remove from index
                if entry.domain in self.domain_index:
                    self.domain_index[entry.domain].remove(entry.entry_id)

            # Keep only recent entries
            self.memories = self.memories[excess:]

            logger.debug(f"Evicted {excess} old memory entries (LRU)")

        self._save_memories()
        logger.info(f"Added memory entry {entry.entry_id} for domain {entry.domain}")

    def find_similar(
        self, input_prompt: str, domain: str, adapter: BaseAdapter, k: int = 3
    ) -> List[MemoryEntry]:
        """
        Find similar past enhancements.

        Args:
            input_prompt: Current input prompt
            domain: Domain identifier
            adapter: Domain adapter for similarity computation
            k: Number of similar entries to retrieve

        Returns:
            List of similar memory entries sorted by similarity
        """
        # Filter by domain and high confidence
        domain_memory_ids = self.domain_index.get(domain, [])
        domain_memories = [
            mem
            for mem in self.memories
            if mem.entry_id in domain_memory_ids and mem.confidence >= 0.7
        ]

        if not domain_memories:
            logger.debug(f"No similar memories found for domain {domain}")
            return []

        # Calculate similarity scores using adapter
        scored_memories = []
        for memory in domain_memories:
            score = adapter.compute_similarity(input_prompt, memory.input_prompt)
            scored_memories.append((memory, score))

        # Sort by score (descending) and take top-k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [mem for mem, score in scored_memories[:k]]

        logger.debug(
            f"Retrieved {len(top_memories)} similar memories for domain {domain} (k={k})"
        )

        return top_memories

    def add_feedback(self, entry_id: str, feedback: str):
        """
        Add user feedback to a memory entry.

        Args:
            entry_id: ID of memory entry
            feedback: User feedback string
        """
        for memory in self.memories:
            if memory.entry_id == entry_id:
                memory.user_feedback = feedback
                self._save_memories()
                logger.info(f"Added feedback for memory entry {entry_id}")
                return

        logger.warning(f"Memory entry {entry_id} not found")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.

        Returns:
            Dictionary with statistics
        """
        # Calculate average confidence by domain
        domain_stats = {}
        for domain, entry_ids in self.domain_index.items():
            domain_memories = [
                mem for mem in self.memories if mem.entry_id in entry_ids
            ]
            if domain_memories:
                avg_confidence = sum(mem.confidence for mem in domain_memories) / len(
                    domain_memories
                )
                domain_stats[domain] = {
                    "count": len(domain_memories),
                    "avg_confidence": avg_confidence,
                }

        # Count entries with feedback
        with_feedback = sum(1 for mem in self.memories if mem.user_feedback)

        # Recent activity (last 7 days)
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_entries = [
            mem
            for mem in self.memories
            if datetime.fromisoformat(mem.created_at) > seven_days_ago
        ]

        return {
            "total_entries": len(self.memories),
            "max_entries": self.max_entries,
            "entries_with_feedback": with_feedback,
            "recent_entries_7days": len(recent_entries),
            "domains": list(self.domain_index.keys()),
            "domain_stats": domain_stats,
        }

    def clear_domain(self, domain: str):
        """
        Clear all memories for a specific domain.

        Args:
            domain: Domain identifier
        """
        if domain not in self.domain_index:
            logger.warning(f"No memories found for domain: {domain}")
            return

        entry_ids = self.domain_index[domain]
        self.memories = [mem for mem in self.memories if mem.entry_id not in entry_ids]
        del self.domain_index[domain]

        self._save_memories()
        logger.info(f"Cleared {len(entry_ids)} memories for domain {domain}")

    def clear_all(self):
        """Clear all memories."""
        self.memories = []
        self.domain_index = {}
        self._save_memories()
        logger.info("Cleared all memories")
