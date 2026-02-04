"""
Base Agent - Generic orchestration for prompt enhancement.

Orchestrates the enhancement pipeline using domain-specific adapters.
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Any

from src.agents.framework.adapters.base import BaseAdapter
from src.agents.framework.core.types import (
    FrameworkConfig,
    GroundingExample,
    MemoryEntry,
    QualityCheck,
    AgentInput,
    AgentOutput,
)
from src.agents.framework.core.examples import ExampleManager
from src.agents.framework.core.reflexion import ReflexionEngine
from src.agents.framework.core.memory import MemorySystem
from src.agents.framework.core.quality import QualityVerifier


logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Generic prompt enhancement agent.

    Orchestrates the enhancement pipeline using domain-specific adapters.
    Works for any domain (Nano, DALL-E, etc.) by swapping the adapter.
    """

    def __init__(self, adapter: BaseAdapter, config: FrameworkConfig = None):
        """
        Initialize agent with domain-specific adapter.

        Args:
            adapter: Domain adapter
            config: Framework configuration
        """
        self.adapter = adapter
        self.config = config or FrameworkConfig()

        # Initialize generic components
        self.example_manager = ExampleManager(
            examples_db_path=self.config.examples_db_path
        )
        self.reflexion_engine = ReflexionEngine(
            max_iterations=self.config.max_reflexion_iterations,
            quality_threshold=self.config.quality_threshold,
        )
        self.memory = MemorySystem(
            memory_db_path=self.config.memory_db_path,
            max_entries=self.config.memory_max_entries,
        )
        self.quality_verifier = QualityVerifier(threshold=self.config.quality_threshold)

        logger.info(f"BaseAgent initialized with {adapter.domain} adapter")

    def enhance(self, agent_input: AgentInput) -> AgentOutput:
        """
        Main enhancement pipeline (generic, works for all domains).

        Pipeline:
        1. Parse input (via adapter)
        2. Retrieve examples (generic)
        3. Enrich context (via adapter)
        4. Generate thinking (via adapter)
        5. Build prompt (via adapter)
        6. Apply techniques (via adapter)
        7. Reflexion loop (generic + adapter)
        8. Quality verify (generic)
        9. Store in memory (generic)

        Args:
            agent_input: User's input prompt

        Returns:
            Enhanced output with all metadata
        """
        start_time = time.time()

        logger.info("=" * 70)
        logger.info(f"ENHANCEMENT PIPELINE ({self.adapter.domain.upper()})")
        logger.info("=" * 70)

        # Step 1: Parse input (via adapter)
        logger.info("Step 1: Parsing input...")
        category, intent = self.adapter.parse_input(agent_input.generic_prompt)
        logger.info(f"  → Category: {category}, Intent: {intent}")

        # Step 2: Retrieve examples (generic)
        logger.info("Step 2: Retrieving examples...")
        examples = self.example_manager.retrieve_relevant(
            agent_input.generic_prompt,
            self.adapter.domain,
            k=3,
        )
        logger.info(f"  → Retrieved {len(examples)} examples")

        # Step 3: Enrich context (via adapter)
        logger.info("Step 3: Enriching context...")
        enriched_input = self.adapter.enrich_context(agent_input)
        logger.info("  → Context enriched")

        # Step 4: Generate thinking (via adapter)
        logger.info("Step 4: Generating thinking...")
        thinking = self.adapter.generate_thinking(
            enriched_input, category, intent, examples
        )
        logger.info(f"  → Thinking generated ({len(thinking)} chars)")

        # Step 5: Build prompt (via adapter)
        logger.info("Step 5: Building prompt...")
        base_prompt = self.adapter.build_prompt(enriched_input, category, intent)
        logger.info(f"  → Base prompt built ({len(base_prompt)} chars)")

        # Step 6: Apply techniques (via adapter)
        logger.info("Step 6: Applying techniques...")
        enhanced_prompt = self.adapter.apply_techniques(base_prompt, thinking)
        logger.info(f"  → Techniques applied ({len(enhanced_prompt)} chars)")

        # Step 7: Reflexion loop (generic + adapter)
        if self.config.enable_reflexion:
            logger.info("Step 7: Reflexion loop...")
            refined_prompt, critique_history = self.reflexion_engine.refine(
                enhanced_prompt, enriched_input, self.adapter
            )
            logger.info(f"  → Refined through {len(critique_history)} iterations")
            final_prompt = refined_prompt
        else:
            final_prompt = enhanced_prompt
            critique_history = []

        # Step 8: Quality verification
        logger.info("Step 8: Verifying quality...")
        quality_check = self.quality_verifier.verify(
            final_prompt, enriched_input, self.adapter, examples
        )
        logger.info(f"  → Quality: {quality_check.confidence:.2f} (passes: {quality_check.passes})")

        # Step 9: Store in memory (generic)
        if self.config.enable_memory and quality_check.confidence >= 0.7:
            logger.info("Step 9: Storing in memory...")
            # Convert enums to strings for JSON serialization
            category_str = str(category) if hasattr(category, 'value') else category
            intent_str = str(intent) if hasattr(intent, 'value') else intent

            memory_entry = MemoryEntry(
                input_prompt=agent_input.generic_prompt,
                enhanced_prompt=final_prompt,
                domain=self.adapter.domain,
                detected_category=category_str,
                detected_intent=intent_str,
                confidence=quality_check.confidence,
                techniques_used=self.adapter._extract_techniques_from_thinking(thinking)
                if hasattr(self.adapter, "_extract_techniques_from_thinking")
                else [],
            )
            self.memory.add_entry(memory_entry)
            logger.info(f"  → Stored in memory (entry_id: {memory_entry.entry_id})")

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Extract techniques from thinking
        techniques_used = (
            self.adapter._extract_techniques_from_thinking(thinking)
            if hasattr(self.adapter, "_extract_techniques_from_thinking")
            else []
        )

        # Create thinking block object if thinking is available (for backward compatibility)
        thinking_block_obj = None
        if thinking and self.adapter.domain == "nano":
            # Create a ThinkingBlock object from the thinking string
            from src.agents.nano.core.types import ThinkingBlock
            # Parse thinking string to extract components
            lines = thinking.split('\n')
            analysis = ""
            techniques = []
            risks = []
            mitigation = []
            context = ""
            current_section = None

            for line in lines:
                line = line.strip()
                if line.startswith("Analysis:"):
                    current_section = "analysis"
                    analysis = line.replace("Analysis:", "").strip()
                elif line.startswith("Techniques to apply:"):
                    current_section = "techniques"
                elif line.startswith("Risks:"):
                    current_section = "risks"
                elif line.startswith("Mitigation:"):
                    current_section = "mitigation"
                elif line.startswith("Context:"):
                    current_section = "context"
                    context = line.replace("Context:", "").strip()
                elif line.startswith("- ") and current_section:
                    item = line[2:].strip()
                    if current_section == "techniques":
                        techniques.append(item)
                    elif current_section == "risks":
                        risks.append(item)
                    elif current_section == "mitigation":
                        mitigation.append(item)
                elif current_section == "analysis" and line:
                    analysis += " " + line

            thinking_block_obj = ThinkingBlock(
                analysis=analysis.strip(),
                techniques=techniques,
                risks=risks,
                mitigation=mitigation,
                context=context.strip(),
            )
        elif thinking:
            # For non-nano domains, just store the thinking string
            thinking_block_obj = thinking

        # Create output
        output = AgentOutput(
            enhanced_prompt=final_prompt,
            detected_category=category,
            detected_intent=intent,
            confidence=quality_check.confidence,
            processing_time_ms=processing_time_ms,
            techniques_used=techniques_used,
            thinking_block=thinking_block_obj,
        )

        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETE ({processing_time_ms:.0f}ms)")
        logger.info("=" * 70)

        return output

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the framework components.

        Returns:
            Dictionary with statistics
        """
        return {
            "adapter": {
                "domain": self.adapter.domain,
            },
            "examples": self.example_manager.get_stats(),
            "memory": self.memory.get_stats(),
            "config": {
                "enable_reflexion": self.config.enable_reflexion,
                "enable_memory": self.config.enable_memory,
                "max_reflexion_iterations": self.config.max_reflexion_iterations,
                "quality_threshold": self.config.quality_threshold,
            },
        }
