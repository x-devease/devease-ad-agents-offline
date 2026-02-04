"""
Prompt Enhancement Agent - Main Agent Class.

Transforms generic prompts into high-fidelity Nano Banana Pro prompts.
Now uses the generic framework with Nano adapter.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.agents.framework.core.types import AgentInput, AgentOutput, FrameworkConfig
from src.agents.nano.core.types import (
    PromptCategory,
    PromptIntent,
)

# Framework imports
from src.agents.framework.core.base_agent import BaseAgent
from src.agents.framework.adapters.nano import NanoAdapter
from src.agents.framework.adapters.base import AdapterConfig

# Legacy imports (for backward compatibility mode)
from src.agents.nano.parsers.input_parser import InputParser, IntentAnalyzer
from src.agents.nano.core.context_enrichment import ContextEnrichmentEngine
from src.agents.nano.core.thinking_engine import ThinkingEngine
from src.agents.nano.formatters.natural_language_builder import NaturalLanguagePromptBuilder
from src.agents.nano.techniques.orchestrator import TechniqueOrchestrator
from src.agents.nano.formatters.technical_specs import TechnicalSpecLayer
from src.agents.nano.formatters.guards import AntiHallucinationGuard
from src.agents.nano.core.quality_verifier import QualityVerifier
from src.agents.nano.formatters.output_formatter import OutputFormatter


logger = logging.getLogger(__name__)


class PromptEnhancementAgent:
    """
    Main agent that transforms generic prompts into high-fidelity NB Pro prompts.

    This is the primary interface for the prompt enhancement system.
    Now uses the generic framework with Nano adapter by default.

    Usage:
        agent = PromptEnhancementAgent()  # Uses framework by default

        input_prompt = AgentInput(
            generic_prompt="Create an ad for our mop",
        )

        output = agent.enhance(input_prompt)

        print(output.enhanced_prompt)  # Ready for Nano Banana Pro

    For legacy behavior:
        agent = PromptEnhancementAgent(use_framework=False)
    """

    def __init__(self, use_framework: bool = True, config_path: Optional[str] = None):
        """
        Initialize the prompt enhancement agent.

        Args:
            use_framework: If True, use the new framework (default: True)
            config_path: Optional path to Nano configuration
        """
        self.use_framework = use_framework

        if use_framework:
            # Use framework with Nano adapter
            logger.info("Initializing PromptEnhancementAgent with Framework")
            adapter_config = AdapterConfig(domain="nano", config_path=config_path)
            self.adapter = NanoAdapter(adapter_config)
            self.framework = BaseAgent(adapter=self.adapter)
            logger.info("PromptEnhancementAgent initialized with Framework")
        else:
            # Use legacy implementation
            logger.info("Initializing PromptEnhancementAgent with Legacy implementation")
            # Initialize all components
            self.input_parser = InputParser()
            self.intent_analyzer = IntentAnalyzer()
            self.context_enrichment = ContextEnrichmentEngine()
            self.thinking_engine = ThinkingEngine()
            self.nl_builder = NaturalLanguagePromptBuilder()
            self.technique_orchestrator = TechniqueOrchestrator()
            self.tech_specs_layer = TechnicalSpecLayer()
            self.anti_hallucination = AntiHallucinationGuard()
            self.quality_verifier = QualityVerifier()
            self.output_formatter = OutputFormatter()
            logger.info("PromptEnhancementAgent initialized with Legacy implementation")

    def enhance(self, agent_input: AgentInput) -> AgentOutput:
        """
        Enhance a generic prompt into a high-fidelity Nano Banana Pro prompt.

        This is the main entry point for the agent.

        Args:
            agent_input: The input containing generic prompt and optional context

        Returns:
            AgentOutput with enhanced prompt and metadata
        """
        if self.use_framework:
            # Use framework
            logger.info("Using Framework for enhancement")
            return self.framework.enhance(agent_input)
        else:
            # Use legacy implementation
            return self._enhance_legacy(agent_input)

    def _enhance_legacy(self, agent_input: AgentInput) -> AgentOutput:
        """
        Legacy enhancement implementation.

        Args:
            agent_input: The input containing generic prompt and optional context

        Returns:
            AgentOutput with enhanced prompt and metadata
        """

        start_time = time.time()

        logger.info(f"Enhancing prompt (legacy): '{agent_input.generic_prompt[:50]}...'")

        # Step 1: Parse input and detect intent
        logger.info("Step 1: Parsing input...")
        category, intent = self.input_parser.parse(agent_input.generic_prompt)

        # Step 2: Analyze request
        logger.info("Step 2: Analyzing request...")
        analysis = self.intent_analyzer.analyze_request_type(
            agent_input.generic_prompt, category
        )

        missing_elements = self.input_parser.identify_missing_elements(
            agent_input.generic_prompt
        )

        # Step 3: Enrich context
        logger.info("Step 3: Enriching context...")
        agent_input = self.context_enrichment.enrich(agent_input)

        # Step 4: Generate thinking
        logger.info("Step 4: Generating thinking...")
        thinking_block = self.thinking_engine.generate_thinking(
            agent_input,
            category,
            intent,
            missing_elements,
            analysis,
        )

        # Step 5: Build natural language prompt
        logger.info("Step 5: Building natural language prompt...")
        intermediate_prompt = self.nl_builder.build(
            agent_input,
            category,
            intent,
            analysis,
        )

        # Step 6: Apply techniques
        logger.info("Step 6: Applying NB techniques...")
        techniques_to_apply = thinking_block.techniques
        enhanced_prompt, applied_techniques = self.technique_orchestrator.apply_techniques(
            intermediate_prompt.prompt_content,
            techniques_to_apply,
            analysis,
        )

        # Update intermediate prompt
        intermediate_prompt.prompt_content = enhanced_prompt
        intermediate_prompt.techniques_applied = [t.technique_name for t in applied_techniques]

        # Step 7: Add technical specifications
        logger.info("Step 7: Adding technical specifications...")
        tech_specs = self.tech_specs_layer.generate_specs(
            intent,
            agent_input.preferred_resolution,
        )
        tech_specs_text = self.tech_specs_layer.format_specs(tech_specs)

        # Step 8: Add anti-hallucination guards
        logger.info("Step 8: Adding anti-hallucination guards...")
        constraints = self.anti_hallucination.generate_constraints(
            has_product_reference=agent_input.product_context is not None,
            has_person_reference=analysis.get("needs_human_element", False),
            intent=intent,
        )
        constraints_text = self.anti_hallucination.format_constraints(constraints)

        # Step 9: Verify quality
        logger.info("Step 9: Verifying quality...")
        passes, confidence, issues = self.quality_verifier.verify(
            enhanced_prompt,
            [t.technique_name for t in applied_techniques],
            has_thinking=agent_input.enable_thinking,
        )

        if not passes:
            logger.warning(f"Quality issues detected: {issues}")

        # Step 10: Format output
        logger.info("Step 10: Formatting output...")
        processing_time_ms = int((time.time() - start_time) * 1000)

        output = self.output_formatter.format(
            thinking_block=thinking_block,
            intermediate_prompt=intermediate_prompt,
            techniques_applied=applied_techniques,
            technical_specs_text=tech_specs_text,
            constraints_text=constraints_text,
            agent_input=agent_input,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )

        # Add classification to output
        output.detected_category = category
        output.detected_intent = intent

        logger.info(
            f"Enhancement complete: confidence={confidence:.2f}, "
            f"techniques={len(applied_techniques)}, time={processing_time_ms}ms"
        )

        return output

    def enhance_simple(self, generic_prompt: str) -> str:
        """
        Simple enhancement interface - returns just the enhanced prompt.

        Convenience method for quick usage.

        Args:
            generic_prompt: The generic input prompt

        Returns:
            Enhanced prompt ready for Nano Banana Pro

        Example:
            agent = PromptEnhancementAgent()
            enhanced = agent.enhance_simple("Create an ad for our mop")
            print(enhanced)
        """

        agent_input = AgentInput(generic_prompt=generic_prompt)
        output = self.enhance(agent_input)

        return output.enhanced_prompt


# Convenience function
def enhance_prompt(generic_prompt: str) -> str:
    """
    Convenience function to enhance a prompt.

    Uses legacy mode for compatibility.

    Args:
        generic_prompt: The generic input prompt

    Returns:
        Enhanced prompt ready for Nano Banana Pro

    Example:
        from src.agents.nano import enhance_prompt

        enhanced = enhance_prompt("Create an ad for our mop")
        print(enhanced)
    """

    # Use framework mode for enhanced prompt quality
    agent = PromptEnhancementAgent(use_framework=True)
    return agent.enhance(AgentInput(generic_prompt=generic_prompt)).enhanced_prompt
