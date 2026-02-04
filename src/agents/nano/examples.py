"""
Example usage of the Nano Banana Pro Prompt Enhancement Agent.

This demonstrates how to use the agent to transform generic prompts
into high-fidelity Nano Banana Pro prompts.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from src.agents.nano.core.agent import PromptEnhancementAgent
from src.agents.nano.core.types import AgentInput


def example_1_ultra_simple():
    """Example 1: Ultra-simple input."""
    print("=" * 80)
    print("EXAMPLE 1: Ultra-Simple Input")
    print("=" * 80)

    agent = PromptEnhancementAgent()

    input_prompt = AgentInput(
        generic_prompt="Create an ad for our mop",
    )

    output = agent.enhance(input_prompt)

    print(f"\nINPUT PROMPT:")
    print(f"  {input_prompt.generic_prompt}")

    print(f"\nOUTPUT PROMPT:")
    print(f"  {output.enhanced_prompt[:500]}...")

    print(f"\nMETADATA:")
    print(f"  Category: {output.detected_category.value}")
    print(f"  Intent: {output.detected_intent.value}")
    print(f"  Confidence: {output.confidence:.2f}")
    print(f"  Techniques: {', '.join(output.techniques_used)}")
    print(f"  Processing Time: {output.processing_time_ms}ms")

    print(f"\nEXPLANATION:")
    print(f"  {output.explanation}")


def example_2_lifestyle():
    """Example 2: Lifestyle advertisement."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Lifestyle Advertisement")
    print("=" * 80)

    agent = PromptEnhancementAgent()

    input_prompt = AgentInput(
        generic_prompt="Create an ad showing someone happy with our clean floor",
        emotion_goal="satisfaction, relief",
        target_audience="homeowners",
    )

    output = agent.enhance(input_prompt)

    print(f"\nINPUT PROMPT:")
    print(f"  {input_prompt.generic_prompt}")

    print(f"\nOUTPUT PROMPT:")
    print(f"  {output.enhanced_prompt[:600]}...")

    print(f"\nMETADATA:")
    print(f"  Category: {output.detected_category.value}")
    print(f"  Intent: {output.detected_intent.value}")
    print(f"  Confidence: {output.confidence:.2f}")


def example_3_comparative():
    """Example 3: Comparative infographic."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Comparative Infographic")
    print("=" * 80)

    agent = PromptEnhancementAgent()

    input_prompt = AgentInput(
        generic_prompt="Make a chart comparing our mop to competitors",
    )

    output = agent.enhance(input_prompt)

    print(f"\nINPUT PROMPT:")
    print(f"  {input_prompt.generic_prompt}")

    print(f"\nOUTPUT PROMPT:")
    print(f"  {output.enhanced_prompt[:600]}...")

    print(f"\nTECHNIQUES APPLIED:")
    for technique in output.applied_techniques:
        print(f"  - {technique.technique_name}: {technique.description}")


def example_4_with_thinking():
    """Example 4: With thinking block enabled."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: With Thinking Block")
    print("=" * 80)

    agent = PromptEnhancementAgent()

    input_prompt = AgentInput(
        generic_prompt="Show a story of someone cleaning their messy kitchen",
        enable_thinking=True,
    )

    output = agent.enhance(input_prompt)

    print(f"\nINPUT PROMPT:")
    print(f"  {input_prompt.generic_prompt}")

    print(f"\nTHINKING BLOCK:")
    print(f"  {output.thinking_block.format()}")

    print(f"\nENHANCED PROMPT (first 400 chars):")
    print(f"  {output.enhanced_prompt[:400]}...")


def example_5_simple_interface():
    """Example 5: Using the simple interface."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Simple Interface")
    print("=" * 80)

    from src.agents.nano.core.agent import enhance_prompt

    enhanced = enhance_prompt("Create a product photo of our mop")

    print(f"\nINPUT:")
    print(f"  'Create a product photo of our mop'")

    print(f"\nOUTPUT (first 300 chars):")
    print(f"  {enhanced[:300]}...")


if __name__ == "__main__":
    # Run examples
    example_1_ultra_simple()
    example_2_lifestyle()
    example_3_comparative()
    example_4_with_thinking()
    example_5_simple_interface()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
