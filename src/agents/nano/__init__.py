"""
Nano Banana Pro Prompt Enhancement Agent.

Transforms generic prompts into high-fidelity Nano Banana Pro prompts
for professional asset production.

Author: Ad System Team
Date: 2026-01-30
Version: 1.0.0
"""

from __future__ import annotations

# Use lazy imports to avoid circular dependencies
# The agent module imports from framework, which imports from nano package
def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == "PromptEnhancementAgent":
        from src.agents.nano.core.agent import PromptEnhancementAgent
        return PromptEnhancementAgent
    elif name == "enhance_prompt":
        from src.agents.nano.core.agent import enhance_prompt
        return enhance_prompt
    elif name == "AgentInput":
        from src.agents.nano.core.types import AgentInput
        return AgentInput
    elif name == "AgentOutput":
        from src.agents.nano.core.types import AgentOutput
        return AgentOutput
    elif name == "PromptIntent":
        from src.agents.nano.core.types import PromptIntent
        return PromptIntent
    elif name == "PromptCategory":
        from src.agents.nano.core.types import PromptCategory
        return PromptCategory
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Note: __all__ is not defined because we use lazy imports via __getattr__
# This avoids circular dependencies with the framework module

# Legacy mode - use legacy implementation to avoid framework bugs
import os
USE_LEGACY = os.environ.get("NANO_LEGACY_MODE", "true").lower() == "true"
