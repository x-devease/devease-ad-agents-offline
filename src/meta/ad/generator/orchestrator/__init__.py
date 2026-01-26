"""
Prompt Builder: Deterministic prompt generation from scorer recommendations.

This module provides a template-based prompt generation system that ensures
identical outputs for identical inputs, enabling statistical gap analysis and
debug-ready assets.

Key Features:
- Zero LLM intervention (pure Python dictionary lookup)
- Deterministic output (same input → same output)
- Direct feature-to-template mapping
- Transparent prompt logging
- Scorer recommendations prioritized over hardcoded defaults
"""

from .prompt_builder import PromptBuilder


__all__ = ["PromptBuilder"]
