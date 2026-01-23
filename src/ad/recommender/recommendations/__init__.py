"""Recommendation system for creative optimization.

This module provides statistical pattern-based recommendations
for improving creative ROAS performance.

Enhanced output features (available but not exported):
- Evidence builder: Statistical significance, confidence intervals, cross-customer validation
- Output structure: Multi-view outputs (executive summary, implementation steps, risk assessment)
- Formatters: Multiple output formats (JSON, HTML, Markdown, Slack)

Note: Conversion of recommendations to creative generation prompts
is out of scope and handled by external systems.
"""

from .rule_engine import RuleEngine

__all__ = [
    "RuleEngine",
]
