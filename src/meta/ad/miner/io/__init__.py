"""I/O module for ad miner.

This module provides JSON I/O and Markdown generation functionality.
"""

from .patterns_io import (
    PatternsIO,
    load_patterns_json,
    save_patterns_json,
    generate_patterns_markdown,
)

__all__ = [
    "PatternsIO",
    "load_patterns_json",
    "save_patterns_json",
    "generate_patterns_markdown",
]
