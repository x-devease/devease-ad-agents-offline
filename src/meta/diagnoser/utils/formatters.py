"""Formatting utilities for diagnostics."""

from typing import Dict, Any


def format_diagnosis(diagnosis: Dict[str, Any]) -> str:
    """Format diagnosis for display.

    Args:
        diagnosis: Dictionary with diagnosis data

    Returns:
        Formatted string representation
    """
    return str(diagnosis)
