"""
Diagnoser AI Agent Team

Automated agent team for continuous detector optimization.
"""

__version__ = "0.1.0"

from .memory_agent import MemoryAgent
from .orchestrator import Orchestrator

__all__ = [
    "MemoryAgent",
    "Orchestrator",
]

# Individual agents are embedded in Orchestrator for this simplified implementation
