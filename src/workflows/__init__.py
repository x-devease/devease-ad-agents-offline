"""
Workflow orchestration for budget allocation operations.

This module provides high-level workflow abstractions for:
- Feature extraction
- Budget allocation
- Parameter tuning

Each workflow encapsulates the business logic separately from CLI concerns,
making them testable, reusable, and maintainable.
"""

from src.workflows.base import Workflow, WorkflowResult
from src.workflows.extract_workflow import ExtractWorkflow
from src.workflows.allocation_workflow import AllocationWorkflow
from src.workflows.tuning_workflow import TuningWorkflow

__all__ = [
    "Workflow",
    "WorkflowResult",
    "ExtractWorkflow",
    "AllocationWorkflow",
    "TuningWorkflow",
]
