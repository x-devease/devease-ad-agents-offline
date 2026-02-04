"""
Meta Diagnoser - Ad Account & Campaign Diagnostics

Diagnoses performance issues and provides actionable insights
for Meta ad accounts and campaigns.
"""

from src.meta.diagnoser.core.diagnoser import Diagnoser
from src.meta.diagnoser.core.issue_detector import IssueDetector

__all__ = [
    "Diagnoser",
    "IssueDetector",
]
