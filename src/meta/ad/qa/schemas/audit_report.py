"""
Data schemas for Ad Reviewer audit reports.

This module defines all dataclass types for audit results from the 4-Guard system.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import json


class GuardStatus(str, Enum):
    """Status of a guard check."""
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"


class RiskLevel(str, Enum):
    """Risk level for cultural compliance."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class GuardResult:
    """Base class for guard results."""
    guard_name: str = ""
    status: GuardStatus = GuardStatus.PENDING
    reasoning: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


@dataclass
class GeometricResult(GuardResult):
    """Result from geometric integrity validation."""
    aspect_ratio_delta: float = 0.0
    num_matched_features: int = 0
    method_used: str = ""  # "homography" or "contour"

    def __post_init__(self):
        if not self.guard_name:
            self.guard_name = "geometric"


@dataclass
class AestheticResult(GuardResult):
    """Result from aesthetic quality check."""
    score: float = 0.0  # 0-10
    issues: List[str] = field(default_factory=list)
    negative_features_detected: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.guard_name:
            self.guard_name = "aesthetic"


@dataclass
class CulturalResult(GuardResult):
    """Result from cultural compliance check."""
    risk_level: RiskLevel = RiskLevel.LOW
    detected_issues: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1

    def __post_init__(self):
        if not self.guard_name:
            self.guard_name = "cultural"


@dataclass
class PerformanceScore(GuardResult):
    """Result from performance scoring."""
    overall_score: int = 0  # 0-100
    psychology_alignment: int = 0  # 0-100
    saliency_clarity: int = 0  # 0-100
    consistency_realism: int = 0  # 0-100

    def __post_init__(self):
        if not self.guard_name:
            self.guard_name = "performance"


@dataclass
class AuditReport:
    """
    Complete audit report for a generated image.

    This report contains the full audit trail from all 4 guards,
    along with generator metadata and execution metrics.
    """
    # Generator Metadata
    session_id: str                          # From generator session.json
    prompt_id: str                           # From generator prompt record
    image_id: str                            # From generator image record
    image_path: str
    product_image_path: str
    generation_model: str                    # e.g., "nano-banana-pro"

    # Strategy/Blueprint Metadata
    blueprint_id: str
    psychology_driver: Optional[str] = None  # From strategy rationale
    timestamp: str = ""

    # Final Status
    status: GuardStatus = GuardStatus.PENDING
    performance_score: Optional[int] = None  # 0-100 if PASS

    # Guard Results
    geometric: Optional[GeometricResult] = None
    aesthetic: Optional[AestheticResult] = None
    cultural: Optional[CulturalResult] = None
    performance: Optional[PerformanceScore] = None

    # Failure Info
    fail_guard: Optional[str] = None  # Which guard failed
    fail_code: Optional[str] = None
    fail_reason: Optional[str] = None

    # Execution Metrics
    total_execution_time_ms: float = 0.0
    api_calls_count: int = 0

    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        data = asdict(self)

        # Convert enums to strings
        if isinstance(data.get('status'), GuardStatus):
            data['status'] = data['status'].value

        if data.get('geometric'):
            if isinstance(data['geometric']['status'], GuardStatus):
                data['geometric']['status'] = data['geometric']['status'].value

        if data.get('aesthetic'):
            if isinstance(data['aesthetic']['status'], GuardStatus):
                data['aesthetic']['status'] = data['aesthetic']['status'].value

        if data.get('cultural'):
            if isinstance(data['cultural']['status'], GuardStatus):
                data['cultural']['status'] = data['cultural']['status'].value
            if isinstance(data['cultural'].get('risk_level'), RiskLevel):
                data['cultural']['risk_level'] = data['cultural']['risk_level'].value

        if data.get('performance'):
            if isinstance(data['performance']['status'], GuardStatus):
                data['performance']['status'] = data['performance']['status'].value

        return data

    def to_json(self, path: str, indent: int = 2):
        """Save report to JSON file."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditReport':
        """Create AuditReport from dict (e.g., loaded from JSON)."""
        # Convert status strings back to enums
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = GuardStatus(data['status'])

        # Convert guard results back to proper types
        if data.get('geometric'):
            if 'status' in data['geometric']:
                data['geometric']['status'] = GuardStatus(data['geometric']['status'])
            data['geometric'] = GeometricResult(**data['geometric'])

        if data.get('aesthetic'):
            if 'status' in data['aesthetic']:
                data['aesthetic']['status'] = GuardStatus(data['aesthetic']['status'])
            data['aesthetic'] = AestheticResult(**data['aesthetic'])

        if data.get('cultural'):
            if 'status' in data['cultural']:
                data['cultural']['status'] = GuardStatus(data['cultural']['status'])
            if 'risk_level' in data['cultural']:
                data['cultural']['risk_level'] = RiskLevel(data['cultural']['risk_level'])
            data['cultural'] = CulturalResult(**data['cultural'])

        if data.get('performance'):
            if 'status' in data['performance']:
                data['performance']['status'] = GuardStatus(data['performance']['status'])
            data['performance'] = PerformanceScore(**data['performance'])

        return cls(**data)

    @property
    def passed(self) -> bool:
        """Check if audit passed."""
        return self.status == GuardStatus.PASS

    @property
    def failed(self) -> bool:
        """Check if audit failed."""
        return self.status == GuardStatus.FAIL

    def get_failure_summary(self) -> str:
        """Get a human-readable summary of the failure."""
        if self.passed:
            return "Audit PASSED"

        if not self.fail_guard:
            return f"Audit FAILED: {self.fail_reason or 'Unknown reason'}"

        return f"Audit FAILED at {self.fail_guard} guard: {self.fail_reason or 'Unknown reason'}"
