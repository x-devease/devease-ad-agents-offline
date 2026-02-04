# Ad Reviewer Module

Visual QA & Risk Matrix (4-Guard System) for validating generated ad images.

## Overview

The Ad Reviewer provides comprehensive quality assurance for generated ad images through four sequential guards:

1. **GeometricGuard** - Product integrity validation using SIFT feature matching
2. **AestheticGuard** - Visual quality checking using VLM
3. **CulturalGuard** - Regional compliance checking using VLM
4. **PerformanceGuard** - Optimization scoring for A/B test prioritization

## Installation

```bash
# Core dependencies (already installed)
pip install opencv-python numpy pyyaml

# VLM dependencies
pip install openai
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from src.meta.ad.reviewer import VisualQAMatrix

# Initialize reviewer with config
reviewer = VisualQAMatrix(
    config_path="config/moprobo/facebook/config.yaml"
)

# Audit all images from a generator session
session_path = Path("results/ad/generator/generated/moprobo/facebook/20260130/tracking/session.json")
reports = reviewer.audit_session(session_path=session_path)

# Separate passed/failed
passed = [r for r in reports if r.status == GuardStatus.PASS]
failed = [r for r in reports if r.status == GuardStatus.FAIL]

# Sort by performance score
passed.sort(key=lambda r: r.performance_score or 0, reverse=True)

print(f"Passed: {len(passed)}/{len(reports)}")
for report in passed[:3]:
    print(f"  {report.image_id}: Score {report.performance_score}/100")
```

### Single Image Audit

```python
# Define blueprint
blueprint = {
    "id": "test_blueprint",
    "strategy_rationale": {
        "psychology_driver": "trust"
    },
    "nano_generation_rules": {
        "negative_prompt": ["dark background", "cluttered layout"]
    }
}

# Run audit
report = reviewer.audit(
    candidate_image_path="results/generator/generated_001.jpg",
    product_image_path="inputs/products/product_001.jpg",
    blueprint=blueprint
)

if report.status == GuardStatus.PASS:
    print(f"✓ PASS: Score {report.performance_score}/100")
else:
    print(f"✗ FAIL: {report.fail_reason}")
```

## Configuration

The reviewer uses the consolidated customer+platform config at `config/{customer}/{platform}/config.yaml`:

```yaml
qa_risk_matrix:
  enabled: true

  # Guard 1: Geometry
  geometry:
    enabled: true
    tolerance: 0.02                    # 2% aspect ratio tolerance
    min_features: 10                   # Min SIFT features
    fallback_to_contour: true          # Use contour if insufficient features

  # Guard 2: Aesthetics
  aesthetics:
    enabled: true
    min_score: 7.0                     # 0-10 scale
    model: "gpt-4o-mini"

  # Guard 3: Cultural
  cultural:
    enabled: true
    target_region: "Middle_East"       # Region for compliance
    risk_threshold: "HIGH"             # FAIL threshold

  # Guard 4: Performance
  performance:
    enabled: true
    psychology_weight: 0.40
    saliency_weight: 0.30
    consistency_weight: 0.30

  # Pipeline settings
  pipeline:
    stop_on_first_fail: true           # Fail-fast behavior
```

## Architecture

```
VisualQAMatrix (Orchestrator)
│
├── GeometricGuard (CPU-local, OpenCV)
│   ├── SIFT feature matching
│   ├── Homography decomposition
│   └── Contour fallback
│
├── AestheticGuard (VLM)
│   ├── Artifact detection
│   ├── Layout analysis
│   └── Negative prompt check
│
├── CulturalGuard (VLM)
│   ├── Region-specific taboos
│   ├── Compliance checking
│   └── Risk classification
│
└── PerformanceGuard (VLM)
    ├── Psychology alignment (40%)
    ├── Saliency & clarity (30%)
    └── Consistency & realism (30%)
```

## Output

Each audit produces an `AuditReport` containing:

- **Metadata**: session_id, prompt_id, image_id, generation_model
- **Status**: PASS or FAIL
- **Performance Score**: 0-100 (if passed)
- **Guard Results**: Detailed results from each guard
- **Failure Info**: Which guard failed and why
- **Execution Metrics**: Time and API call count

Example report:
```json
{
  "session_id": "20260130_143022",
  "prompt_id": "prompt_001",
  "image_id": "img_001",
  "status": "pass",
  "performance_score": 82,
  "geometric": {
    "status": "pass",
    "aspect_ratio_delta": 0.003,
    "method_used": "homography"
  },
  "aesthetic": {
    "status": "pass",
    "score": 8.2
  },
  "cultural": {
    "status": "pass",
    "risk_level": "low"
  },
  "performance": {
    "psychology_alignment": 85,
    "saliency_clarity": 78,
    "consistency_realism": 82
  }
}
```

## Testing

```bash
# Run all tests
pytest src/meta/ad/reviewer/tests/

# Run specific test file
pytest src/meta/ad/reviewer/tests/test_geometric_guard.py -v

# Run with coverage
pytest src/meta/ad/reviewer/tests/ --cov=src/meta/ad/reviewer --cov-report=html
```

## Module Structure

```
src/meta/ad/reviewer/
├── __init__.py
├── pipeline.py                    # VisualQAMatrix orchestrator
├── guards/
│   ├── geometric_guard.py
│   ├── aesthetic_guard.py
│   ├── cultural_guard.py
│   └── performance_guard.py
├── vlms/
│   ├── base.py                    # Abstract VLM interface
│   └── openai_vlm.py              # GPT-4o implementation
├── schemas/
│   └── audit_report.py            # Dataclass definitions
├── utils/
│   ├── image_processor.py         # OpenCV utilities
│   └── geometry_utils.py          # SIFT, homography
└── tests/
    ├── test_geometric_guard.py
    └── test_pipeline.py
```

## API Reference

### VisualQAMatrix

```python
class VisualQAMatrix:
    def __init__(self, config_path: str | Path, vlm_client: Optional[VLMClient] = None):
        """Initialize the Visual QA pipeline."""

    def audit(self, candidate_image_path: str, product_image_path: str, blueprint: Dict) -> AuditReport:
        """Run complete 4-guard audit on a generated image."""

    def audit_session(self, session_path: str | Path, blueprint: Optional[Dict] = None) -> List[AuditReport]:
        """Audit all images from a generator session."""
```

### AuditReport

```python
@dataclass
class AuditReport:
    session_id: str
    prompt_id: str
    image_id: str
    image_path: str
    product_image_path: str
    generation_model: str
    blueprint_id: str
    psychology_driver: Optional[str]
    timestamp: str
    status: GuardStatus
    performance_score: Optional[int]
    geometric: Optional[GeometricResult]
    aesthetic: Optional[AestheticResult]
    cultural: Optional[CulturalResult]
    performance: Optional[PerformanceScore]
    fail_guard: Optional[str]
    fail_code: Optional[str]
    fail_reason: Optional[str]
    total_execution_time_ms: float
    api_calls_count: int

    @property
    def passed(self) -> bool: ...

    @property
    def failed(self) -> bool: ...

    def to_dict(self) -> Dict[str, Any]: ...

    def to_json(self, path: str) -> None: ...
```

## Design Document

See `docs/ad_reviewer_v2_design.md` for complete design documentation.
