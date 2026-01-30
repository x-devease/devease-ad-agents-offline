# Test Suite for Template-Driven Ad Generator

## Test Structure

```
tests/
├── unit/                           # Unit tests for individual modules
│   ├── test_generator_paths.py     # Path management tests
│   ├── test_product_preprocessor.py # Product preprocessing tests
│   ├── test_template_selector.py    # Template selection tests
│   ├── test_smart_typer.py          # Smart text overlay tests
│   ├── test_background_generator.py # Background generation tests
│   └── test_physics_compositor.py   # Physics compositing tests
└── integration/                     # Integration tests
    ├── test_pipeline_integration.py # Pipeline integration tests
    └── test_pipeline_end_to_end.py   # End-to-end pipeline tests
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Unit Tests Only
```bash
pytest tests/unit/
```

### Run Integration Tests Only
```bash
pytest tests/integration/
```

### Run Specific Test File
```bash
pytest tests/unit/test_generator_paths.py
```

### Run Specific Test Class
```bash
pytest tests/unit/test_generator_paths.py::TestGeneratorPaths
```

### Run Specific Test
```bash
pytest tests/unit/test_generator_paths.py::TestGeneratorPaths::test_init_basic
```

### Run with Coverage
```bash
pytest --cov=src/meta/ad/generator/template_system --cov-report=html
```

### Run Verbose Output
```bash
pytest -v
```

## Test Coverage

Current test coverage focuses on:

### Path Management (`test_generator_paths.py`)
- ✅ Customer/platform name normalization
- ✅ Config path resolution
- ✅ Blueprint path resolution
- ✅ Campaign content path (with platform overrides)
- ✅ Ad Miner output paths
- ✅ Ad Generator output paths (generated, backgrounds, composited)
- ✅ Product input paths
- ✅ Directory creation

### Product Preprocessor (`test_product_preprocessor.py`)
- ✅ Perspective type enum
- ✅ PreprocessorResult dataclass
- ✅ Transparency trimming
- ✅ Perspective detection (HIGH_ANGLE vs EYE_LEVEL)
- ✅ Mask generation
- ✅ Full preprocessing pipeline

### Template Selector (`test_template_selector.py`)
- ✅ TemplateSpec dataclass
- ✅ PsychologySpec dataclass
- ✅ Template loading from YAML
- ✅ Psychology catalog loading
- ✅ Template retrieval by ID
- ✅ Template filtering by psychology driver
- ✅ Psychology-driven auto-selection
- ✅ Blueprint-based selection

### Smart Typer (`test_smart_typer.py`)
- ✅ CampaignContent dataclass
- ✅ SmartColorCalculator (luminance, auto-contrast)
- ✅ CollisionDetector (IoU calculation)
- ✅ Non-colliding position finding
- ✅ Text rendering
- ✅ Psychology-enhanced rendering

### Background Generator (`test_background_generator.py`)
- ✅ AspectRatio enum
- ✅ GenerationConfig dataclass
- ✅ BackgroundPrompt perspective injection
- ✅ NanoBackgroundGenerator class
- ✅ GeneratedBackground dataclass
- ✅ Perspective injection tests (HIGH_ANGLE vs EYE_LEVEL)
- ✅ Various aspect ratios (1:1, 3:4, 4:3, 9:16)
- ✅ Config variations (steps, CFG scale, batch size)

### Physics Compositor (`test_physics_compositor.py`)
- ✅ ShadowDirection enum
- ✅ CompositingConfig dataclass
- ✅ CompositingResult dataclass
- ✅ PhysicsCompositor class
- ✅ Dual-layer shadow stack generation
- ✅ Contact shadow creation
- ✅ Cast shadow creation (affine skew)
- ✅ Shadow direction variations (left, right, top, bottom)
- ✅ Light matching application
- ✅ Light wrap application
- ✅ Edge mask creation
- ✅ Light wrap intensity variations
- ✅ Light match opacity variations

### Pipeline Integration (`test_pipeline_integration.py`, `test_pipeline_end_to_end.py`)
- ✅ Product preprocessing integration
- ✅ Template selection integration
- ✅ Path resolution consistency
- ✅ Cross-module integration
- ✅ Complete environment setup
- ✅ Config loading flow
- ✅ Blueprint to template flow
- ✅ Campaign content to text rendering flow
- ✅ Pipeline initialization
- ✅ Output saving
- ✅ Multiple variants handling
- ✅ Background handling
- ✅ Error handling (missing files)
- ✅ Metadata tracking
- ✅ Performance tests (slow)

## Fixtures

Test fixtures are defined in each test file:
- `sample_templates_yaml`: Template configuration
- `sample_psychology_yaml`: Psychology catalog
- `sample_master_blueprint`: Master blueprint
- `sample_product_image`: Product image for testing
- `sample_campaign_content`: Campaign content
- `sample_template_spec`: Template specification

## Writing New Tests

### Unit Test Template

```python
"""
Unit tests for [module_name].
"""

import pytest
from src.meta.ad.generator.template_system.[module_name] import (
    ClassToTest,
)


class TestClassToTest:
    """Test ClassToTest functionality."""

    def test_init_default(self):
        """Test default initialization."""
        obj = ClassToTest()
        assert obj.attribute == expected_value

    def test_method_name(self):
        """Test method description."""
        obj = ClassToTest()
        result = obj.method()
        assert result == expected_result
```

### Integration Test Template

```python
"""
Integration tests for [feature].
"""

import pytest
from pathlib import Path
from src.meta.ad.generator.template_system import generate_ads


class TestEndToEnd:
    """Test end-to-end pipeline."""

    def test_full_pipeline(self, tmp_path):
        """Test complete generation pipeline."""
        # Setup
        # Execute
        # Verify
        assert result is not None
```

## Continuous Integration

Tests are configured to run with:
- pytest as test runner
- pytest-cov for coverage reporting
- pytest-html for HTML reports (optional)

## Future Enhancements

1. ✅ All core modules have unit tests
2. ✅ Integration tests cover pipeline flow
3. ⏳ Add visual regression tests for generated images
4. ⏳ Add performance benchmarks
5. ⏳ Add property-based testing with Hypothesis
6. ⏳ Add Golden Master tests for reference outputs
