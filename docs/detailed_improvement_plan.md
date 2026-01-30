# Ad Miner Detailed Improvement Plan: Schemas, Paths & Refactoring

**Status:** Detailed Implementation Plan
**Date:** 2026-01-27
**Branch:** ad-reviewer
**Estimated Duration:** 6-8 weeks
**Version:** 2.0

---

## Table of Contents

1. [Overview](#1-overview)
2. [Terminology Changes](#2-terminology-changes)
3. [Input Schema Specification](#3-input-schema-specification)
4. [Output Schema Specification](#4-output-schema-specification)
5. [Path Structure Design](#5-path-structure-design)
6. [Code Refactoring Plan](#6-code-refactoring-plan)
7. [Implementation Phases](#7-implementation-phases)
8. [File-by-File Changes](#8-file-by-file-changes)

---

## 1. Overview

### 1.1 Objectives

**Primary Goals:**
1. Rename "recommendation" → "pattern" throughout codebase for clarity
2. Implement structured, validated input schemas (CSV)
3. Implement structured, validated output schemas (JSON)
4. Design organized, hierarchical path structure
5. Add context-awareness (goal/product/branch segmentation)

**Key Changes:**
- `recommendation` → `pattern` (function names, variables, files)
- Generic CSV → Structured schema with metadata columns
- MD-only output → JSON primary + MD generated
- Flat paths → Hierarchical segmentation paths

---

## 2. Terminology Changes

### 2.1 Concept Rename: Recommendation → Pattern

**Rationale:** "Pattern" better describes what the system does: discovers statistical patterns in creative data.

| Old Term | New Term | Example |
|----------|----------|---------|
| `recommendation` | `pattern` | Pattern detection, pattern scoring |
| `generate_recommendations()` | `mine_patterns()` | Mine patterns from data |
| `load_recommendations()` | `load_patterns()` | Load mined patterns |
| `recommendations.md` | `patterns.md` | Output file name |
| `ad_recommender` | `ad_miner` | Already done ✓ |
| `RecommendationEngine` | `PatternMiner` | Class name |

### 2.2 Complete Rename Mapping

**Files to rename:**
```bash
# Python files
src/meta/ad/miner/recommendations/rule_engine.py
  → src/meta/ad/miner/patterns/rule_engine.py

src/meta/ad/miner/recommendations/md_io.py
  → src/meta/ad/miner/patterns/json_io.py  (Now handles JSON)

src/meta/ad/miner/recommendations/prompt_formatter.py
  → src/meta/ad/miner/patterns/prompt_formatter.py

src/meta/ad/miner/recommendations/evidence_builder.py
  → src/meta/ad/miner/patterns/evidence_builder.py

src/meta/ad/miner/recommendations/formatters.py
  → src/meta/ad/miner/patterns/formatters.py

tests/unit/ad/miner/test_rule_engine_gen.py
  → tests/unit/ad/miner/test_pattern_mining.py

tests/integration/test_ad_mining_generation.py
  → tests/integration/test_pattern_mining_generation.py
```

**Code changes (function/class names):**
```python
# Old
class RuleEngine:
    def generate_recommendations(creative)
    def load_patterns(recommendations)

# New
class PatternMiner:
    def mine_patterns(creative)
    def load_patterns(patterns)

# Old
def export_recommendations_md(data, path)
def load_recommendations_file(path)

# New
def export_patterns_json(data, path)
def load_patterns_file(path)
```

---

## 3. Input Schema Specification

### 3.1 Schema Definition File

**Path:** `src/meta/ad/miner/schemas/input_schema.yaml`

```yaml
schema:
  version: "1.0"
  name: "creative_features_with_metadata"
  format: "CSV"
  encoding: "utf-8"

# Metadata about the schema
metadata:
  author: "Ad Miner Team"
  created_date: "2026-01-27"
  description: "Creative features + campaign metadata for pattern mining"

# Column definitions
columns:
  # === IDENTIFIER COLUMNS ===
  - name: "creative_id"
    type: "string"
    required: true
    unique: true
    description: "Unique creative identifier from ad platform"
    example: "moprobo_meta_20250115_001234"
    validation:
      pattern: "^[a-z_]+_[0-9]+$"

  - name: "filename"
    type: "string"
    required: true
    description: "Image filename"
    example: "moprobo_meta_001.jpg"
    validation:
      pattern: "^.+\\.(jpg|jpeg|png)$"

  # === PERFORMANCE METRICS ===
  - name: "roas"
    type: "float"
    required: true
    description: "Return on ad spend (revenue / spend)"
    range: [0.0, null]
    example: 2.45
    validation:
      min: 0.0
      allow_null: false

  - name: "spend"
    type: "float"
    required: false
    description: "Amount spent on this creative"
    range: [0.0, null]
    example: 150.50

  - name: "impressions"
    type: "integer"
    required: false
    description: "Number of impressions"
    range: [0, null]
    example: 15000

  - name: "clicks"
    type: "integer"
    required: false
    description: "Number of clicks"
    range: [0, null]
    example: 450

  # === CONTEXT METADATA (SEGMENTATION KEYS) ===
  - name: "campaign_goal"
    type: "enum"
    required: true
    description: "Primary campaign objective"
    values:
      - "awareness"
      - "conversion"
      - "traffic"
      - "lead_generation"
      - "app_installs"
      - "engagement"
      - "sales"
      - "unknown"
    default: "unknown"
    example: "conversion"

  - name: "product"
    type: "string"
    required: true
    description: "Product being advertised"
    example: "Power Station"
    default: "unknown"
    validation:
      max_length: 100

  - name: "branch"
    type: "enum"
    required: true
    description: "Regional or organizational branch"
    values:
      - "US"
      - "EU"
      - "UK"
      - "APAC"
      - "LATAM"
      - "Global"
      - "unknown"
    default: "unknown"
    example: "US"

  - name: "campaign_id"
    type: "string"
    required: false
    description: "Campaign identifier for grouping"
    example: "moprobo_conversion_2025_01"
    validation:
      max_length: 100

  - name: "adset_id"
    type: "string"
    required: false
    description: "Adset identifier for grouping"
    example: "moprobo_conversion_25-34_us"
    validation:
      max_length: 100

  # === VISUAL FEATURES (29 features total) ===
  - name: "direction"
    type: "enum"
    required: false
    description: "Camera angle relative to product"
    values:
      - "front"
      - "side"
      - "overhead"
      - "45-degree"
      - "low_angle"
      - "dutch_angle"
      - "unknown"
    example: "overhead"

  - name: "lighting_style"
    type: "enum"
    required: false
    description: "Lighting setup style"
    values:
      - "studio"
      - "natural"
      - "artificial"
      - "mixed"
      - "unknown"
    example: "studio"

  - name: "lighting_type"
    type: "enum"
    required: false
    description: "Type of lighting source"
    values:
      - "Artificial"
      - "Natural"
      - "Mixed"
      - "unknown"
    example: "Artificial"

  - name: "mood_lighting"
    type: "enum"
    required: false
    description: "Mood or atmosphere created by lighting"
    values:
      - "clinical"
      - "energetic"
      - "natural"
      - "dramatic"
      - "romantic"
      - "mysterious"
      - "unknown"
    example: "energetic"

  - name: "primary_colors"
    type: "list"
    item_type: "string"
    required: false
    description: "Primary colors present in image (comma-separated)"
    example: "green, white, gray"
    delimiter: ", "

  - name: "color_balance"
    type: "enum"
    required: false
    description: "Overall color temperature balance"
    values:
      - "cool-dominant"
      - "warm-dominant"
      - "neutral"
      - "balanced"
      - "unknown"
    example: "cool-dominant"

  - name: "temperature"
    type: "enum"
    required: false
    description: "Color temperature"
    values:
      - "Cool"
      - "Warm"
      - "Neutral"
      - "unknown"
    example: "Cool"

  - name: "color_saturation"
    type: "enum"
    required: false
    description: "Color intensity"
    values:
      - "high"
      - "medium"
      - "low"
      - "unknown"
    example: "high"

  - name: "color_vibrancy"
    type: "enum"
    required: false
    description: "Color vibrancy"
    values:
      - "vibrant"
      - "moderate"
      - "muted"
      - "unknown"
    example: "vibrant"

  - name: "product_position"
    type: "enum"
    required: false
    description: "Product position in frame"
    values:
      - "left"
      - "center"
      - "right"
      - "top-left"
      - "top-right"
      - "bottom-left"
      - "bottom-right"
      - "unknown"
    example: "bottom-right"

  - name: "product_placement"
    type: "enum"
    required: false
    description: "Where product is placed"
    values:
      - "left"
      - "right"
      - "center"
      - "unknown"
    example: "right"

  - name: "product_visibility"
    type: "enum"
    required: false
    description: "How much of product is visible"
    values:
      - "full"
      - "partial"
      - "minimal"
      - "obscured"
      - "unknown"
    example: "partial"

  - name: "visual_prominence"
    type: "enum"
    required: false
    description: "Visual dominance of product"
    values:
      - "dominant"
      - "prominent"
      - "subdued"
      - "unknown"
    example: "dominant"

  - name: "human_elements"
    type: "enum"
    required: false
    description: "Presence of humans"
    values:
      - "Lifestyle context"
      - "Face visible"
      - "Body visible"
      - "None"
      - "unknown"
    example: "Lifestyle context"

  - name: "product_context"
    type: "enum"
    required: false
    description: "Product usage context"
    values:
      - "isolated"
      - "in-use"
      - "lifestyle"
      - "unknown"
    example: "isolated"

  - name: "context_richness"
    type: "enum"
    required: false
    description: "Background detail level"
    values:
      - "rich"
      - "moderate"
      - "minimal"
      - "unknown"
    example: "moderate"

  - name: "background_content_type"
    type: "enum"
    required: false
    description: "Type of background"
    values:
      - "solid-color"
      - "textured"
      - "environment"
      - "blurred"
      - "unknown"
    example: "solid-color"

  - name: "relationship_depiction"
    type: "enum"
    required: false
    description: "Product-people relationship"
    values:
      - "product-alone"
      - "product-with-people"
      - "product-in-environment"
      - "unknown"
    example: "product-in-environment"

  - name: "visual_flow"
    type: "enum"
    required: false
    description: "Eye movement pattern"
    values:
      - "forced"
      - "natural"
      - "z-pattern"
      - "f-pattern"
      - "circular"
      - "unknown"
    example: "forced"

  - name: "composition_style"
    type: "enum"
    required: false
    description: "Overall composition approach"
    values:
      - "balanced"
      - "asymmetrical"
      - "minimal"
      - "complex"
      - "unknown"
    example: "balanced"

  - name: "depth_layers"
    type: "enum"
    required: false
    description: "Number of visual depth layers"
    values:
      - "shallow"
      - "moderate"
      - "deep"
      - "unknown"
    example: "shallow"

  - name: "contrast_level"
    type: "enum"
    required: false
    description: "Contrast between light and dark"
    values:
      - "high"
      - "medium"
      - "low"
      - "unknown"
    example: "high"

  - name: "background_tone_contrast"
    type: "enum"
    required: false
    description: "Product vs background contrast"
    values:
      - "high"
      - "medium"
      - "low"
      - "unknown"
    example: "high"

  - name: "local_contrast"
    type: "enum"
    required: false
    description: "Contrast within product"
    values:
      - "high"
      - "medium"
      - "low"
      - "unknown"
    example: "high"

  - name: "image_style"
    type: "enum"
    required: false
    description: "Overall image aesthetic"
    values:
      - "professional"
      - "casual"
      - "lifestyle"
      - "editorial"
      - "unknown"
    example: "professional"

  - name: "visual_complexity"
    type: "enum"
    required: false
    description: "Visual complexity"
    values:
      - "simple"
      - "moderate"
      - "complex"
      - "unknown"
    example: "simple"

  - name: "product_angle"
    type: "enum"
    required: false
    description: "Product camera angle"
    values:
      - "front"
      - "45-degree"
      "side"
      - "back"
      - "top-down"
      - "unknown"
    example: "45-degree"

  - name: "product_presentation"
    type: "enum"
    required: false
    description: "How product is presented"
    values:
      - "Full product"
      - "Partial"
      - "Close-up"
      - "Multiple views"
      - "unknown"
    example: "Full product"

  - name: "framing"
    type: "enum"
    required: false
    description: "Camera framing"
    values:
      - "Close-up"
      - "Medium shot"
      - "Wide shot"
      - "Extreme close-up"
      - "unknown"
    example: "Medium shot"

  - name: "architectural_elements_presence"
    type: "enum"
    required: false
    description: "Presence of architectural elements"
    values:
      - "yes"
      - "no"
      - "unknown"
    example: "no"

  - name: "primary_focal_point"
    type: "enum"
    required: false
    description: "Primary visual focus"
    values:
      - "product"
      - "person"
      - "text"
      - "background"
      - "unknown"
    example: "product"

  # === HUMAN SUBJECT FEATURES ===
  - name: "person_count"
    type: "enum"
    required: false
    description: "Number of people in image"
    values:
      - "single"
      - "couple"
      - "multiple"
      - "crowd"
      - "none"
      - "unknown"
    example: "single"

  - name: "person_relationship_type"
    type: "enum"
    required: false
    description: "Relationship between people"
    values:
      - "individual"
      - "couple"
      - "family"
      - "friends"
      - "colleagues"
      - "unknown"
    example: "individual"

  - name: "person_gender"
    type: "enum"
    required: false
    description: "Gender of primary person"
    values:
      - "male"
      - "female"
      - "multiple"
      - "unknown"
    example: "male"

  - name: "person_age_group"
    type: "enum"
    required: false
    description: "Age group of primary person"
    values:
      - "child"
      - "teen"
      - "young_adult"
      - "adult"
      - "senior"
      - "mixed"
      - "unknown"
    example: "adult"

  - name: "person_activity"
    type: "enum"
    required: false
    description: "What person is doing"
    values:
      - "posing"
      - "using_product"
      - "interacting"
      - "observing"
      - "working"
      - "playing"
      - "unknown"
    example: "posing"

  # === TEXT & CTA FEATURES ===
  - name: "text_elements"
    type: "list"
    item_type: "string"
    required: false
    description: "Text elements present (comma-separated)"
    example: "Headline, Subheadline, Feature Icons"
    delimiter: ", "

  - name: "cta_visuals"
    type: "list"
    item_type: "string"
    required: false
    description: "Call-to-action visual elements (comma-separated)"
    example: "Highlighting, Button"
    delimiter: ", "

  - name: "problem_solution_narrative"
    type: "enum"
    required: false
    description: "Narrative structure"
    values:
      - "problem"
      - "solution"
      - "both"
      - "neither"
      - "unknown"
    example: "both"

  - name: "emotional_tone"
    type: "enum"
    required: false
    description: "Emotional tone of image"
    values:
      - "Exciting"
      - "Calm"
      - "Urgent"
      - "Trustworthy"
      - "Professional"
      - "Playful"
      - "unknown"
    example: "Exciting"

  - name: "activity_level"
    type: "enum"
    required: false
    description: "Activity level in image"
    values:
      - "active"
      - "passive"
      - "neutral"
      - "unknown"
    example: "active"

# Validation rules
validation:
  - rule: "creative_id is unique"
    message: "Duplicate creative_id found"

  - rule: "roas >= 0"
    message: "ROAS cannot be negative"

  - rule: "campaign_goal in valid_goals"
    message: "Invalid campaign_goal value"

  - rule: "branch in valid_branches"
    message: "Invalid branch value"

  - rule: "filename matches image pattern"
    message: "Filename must be an image file"

  - rule: "If spend is present, spend >= 0"
    message: "Spend cannot be negative"

  - rule: "If impressions is present, impressions >= 0"
    message: "Impressions cannot be negative"
```

---

### 3.2 Schema Validator Implementation

**Path:** `src/meta/ad/miner/validation/input_validator.py`

```python
"""Input schema validator for creative features CSV."""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import logging
import yaml

logger = logging.getLogger(__name__)


class InputSchemaValidator:
    """
    Validate creative features CSV against schema definition.

    Loads schema from YAML and validates:
    - Required columns present
    - Data types correct
    - Enum values valid
    - Value ranges respected
    - Unique constraints satisfied
    """

    SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "input_schema.yaml"

    def __init__(self, csv_path: str | Path, schema_path: str | Path = None):
        """
        Initialize validator.

        Args:
            csv_path: Path to CSV file to validate
            schema_path: Optional path to schema YAML (defaults to input_schema.yaml)
        """
        self.csv_path = Path(csv_path)
        self.schema_path = Path(schema_path) if schema_path else self.SCHEMA_PATH
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.schema: Dict[str, Any] = {}

        # Load schema
        self._load_schema()

    def _load_schema(self) -> None:
        """Load schema definition from YAML."""
        try:
            with open(self.schema_path) as f:
                self.schema = yaml.safe_load(f)
            logger.info(f"Loaded schema from {self.schema_path}")
        except Exception as e:
            raise ValueError(f"Failed to load schema from {self.schema_path}: {e}")

    def validate(self) -> bool:
        """
        Validate CSV against schema.

        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []

        # Load CSV
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            self.errors.append(f"Failed to read CSV: {e}")
            return False

        # Run validation checks
        self._validate_required_columns(df)
        self._validate_data_types(df)
        self._validate_enums(df)
        self._validate_ranges(df)
        self._validate_uniques(df)
        self._validate_patterns(df)

        is_valid = len(self.errors) == 0

        if not is_valid:
            logger.error(f"Validation failed with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  ✗ {error}")

        if self.warnings:
            logger.warning(f"Validation produced {len(self.warnings)} warnings")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")

        return is_valid

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Check that all required columns exist."""
        columns_def = self.schema.get("columns", [])
        required_columns = [
            col["name"] for col in columns_def
            if col.get("required", False)
        ]

        missing = set(required_columns) - set(df.columns)
        if missing:
            self.errors.append(f"Missing required columns: {sorted(missing)}")

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate column data types."""
        columns_def = self.schema.get("columns", [])

        for col_def in columns_def:
            col_name = col_def["name"]
            if col_name not in df.columns:
                continue

            col_type = col_def.get("type")

            # Numeric types
            if col_type in ["float", "integer"]:
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    self.errors.append(
                        f"Column '{col_name}' must be numeric (type: {col_type})"
                    )

    def _validate_enums(self, df: pd.DataFrame) -> None:
        """Validate enum columns have valid values."""
        columns_def = self.schema.get("columns", [])

        for col_def in columns_def:
            if col_def.get("type") != "enum":
                continue

            col_name = col_def["name"]
            if col_name not in df.columns:
                continue

            valid_values = col_def.get("values", [])
            if not valid_values:
                continue

            # Check for invalid values
            invalid_mask = ~df[col_name].isin(valid_values) & df[col_name].notna()
            invalid_values = df.loc[invalid_mask, col_name].unique()

            if len(invalid_values) > 0:
                self.errors.append(
                    f"Column '{col_name}' has invalid values: {invalid_values}. "
                    f"Valid values: {valid_values}"
                )

    def _validate_ranges(self, df: pd.DataFrame) -> None:
        """Validate numeric column ranges."""
        columns_def = self.schema.get("columns", [])

        for col_def in columns_def:
            col_name = col_def["name"]
            if col_name not in df.columns:
                continue

            validation = col_def.get("validation", {})
            min_val = validation.get("min")
            max_val = validation.get("max")

            if min_val is not None and col_name in df.columns:
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    invalid = df[df[col_name] < min_val]
                    if len(invalid) > 0:
                        self.errors.append(
                            f"Column '{col_name}' has {len(invalid)} values < {min_val}"
                        )

    def _validate_uniques(self, df: pd.DataFrame) -> None:
        """Check unique constraints."""
        columns_def = self.schema.get("columns", [])

        for col_def in columns_def:
            col_name = col_def["name"]
            if not col_def.get("unique", False):
                continue

            if col_name not in df.columns:
                continue

            duplicates = df[col_name].duplicated()
            if duplicates.sum() > 0:
                self.errors.append(
                    f"Column '{col_name}' must be unique. "
                    f"Found {duplicates.sum()} duplicates."
                )

    def _validate_patterns(self, df: pd.DataFrame) -> None:
        """Validate regex patterns."""
        columns_def = self.schema.get("columns", [])

        for col_def in columns_def:
            col_name = col_def["name"]
            if col_name not in df.columns:
                continue

            validation = col_def.get("validation", {})
            pattern_str = validation.get("pattern")

            if not pattern_str:
                continue

            import re
            pattern = re.compile(pattern_str)

            # Check non-null values
            non_null = df[df[col_name].notna()]
            invalid = non_null[~non_null[col_name].astype(str).str.match(pattern)]

            if len(invalid) > 0:
                self.errors.append(
                    f"Column '{col_name}' has {len(invalid)} values not matching pattern '{pattern_str}'"
                )

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get validation report.

        Returns:
            Dict with validation results
        """
        return {
            "valid": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "schema_version": self.schema.get("schema", {}).get("version"),
        }
```

---

### 3.3 Data Loader with Schema Validation

**Path:** `src/meta/ad/miner/data/loader.py`

```python
"""Load and validate creative features data."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import logging

from src.meta.ad.miner.validation.input_validator import InputSchemaValidator

logger = logging.getLogger(__name__)


def load_features(
    csv_path: str | Path,
    validate: bool = True,
    fill_defaults: bool = True,
    add_derived_columns: bool = True
) -> pd.DataFrame:
    """
    Load creative features CSV with optional validation.

    Args:
        csv_path: Path to CSV file
        validate: If True, validate against schema before returning
        fill_defaults: If True, fill missing values with defaults
        add_derived_columns: If True, add computed columns

    Returns:
        DataFrame with features + metadata

    Raises:
        ValueError: If validation fails and validate=True
    """
    csv_path = Path(csv_path)

    # Validate if requested
    if validate:
        validator = InputSchemaValidator(csv_path)
        if not validator.validate():
            report = validator.get_validation_report()
            raise ValueError(
                f"Input validation failed:\n"
                f"  {report['error_count']} errors, {report['warning_count']} warnings\n"
                f"  Errors:\n" + "\n  ".join(report["errors"])
            )
        logger.info("✓ Input validation passed")

    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} creatives from {csv_path}")

    # Fill defaults for missing metadata
    if fill_defaults:
        df = _fill_metadata_defaults(df)
        logger.info("✓ Filled missing metadata with defaults")

    # Add derived columns
    if add_derived_columns:
        df = _add_derived_columns(df)
        logger.info("✓ Added derived columns")

    # Log summary
    _log_data_summary(df)

    return df


def _fill_metadata_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing metadata with default values."""
    defaults = {
        "campaign_goal": "unknown",
        "product": "unknown",
        "branch": "unknown",
    }

    for col, default_val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)

    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns for analysis."""
    # Add performance quartile
    if "roas" in df.columns:
        df["performance_quartile"] = pd.qcut(
            df["roas"],
            q=4,
            labels=["bottom", "mid-low", "mid-high", "top"],
            duplicates="drop"
        )

    # Add segment key for grouping
    df["segment_key"] = (
        df["product"].astype(str) + "|" +
        df["branch"].astype(str) + "|" +
        df["campaign_goal"].astype(str)
    )

    # Add metrics (if spend and impressions available)
    if "spend" in df.columns and "clicks" in df.columns:
        df["cpc"] = df["spend"] / df["clicks"].replace(0, pd.NA)

    if "impressions" in df.columns and "clicks" in df.columns:
        df["ctr"] = (df["clicks"] / df["impressions"]) * 100

    return df


def _log_data_summary(df: pd.DataFrame) -> None:
    """Log summary statistics."""
    logger.info("=" * 70)
    logger.info("DATA SUMMARY")
    logger.info("=" * 70)

    # Sample size
    logger.info(f"Total creatives: {len(df)}")

    # Segments
    if "segment_key" in df.columns:
        segments = df["segment_key"].nunique()
        logger.info(f"Unique segments: {segments}")

        # Segment sizes
        segment_counts = df.groupby("segment_key").size()
        logger.info(f"Segment sizes: Min={segment_counts.min()}, "
                   f"Max={segment_counts.max()}, "
                   f"Mean={segment_counts.mean():.1f}")

    # Goals
    if "campaign_goal" in df.columns:
        goals = df["campaign_goal"].value_counts()
        logger.info(f"Goals: {dict(goals)}")

    # Products
    if "product" in df.columns:
        products = df["product"].value_counts()
        logger.info(f"Products: {dict(products)}")

    # Branches
    if "branch" in df.columns:
        branches = df["branch"].value_counts()
        logger.info(f"Branches: {dict(branches)}")

    # ROAS stats
    if "roas" in df.columns:
        logger.info(f"ROAS: Min={df['roas'].min():.2f}, "
                   f"Max={df['roas'].max():.2f}, "
                   f"Mean={df['roas'].mean():.2f}, "
                   f"Median={df['roas'].median():.2f}")

    logger.info("=" * 70)


def get_segment_sizes(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get sample sizes for all segments.

    Returns:
        Dict mapping segment_key to sample size
    """
    if "segment_key" not in df.columns:
        return {"all": len(df)}

    return df.groupby("segment_key").size().to_dict()


def filter_to_segment(
    df: pd.DataFrame,
    product: Optional[str] = None,
    branch: Optional[str] = None,
    campaign_goal: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter DataFrame to specific segment.

    Args:
        df: Source DataFrame
        product: Product filter (None = all)
        branch: Branch filter (None = all)
        campaign_goal: Goal filter (None = all)

    Returns:
        Filtered DataFrame
    """
    result = df.copy()

    if product is not None and "product" in df.columns:
        result = result[result["product"] == product]

    if branch is not None and "branch" in df.columns:
        result = result[result["branch"] == branch]

    if campaign_goal is not None and "campaign_goal" in df.columns:
        result = result[result["campaign_goal"] == campaign_goal]

    return result
```

---

## 4. Output Schema Specification

### 4.1 Output Schema Definition

**Path:** `src/meta/ad/miner/schemas/output_schema.yaml`

```yaml
schema:
  version: "2.0"
  name: "mined_patterns"
  format: "JSON"
  encoding: "utf-8"

metadata:
  author: "Ad Miner Team"
  created_date: "2026-01-27"
  description: "Mined creative patterns with context metadata"

root:
  type: "object"
  required:
    - "metadata"
    - "patterns"
    - "anti_patterns"

  properties:
    # === METADATA SECTION ===
    metadata:
      type: "object"
      required: true
      description: "Information about the pattern mining analysis"

      properties:
        schema_version:
          type: "string"
          required: true
          description: "Schema version"
          example: "2.0"

        customer:
          type: "string"
          required: true
          description: "Customer/account name"
          example: "moprobo"

        product:
          type: "string"
          required: true
          description: "Product name (or 'all' for generic)"
          example: "Power Station"

        branch:
          type: "string"
          required: true
          description: "Branch/region (or 'all' for generic)"
          example: "US"

        campaign_goal:
          type: "string"
          required: true
          description: "Campaign objective (or 'all' for generic)"
          example: "conversion"

        granularity_level:
          type: "integer"
          required: true
          description: "Granularity level (1-4)"
          range: [1, 4]
          example: 1

        sample_size:
          type: "integer"
          required: true
          description: "Number of creatives analyzed"
          range: [1, null]
          example: 342

        min_threshold:
          type: "integer"
          required: true
          description: "Minimum sample size for this granularity level"
          example: 200

        analysis_date:
          type: "string"
          format: "date"
          required: true
          description: "When analysis was run (YYYY-MM-DD)"
          example: "2026-01-27"

        fallback_used:
          type: "boolean"
          required: true
          description: "Whether fallback to broader level was used"
          example: false

        fallback_level:
          type: "integer"
          required: false
          description: "If fallback used, which level was returned"
          range: [2, 4]
          example: 3

        data_quality:
          type: "object"
          required: false
          description: "Data quality metrics"

          properties:
            completeness_score:
              type: "float"
              range: [0.0, 1.0]
              description: "Completeness of feature data"
              example: 0.95

            avg_roas:
              type: "float"
              description: "Average ROAS in this segment"
              example: 2.34

            top_quartile_roas:
              type: "float"
              description: "Average ROAS in top quartile"
              example: 4.56

            bottom_quartile_roas:
              type: "float"
              description: "Average ROAS in bottom quartile"
              example: 0.98

            roas_range:
              type: "float"
              description: "Top quartile ROAS / bottom quartile ROAS"
              example: 4.65

            top_quartile_size:
              type: "integer"
              description: "Number of creatives in top quartile"
              example: 85

            bottom_quartile_size:
              type: "integer"
              description: "Number of creatives in bottom quartile"
              example: 85

    # === POSITIVE PATTERNS SECTION ===
    patterns:
      type: "array"
      required: true
      description: "Positive patterns (DOs) ranked by priority_score"
      min_items: 0

      items:
        type: "object"
        required:
          - "feature"
          - "value"
          - "pattern_type"
          - "confidence"
          - "roas_lift_multiple"
          - "roas_lift_pct"
          - "top_quartile_prevalence"
          - "priority_score"

        properties:
          feature:
            type: "string"
            required: true
            description: "Feature name (from input schema)"
            example: "product_position"

          current_value:
            type: "string"
            required: false
            description: "Current value in underperforming creatives"
            example: "center"

          value:
            type: "string"
            required: true
            description: "Recommended value (pattern to implement)"
            example: "bottom-right"

          pattern_type:
            type: "enum"
            required: true
            values: ["DO", "DO_CONVERSION", "DO_AWARENESS", "DO_TRAFFIC"]
            description: "Type of positive pattern"

          confidence:
            type: "enum"
            required: true
            values: ["high", "medium", "low"]
            description: "Confidence level based on statistical significance"

          roas_lift_multiple:
            type: "float"
            required: true
            range: [1.0, null]
            description: "ROAS multiple when using this value"
            example: 2.8

          roas_lift_pct:
            type: "float"
            required: true
            description: "Percentage ROAS lift (roas_lift_multiple - 1) * 100"
            example: 180.0

          top_quartile_prevalence:
            type: "float"
            required: true
            range: [0.0, 1.0]
            description: "Prevalence in top performers (0-1)"
            example: 0.67

          bottom_quartile_prevalence:
            type: "float"
            required: false
            range: [0.0, 1.0]
            description: "Prevalence in bottom performers (0-1)"
            example: 0.12

          prevalence_lift:
            type: "float"
            required: false
            description: "Prevalence difference (top - bottom)"
            example: 0.55

          goal_specific:
            type: "boolean"
            required: true
            description: "Whether this pattern is specific to campaign_goal"
            example: true

          product_specific:
            type: "boolean"
            required: true
            description: "Whether this pattern is specific to product"
            example: true

          branch_specific:
            type: "boolean"
            required: true
            description: "Whether this pattern is specific to branch"
            example: false

          reason:
            type: "string"
            required: true
            description: "Human-readable explanation"
            example: "For conversion campaigns with Power Station in US, products positioned bottom-right show 2.8x higher ROAS. Present in 67% of top performers vs 12% in bottom quartile."

          maps_to_template:
            type: "string"
            required: true
            description: "Template placeholder this maps to"
            example: "product_position"

          priority_score:
            type: "float"
            required: true
            range: [0.0, 10.0]
            description: "Priority score for ranking (higher = more important)"
            example: 9.5

          sample_count:
            type: "integer"
            required: false
            description: "Number of creatives with this value"
            example: 89

          statistical_significance:
            type: "object"
            required: false
            description: "Statistical test results"

            properties:
              chi_square_stat:
                type: "float"
                description: "Chi-square test statistic"
                example: 45.23

              p_value:
                type: "float"
                range: [0.0, 1.0]
                description: "P-value from chi-square test"
                example: 0.00001

              significant:
                type: "boolean"
                description: "Whether result is statistically significant (p < 0.05)"
                example: true

              confidence_interval:
                type: "object"
                description: "95% confidence interval for ROAS lift"

                properties:
                  lower:
                    type: "float"
                    example: 2.2

                  upper:
                    type: "float"
                    example: 3.4

          supporting_evidence:
            type: "array"
            required: false
            description: "Additional evidence for this pattern"

            items:
              type: "object"
              properties:
                type:
                  type: "enum"
                  values: ["example_creative", "statistical", "expert_review"]
                  example: "example_creative"

                description:
                  type: "string"
                  example: "Creative moprobo_meta_001 shows this pattern with ROAS 4.5"

                creative_id:
                  type: "string"
                  example: "moprobo_meta_20250115_001234"

                roas:
                  type: "float"
                  example: 4.5

    # === NEGATIVE PATTERNS SECTION ===
    anti_patterns:
      type: "array"
      required: true
      description: "Negative patterns (DON'Ts) to avoid"
      min_items: 0

      items:
        type: "object"
        required:
          - "feature"
          - "avoid_value"
          - "pattern_type"
          - "confidence"
          - "roas_penalty_multiple"
          - "roas_penalty_pct"
          - "bottom_quartile_prevalence"

        properties:
          feature:
            type: "string"
            required: true
            description: "Feature name"
            example: "product_position"

          avoid_value:
            type: "string"
            required: true
            description: "Value to avoid"
            example: "top-left"

          pattern_type:
            type: "enum"
            required: true
            values: ["DON'T", "ANTI_PATTERN"]
            description: "Type of negative pattern"

          confidence:
            type: "enum"
            required: true
            values: ["high", "medium", "low"]
            description: "Confidence level"

          roas_penalty_multiple:
            type: "float"
            required: true
            range: [0.0, 1.0]
            description: "ROAS multiple when using this value"
            example: 0.6

          roas_penalty_pct:
            type: "float"
            required: true
            description: "Percentage ROAS penalty (roas_penalty_multiple - 1) * 100"
            example: -40.0

          bottom_quartile_prevalence:
            type: "float"
            required: true
            range: [0.0, 1.0]
            description: "Prevalence in bottom performers"
            example: 0.65

          top_quartile_prevalence:
            type: "float"
            required: false
            range: [0.0, 1.0]
            description: "Prevalence in top performers"
            example: 0.15

          reason:
            type: "string"
            required: true
            description: "Explanation"
            example: "Used in 65% of worst performers, 40% lower ROAS than average"

          maps_to_template:
            type: "string"
            required: true
            description: "Template placeholder"
            example: "product_position"

          sample_count:
            type: "integer"
            required: false
            description: "Number of creatives with this value"
            example: 65

          statistical_significance:
            type: "object"
            required: false
            description: "Statistical test results"

    # === LOW-PRIORITY INSIGHTS SECTION ===
    low_priority_insights:
      type: "array"
      required: true
      description: "Minor trends worth watching but not acting on"
      min_items: 0

      items:
        type: "object"
        required:
          - "feature"
          - "value"
          - "confidence"
          - "roas_lift_multiple"
          - "reason"

        properties:
          feature:
            type: "string"
            required: true
            example: "contrast_level"

          value:
            type: "string"
            required: true
            example: "high"

          roas_lift_multiple:
            type: "float"
            required: true
            example: 1.05

          roas_lift_pct:
            type: "float"
            required: true
            example: 5.0

          confidence:
            type: "enum"
            required: true
            values: ["low"]
            example: "low"

          reason:
            type: "string"
            required: true
            example: "Slight positive trend (5% lift), but not statistically significant (p=0.15)"

          trend_direction:
            type: "enum"
            values: ["positive", "negative", "neutral"]
            example: "positive"

    # === GENERATION INSTRUCTIONS SECTION ===
    generation_instructions:
      type: "object"
      required: false
      description: "Instructions for ad generator"

      properties:
        must_include:
          type: "array"
          description: "Feature names that must be included"
          items:
            type: "string"
          example: ["product_position", "lighting_style"]

        prioritize:
          type: "array"
          description: "Feature names to prioritize"
          items:
            type: "string"
          example: ["visual_prominence", "color_balance"]

        avoid:
          type: "array"
          description: "Feature names to avoid (can use feature:value format)"
          items:
            type: "string"
          example: ["product_position:top-left", "lighting_style:natural"]

        min_coverage:
          type: "float"
          range: [0.0, 1.0]
          description: "Minimum feature coverage required"
          example: 0.8

        max_features:
          type: "integer"
          description: "Maximum number of features to include"
          example: 15
```

---

### 4.2 Output Validator Implementation

**Path:** `src/meta/ad/miner/validation/output_validator.py`

```python
"""Output schema validator for mined patterns JSON."""

from pathlib import Path
from typing import Any, Dict
import json
import logging

logger = logging.getLogger(__name__)


class OutputSchemaValidator:
    """Validate mined patterns JSON against schema."""

    SCHEMA_VERSION = "2.0"

    REQUIRED_TOP_LEVEL_KEYS = [
        "metadata",
        "patterns",
        "anti_patterns",
        "low_priority_insights",
    ]

    REQUIRED_METADATA_KEYS = [
        "schema_version",
        "customer",
        "product",
        "branch",
        "campaign_goal",
        "granularity_level",
        "sample_size",
        "analysis_date",
    ]

    # Valid enum values
    VALID_CONFIDENCE = ["high", "medium", "low"]
    VALID_PATTERN_TYPES = ["DO", "DO_CONVERSION", "DO_AWARENESS", "DO_TRAFFIC", "DON'T", "ANTI_PATTERN"]

    def __init__(self, json_path: str | Path):
        """
        Initialize validator.

        Args:
            json_path: Path to JSON file to validate
        """
        self.json_path = Path(json_path)
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> bool:
        """
        Validate JSON against schema.

        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []

        # Load JSON
        try:
            with open(self.json_path) as f:
                data = json.load(f)
        except Exception as e:
            self.errors.append(f"Failed to read JSON: {e}")
            return False

        # Run validation checks
        self._validate_top_level(data)
        self._validate_metadata(data.get("metadata", {}))
        self._validate_patterns(data.get("patterns", []))
        self._validate_anti_patterns(data.get("anti_patterns", []))
        self._validate_low_priority_insights(data.get("low_priority_insights", []))
        self._validate_ranges(data)
        self._validate_consistency(data)

        is_valid = len(self.errors) == 0

        if not is_valid:
            logger.error(f"Output validation failed with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  ✗ {error}")

        if self.warnings:
            logger.warning(f"Output validation produced {len(self.warnings)} warnings")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")

        return is_valid

    def _validate_top_level(self, data: Dict) -> None:
        """Validate top-level structure."""
        if not isinstance(data, dict):
            self.errors.append(f"Root must be object, got {type(data)}")
            return

        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            if key not in data:
                self.errors.append(f"Missing required top-level key: {key}")

    def _validate_metadata(self, metadata: Dict) -> None:
        """Validate metadata section."""
        if not isinstance(metadata, dict):
            self.errors.append("metadata must be object")
            return

        for key in self.REQUIRED_METADATA_KEYS:
            if key not in metadata:
                self.errors.append(f"Missing required metadata key: {key}")

        # Validate schema version
        if metadata.get("schema_version") != self.SCHEMA_VERSION:
            self.warnings.append(
                f"Schema version mismatch: expected {self.SCHEMA_VERSION}, "
                f"got {metadata.get('schema_version')}"
            )

        # Validate granularity level
        granularity = metadata.get("granularity_level")
        if granularity is not None:
            if not isinstance(granularity, int) or not (1 <= granularity <= 4):
                self.errors.append(f"granularity_level must be integer 1-4, got {granularity}")

        # Validate sample size
        sample_size = metadata.get("sample_size")
        if sample_size is not None:
            if not isinstance(sample_size, int) or sample_size <= 0:
                self.errors.append(f"sample_size must be positive integer, got {sample_size}")

        # Validate data quality if present
        data_quality = metadata.get("data_quality", {})
        if data_quality:
            completeness = data_quality.get("completeness_score")
            if completeness is not None:
                if not (0.0 <= completeness <= 1.0):
                    self.errors.append(f"completeness_score must be 0-1, got {completeness}")

    def _validate_patterns(self, patterns: list) -> None:
        """Validate patterns array."""
        if not isinstance(patterns, list):
            self.errors.append("patterns must be array")
            return

        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, dict):
                self.errors.append(f"Pattern {i}: must be object")
                continue

            # Required fields
            required_fields = [
                "feature", "value", "pattern_type", "confidence",
                "roas_lift_multiple", "roas_lift_pct",
                "top_quartile_prevalence", "priority_score"
            ]
            for field in required_fields:
                if field not in pattern:
                    self.errors.append(f"Pattern {i}: missing required field '{field}'")

            # Validate confidence
            confidence = pattern.get("confidence")
            if confidence not in self.VALID_CONFIDENCE:
                self.errors.append(f"Pattern {i}: invalid confidence '{confidence}'")

            # Validate pattern_type
            pattern_type = pattern.get("pattern_type")
            if pattern_type not in self.VALID_PATTERN_TYPES:
                self.errors.append(f"Pattern {i}: invalid pattern_type '{pattern_type}'")

            # Validate ranges
            roas_lift = pattern.get("roas_lift_multiple")
            if roas_lift is not None and roas_lift < 1.0:
                self.errors.append(f"Pattern {i}: roas_lift_multiple must be >= 1.0, got {roas_lift}")

            top_prev = pattern.get("top_quartile_prevalence")
            if top_prev is not None and not (0.0 <= top_prev <= 1.0):
                self.errors.append(f"Pattern {i}: top_quartile_prevalence must be 0-1, got {top_prev}")

            priority_score = pattern.get("priority_score")
            if priority_score is not None and not (0.0 <= priority_score <= 10.0):
                self.errors.append(f"Pattern {i}: priority_score must be 0-10, got {priority_score}")

    def _validate_anti_patterns(self, anti_patterns: list) -> None:
        """Validate anti_patterns array."""
        if not isinstance(anti_patterns, list):
            self.errors.append("anti_patterns must be array")
            return

        for i, pattern in enumerate(anti_patterns):
            if not isinstance(pattern, dict):
                self.errors.append(f"Anti-pattern {i}: must be object")
                continue

            required_fields = [
                "feature", "avoid_value", "pattern_type", "confidence",
                "roas_penalty_multiple", "roas_penalty_pct", "bottom_quartile_prevalence"
            ]
            for field in required_fields:
                if field not in pattern:
                    self.errors.append(f"Anti-pattern {i}: missing required field '{field}'")

            # Validate confidence
            confidence = pattern.get("confidence")
            if confidence not in self.VALID_CONFIDENCE:
                self.errors.append(f"Anti-pattern {i}: invalid confidence '{confidence}'")

            # Validate penalty
            penalty = pattern.get("roas_penalty_multiple")
            if penalty is not None and not (0.0 <= penalty <= 1.0):
                self.errors.append(f"Anti-pattern {i}: roas_penalty_multiple must be 0-1, got {penalty}")

    def _validate_low_priority_insights(self, insights: list) -> None:
        """Validate low_priority_insights array."""
        if not isinstance(insights, list):
            self.errors.append("low_priority_insights must be array")
            return

        for i, insight in enumerate(insights):
            if not isinstance(insight, dict):
                self.errors.append(f"Insight {i}: must be object")
                continue

            required_fields = ["feature", "value", "confidence", "roas_lift_multiple", "reason"]
            for field in required_fields:
                if field not in insight:
                    self.errors.append(f"Insight {i}: missing required field '{field}'")

            # Low priority insights should always have "low" confidence
            confidence = insight.get("confidence")
            if confidence != "low":
                self.warnings.append(f"Insight {i}: has '{confidence}' confidence, expected 'low'")

    def _validate_ranges(self, data: Dict) -> None:
        """Validate value ranges across document."""
        metadata = data.get("metadata", {})

        # Validate data quality ranges
        data_quality = metadata.get("data_quality", {})
        for key in ["avg_roas", "top_quartile_roas", "bottom_quartile_roas"]:
            value = data_quality.get(key)
            if value is not None and value < 0:
                self.errors.append(f"data_quality.{key} must be >= 0, got {value}")

    def _validate_consistency(self, data: Dict) -> None:
        """Validate internal consistency."""
        metadata = data.get("metadata", {})
        patterns = data.get("patterns", [])
        anti_patterns = data.get("anti_patterns", [])

        # Check that pattern count doesn't exceed sample size
        sample_size = metadata.get("sample_size", 0)
        total_patterns = len(patterns) + len(anti_patterns)

        if total_patterns > sample_size:
            self.warnings.append(
                f"Total patterns ({total_patterns}) exceeds sample size ({sample_size})"
            )

        # Check that priority scores are sorted (descending)
        priorities = [p.get("priority_score", 0) for p in patterns]
        if priorities != sorted(priorities, reverse=True):
            self.warnings.append("Patterns are not sorted by priority_score (descending)")

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get validation report.

        Returns:
            Dict with validation results
        """
        return {
            "valid": len(self.errors) == 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "schema_version": self.SCHEMA_VERSION,
        }
```

---

## 5. Path Structure Design

### 5.1 Complete Path Hierarchy

```
# ============================================
# INPUT DATA PATHS
# ============================================
data/
├── creative_features/
│   ├── moprobo_features.csv                 # All moprobo creatives
│   ├── moprobo_features_2026_01.csv         # Monthly snapshot
│   └── moprobo_features_2026_02.csv
│
├── creative_images/
│   ├── moprobo/
│   │   ├── meta/
│   │   │   ├── moprobo_meta_001.jpg
│   │   │   └── ...
│   │   └── taboola/
│   │       └── ...
│   └── customer_b/
│       └── ...
│
└── reference_data/
    ├── product_catalog.json                 # Product definitions
    └── branch_definitions.json              # Branch mappings

# ============================================
# CONFIG PATHS
# ============================================
config/ad/miner/
├── schemas/
│   ├── input_schema.yaml                    # Input CSV schema
│   └── output_schema.yaml                   # Output JSON schema
│
├── gpt4/
│   ├── features.yaml                        # GPT-4 feature definitions
│   └── prompts.yaml                         # GPT-4 prompt templates
│
├── customers/                               # Per-customer configuration
│   ├── moprobo/
│   │   ├── products.yaml                     # Product list
│   │   ├── branches.yaml                    # Branch list
│   │   └── goals.yaml                       # Goal definitions
│   └── customer_b/
│       └── ...
│
└── defaults/
    ├── mining_params.yaml                   # Default mining parameters
    └── thresholds.yaml                      # Default thresholds

# ============================================
# OUTPUT PATHS: MINED PATTERNS
# ============================================
config/ad/miner/mined_patterns/
├── moprobo/                                 # Customer
│   ├── segment_index.json                   # Index of all segments
│   │
│   ├── Power_Station/                       # Product (Level 1)
│   │   ├── US/                               # Branch (Level 1)
│   │   │   ├── conversion/                   # Goal (Level 1)
│   │   │   │   ├── patterns.json             # Primary: Mined patterns
│   │   │   │   ├── patterns.md               # Generated: Human-readable
│   │   │   │   ├── metadata.json             # Analysis metadata
│   │   │   │   └── statistics.json           # Statistical summary
│   │   │   │
│   │   │   ├── awareness/                    # Goal (Level 1)
│   │   │   │   └── patterns.json
│   │   │   │
│   │   │   └── traffic/                      # Goal (Level 1)
│   │   │       └── patterns.json
│   │   │
│   │   ├── EU/                               # Branch (Level 1)
│   │   │   ├── conversion/
│   │   │   │   └── patterns.json
│   │   │   └── awareness/
│   │   │       └── patterns.json
│   │   │
│   │   ├── conversion/                       # Goal (Level 2 fallback)
│   │   │   ├── patterns.json                 # Merged across branches
│   │   │   └── patterns.md
│   │   │
│   │   └── patterns.json                     # Product-level fallback (Level 2)
│   │
│   ├── MoProBo/                              # Product (Level 1)
│   │   ├── US/
│   │   │   ├── conversion/
│   │   │   │   └── patterns.json
│   │   │   └── awareness/
│   │   │       └── patterns.json
│   │   │
│   │   └── awareness/                       # Goal (Level 2 fallback)
│   │       └── patterns.json
│   │
│   ├── conversion/                          # Goal (Level 3 fallback)
│   │   ├── patterns.json                     # Merged across products
│   │   ├── patterns.md
│   │   └── metadata.json
│   │
│   ├── awareness/                           # Goal (Level 3 fallback)
│   │   └── patterns.json
│   │
│   ├── traffic/                              # Goal (Level 3 fallback)
│   │   └── patterns.json
│   │
│   └── patterns.json                         # Customer-level fallback (Level 4)
│
└── customer_b/
    └── ...

# ============================================
# INTERMEDIATE OUTPUT PATHS
# ============================================
cache/ad/miner/
├── moprobo/
│   ├── extracted_features/                   # Raw GPT-4 extractions
│   │   ├── batch_001.json
│   │   └── batch_002.json
│   │
│   ├── feature_matrices/                     # Processed feature matrices
│   │   ├── moprobo_meta_features.csv
│   │   └── moprobo_taboola_features.csv
│   │
│   └── checkpoints/                          # Analysis checkpoints
│       ├── Power_Station_US_conversion_checkpoint.json
│       └── ...

# ============================================
# RESULTS PATHS
# ============================================
results/ad/miner/
├── moprobo/
│   ├── 2026-01-27/                           # Analysis date
│   │   ├── analysis_report.md                # Human-readable report
│   │   ├── segment_summary.json              # Segment statistics
│   │   ├── quality_metrics.json              # Data quality scores
│   │   └── visualizations/                   # Charts and graphs
│   │       ├── roas_distribution.png
│   │       ├── pattern_heatmap.png
│   │       └── feature_importance.png
│   │
│   └── 2026-02-03/
│       └── ...
│
└── customer_b/
    └── ...

# ============================================
# LOG PATHS
# ============================================
logs/ad/miner/
├── moprobo/
│   ├── pattern_mining_20260127.log
│   ├── validation_errors_20260127.log
│   └── performance_20260127.log
└── customer_b/
```

---

### 5.2 Path Utility Implementation

**Path:** `src/meta/ad/miner/utils/paths.py`

```python
"""Path management for ad miner."""

from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MinerPaths:
    """
    Manage all paths for ad miner.

    Provides centralized path management with methods for:
    - Input data paths
    - Output paths (with granularity levels)
    - Cache paths
    - Results paths
    """

    def __init__(
        self,
        customer: str,
        product: Optional[str] = None,
        branch: Optional[str] = None,
        campaign_goal: Optional[str] = None,
        granularity_level: int = 1
    ):
        """
        Initialize paths.

        Args:
            customer: Customer/account name
            product: Optional product name
            branch: Optional branch name
            campaign_goal: Optional campaign goal
            granularity_level: Granularity level (1-4)
        """
        self.customer = customer.lower().replace(" ", "_")
        self.product = product.lower().replace(" ", "_") if product else None
        self.branch = branch.lower().replace(" ", "_") if branch else None
        self.campaign_goal = campaign_goal.lower().replace(" ", "_") if campaign_goal else None
        self.granularity_level = granularity_level

        # Base paths
        self.project_root = Path(__file__).resolve().parents[4]  # Up to project root
        self.data_dir = self.project_root / "data" / "creative_features"
        self.config_dir = self.project_root / "config" / "ad" / "miner"
        self.cache_dir = self.project_root / "cache" / "ad" / "miner"
        self.results_dir = self.project_root / "results" / "ad" / "miner"

    # ============================================
    # INPUT PATHS
    # ============================================

    def input_csv(self, filename: Optional[str] = None) -> Path:
        """
        Get input CSV path.

        Args:
            filename: Optional filename (defaults to {customer}_features.csv)

        Returns:
            Path to input CSV
        """
        if filename is None:
            filename = f"{self.customer}_features.csv"
        return self.data_dir / filename

    def images_dir(self, platform: str) -> Path:
        """
        Get directory for creative images.

        Args:
            platform: Platform name (meta, taboola, etc.)

        Returns:
            Path to images directory
        """
        return (
            self.project_root / "data" / "creative_images" /
            self.customer / platform
        )

    # ============================================
    # OUTPUT PATHS: MINED PATTERNS
    # ============================================

    def mined_patterns_dir(self) -> Path:
        """
        Get base directory for mined patterns.

        Returns:
            Path to mined_patterns directory
        """
        return self.config_dir / "mined_patterns" / self.customer

    def segment_patterns_dir(self) -> Path:
        """
        Get directory for specific segment patterns.

        Constructs path based on granularity_level:
        - Level 1: mined_patterns/{customer}/{product}/{branch}/{goal}/
        - Level 2: mined_patterns/{customer}/{product}/{goal}/
        - Level 3: mined_patterns/{customer}/{goal}/
        - Level 4: mined_patterns/{customer}/

        Returns:
            Path to segment-specific patterns directory
        """
        base = self.mined_patterns_dir()

        if self.granularity_level == 1 and self.product and self.branch and self.campaign_goal:
            # Level 1: All three dimensions
            return base / self.product / self.branch / self.campaign_goal

        elif self.granularity_level == 2 and self.product and self.campaign_goal:
            # Level 2: Product + Goal
            return base / self.product / self.campaign_goal

        elif self.granularity_level == 3 and self.campaign_goal:
            # Level 3: Goal only
            return base / self.campaign_goal

        else:
            # Level 4: Customer level
            return base

    def patterns_json(self) -> Path:
        """
        Get path to patterns.json (primary output).

        Returns:
            Path to patterns.json
        """
        return self.segment_patterns_dir() / "patterns.json"

    def patterns_md(self) -> Path:
        """
        Get path to patterns.md (generated human-readable).

        Returns:
            Path to patterns.md
        """
        return self.segment_patterns_dir() / "patterns.md"

    def patterns_metadata(self) -> Path:
        """
        Get path to patterns metadata file.

        Returns:
            Path to metadata.json
        """
        return self.segment_patterns_dir() / "metadata.json"

    def patterns_statistics(self) -> Path:
        """
        Get path to patterns statistics file.

        Returns:
            Path to statistics.json
        """
        return self.segment_patterns_dir() / "statistics.json"

    def segment_index(self) -> Path:
        """
        Get path to segment index file.

        Returns:
            Path to segment_index.json
        """
        return self.mined_patterns_dir() / "segment_index.json"

    # ============================================
    # CACHE PATHS
    # ============================================

    def extracted_features_cache(self, batch_id: str) -> Path:
        """
        Get path to extracted features cache.

        Args:
            batch_id: Batch identifier

        Returns:
            Path to cached features
        """
        return (
            self.cache_dir / self.customer / "extracted_features" / f"{batch_id}.json"
        )

    def feature_matrix_cache(self, platform: str) -> Path:
        """
        Get path to feature matrix cache.

        Args:
            platform: Platform name

        Returns:
            Path to cached feature matrix
        """
        return (
            self.cache_dir / self.customer / "feature_matrices" /
            f"{self.customer}_{platform}_features.csv"
        )

    def checkpoint_path(self, segment_key: str) -> Path:
        """
        Get path to checkpoint file.

        Args:
            segment_key: Segment identifier

        Returns:
            Path to checkpoint file
        """
        return (
            self.cache_dir / self.customer / "checkpoints" /
            f"{segment_key}_checkpoint.json"
        )

    # ============================================
    # RESULTS PATHS
    # ============================================

    def analysis_results_dir(self, date: str) -> Path:
        """
        Get directory for analysis results.

        Args:
            date: Analysis date (YYYY-MM-DD)

        Returns:
            Path to results directory
        """
        return self.results_dir / self.customer / date

    def analysis_report(self, date: str) -> Path:
        """
        Get path to analysis report.

        Args:
            date: Analysis date

        Returns:
            Path to report markdown
        """
        return self.analysis_results_dir(date) / "analysis_report.md"

    def segment_summary(self, date: str) -> Path:
        """
        Get path to segment summary.

        Args:
            date: Analysis date

        Returns:
            Path to segment summary JSON
        """
        return self.analysis_results_dir(date) / "segment_summary.json"

    def visualizations_dir(self, date: str) -> Path:
        """
        Get directory for visualization outputs.

        Args:
            date: Analysis date

        Returns:
            Path to visualizations directory
        """
        viz_dir = self.analysis_results_dir(date) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        return viz_dir

    # ============================================
    # UTILITY METHODS
    # ============================================

    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.data_dir,
            self.mined_patterns_dir(),
            self.segment_patterns_dir(),
            self.cache_dir / self.customer,
            self.cache_dir / self.customer / "extracted_features",
            self.cache_dir / self.customer / "feature_matrices",
            self.cache_dir / self.customer / "checkpoints",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Ensured {len(directories)} directories exist")

    def get_all_segment_paths(self) -> list[Path]:
        """
        Get all existing segment pattern paths.

        Returns:
            List of paths to patterns.json files
        """
        base = self.mined_patterns_dir()
        if not base.exists():
            return []

        return list(base.glob("**/patterns.json"))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MinerPaths(customer={self.customer}, "
            f"product={self.product}, branch={self.branch}, "
            f"goal={self.campaign_goal}, level={self.granularity_level})"
        )
```

---

### 5.3 Path Resolution Examples

```python
# Example 1: Level 1 (most specific)
paths = MinerPaths(
    customer="moprobo",
    product="Power Station",
    branch="US",
    campaign_goal="conversion",
    granularity_level=1
)
# patterns.json → config/ad/miner/mined_patterns/moprobo/Power_Station/US/conversion/patterns.json

# Example 2: Level 2 (product + goal)
paths = MinerPaths(
    customer="moprobo",
    product="Power Station",
    branch=None,
    campaign_goal="conversion",
    granularity_level=2
)
# patterns.json → config/ad/miner/mined_patterns/moprobo/Power_Station/conversion/patterns.json

# Example 3: Level 3 (goal only)
paths = MinerPaths(
    customer="moprobo",
    product=None,
    branch=None,
    campaign_goal="conversion",
    granularity_level=3
)
# patterns.json → config/ad/miner/mined_patterns/moprobo/conversion/patterns.json

# Example 4: Level 4 (customer fallback)
paths = MinerPaths(
    customer="moprobo",
    product=None,
    branch=None,
    campaign_goal=None,
    granularity_level=4
)
# patterns.json → config/ad/miner/mined_patterns/moprobo/patterns.json
```

---

## 6. Code Refactoring Plan

### 6.1 Rename Mapping Summary

**Directory rename:**
```bash
src/meta/ad/miner/recommendations/
  → src/meta/ad/miner/patterns/
```

**File renames:**
```bash
# Core pattern mining
rule_engine.py
  → pattern_miner.py

# I/O
md_io.py
  → json_io.py  # Now handles JSON + generates MD

# Evidence & formatters
evidence_builder.py
  → pattern_evidence.py

formatters.py
  → pattern_formatters.py

# Tests
test_rule_engine_gen.py
  → test_pattern_miner.py

test_md_io.py
  → test_json_io.py
```

**Function/class renames:**
```python
# PatternMiner class
class RuleEngine → class PatternMiner

# Mining functions
generate_recommendations() → mine_patterns()
load_recommendations() → load_patterns()
export_recommendations() → export_patterns()

# Pattern types
RecommendationType → PatternType
RECOMMENDATION_DO → PATTERN_DO
RECOMMENDATION_DONT → PATTERN_DONT
```

---

### 6.2 File-by-File Refactoring

#### File 1: `rule_engine.py` → `pattern_miner.py`

**Old structure:**
```python
class RuleEngine:
    def generate_recommendations(self, creative)
    def load_patterns(self, recommendations)
```

**New structure:**
```python
class PatternMiner:
    """Mine statistical patterns from creative data."""

    TOP_PCT = 0.25
    BOTTOM_PCT = 0.25
    LIFT_MIN = 1.5
    PREVALENCE_MIN = 0.10

    def mine_patterns(self, creative: Dict) -> List[Dict]:
        """
        Mine patterns from a single creative.

        Returns:
            List of pattern dictionaries
        """
        current_roas = creative.get("roas", 0)
        patterns = []

        # Check against high-confidence patterns
        for pattern in self.patterns:
            if not self._has_feature(creative, pattern):
                self._add_improvement_pattern(creative, pattern, patterns)

        # Check anti-patterns
        for anti_pattern in self.anti_patterns:
            if self._has_anti_pattern(creative, anti_pattern):
                self._add_anti_pattern(creative, anti_pattern, patterns)

        return patterns

    def mine_patterns_from_df(
        self,
        df: pd.DataFrame,
        top_pct: float = None,
        bottom_pct: float = None
    ) -> List[Dict]:
        """
        Mine patterns from DataFrame.

        Args:
            df: DataFrame with features
            top_pct: Top quartile threshold
            bottom_pct: Bottom quartile threshold

        Returns:
            List of pattern dictionaries
        """
        top_pct = top_pct or self.TOP_PCT
        bottom_pct = bottom_pct or self.BOTTOM_PCT

        # Split by performance
        top_threshold = df["roas"].quantile(1 - top_pct)
        bottom_threshold = df["roas"].quantile(bottom_pct)

        top_df = df[df["roas"] >= top_threshold]
        bottom_df = df[df["roas"] <= bottom_threshold]

        # Analyze each feature
        patterns = []
        for feature in self._get_feature_columns(df):
            feature_patterns = self._analyze_feature(
                feature, top_df, bottom_df
            )
            patterns.extend(feature_patterns)

        # Sort by ROAS lift
        patterns.sort(key=lambda p: p["roas_lift_multiple"], reverse=True)

        return patterns

    def _analyze_feature(
        self,
        feature: str,
        top_df: pd.DataFrame,
        bottom_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Analyze a single feature for patterns.

        Returns:
            List of patterns for this feature
        """
        patterns = []

        # Get unique values
        all_values = set(top_df[feature].unique()) | set(bottom_df[feature].unique())

        for value in all_values:
            # Calculate prevalence
            top_count = (top_df[feature] == value).sum()
            bottom_count = (bottom_df[feature] == value).sum()
            top_total = len(top_df)
            bottom_total = len(bottom_df)

            top_prevalence = top_count / top_total if top_total > 0 else 0
            bottom_prevalence = bottom_count / bottom_total if bottom_total > 0 else 0

            # Calculate ROAS lift
            top_roas = top_df[top_df[feature] == value]["roas"].mean()
            bottom_roas = bottom_df[bottom_df[feature] == value]["roas"].mean()
            roas_lift = top_roas / bottom_roas if bottom_roas > 0 else 1.0

            # Determine if this is a pattern
            if (roas_lift >= self.LIFT_MIN and
                top_prevalence >= self.PREVALENCE_MIN and
                top_prevalence > bottom_prevalence):

                # Calculate priority score
                priority_score = self._calculate_priority_score(
                    roas_lift, top_prevalence, top_count
                )

                # Determine confidence
                confidence = self._determine_confidence(
                    roas_lift, top_prevalence, top_count
                )

                pattern = {
                    "feature": feature,
                    "value": value,
                    "pattern_type": "DO",
                    "confidence": confidence,
                    "roas_lift_multiple": roas_lift,
                    "roas_lift_pct": (roas_lift - 1) * 100,
                    "top_quartile_prevalence": top_prevalence,
                    "bottom_quartile_prevalence": bottom_prevalence,
                    "prevalence_lift": top_prevalence - bottom_prevalence,
                    "priority_score": priority_score,
                    "sample_count": top_count,
                    "top_quartile_roas": top_roas,
                    "bottom_quartile_roas": bottom_roas,
                }

                # Add statistical significance
                pattern["statistical_significance"] = self._calculate_significance(
                    top_count, bottom_count, top_total, bottom_total
                )

                patterns.append(pattern)

        return patterns

    def _calculate_priority_score(
        self,
        roas_lift: float,
        prevalence: float,
        sample_count: int
    ) -> float:
        """
        Calculate priority score for ranking.

        Score = roas_lift * 3.0 + prevalence * 2.0 + log(sample_count) * 1.0

        Returns:
            Priority score (0-10)
        """
        import math

        score = (
            roas_lift * 3.0 +
            prevalence * 2.0 +
            math.log(sample_count + 1) * 1.0
        )

        # Normalize to 0-10 range
        return min(max(score, 0.0), 10.0)

    def _determine_confidence(
        self,
        roas_lift: float,
        prevalence: float,
        sample_count: int
    ) -> str:
        """
        Determine confidence level.

        Returns:
            "high", "medium", or "low"
        """
        if (roas_lift >= 2.5 and prevalence >= 0.5 and sample_count >= 30):
            return "high"
        elif (roas_lift >= 2.0 or prevalence >= 0.3):
            return "medium"
        else:
            return "low"

    def _calculate_significance(
        self,
        top_count: int,
        bottom_count: int,
        top_total: int,
        bottom_total: int
    ) -> Dict:
        """
        Calculate statistical significance using chi-square test.

        Returns:
            Dict with chi_square_stat, p_value, significant
        """
        from scipy.stats import chi2_contingency

        # Create contingency table
        table = [
            [top_count, top_total - top_count],
            [bottom_count, bottom_total - bottom_count]
        ]

        # Chi-square test
        chi2, p_value, _, _ = chi2_contingency(table)

        return {
            "chi_square_stat": chi2,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
```

---

#### File 2: `md_io.py` → `json_io.py`

**Path:** `src/meta/ad/miner/patterns/json_io.py`

```python
"""JSON I/O for mined patterns."""

from pathlib import Path
from typing import Any, Dict
import json
import logging

from src.meta.ad.miner.patterns.markdown_generator import generate_markdown

logger = logging.getLogger(__name__)


def export_patterns_json(
    patterns_data: Dict[str, Any],
    output_path: str | Path,
    validate: bool = True,
    format_json: bool = True,
    generate_md: bool = True
) -> None:
    """
    Export patterns to JSON file.

    Args:
        patterns_data: Dict with patterns, anti_patterns, etc.
        output_path: Path to output JSON file
        validate: If True, validate before writing
        format_json: If True, format with indentation
        generate_md: If True, also generate .md file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate if requested
    if validate:
        from src.meta.ad.miner.validation.output_validator import OutputSchemaValidator

        # Write to temp file for validation
        temp_path = output_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(patterns_data, f, indent=2 if format_json else None)

        validator = OutputSchemaValidator(temp_path)
        if not validator.validate():
            report = validator.get_validation_report()
            raise ValueError(
                f"Output validation failed:\n"
                f"{report['error_count']} errors, {report['warning_count']} warnings\n"
                f"Errors: {report['errors']}"
            )

        logger.info("✓ Output validation passed")

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(patterns_data, f, indent=2 if format_json else None)

    logger.info(f"✓ Wrote patterns to {output_path}")

    # Generate MD if requested
    if generate_md:
        md_path = output_path.with_suffix(".md")
        md_content = generate_markdown(patterns_data)
        with open(md_path, "w") as f:
            f.write(md_content)
        logger.info(f"✓ Generated markdown at {md_path}")


def load_patterns_file(
    path: str | Path
) -> Dict[str, Any]:
    """
    Load patterns from JSON file.

    Args:
        path: Path to patterns.json file

    Returns:
        Dict with patterns, anti_patterns, etc.

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Patterns file not found: {path}")

    # Try JSON first
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)

        # Validate
        from src.meta.ad.miner.validation.output_validator import OutputSchemaValidator
        validator = OutputSchemaValidator(path)
        if not validator.validate():
            logger.warning(f"Patterns file validation failed: {path}")

        return data

    # Fallback to MD (legacy format)
    elif path.suffix == ".md":
        logger.warning(f"Loading legacy MD format from {path}")
        return _load_patterns_from_md(path)

    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def _load_patterns_from_md(path: Path) -> Dict[str, Any]:
    """
    Load patterns from legacy MD format.

    Args:
        path: Path to .md file

    Returns:
        Dict in v2.0 format
    """
    # Parse MD and convert to v2.0 format
    # Implementation would go here
    raise NotImplementedError("MD loading not yet implemented")


def update_segment_index(
    customer: str,
    segment_info: Dict[str, Any],
    index_path: str | Path = None
) -> None:
    """
    Update segment index with new segment.

    Args:
        customer: Customer name
        segment_info: Dict with product, branch, goal, granularity, etc.
        index_path: Optional custom index path
    """
    if index_path is None:
        from src.meta.ad.miner.utils.paths import MinerPaths
        paths = MinerPaths(customer=customer)
        index_path = paths.segment_index()
    else:
        index_path = Path(index_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing index
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {
            "customer": customer,
            "schema_version": "2.0",
            "last_updated": "",
            "total_segments": 0,
            "segments": []
        }

    # Check if segment already exists
    segment_key = (
        f"{segment_info['product']}/"
        f"{segment_info['branch']}/"
        f"{segment_info['campaign_goal']}"
    )

    # Update or add segment
    existing = next(
        (s for s in index["segments"] if s["segment_key"] == segment_key),
        None
    )

    if existing:
        existing.update(segment_info)
    else:
        segment_info["segment_key"] = segment_key
        index["segments"].append(segment_info)
        index["total_segments"] += 1

    # Update timestamp
    from datetime import datetime
    index["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    # Write index
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"✓ Updated segment index: {segment_key}")
```

---

#### File 3: Markdown Generator (New)

**Path:** `src/meta/ad/miner/patterns/markdown_generator.py`

```python
"""Generate human-readable markdown from mined patterns JSON."""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def generate_markdown(patterns_data: Dict[str, Any]) -> str:
    """
    Generate human-readable markdown from patterns JSON.

    Args:
        patterns_data: Patterns dict (JSON format)

    Returns:
        Markdown string
    """
    metadata = patterns_data.get("metadata", {})
    patterns = patterns_data.get("patterns", [])
    anti_patterns = patterns_data.get("anti_patterns", [])
    insights = patterns_data.get("low_priority_insights", [])

    md_lines = []

    # Header
    md_lines.extend(_generate_header(metadata))

    # Data quality
    if metadata.get("data_quality"):
        md_lines.extend(_generate_data_quality(metadata["data_quality"]))

    # High-impact patterns
    if patterns:
        md_lines.extend(_generate_patterns_section(patterns, metadata))

    # Anti-patterns
    if anti_patterns:
        md_lines.extend(_generate_anti_patterns_section(anti_patterns, metadata))

    # Low-priority insights
    if insights:
        md_lines.extend(_generate_insights_section(insights))

    # Generation instructions
    if patterns_data.get("generation_instructions"):
        md_lines.extend(_generate_instructions_section(
            patterns_data["generation_instructions"]
        ))

    return "\n".join(md_lines)


def _generate_header(metadata: Dict) -> list:
    """Generate markdown header."""
    return [
        "# Mined Creative Patterns",
        "",
        f"**Customer:** {metadata.get('customer')} | "
        f"**Product:** {metadata.get('product')} | "
        f"**Branch:** {metadata.get('branch')} | "
        f"**Goal:** {metadata.get('campaign_goal')}",
        "",
        f"**Analysis Date:** {metadata.get('analysis_date')} | "
        f"**Sample:** {metadata.get('sample_size')} creatives | "
        f"**Granularity:** Level {metadata.get('granularity_level')}",
        "",
    ]


def _generate_data_quality(data_quality: Dict) -> list:
    """Generate data quality section."""
    lines = [
        "## 📊 Data Quality",
        "",
        f"- **Avg ROAS:** {data_quality.get('avg_roas', 0):.2f}",
        f"- **Top Quartile ROAS:** {data_quality.get('top_quartile_roas', 0):.2f}",
        f"- **Bottom Quartile ROAS:** {data_quality.get('bottom_quartile_roas', 0):.2f}",
        f"- **ROAS Range:** {data_quality.get('roas_range', 0):.2f}x",
        "",
    ]
    return lines


def _generate_patterns_section(patterns: list, metadata: Dict) -> list:
    """Generate patterns section."""
    lines = [
        "## 🎯 High-Impact Patterns (Priority Order)",
        "",
        "*Implement these patterns first for maximum ROAS lift*",
        "",
    ]

    for i, pattern in enumerate(patterns, 1):
        lines.extend(_generate_pattern_item(i, pattern, metadata))

    return lines


def _generate_pattern_item(index: int, pattern: Dict, metadata: Dict) -> list:
    """Generate single pattern item."""
    lines = [
        f"### {index}. {pattern['feature'].title()}: `{pattern['value']}`",
        "",
        f"- **Impact:** +{pattern['roas_lift_pct']:.0f}% ROAS "
         f"({pattern['roas_lift_multiple']:.1f}x lift)",
        f"- **Evidence:** Used in {pattern['top_quartile_prevalence']:.0%} "
         f"of top performers",
        f"- **Confidence:** {pattern['confidence'].title()}",
    ]

    # Add context flags
    flags = []
    if pattern.get("goal_specific"):
        flags.append(f"{metadata.get('campaign_goal').title()}-specific")
    if pattern.get("product_specific"):
        flags.append(f"{metadata.get('product').title()}-specific")
    if pattern.get("branch_specific"):
        flags.append(f"{metadata.get('branch').title()}-specific")

    if flags:
        lines.append(f"- **Context:** {', '.join(flags)}")

    lines.extend([
        "",
        f"**Why:** {pattern['reason']}",
        "",
    ])

    return lines


def _generate_anti_patterns_section(anti_patterns: list, metadata: Dict) -> list:
    """Generate anti-patterns section."""
    lines = [
        "## ⚠️ Anti-Patterns to Avoid",
        "",
        "*These patterns consistently underperform*",
        "",
    ]

    for i, pattern in enumerate(anti_patterns, 1):
        lines.extend([
            f"### {i}. {pattern['feature'].title()}: Avoid `{pattern['avoid_value']}`",
            "",
            f"- **Penalty:** {pattern['roas_penalty_pct']:.0f}% ROAS "
             f"({pattern['roas_penalty_multiple']:.1f}x of average)",
            f"- **Evidence:** Used in {pattern['bottom_quartile_prevalence']:.0%} "
             f"of worst performers",
            f"- **Confidence:** {pattern['confidence'].title()}",
            "",
            f"**Why:** {pattern['reason']}",
            "",
        ])

    return lines


def _generate_insights_section(insights: list) -> list:
    """Generate low-priority insights section."""
    lines = [
        "## 📊 Low-Priority Insights",
        "",
        "*Minor trends worth watching but not acting on yet*",
        "",
    ]

    for insight in insights:
        lines.extend([
            f"- **{insight['feature'].title()}:** `{insight['value']}`",
            f"  - Impact: +{insight['roas_lift_pct']:.0f}% ROAS (trend, not conclusive)",
            f"  - {insight['reason']}",
            "",
        ])

    return lines


def _generate_instructions_section(instructions: Dict) -> list:
    """Generate generation instructions section."""
    lines = [
        "---",
        "",
        "## 🤖 For Ad Generator",
        "",
        f"**Must Include:** " + ", ".join(instructions.get("must_include", [])),
        "",
        f"**Prioritize:** " + ", ".join(instructions.get("prioritize", [])),
        "",
        f"**Avoid:** " + ", ".join(instructions.get("avoid", [])),
        "",
    ]
    return lines
```

---

## 7. Implementation Phases (Detailed)

### Phase 1: Schema Definition (Week 1)

**Tasks:**
1. ⏳ Create schema YAML files
   - `src/meta/ad/miner/schemas/input_schema.yaml`
   - `src/meta/ad/miner/schemas/output_schema.yaml`

2. ⏳ Implement validators
   - `src/meta/ad/miner/validation/input_validator.py`
   - `src/meta/ad/miner/validation/output_validator.py`

3. ⏳ Write validator tests
   - `tests/unit/ad/miner/validation/test_input_validator.py`
   - `tests/unit/ad/miner/validation/test_output_validator.py`

4. ⏳ Update path utilities
   - `src/meta/ad/miner/utils/paths.py` (new structured paths)

**Deliverables:**
- ✓ Schema definitions in YAML
- ✓ Validators working
- ✓ Path utility class
- ✓ Test coverage >90%

---

### Phase 2: Rename to "Pattern" Terminology (Week 1-2)

**Tasks:**
1. ⏳ Rename directories
   ```bash
   git mv src/meta/ad/miner/recommendations src/meta/ad/miner/patterns
   git mv tests/unit/ad/miner/test_rule_engine_gen.py tests/unit/ad/miner/test_pattern_miner.py
   ```

2. ⏳ Rename and refactor files
   - `rule_engine.py` → `pattern_miner.py`
   - `md_io.py` → `json_io.py`
   - Update all imports

3. ⏳ Update class names
   - `RuleEngine` → `PatternMiner`
   - Update all references

4. ⏳ Update function names
   - `generate_recommendations()` → `mine_patterns()`
   - `load_recommendations()` → `load_patterns()`
   - Update all call sites

5. ⏳ Update variable names
   - `recommendation` → `pattern`
   - `recommendations` → `patterns`
   - Global search and replace

**Deliverables:**
- ✓ All files renamed
- ✓ All code updated
- ✓ All tests updated
- ✓ Documentation updated

---

### Phase 3: Implement New Data Loader (Week 2)

**Tasks:**
1. ⏳ Create data loader
   - `src/meta/ad/miner/data/loader.py`
   - Load with validation
   - Fill defaults
   - Add derived columns

2. ⏳ Update feature extraction
   - Add metadata columns to GPT-4 extraction
   - Backfill script for existing data

3. ⏳ Add derived columns
   - `performance_quartile`
   - `segment_key`
   - `cpc`, `ctr` (if spend/impressions available)

**Deliverables:**
- ✓ Data loader working
- ✓ Feature extraction updated
- ✓ Backfill script created
- ✓ Data validation working

---

### Phase 4: Implement Pattern Mining (Week 2-3)

**Tasks:**
1. ⏳ Refactor PatternMiner class
   - Rename from RuleEngine
   - Implement `mine_patterns_from_df()`
   - Add statistical significance testing
   - Add priority scoring

2. ⏳ Implement segmentation
   - `SegmentAnalyzer` class
   - Granularity selection
   - Context-aware pattern detection

3. ⏳ Update pattern types
   - `DO`, `DO_CONVERSION`, `DO_AWARENESS`
   - `DON'T`, `ANTI_PATTERN`

**Deliverables:**
- ✓ PatternMiner working
- ✓ Segmentation working
- ✓ Statistical testing working
- ✓ Context-aware detection working

---

### Phase 5: Implement JSON Output (Week 3-4)

**Tasks:**
1. ⏳ Implement JSON I/O
   - `json_io.py` (renamed from md_io.py)
   - Export to JSON format
   - Validate before writing

2. ⏳ Implement markdown generator
   - `markdown_generator.py`
   - Generate MD from JSON
   - Clean, non-redundant structure

3. ⏳ Implement segment index
   - Track all segments
   - Update index on write
   - Fast lookup

4. ⏳ Update CLI
   - `python run.py mine-patterns --product X --branch Y --goal Z`
   - Support context parameters

**Deliverables:**
- ✓ JSON output working
- ✓ MD generation working
- ✓ Segment index working
- ✓ CLI updated

---

### Phase 6: Implement Query API (Week 4-5)

**Tasks:**
1. ⏳ Create query module
   - `src/meta/ad/miner/queries.py`
   - `load_patterns_with_fallback()`
   - Automatic granularity selection

2. ⏳ Update ad generator integration
   - Update `ad_miner_adapter.py`
   - Use new query API
   - Handle v2.0 format

3. ⏳ Add fallback logging
   - Log when fallback occurs
   - Warn users about data limitations

**Deliverables:**
- ✓ Query API working
- ✓ Fallback logic working
- ✓ Ad generator integration updated
- ✓ Logging working

---

### Phase 7: Fix Feature Mapping (Week 5)

**Tasks:**
1. ⏳ Audit feature mapping
   - Identify unmapped features
   - Document mapping gaps

2. ⏳ Create new mappings
   - Add to `recommendation_mapping.py`
   - Handle list values
   - Handle complex values

3. ⏳ Expand template placeholders
   - Add `text_overlay` placeholder
   - Add `cta_style` placeholder
   - Add `color_palette` placeholder

4. ⏳ Test 100% coverage
   - Ensure all features map
   - Update transformers

**Deliverables:**
- ✓ Feature mapping complete
- ✓ 100% coverage achieved
- ✓ Templates updated
- ✓ Tests passing

---

### Phase 8: Testing & Validation (Week 6)

**Tasks:**
1. ⏳ Unit tests
   - Validator tests
   - Pattern mining tests
   - Segmentation tests
   - Path utility tests

2. ⏳ Integration tests
   - End-to-end pattern mining
   - Query with fallback
   - Feature mapping

3. ⏳ Performance tests
   - Large datasets
   - Query latency

**Deliverables:**
- ✓ Test suite complete
- ✓ >80% coverage
- ✓ All tests passing
- ✓ Performance benchmarks met

---

### Phase 9: Documentation (Week 6-7)

**Tasks:**
1. ⏳ Update code documentation
   - Docstrings for all classes
   - Type hints everywhere
   - Usage examples

2. ⏳ Create user documentation
   - Schema documentation
   - Path structure guide
   - Migration guide

3. ⏳ Create tutorials
   - Getting started
   - Advanced segmentation
   - Custom pattern detection

**Deliverables:**
- ✓ Documentation complete
- ✓ Examples working
- ✓ Tutorials tested

---

### Phase 10: Rollout (Week 7-8)

**Tasks:**
1. ⏳ Canary deployment
2. ⏳ Gradual rollout
3. ⏳ Monitor metrics
4. ⏳ Collect feedback

**Deliverables:**
- ✓ Successful deployment
- ✓ Metrics monitored
- ✓ Feedback collected

---

## 8. Summary

This detailed improvement plan provides:

1. ✅ **Structured schemas** for input (CSV) and output (JSON)
2. ✅ **Organized path structure** with clear hierarchy
3. ✅ **Complete refactoring plan**: recommendation → pattern
4. ✅ **File-by-file changes** with code examples
5. ✅ **10-phase implementation** with specific tasks
6. ✅ **Validation framework** for data quality
7. ✅ **Backward compatibility** strategy

**Ready for implementation!**

---

## Appendix: Quick Reference

### File Structure Summary

```
src/meta/ad/miner/
├── schemas/
│   ├── input_schema.yaml          (NEW)
│   └── output_schema.yaml         (NEW)
├── validation/
│   ├── input_validator.py         (NEW)
│   └── output_validator.py        (NEW)
├── patterns/                      (RENAMED from recommendations/)
│   ├── pattern_miner.py           (RENAMED from rule_engine.py)
│   ├── json_io.py                (RENAMED from md_io.py)
│   ├── markdown_generator.py     (NEW)
│   ├── pattern_evidence.py        (RENAMED from evidence_builder.py)
│   └── pattern_formatters.py     (RENAMED from formatters.py)
├── data/
│   └── loader.py                  (NEW)
├── queries.py                     (NEW)
├── segmentation.py                (NEW)
└── utils/
    ├── paths.py                   (UPDATED)
    └── compatibility.py           (NEW)
```

### Naming Convention Summary

| Old | New |
|-----|-----|
| `recommendation` | `pattern` |
| `recommendations` | `patterns` |
| `RuleEngine` | `PatternMiner` |
| `generate_recommendations()` | `mine_patterns()` |
| `load_recommendations()` | `load_patterns()` |
| `export_recommendations()` | `export_patterns()` |
| `recommendations.md` | `patterns.md` |
| `recommendations.json` | `patterns.json` |

### Path Convention Summary

**Input:**
- Data: `data/creative_features/{customer}_features.csv`
- Images: `data/creative_images/{customer}/{platform}/`

**Output:**
- Patterns: `config/ad/miner/mined_patterns/{customer}/{product}/{branch}/{goal}/`
- Index: `config/ad/miner/mined_patterns/{customer}/segment_index.json`
- Cache: `cache/ad/miner/{customer}/`
- Results: `results/ad/miner/{customer}/{date}/`
