# Input/Output Schema Design for Context-Aware Ad Miner

**Author:** Claude
**Date:** 2026-01-27
**Status:** Detailed Schema Specification
**Branch:** ad-reviewer

---

## Overview

This document defines **structured schemas** for both input (creative features with metadata) and output (context-aware recommendations) to support the multi-level granularity system.

**Design principles:**
1. **Self-describing**: Schema includes metadata, validation rules, type information
2. **Extensible**: Easy to add new metadata fields without breaking existing code
3. **Validatable**: Can validate against schema before processing
4. **Backward compatible**: Existing data can be migrated with defaults
5. **Queryable**: Efficient filtering and aggregation by metadata dimensions

---

## Part 1: Input Schema

### 1.1 CSV Schema Definition

**File:** `data/creative_features_with_metadata_{customer}.csv`

**Schema version:** `1.0`

```yaml
schema:
  version: "1.0"
  name: "creative_features_with_metadata"
  description: "Creative features + campaign metadata for pattern mining"
  format: "CSV"

# Required columns (must be present)
required_columns:
  # Identifiers
  - name: "creative_id"
    type: "string"
    description: "Unique creative identifier"
    example: "moprobo_meta_20250115_001234"

  - name: "filename"
    type: "string"
    description: "Image filename"
    example: "moprobo_meta_001.jpg"

  # Performance metrics
  - name: "roas"
    type: "float"
    description: "Return on ad spend"
    range: [0, null]
    example: 2.45

  # Metadata (NEW - required for segmentation)
  - name: "campaign_goal"
    type: "enum"
    values: ["awareness", "conversion", "traffic", "lead_generation", "app_installs", "unknown"]
    description: "Primary campaign objective"
    example: "conversion"
    default: "unknown"

  - name: "product"
    type: "string"
    description: "Product being advertised"
    example: "Power Station"
    default: "unknown"

  - name: "branch"
    type: "enum"
    values: ["US", "EU", "APAC", "LATAM", "Global", "unknown"]
    description: "Regional/organizational branch"
    example: "US"
    default: "unknown"

  - name: "campaign_id"
    type: "string"
    description: "Campaign identifier (optional, for grouping)"
    example: "moprobo_conversion_2025_01"
    required: false

  - name: "adset_id"
    type: "string"
    description: "Adset identifier (optional, for grouping)"
    example: "moprobo_conversion_25-34_us"
    required: false

# Optional columns (feature columns)
optional_columns:
  # Visual features (extracted by GPT-4 Vision)
  - name: "direction"
    type: "enum"
    values: ["front", "side", "overhead", "45-degree", "unknown"]
    description: "Camera angle relative to product"

  - name: "lighting_style"
    type: "enum"
    values: ["studio", "natural", "artificial", "unknown"]
    description: "Lighting setup style"

  - name: "primary_colors"
    type: "list"
    item_type: "string"
    description: "Primary colors in image (comma-separated)"
    example: "green, white, gray"

  - name: "product_position"
    type: "enum"
    values: ["left", "center", "right", "top-left", "top-right", "bottom-left", "bottom-right", "unknown"]
    description: "Product position in frame"

  # ... (all 29 features from feature extraction)

# Validation rules
validation:
  - rule: "roas >= 0"
    message: "ROAS cannot be negative"

  - rule: "campaign_goal in valid_goals"
    message: "Invalid campaign_goal"

  - rule: "filename not null"
    message: "Filename is required"

  - rule: "creative_id is unique"
    message: "Duplicate creative_id found"

# Metadata columns (for segmentation)
segmentation_keys:
  - "campaign_goal"
  - "product"
  - "branch"
```

---

### 1.2 CSV Example Data

**File:** `data/creative_features_with_metadata_moprobo.csv`

```csv
creative_id,filename,roas,campaign_goal,product,branch,campaign_id,adset_id,direction,lighting_style,primary_colors,product_position,lighting_type,human_elements,product_visibility,visual_prominence,color_balance,temperature,context_richness,product_context,relationship_depiction,visual_flow,composition_style,depth_layers,contrast_level,color_saturation,color_vibrancy,background_content_type,mood_lighting,emotional_tone,activity_level,primary_focal_point,framing,architectural_elements_presence,person_count,person_relationship_type,person_gender,person_age_group,person_activity,text_elements,cta_visuals,problem_solution narratives
moprobo_meta_20250115_000001,moprobo_meta_001.jpg,3.45,conversion,Power Station,US,moprobo_conversion_2025_01,moprobo_conversion_25-34_us,overhead,studio,green, white, gray, black, beige,bottom-right,Artificial,Lifestyle context,partial,dominant,cool-dominant,Cool,moderate,isolated,product-in-environment,forced,balanced,shallow,high,high,vibrant,solid-color,energetic,Exciting,active,product,Medium shot,no,single,individual,male,adult,posing,Headline, Subheadline, Feature Icons,Highlighting, Button,both
moprobo_meta_20250115_000002,moprobo_meta_002.jpg,1.23,awareness,MoProBo,US,moprobo_awareness_2025_01,moprobo_awareness_18-24_us,front,natural,brown, white, green, beige,center,natural,Face visible,full,dominant,warm-dominant,Warm,rich,in-use,product-with-people,natural,lifestyle,deep,medium,medium,muted,environment,natural,Exciting,active,person,Medium shot,yes,multiple,group,female,child,playing,None,None,both
moprobo_meta_20250115_000003,moprobo_meta_003.jpg,0.87,traffic,Power Station,EU,moprobo_traffic_2025_01,moprobo_traffic_35-44_eu,side,studio,gray, white, green, beige,top-left,Artificial,None,isolated,subdued,neutral,Neutral,minimal,isolated,product-alone,static,minimal,shallow,medium,low,low,solid-color,clinical,Neutral,low,product,Close-up,no,single,individual,male,adult,posing,Headline,None,problem
moprobo_meta_20250115_000004,moprobo_meta_004.jpg,2.56,conversion,Power Station,US,moprobo_conversion_2025_01,moprobo_conversion_45-54_us,45-degree,studio,green, white, gray, black, beige,bottom-right,Artificial,Lifestyle context,partial,dominant,cool-dominant,Cool,moderate,isolated,product-in-environment,forced,balanced,shallow,high,high,vibrant,solid-color,energetic,Exciting,active,product,Medium shot,no,single,individual,female,adult,posing,Headline, Subheadline,Highlighting, Button,both
...
```

---

### 1.3 Input Validation Module

**File:** `src/meta/ad/miner/validation/input_validator.py`

```python
"""Validate input CSV against schema."""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class InputSchemaValidator:
    """Validate creative features CSV against schema."""

    # Schema definition (could be loaded from YAML)
    SCHEMA_VERSION = "1.0"
    REQUIRED_COLUMNS = [
        "creative_id",
        "filename",
        "roas",
        "campaign_goal",
        "product",
        "branch",
    ]

    ENUM_COLUMNS = {
        "campaign_goal": ["awareness", "conversion", "traffic", "lead_generation", "app_installs", "unknown"],
        "branch": ["US", "EU", "APAC", "LATAM", "Global", "unknown"],
    }

    def __init__(self, csv_path: str | Path):
        """Initialize validator with CSV path."""
        self.csv_path = Path(csv_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> bool:
        """
        Validate CSV against schema.

        Returns:
            True if valid, False otherwise
        """
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            self.errors.append(f"Failed to read CSV: {e}")
            return False

        # Check required columns
        self._validate_required_columns(df)

        # Check data types
        self._validate_data_types(df)

        # Check enum values
        self._validate_enums(df)

        # Check value ranges
        self._validate_ranges(df)

        # Check for missing critical values
        self._validate_missing_values(df)

        # Check for duplicates
        self._validate_duplicates(df)

        is_valid = len(self.errors) == 0

        if not is_valid:
            logger.error(f"Validation failed with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning(f"Validation produced {len(self.warnings)} warnings")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        return is_valid

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Check that all required columns exist."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            self.errors.append(f"Missing required columns: {missing}")

    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate column data types."""
        # ROAS must be numeric
        if "roas" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["roas"]):
                self.errors.append("Column 'roas' must be numeric")

    def _validate_enums(self, df: pd.DataFrame) -> None:
        """Validate enum columns have valid values."""
        for column, valid_values in self.ENUM_COLUMNS.items():
            if column not in df.columns:
                continue

            invalid = df[~df[column].isin(valid_values) & df[column].notna()]
            if len(invalid) > 0:
                unique_invalid = invalid[column].unique()
                self.errors.append(
                    f"Column '{column}' has invalid values: {unique_invalid}. "
                    f"Valid: {valid_values}"
                )

    def _validate_ranges(self, df: pd.DataFrame) -> None:
        """Validate numeric column ranges."""
        if "roas" in df.columns:
            negative_roas = df[df["roas"] < 0]
            if len(negative_roas) > 0:
                self.errors.append(f"ROAS cannot be negative. Found {len(negative_roas)} rows")

    def _validate_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing critical values."""
        critical_columns = ["creative_id", "filename", "roas"]

        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    self.errors.append(
                        f"Column '{col}' has {missing_count} missing values"
                    )

    def _validate_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate creative IDs."""
        if "creative_id" in df.columns:
            duplicates = df["creative_id"].duplicated()
            if duplicates.sum() > 0:
                self.errors.append(
                    f"Found {duplicates.sum()} duplicate creative_id values"
                )

    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get validation report as dict.

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

### 1.4 Input Data Loader

**File:** `src/meta/ad/miner/data/loader.py`

```python
"""Load and validate input data with metadata."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import logging

from src.meta.ad.miner.validation.input_validator import InputSchemaValidator

logger = logging.getLogger(__name__)


def load_features_with_metadata(
    csv_path: str | Path,
    validate: bool = True,
    fill_defaults: bool = True
) -> pd.DataFrame:
    """
    Load creative features CSV with metadata columns.

    Args:
        csv_path: Path to CSV file
        validate: If True, validate against schema before returning
        fill_defaults: If True, fill missing metadata with defaults

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
                f"{report['error_count']} errors, {report['warning_count']} warnings\n"
                f"Errors: {report['errors']}"
            )

        logger.info("Input validation passed")

    # Load data
    df = pd.read_csv(csv_path)

    # Fill defaults for missing metadata
    if fill_defaults:
        df = _fill_metadata_defaults(df)

    # Add derived columns
    df = _add_derived_columns(df)

    logger.info(
        f"Loaded {len(df)} creatives with metadata: "
        f"goals={df['campaign_goal'].nunique()}, "
        f"products={df['product'].nunique()}, "
        f"branches={df['branch'].nunique()}"
    )

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
    """Add derived columns for analysis."""
    # Add performance quartile (for analysis)
    if "roas" in df.columns:
        df["performance_quartile"] = pd.qcut(
            df["roas"],
            q=4,
            labels=["bottom", "mid-low", "mid-high", "top"],
            duplicates="drop"
        )

    # Add segment key (for grouping)
    df["segment_key"] = (
        df["product"].astype(str) + "|" +
        df["branch"].astype(str) + "|" +
        df["campaign_goal"].astype(str)
    )

    return df


def get_segment_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about data segments.

    Returns:
        Dict with segment counts and sizes
    """
    segments = df.groupby(["product", "branch", "campaign_goal"]).size()

    return {
        "total_creatives": len(df),
        "num_segments": len(segments),
        "segments": [
            {
                "product": product,
                "branch": branch,
                "campaign_goal": goal,
                "sample_size": count
            }
            for (product, branch, goal), count in segments.items()
        ],
        "min_sample_size": segments.min(),
        "max_sample_size": segments.max(),
        "avg_sample_size": segments.mean(),
    }
```

---

## Part 2: Output Schema

### 2.1 Recommendations JSON Schema

**File:** `config/ad/miner/{customer}/{product}/{branch}/{goal}/recommendations.json`

**Schema version:** `2.0` (upgraded from 1.0 to support context-awareness)

```yaml
schema:
  version: "2.0"
  name: "context_aware_recommendations"
  description: "Creative recommendations with context metadata"
  format: "JSON"

# Top-level structure
root:
  type: "object"
  properties:
    # Metadata section
    metadata:
      type: "object"
      required: true
      properties:
        schema_version:
          type: "string"
          description: "Schema version"
          example: "2.0"

        customer:
          type: "string"
          description: "Customer name"
          example: "moprobo"

        product:
          type: "string"
          description: "Product name"
          example: "Power Station"

        branch:
          type: "string"
          description: "Branch/region"
          example: "US"

        campaign_goal:
          type: "string"
          description: "Campaign objective"
          example: "conversion"

        granularity_level:
          type: "integer"
          description: "Granularity level (1-4)"
          range: [1, 4]
          example: 1

        sample_size:
          type: "integer"
          description: "Number of creatives in this segment"
          example: 342

        min_threshold:
          type: "integer"
          description: "Minimum sample size for this level"
          example: 200

        analysis_date:
          type: "string"
          format: "date"
          description: "When analysis was run"
          example: "2026-01-27"

        fallback_used:
          type: "boolean"
          description: "Whether fallback to broader level was used"
          example: false

        fallback_level:
          type: "integer"
          description: "If fallback used, which level was returned"
          required: false
          example: 3

        data_quality:
          type: "object"
          properties:
            completeness_score:
              type: "float"
              range: [0, 1]
              example: 0.95

            avg_roas:
              type: "float"
              example: 2.34

            top_quartile_roas:
              type: "float"
              example: 4.56

            bottom_quartile_roas:
              type: "float"
              example: 0.98

    # High-impact recommendations section
    high_impact_recommendations:
      type: "array"
      description: "Top 5-10 recommendations by priority score"
      items:
        type: "object"
        properties:
          feature:
            type: "string"
            description: "Feature name (from schema)"
            example: "product_position"

          current_value:
            type: "string"
            description: "Current value in underperforming creatives"
            example: "center"

          recommended_value:
            type: "string"
            description: "Recommended value"
            example: "bottom-right"

          roas_lift_multiple:
            type: "float"
            description: "ROAS multiple when using this value"
            example: 2.8

          roas_lift_pct:
            type: "float"
            description: "ROAS percentage lift"
            example: 180.0

          top_quartile_prevalence:
            type: "float"
            range: [0, 1]
            description: "Prevalence in top performers"
            example: 0.67

          bottom_quartile_prevalence:
            type: "float"
            range: [0, 1]
            description: "Prevalence in bottom performers"
            example: 0.12

          confidence:
            type: "string"
            enum: ["high", "medium", "low"]
            example: "high"

          type:
            type: "string"
            enum: ["DO", "DON'T"]
            example: "DO"

          goal_specific:
            type: "boolean"
            description: "Whether this pattern is specific to campaign goal"
            example: true

          product_specific:
            type: "boolean"
            description: "Whether this pattern is specific to product"
            example: false

          branch_specific:
            type: "boolean"
            description: "Whether this pattern is specific to branch"
            example: false

          reason:
            type: "string"
            description: "Human-readable explanation"
            example: "For conversion campaigns with Power Station in US, products positioned bottom-right show 2.8x higher ROAS. Present in 67% of top performers."

          maps_to_template:
            type: "string"
            description: "Template placeholder this maps to"
            example: "product_position"

          priority_score:
            type: "float"
            description: "Priority score for ranking (higher = more important)"
            example: 9.5

          sample_count:
            type: "integer"
            description: "Number of creatives with this value"
            example: 89

          statistical_significance:
            type: "object"
            properties:
              chi_square_stat:
                type: "float"
                example: 23.45

              p_value:
                type: "float"
                range: [0, 1]
                example: 0.0001

              significant:
                type: "boolean"
                example: true

    # Negative guidance section (DON'Ts)
    negative_guidance:
      type: "array"
      description: "Anti-patterns to avoid"
      items:
        type: "object"
        properties:
          feature:
            type: "string"
            example: "product_position"

          avoid_value:
            type: "string"
            description: "Value to avoid"
            example: "top-left"

          roas_penalty_multiple:
            type: "float"
            description: "ROAS penalty when using this value"
            example: 0.6

          roas_penalty_pct:
            type: "float"
            description: "ROAS percentage penalty"
            example: -40.0

          bottom_quartile_prevalence:
            type: "float"
            example: 0.65

          confidence:
            type: "string"
            enum: ["high", "medium", "low"]
            example: "high"

          reason:
            type: "string"
            example: "Used in 65% of worst performers, 40% lower ROAS than average"

          maps_to_template:
            type: "string"
            example: "product_position"

    # Low-priority insights section
    low_priority_insights:
      type: "array"
      description: "Minor trends worth watching but not acting on"
      items:
        type: "object"
        properties:
          feature:
            type: "string"
            example: "contrast_level"

          value:
            type: "string"
            example: "high"

          roas_lift_multiple:
            type: "float"
            example: 1.05

          confidence:
            type: "string"
            enum: ["low"]
            example: "low"

          reason:
            type: "string"
            example: "Slight positive trend, but not statistically significant"

    # Generation instructions (for ad generator)
    generation_instructions:
      type: "object"
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
          description: "Feature names to avoid"
          items:
            type: "string"
          example: ["product_position:top-left"]
```

---

### 2.2 Example Output JSON

**File:** `config/ad/miner/moprobo/Power_Station/US/conversion/recommendations.json`

```json
{
  "metadata": {
    "schema_version": "2.0",
    "customer": "moprobo",
    "product": "Power Station",
    "branch": "US",
    "campaign_goal": "conversion",
    "granularity_level": 1,
    "sample_size": 342,
    "min_threshold": 200,
    "analysis_date": "2026-01-27",
    "fallback_used": false,
    "data_quality": {
      "completeness_score": 0.95,
      "avg_roas": 2.34,
      "top_quartile_roas": 4.56,
      "bottom_quartile_roas": 0.98
    }
  },
  "high_impact_recommendations": [
    {
      "feature": "product_position",
      "current_value": "center",
      "recommended_value": "bottom-right",
      "roas_lift_multiple": 2.8,
      "roas_lift_pct": 180.0,
      "top_quartile_prevalence": 0.67,
      "bottom_quartile_prevalence": 0.12,
      "confidence": "high",
      "type": "DO",
      "goal_specific": true,
      "product_specific": true,
      "branch_specific": false,
      "reason": "For conversion campaigns with Power Station in US, products positioned bottom-right show 2.8x higher ROAS. Present in 67% of top performers vs 12% in bottom quartile.",
      "maps_to_template": "product_position",
      "priority_score": 9.5,
      "sample_count": 89,
      "statistical_significance": {
        "chi_square_stat": 45.23,
        "p_value": 0.00001,
        "significant": true
      }
    },
    {
      "feature": "lighting_style",
      "current_value": "natural",
      "recommended_value": "studio",
      "roas_lift_multiple": 1.7,
      "roas_lift_pct": 70.0,
      "top_quartile_prevalence": 0.58,
      "bottom_quartile_prevalence": 0.21,
      "confidence": "high",
      "type": "DO",
      "goal_specific": true,
      "product_specific": false,
      "branch_specific": false,
      "reason": "For conversion campaigns, studio lighting shows 1.7x higher ROAS. Used in 58% of top performers.",
      "maps_to_template": "lighting_detail",
      "priority_score": 8.3,
      "sample_count": 112,
      "statistical_significance": {
        "chi_square_stat": 28.91,
        "p_value": 0.0001,
        "significant": true
      }
    },
    {
      "feature": "human_elements",
      "current_value": "Face visible",
      "recommended_value": "Lifestyle context",
      "roas_lift_multiple": 1.4,
      "roas_lift_pct": 40.0,
      "top_quartile_prevalence": 0.52,
      "bottom_quartile_prevalence": 0.28,
      "confidence": "medium",
      "type": "DO",
      "goal_specific": false,
      "product_specific": true,
      "branch_specific": false,
      "reason": "For Power Station, lifestyle context shows 1.4x higher ROAS than face visible.",
      "maps_to_template": "human_elements",
      "priority_score": 7.1,
      "sample_count": 76,
      "statistical_significance": {
        "chi_square_stat": 12.34,
        "p_value": 0.002,
        "significant": true
      }
    }
  ],
  "negative_guidance": [
    {
      "feature": "product_position",
      "avoid_value": "top-left",
      "roas_penalty_multiple": 0.6,
      "roas_penalty_pct": -40.0,
      "bottom_quartile_prevalence": 0.65,
      "confidence": "high",
      "reason": "Used in 65% of worst performers, 40% lower ROAS than average",
      "maps_to_template": "product_position"
    },
    {
      "feature": "lighting_style",
      "avoid_value": "natural",
      "roas_penalty_multiple": 0.75,
      "roas_penalty_pct": -25.0,
      "bottom_quartile_prevalence": 0.58,
      "confidence": "medium",
      "reason": "For conversion campaigns, natural lighting underperforms by 25%",
      "maps_to_template": "lighting_detail"
    }
  ],
  "low_priority_insights": [
    {
      "feature": "contrast_level",
      "value": "high",
      "roas_lift_multiple": 1.05,
      "roas_lift_pct": 5.0,
      "confidence": "low",
      "reason": "Slight positive trend (5% lift), but not statistically significant (p=0.15)"
    },
    {
      "feature": "color_saturation",
      "value": "high",
      "roas_lift_multiple": 1.03,
      "roas_lift_pct": 3.0,
      "confidence": "low",
      "reason": "Minor positive trend, inconclusive"
    }
  ],
  "generation_instructions": {
    "must_include": [
      "product_position",
      "lighting_style"
    ],
    "prioritize": [
      "visual_prominence",
      "color_balance",
      "human_elements"
    ],
    "avoid": [
      "product_position:top-left",
      "lighting_style:natural"
    ]
  }
}
```

---

### 2.3 Output Validator

**File:** `src/meta/ad/miner/validation/output_validator.py`

```python
"""Validate recommendations JSON against schema."""

from pathlib import Path
from typing import Any, Dict
import json
import logging

logger = logging.getLogger(__name__)


class OutputSchemaValidator:
    """Validate recommendations JSON against schema."""

    SCHEMA_VERSION = "2.0"

    REQUIRED_TOP_LEVEL_KEYS = [
        "metadata",
        "high_impact_recommendations",
        "negative_guidance",
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

    def __init__(self, json_path: str | Path):
        """Initialize validator with JSON path."""
        self.json_path = Path(json_path)
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self) -> bool:
        """
        Validate JSON against schema.

        Returns:
            True if valid, False otherwise
        """
        try:
            with open(self.json_path) as f:
                data = json.load(f)
        except Exception as e:
            self.errors.append(f"Failed to read JSON: {e}")
            return False

        # Validate top-level structure
        self._validate_top_level(data)

        # Validate metadata
        if "metadata" in data:
            self._validate_metadata(data["metadata"])

        # Validate recommendations
        if "high_impact_recommendations" in data:
            self._validate_recommendations(data["high_impact_recommendations"])

        # Validate negative guidance
        if "negative_guidance" in data:
            self._validate_negative_guidance(data["negative_guidance"])

        # Validate value ranges
        self._validate_ranges(data)

        is_valid = len(self.errors) == 0

        if not is_valid:
            logger.error(f"Validation failed with {len(self.errors)} errors")

        return is_valid

    def _validate_top_level(self, data: Dict) -> None:
        """Validate top-level structure."""
        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            if key not in data:
                self.errors.append(f"Missing required top-level key: {key}")

    def _validate_metadata(self, metadata: Dict) -> None:
        """Validate metadata section."""
        for key in self.REQUIRED_METADATA_KEYS:
            if key not in metadata:
                self.errors.append(f"Missing required metadata key: {key}")

        # Validate schema version
        if metadata.get("schema_version") != self.SCHEMA_VERSION:
            self.warnings.append(
                f"Schema version mismatch: expected {self.SCHEMA_VERSION}, "
                f"got {metadata.get('schema_version')}"
            )

        # Validate granularity level range
        granularity = metadata.get("granularity_level")
        if granularity is not None and not (1 <= granularity <= 4):
            self.errors.append(f"granularity_level must be 1-4, got {granularity}")

    def _validate_recommendations(self, recommendations: list) -> None:
        """Validate recommendations array."""
        for i, rec in enumerate(recommendations):
            # Check required fields
            required_fields = ["feature", "recommended_value", "confidence", "type"]
            for field in required_fields:
                if field not in rec:
                    self.errors.append(
                        f"Recommendation {i}: missing required field '{field}'"
                    )

            # Validate confidence values
            confidence = rec.get("confidence")
            if confidence not in ["high", "medium", "low"]:
                self.errors.append(
                    f"Recommendation {i}: invalid confidence '{confidence}'"
                )

            # Validate type values
            rec_type = rec.get("type")
            if rec_type not in ["DO", "DON'T"]:
                self.errors.append(
                    f"Recommendation {i}: invalid type '{rec_type}'"
                )

            # Validate ranges
            if "roas_lift_multiple" in rec:
                if rec["roas_lift_multiple"] < 1.0:
                    self.errors.append(
                        f"Recommendation {i}: roas_lift_multiple must be >= 1.0"
                    )

            if "top_quartile_prevalence" in rec:
                if not (0 <= rec["top_quartile_prevalence"] <= 1):
                    self.errors.append(
                        f"Recommendation {i}: top_quartile_prevalence must be 0-1"
                    )

    def _validate_negative_guidance(self, negative_guidance: list) -> None:
        """Validate negative guidance array."""
        for i, rec in enumerate(negative_guidance):
            required_fields = ["feature", "avoid_value", "confidence"]
            for field in required_fields:
                if field not in rec:
                    self.errors.append(
                        f"Negative guidance {i}: missing required field '{field}'"
                    )

    def _validate_ranges(self, data: Dict) -> None:
        """Validate value ranges across the document."""
        metadata = data.get("metadata", {})

        # Validate sample size > 0
        if metadata.get("sample_size", 0) <= 0:
            self.errors.append("sample_size must be > 0")

        # Validate completeness score range
        data_quality = metadata.get("data_quality", {})
        completeness = data_quality.get("completeness_score")
        if completeness is not None and not (0 <= completeness <= 1):
            self.errors.append("completeness_score must be 0-1")

    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation report."""
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

### 2.4 Recommendations Writer

**File:** `src/meta/ad/miner/output/writer.py`

```python
"""Write recommendations to structured JSON format."""

from pathlib import Path
from typing import Any, Dict
import json
import logging
from datetime import datetime

from src.meta.ad.miner.validation.output_validator import OutputSchemaValidator

logger = logging.getLogger(__name__)


def write_recommendations(
    recommendations: Dict[str, Any],
    output_path: str | Path,
    validate: bool = True,
    format_json: bool = True
) -> None:
    """
    Write recommendations to JSON file.

    Args:
        recommendations: Recommendations dict
        output_path: Output file path
        validate: If True, validate before writing
        format_json: If True, format JSON with indentation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata if not present
    if "metadata" not in recommendations:
        recommendations["metadata"] = {}

    recommendations["metadata"]["schema_version"] = "2.0"
    recommendations["metadata"]["analysis_date"] = datetime.now().strftime("%Y-%m-%d")

    # Validate if requested
    if validate:
        # Write to temp file for validation
        temp_path = output_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(recommendations, f, indent=2 if format_json else None)

        validator = OutputSchemaValidator(temp_path)
        if not validator.validate():
            report = validator.get_validation_report()
            raise ValueError(
                f"Output validation failed:\n"
                f"{report['error_count']} errors, {report['warning_count']} warnings\n"
                f"Errors: {report['errors']}"
            )

        logger.info("Output validation passed")

    # Write to final path
    with open(output_path, "w") as f:
        json.dump(recommendations, f, indent=2 if format_json else None)

    logger.info(f"Wrote recommendations to {output_path}")


def write_segment_index(
    customer: str,
    segments: list[Dict[str, Any]],
    output_path: str | Path = None
) -> None:
    """
    Write segment index file.

    Args:
        customer: Customer name
        segments: List of segment dicts with keys:
            - product
            - branch
            - campaign_goal
            - granularity_level
            - sample_size
            - file_path
        output_path: Optional custom output path
    """
    if output_path is None:
        output_path = Path(f"config/ad/miner/{customer}/segment_index.json")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    index_data = {
        "customer": customer,
        "schema_version": "2.0",
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "total_segments": len(segments),
        "segments": segments,
    }

    with open(output_path, "w") as f:
        json.dump(index_data, f, indent=2)

    logger.info(f"Wrote segment index with {len(segments)} segments to {output_path}")
```

---

### 2.5 Markdown Generator (Human-Readable View)

**File:** `src/meta/ad/miner/output/markdown_generator.py`

```python
"""Generate human-readable markdown from JSON recommendations."""

from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def generate_markdown(recommendations: Dict[str, Any]) -> str:
    """
    Generate human-readable markdown from recommendations JSON.

    Args:
        recommendations: Recommendations dict (from JSON)

    Returns:
        Markdown string
    """
    metadata = recommendations.get("metadata", {})
    high_impact = recommendations.get("high_impact_recommendations", [])
    negative = recommendations.get("negative_guidance", [])
    low_priority = recommendations.get("low_priority_insights", [])

    md_lines = []

    # Header
    md_lines.append("# Ad Creative Recommendations")
    md_lines.append("")
    md_lines.append(
        f"**Customer:** {metadata.get('customer')} | "
        f"**Product:** {metadata.get('product')} | "
        f"**Branch:** {metadata.get('branch')} | "
        f"**Goal:** {metadata.get('campaign_goal')}"
    )
    md_lines.append("")
    md_lines.append(
        f"**Analysis Date:** {metadata.get('analysis_date')} | "
        f"**Sample:** {metadata.get('sample_size')} creatives | "
        f"**Granularity:** Level {metadata.get('granularity_level')}"
    )
    md_lines.append("")

    # Data quality summary
    data_quality = metadata.get("data_quality", {})
    if data_quality:
        md_lines.append("## 📊 Data Quality")
        md_lines.append("")
        md_lines.append(f"- **Avg ROAS:** {data_quality.get('avg_roas', 0):.2f}")
        md_lines.append(f"- **Top Quartile ROAS:** {data_quality.get('top_quartile_roas', 0):.2f}")
        md_lines.append(f"- **Bottom Quartile ROAS:** {data_quality.get('bottom_quartile_roas', 0):.2f}")
        md_lines.append("")

    # High-impact recommendations
    if high_impact:
        md_lines.append("## 🎯 High-Impact Changes (Priority Order)")
        md_lines.append("")
        md_lines.append("*Implement these changes first for maximum ROAS lift*")
        md_lines.append("")

        for i, rec in enumerate(high_impact, 1):
            md_lines.append(f"### {i}. {rec['feature'].title()}: {rec['recommended_value']}")
            md_lines.append("")
            md_lines.append(f"- **Current:** {rec.get('current_value', 'N/A')}")
            md_lines.append(f"- **Recommended:** {rec['recommended_value']}")
            md_lines.append(f"- **Impact:** +{rec['roas_lift_pct']:.0f}% ROAS ({rec['roas_lift_multiple']:.1f}x lift)")
            md_lines.append(f"- **Evidence:** Used in {rec['top_quartile_prevalence']:.0%} of top performers")
            md_lines.append(f"- **Confidence:** {rec['confidence'].title()}")
            if rec.get('goal_specific'):
                md_lines.append(f"- **Note:** Specific to {metadata.get('campaign_goal')} campaigns")
            md_lines.append("")
            md_lines.append(f"**Why:** {rec['reason']}")
            md_lines.append("")

    # Negative guidance
    if negative:
        md_lines.append("## ⚠️ Avoid These")
        md_lines.append("")
        md_lines.append("*These patterns consistently underperform*")
        md_lines.append("")

        for i, rec in enumerate(negative, 1):
            md_lines.append(f"### {i}. {rec['feature'].title()}: Avoid {rec['avoid_value']}")
            md_lines.append("")
            md_lines.append(f"- **Penalty:** {rec['roas_penalty_pct']:.0f}% ROAS ({rec['roas_penalty_multiple']:.1f}x of average)")
            md_lines.append(f"- **Evidence:** Used in {rec['bottom_quartile_prevalence']:.0%} of worst performers")
            md_lines.append(f"- **Confidence:** {rec['confidence'].title()}")
            md_lines.append("")
            md_lines.append(f"**Why:** {rec['reason']}")
            md_lines.append("")

    # Low-priority insights
    if low_priority:
        md_lines.append("## 📊 Low-Priority Insights")
        md_lines.append("")
        md_lines.append("*Minor trends worth watching but not acting on yet*")
        md_lines.append("")

        for rec in low_priority:
            md_lines.append(f"- **{rec['feature'].title()}:** {rec['value']}")
            md_lines.append(f"  - Impact: +{rec['roas_lift_pct']:.0f}% ROAS (trend, not conclusive)")
            md_lines.append(f"  - {rec['reason']}")
            md_lines.append("")

    # Generation instructions
    gen_instructions = recommendations.get("generation_instructions", {})
    if gen_instructions:
        md_lines.append("---")
        md_lines.append("")
        md_lines.append("## 🤖 For Ad Generator")
        md_lines.append("")
        md_lines.append("**Must Include:** " + ", ".join(gen_instructions.get("must_include", [])))
        md_lines.append("")
        md_lines.append("**Prioritize:** " + ", ".join(gen_instructions.get("prioritize", [])))
        md_lines.append("")
        md_lines.append("**Avoid:** " + ", ".join(gen_instructions.get("avoid", [])))
        md_lines.append("")

    return "\n".join(md_lines)


def write_markdown(
    recommendations: Dict[str, Any],
    output_path: str | Path
) -> None:
    """
    Write recommendations as markdown file.

    Args:
        recommendations: Recommendations dict
        output_path: Output .md file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = generate_markdown(recommendations)

    with open(output_path, "w") as f:
        f.write(md_content)

    logger.info(f"Wrote markdown to {output_path}")
```

---

### 2.6 Example Generated Markdown

**File:** `config/ad/miner/moprobo/Power_Station/US/conversion/recommendations.md`

```markdown
# Ad Creative Recommendations

**Customer:** moprobo | **Product:** Power Station | **Branch:** US | **Goal:** conversion

**Analysis Date:** 2026-01-27 | **Sample:** 342 creatives | **Granularity:** Level 1

## 📊 Data Quality

- **Avg ROAS:** 2.34
- **Top Quartile ROAS:** 4.56
- **Bottom Quartile ROAS:** 0.98

## 🎯 High-Impact Changes (Priority Order)

*Implement these changes first for maximum ROAS lift*

### 1. Product_Position: bottom-right

- **Current:** center
- **Recommended:** bottom-right
- **Impact:** +180% ROAS (2.8x lift)
- **Evidence:** Used in 67% of top performers
- **Confidence:** High
- **Note:** Specific to conversion campaigns

**Why:** For conversion campaigns with Power Station in US, products positioned bottom-right show 2.8x higher ROAS. Present in 67% of top performers vs 12% in bottom quartile.

### 2. Lighting_Style: studio

- **Current:** natural
- **Recommended:** studio
- **Impact:** +70% ROAS (1.7x lift)
- **Evidence:** Used in 58% of top performers
- **Confidence:** High
- **Note:** Specific to conversion campaigns

**Why:** For conversion campaigns, studio lighting shows 1.7x higher ROAS. Used in 58% of top performers.

### 3. Human_Elements: Lifestyle context

- **Current:** Face visible
- **Recommended:** Lifestyle context
- **Impact:** +40% ROAS (1.4x lift)
- **Evidence:** Used in 52% of top performers
- **Confidence:** Medium
- **Note:** Specific to Power Station product

**Why:** For Power Station, lifestyle context shows 1.4x higher ROAS than face visible.

## ⚠️ Avoid These

*These patterns consistently underperform*

### 1. Product_Position: Avoid top-left

- **Penalty:** -40% ROAS (0.6x of average)
- **Evidence:** Used in 65% of worst performers
- **Confidence:** High

**Why:** Used in 65% of worst performers, 40% lower ROAS than average

### 2. Lighting_Style: Avoid natural

- **Penalty:** -25% ROAS (0.75x of average)
- **Evidence:** Used in 58% of worst performers
- **Confidence:** Medium

**Why:** For conversion campaigns, natural lighting underperforms by 25%

## 📊 Low-Priority Insights

*Minor trends worth watching but not acting on yet*

- **Contrast_Level:** high
  - Impact: +5% ROAS (trend, not conclusive)
  - Slight positive trend (5% lift), but not statistically significant (p=0.15)

- **Color_Saturation:** high
  - Impact: +3% ROAS (trend, not conclusive)
  - Minor positive trend, inconclusive

---

## 🤖 For Ad Generator

**Must Include:** product_position, lighting_style

**Prioritize:** visual_prominence, color_balance, human_elements

**Avoid:** product_position:top-left, lighting_style:natural
```

---

## Part 3: Migration Strategy

### 3.1 Schema Versioning

**Version 1.0** (Current)
- Generic recommendations only
- Markdown-only output
- No metadata

**Version 2.0** (Proposed)
- Context-aware recommendations
- JSON primary, MD generated
- Rich metadata
- Multi-level granularity

**Migration path:**
1. Add `schema_version` field to all outputs
2. Maintain backward compatibility for reading v1.0 files
3. Auto-migrate v1.0 → v2.0 on read with default metadata
4. Phase out v1.0 support after 6 months

---

### 3.2 Backward Compatibility Layer

**File:** `src/meta/ad/miner/output/compatibility.py`

```python
"""Backward compatibility for v1.0 recommendations."""

from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


def migrate_v1_to_v2(v1_recommendations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate v1.0 recommendations to v2.0 format.

    Args:
        v1_recommendations: v1.0 format recommendations

    Returns:
        v2.0 format recommendations with default metadata
    """
    v2_recommendations = {
        "metadata": {
            "schema_version": "2.0",
            "customer": "unknown",
            "product": "unknown",
            "branch": "unknown",
            "campaign_goal": "unknown",
            "granularity_level": 4,
            "sample_size": v1_recommendations.get("sample_size", 0),
            "analysis_date": "2026-01-27",
            "fallback_used": False,
            "migrated_from_v1": True,
        },
        "high_impact_recommendations": _migrate_recommendations(
            v1_recommendations.get("recommendations", [])
        ),
        "negative_guidance": [],
        "low_priority_insights": [],
        "generation_instructions": {
            "must_include": [],
            "prioritize": [],
            "avoid": [],
        },
    }

    return v2_recommendations


def _migrate_recommendations(v1_recs: list) -> list:
    """Migrate v1.0 recommendation items to v2.0 format."""
    v2_recs = []

    for rec in v1_recs:
        if rec.get("type") == "anti_pattern":
            continue  # Skip for now, would go to negative_guidance

        v2_rec = {
            "feature": rec.get("feature"),
            "current_value": rec.get("current"),
            "recommended_value": rec.get("recommended"),
            "roas_lift_multiple": 1.5,  # Default for v1.0
            "roas_lift_pct": 50.0,
            "top_quartile_prevalence": rec.get("high_performer_pct", 0.5),
            "bottom_quartile_prevalence": 0.0,  # Not tracked in v1.0
            "confidence": rec.get("confidence", "medium"),
            "type": "DO",
            "goal_specific": False,
            "product_specific": False,
            "branch_specific": False,
            "reason": rec.get("reason", ""),
            "maps_to_template": "",  # Not tracked in v1.0
            "priority_score": 5.0,  # Default for v1.0
            "sample_count": 0,
            "statistical_significance": None,
        }
        v2_recs.append(v2_rec)

    return v2_recs


def load_recommendations_with_migration(
    path: str | Path
) -> Dict[str, Any]:
    """
    Load recommendations, migrating v1.0 to v2.0 if needed.

    Args:
        path: Path to recommendations file (.json or .md)

    Returns:
        v2.0 format recommendations
    """
    path = Path(path)

    # Try JSON first
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)

        # Check version
        if data.get("metadata", {}).get("schema_version") == "2.0":
            return data  # Already v2.0
        else:
            logger.warning(f"Migrating v1.0 recommendations from {path}")
            return migrate_v1_to_v2(data)

    # Try MD (legacy v1.0 format)
    elif path.suffix == ".md":
        logger.warning(f"Loading legacy MD format from {path}")
        # Would need to parse MD and convert
        # For now, raise error
        raise NotImplementedError(
            "MD format migration not yet implemented. "
            "Please use JSON format or regenerate recommendations."
        )

    else:
        raise ValueError(f"Unknown file format: {path.suffix}")
```

---

## Summary

This design provides:

### Input Side
1. **Validated CSV schema** with required metadata columns
2. **Type-safe enum columns** for goals, products, branches
3. **Input validator** to catch data quality issues early
4. **Data loader** with default filling and derived columns

### Output Side
1. **Structured JSON schema** (v2.0) with rich metadata
2. **Output validator** to ensure correctness
3. **Markdown generator** for human-readable views
4. **Backward compatibility layer** for v1.0 migration

### Key Benefits
- **Self-documenting**: Schema includes descriptions, types, examples
- **Validatable**: Catch errors before processing/consumption
- **Queryable**: Easy to filter and aggregate by metadata
- **Extensible**: Easy to add new fields without breaking changes
- **Migrate-able**: Clear path from v1.0 to v2.0
