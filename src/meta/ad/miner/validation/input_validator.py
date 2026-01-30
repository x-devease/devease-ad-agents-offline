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

            if max_val is not None and col_name in df.columns:
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    invalid = df[df[col_name] > max_val]
                    if len(invalid) > 0:
                        self.errors.append(
                            f"Column '{col_name}' has {len(invalid)} values > {max_val}"
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
            try:
                pattern = re.compile(pattern_str)
            except re.error as e:
                self.errors.append(
                    f"Column '{col_name}' has invalid regex pattern: {e}"
                )
                continue

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
