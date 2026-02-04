"""
Schema Validator for Agent Outputs.

Validates agent outputs against JSON schemas to ensure correctness.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

try:
    from jsonschema import validate, ValidationError, Draft7Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates agent outputs against JSON schemas.

    Loads all schemas from the schemas directory and provides
    validation methods for different agent outputs.
    """

    def __init__(self, schema_dir: Optional[str] = None):
        """
        Initialize schema validator.

        Args:
            schema_dir: Path to schemas directory. If None, uses default path.
        """
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError(
                "jsonschema package is required. Install with: pip install jsonschema"
            )

        if schema_dir is None:
            schema_dir = Path(__file__).parent / "schemas"
        else:
            schema_dir = Path(schema_dir)

        self.schema_dir = schema_dir
        self.schemas = {}
        self.validators = {}

        self._load_schemas()
        logger.info(f"Loaded {len(self.schemas)} schemas from {schema_dir}")

    def _load_schemas(self):
        """Load all JSON schemas from schema directory."""
        if not self.schema_dir.exists():
            raise FileNotFoundError(f"Schema directory not found: {self.schema_dir}")

        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                    schema_name = schema_file.stem
                    self.schemas[schema_name] = schema
                    self.validators[schema_name] = Draft7Validator(schema)
                    logger.debug(f"Loaded schema: {schema_name}")
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")

    def validate(self, data: Dict[str, Any], schema_name: str) -> Tuple[bool, str]:
        """
        Validate data against a schema.

        Args:
            data: Data to validate
            schema_name: Name of schema to validate against

        Returns:
            Tuple of (is_valid, error_message)

        Examples:
            >>> is_valid, error = validator.validate(spec, "experiment_spec")
            >>> if not is_valid:
            ...     print(f"Validation failed: {error}")
        """
        if schema_name not in self.schemas:
            return False, f"Schema '{schema_name}' not found"

        try:
            validate(instance=data, schema=self.schemas[schema_name])
            return True, "Valid"
        except ValidationError as e:
            error_path = " -> ".join(str(p) for p in e.path)
            error_msg = f"Validation error at '{error_path}': {e.message}"
            return False, error_msg
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"

    def validate_with_details(
        self,
        data: Dict[str, Any],
        schema_name: str
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validate data and return detailed errors.

        Args:
            data: Data to validate
            schema_name: Name of schema to validate against

        Returns:
            Tuple of (is_valid, error_message, details)

        Examples:
            >>> valid, msg, details = validator.validate_with_details(spec, "experiment_spec")
            >>> if not valid:
            ...     print(f"Errors: {details['errors']}")
        """
        if schema_name not in self.validators:
            return False, f"Schema '{schema_name}' not found", None

        validator = self.validators[schema_name]
        errors = []

        for error in validator.iter_errors(data):
            error_path = " -> ".join(str(p) for p in error.path)
            errors.append({
                "path": error_path,
                "message": error.message,
                "validator": error.validator,
                "schema_path": " -> ".join(str(p) for p in error.absolute_schema_path)
            })

        if errors:
            return False, f"Validation failed with {len(errors)} error(s)", {
                "errors": errors,
                "error_count": len(errors)
            }

        return True, "Valid", None

    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema by name.

        Args:
            schema_name: Name of schema

        Returns:
            Schema dictionary or None if not found
        """
        return self.schemas.get(schema_name)

    def list_schemas(self) -> list:
        """
        List all available schema names.

        Returns:
            List of schema names
        """
        return list(self.schemas.keys())


# Convenience functions for validation
def validate_experiment_spec(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate experiment spec."""
    validator = SchemaValidator()
    return validator.validate(data, "experiment_spec")


def validate_implementation_result(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate implementation result."""
    validator = SchemaValidator()
    return validator.validate(data, "implementation_result")


def validate_review_result(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate review result."""
    validator = SchemaValidator()
    return validator.validate(data, "review_result")


def validate_judge_decision(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate judge decision."""
    validator = SchemaValidator()
    return validator.validate(data, "judge_decision")


# Singleton instance
_global_validator: Optional[SchemaValidator] = None


def get_global_validator() -> SchemaValidator:
    """Get global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = SchemaValidator()
    return _global_validator
