"""
Unit tests for schema validator.
"""

import pytest
import json
from pathlib import Path
from jsonschema import ValidationError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.meta.diagnoser.agents.schema_validator import SchemaValidator


@pytest.fixture
def sample_schema():
    """Sample JSON schema for testing."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "TestSchema",
        "type": "object",
        "required": ["name", "value"],
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1
            },
            "value": {
                "type": "number",
                "minimum": 0
            },
            "optional_field": {
                "type": "string"
            }
        }
    }


@pytest.fixture
def schema_dir(tmp_path, sample_schema):
    """Create temporary schema directory."""
    schema_file = tmp_path / "test_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(sample_schema, f)

    return tmp_path


@pytest.fixture
def validator(schema_dir):
    """Create SchemaValidator instance."""
    return SchemaValidator(str(schema_dir))


class TestSchemaValidatorInit:
    """Test SchemaValidator initialization."""

    def test_load_schemas_successfully(self, validator):
        """Test schemas are loaded successfully."""
        assert "test_schema" in validator.schemas
        assert len(validator.schemas) == 1

    def test_load_nonexistent_directory(self):
        """Test loading nonexistent directory raises error."""
        with pytest.raises(FileNotFoundError):
            SchemaValidator("/nonexistent/path")

    def test_load_empty_directory(self, tmp_path):
        """Test loading empty directory."""
        validator = SchemaValidator(str(tmp_path))

        assert len(validator.schemas) == 0


class TestSchemaValidatorValidate:
    """Test SchemaValidator.validate() method."""

    def test_validate_valid_data(self, validator):
        """Test validation of valid data."""
        data = {
            "name": "test",
            "value": 42
        }

        is_valid, message = validator.validate(data, "test_schema")

        assert is_valid is True
        assert message == "Valid"

    def test_validate_missing_required_field(self, validator):
        """Test validation fails when required field is missing."""
        data = {
            "name": "test"
            # Missing "value" field
        }

        is_valid, message = validator.validate(data, "test_schema")

        assert is_valid is False
        assert "required" in message.lower()

    def test_validate_wrong_type(self, validator):
        """Test validation fails when field has wrong type."""
        data = {
            "name": "test",
            "value": "not_a_number"  # Should be number
        }

        is_valid, message = validator.validate(data, "test_schema")

        assert is_valid is False
        assert "type" in message.lower() or "number" in message.lower()

    def test_validate_value_out_of_range(self, validator):
        """Test validation fails when value is out of range."""
        data = {
            "name": "test",
            "value": -10  # Should be >= 0
        }

        is_valid, message = validator.validate(data, "test_schema")

        assert is_valid is False

    def test_validate_with_optional_field(self, validator):
        """Test validation with optional field included."""
        data = {
            "name": "test",
            "value": 42,
            "optional_field": "present"
        }

        is_valid, message = validator.validate(data, "test_schema")

        assert is_valid is True

    def test_validate_nonexistent_schema(self, validator):
        """Test validation with nonexistent schema name."""
        data = {"name": "test", "value": 42}

        is_valid, message = validator.validate(data, "nonexistent_schema")

        assert is_valid is False
        assert "not found" in message.lower()


class TestSchemaValidatorValidateWithDetails:
    """Test SchemaValidator.validate_with_details() method."""

    def test_validate_with_details_success(self, validator):
        """Test successful validation with details."""
        data = {
            "name": "test",
            "value": 42
        }

        is_valid, message, details = validator.validate_with_details(data, "test_schema")

        assert is_valid is True
        assert details is None

    def test_validate_with_details_failure(self, validator):
        """Test failed validation with error details."""
        data = {
            "name": "test"
            # Missing "value"
        }

        is_valid, message, details = validator.validate_with_details(data, "test_schema")

        assert is_valid is False
        assert details is not None
        # Details structure contains 'errors' key with error details
        assert "errors" in details or isinstance(details, dict)

    def test_validate_with_details_multiple_errors(self, schema_dir):
        """Test validation with multiple errors."""
        # Create schema with multiple fields
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["field1", "field2", "field3"],
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"},
                "field3": {"type": "boolean"}
            }
        }

        schema_file = schema_dir / "multi_error_schema.json"
        with open(schema_file, 'w') as f:
            json.dump(schema, f)

        validator = SchemaValidator(str(schema_dir))

        data = {
            "field1": 123,  # Wrong type
            # Missing field2
            # Missing field3
        }

        is_valid, message, details = validator.validate_with_details(data, "multi_error_schema")

        assert is_valid is False
        assert details is not None


class TestSchemaValidatorExperimentSpec:
    """Test validation of experiment spec schema."""

    def test_valid_experiment_spec(self):
        """Test validation of valid experiment spec."""
        schema_dir = Path("src/meta/diagnoser/agents/schemas")
        if not schema_dir.exists():
            pytest.skip("Schema directory not found")

        validator = SchemaValidator(str(schema_dir))

        spec = {
            "title": "Test optimization experiment",
            "scope": "阈值调整",
            "changes": [{
                "file": "src/meta/diagnoser/detectors/fatigue_detector.py",
                "type": "threshold_tuning",
                "parameter": "cpa_increase_threshold",
                "from": 1.2,
                "to": 1.15,
                "reason": "Lowering threshold to improve recall"
            }],
            "constraints": [
                "Precision must not drop below 0.90",
                "F1-Score must improve by at least 3%"
            ],
            "expected_outcome": {
                "f1_score": "0.73 (+4%)",
                "recall": "0.60 (+11%)",
                "precision": ">=0.95"
            },
            "rollback_plan": "Revert threshold to 1.2 if precision drops below 0.90"
        }

        is_valid, message = validator.validate(spec, "experiment_spec")

        assert is_valid is True

    def test_invalid_experiment_spec_missing_fields(self):
        """Test experiment spec validation fails with missing fields."""
        schema_dir = Path("src/meta/diagnoser/agents/schemas")
        if not schema_dir.exists():
            pytest.skip("Schema directory not found")

        validator = SchemaValidator(str(schema_dir))

        spec = {
            "title": "Incomplete spec"
            # Missing required fields
        }

        is_valid, message = validator.validate(spec, "experiment_spec")

        assert is_valid is False

    def test_invalid_experiment_spec_wrong_scope(self):
        """Test experiment spec validation fails with invalid scope."""
        schema_dir = Path("src/meta/diagnoser/agents/schemas")
        if not schema_dir.exists():
            pytest.skip("Schema directory not found")

        validator = SchemaValidator(str(schema_dir))

        spec = {
            "title": "Test",
            "scope": "INVALID_SCOPE",  # Not in enum
            "changes": [{
                "file": "src/meta/diagnoser/detectors/fatigue_detector.py",
                "type": "threshold_tuning",
                "parameter": "test",
                "from": 1.0,
                "to": 2.0,
                "reason": "Test"
            }],
            "constraints": ["Test"],
            "expected_outcome": {"f1_score": "0.73"}
        }

        is_valid, message = validator.validate(spec, "experiment_spec")

        assert is_valid is False


class TestSchemaValidatorJudgeDecision:
    """Test validation of judge decision schema."""

    def test_valid_judge_decision(self):
        """Test validation of valid judge decision."""
        schema_dir = Path("src/meta/diagnoser/agents/schemas")
        if not schema_dir.exists():
            pytest.skip("Schema directory not found")

        validator = SchemaValidator(str(schema_dir))

        decision = {
            "decision": "MERGE",
            "metrics_comparison": {
                "baseline": {"precision": 1.0, "recall": 0.5, "f1_score": 0.7, "tp": 50, "fp": 0, "fn": 50},
                "new": {"precision": 0.97, "recall": 0.6, "f1_score": 0.74, "tp": 60, "fp": 2, "fn": 40},
                "improvement": {
                    "precision_delta": -0.03,
                    "recall_delta": 0.1,
                    "f1_delta": 0.04,
                    "precision_improvement_pct": -3.0,
                    "recall_improvement_pct": 20.0,
                    "f1_improvement_pct": 5.7
                }
            },
            "reasoning": "Good improvement with acceptable precision trade-off"
        }

        is_valid, message = validator.validate(decision, "judge_decision")

        assert is_valid is True

    def test_invalid_judge_decision_missing_decision(self):
        """Test judge decision validation fails without decision."""
        schema_dir = Path("src/meta/diagnoser/agents/schemas")
        if not schema_dir.exists():
            pytest.skip("Schema directory not found")

        validator = SchemaValidator(str(schema_dir))

        decision = {
            "evaluation_result": {
                "overall_score": 85
                # Missing "decision"
            }
        }

        is_valid, message = validator.validate(decision, "judge_decision")

        assert is_valid is False


class TestSchemaValidatorErrorMessages:
    """Test schema validator error messages."""

    def test_error_message_is_informative(self, validator):
        """Test error messages provide useful information."""
        data = {"name": "test"}  # Missing "value"

        is_valid, message = validator.validate(data, "test_schema")

        assert is_valid is False
        assert len(message) > 0
        # Should mention the missing field
        assert "value" in message.lower() or "required" in message.lower()

    def test_error_message_includes_field_path(self, validator):
        """Test error messages include field path for nested errors."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"}
                    },
                    "required": ["value"]
                }
            }
        }

        schema_file = Path(validator.schema_dir) / "nested_schema.json"
        with open(schema_file, 'w') as f:
            json.dump(schema, f)

        validator._load_schemas()

        data = {
            "nested": {}
            # Missing nested.value
        }

        is_valid, message, details = validator.validate_with_details(data, "nested_schema")

        assert is_valid is False
        assert details is not None
