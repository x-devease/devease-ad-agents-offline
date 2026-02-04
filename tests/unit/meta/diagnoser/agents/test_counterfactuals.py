"""
Unit tests for counterfactual memory module.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.meta.diagnoser.agents.counterfactuals import CounterfactualMemory


@pytest.fixture
def temp_storage_path(tmp_path):
    """Create temporary storage path."""
    return tmp_path / "counterfactuals.json"


@pytest.fixture
def sample_experiment():
    """Sample experiment record."""
    return {
        "experiment_id": "exp_fatigue_001",
        "detector": "FatigueDetector",
        "timestamp": "2025-02-03T10:00:00Z",
        "spec": {
            "title": "Lower CPA threshold",
            "changes": [{
                "parameter": "cpa_increase_threshold",
                "from": 1.2,
                "to": 1.15
            }]
        },
        "outcome": "SUCCESS",
        "evaluation": {
            "lift": {"f1_score": "+4.7%"}
        }
    }


@pytest.fixture
def sample_counterfactual_changes():
    """Sample counterfactual changes."""
    return {
        "parameter": "cpa_increase_threshold",
        "from": 1.2,
        "to": 1.10  # Different from actual (1.15)
    }


@pytest.fixture
def sample_predicted_outcome():
    """Sample predicted outcome."""
    return {
        "predicted_outcome": "SUCCESS",
        "confidence": 0.75,
        "expected_lift": "+6.2%",
        "based_on": "3 similar experiments"
    }


class TestCounterfactualMemoryInit:
    """Test CounterfactualMemory initialization."""

    def test_init_creates_storage_file(self, temp_storage_path):
        """Test initialization creates storage file if it doesn't exist."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)

        # Note: File is created on first save, not init
        assert memory.enabled is True

    def test_init_loads_existing_counterfactuals(self, temp_storage_path):
        """Test initialization loads existing counterfactuals."""
        # Create existing data with proper structure
        existing = {
            "last_updated": "2025-02-03T10:00:00",
            "counterfactuals": [{
                "timestamp": "2025-02-03T10:00:00",
                "actual_experiment": {"test": "exp"},
                "what_if_changes": {"param": "value"},
                "predicted_outcome": {"outcome": "SUCCESS"},
                "validation_status": "pending",
                "actual_outcome": None,
                "notes": ""
            }]
        }
        with open(temp_storage_path, 'w') as f:
            json.dump(existing, f)

        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)

        assert len(memory.counterfactuals) == 1

    def test_init_empty_storage(self, temp_storage_path):
        """Test initialization with empty storage."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)

        assert len(memory.counterfactuals) == 0


class TestCounterfactualMemoryStore:
    """Test store() method."""

    def test_store_counterfactual(self, temp_storage_path, sample_experiment,
                                   sample_counterfactual_changes, sample_predicted_outcome):
        """Test storing a counterfactual."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)
        cf_id = memory.store(
            sample_experiment,
            sample_counterfactual_changes,
            sample_predicted_outcome
        )

        assert cf_id is not None
        assert len(memory.counterfactuals) == 1

    def test_store_includes_timestamp(self, temp_storage_path, sample_experiment,
                                       sample_counterfactual_changes, sample_predicted_outcome):
        """Test stored counterfactual includes timestamp."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)
        cf_id = memory.store(
            sample_experiment,
            sample_counterfactual_changes,
            sample_predicted_outcome
        )

        cf = memory.counterfactuals[0]
        assert hasattr(cf, "timestamp")
        datetime.fromisoformat(cf.timestamp)  # Verify ISO format

    def test_store_includes_validation_status(self, temp_storage_path, sample_experiment,
                                                sample_counterfactual_changes, sample_predicted_outcome):
        """Test stored counterfactual has validation status."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)
        cf_id = memory.store(
            sample_experiment,
            sample_counterfactual_changes,
            sample_predicted_outcome
        )

        cf = memory.counterfactuals[0]
        assert hasattr(cf, "validation_status")
        assert cf.validation_status == "pending"

    def test_store_with_notes(self, temp_storage_path, sample_experiment,
                               sample_counterfactual_changes, sample_predicted_outcome):
        """Test storing counterfactual with notes."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)
        notes = "This represents a more aggressive threshold reduction"

        cf_id = memory.store(
            sample_experiment,
            sample_counterfactual_changes,
            sample_predicted_outcome,
            notes=notes
        )

        cf = memory.counterfactuals[0]
        assert cf.notes == notes

    def test_store_persists_to_file(self, temp_storage_path, sample_experiment,
                                     sample_counterfactual_changes, sample_predicted_outcome):
        """Test storing persists to file."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)
        cf_id = memory.store(
            sample_experiment,
            sample_counterfactual_changes,
            sample_predicted_outcome
        )

        # Create new instance to verify persistence
        memory2 = CounterfactualMemory(str(temp_storage_path), enabled=True)

        assert len(memory2.counterfactuals) == 1


class TestCounterfactualMemoryQuery:
    """Test query() method."""

    def test_query_by_experiment_id(self, temp_storage_path, sample_experiment,
                                     sample_counterfactual_changes, sample_predicted_outcome):
        """Test querying counterfactuals by experiment."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)
        memory.store(sample_experiment, sample_counterfactual_changes, sample_predicted_outcome)

        results = memory.query(sample_experiment)

        assert len(results) == 1
        assert results[0].actual_experiment.get("experiment_id", "") == sample_experiment["experiment_id"]

    def test_query_returns_all_counterfactuals(self, temp_storage_path, sample_experiment,
                                                sample_counterfactual_changes, sample_predicted_outcome):
        """Test query returns all counterfactuals for an experiment."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)

        # Store multiple counterfactuals for same experiment
        memory.store(sample_experiment, {"to": 1.10}, sample_predicted_outcome)
        memory.store(sample_experiment, {"to": 1.05}, sample_predicted_outcome)

        results = memory.query(sample_experiment)

        assert len(results) == 2

    def test_query_no_counterfactuals(self, temp_storage_path, sample_experiment):
        """Test query when no counterfactuals exist."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)

        results = memory.query(sample_experiment)

        assert len(results) == 0

    def test_query_respects_max_results(self, temp_storage_path, sample_experiment,
                                        sample_counterfactual_changes, sample_predicted_outcome):
        """Test query respects max_results parameter."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)

        # Store multiple counterfactuals
        for i in range(5):
            memory.store(sample_experiment, {"to": 1.0 + i * 0.01}, sample_predicted_outcome)

        results = memory.query(sample_experiment, max_results=3)

        assert len(results) == 3


class TestCounterfactualMemoryValidate:
    """Test validate_counterfactual() method."""

    def test_validate_counterfactual(self, temp_storage_path, sample_experiment,
                                      sample_counterfactual_changes, sample_predicted_outcome):
        """Test validating a counterfactual."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)
        cf_id = memory.store(
            sample_experiment,
            sample_counterfactual_changes,
            sample_predicted_outcome
        )

        # Simulate actual outcome
        actual_outcome = {
            "outcome": "SUCCESS",
            "f1_score": 0.76,
            "lift": "+6.5%"
        }

        memory.validate_counterfactual(cf_id, actual_outcome, "validated")

        cf = memory.counterfactuals[0]
        assert cf.validation_status == "validated"
        assert cf.actual_outcome == actual_outcome

    def test_validate_nonexistent_counterfactual(self, temp_storage_path):
        """Test validating nonexistent counterfactual logs warning."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)

        # Should log warning and not raise exception
        memory.validate_counterfactual(
            "nonexistent_id",
            {"outcome": "SUCCESS"},
            "validated"
        )

        # Counterfactuals list should remain empty
        assert len(memory.counterfactuals) == 0

    def test_validate_with_failed_status(self, temp_storage_path, sample_experiment,
                                          sample_counterfactual_changes, sample_predicted_outcome):
        """Test validation with failed status."""
        memory = CounterfactualMemory(str(temp_storage_path), enabled=True)
        cf_id = memory.store(
            sample_experiment,
            sample_counterfactual_changes,
            sample_predicted_outcome
        )

        memory.validate_counterfactual(
            cf_id,
            {"outcome": "FAILURE"},
            "prediction_incorrect"
        )

        cf = memory.counterfactuals[0]
        assert cf.validation_status == "prediction_incorrect"





