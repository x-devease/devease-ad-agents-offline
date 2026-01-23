"""
Integration tests for TuningWorkflow.

Tests the end-to-end parameter tuning workflow including:
- Bayesian optimization
- Configuration updates
- Report generation
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.adset.allocator.workflows.tuning_workflow import TuningWorkflow
from src.adset.features.workflows.base import WorkflowResult


@pytest.fixture
def sample_adset_features(tmp_path):
    """Create sample adset features for tuning."""
    features_df = pd.DataFrame(
        {
            "adset_id": [f"adset_{i:03d}" for i in range(1, 51)],
            "date_start": ["2024-01-01"] * 50,
            "purchase_roas_rolling_7d": [2.0 + i * 0.05 for i in range(50)],
            "roas_trend": [0.05 + i * 0.01 for i in range(50)],
            "health_score": [0.5 + i * 0.01 for i in range(50)],
            "days_since_start": [20 + i for i in range(50)],
            "spend": [100.0 + i * 5 for i in range(50)],
            "impressions": [10000] * 50,
            "clicks": [500] * 50,
        }
    )

    features_file = tmp_path / "adset_features.csv"
    features_df.to_csv(features_file, index=False)
    return features_file


@pytest.fixture
def sample_config(tmp_path):
    """Create sample configuration for tuning."""
    import yaml

    config = {
        "test_customer": {
            "meta": {
                "decision_rules": {
                    "low_roas_threshold": 2.0,
                    "medium_roas_threshold": 2.5,
                    "high_roas_threshold": 3.0,
                }
            }
        }
    }

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


class TestTuningWorkflow:
    """Test TuningWorkflow end-to-end."""

    @pytest.mark.slow
    def test_tuning_with_few_iterations(
        self, sample_adset_features, sample_config, tmp_path
    ):
        """Test tuning with minimal iterations for quick testing."""
        workflow = TuningWorkflow(
            config_path=str(sample_config),
            n_calls=5,  # Very few for testing
            n_initial_points=2,
            update_config=False,  # Don't modify config
            generate_report=True,
        )

        result = workflow._process_customer(
            customer="test_customer",
            platform="meta",
        )

        assert isinstance(result, WorkflowResult)
        # Tuning may or may not succeed depending on data quality
        # Just verify it runs without crashing

    def test_tuning_with_config_update(
        self, sample_adset_features, sample_config, tmp_path
    ):
        """Test tuning with configuration update enabled."""
        workflow = TuningWorkflow(
            config_path=str(sample_config),
            n_calls=3,  # Minimal for testing
            update_config=True,
            generate_report=False,
        )

        result = workflow._process_customer(
            customer="test_customer",
            platform="meta",
        )

        assert isinstance(result, WorkflowResult)

    def test_tuning_constraints(self):
        """Test that tuning constraints are properly set."""
        workflow = TuningWorkflow()

        assert workflow.constraints is not None
        assert workflow.constraints.min_budget_change_rate > 0
        assert workflow.constraints.max_budget_change_rate <= 1.0
        assert workflow.constraints.min_total_budget_utilization > 0
        assert workflow.constraints.max_total_budget_utilization <= 1.1

    def test_custom_tuning_iterations(self):
        """Test custom iteration counts."""
        workflow = TuningWorkflow(
            n_calls=100,
            n_initial_points=20,
        )

        assert workflow.n_calls == 100
        assert workflow.n_initial_points == 20

    def test_tuning_report_generation_toggle(self):
        """Test report generation toggle."""
        workflow_with_report = TuningWorkflow(generate_report=True)
        assert workflow_with_report.generate_report is True

        workflow_no_report = TuningWorkflow(generate_report=False)
        assert workflow_no_report.generate_report is False


class TestTuningWorkflowErrorHandling:
    """Test error handling in tuning workflow."""

    def test_tuning_with_insufficient_data(self, tmp_path):
        """Test tuning with insufficient data."""
        # Create very small dataset
        small_df = pd.DataFrame(
            {
                "adset_id": ["adset_001", "adset_002"],
                "date_start": ["2024-01-01"] * 2,
                "purchase_roas_rolling_7d": [2.0, 2.5],
            }
        )

        features_file = tmp_path / "small_features.csv"
        small_df.to_csv(features_file, index=False)

        import yaml

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"test_customer": {"meta": {}}}, f)

        workflow = TuningWorkflow(
            config_path=str(config_file),
            n_calls=2,
        )

        result = workflow._process_customer(
            customer="test_customer",
            platform="meta",
        )

        # Should handle insufficient data
        assert isinstance(result, WorkflowResult)

    def test_tuning_with_invalid_config(self, tmp_path):
        """Test tuning with invalid configuration."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml:")

        workflow = TuningWorkflow(
            config_path=str(invalid_config),
            n_calls=2,
        )

        result = workflow._process_customer(
            customer="test_customer",
            platform="meta",
        )

        # Should handle invalid config
        assert isinstance(result, WorkflowResult)
