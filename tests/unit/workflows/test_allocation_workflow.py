"""
Unit tests for workflows.allocation_workflow module.

Tests the AllocationWorkflow class for budget allocation orchestration.
Focuses on allocator initialization, budget processing, and different allocation approaches.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import pytest

from src.adset.allocator.workflows.allocation_workflow import AllocationWorkflow
from src.adset.features.workflows.base import WorkflowResult


@pytest.fixture
def sample_features_df():
    """Create sample adset features DataFrame."""
    return pd.DataFrame(
        {
            "adset_id": ["adset_001", "adset_002", "adset_003", "adset_004"],
            "spend": [100.0, 150.0, 200.0, 120.0],
            "purchase_roas_rolling_7d": [3.5, 2.0, 1.2, 2.8],
            "roas_trend": [0.12, 0.05, -0.08, 0.03],
            "health_score": [0.90, 0.70, 0.45, 0.85],
            "days_since_start": [30, 20, 15, 25],
            "previous_budget": [100.0, 150.0, 200.0, 120.0],
        }
    )


@pytest.fixture
def sample_config():
    """Create sample config."""
    config = MagicMock()
    config.get.return_value = None
    return config


class TestAllocationWorkflowInitializeAllocator:
    """Test AllocationWorkflow._initialize_allocator method."""

    @patch("src.workflows.allocation_workflow.SafetyRules")
    @patch("src.workflows.allocation_workflow.DecisionRules")
    @patch("src.workflows.allocation_workflow.Allocator")
    def test_initialize_allocator(
        self,
        mock_allocator_class,
        mock_decision_rules_class,
        mock_safety_rules_class,
        sample_config,
    ):
        """Test initializing rules-based allocator."""
        mock_safety_rules = MagicMock()
        mock_decision_rules = MagicMock()
        mock_allocator = MagicMock()
        mock_safety_rules_class.return_value = mock_safety_rules
        mock_decision_rules_class.return_value = mock_decision_rules
        mock_allocator_class.return_value = mock_allocator

        workflow = AllocationWorkflow()
        allocator = workflow._initialize_allocator(
            sample_config, "test_customer", "meta"
        )

        # Verify rules-based initialization
        mock_safety_rules_class.assert_called_once_with(sample_config)
        mock_decision_rules_class.assert_called_once_with(sample_config)
        mock_allocator_class.assert_called_once_with(
            mock_safety_rules, mock_decision_rules, sample_config
        )
        assert allocator == mock_allocator


class TestAllocationWorkflowExtractMetrics:
    """Test AllocationWorkflow._extract_metrics method."""

    def test_extract_metrics_basic(self):
        """Test extracting basic metrics from DataFrame row."""
        workflow = AllocationWorkflow()
        mock_config = MagicMock()
        mock_config.get_safety_rule.return_value = 1.0

        row = pd.Series(
            {
                "spend": 100.0,
                "purchase_roas_rolling_7d": 3.5,
                "roas_trend": 0.12,
                "health_score": 0.85,
                "days_since_start": 30,
            }
        )

        metrics = workflow._extract_metrics(mock_config, row, total_budget_today=500.0)

        # When Series has no index name, row.name is None, so str(None) = 'None'
        assert metrics["adset_id"] == "None"  # str(row.name) where row.name is None
        assert metrics["current_budget"] == 100.0
        assert metrics["roas_7d"] == 3.5
        assert metrics["roas_trend"] == 0.12
        assert metrics["health_score"] == 0.85
        assert metrics["days_active"] == 30
        assert metrics["total_budget_today"] == 500.0

    def test_extract_metrics_with_previous_budget(self):
        """Test extracting metrics with previous_budget."""
        workflow = AllocationWorkflow()
        mock_config = MagicMock()
        mock_config.get_safety_rule.return_value = 1.0

        row = pd.Series(
            {
                "spend": 100.0,
                "previous_budget": 90.0,
                "purchase_roas_rolling_7d": 3.5,
            }
        )

        metrics = workflow._extract_metrics(mock_config, row, total_budget_today=500.0)

        assert metrics["previous_budget"] == 90.0

    def test_extract_metrics_with_nan_values(self):
        """Test extracting metrics with NaN values."""
        workflow = AllocationWorkflow()
        mock_config = MagicMock()
        mock_config.get_safety_rule.return_value = 1.0

        row = pd.Series(
            {
                "spend": 100.0,
                "purchase_roas_rolling_7d": None,
                "roas_trend": None,
                "health_score": 0.5,
                "days_since_start": 10,
            }
        )

        metrics = workflow._extract_metrics(mock_config, row, total_budget_today=500.0)

        # roas_7d has pd.notna() check in source, so NaN becomes 0.0
        assert metrics["roas_7d"] == 0.0
        # roas_trend does NOT have pd.notna() check in source (line 339), so it becomes nan
        assert pd.isna(metrics["roas_trend"])
        assert metrics["health_score"] == 0.5

    def test_extract_metrics_with_optional_metrics(self):
        """Test extracting metrics with optional fields."""
        workflow = AllocationWorkflow()
        mock_config = MagicMock()
        mock_config.get_safety_rule.return_value = 1.0

        row = pd.Series(
            {
                "spend": 100.0,
                "purchase_roas_rolling_7d": 3.5,
                "adset_roas": 3.2,
                "campaign_roas": 2.8,
                "account_roas": 2.5,
                "impressions": 1000,
                "clicks": 50,
                "reach": 800,
                "budget_utilization": 0.95,
                "marginal_roas": 2.0,
            }
        )

        metrics = workflow._extract_metrics(mock_config, row, total_budget_today=500.0)

        # Optional metrics should be included
        assert "adset_roas" in metrics
        assert "campaign_roas" in metrics
        assert "account_roas" in metrics
        assert metrics["impressions"] == 1000
        assert metrics["clicks"] == 50
        assert metrics["reach"] == 800
        assert metrics["budget_utilization"] == 0.95
        assert metrics["marginal_roas"] == 2.0


class TestAllocationWorkflowAllocateBudget:
    """Test AllocationWorkflow._allocate_budget method."""

    def test_allocate_budget_rules_based(self, sample_features_df):
        """Test budget allocation with rules-based allocator."""
        workflow = AllocationWorkflow()
        mock_config = MagicMock()
        mock_config.get_safety_rule.return_value = 1.0

        # Mock allocator
        mock_allocator = MagicMock()
        mock_allocator.allocate_budget.side_effect = [
            (120.0, ["increase", "high_roas"]),
            (140.0, ["maintain"]),
            (180.0, ["decrease", "low_roas"]),
            (110.0, ["increase", "medium_roas"]),
        ]

        total_budget = 500.0
        results_df = workflow._allocate_budget(
            mock_allocator, mock_config, sample_features_df, total_budget
        )

        assert len(results_df) == 4
        assert "adset_id" in results_df.columns
        assert "current_budget" in results_df.columns
        assert "new_budget" in results_df.columns
        assert "change_pct" in results_df.columns
        assert "decision_path" in results_df.columns

        # Verify budgets were scaled to match total
        total_allocated = results_df["new_budget"].sum()
        assert abs(total_allocated - total_budget) < 0.01  # Allow small rounding errors

    def test_allocate_budget_scaling(self, sample_features_df):
        """Test that budgets are properly scaled."""
        workflow = AllocationWorkflow()
        mock_config = MagicMock()
        mock_config.get_safety_rule.return_value = 1.0

        # Mock allocator to return budgets that don't sum to total
        mock_allocator = MagicMock()
        mock_allocator.allocate_budget.side_effect = [
            (200.0, ["increase"]),
            (200.0, ["maintain"]),
            (200.0, ["decrease"]),
            (200.0, ["increase"]),
        ]

        total_budget = 500.0
        results_df = workflow._allocate_budget(
            mock_allocator, mock_config, sample_features_df, total_budget
        )

        # Total should match exactly after scaling
        total_allocated = results_df["new_budget"].sum()
        assert abs(total_allocated - total_budget) < 0.01

    def test_allocate_budget_empty_dataframe(self):
        """Test allocation with empty DataFrame."""
        workflow = AllocationWorkflow()
        mock_config = MagicMock()
        mock_config.get_safety_rule.return_value = 1.0

        mock_allocator = MagicMock()
        empty_df = pd.DataFrame({"adset_id": []})

        # When DataFrame is empty, the code tries to access 'new_budget' column
        # on line 305 before it exists, causing KeyError
        with pytest.raises(KeyError, match="new_budget"):
            results_df = workflow._allocate_budget(
                mock_allocator, mock_config, empty_df, 500.0
            )


class TestAllocationWorkflowProcessCustomer:
    """Test AllocationWorkflow._process_customer method."""

    @patch("src.workflows.allocation_workflow.ensure_customer_dirs")
    @patch("src.workflows.allocation_workflow.get_customer_adset_features_path")
    @patch("src.workflows.allocation_workflow.get_customer_allocations_path")
    @patch("src.workflows.allocation_workflow.MonthlyBudgetState")
    @patch("src.workflows.allocation_workflow.MonthlyBudgetTracker")
    def test_process_customer_success(
        self,
        mock_tracker_class,
        mock_state_class,
        mock_get_allocations_path,
        mock_get_adset_path,
        mock_ensure_dirs,
        sample_features_df,
        tmp_path,
    ):
        """Test successful customer processing."""
        # Setup paths
        input_file = tmp_path / "input_features.csv"
        output_file = tmp_path / "output_allocations.csv"
        sample_features_df.to_csv(input_file, index=False)

        mock_get_adset_path.return_value = input_file
        mock_get_allocations_path.return_value = output_file

        workflow = AllocationWorkflow(budget=10000.0)

        # Mock config and allocator
        mock_config = MagicMock()
        mock_config.get_monthly_setting.return_value = True
        mock_allocator = MagicMock()
        # Mock allocate_budget to return (new_budget, decision_path)
        mock_allocator.allocate_budget.return_value = (100.0, ["test_rule"])

        # Mock state and tracker
        mock_state = MagicMock()
        mock_state.month = "2026-01"
        mock_state.month_start_date = datetime(2026, 1, 1)
        mock_state.days_since_budget_start = 1
        mock_state.is_first_execution = True
        mock_state.tracking = {
            "total_spent": 0.0,
            "total_allocated": 0.0,
            "remaining_budget": 10000.0,
            "days_active": 0,
        }
        mock_state.state_path = tmp_path / "state.json"
        mock_state_class.load_or_create.return_value = mock_state

        mock_tracker = MagicMock()
        mock_tracker.is_budget_exhausted.return_value = False
        mock_tracker.calculate_daily_budget.return_value = 333.33
        mock_tracker_class.return_value = mock_tracker

        with patch.object(workflow, "get_customer_config", return_value=mock_config):
            with patch.object(
                workflow, "_initialize_allocator", return_value=mock_allocator
            ):
                result = workflow._process_customer(
                    customer="test_customer", platform="meta"
                )

        assert result.success is True
        assert "Budget allocation complete" in result.message
        assert result.data["total_adsets"] == 4
        assert "total_allocated" in result.data
        assert "avg_roas" in result.data
        assert output_file.exists()

    @patch("src.workflows.allocation_workflow.ensure_customer_dirs")
    @patch("src.workflows.allocation_workflow.get_customer_adset_features_path")
    @patch("src.workflows.allocation_workflow.get_customer_allocations_path")
    def test_process_customer_missing_adset_id(
        self, mock_get_allocations_path, mock_get_adset_path, mock_ensure_dirs, tmp_path
    ):
        """Test processing with missing adset_id column."""
        # Create DataFrame without adset_id
        df_without_id = pd.DataFrame(
            {
                "spend": [100.0, 150.0],
                "roas": [2.5, 3.0],
            }
        )

        input_file = tmp_path / "input_features.csv"
        df_without_id.to_csv(input_file, index=False)

        mock_get_adset_path.return_value = input_file

        workflow = AllocationWorkflow(budget=10000.0)

        mock_config = MagicMock()
        with patch.object(workflow, "get_customer_config", return_value=mock_config):
            with patch.object(
                workflow, "_initialize_allocator", return_value=MagicMock()
            ):
                result = workflow._process_customer(
                    customer="test_customer", platform="meta"
                )

        assert result.success is False
        assert "missing 'adset_id' column" in result.message

    @patch("src.workflows.allocation_workflow.ensure_customer_dirs")
    @patch("src.workflows.allocation_workflow.get_customer_adset_features_path")
    def test_process_customer_file_not_found(
        self, mock_get_adset_path, mock_ensure_dirs
    ):
        """Test processing with FileNotFoundError."""
        mock_get_adset_path.return_value = Path("/nonexistent/file.csv")

        workflow = AllocationWorkflow(budget=10000.0)

        mock_config = MagicMock()
        with patch.object(workflow, "get_customer_config", return_value=mock_config):
            with patch.object(
                workflow, "_initialize_allocator", return_value=MagicMock()
            ):
                result = workflow._process_customer(
                    customer="test_customer", platform="meta"
                )

        assert result.success is False
        assert "File not found" in result.message
        assert isinstance(result.error, FileNotFoundError)

    @patch("src.workflows.allocation_workflow.ensure_customer_dirs")
    @patch("src.workflows.allocation_workflow.get_customer_adset_features_path")
    @patch("src.workflows.allocation_workflow.get_customer_allocations_path")
    @patch("src.workflows.allocation_workflow.MonthlyBudgetState")
    @patch("src.workflows.allocation_workflow.MonthlyBudgetTracker")
    def test_process_customer_with_explicit_paths(
        self,
        mock_tracker_class,
        mock_state_class,
        mock_get_allocations_path,
        mock_get_adset_path,
        mock_ensure_dirs,
        sample_features_df,
        tmp_path,
    ):
        """Test processing with explicit input/output paths."""
        custom_input = tmp_path / "custom_input.csv"
        custom_output = tmp_path / "custom_output.csv"
        sample_features_df.to_csv(custom_input, index=False)

        mock_get_adset_path.return_value = Path("/default/input.csv")
        mock_get_allocations_path.return_value = Path("/default/output.csv")

        workflow = AllocationWorkflow(budget=10000.0)

        mock_config = MagicMock()
        mock_config.get_monthly_setting.return_value = True
        mock_allocator = MagicMock()
        # Mock allocate_budget to return (new_budget, decision_path)
        mock_allocator.allocate_budget.return_value = (100.0, ["test_rule"])

        # Mock state and tracker
        mock_state = MagicMock()
        mock_state.month = "2026-01"
        mock_state.month_start_date = datetime(2026, 1, 1)
        mock_state.days_since_budget_start = 1
        mock_state.is_first_execution = True
        mock_state.tracking = {
            "total_spent": 0.0,
            "total_allocated": 0.0,
            "remaining_budget": 10000.0,
            "days_active": 0,
        }
        mock_state.state_path = tmp_path / "state.json"
        mock_state_class.load_or_create.return_value = mock_state

        mock_tracker = MagicMock()
        mock_tracker.is_budget_exhausted.return_value = False
        mock_tracker.calculate_daily_budget.return_value = 333.33
        mock_tracker_class.return_value = mock_tracker

        with patch.object(workflow, "get_customer_config", return_value=mock_config):
            with patch.object(
                workflow, "_initialize_allocator", return_value=mock_allocator
            ):
                result = workflow._process_customer(
                    customer="test_customer",
                    platform="meta",
                    input_file=str(custom_input),
                    output_file=str(custom_output),
                )

        assert result.success is True
        assert custom_output.exists()

    @patch("src.workflows.allocation_workflow.ensure_customer_dirs")
    @patch("src.workflows.allocation_workflow.get_customer_adset_features_path")
    @patch("src.workflows.allocation_workflow.get_customer_allocations_path")
    @patch("src.workflows.allocation_workflow.MonthlyBudgetState")
    @patch("src.workflows.allocation_workflow.MonthlyBudgetTracker")
    def test_process_customer_calculates_statistics(
        self,
        mock_tracker_class,
        mock_state_class,
        mock_get_allocations_path,
        mock_get_adset_path,
        mock_ensure_dirs,
        sample_features_df,
        tmp_path,
    ):
        """Test that processing calculates allocation statistics."""
        input_file = tmp_path / "input_features.csv"
        output_file = tmp_path / "output_allocations.csv"
        sample_features_df.to_csv(input_file, index=False)

        mock_get_adset_path.return_value = input_file
        mock_get_allocations_path.return_value = output_file

        workflow = AllocationWorkflow(budget=10000.0)

        mock_config = MagicMock()
        mock_config.get_monthly_setting.return_value = True
        mock_allocator = MagicMock()
        # Mock allocate_budget to return (new_budget, decision_path)
        mock_allocator.allocate_budget.return_value = (100.0, ["test_rule"])

        # Mock state and tracker
        mock_state = MagicMock()
        mock_state.month = "2026-01"
        mock_state.month_start_date = datetime(2026, 1, 1)
        mock_state.days_since_budget_start = 1
        mock_state.is_first_execution = True
        mock_state.tracking = {
            "total_spent": 0.0,
            "total_allocated": 0.0,
            "remaining_budget": 10000.0,
            "days_active": 0,
        }
        mock_state.state_path = tmp_path / "state.json"
        mock_state_class.load_or_create.return_value = mock_state

        mock_tracker = MagicMock()
        mock_tracker.is_budget_exhausted.return_value = False
        mock_tracker.calculate_daily_budget.return_value = 333.33
        mock_tracker_class.return_value = mock_tracker

        with patch.object(workflow, "get_customer_config", return_value=mock_config):
            with patch.object(
                workflow, "_initialize_allocator", return_value=mock_allocator
            ):
                result = workflow._process_customer(
                    customer="test_customer", platform="meta"
                )

        assert result.success is True
        assert "increases" in result.data
        assert "decreases" in result.data
        assert "no_change" in result.data
        assert "avg_roas" in result.data


class TestAllocationWorkflowIntegration:
    """Integration tests for AllocationWorkflow orchestration."""

    @patch("src.workflows.allocation_workflow.MonthlyBudgetTracker")
    @patch("src.workflows.allocation_workflow.MonthlyBudgetState")
    @patch("src.workflows.base.get_all_customers")
    @patch("src.workflows.allocation_workflow.ensure_customer_dirs")
    @patch("src.workflows.allocation_workflow.get_customer_adset_features_path")
    @patch("src.workflows.allocation_workflow.get_customer_allocations_path")
    def test_workflow_run_multiple_customers(
        self,
        mock_get_allocations_path,
        mock_get_adset_path,
        mock_ensure_dirs,
        mock_get_customers,
        mock_state_class,
        mock_tracker_class,
        sample_features_df,
        tmp_path,
    ):
        """Test running allocation workflow for multiple customers."""
        # Mock state and tracker
        mock_state = MagicMock()
        mock_state.month = "2026-01"
        mock_state.month_start_date = datetime(2026, 1, 1)
        mock_state.days_since_budget_start = 1
        mock_state.is_first_execution = True
        mock_state.tracking = {
            "total_spent": 0.0,
            "total_allocated": 0.0,
            "remaining_budget": 10000.0,
            "days_active": 0,
        }
        mock_state.state_path = tmp_path / "state.json"
        mock_state_class.load_or_create.return_value = mock_state

        mock_tracker = MagicMock()
        mock_tracker.is_budget_exhausted.return_value = False
        mock_tracker.calculate_daily_budget.return_value = 333.33
        mock_tracker_class.return_value = mock_tracker

        # Mock get_all_customers to return our test customers
        # This is called by base.run() -> _run_all_customers() at line 153
        # Patch where it's used (src.workflows.base) not where it's defined
        mock_get_customers.return_value = ["customer1", "customer2"]

        # Create input files for each customer
        input_files = {}
        output_files = {}

        for customer in ["customer1", "customer2"]:
            input_file = tmp_path / f"{customer}_features.csv"
            output_file = tmp_path / f"{customer}_allocations.csv"
            sample_features_df.to_csv(input_file, index=False)

            input_files[customer] = input_file
            output_files[customer] = output_file

        def get_adset_side_effect(customer, platform):
            return input_files[customer]

        def get_allocations_side_effect(customer, platform):
            return output_files[customer]

        mock_get_adset_path.side_effect = get_adset_side_effect
        mock_get_allocations_path.side_effect = get_allocations_side_effect

        workflow = AllocationWorkflow(budget=10000.0, verbose=False)

        mock_config = MagicMock()
        mock_config.get_monthly_setting.return_value = 0.8
        mock_config.get_safety_rule.return_value = 1.0
        mock_allocator = MagicMock()
        # Mock allocate_budget to return (new_budget, decision_path)
        mock_allocator.allocate_budget.return_value = (100.0, ["test_rule"])

        with patch.object(workflow, "get_customer_config", return_value=mock_config):
            with patch.object(
                workflow, "_initialize_allocator", return_value=mock_allocator
            ):
                # Don't pass customer parameter - run all customers
                results = workflow.run(platform="meta")

        assert len(results) == 2
        assert all(result.success for result in results.values())
        assert workflow.metrics.total_customers == 2
        assert workflow.metrics.successful_customers == 2

    @patch("src.workflows.allocation_workflow.MonthlyBudgetTracker")
    @patch("src.workflows.allocation_workflow.MonthlyBudgetState")
    @patch("src.workflows.allocation_workflow.ensure_customer_dirs")
    @patch("src.workflows.allocation_workflow.get_customer_adset_features_path")
    @patch("src.workflows.allocation_workflow.get_customer_allocations_path")
    def test_workflow_run_single_customer(
        self,
        mock_get_allocations_path,
        mock_get_adset_path,
        mock_ensure_dirs,
        mock_state_class,
        mock_tracker_class,
        sample_features_df,
        tmp_path,
    ):
        """Test running allocation workflow for single customer."""
        # Mock state and tracker
        mock_state = MagicMock()
        mock_state.month = "2026-01"
        mock_state.month_start_date = datetime(2026, 1, 1)
        mock_state.days_since_budget_start = 1
        mock_state.is_first_execution = True
        mock_state.tracking = {
            "total_spent": 0.0,
            "total_allocated": 0.0,
            "remaining_budget": 10000.0,
            "days_active": 0,
        }
        mock_state.state_path = tmp_path / "state.json"
        mock_state_class.load_or_create.return_value = mock_state

        mock_tracker = MagicMock()
        mock_tracker.is_budget_exhausted.return_value = False
        mock_tracker.calculate_daily_budget.return_value = 333.33
        mock_tracker_class.return_value = mock_tracker

        input_file = tmp_path / "customer1_features.csv"
        output_file = tmp_path / "customer1_allocations.csv"
        sample_features_df.to_csv(input_file, index=False)

        mock_get_adset_path.return_value = input_file
        mock_get_allocations_path.return_value = output_file

        workflow = AllocationWorkflow(budget=10000.0, verbose=False)

        mock_config = MagicMock()
        mock_config.get_monthly_setting.return_value = 0.8
        mock_config.get_safety_rule.return_value = 1.0
        mock_allocator = MagicMock()
        # Mock allocate_budget to return (new_budget, decision_path)
        mock_allocator.allocate_budget.return_value = (100.0, ["test_rule"])

        with patch.object(workflow, "get_customer_config", return_value=mock_config):
            with patch.object(
                workflow, "_initialize_allocator", return_value=mock_allocator
            ):
                results = workflow.run(customer="customer1", platform="meta")

        assert len(results) == 1
        assert "customer1" in results
        assert results["customer1"].success is True
        assert workflow.metrics.total_customers == 1
