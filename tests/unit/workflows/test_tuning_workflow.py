"""
Unit tests for workflows.tuning_workflow module.

Tests the TuningWorkflow class for parameter tuning orchestration.
Focuses on Bayesian optimization, constraint handling, and result reporting.
"""

from unittest.mock import Mock, patch, MagicMock
import pytest

from src.adset.allocator.workflows.tuning_workflow import TuningWorkflow
from src.adset.features.workflows.base import WorkflowResult
from src.adset.allocator.optimizer.tuning import TuningResult


@pytest.fixture
def sample_tuning_result():
    """Create sample TuningResult."""
    return TuningResult(
        param_config={
            "low_roas_threshold": 2.0,
            "high_roas_threshold": 3.0,
            "aggressive_increase_pct": 0.20,
        },
        total_adsets=10,
        adsets_with_changes=7,
        change_rate=0.7,
        total_budget_allocated=1000.0,
        budget_utilization=0.95,
        avg_roas=2.8,
        weighted_avg_roas=2.8,
        total_revenue=2800.0,
        revenue_efficiency=0.85,
        max_single_adset_pct=0.25,
        budget_gini=0.35,
        budget_entropy=1.85,
    )


class TestTuningWorkflowInit:
    """Test TuningWorkflow initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        workflow = TuningWorkflow()

        assert workflow.config_path == "config/adset/allocator/rules.yaml"
        assert workflow.n_calls == 50
        assert workflow.n_initial_points == 10
        assert workflow.update_config is True
        assert workflow.generate_report is True
        assert workflow.verbose is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        workflow = TuningWorkflow(
            config_path="custom/config.yaml",
            n_calls=100,
            n_initial_points=20,
            update_config=False,
            generate_report=False,
            verbose=False,
        )

        assert workflow.config_path == "custom/config.yaml"
        assert workflow.n_calls == 100
        assert workflow.n_initial_points == 20
        assert workflow.update_config is False
        assert workflow.generate_report is False
        assert workflow.verbose is False

    def test_init_default_constraints(self):
        """Test that default constraints are set correctly."""
        workflow = TuningWorkflow()

        assert workflow.constraints.min_budget_change_rate == 0.10
        assert workflow.constraints.max_budget_change_rate == 0.50
        assert workflow.constraints.min_total_budget_utilization == 0.85
        assert workflow.constraints.max_total_budget_utilization == 1.05
        assert workflow.constraints.min_avg_roas == 1.5
        assert workflow.constraints.min_revenue_efficiency == 0.8


class TestTuningWorkflowProcessCustomer:
    """Test TuningWorkflow._process_customer method."""

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_success(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test successful customer tuning."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is True
        assert "Parameter tuning complete" in result.message
        assert result.data["weighted_avg_roas"] == 2.8
        assert result.data["budget_utilization"] == 0.95
        assert result.data["change_rate"] == 0.7
        assert result.data["revenue_efficiency"] == 0.85
        assert result.data["total_revenue"] == 2800.0
        assert result.data["budget_gini"] == 0.35
        assert result.data["budget_entropy"] == 1.85
        assert "param_config" in result.data

        # Verify tuner was initialized correctly
        mock_tuner_class.assert_called_once()
        call_kwargs = mock_tuner_class.call_args[1]
        assert call_kwargs["config_path"] == workflow.config_path
        assert call_kwargs["constraints"] == workflow.constraints
        assert call_kwargs["n_calls"] == 50
        assert call_kwargs["n_initial_points"] == 10

        # Verify tune_customer was called
        mock_tuner.tune_customer.assert_called_once_with("test_customer")

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_with_custom_optimization_params(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test tuning with custom optimization parameters."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(n_calls=100, n_initial_points=20, verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is True

        # Verify tuner was initialized with custom params
        mock_tuner_class.assert_called_once()
        call_kwargs = mock_tuner_class.call_args[1]
        assert call_kwargs["n_calls"] == 100
        assert call_kwargs["n_initial_points"] == 20

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_updates_config(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test that config is updated when update_config=True."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(update_config=True, verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is True
        mock_tuner.update_config_with_results.assert_called_once_with(
            {"test_customer": sample_tuning_result}
        )

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_skips_config_update(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test that config is not updated when update_config=False."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(update_config=False, verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is True
        mock_tuner.update_config_with_results.assert_not_called()

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_no_valid_configuration(
        self, mock_tuner_class, mock_ensure_dirs
    ):
        """Test handling when tuner returns None (no valid configuration)."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = None
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is False
        assert "no valid configuration found" in result.message
        assert result.data is None
        assert result.error is None

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_file_not_found(self, mock_tuner_class, mock_ensure_dirs):
        """Test handling of FileNotFoundError during tuning."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.side_effect = FileNotFoundError("Config not found")
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is False
        assert "File not found" in result.message
        assert isinstance(result.error, FileNotFoundError)

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_unexpected_error(
        self, mock_tuner_class, mock_ensure_dirs
    ):
        """Test handling of unexpected Exception during tuning."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.side_effect = RuntimeError("Optimization failed")
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is False
        assert "Tuning error" in result.message
        assert isinstance(result.error, RuntimeError)

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_with_verbose_logging(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result, caplog
    ):
        """Test verbose logging during processing."""
        import logging

        caplog.set_level(logging.INFO)

        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=True)

        with patch.object(workflow, "_print_result_summary") as mock_print:
            result = workflow._process_customer(
                customer="test_customer", platform="meta"
            )

        assert result.success is True
        # Verify _print_result_summary was called
        mock_print.assert_called_once_with("test_customer", sample_tuning_result)

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_process_customer_without_report_generation(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test processing without report generation."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(generate_report=False, verbose=True)

        with patch.object(workflow, "_print_result_summary") as mock_print:
            result = workflow._process_customer(
                customer="test_customer", platform="meta"
            )

        assert result.success is True
        # Verify _print_result_summary was not called
        mock_print.assert_not_called()


class TestTuningWorkflowPrintResultSummary:
    """Test TuningWorkflow._print_result_summary method."""

    def test_print_result_summary(self, caplog):
        """Test printing tuning result summary."""
        import logging

        caplog.set_level(logging.INFO)

        workflow = TuningWorkflow(verbose=True)

        result = TuningResult(
            param_config={
                "low_roas_threshold": 2.0,
                "high_roas_threshold": 3.0,
            },
            total_adsets=10,
            adsets_with_changes=7,
            change_rate=0.7,
            total_budget_allocated=1000.0,
            budget_utilization=0.95,
            avg_roas=2.8,
            weighted_avg_roas=2.8,
            total_revenue=2800.0,
            revenue_efficiency=0.85,
            max_single_adset_pct=0.25,
            budget_gini=0.35,
            budget_entropy=1.85,
        )

        workflow._print_result_summary("test_customer", result)

        logs = [record.message for record in caplog.records]

        # Check for headers
        assert any("BEST PARAMETERS" in msg for msg in logs)
        assert any("METRICS" in msg for msg in logs)

        # Check for parameters
        assert any("low_roas_threshold: 2.0" in msg for msg in logs)
        assert any("high_roas_threshold: 3.0" in msg for msg in logs)

        # Check for metrics
        assert any("Weighted Avg ROAS: 2.8000" in msg for msg in logs)
        assert any("Budget Utilization: 95.00%" in msg for msg in logs)
        assert any("Change Rate: 70.00%" in msg for msg in logs)
        assert any("Revenue Efficiency: 0.8500" in msg for msg in logs)
        assert any("Total Revenue: $2,800.00" in msg for msg in logs)
        assert any("Budget Gini: 0.350" in msg for msg in logs)
        assert any("Budget Entropy: 1.850" in msg for msg in logs)


class TestTuningWorkflowIntegration:
    """Integration tests for TuningWorkflow orchestration."""

    @patch("src.workflows.base.get_all_customers")
    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_run_multiple_customers(
        self,
        mock_tuner_class,
        mock_ensure_dirs,
        mock_get_customers,
        sample_tuning_result,
    ):
        """Test running tuning workflow for multiple customers."""
        mock_get_customers.return_value = ["customer1", "customer2"]

        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        results = workflow.run(platform="meta")

        assert len(results) == 2
        assert all(result.success for result in results.values())
        assert workflow.metrics.total_customers == 2
        assert workflow.metrics.successful_customers == 2

        # Verify update_config_with_results was called for each customer
        assert mock_tuner.update_config_with_results.call_count == 2
        # Verify both customers were updated
        call_args_list = [
            call[0][0] for call in mock_tuner.update_config_with_results.call_args_list
        ]
        updated_customers = []
        for call_arg in call_args_list:
            updated_customers.extend(call_arg.keys())
        assert "customer1" in updated_customers
        assert "customer2" in updated_customers

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_run_single_customer(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test running tuning workflow for single customer."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        results = workflow.run(customer="customer1", platform="meta")

        assert len(results) == 1
        assert "customer1" in results
        assert results["customer1"].success is True
        assert workflow.metrics.total_customers == 1
        assert workflow.metrics.successful_customers == 1

    @patch("src.workflows.base.get_all_customers")
    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_run_mixed_results(
        self,
        mock_tuner_class,
        mock_ensure_dirs,
        mock_get_customers,
        sample_tuning_result,
    ):
        """Test running tuning workflow with mixed success/failure."""
        mock_get_customers.return_value = ["customer1", "customer2", "customer3"]

        mock_tuner = MagicMock()
        # customer2 fails (returns None)
        mock_tuner.tune_customer.side_effect = [
            sample_tuning_result,
            None,
            sample_tuning_result,
        ]
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        results = workflow.run(platform="meta")

        assert len(results) == 3
        assert results["customer1"].success is True
        assert results["customer2"].success is False
        assert results["customer3"].success is True

        # Check metrics
        assert workflow.metrics.total_customers == 3
        assert workflow.metrics.successful_customers == 2
        assert workflow.metrics.failed_customers == 1

    @patch("src.workflows.base.get_all_customers")
    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_continues_on_error(
        self,
        mock_tuner_class,
        mock_ensure_dirs,
        mock_get_customers,
        sample_tuning_result,
    ):
        """Test that workflow continues even when one customer fails."""
        mock_get_customers.return_value = ["customer1", "customer2", "customer3"]

        mock_tuner = MagicMock()
        # customer2 raises exception
        mock_tuner.tune_customer.side_effect = [
            sample_tuning_result,
            FileNotFoundError("Config for customer2 not found"),
            sample_tuning_result,
        ]
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        results = workflow.run(platform="meta")

        # All customers should be in results
        assert len(results) == 3
        assert results["customer1"].success is True
        assert results["customer2"].success is False
        assert results["customer3"].success is True

        # Metrics should reflect all attempts
        assert workflow.metrics.total_customers == 3
        assert workflow.metrics.successful_customers == 2
        assert workflow.metrics.failed_customers == 1

    @patch("src.workflows.base.get_all_customers")
    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_with_constraints(
        self,
        mock_tuner_class,
        mock_ensure_dirs,
        mock_get_customers,
        sample_tuning_result,
    ):
        """Test that constraints are properly passed to tuner."""
        mock_get_customers.return_value = ["customer1"]

        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(verbose=False)

        results = workflow.run(platform="meta")

        # Verify constraints were passed to BayesianTuner
        mock_tuner_class.assert_called_once()
        call_kwargs = mock_tuner_class.call_args[1]
        assert "constraints" in call_kwargs
        constraints = call_kwargs["constraints"]
        assert constraints.min_budget_change_rate == 0.10
        assert constraints.max_budget_change_rate == 0.50
        assert constraints.min_total_budget_utilization == 0.85
        assert constraints.max_total_budget_utilization == 1.05
        assert constraints.min_avg_roas == 1.5
        assert constraints.min_revenue_efficiency == 0.8

    @patch("src.workflows.base.get_all_customers")
    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_empty_customer_list(
        self, mock_tuner_class, mock_ensure_dirs, mock_get_customers
    ):
        """Test workflow when no customers are found."""
        mock_get_customers.return_value = []

        workflow = TuningWorkflow(verbose=False)

        results = workflow.run(platform="meta")

        assert results == {}
        assert workflow.metrics.total_customers == 0
        # BayesianTuner should not be initialized
        mock_tuner_class.assert_not_called()


class TestTuningWorkflowWithDifferentModes:
    """Test TuningWorkflow with different tuning modes."""

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_bayesian_mode(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test workflow in Bayesian optimization mode (default)."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(n_calls=50, verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is True

        # Verify BayesianTuner initialization
        mock_tuner_class.assert_called_once()
        call_kwargs = mock_tuner_class.call_args[1]
        assert call_kwargs["n_calls"] == 50
        assert call_kwargs["n_initial_points"] == 10

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_with_more_iterations(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test workflow with more optimization iterations."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(n_calls=200, n_initial_points=30, verbose=False)

        result = workflow._process_customer(customer="test_customer", platform="meta")

        assert result.success is True

        # Verify parameters
        mock_tuner_class.assert_called_once()
        call_kwargs = mock_tuner_class.call_args[1]
        assert call_kwargs["n_calls"] == 200
        assert call_kwargs["n_initial_points"] == 30

    @patch("src.workflows.tuning_workflow.ensure_customer_dirs")
    @patch("src.workflows.tuning_workflow.BayesianTuner")
    def test_workflow_diagnose_mode(
        self, mock_tuner_class, mock_ensure_dirs, sample_tuning_result
    ):
        """Test workflow in diagnose mode (generate_report=True)."""
        mock_tuner = MagicMock()
        mock_tuner.tune_customer.return_value = sample_tuning_result
        mock_tuner_class.return_value = mock_tuner

        workflow = TuningWorkflow(generate_report=True, verbose=True)

        with patch.object(workflow, "_print_result_summary") as mock_print:
            result = workflow._process_customer(
                customer="test_customer", platform="meta"
            )

        assert result.success is True
        mock_print.assert_called_once()
