"""
Parameter tuning workflow.

Tunes budget allocation parameters using Bayesian optimization.
"""

import logging
from typing import Optional

from src.adset.allocator.optimizer.lib.bayesian_tuner import BayesianTuner
from src.adset.allocator.optimizer.tuning import TuningConstraints
from src.utils.customer_paths import ensure_customer_dirs
from src.adset.features.workflows.base import Workflow, WorkflowResult

logger = logging.getLogger(__name__)


class TuningWorkflow(Workflow):
    """
    Workflow for tuning budget allocation parameters.

    Performs:
    1. Load customer feature data
    2. Run Bayesian optimization
    3. Update configuration with best parameters
    4. Generate tuning report
    """

    def __init__(
        self,
        config_path: str = "config/adset/allocator/rules.yaml",
        n_calls: int = 50,
        n_initial_points: int = 10,
        update_config: bool = True,
        generate_report: bool = True,
        **kwargs,
    ):
        """Initialize tuning workflow.

        Args:
            config_path: Path to configuration file.
            n_calls: Number of optimization iterations.
            n_initial_points: Number of initial random evaluations.
            update_config: Whether to update config with best parameters.
            generate_report: Whether to generate tuning report.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(config_path, **kwargs)
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.update_config = update_config
        self.generate_report = generate_report

        # Define tuning constraints
        self.constraints = TuningConstraints(
            min_budget_change_rate=0.10,  # At least 10% changes
            max_budget_change_rate=0.50,  # At most 50% changes
            min_total_budget_utilization=0.85,  # Use 85%+ budget
            max_total_budget_utilization=1.05,  # Don't exceed 105%
            min_avg_roas=1.5,  # Minimum average ROAS
            min_revenue_efficiency=0.8,  # Revenue efficiency
        )

    def _process_customer(
        self,
        customer: str,
        platform: Optional[str],
        **kwargs,
    ) -> WorkflowResult:
        """
        Tune parameters for a single customer.

        Args:
            customer: Customer name.
            platform: Platform name.
            **kwargs: Additional arguments.

        Returns:
            WorkflowResult with tuning status.
        """
        try:
            # Ensure directories exist
            ensure_customer_dirs(customer, platform)

            if self.verbose:
                self._print_header(f"Tuning Parameters: {customer}")

            # Initialize tuner
            tuner = BayesianTuner(
                config_path=self.config_path,
                constraints=self.constraints,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
            )

            # Tune customer
            result = tuner.tune_customer(customer)

            if result is None:
                return WorkflowResult(
                    success=False,
                    customer=customer,
                    message="Tuning failed - no valid configuration found",
                )

            # Update config
            if self.update_config:
                if self.verbose:
                    logger.info("Updating configuration...")
                tuner.update_config_with_results({customer: result})

            # Generate report
            if self.generate_report and self.verbose:
                self._print_result_summary(customer, result)

            return WorkflowResult(
                success=True,
                customer=customer,
                message="Parameter tuning complete",
                data={
                    "weighted_avg_roas": result.weighted_avg_roas,
                    "budget_utilization": result.budget_utilization,
                    "change_rate": result.change_rate,
                    "revenue_efficiency": result.revenue_efficiency,
                    "total_revenue": result.total_revenue,
                    "budget_gini": result.budget_gini,
                    "budget_entropy": result.budget_entropy,
                    "param_config": result.param_config,
                },
            )

        except FileNotFoundError as err:
            logger.error(f"File not found: {err}")
            return WorkflowResult(
                success=False,
                customer=customer,
                message=f"File not found: {err}",
                error=err,
            )
        except Exception as err:
            logger.exception(f"Tuning error: {err}")
            return WorkflowResult(
                success=False,
                customer=customer,
                message=f"Tuning error: {err}",
                error=err,
            )

    def _print_result_summary(self, customer: str, result):
        """Print tuning result summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("BEST PARAMETERS")
        logger.info("=" * 70)
        for key, value in result.param_config.items():
            logger.info(f"  {key}: {value}")

        logger.info("")
        logger.info("=" * 70)
        logger.info("METRICS")
        logger.info("=" * 70)
        logger.info(f"  Weighted Avg ROAS: {result.weighted_avg_roas:.4f}")
        logger.info(f"  Budget Utilization: {result.budget_utilization:.2%}")
        logger.info(f"  Change Rate: {result.change_rate:.2%}")
        logger.info(f"  Revenue Efficiency: {result.revenue_efficiency:.4f}")
        logger.info(f"  Total Revenue: ${result.total_revenue:,.2f}")
        logger.info(f"  Budget Gini: {result.budget_gini:.3f}")
        logger.info(f"  Budget Entropy: {result.budget_entropy:.3f}")
        logger.info("=" * 70)
