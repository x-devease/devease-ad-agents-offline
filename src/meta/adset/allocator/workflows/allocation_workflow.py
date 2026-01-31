"""
Budget allocation workflow.

Executes rule-based budget allocation for adsets.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.meta.adset import Allocator, DecisionRules, SafetyRules
from src.meta.adset.allocator.utils.parser import Parser
from src.utils.customer_paths import (
    ensure_customer_dirs,
    get_customer_adset_features_path,
    get_customer_allocations_path,
)
from src.meta.adset.allocator.features.workflows.base import Workflow, WorkflowResult

# Import budget tracking module
from src.meta.adset.allocator.budget import MonthlyBudgetTracker, MonthlyBudgetState

logger = logging.getLogger(__name__)


class AllocationWorkflow(Workflow):
    """
    Workflow for budget allocation.

    Performs:
    1. Load adset features
    2. Initialize rules-based allocator
    3. Allocate budget to adsets
    4. Save allocation results
    """

    def __init__(
        self,
        config_path: str = "config/adset/allocator/rules.yaml",
        budget: float = 10000.0,
        **kwargs,
    ):
        """Initialize allocation workflow.

        Args:
            config_path: Path to configuration file.
            budget: Total budget to allocate.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(config_path, **kwargs)
        self.budget = budget

    def _process_customer(
        self,
        customer: str,
        platform: Optional[str],
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        **kwargs,
    ) -> WorkflowResult:
        """
        Allocate budget for a single customer.

        Args:
            customer: Customer name.
            platform: Platform name.
            input_file: Explicit path to input features file.
            output_file: Explicit path to output allocations file.
            **kwargs: Additional arguments.

        Returns:
            WorkflowResult with allocation status.
        """
        try:
            # Ensure directories exist
            ensure_customer_dirs(customer, platform)

            # Load configuration
            config = self.get_customer_config(customer)

            # Initialize allocator (pass platform for decision engine)
            allocator = self._initialize_allocator(config, customer, platform or "meta")

            # Determine paths
            if input_file is None:
                input_file = str(get_customer_adset_features_path(customer, platform))
            if output_file is None:
                output_file = str(
                    get_customer_allocations_path(customer, platform=platform)
                )

            # Load features
            df_features = pd.read_csv(input_file)

            if "adset_id" not in df_features.columns:
                return WorkflowResult(
                    success=False,
                    customer=customer,
                    message="Input missing 'adset_id' column",
                )

            # Filter to most recent date's rolling metrics
            # Don't filter to single date - use pre-computed rolling metrics directly
            if "date_start" in df_features.columns:
                # Just get the most recent date for each adset to use their rolling metrics
                df_features = (
                    df_features.sort_values("date_start")
                    .groupby("adset_id")
                    .last()
                    .reset_index()
                )

            # === Budget Tracking Integration ===
            from datetime import datetime

            # Load or create state (monthly tracking always enabled)
            state = MonthlyBudgetState.load_or_create(
                customer=customer,
                platform=platform or "meta",
                monthly_budget=self.budget,
            )
            tracker = MonthlyBudgetTracker(state)

            # Check if budget exhausted
            if tracker.is_budget_exhausted():
                return WorkflowResult(
                    success=False,
                    customer=customer,
                    message="Monthly budget exhausted",
                    data={
                        "total_spent": state.tracking["total_spent"],
                        "monthly_budget": self.budget,
                    },
                )

            # Calculate today's budget
            total_budget_today = tracker.calculate_daily_budget(datetime.now())

            logger.info(f"Monthly tracking active for {state.month}")
            logger.info(
                f"Spent: ${state.tracking['total_spent']:.2f} of ${self.budget:.2f}"
            )
            logger.info(f"Today's budget: ${total_budget_today:.2f}")

            # NEW: Day 1 special handling
            if state.is_first_execution:
                logger.info("=" * 70)
                logger.info("FIRST EXECUTION OF MONTHLY BUDGET PERIOD")
                logger.info("=" * 70)
                logger.info(f"Month: {state.month}")
                logger.info(f"Monthly Budget: ${self.budget:.2f}")
                logger.info(
                    f"Start Date: {state.month_start_date.strftime('%Y-%m-%d')}"
                )
                logger.info(f"Days in Period: {state.days_since_budget_start}")

                # Apply day 1 conservative multiplier (configurable)
                day1_multiplier = config.get(
                    "monthly_budget.day1_budget_multiplier", 0.8
                )
                total_budget_today = total_budget_today * day1_multiplier
                logger.info(f"Day 1 Conservative Multiplier: {day1_multiplier}")
                logger.info(f"Adjusted Daily Budget: ${total_budget_today:.2f}")

            # Perform allocation
            results_df = self._allocate_budget(
                allocator, config, df_features, total_budget_today
            )

            # Save results
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to: {output_file}")

            # Record allocation
            actual_spend = results_df["current_budget"].sum()

            # Archive allocation
            archive_path = self._archive_allocation(
                output_file, datetime.now(), customer, platform
            )

            # NEW: Extract per-adset allocations for tracking
            adset_allocations = {
                row["adset_id"]: row["new_budget"] for _, row in results_df.iterrows()
            }

            # Record execution
            tracker.record_allocation(
                execution_date=datetime.now(),
                allocated=total_budget_today,
                spent=actual_spend,
                num_adsets=len(results_df),
                allocation_file=str(archive_path),
                adset_allocations=adset_allocations,  # NEW
            )

            # Save state
            state.save()
            logger.info(f"State saved to: {state.state_path}")

            # Calculate summary statistics
            total_allocated = results_df["new_budget"].sum()
            increases = (results_df["change_pct"] > 0).sum()
            decreases = (results_df["change_pct"] < 0).sum()
            no_change = (results_df["change_pct"] == 0).sum()
            avg_roas = results_df["roas_7d"].mean()

            # Build result data
            result_data = {
                "total_adsets": len(results_df),
                "total_allocated": total_allocated,
                "avg_roas": avg_roas,
                "increases": increases,
                "decreases": decreases,
                "no_change": no_change,
                "output_file": str(output_file),
                "monthly_tracking": {
                    "month": state.month,
                    "total_spent": state.tracking["total_spent"],
                    "remaining_budget": state.tracking["remaining_budget"],
                    "days_active": state.tracking["days_active"],
                },
            }

            return WorkflowResult(
                success=True,
                customer=customer,
                message="Budget allocation complete",
                data=result_data,
            )

        except FileNotFoundError as err:
            logger.error(f"File not found: {err}")
            return WorkflowResult(
                success=False,
                customer=customer,
                message=f"File not found: {err}",
                error=err,
            )
        except pd.errors.EmptyDataError as err:
            logger.error(f"Empty data: {err}")
            return WorkflowResult(
                success=False,
                customer=customer,
                message=f"Empty input file: {err}",
                error=err,
            )
        except Exception as err:
            logger.exception(f"Allocation error: {err}")
            return WorkflowResult(
                success=False,
                customer=customer,
                message=f"Allocation error: {err}",
                error=err,
            )

    def _initialize_allocator(
        self, config: Parser, customer: str, platform: str = "meta"
    ):
        """Initialize rules-based allocator."""
        logger.info("Using rules-based allocator")
        safety_rules = SafetyRules(config)
        decision_rules = DecisionRules(config)
        return Allocator(safety_rules, decision_rules, config)

    def _filter_to_most_recent_active_date(
        self, df_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter features to the most recent date with active spend.

        Args:
            df_features: DataFrame with adset features.

        Returns:
            Filtered DataFrame for the most recent active date.
        """
        # Find dates with any spend > 0
        active_dates = df_features[df_features["spend"] > 0]["date_start"].unique()

        if len(active_dates) == 0:
            # No active dates, use the most recent date
            most_recent_date = df_features["date_start"].max()
            logger.warning(
                f"No dates with spend > 0 found. Using most recent date: {most_recent_date}"
            )
        else:
            # Use the most recent active date
            most_recent_date = pd.Series(active_dates).max()
            logger.info(f"Using most recent active date: {most_recent_date}")

        # Filter to that date
        df_filtered = df_features[df_features["date_start"] == most_recent_date].copy()
        logger.info(
            f"Filtered to {len(df_filtered)} adsets on {most_recent_date} "
            f"({(df_filtered['spend'] > 0).sum()} with spend > 0)"
        )

        return df_filtered

    def _allocate_budget(
        self,
        allocator,
        config,
        df_features: pd.DataFrame,
        total_budget_today: float,
    ) -> pd.DataFrame:
        """Allocate budget to all adsets using rules-based approach.

        Args:
            allocator: Allocator instance.
            df_features: DataFrame with adset features.
            total_budget_today: Total budget for today.

        Returns:
            DataFrame with allocation results.
        """
        adset_groups = df_features.groupby("adset_id").first()

        results = []
        for adset_id, row in adset_groups.iterrows():
            metrics = self._extract_metrics(config, row, total_budget_today)
            new_budget, decision_path = allocator.allocate_budget(**metrics)

            current_budget = metrics["current_budget"]
            change_pct = (
                ((new_budget / current_budget) - 1) * 100 if current_budget > 0 else 0
            )

            results.append(
                {
                    "adset_id": adset_id,
                    "current_budget": current_budget,
                    "new_budget": new_budget,
                    "change_pct": change_pct,
                    "roas_7d": metrics.get("roas_7d", 0),
                    "health_score": metrics.get("health_score", 0.5),
                    "days_active": metrics.get("days_active", 0),
                    "decision_path": (
                        " -> ".join(decision_path) if decision_path else "default"
                    ),
                }
            )

        results_df = pd.DataFrame(results)

        # Scale budgets to match total_budget_today
        total_new_budget = results_df["new_budget"].sum()
        if total_new_budget > 0:
            scale_factor = total_budget_today / total_new_budget
            results_df["new_budget"] = results_df["new_budget"] * scale_factor
            results_df["change_pct"] = (
                (results_df["new_budget"] / results_df["current_budget"]) - 1
            ) * 100

        return results_df

    def _is_truly_new_adset(self, config, row: pd.Series) -> bool:
        """
        Determine if adset is truly new (launched recently) vs frozen/underperforming.

        Checks:
        1. days_since_start <= cold_start_days (config, default 3)
        2. No historical spend (all spend metrics are 0/NaN)
        3. Not in first allocation dates (if state available)

        Args:
            row: Row from adset features DataFrame

        Returns:
            True if this is a truly new adset that should get conservative initial budget
        """
        cold_start_days = config.get_safety_rule("cold_start_days", 3)
        days_active = int(row.get("days_since_start", 0))

        # Must be recently created
        if days_active > cold_start_days:
            return False

        # Must have no spend history
        has_spend = (
            pd.notna(row.get("spend", 0))
            and row.get("spend", 0) > 0
            or pd.notna(row.get("spend_rolling_7d", 0))
            and row.get("spend_rolling_7d", 0) > 0
        )
        if has_spend:
            return False

        return True

    def _calculate_initial_budget(
        self,
        config,
        row: pd.Series,
        total_budget_today: float,
        num_adsets: int,
    ) -> float:
        """
        Calculate initial budget for a truly new adset.

        Strategy:
        1. Start with minimum budget (from config)
        2. Add fraction of equal share (configurable, default 50%)
        3. Respect max initial budget cap (from config)

        Formula:
            initial = min_budget + (equal_share * initial_fraction)
            initial = min(initial, max_initial_budget)

        Example:
            total_budget_today = $1000
            num_adsets = 10
            min_budget = $1
            initial_fraction = 0.5
            max_initial_budget = $100

            equal_share = 1000 / 10 = $100
            initial = 1 + (100 * 0.5) = $51

        Args:
            row: Row from adset features DataFrame
            total_budget_today: Total budget for today
            num_adsets: Number of adsets

        Returns:
            Initial budget for new adset
        """
        # Get config values
        min_budget = config.get_safety_rule("min_budget", 1.0)
        initial_fraction = config.get_safety_rule("new_adset_initial_fraction", 0.5)
        max_initial_budget = config.get_safety_rule(
            "new_adset_max_initial_budget", 100.0
        )

        # Calculate equal share
        equal_share = total_budget_today / max(1, num_adsets)

        # Calculate initial budget
        initial_budget = min_budget + (equal_share * initial_fraction)
        initial_budget = min(initial_budget, max_initial_budget)

        # Sanity check: don't exceed total budget
        initial_budget = min(initial_budget, total_budget_today)

        logger.debug(
            f"New adset {row.name}: min={min_budget}, equal_share={equal_share:.2f}, "
            f"initial={initial_budget:.2f}"
        )

        return initial_budget

    def _get_current_budget(
        self, config, row: pd.Series, total_budget_today: float, num_adsets: int
    ) -> float:
        """Get current budget with proper fallback logic.

        Hierarchy:
        1. adset_daily_budget (configured) - use as-is
        2. spend_rolling_7d (recent actual) - use as-is
        3. spend (today's actual) - use as-is
        4. NEW: Smart initial budget for truly new adsets
        5. Fallback: Equal allocation

        Args:
            row: Row from adset features DataFrame.
            total_budget_today: Total budget for today.
            num_adsets: Number of adsets (for fallback calculation).

        Returns:
            Current budget value.
        """
        # First try: adset_daily_budget (configured budget)
        budget = row.get("adset_daily_budget")
        if pd.notna(budget) and budget > 0:
            return float(budget)

        # Second try: spend_rolling_7d (recent actual spend)
        recent_spend = row.get("spend_rolling_7d")
        if pd.notna(recent_spend) and recent_spend > 0:
            return float(recent_spend)

        # Third try: current spend (today's actual spend)
        current_spend = row.get("spend")
        if pd.notna(current_spend) and current_spend > 0:
            return float(current_spend)

        # NEW: Check if this is a truly new adset
        if self._is_truly_new_adset(config, row):
            return self._calculate_initial_budget(
                config, row, total_budget_today, num_adsets
            )

        # Last resort: equal allocation
        return total_budget_today / num_adsets

    def _extract_metrics(
        self, config, row: pd.Series, total_budget_today: float
    ) -> dict:
        """Extract metrics from DataFrame row.

        Args:
            row: Row from adset features DataFrame.
            total_budget_today: Total budget for today.

        Returns:
            Dictionary of metrics for allocator.
        """
        # Use standardized current_budget calculation
        current_budget = self._get_current_budget(
            config, row, total_budget_today, len(row)
        )

        # Get rolling coverage (P0-2: Use rolling coverage to gate decisions)
        rolling_7d_coverage = float(row.get("rolling_7d_coverage", 1.0))
        rolling_14d_coverage = float(row.get("rolling_14d_coverage", 1.0))

        # Common metric extraction logic
        metrics = {
            "adset_id": str(row.name),
            "current_budget": current_budget,
            "previous_budget": (
                float(row.get("previous_budget", 0))
                if pd.notna(row.get("previous_budget"))
                else None
            ),
            "roas_7d": (
                float(row.get("purchase_roas_rolling_7d", 0))
                if pd.notna(row.get("purchase_roas_rolling_7d"))
                else 0.0
            ),
            "roas_trend": float(row.get("roas_trend", 0)),
            "health_score": float(row.get("health_score", 0.5)),
            "days_active": int(row.get("days_since_start", 0)),
            "total_budget_today": total_budget_today,
            # P0-2: Add rolling coverage for confidence weighting
            "rolling_7d_coverage": rolling_7d_coverage,
            "rolling_14d_coverage": rolling_14d_coverage,
        }

        # Add optional metrics if available
        optional_metrics = [
            "adset_roas",
            "campaign_roas",
            "account_roas",
            "roas_vs_adset",
            "roas_vs_campaign",
            "roas_vs_account",
            "efficiency",
            "revenue_per_impression",
            "revenue_per_click",
            "spend",
            "impressions",
            "clicks",
            "reach",
            "budget_utilization",
            "marginal_roas",
            # Shopify integration: actual revenue-based ROAS
            "shopify_roas",
            "shopify_revenue",
        ]

        for metric in optional_metrics:
            if metric in row.index and pd.notna(row[metric]):
                metrics[metric] = float(row[metric])

        return metrics

    def _archive_allocation(
        self, output_file: str, execution_date, customer: str, platform: Optional[str]
    ) -> str:
        """Archive allocation file with date suffix.

        Args:
            output_file: Path to allocation output file
            execution_date: Date of execution
            customer: Customer name
            platform: Platform name

        Returns:
            Path to archived file
        """
        from src.config.path_manager import get_path_manager

        path_manager = get_path_manager(customer, platform or "meta")
        archive_dir = path_manager.archive_dir(
            customer, platform or "meta", execution_date.strftime("%Y-%m")
        )

        output_path = Path(output_file)
        archive_name = f"{output_path.stem}_{execution_date.strftime('%Y-%m-%d')}{output_path.suffix}"
        archive_path = archive_dir / archive_name

        archive_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copy2(output_file, archive_path)
        logger.info(f"Archived allocation to: {archive_path}")
        return str(archive_path)
