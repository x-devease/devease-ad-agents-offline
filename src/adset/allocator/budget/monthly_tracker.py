"""Monthly budget tracker for daily budget calculation.

Implements conservative budget tracking to prevent over-spending monthly caps.
Uses the formula: daily_budget = (monthly_budget - spent) / remaining_days * conservative_factor
"""

from datetime import datetime
from typing import Optional

from .state_manager import MonthlyBudgetState


class MonthlyBudgetTracker:
    """Tracks monthly budget and calculates daily allocations.

    Features:
    - Conservative daily budget calculation with configurable buffer
    - Budget exhaustion detection
    - Automatic state recording and persistence
    - Remaining days calculation including current day
    """

    def __init__(
        self,
        state: MonthlyBudgetState,
        conservative_factor: float = 0.95,
    ):
        """Initialize monthly budget tracker.

        Args:
            state: MonthlyBudgetState instance
            conservative_factor: Safety buffer (default 0.95 = 5% buffer)
        """
        self.state = state
        self.conservative_factor = conservative_factor

    def calculate_daily_budget(
        self,
        execution_date: Optional[datetime] = None,
    ) -> float:
        """Calculate daily budget based on remaining monthly budget.

        Formula: daily_budget = (monthly_budget - spent) / remaining_days * conservative_factor

        The conservative factor provides a buffer to prevent:
        - Front-loading spend early in month
        - Over-spending due to daily variations
        - Unexpected costs near month-end

        Args:
            execution_date: Date of execution (defaults to now)

        Returns:
            Daily budget amount. Returns 0 if budget exhausted.

        Examples:
            >>> # Day 1 of month, $10,000 budget, nothing spent yet
            >>> tracker.calculate_daily_budget()
            322.58  # (10000 - 0) / 30 * 0.95

            >>> # Day 20 of month, $10,000 budget, $7,000 spent
            >>> tracker.calculate_daily_budget()
            159.09  # (10000 - 7000) / 11 * 0.95
        """
        if execution_date is None:
            execution_date = datetime.now()

        # Check if budget exhausted
        if self.is_budget_exhausted():
            return 0.0

        # Calculate remaining days in month (including today)
        remaining_days = self._get_remaining_days(execution_date)

        if remaining_days <= 0:
            return 0.0

        # Calculate remaining budget
        remaining_budget = (
            self.state.budget["monthly_budget_cap"] - self.state.tracking["total_spent"]
        )

        # Apply conservative factor
        daily_budget = (remaining_budget / remaining_days) * self.conservative_factor

        # Ensure non-negative
        return max(0.0, daily_budget)

    def _get_remaining_days(self, execution_date: datetime) -> int:
        """Calculate remaining days in period including current day.

        Uses the actual start date of the budget period (month_start_date) rather
        than just calendar month, to handle mid-month starts correctly.

        Args:
            execution_date: Current execution date

        Returns:
            Number of days remaining in period (including today)
        """
        import calendar

        # Calculate days actually elapsed since budget period started
        # (subtract start date from execution date, gives 0 on first day)
        days_elapsed = (execution_date.date() - self.state.month_start_date.date()).days

        # Determine period length
        if self.state.month_start_date.day == 1:
            # Standard calendar month - use actual days in month
            year = execution_date.year
            month = execution_date.month
            period_length_days = calendar.monthrange(year, month)[1]
        else:
            # Custom period (e.g., started mid-month)
            # Use configurable period_length_days (default 30)
            period_length_days = self.state.budget.get("period_length_days", 30)

        # Calculate remaining days (including today)
        remaining_days = period_length_days - days_elapsed

        return max(1, remaining_days)  # At least 1 day (today)

    def is_budget_exhausted(
        self,
        threshold: float = 0.01,
    ) -> bool:
        """Check if monthly budget is exhausted or nearly exhausted.

        Args:
            threshold: Minimum remaining budget percentage (default 1%)

        Returns:
            True if budget is exhausted below threshold
        """
        monthly_budget = self.state.budget["monthly_budget_cap"]
        total_spent = self.state.tracking["total_spent"]

        # Calculate remaining budget
        remaining_budget = monthly_budget - total_spent

        # Check if below threshold
        return remaining_budget <= (monthly_budget * threshold)

    def record_allocation(
        self,
        execution_date: datetime,
        allocated: float,
        spent: float,
        num_adsets: int,
        allocation_file: str,
        daily_budget: Optional[float] = None,
        adset_allocations: Optional[dict] = None,
    ) -> None:
        """Record an allocation execution in state.

        Args:
            execution_date: When allocation occurred
            allocated: Amount allocated today
            spent: Amount actually spent (from previous execution)
            num_adsets: Number of adsets in allocation
            allocation_file: Path to allocation output file
            daily_budget: Budget calculated for the day (auto-calculated if None)
            adset_allocations: Optional dict of {adset_id: new_budget}
        """
        if daily_budget is None:
            daily_budget = self.calculate_daily_budget(execution_date)

        # Record execution in state
        self.state.add_execution(
            execution_date=execution_date,
            daily_budget=daily_budget,
            allocated=allocated,
            spent=spent,
            num_adsets=num_adsets,
            allocation_file=allocation_file,
            adset_allocations=adset_allocations,
        )

    def get_tracking_summary(self) -> dict:
        """Get summary of current tracking status.

        Returns:
            Dict with tracking summary including:
            - month: Current tracking month
            - monthly_budget: Monthly budget cap
            - total_spent: Total amount spent
            - total_allocated: Total amount allocated
            - remaining_budget: Remaining budget amount
            - remaining_pct: Remaining budget percentage
            - days_active: Number of active days
            - days_in_month: Total days in month
            - is_exhausted: Whether budget is exhausted
        """
        monthly_budget = self.state.budget["monthly_budget_cap"]
        total_spent = self.state.tracking["total_spent"]
        remaining_budget = self.state.tracking["remaining_budget"]

        return {
            "month": self.state.month,
            "monthly_budget": monthly_budget,
            "total_spent": total_spent,
            "total_allocated": self.state.tracking["total_allocated"],
            "remaining_budget": remaining_budget,
            "remaining_pct": (
                (remaining_budget / monthly_budget * 100) if monthly_budget > 0 else 0.0
            ),
            "days_active": self.state.tracking["days_active"],
            "days_in_month": self.state.tracking["days_in_month"],
            "is_exhausted": self.is_budget_exhausted(),
        }
