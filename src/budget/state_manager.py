"""Monthly budget state management.

Handles persistent storage of monthly budget tracking data using JSON files.
Provides atomic writes to prevent corruption and automatic month rollover detection.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MonthlyBudgetState:
    """Manages monthly budget tracking state with JSON persistence.

    State file location: results/{customer}/{platform}/monthly_state_YYYY-MM.json

    The state tracks:
    - Monthly budget cap
    - Total spent and allocated amounts
    - Execution history with daily allocations
    - Auto-reset on new month
    """

    def __init__(
        self,
        customer: str,
        platform: str,
        monthly_budget: float,
        month: str,
        state_path: Path,
        month_start_date: Optional[datetime] = None,
    ):
        """Initialize monthly budget state.

        Args:
            customer: Customer name (e.g., "moprobo")
            platform: Platform name (e.g., "meta")
            monthly_budget: Monthly budget cap amount
            month: Month string in "YYYY-MM" format
            state_path: Path to state JSON file
            month_start_date: When the budget period actually started (defaults to now)
        """
        self.customer = customer
        self.platform = platform
        self.month = month
        self.state_path = state_path

        # If month_start_date not provided, use current date (start of day)
        if month_start_date is None:
            month_start_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        self.month_start_date = month_start_date

        # Initialize state structure
        self.data: Dict[str, Any] = {
            "metadata": {
                "customer": customer,
                "platform": platform,
                "month": month,
                "month_start_date": month_start_date.isoformat(),
                "last_updated": datetime.now().isoformat(),
            },
            "budget": {
                "monthly_budget_cap": monthly_budget,
                "source": "config",
            },
            "tracking": {
                "total_spent": 0.0,
                "total_allocated": 0.0,
                "remaining_budget": monthly_budget,
                "days_active": 0,
                "days_in_month": self._get_days_in_month(month),
            },
            "execution_history": [],
        }

    @property
    def tracking(self) -> Dict[str, Any]:
        """Access tracking section."""
        return self.data["tracking"]

    @property
    def budget(self) -> Dict[str, Any]:
        """Access budget section."""
        return self.data["budget"]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Access metadata section."""
        return self.data["metadata"]

    @property
    def execution_history(self) -> List[Dict[str, Any]]:
        """Access execution history."""
        return self.data["execution_history"]

    @property
    def is_first_execution(self) -> bool:
        """Check if this is the first execution of the budget period."""
        return self.tracking["days_active"] == 0

    @property
    def days_since_budget_start(self) -> int:
        """Calculate days since budget period actually started."""
        if not hasattr(self, "month_start_date") or self.month_start_date is None:
            return 0
        return (datetime.now() - self.month_start_date).days + 1

    def get_adset_first_allocation_date(self, adset_id: str) -> Optional[datetime]:
        """Get the date when an adset first received budget.

        Args:
            adset_id: Adset identifier

        Returns:
            First allocation date, or None if adset not found in tracking
        """
        dates = self.tracking.get("adset_first_allocation_dates", {})
        if adset_id in dates:
            return datetime.fromisoformat(dates[adset_id])
        return None

    def is_adset_new(self, adset_id: str, days_threshold: int = 7) -> bool:
        """
        Check if adset is considered "new" (recently first allocated).

        Args:
            adset_id: Adset identifier
            days_threshold: Days after first allocation to still be considered "new"

        Returns:
            True if adset was first allocated within days_threshold days
        """
        first_date = self.get_adset_first_allocation_date(adset_id)
        if first_date is None:
            return False  # No allocation history, not "new", just unknown

        days_since_first = (datetime.now() - first_date).days
        return days_since_first <= days_threshold

    @staticmethod
    def _get_days_in_month(month: str) -> int:
        """Get number of days in month from YYYY-MM string."""
        try:
            year, month_num = map(int, month.split("-"))
            import calendar

            return calendar.monthrange(year, month_num)[1]
        except (ValueError, IndexError):
            # Default to 30 if parsing fails
            return 30

    @staticmethod
    def load_or_create(
        customer: str,
        platform: str,
        monthly_budget: float,
        state_path: Optional[Path] = None,
    ) -> "MonthlyBudgetState":
        """Load existing state or create new for current month.

        Args:
            customer: Customer name
            platform: Platform name
            monthly_budget: Monthly budget cap
            state_path: Path to state file (auto-derived if None)

        Returns:
            MonthlyBudgetState instance (loaded or new)

        Raises:
            ValueError: If state file exists but for different customer/platform
        """
        from src.config.path_manager import get_path_manager

        # Auto-derive state path if not provided
        if state_path is None:
            path_manager = get_path_manager(customer, platform)
            current_month = datetime.now().strftime("%Y-%m")
            state_path = path_manager.monthly_state_path(
                customer=customer, platform=platform, month=current_month
            )

        # Check if state file exists
        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    saved_data = json.load(f)

                saved_month = saved_data.get("metadata", {}).get("month")
                saved_customer = saved_data.get("metadata", {}).get("customer")
                saved_platform = saved_data.get("metadata", {}).get("platform")

                # Validate customer/platform match
                if saved_customer != customer or saved_platform != platform:
                    raise ValueError(
                        f"State file belongs to {saved_customer}/{saved_platform}, "
                        f"not {customer}/{platform}"
                    )

                # Check if month rollover needed
                current_month = datetime.now().strftime("%Y-%m")
                if saved_month != current_month:
                    # Create new state for new month
                    state = MonthlyBudgetState(
                        customer=customer,
                        platform=platform,
                        monthly_budget=monthly_budget,
                        month=current_month,
                        state_path=state_path,
                    )
                else:
                    # Load existing state
                    state = MonthlyBudgetState.__new__(MonthlyBudgetState)
                    state.customer = customer
                    state.platform = platform
                    state.month = saved_month
                    state.state_path = state_path
                    state.data = saved_data

                    # Load month_start_date if available (for backward compatibility)
                    saved_start_date = saved_data.get("metadata", {}).get(
                        "month_start_date"
                    )
                    if saved_start_date:
                        state.month_start_date = datetime.fromisoformat(
                            saved_start_date
                        )
                    else:
                        # Backward compatibility: use first of month
                        state.month_start_date = datetime.strptime(
                            f"{saved_month}-01", "%Y-%m-%d"
                        )

                return state

            except (json.JSONDecodeError, KeyError) as e:
                # Corrupted state file, create new
                import warnings

                warnings.warn(
                    f"Corrupted state file at {state_path}: {e}. Creating new state."
                )

        # Create new state
        current_month = datetime.now().strftime("%Y-%m")
        return MonthlyBudgetState(
            customer=customer,
            platform=platform,
            monthly_budget=monthly_budget,
            month=current_month,
            state_path=state_path,
        )

    def should_reset_month(self) -> bool:
        """Check if month has rolled over since last execution.

        Returns:
            True if current month differs from state month
        """
        current_month = datetime.now().strftime("%Y-%m")
        return self.month != current_month

    def add_execution(
        self,
        execution_date: datetime,
        daily_budget: float,
        allocated: float,
        spent: float,
        num_adsets: int,
        allocation_file: str,
        adset_allocations: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record an allocation execution in history.

        Args:
            execution_date: When allocation occurred
            daily_budget: Budget allocated for the day
            allocated: Actual amount allocated
            spent: Actual amount spent (from previous day)
            num_adsets: Number of adsets in allocation
            allocation_file: Path to allocation output file
            adset_allocations: Optional dict of {adset_id: new_budget}
                          If provided, track first allocation dates per adset
        """
        execution_record = {
            "date": execution_date.strftime("%Y-%m-%d"),
            "daily_budget": daily_budget,
            "allocated": allocated,
            "spent": spent,
            "num_adsets": num_adsets,
            "allocation_file": allocation_file,
        }

        self.execution_history.append(execution_record)

        # NEW: Track first allocation dates per adset
        if adset_allocations:
            if "adset_first_allocation_dates" not in self.tracking:
                self.tracking["adset_first_allocation_dates"] = {}

            for adset_id in adset_allocations.keys():
                # Only record if not already tracked (first allocation)
                if adset_id not in self.tracking["adset_first_allocation_dates"]:
                    self.tracking["adset_first_allocation_dates"][
                        adset_id
                    ] = execution_date.isoformat()

        # Update tracking totals
        self.tracking["total_allocated"] += allocated
        self.tracking["total_spent"] += spent
        self.tracking["days_active"] = self._get_unique_days_active()
        self.tracking["remaining_budget"] = (
            self.budget["monthly_budget_cap"] - self.tracking["total_spent"]
        )

        # Update metadata timestamp
        self.metadata["last_updated"] = datetime.now().isoformat()

    def _get_unique_days_active(self) -> int:
        """Count unique days with executions."""
        unique_dates = set(record.get("date", "") for record in self.execution_history)
        return len(unique_dates)

    def save(self) -> None:
        """Atomically save state to JSON file.

        Uses write-and-rename pattern to prevent corruption.
        Creates parent directories if needed.
        """
        # Ensure parent directory exists
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first
        temp_path = self.state_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self.data, f, indent=2)

        # Atomic rename
        temp_path.replace(self.state_path)

    def reset_month(self, new_monthly_budget: float) -> None:
        """Reset state for new month.

        Preserves customer/platform but resets all tracking.

        Args:
            new_monthly_budget: New month's budget cap
        """
        current_month = datetime.now().strftime("%Y-%m")
        self.month = current_month

        # Reset month_start_date to first of new month
        self.month_start_date = datetime.strptime(f"{current_month}-01", "%Y-%m-%d")

        # Reset tracking data
        self.data["tracking"] = {
            "total_spent": 0.0,
            "total_allocated": 0.0,
            "remaining_budget": new_monthly_budget,
            "days_active": 0,
            "days_in_month": self._get_days_in_month(current_month),
        }

        # Clear execution history
        self.data["execution_history"] = []

        # Update budget
        self.budget["monthly_budget_cap"] = new_monthly_budget

        # Update metadata
        self.metadata["month"] = current_month
        self.metadata["month_start_date"] = self.month_start_date.isoformat()
        self.metadata["last_updated"] = datetime.now().isoformat()
