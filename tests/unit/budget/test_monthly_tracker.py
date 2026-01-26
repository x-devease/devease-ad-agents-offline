"""Unit tests for MonthlyBudgetTracker."""

from datetime import datetime
from pathlib import Path

import pytest

from src.meta.adset.allocator.budget.monthly_tracker import MonthlyBudgetTracker
from src.meta.adset.allocator.budget.state_manager import MonthlyBudgetState


class TestMonthlyBudgetTracker:
    """Test MonthlyBudgetTracker class."""

    def test_initialization(self):
        """Test MonthlyBudgetTracker initialization."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state, conservative_factor=0.95)

        assert tracker.state is state
        assert tracker.conservative_factor == 0.95

    def test_default_conservative_factor(self):
        """Test default conservative_factor is 0.95."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state)

        assert tracker.conservative_factor == 0.95


class TestCalculateDailyBudget:
    """Test calculate_daily_budget method."""

    def test_calculate_daily_budget_start_of_month(self):
        """Test daily budget calculation at start of month."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2026, 1, 1),
        )

        tracker = MonthlyBudgetTracker(state, conservative_factor=0.95)

        # Day 1 of January (31 days)
        # Formula: (10000 - 0) / 31 * 0.95 = 306.45
        execution_date = datetime(2026, 1, 1)
        daily_budget = tracker.calculate_daily_budget(execution_date)

        expected = (10000.0 - 0.0) / 31 * 0.95
        assert abs(daily_budget - expected) < 0.01

    def test_calculate_daily_budget_mid_month(self):
        """Test daily budget calculation mid-month."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2026, 1, 1),
        )

        tracker = MonthlyBudgetTracker(state, conservative_factor=0.95)

        # Add some spent amount (simulating mid-month)
        state.tracking["total_spent"] = 5000.0

        # Day 15 of January (17 days remaining including today)
        # Formula: (10000 - 5000) / 17 * 0.95 = 279.41
        execution_date = datetime(2026, 1, 15)
        daily_budget = tracker.calculate_daily_budget(execution_date)

        remaining_days = 31 - 15 + 1  # 17 days
        expected = (10000.0 - 5000.0) / remaining_days * 0.95
        assert abs(daily_budget - expected) < 0.01

    def test_calculate_daily_budget_end_of_month(self):
        """Test daily budget calculation near end of month."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2026, 1, 1),
        )

        tracker = MonthlyBudgetTracker(state, conservative_factor=0.95)

        # Add some spent amount
        state.tracking["total_spent"] = 9500.0

        # Day 30 of January (2 days remaining including today)
        # Formula: (10000 - 9500) / 2 * 0.95 = 237.50
        execution_date = datetime(2026, 1, 30)
        daily_budget = tracker.calculate_daily_budget(execution_date)

        remaining_days = 31 - 30 + 1  # 2 days
        expected = (10000.0 - 9500.0) / remaining_days * 0.95
        assert abs(daily_budget - expected) < 0.01

    def test_calculate_daily_budget_returns_zero_when_exhausted(self):
        """Test calculate_daily_budget returns 0 when budget exhausted."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state, conservative_factor=0.95)

        # Spend more than monthly budget
        state.tracking["total_spent"] = 10000.0

        execution_date = datetime(2026, 1, 15)
        daily_budget = tracker.calculate_daily_budget(execution_date)

        assert daily_budget == 0.0

    def test_calculate_daily_budget_custom_conservative_factor(self):
        """Test calculate_daily_budget with custom conservative factor."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2026, 1, 1),
        )

        tracker = MonthlyBudgetTracker(state, conservative_factor=0.90)

        # Day 1 of January (31 days)
        # Formula: (10000 - 0) / 31 * 0.90 = 290.32
        execution_date = datetime(2026, 1, 1)
        daily_budget = tracker.calculate_daily_budget(execution_date)

        expected = (10000.0 - 0.0) / 31 * 0.90
        assert abs(daily_budget - expected) < 0.01

    def test_calculate_daily_budget_negative_is_clamped_to_zero(self):
        """Test calculate_daily_budget clamps negative values to zero."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state, conservative_factor=0.95)

        # Spend more than monthly budget
        state.tracking["total_spent"] = 11000.0

        execution_date = datetime(2026, 1, 15)
        daily_budget = tracker.calculate_daily_budget(execution_date)

        assert daily_budget == 0.0


class TestGetRemainingDays:
    """Test _get_remaining_days method."""

    def test_get_remaining_days_first_of_month(self):
        """Test _get_remaining_days on first day of month."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2026, 1, 1),
        )

        tracker = MonthlyBudgetTracker(state)

        execution_date = datetime(2026, 1, 1)
        remaining_days = tracker._get_remaining_days(execution_date)

        # January has 31 days, so remaining = 31 - 1 + 1 = 31
        assert remaining_days == 31

    def test_get_remaining_days_mid_month(self):
        """Test _get_remaining_days mid-month."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2026, 1, 1),
        )

        tracker = MonthlyBudgetTracker(state)

        execution_date = datetime(2026, 1, 15)
        remaining_days = tracker._get_remaining_days(execution_date)

        # January has 31 days, so remaining = 31 - 15 + 1 = 17
        assert remaining_days == 17

    def test_get_remaining_days_last_day_of_month(self):
        """Test _get_remaining_days on last day of month."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2026, 1, 1),
        )

        tracker = MonthlyBudgetTracker(state)

        execution_date = datetime(2026, 1, 31)
        remaining_days = tracker._get_remaining_days(execution_date)

        # January has 31 days, so remaining = 31 - 31 + 1 = 1
        assert remaining_days == 1

    def test_get_remaining_days_february_non_leap(self):
        """Test _get_remaining_days for February (non-leap year)."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-02",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2026, 2, 1),
        )

        tracker = MonthlyBudgetTracker(state)

        execution_date = datetime(2026, 2, 15)
        remaining_days = tracker._get_remaining_days(execution_date)

        # February 2026 has 28 days, so remaining = 28 - 15 + 1 = 14
        assert remaining_days == 14

    def test_get_remaining_days_february_leap_year(self):
        """Test _get_remaining_days for February (leap year)."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2024-02",
            state_path=Path("/tmp/test.json"),
            month_start_date=datetime(2024, 2, 1),
        )

        tracker = MonthlyBudgetTracker(state)

        execution_date = datetime(2024, 2, 15)
        remaining_days = tracker._get_remaining_days(execution_date)

        # February 2024 has 29 days (leap year), so remaining = 29 - 15 + 1 = 15
        assert remaining_days == 15


class TestIsBudgetExhausted:
    """Test is_budget_exhausted method."""

    def test_is_budget_exhausted_false_when_full_budget(self):
        """Test is_budget_exhausted returns False when budget is full."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state)

        assert tracker.is_budget_exhausted() is False

    def test_is_budget_exhausted_true_when_spent_equals_budget(self):
        """Test is_budget_exhausted returns True when spent equals budget."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state)

        # Spend exactly the budget
        state.tracking["total_spent"] = 10000.0

        assert tracker.is_budget_exhausted() is True

    def test_is_budget_exhausted_true_when_near_budget(self):
        """Test is_budget_exhausted returns True when near threshold."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state)

        # Spend 99% of budget (below default 1% threshold)
        state.tracking["total_spent"] = 9900.0

        assert tracker.is_budget_exhausted() is True

    def test_is_budget_exhausted_false_when_above_threshold(self):
        """Test is_budget_exhausted returns False when above threshold."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state)

        # Spend 90% of budget (above 1% threshold)
        state.tracking["total_spent"] = 9000.0

        assert tracker.is_budget_exhausted() is False

    def test_is_budget_exhausted_custom_threshold(self):
        """Test is_budget_exhausted with custom threshold."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test.json"),
        )

        tracker = MonthlyBudgetTracker(state)

        # Spend 95% of budget
        state.tracking["total_spent"] = 9500.0

        # Default threshold (1%) should return False
        assert tracker.is_budget_exhausted(threshold=0.01) is False

        # 5% threshold should return True
        assert tracker.is_budget_exhausted(threshold=0.05) is True


class TestRecordAllocation:
    """Test record_allocation method."""

    def test_record_allocation(self, tmp_path):
        """Test recording an allocation updates state."""
        state_path = tmp_path / "state.json"
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=state_path,
        )

        tracker = MonthlyBudgetTracker(state)

        execution_date = datetime(2026, 1, 1)
        tracker.record_allocation(
            execution_date=execution_date,
            allocated=320.0,
            spent=315.50,
            num_adsets=45,
            allocation_file="/tmp/alloc.csv",
        )

        # Verify state updated
        assert len(state.execution_history) == 1
        assert state.tracking["total_spent"] == 315.50
        assert state.tracking["total_allocated"] == 320.0
        assert state.tracking["days_active"] == 1

        # Verify execution record
        record = state.execution_history[0]
        assert record["date"] == "2026-01-01"
        assert record["allocated"] == 320.0
        assert record["spent"] == 315.50

    def test_record_allocation_with_custom_daily_budget(self, tmp_path):
        """Test record_allocation with explicit daily_budget."""
        state_path = tmp_path / "state.json"
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=state_path,
        )

        tracker = MonthlyBudgetTracker(state)

        execution_date = datetime(2026, 1, 1)
        tracker.record_allocation(
            execution_date=execution_date,
            allocated=320.0,
            spent=315.50,
            num_adsets=45,
            allocation_file="/tmp/alloc.csv",
            daily_budget=350.0,  # Explicit daily budget
        )

        # Verify execution record uses explicit daily_budget
        record = state.execution_history[0]
        assert record["daily_budget"] == 350.0


class TestGetTrackingSummary:
    """Test get_tracking_summary method."""

    def test_get_tracking_summary(self, tmp_path):
        """Test get_tracking_summary returns correct summary."""
        state_path = tmp_path / "state.json"
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=state_path,
        )

        tracker = MonthlyBudgetTracker(state)

        # Add some spend
        state.tracking["total_spent"] = 5000.0
        state.tracking["total_allocated"] = 5200.0
        state.tracking["remaining_budget"] = 5000.0  # Manually update remaining
        state.tracking["days_active"] = 15

        summary = tracker.get_tracking_summary()

        assert summary["month"] == "2026-01"
        assert summary["monthly_budget"] == 10000.0
        assert summary["total_spent"] == 5000.0
        assert summary["total_allocated"] == 5200.0
        assert summary["remaining_budget"] == 5000.0
        assert summary["remaining_pct"] == 50.0
        assert summary["days_active"] == 15
        assert summary["days_in_month"] == 31
        assert summary["is_exhausted"] is False

    def test_get_tracking_summary_exhausted(self, tmp_path):
        """Test get_tracking_summary when budget is exhausted."""
        state_path = tmp_path / "state.json"
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=state_path,
        )

        tracker = MonthlyBudgetTracker(state)

        # Spend entire budget
        state.tracking["total_spent"] = 10000.0
        state.tracking["remaining_budget"] = 0.0  # Manually update remaining

        summary = tracker.get_tracking_summary()

        assert summary["remaining_budget"] == 0.0
        assert summary["remaining_pct"] == 0.0
        assert summary["is_exhausted"] is True

    def test_get_tracking_summary_zero_monthly_budget(self, tmp_path):
        """Test get_tracking_summary handles zero monthly budget."""
        state_path = tmp_path / "state.json"
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=0.0,
            month="2026-01",
            state_path=state_path,
        )

        tracker = MonthlyBudgetTracker(state)

        summary = tracker.get_tracking_summary()

        assert summary["remaining_pct"] == 0.0
