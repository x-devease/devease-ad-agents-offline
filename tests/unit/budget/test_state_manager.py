"""Unit tests for MonthlyBudgetState."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.budget.state_manager import MonthlyBudgetState


class TestMonthlyBudgetState:
    """Test MonthlyBudgetState class."""

    def test_initialization(self):
        """Test MonthlyBudgetState initialization."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test_state.json"),
        )

        assert state.customer == "test_customer"
        assert state.platform == "meta"
        assert state.month == "2026-01"
        assert state.budget["monthly_budget_cap"] == 10000.0
        assert state.tracking["total_spent"] == 0.0
        assert state.tracking["total_allocated"] == 0.0
        assert state.tracking["remaining_budget"] == 10000.0
        assert state.tracking["days_active"] == 0
        assert len(state.execution_history) == 0

    def test_tracking_property(self):
        """Test tracking property access."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test_state.json"),
        )

        tracking = state.tracking
        assert tracking["total_spent"] == 0.0
        # monthly_budget_cap is in budget, not tracking
        assert "monthly_budget_cap" not in tracking

    def test_budget_property(self):
        """Test budget property access."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test_state.json"),
        )

        budget = state.budget
        assert budget["monthly_budget_cap"] == 10000.0

    def test_metadata_property(self):
        """Test metadata property access."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test_state.json"),
        )

        metadata = state.metadata
        assert metadata["customer"] == "test_customer"
        assert metadata["platform"] == "meta"
        assert metadata["month"] == "2026-01"
        assert "last_updated" in metadata

    def test_execution_history_property(self):
        """Test execution_history property access."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=Path("/tmp/test_state.json"),
        )

        history = state.execution_history
        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_days_in_month(self):
        """Test _get_days_in_month static method."""
        # January (31 days)
        assert MonthlyBudgetState._get_days_in_month("2026-01") == 31

        # February 2026 (28 days, not leap year)
        assert MonthlyBudgetState._get_days_in_month("2026-02") == 28

        # February 2024 (29 days, leap year)
        assert MonthlyBudgetState._get_days_in_month("2024-02") == 29

        # April (30 days)
        assert MonthlyBudgetState._get_days_in_month("2026-04") == 30

    def test_get_days_in_month_invalid_format(self):
        """Test _get_days_in_month with invalid format returns default."""
        # Invalid format should default to 30
        assert MonthlyBudgetState._get_days_in_month("invalid") == 30


class TestMonthlyBudgetStateLoadOrCreate:
    """Test MonthlyBudgetState.load_or_create method."""

    def test_load_or_create_creates_new_state(self, tmp_path):
        """Test load_or_create creates new state when none exists."""
        state_path = tmp_path / "monthly_state_2026-01.json"

        state = MonthlyBudgetState.load_or_create(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            state_path=state_path,
        )

        assert state.customer == "test_customer"
        assert state.platform == "meta"
        assert state.budget["monthly_budget_cap"] == 10000.0
        assert state.tracking["total_spent"] == 0.0

    def test_load_or_create_loads_existing_state(self, tmp_path):
        """Test load_or_create loads existing state file."""
        state_path = tmp_path / "monthly_state_2026-01.json"

        # Create initial state
        state1 = MonthlyBudgetState.load_or_create(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            state_path=state_path,
        )
        state1.add_execution(
            execution_date=datetime(2026, 1, 1),
            daily_budget=322.58,
            allocated=320.0,
            spent=315.50,
            num_adsets=45,
            allocation_file="/tmp/alloc_2026-01-01.csv",
        )
        state1.save()

        # Load existing state
        state2 = MonthlyBudgetState.load_or_create(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            state_path=state_path,
        )

        assert state2.tracking["total_spent"] == 315.50
        assert state2.tracking["total_allocated"] == 320.0
        assert state2.tracking["days_active"] == 1
        assert len(state2.execution_history) == 1

    def test_load_or_create_wrong_customer_raises_error(self, tmp_path):
        """Test load_or_create raises error for different customer."""
        state_path = tmp_path / "monthly_state_2026-01.json"

        # Create state for customer1
        state1 = MonthlyBudgetState.load_or_create(
            customer="customer1",
            platform="meta",
            monthly_budget=10000.0,
            state_path=state_path,
        )
        state1.save()

        # Try to load for customer2 (should raise error)
        with pytest.raises(ValueError, match="belongs to customer1"):
            MonthlyBudgetState.load_or_create(
                customer="customer2",
                platform="meta",
                monthly_budget=10000.0,
                state_path=state_path,
            )

    def test_load_or_create_corrupted_state_creates_new(self, tmp_path):
        """Test load_or_create handles corrupted state file."""
        state_path = tmp_path / "monthly_state_2026-01.json"

        # Create corrupted JSON file
        state_path.write_text("{invalid json content")

        # Should create new state with warning
        with pytest.warns(UserWarning, match="Corrupted state file"):
            state = MonthlyBudgetState.load_or_create(
                customer="test_customer",
                platform="meta",
                monthly_budget=10000.0,
                state_path=state_path,
            )

        assert state.tracking["total_spent"] == 0.0


class TestMonthlyBudgetStateAddExecution:
    """Test MonthlyBudgetState.add_execution method."""

    def test_add_execution_single(self, tmp_path):
        """Test adding a single execution record."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=tmp_path / "state.json",
        )

        state.add_execution(
            execution_date=datetime(2026, 1, 1),
            daily_budget=322.58,
            allocated=320.0,
            spent=315.50,
            num_adsets=45,
            allocation_file="/tmp/alloc_2026-01-01.csv",
        )

        assert len(state.execution_history) == 1
        assert state.tracking["total_spent"] == 315.50
        assert state.tracking["total_allocated"] == 320.0
        assert state.tracking["days_active"] == 1
        assert state.tracking["remaining_budget"] == 10000.0 - 315.50

        # Check execution record
        record = state.execution_history[0]
        assert record["date"] == "2026-01-01"
        assert record["daily_budget"] == 322.58
        assert record["allocated"] == 320.0
        assert record["spent"] == 315.50
        assert record["num_adsets"] == 45
        assert record["allocation_file"] == "/tmp/alloc_2026-01-01.csv"

    def test_add_execution_multiple(self, tmp_path):
        """Test adding multiple execution records."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=tmp_path / "state.json",
        )

        # Add first execution
        state.add_execution(
            execution_date=datetime(2026, 1, 1),
            daily_budget=322.58,
            allocated=320.0,
            spent=315.50,
            num_adsets=45,
            allocation_file="/tmp/alloc_2026-01-01.csv",
        )

        # Add second execution
        state.add_execution(
            execution_date=datetime(2026, 1, 2),
            daily_budget=325.0,
            allocated=323.0,
            spent=318.0,
            num_adsets=46,
            allocation_file="/tmp/alloc_2026-01-02.csv",
        )

        assert len(state.execution_history) == 2
        assert state.tracking["total_spent"] == 315.50 + 318.0
        assert state.tracking["total_allocated"] == 320.0 + 323.0
        assert state.tracking["days_active"] == 2


class TestMonthlyBudgetStateSave:
    """Test MonthlyBudgetState.save method."""

    def test_save_creates_file(self, tmp_path):
        """Test save creates JSON file."""
        state_path = tmp_path / "monthly_state_2026-01.json"
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=state_path,
        )

        state.save()

        assert state_path.exists()

    def test_save_creates_parent_directories(self, tmp_path):
        """Test save creates parent directories."""
        state_path = (
            tmp_path
            / "results"
            / "test_customer"
            / "meta"
            / "monthly_state_2026-01.json"
        )
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=state_path,
        )

        state.save()

        assert state_path.exists()
        assert state_path.parent.exists()

    def test_save_atomic_write(self, tmp_path):
        """Test save uses atomic write (no .tmp file left)."""
        state_path = tmp_path / "monthly_state_2026-01.json"
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=state_path,
        )

        state.add_execution(
            execution_date=datetime(2026, 1, 1),
            daily_budget=322.58,
            allocated=320.0,
            spent=315.50,
            num_adsets=45,
            allocation_file="/tmp/alloc.csv",
        )
        state.save()

        # Check no .tmp file exists
        tmp_file = tmp_path / "monthly_state_2026-01.json.tmp"
        assert not tmp_file.exists()

        # Verify state file was created
        assert state_path.exists()

        # Load and verify content
        with open(state_path, "r") as f:
            saved_data = json.load(f)

        assert saved_data["tracking"]["total_spent"] == 315.50
        assert len(saved_data["execution_history"]) == 1

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test save and load roundtrip preserves data."""
        state_path = tmp_path / "monthly_state_2026-01.json"

        # Create and save state
        state1 = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=state_path,
        )
        state1.add_execution(
            execution_date=datetime(2026, 1, 1),
            daily_budget=322.58,
            allocated=320.0,
            spent=315.50,
            num_adsets=45,
            allocation_file="/tmp/alloc.csv",
        )
        state1.save()

        # Load state
        state2 = MonthlyBudgetState.load_or_create(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            state_path=state_path,
        )

        # Verify data preserved
        assert state2.tracking["total_spent"] == state1.tracking["total_spent"]
        assert state2.tracking["total_allocated"] == state1.tracking["total_allocated"]
        assert len(state2.execution_history) == len(state1.execution_history)


class TestMonthlyBudgetStateReset:
    """Test MonthlyBudgetState.reset_month method."""

    def test_reset_month_clears_tracking(self, tmp_path):
        """Test reset_month clears tracking data."""
        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2026-01",
            state_path=tmp_path / "state.json",
        )

        # Add some execution history
        state.add_execution(
            execution_date=datetime(2026, 1, 1),
            daily_budget=322.58,
            allocated=320.0,
            spent=315.50,
            num_adsets=45,
            allocation_file="/tmp/alloc.csv",
        )

        assert state.tracking["total_spent"] == 315.50
        assert len(state.execution_history) == 1

        # Reset month
        state.reset_month(new_monthly_budget=12000.0)

        # Verify tracking cleared
        assert state.tracking["total_spent"] == 0.0
        assert state.tracking["total_allocated"] == 0.0
        assert state.tracking["remaining_budget"] == 12000.0
        assert state.tracking["days_active"] == 0
        assert len(state.execution_history) == 0
        assert state.budget["monthly_budget_cap"] == 12000.0

    def test_should_reset_month(self):
        """Test should_reset_month detects month rollover."""
        from unittest.mock import patch

        state = MonthlyBudgetState(
            customer="test_customer",
            platform="meta",
            monthly_budget=10000.0,
            month="2025-12",
            state_path=Path("/tmp/test.json"),
        )

        # Mock datetime.now() to return January 2026
        with patch("src.budget.state_manager.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01"

            # Should detect rollover
            assert state.should_reset_month() is True

        # Reset to current month
        state.reset_month(10000.0)
        assert state.month == "2026-01"

        # Should not detect rollover anymore
        assert state.should_reset_month() is False
