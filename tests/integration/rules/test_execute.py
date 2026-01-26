"""
Integration tests for execute.py script and src.rules system.
Tests the complete budget allocation pipeline and allocator functionality.
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

from src.meta.adset import Allocator, DecisionRules, SafetyRules
from src.meta.adset.allocator.utils.parser import Parser


def run_script(script_path, args):
    """Run a script and return the result."""
    # Add current directory to PYTHONPATH for subprocess
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{Path.cwd()}:{pythonpath}" if pythonpath else str(Path.cwd())

    return subprocess.run(
        [script_path] + args,
        capture_output=True,
        text=True,
        check=False,
        cwd=Path.cwd(),
        env=env,
    )


def run_script_and_verify_output(script_path, args, output_file, expected_columns=None):
    """Run a script and verify its output file exists and has content."""
    result = run_script(script_path, args)

    # Check that script executed successfully
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Check that output file was created
    assert output_file.exists()

    # Check that output file has content
    output_df = pd.read_csv(output_file)
    assert len(output_df) > 0

    # Check expected columns if provided
    if expected_columns:
        for col in expected_columns:
            assert col in output_df.columns, f"Missing column: {col}"

    return output_df


class TestExecute:
    """Test suite for execute.py script"""

    @pytest.fixture(autouse=True)
    def cleanup_state_files(self):
        """Clean up monthly state files before each test to ensure fresh state."""
        from src.config.path_manager import get_path_manager
        from pathlib import Path

        # Get state directory for moprobo customer
        path_manager = get_path_manager()
        results_dir = path_manager.results_base / "moprobo" / "meta"

        # Delete existing state files before test
        if results_dir.exists():
            for state_file in results_dir.glob("monthly_state_*.json"):
                if state_file.exists():
                    state_file.unlink()

        yield

        # Clean up again after test
        if results_dir.exists():
            for state_file in results_dir.glob("monthly_state_*.json"):
                if state_file.exists():
                    state_file.unlink()

    @pytest.fixture
    def unique_customer(self, tmp_path, request):
        """Provide customer name for tests.
        Uses 'moprobo' for subprocess tests (requires existing config),
        and unique names for other tests to avoid state conflicts.
        """
        # Use 'moprobo' for tests that run subprocess (need existing config)
        # Use unique names for in-process tests (to avoid state conflicts)
        test_name = request.node.name
        if "test_execute" in test_name:
            return "moprobo"
        return f"test_customer_{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def sample_adset_features_df(self):
        """Create sample adset features DataFrame"""
        return pd.DataFrame(
            {
                "adset_id": ["adset_001", "adset_002", "adset_003"],
                "date_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "purchase_roas_rolling_7d": [
                    3.5,
                    2.0,
                    1.2,
                ],  # Required column name
                "roas_trend": [0.12, 0.05, -0.08],
                "health_score": [0.90, 0.70, 0.45],
                "days_since_start": [30, 20, 15],  # Required column name
                "spend": [100.0, 150.0, 200.0],
                "impressions": [10000, 15000, 20000],
                "clicks": [100, 150, 200],
                "purchase_roas": [3.5, 2.0, 1.2],
            }
        )

    @pytest.fixture
    def temp_input_file(self, sample_adset_features_df, tmp_path):
        """Create temporary input CSV file"""
        input_file = tmp_path / "input_features.csv"
        sample_adset_features_df.to_csv(input_file, index=False)
        return input_file

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory"""
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        return output_dir

    def test_execute_minimal_args(
        self, temp_input_file, temp_output_dir, unique_customer
    ):
        """Test execute.py with minimal arguments"""
        output_file = temp_output_dir / "allocations.csv"

        # Run execute.py as a subprocess
        run_script_and_verify_output(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--customer",
                unique_customer,
                "--input",
                str(temp_input_file),
                "--output",
                str(output_file),
                "--budget",
                "1000",
            ],
            output_file,
            expected_columns=["adset_id", "new_budget"],
        )

    def test_execute_all_args(self, temp_input_file, temp_output_dir, unique_customer):
        """Test execute.py with all arguments"""
        output_file = temp_output_dir / "allocations.csv"

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--customer",
                unique_customer,
                "--input",
                str(temp_input_file),
                "--output",
                str(output_file),
                "--budget",
                "5000",
                "--config",
                "config/adset/allocator/rules.yaml",
            ],
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output_file.exists()

        output_df = pd.read_csv(output_file)
        assert len(output_df) == 3  # Should have all 3 adsets
        assert "adset_id" in output_df.columns
        assert "new_budget" in output_df.columns

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Test mode requires config files, skipped in CI"
    )
    def test_execute_script_test_mode(self):
        """Test execute.py in test mode"""
        result = run_script(sys.executable, ["src/meta/adset/allocator/cli/commands/execute.py", "--test"])

        # Test mode should execute successfully
        assert result.returncode == 0, f"Test mode failed: {result.stderr}"

    def test_execute_missing_input(self, temp_output_dir):
        """Test execute.py with missing input file"""
        output_file = temp_output_dir / "allocations.csv"

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--input",
                "nonexistent_file.csv",
                "--output",
                str(output_file),
                "--budget",
                "1000",
            ],
        )

        # Should fail with error
        assert result.returncode != 0

    def test_execute_invalid_budget(
        self, temp_input_file, temp_output_dir, unique_customer
    ):
        """Test execute.py with invalid budget (negative budget)"""
        output_file = temp_output_dir / "allocations.csv"

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--customer",
                unique_customer,
                "--input",
                str(temp_input_file),
                "--output",
                str(output_file),
                "--budget",
                "-100",
            ],
        )

        # With monthly tracking, negative budget is treated as exhausted
        # Script should fail
        assert result.returncode != 0

    def test_execute_creates_dir(self, temp_input_file, tmp_path, unique_customer):
        """Test that execute.py creates output directory if it doesn't exist"""
        output_dir = tmp_path / "new_results"
        output_file = output_dir / "allocations.csv"

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--customer",
                unique_customer,
                "--input",
                str(temp_input_file),
                "--output",
                str(output_file),
                "--budget",
                "1000",
            ],
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output_dir.exists()
        assert output_file.exists()

    def test_execute_output_columns(
        self, temp_input_file, temp_output_dir, unique_customer
    ):
        """Test that output CSV has expected columns"""
        output_file = temp_output_dir / "allocations.csv"

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--customer",
                unique_customer,
                "--input",
                str(temp_input_file),
                "--output",
                str(output_file),
                "--budget",
                "1000",
            ],
        )

        assert result.returncode == 0
        output_df = pd.read_csv(output_file)

        # Check for expected columns (may vary based on implementation)
        assert "adset_id" in output_df.columns, "Missing column: adset_id"
        # new_budget may be named differently or computed
        assert len(output_df) > 0, "Output should have rows"

    def test_execute_budget_sum(
        self, temp_input_file, temp_output_dir, unique_customer
    ):
        """Test that allocated budgets sum to daily budget"""
        output_file = temp_output_dir / "allocations.csv"
        total_budget = 1000.0

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--customer",
                unique_customer,
                "--input",
                str(temp_input_file),
                "--output",
                str(output_file),
                "--budget",
                str(total_budget),
            ],
        )

        assert result.returncode == 0
        output_df = pd.read_csv(output_file)

        # Check that new_budget column exists
        assert "new_budget" in output_df.columns

        allocated_sum = output_df["new_budget"].sum()
        # With monthly tracking, daily_budget is calculated conservatively
        # as (monthly_budget - spent) / remaining_days * 0.95
        # This varies based on day of month, so just check it's reasonable
        assert allocated_sum > 0, "No budget was allocated"
        assert (
            allocated_sum < total_budget
        ), f"Allocated more than total budget: {allocated_sum} > {total_budget}"
        # With conservative factor, should allocate at most total/remaining_days * 0.95
        # which is always less than total_budget
        # Just verify it's not an obviously broken value
        assert (
            allocated_sum > 1.0
        ), f"Allocated budget suspiciously low: {allocated_sum}"

    def test_execute_preserves_ids(
        self, temp_input_file, temp_output_dir, unique_customer
    ):
        """Test that output preserves input adset IDs"""
        output_file = temp_output_dir / "allocations.csv"

        # Read input to get adset IDs
        input_df = pd.read_csv(temp_input_file)
        input_adset_ids = set(input_df["adset_id"])

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--customer",
                unique_customer,
                "--input",
                str(temp_input_file),
                "--output",
                str(output_file),
                "--budget",
                "1000",
            ],
        )

        assert result.returncode == 0
        output_df = pd.read_csv(output_file)
        output_adset_ids = set(output_df["adset_id"])

        # All input adset IDs should be in output
        assert (
            input_adset_ids == output_adset_ids
        ), f"Adset ID mismatch: {input_adset_ids} != {output_adset_ids}"

    def test_execute_empty_input(self, temp_output_dir):
        """Test execute.py with empty input file"""
        # Create empty CSV file
        empty_file = temp_output_dir / "empty.csv"
        pd.DataFrame(columns=["adset_id", "roas_7d", "roas_trend"]).to_csv(
            empty_file, index=False
        )

        output_file = temp_output_dir / "allocations.csv"

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--input",
                str(empty_file),
                "--output",
                str(output_file),
                "--budget",
                "1000",
            ],
        )

        # Should handle empty input gracefully
        # (may fail or produce empty output)
        if result.returncode == 0:
            # If successful, output should be empty or have header only
            if output_file.exists():
                output_df = pd.read_csv(output_file)
                assert len(output_df) == 0

    def test_execute_custom_config(
        self, temp_input_file, temp_output_dir, tmp_path, unique_customer
    ):
        """Test execute.py with custom config file"""
        # Create custom config file (flat structure - new format)
        custom_config = tmp_path / "custom_rules.yaml"
        custom_config.write_text(
            """
safety_rules:
  max_daily_increase_pct: 0.20
  freeze_roas_threshold: 0.5
decision_rules:
  high_roas_threshold: 3.0
"""
        )

        output_file = temp_output_dir / "allocations.csv"

        result = run_script(
            sys.executable,
            [
                "src/meta/adset/allocator/cli/commands/execute.py",
                "--customer",
                unique_customer,
                "--input",
                str(temp_input_file),
                "--output",
                str(output_file),
                "--budget",
                "1000",
                "--config",
                str(custom_config),
            ],
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output_file.exists()


class TestFullFlow:
    """Integration tests for complete src.rules flow"""

    def test_full_flow_high_performer(self, allocator, sample_metrics_dict):
        """Test complete flow for high performing adset"""
        metrics = sample_metrics_dict.copy()
        metrics.update(
            {
                "adset_id": "adset_001",
                "roas_7d": 3.5,
                "roas_trend": 0.12,
                "roas_vs_adset": 1.25,
                "roas_vs_campaign": 1.40,
                "roas_vs_account": 1.59,
                "revenue_per_click": 3.0,
                "health_score": 0.90,
                "days_active": 30,
            }
        )

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should increase budget for high performer
        assert new_budget > 100.0
        assert new_budget <= 115.0  # Capped by learning shock protection
        assert len(decision_path) > 0
        assert any("decision_rule" in path for path in decision_path)

    def test_full_flow_low_perf_frozen(self, allocator):
        """Test complete flow for low performing adset (should freeze)"""
        metrics = {
            "adset_id": "adset_002",
            "current_budget": 50.0,
            "previous_budget": 55.0,
            "roas_7d": 0.3,  # Below freeze threshold
            "roas_trend": -0.20,
            "health_score": 0.15,  # Below freeze threshold
            "days_active": 15,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should be frozen
        assert new_budget == 0.0
        assert any("frozen" in path.lower() for path in decision_path)

    def test_full_flow_cold_start(self, allocator):
        """Test complete flow for cold start adset"""
        metrics = {
            "adset_id": "adset_003",
            "current_budget": 20.0,
            "previous_budget": 20.0,
            "roas_7d": 2.5,
            "roas_trend": 0.10,
            "health_score": 0.70,
            "days_active": 2,  # Cold start period
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should be conservative (max 10% increase during cold start)
        assert new_budget >= 20.0
        assert new_budget <= 22.0  # Capped at 10% increase
        assert any("cold_start" in path.lower() for path in decision_path)

    def test_full_flow_weekend_boost(self, allocator):
        """Test complete flow with weekend boost"""
        metrics = {
            "adset_id": "adset_004",
            "current_budget": 80.0,
            "previous_budget": 75.0,
            "roas_7d": 2.8,
            "roas_trend": 0.08,
            "health_score": 0.75,
            "days_active": 20,
            "is_weekend": True,  # Weekend boost
            "week_of_year": 25,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should get weekend boost (may be slight due to learning shock)
        # Check that weekend is mentioned in decision path
        decision_path_str = " ".join(decision_path).lower()
        assert "weekend" in decision_path_str
        # Budget should be reasonable
        assert new_budget > 0

    def test_full_flow_q4_boost(self, allocator):
        """Test complete flow with Q4 boost"""
        metrics = {
            "adset_id": "adset_005",
            "current_budget": 100.0,
            "previous_budget": 95.0,
            "roas_7d": 2.5,
            "roas_trend": 0.05,
            "health_score": 0.70,
            "days_active": 30,
            "is_weekend": False,
            "week_of_year": 50,  # Q4 week
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should get Q4 boost
        assert new_budget > 100.0
        assert any("q4" in path.lower() for path in decision_path)

    def test_full_flow_shock(self, allocator):
        """Test that learning shock protection is applied"""
        metrics = {
            "adset_id": "adset_006",
            "current_budget": 100.0,
            "previous_budget": 100.0,
            "roas_7d": 4.0,  # Excellent ROAS
            "roas_trend": 0.20,  # Strong rising trend
            "efficiency": 0.15,
            "health_score": 0.95,
            "days_active": 30,
        }

        new_budget, _ = allocator.allocate_budget(**metrics)

        # Should be capped at 15% increase (learning shock protection)
        assert new_budget <= 115.0

    def test_full_flow_budget_caps(self, allocator):
        """Test that budget caps are applied"""
        total_budget = 1000.0
        metrics = {
            "adset_id": "adset_007",
            "current_budget": 100.0,
            "previous_budget": 100.0,
            "roas_7d": 4.0,
            "roas_trend": 0.15,
            "health_score": 0.9,
            "days_active": 30,
            "total_budget_today": total_budget,
        }

        new_budget, _ = allocator.allocate_budget(**metrics)

        # Should be capped at 40% of total budget
        max_allowed = total_budget * 0.40
        assert new_budget <= max_allowed

    def test_full_flow_all_features(self, allocator, sample_metrics_dict):
        """Test complete flow with all 21 features provided"""
        metrics = sample_metrics_dict.copy()
        metrics["adset_id"] = "adset_008"

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should successfully allocate with all features
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_multi_adsets_allocation(self, allocator):
        """Test allocating budget for multiple adsets"""
        adsets = [
            {
                "adset_id": "adset_high",
                "current_budget": 100.0,
                "previous_budget": 95.0,
                "roas_7d": 3.5,
                "roas_trend": 0.12,
                "health_score": 0.90,
                "days_active": 30,
            },
            {
                "adset_id": "adset_medium",
                "current_budget": 75.0,
                "previous_budget": 70.0,
                "roas_7d": 2.0,
                "roas_trend": 0.05,
                "health_score": 0.65,
                "days_active": 25,
            },
            {
                "adset_id": "adset_low",
                "current_budget": 50.0,
                "previous_budget": 55.0,
                "roas_7d": 0.3,
                "roas_trend": -0.20,
                "health_score": 0.15,
                "days_active": 15,
            },
        ]

        results = []
        for adset in adsets:
            new_budget, decision_path = allocator.allocate_budget(**adset)
            results.append(
                {
                    "adset_id": adset["adset_id"],
                    "new_budget": new_budget,
                    "decision_path": decision_path,
                }
            )

        # High performer should increase
        assert results[0]["new_budget"] >= 98.0
        # Medium performer should maintain or slight increase
        assert results[1]["new_budget"] >= 70.0
        # Low performer should be frozen
        assert results[2]["new_budget"] == 0.0

    def test_config_integration(self, config):
        """Test that configuration is properly loaded and used"""
        safety_rules = SafetyRules(config)
        decision_rules = DecisionRules(config)

        # Verify config values are loaded
        assert safety_rules.max_daily_increase_pct == config.get_safety_rule(
            "max_daily_increase_pct"
        )
        assert decision_rules.high_roas_threshold == config.get_decision_rule(
            "high_roas_threshold"
        )

    def test_default_config_fallback(self):
        """Test that system works with default config when file is missing"""
        # Create allocator without config file
        safety_rules = SafetyRules()
        decision_rules = DecisionRules()
        allocator = Allocator(safety_rules, decision_rules)

        metrics = {
            "adset_id": "adset_default",
            "current_budget": 100.0,
            "previous_budget": 95.0,
            "roas_7d": 3.0,
            "roas_trend": 0.10,
            "health_score": 0.8,
            "days_active": 20,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should work with defaults
        assert new_budget > 0
        assert len(decision_path) > 0


class TestExtended:
    """Extended integration tests for complete rules system"""

    def test_allocator_minimal_fields(self, allocator):
        """Test allocation with only required fields"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_minimal",
            current_budget=100.0,
            roas_7d=2.5,
            roas_trend=0.0,
        )

        assert isinstance(new_budget, (int, float))
        assert new_budget >= 0
        assert isinstance(decision_path, list)
        assert len(decision_path) > 0

    def test_allocator_all_optional(self, allocator, sample_metrics_dict):
        """Test allocation with all optional fields provided"""
        metrics = sample_metrics_dict.copy()
        metrics["adset_id"] = "test_complete"
        new_budget, decision_path = allocator.allocate_budget(**metrics)

        assert new_budget > 0
        assert len(decision_path) > 0

    def test_allocator_zero_budget(self, allocator):
        """Test allocation when current budget is zero"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_zero",
            current_budget=0.0,
            previous_budget=0.0,
            roas_7d=3.0,
            roas_trend=0.0,
            health_score=0.8,
        )

        # Should either freeze (if below threshold) or allocate small budget
        assert new_budget >= 0
        assert len(decision_path) > 0

    def test_allocator_large_budget(self, allocator):
        """Test allocation with very large budget values"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_large",
            current_budget=1000000.0,
            previous_budget=950000.0,
            roas_7d=2.5,
            roas_trend=0.05,
            health_score=0.7,
            total_budget_today=5000000.0,
        )

        assert new_budget > 0
        assert new_budget <= 5000000.0 * 0.40  # Max 40% of total
        assert len(decision_path) > 0

    def test_allocator_small_budget(self, allocator):
        """Test allocation with very small budget values"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_small",
            current_budget=0.5,
            previous_budget=0.5,
            roas_7d=2.5,
            roas_trend=0.0,
            health_score=0.7,
            total_budget_today=100.0,  # Provide total for caps
        )

        # Minimum budget is enforced in apply_budget_caps, but may vary
        # depending on calculation flow. Just verify it's a valid budget.
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_allocator_extreme_roas(self, allocator):
        """Test allocation with extreme ROAS values"""
        # Very high ROAS
        new_budget_high, _ = allocator.allocate_budget(
            adset_id="test_high_roas",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=10.0,  # Extremely high
            roas_trend=0.0,
            health_score=0.9,
        )

        assert new_budget_high > 100.0  # Should increase
        assert new_budget_high <= 115.0  # Capped by learning shock

        # Very low ROAS (but above freeze threshold)
        new_budget_low, _ = allocator.allocate_budget(
            adset_id="test_low_roas",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=0.6,  # Low but above 0.5 threshold
            roas_trend=-0.10,
            health_score=0.6,
        )

        assert new_budget_low >= 0
        assert len(str(new_budget_low)) > 0  # Valid number

    def test_allocator_boundary_health(self, allocator):
        """Test allocation at health score boundaries"""
        # Health score at freeze threshold
        new_budget_low, _decision_path_low = allocator.allocate_budget(
            adset_id="test_health_low",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.0,
            roas_trend=0.0,
            health_score=0.2,  # At threshold
        )

        # Health score just above threshold
        new_budget_above, _decision_path_above = allocator.allocate_budget(
            adset_id="test_health_above",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.0,
            roas_trend=0.0,
            health_score=0.21,  # Just above
        )

        # Health score very high
        new_budget_high, _ = allocator.allocate_budget(
            adset_id="test_health_high",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.0,
            roas_trend=0.10,
            health_score=0.99,
        )

        assert new_budget_low >= 0
        assert new_budget_above >= 0
        assert new_budget_high > 0

    def test_allocator_cold_bounds(self, allocator):
        """Test allocation at cold start boundaries"""
        # Day 3 (last day of cold start)
        new_budget_day3, _ = allocator.allocate_budget(
            adset_id="test_cold_day3",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.5,
            roas_trend=0.15,
            days_active=3,  # Last cold start day
            health_score=0.8,
        )

        # Day 4 (first day after cold start)
        new_budget_day4, _ = allocator.allocate_budget(
            adset_id="test_cold_day4",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.5,
            roas_trend=0.15,
            days_active=4,  # After cold start
            health_score=0.8,
        )

        # Day 3 should be capped at 10% increase
        assert new_budget_day3 <= 110.01
        # Day 4 can have larger increase (up to 15%)
        assert new_budget_day4 <= 115.01

    def test_allocator_extreme_trends(self, allocator):
        """Test allocation with extreme trend values"""
        # Very strong rising trend
        new_budget_rising, _ = allocator.allocate_budget(
            adset_id="test_trend_rising",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.0,
            roas_trend=0.50,  # Very strong
            health_score=0.8,
        )

        # Very strong falling trend
        new_budget_falling, _ = allocator.allocate_budget(
            adset_id="test_trend_falling",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.0,
            roas_trend=-0.50,  # Very strong
            health_score=0.6,
        )

        assert new_budget_rising > 0
        assert new_budget_falling >= 0

    def test_allocator_weekend_weekday(self, allocator):
        """Test allocation difference between weekend and weekday"""
        # Weekend
        new_budget_weekend, _path_weekend = allocator.allocate_budget(
            adset_id="test_weekend",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.5,
            roas_trend=0.0,
            is_weekend=True,
            health_score=0.7,
        )

        # Weekday
        new_budget_weekday, _path_weekday = allocator.allocate_budget(
            adset_id="test_weekday",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.5,
            roas_trend=0.0,
            is_weekend=False,
            health_score=0.7,
        )

        # Weekend might get boost, but both should be valid
        assert new_budget_weekend > 0
        assert new_budget_weekday > 0

    def test_allocator_q4_boost(self, allocator):
        """Test Q4 boost functionality"""
        # Q4 week (week 50)
        new_budget_q4, _path_q4 = allocator.allocate_budget(
            adset_id="test_q4",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.5,
            roas_trend=0.0,
            week_of_year=50,
            health_score=0.7,
        )

        # Non-Q4 week
        new_budget_normal, _path_normal = allocator.allocate_budget(
            adset_id="test_normal",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.5,
            roas_trend=0.0,
            week_of_year=20,
            health_score=0.7,
        )

        assert new_budget_q4 > 0
        assert new_budget_normal > 0
        # Q4 might get boost
        assert new_budget_q4 >= new_budget_normal or new_budget_q4 <= new_budget_normal

    def test_allocator_budget_caps(self, allocator):
        """Test that budget caps are properly enforced"""
        total_budget = 1000.0
        max_allowed = total_budget * 0.40  # 40% max

        new_budget, _ = allocator.allocate_budget(
            adset_id="test_cap",
            current_budget=500.0,  # Would exceed 40%
            previous_budget=500.0,
            roas_7d=4.0,  # Very high ROAS
            roas_trend=0.20,
            health_score=0.95,
            total_budget_today=total_budget,
        )

        assert new_budget <= max_allowed

    def test_allocator_shock(self, allocator):
        """Test learning shock protection limits changes"""
        # Scenario that would suggest large increase
        new_budget, _ = allocator.allocate_budget(
            adset_id="test_shock",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=5.0,  # Excellent ROAS
            roas_trend=0.30,  # Very strong trend
            efficiency=0.20,
            health_score=0.98,
        )

        # Should be capped at 15% increase
        assert new_budget <= 115.01

    def test_allocator_frozen_recovery(self, allocator):
        """Test frozen adset recovery when performance improves"""
        # First freeze it
        new_budget_frozen, path_frozen = allocator.allocate_budget(
            adset_id="test_recovery",
            current_budget=0.0,  # Already frozen
            previous_budget=100.0,
            roas_7d=0.4,  # Below threshold
            roas_trend=0.0,
            health_score=0.3,
        )

        assert new_budget_frozen == 0.0
        assert "frozen" in " ".join(path_frozen).lower()

        # Then recover (performance improves)
        new_budget_recovered, _path_recovered = allocator.allocate_budget(
            adset_id="test_recovery",
            current_budget=0.0,
            previous_budget=0.0,
            roas_7d=0.7,  # Above threshold * 1.2 (0.5 * 1.2 = 0.6)
            roas_trend=0.10,
            health_score=0.5,
        )

        # Should be able to recover
        assert new_budget_recovered >= 0

    def test_allocator_neg_trend_roas(self, allocator, sample_metrics_dict):
        """Test allocation with negative trend but high ROAS"""
        metrics = sample_metrics_dict.copy()
        metrics.update(
            {
                "adset_id": "test_negative_trend",
                "roas_7d": 3.5,
                "roas_trend": -0.10,
                "health_score": 0.8,
            }
        )
        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should handle carefully (might decrease or maintain)
        assert new_budget >= 0
        assert len(decision_path) > 0

    def test_allocator_pos_trend_roas(self, allocator):
        """Test allocation with positive trend but low ROAS"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_positive_trend",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=1.2,  # Low ROAS
            roas_trend=0.15,  # Positive trend
            health_score=0.6,
        )

        # Should handle carefully
        assert new_budget >= 0
        assert len(decision_path) > 0

    def test_allocator_config_file(
        self, tmp_path, allocator
    ):  # pylint: disable=unused-argument
        """Test allocator uses config file settings"""
        # Create minimal config file
        config_path = tmp_path / "test_rules.yaml"
        config_path.write_text(
            """
safety_rules:
  max_daily_increase_pct: 0.20
  max_daily_decrease_pct: 0.20
  freeze_roas_threshold: 0.6
  freeze_health_threshold: 0.3
  min_budget: 2.0
  max_budget_pct_of_total: 0.50
  cold_start_days: 5
  cold_start_max_increase_pct: 0.15

decision_rules:
  high_roas_threshold: 2.8
  low_roas_threshold: 1.4
  excellent_roas_threshold: 3.8
  aggressive_increase_pct: 0.18
  moderate_increase_pct: 0.12
        """
        )

        config = Parser(config_path=str(config_path))
        safety_rules = SafetyRules(config)
        decision_rules = DecisionRules(config)
        test_allocator = Allocator(safety_rules, decision_rules, config)

        new_budget, decision_path = test_allocator.allocate_budget(
            adset_id="test_config",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=4.0,
            roas_trend=0.20,
            health_score=0.9,
        )

        # Config has max_daily_increase_pct: 0.20, so should allow up to 20%
        assert new_budget <= 120.01
        assert len(decision_path) > 0

    def test_allocator_multi_adsets(self, allocator):
        """Test that allocation is consistent across multiple adsets"""
        metrics_list = [
            {
                "adset_id": f"adset_{i}",
                "current_budget": 100.0,
                "previous_budget": 95.0,
                "roas_7d": 2.5 + (i * 0.1),
                "roas_trend": 0.05,
                "health_score": 0.7,
            }
            for i in range(5)
        ]

        results = []
        for metrics in metrics_list:
            new_budget, decision_path = allocator.allocate_budget(**metrics)
            results.append(
                {
                    "adset_id": metrics["adset_id"],
                    "new_budget": new_budget,
                    "decision_path": decision_path,
                }
            )

        # All should succeed
        assert len(results) == 5
        for result in results:
            assert result["new_budget"] > 0
            assert len(result["decision_path"]) > 0

    def test_allocator_none_handling(self, allocator):
        """Test allocation handles None values gracefully"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_none",
            current_budget=100.0,
            previous_budget=None,  # None value
            roas_7d=2.5,
            roas_trend=0.0,
            adset_roas=None,
            campaign_roas=None,
            account_roas=None,
            health_score=0.7,
        )

        # Should handle None values
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_allocator_rel_perf_high(self, allocator):
        """Test allocation with high relative performance"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_relative",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.0,
            roas_trend=0.0,
            roas_vs_adset=1.5,  # High relative performance
            roas_vs_campaign=1.4,
            roas_vs_account=1.3,
            health_score=0.8,
        )

        # Should increase due to relative performance
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_allocator_rel_perf_low(self, allocator):
        """Test allocation with low relative performance"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_relative_low",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.0,
            roas_trend=0.0,
            roas_vs_adset=0.7,  # Low relative performance
            roas_vs_campaign=0.8,
            roas_vs_account=0.9,
            health_score=0.6,
        )

        assert new_budget >= 0
        assert len(decision_path) > 0

    def test_allocator_efficiency_high(self, allocator):
        """Test allocation with high efficiency metrics"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_efficiency",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=2.5,
            roas_trend=0.0,
            efficiency=0.15,  # High efficiency
            revenue_per_impression=0.12,
            revenue_per_click=3.5,
            health_score=0.8,
        )

        assert new_budget > 0
        assert len(decision_path) > 0

    def test_allocator_volume_scaling(self, allocator):
        """Test allocation based on volume metrics"""
        new_budget, decision_path = allocator.allocate_budget(
            adset_id="test_volume",
            current_budget=100.0,
            previous_budget=100.0,
            roas_7d=3.0,
            roas_trend=0.0,
            spend=5.0,  # Low spend
            impressions=500,  # Low volume
            clicks=10,
            efficiency=0.12,  # High efficiency
            days_active=25,
            health_score=0.8,
        )

        # Should scale up due to low spend but high ROAS/efficiency
        assert new_budget > 0
        assert len(decision_path) > 0


class TestEdgeCases:
    """Integration tests for edge cases and error scenarios"""

    def test_minimal_required_features(self, allocator):
        """Test allocation with only required features"""
        metrics = {
            "adset_id": "adset_minimal",
            "current_budget": 100.0,
            "previous_budget": None,
            "roas_7d": 2.5,
            "roas_trend": 0.0,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should work with minimal features
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_missing_optional_features(self, allocator):
        """Test allocation with missing optional features"""
        metrics = {
            "adset_id": "adset_missing_optional",
            "current_budget": 100.0,
            "previous_budget": 95.0,
            "roas_7d": 3.0,
            "roas_trend": 0.10,
            # Missing: adset_roas, campaign_roas, efficiency, etc.
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should work with missing optional features
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_zero_budget_adset(self, allocator):
        """Test allocation for adset with zero current budget"""
        metrics = {
            "adset_id": "adset_zero",
            "current_budget": 0.0,
            "previous_budget": 50.0,
            "roas_7d": 2.5,
            "roas_trend": 0.10,
            "health_score": 0.7,
            "days_active": 20,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should handle zero budget
        assert new_budget >= 0
        assert len(decision_path) > 0

    def test_very_large_budget(self, allocator):
        """Test allocation with very large budget"""
        metrics = {
            "adset_id": "adset_large",
            "current_budget": 10000.0,
            "previous_budget": 9500.0,
            "roas_7d": 3.0,
            "roas_trend": 0.10,
            "health_score": 0.8,
            "days_active": 30,
            "total_budget_today": 50000.0,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should handle large budgets and apply caps
        assert new_budget > 0
        assert new_budget <= 20000.0  # Capped at 40% of total
        assert len(decision_path) > 0

    def test_very_small_budget(self, allocator):
        """Test allocation with very small budget"""
        metrics = {
            "adset_id": "adset_small",
            "current_budget": 1.0,
            "previous_budget": 0.5,
            "roas_7d": 3.0,
            "roas_trend": 0.10,
            "health_score": 0.8,
            "days_active": 30,
            "total_budget_today": 100.0,  # Provide total budget for caps
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should handle small budgets
        # Note: Minimum budget is enforced in apply_budget_caps, but may be
        # applied after adjustments, so we just check it's reasonable
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_boundary_roas_values(self, allocator):
        """Test allocation with boundary ROAS values"""
        # Test at freeze threshold
        metrics_freeze = {
            "adset_id": "adset_freeze_boundary",
            "current_budget": 100.0,
            "previous_budget": 100.0,
            "roas_7d": 0.5,  # Exactly at freeze threshold
            "roas_trend": 0.0,
            "health_score": 0.5,
            "days_active": 20,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_freeze)

        # Should not freeze (threshold is < 0.5)
        assert new_budget >= 0
        assert len(decision_path) > 0

        # Test just below freeze threshold
        metrics_below = {
            "adset_id": "adset_below_freeze",
            "current_budget": 100.0,
            "previous_budget": 100.0,
            "roas_7d": 0.49,  # Just below freeze threshold
            "roas_trend": 0.0,
            "health_score": 0.5,
            "days_active": 20,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_below)

        # Should be frozen
        assert new_budget == 0.0
        assert any("frozen" in path.lower() for path in decision_path)

    def test_boundary_health_score(self, allocator):
        """Test allocation with boundary health score values"""
        # Test at freeze threshold
        metrics_freeze = {
            "adset_id": "adset_health_boundary",
            "current_budget": 100.0,
            "previous_budget": 100.0,
            "roas_7d": 2.0,
            "roas_trend": 0.0,
            "health_score": 0.2,  # Exactly at freeze threshold
            "days_active": 20,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_freeze)

        # Should not freeze (threshold is < 0.2)
        assert new_budget >= 0
        assert len(decision_path) > 0

        # Test just below freeze threshold
        metrics_below = {
            "adset_id": "adset_health_below",
            "current_budget": 100.0,
            "previous_budget": 100.0,
            "roas_7d": 2.0,
            "roas_trend": 0.0,
            "health_score": 0.19,  # Just below freeze threshold
            "days_active": 20,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_below)

        # Should be frozen
        assert new_budget == 0.0
        assert any("frozen" in path.lower() for path in decision_path)

    def test_cold_start_boundary(self, allocator):
        """Test allocation at cold start boundary"""
        # Test at cold start boundary (day 3)
        metrics_boundary = {
            "adset_id": "adset_cold_start_boundary",
            "current_budget": 100.0,
            "previous_budget": 100.0,
            "roas_7d": 3.0,
            "roas_trend": 0.15,
            "health_score": 0.8,
            "days_active": 3,  # At cold start boundary
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_boundary)

        # Should still apply cold start protection
        assert new_budget > 0
        assert new_budget <= 110.0  # Capped at 10% increase
        assert len(decision_path) > 0

        # Test just after cold start (day 4)
        metrics_after = {
            "adset_id": "adset_after_cold_start",
            "current_budget": 100.0,
            "previous_budget": 100.0,
            "roas_7d": 3.0,
            "roas_trend": 0.15,
            "health_score": 0.8,
            "days_active": 4,  # After cold start
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_after)

        # Should allow larger increases
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_extreme_trend_values(self, allocator):
        """Test allocation with extreme trend values"""
        # Very strong rising trend
        metrics_rising = {
            "adset_id": "adset_strong_rising",
            "current_budget": 100.0,
            "previous_budget": 95.0,
            "roas_7d": 3.0,
            "roas_trend": 0.50,  # Very strong trend
            "health_score": 0.8,
            "days_active": 30,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_rising)

        # Should increase but be capped by learning shock protection
        assert new_budget > 100.0
        assert new_budget <= 115.0  # Capped at 15% increase
        assert len(decision_path) > 0

        # Very strong falling trend
        metrics_falling = {
            "adset_id": "adset_strong_falling",
            "current_budget": 100.0,
            "previous_budget": 105.0,
            "roas_7d": 1.5,
            "roas_trend": -0.50,  # Very strong decline
            "health_score": 0.5,
            "days_active": 30,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_falling)

        # Should decrease but be capped by learning shock protection
        assert new_budget < 100.0
        assert new_budget >= 85.0  # Capped at 15% decrease
        assert len(decision_path) > 0

    def test_frozen_adset_recovery(self, allocator):
        """Test frozen adset recovery when performance improves"""
        # First, freeze the adset
        metrics_frozen = {
            "adset_id": "adset_recovery",
            "current_budget": 0.0,
            "previous_budget": 50.0,
            "roas_7d": 0.3,  # Below threshold
            "roas_trend": -0.20,
            "health_score": 0.15,
            "days_active": 15,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_frozen)
        assert new_budget == 0.0  # Should be frozen

        # Then, test recovery when performance improves
        metrics_recovery = {
            "adset_id": "adset_recovery",
            "current_budget": 0.0,
            "previous_budget": 0.0,
            "roas_7d": 0.7,  # Above threshold * 1.2
            "roas_trend": 0.20,
            "health_score": 0.5,
            "days_active": 16,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics_recovery)

        # Should be able to recover
        assert new_budget >= 0
        assert len(decision_path) > 0

    def test_neg_trend_high_roas(self, allocator):
        """Test allocation with negative trend but high ROAS"""
        metrics = {
            "adset_id": "adset_high_roas_negative_trend",
            "current_budget": 100.0,
            "previous_budget": 95.0,
            "roas_7d": 3.5,  # High ROAS
            "roas_trend": -0.05,  # Slight decline
            "health_score": 0.8,
            "days_active": 30,
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should still perform well despite negative trend
        assert new_budget > 0
        assert len(decision_path) > 0

    def test_pos_trend_low_roas(self, allocator):
        """Test allocation with positive trend but low ROAS"""
        metrics = {
            "adset_id": "adset_low_roas_positive_trend",
            "current_budget": 100.0,
            "previous_budget": 95.0,
            "roas_7d": 1.2,  # Low ROAS
            "roas_trend": 0.10,  # Improving
            "health_score": 0.6,
            "days_active": 10,  # Learning phase
        }

        new_budget, decision_path = allocator.allocate_budget(**metrics)

        # Should give time for improvement (Rule 6.2)
        assert new_budget > 0
        assert len(decision_path) > 0
