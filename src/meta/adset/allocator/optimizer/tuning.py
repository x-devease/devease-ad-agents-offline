"""
Rule parameter tuning utilities for budget allocation.

This module provides core evaluation utilities for the automated Bayesian tuner.
It handles simulation of budget allocation and evaluation of results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.meta.adset import Allocator, DecisionRules, SafetyRules
from src.meta.adset.allocator.utils.parser import Parser
from src.meta.adset.allocator.lib.models import BudgetAllocationMetrics


@dataclass(frozen=True)
class TuningConstraints:
    """Constraints for budget allocation tuning.

    These constraints ensure that tuned parameters produce reasonable
    budget allocation behavior.
    """

    # Budget adjustment constraints
    min_budget_change_rate: float = (
        0.05  # At least 5% of adsets should have budget changes
    )
    max_budget_change_rate: float = (
        0.50  # At most 50% of adsets should have budget changes
    )
    min_total_budget_utilization: float = (
        0.80  # At least 80% of total budget should be allocated
    )
    max_total_budget_utilization: float = 1.05  # At most 105% of total budget allocated

    # Performance constraints
    min_avg_roas: float = 1.5  # Minimum average ROAS for allocated budgets
    min_revenue_efficiency: float = 0.8  # Revenue efficiency (revenue / total_budget)

    # Budget distribution constraints
    min_adsets_with_budget: int = 1  # Minimum number of adsets that receive budget
    max_single_adset_budget_pct: float = 0.50  # Max 50% of budget to single adset


@dataclass(frozen=True)
class TuningResult:
    """Result of a single tuning iteration."""

    param_config: Dict[str, Any]  # Parameter configuration tested
    total_adsets: int  # Total number of adsets processed
    adsets_with_changes: int  # Number of adsets with budget changes
    change_rate: float  # Percentage of adsets with changes
    total_budget_allocated: float  # Total budget allocated
    budget_utilization: float  # Budget utilization (allocated / available)
    avg_roas: float  # Average ROAS of adsets with allocated budget
    weighted_avg_roas: float  # Budget-weighted average ROAS
    total_revenue: float  # Estimated total revenue (budget * ROAS)
    revenue_efficiency: float  # Revenue efficiency (revenue / total_budget)
    max_single_adset_pct: float  # Maximum percentage of budget to single adset

    # Additional metrics for improved evaluation
    roas_std: float = 0.0  # Standard deviation of ROAS (stability metric)
    budget_gini: float = (
        0.0  # Gini coefficient of budget distribution (0=equal, 1=unequal)
    )
    budget_entropy: float = (
        0.0  # Entropy of budget distribution (higher = more diverse)
    )
    avg_budget_change_pct: float = 0.0  # Average percentage change in budgets
    budget_change_volatility: float = 0.0  # Std dev of budget changes (stability)
    high_roas_budget_pct: float = (
        0.0  # Percentage of budget allocated to high ROAS adsets
    )
    low_roas_budget_pct: float = (
        0.0  # Percentage of budget allocated to low ROAS adsets
    )
    baseline_comparison: Optional[float] = None  # Comparison vs baseline (if provided)

    # CTR metrics (for engagement optimization)
    avg_ctr: float = 0.0  # Average CTR of adsets with allocated budget
    weighted_avg_ctr: float = 0.0  # Budget-weighted average CTR
    ctr_std: float = 0.0  # Standard deviation of CTR
    high_ctr_budget_pct: float = 0.0  # Percentage of budget to high CTR adsets


@dataclass(frozen=True)
class OptimizationObjectives:
    """Customer-specific optimization objectives.

    Defines how to balance multiple competing objectives (ROAS, CTR, stability)
    and what constraints to apply during optimization.
    """

    # Objective weights (must sum to 1.0)
    roas_weight: float = 0.60
    ctr_weight: float = 0.20
    stability_weight: float = 0.15
    budget_utilization_weight: float = 0.05

    # Target values for optimization
    target_roas: Optional[float] = None
    target_ctr: Optional[float] = None
    min_acceptable_roas: Optional[float] = None
    min_acceptable_ctr: Optional[float] = None

    # Trade-off preferences
    accept_lower_roas_for_higher_ctr: bool = False
    accept_higher_volatility_for_higher_roas: bool = True
    prefer_stable_over_optimal: bool = False

    # Constraints (override TuningConstraints if provided)
    max_daily_change_pct: Optional[float] = None
    min_budget_utilization: Optional[float] = None
    max_budget_utilization: Optional[float] = None
    max_single_adset_pct: Optional[float] = None


def load_objectives(
    config_path: str,
    customer_name: str = "moprobo",
    campaign_name: Optional[str] = None,
) -> OptimizationObjectives:
    """Load optimization objectives from config file.

    Args:
        config_path: Path to objectives.yaml config file.
        customer_name: Customer name (e.g., "moprobo").
        campaign_name: Optional campaign name for campaign-specific overrides.

    Returns:
        OptimizationObjectives with customer/campaign-specific configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If customer not found in config.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        # Return default objectives if config doesn't exist
        return OptimizationObjectives()

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if customer_name not in config:
        raise ValueError(f"Customer '{customer_name}' not found in {config_path}")

    customer_config = config[customer_name]

    # Start with default objectives
    objectives_dict = {
        "roas_weight": 0.60,
        "ctr_weight": 0.20,
        "stability_weight": 0.15,
        "budget_utilization_weight": 0.05,
    }

    # Load customer-level objectives
    if "objectives" in customer_config:
        obj = customer_config["objectives"]

        # Load weights
        if "weights" in obj:
            weights = obj["weights"]
            objectives_dict["roas_weight"] = weights.get("roas", 0.60)
            objectives_dict["ctr_weight"] = weights.get("ctr", 0.20)
            objectives_dict["stability_weight"] = weights.get("stability", 0.15)
            objectives_dict["budget_utilization_weight"] = weights.get(
                "budget_utilization", 0.05
            )

        # Load targets
        objectives_dict["target_roas"] = obj.get("target_roas")
        objectives_dict["target_ctr"] = obj.get("target_ctr")
        objectives_dict["min_acceptable_roas"] = obj.get("trade_offs", {}).get(
            "min_acceptable_roas"
        )
        objectives_dict["min_acceptable_ctr"] = obj.get("trade_offs", {}).get(
            "min_acceptable_ctr"
        )

        # Load trade-offs
        if "trade_offs" in obj:
            to = obj["trade_offs"]
            # Note: can't use bool() as field names, so we load them differently
            # These will be set as attributes after initialization

        # Load constraints
        if "constraints" in obj:
            constraints = obj["constraints"]
            objectives_dict["max_daily_change_pct"] = constraints.get(
                "max_daily_change_pct"
            )
            objectives_dict["min_budget_utilization"] = constraints.get(
                "min_budget_utilization"
            )
            objectives_dict["max_budget_utilization"] = constraints.get(
                "max_budget_utilization"
            )
            objectives_dict["max_single_adset_pct"] = constraints.get(
                "max_single_adset_pct"
            )

    # Helper function to find campaign config in year/month structure
    def _find_campaign_config(
        campaign_objectives: Dict[str, Any], campaign_name: str
    ) -> Optional[Dict[str, Any]]:
        """Find campaign config in year/month grouped structure.

        Args:
            campaign_objectives: Campaign objectives section from config.
            campaign_name: Name of campaign to find.

        Returns:
            Campaign config dict if found, None otherwise.
        """
        # First, try direct lookup (old flat structure for backwards compatibility)
        if campaign_name in campaign_objectives:
            return campaign_objectives[campaign_name]

        # Then, search in year/month groups
        for year_key, year_data in campaign_objectives.items():
            if year_key == "evergreen":
                # Special section for always-running campaigns
                if campaign_name in year_data:
                    return year_data[campaign_name]
                continue

            if not isinstance(year_data, dict):
                continue

            for month_key, month_data in year_data.items():
                if not isinstance(month_data, dict):
                    continue

                if campaign_name in month_data:
                    return month_data[campaign_name]

        return None

    # Apply campaign-specific overrides if specified
    if campaign_name and "campaign_objectives" in customer_config:
        campaign_config = _find_campaign_config(
            customer_config["campaign_objectives"], campaign_name
        )
        if campaign_config:
            # Update weights
            if "weights" in campaign_config:
                weights = campaign_config["weights"]
                objectives_dict["roas_weight"] = weights.get("roas", 0.60)
                objectives_dict["ctr_weight"] = weights.get("ctr", 0.20)
                objectives_dict["stability_weight"] = weights.get("stability", 0.15)
                objectives_dict["budget_utilization_weight"] = weights.get(
                    "budget_utilization", 0.05
                )

            # Update targets
            if "target_roas" in campaign_config:
                objectives_dict["target_roas"] = campaign_config["target_roas"]
            if "target_ctr" in campaign_config:
                objectives_dict["target_ctr"] = campaign_config["target_ctr"]

            # Update trade-offs
            if "trade_offs" in campaign_config:
                to = campaign_config["trade_offs"]
                if "accept_lower_roas_for_higher_ctr" in to:
                    objectives_dict["accept_lower_roas_for_higher_ctr"] = to[
                        "accept_lower_roas_for_higher_ctr"
                    ]
                if "accept_higher_volatility_for_higher_roas" in to:
                    objectives_dict["accept_higher_volatility_for_higher_roas"] = to[
                        "accept_higher_volatility_for_higher_roas"
                    ]
                if "prefer_stable_over_optimal" in to:
                    objectives_dict["prefer_stable_over_optimal"] = to[
                        "prefer_stable_over_optimal"
                    ]
                if "min_acceptable_roas" in to:
                    objectives_dict["min_acceptable_roas"] = to["min_acceptable_roas"]
                if "min_acceptable_ctr" in to:
                    objectives_dict["min_acceptable_ctr"] = to["min_acceptable_ctr"]

            # Update constraints
            if "max_daily_change_pct" in campaign_config:
                objectives_dict["max_daily_change_pct"] = campaign_config[
                    "max_daily_change_pct"
                ]

    # Extract trade-offs from customer config for final values
    trade_offs = customer_config.get("objectives", {}).get("trade_offs", {})
    accept_lower_roas = objectives_dict.get(
        "accept_lower_roas_for_higher_ctr",
        trade_offs.get("accept_lower_roas_for_higher_ctr", False),
    )
    accept_higher_volatility = objectives_dict.get(
        "accept_higher_volatility_for_higher_roas",
        trade_offs.get("accept_higher_volatility_for_higher_roas", True),
    )
    prefer_stable = objectives_dict.get(
        "prefer_stable_over_optimal",
        trade_offs.get("prefer_stable_over_optimal", False),
    )

    return OptimizationObjectives(
        roas_weight=objectives_dict["roas_weight"],
        ctr_weight=objectives_dict["ctr_weight"],
        stability_weight=objectives_dict["stability_weight"],
        budget_utilization_weight=objectives_dict["budget_utilization_weight"],
        target_roas=objectives_dict.get("target_roas"),
        target_ctr=objectives_dict.get("target_ctr"),
        min_acceptable_roas=objectives_dict.get("min_acceptable_roas"),
        min_acceptable_ctr=objectives_dict.get("min_acceptable_ctr"),
        accept_lower_roas_for_higher_ctr=accept_lower_roas,
        accept_higher_volatility_for_higher_roas=accept_higher_volatility,
        prefer_stable_over_optimal=prefer_stable,
        max_daily_change_pct=objectives_dict.get("max_daily_change_pct"),
        min_budget_utilization=objectives_dict.get("min_budget_utilization"),
        max_budget_utilization=objectives_dict.get("max_budget_utilization"),
        max_single_adset_pct=objectives_dict.get("max_single_adset_pct"),
    )


def _create_allocator_with_config(
    base_config_path: str,
    param_overrides: Dict[str, Any],
    customer_name: str = "moprobo",
) -> Allocator:
    """Create an Allocator instance with parameter overrides.

    Args:
        base_config_path: Path to base rules.yaml config file.
        param_overrides: Dictionary of parameter overrides to apply.
            Keys should be dot-separated paths (e.g., "decision_rules.low_roas_threshold").
        customer_name: Customer name for config section. Defaults to "moprobo".

    Returns:
        Configured Allocator instance.
    """
    import tempfile
    import yaml
    from pathlib import Path

    # Load base config
    base_parser = Parser(base_config_path, customer_name=customer_name, platform="meta")
    config = base_parser.config.copy()

    # Get customer config section
    if customer_name in config:
        customer_config = config[customer_name].copy()
    elif "safety_rules" in config or "decision_rules" in config:
        customer_config = config.copy()
    else:
        customer_config = {}

    # Apply parameter overrides to customer config
    # Navigate through nested structure (e.g., decision_rules.low_roas_threshold)
    for key_path, value in param_overrides.items():
        keys = key_path.split(".")
        current = customer_config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    # Update config with modified customer config
    if customer_name in config:
        config[customer_name] = customer_config
    else:
        config.update(customer_config)

    # Write modified config to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp_file:
        yaml.dump(config, tmp_file, default_flow_style=False)
        tmp_config_path = tmp_file.name

    try:
        # Create parser with modified config
        parser = Parser(tmp_config_path, customer_name=customer_name, platform="meta")

        # Create rules with updated config
        safety_rules = SafetyRules(parser)
        decision_rules = DecisionRules(parser)
        allocator = Allocator(safety_rules, decision_rules, parser)

        return allocator
    finally:
        # Clean up temp file
        Path(tmp_config_path).unlink(missing_ok=True)


def simulate_allocation(
    df_features: pd.DataFrame,
    allocator: Allocator,
    total_budget: float,
) -> pd.DataFrame:
    """Simulate budget allocation on feature data.

    Args:
        df_features: DataFrame with adset features (must include required columns).
        allocator: Allocator instance to use for allocation.
        total_budget: Total available budget to allocate.

    Returns:
        DataFrame with allocation results including new_budget column.
    """
    results = []

    for _, row in df_features.iterrows():
        # Convert row to metrics dict
        metrics_dict = row.to_dict()

        # Ensure required fields exist
        if "adset_id" not in metrics_dict:
            metrics_dict["adset_id"] = (
                str(row.name) if hasattr(row, "name") else "unknown"
            )
        if "current_budget" not in metrics_dict:
            metrics_dict["current_budget"] = metrics_dict.get("spend", 0.0)
        if "roas_7d" not in metrics_dict:
            # Try alternative column names
            metrics_dict["roas_7d"] = metrics_dict.get(
                "purchase_roas_rolling_7d",
                metrics_dict.get("purchase_roas", 0.0),
            )
        if "roas_trend" not in metrics_dict:
            metrics_dict["roas_trend"] = 0.0

        # Map common column name variations
        column_mapping = {
            "days_since_start": "days_active",
            "purchase_roas_rolling_7d": "roas_7d",
            "purchase_roas": "roas_7d",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in metrics_dict and new_name not in metrics_dict:
                metrics_dict[new_name] = metrics_dict[old_name]

        try:
            new_budget, decision_path = allocator.allocate_budget(**metrics_dict)
            results.append(
                {
                    "adset_id": metrics_dict["adset_id"],
                    "current_budget": metrics_dict["current_budget"],
                    "new_budget": new_budget,
                    "roas_7d": metrics_dict["roas_7d"],
                    "decision_path": (
                        " -> ".join(decision_path) if decision_path else "none"
                    ),
                }
            )
        except Exception as e:
            # Skip rows that cause errors
            continue

    return pd.DataFrame(results)


def _calculate_gini_coefficient(values: pd.Series) -> float:
    """Calculate Gini coefficient for budget distribution.

    Gini coefficient measures inequality:
    - 0 = perfect equality (all adsets get same budget)
    - 1 = perfect inequality (one adset gets all budget)

    Args:
        values: Series of budget values.

    Returns:
        Gini coefficient between 0 and 1.
    """
    if len(values) == 0 or values.sum() == 0:
        return 0.0

    values = values[values > 0].sort_values()
    n = len(values)
    cumsum = values.cumsum()

    # Gini formula: 1 - (2 * sum of cumulative proportions) / n
    return float(1 - (2 * (cumsum / cumsum.iloc[-1]).sum() / (n + 1)))


def _calculate_entropy(values: pd.Series) -> float:
    """Calculate entropy of budget distribution.

    Higher entropy = more diverse/even distribution.
    Lower entropy = more concentrated distribution.

    Args:
        values: Series of budget values.

    Returns:
        Entropy value (bits).
    """
    if len(values) == 0 or values.sum() == 0:
        return 0.0

    # Normalize to probabilities
    probs = values[values > 0] / values.sum()
    # Calculate entropy: -sum(p * log2(p))
    entropy = -((probs * np.log2(probs)).sum())
    return float(entropy)


def evaluate_allocation(
    allocation_results: pd.DataFrame,
    df_features: pd.DataFrame,
    total_budget: float,
    baseline_allocation: Optional[pd.DataFrame] = None,
    target_roas: Optional[float] = None,
    target_ctr: Optional[float] = None,
) -> TuningResult:
    """Evaluate budget allocation results.

    Args:
        allocation_results: DataFrame with allocation results from simulate_allocation.
        df_features: Original features DataFrame (for ROAS and CTR lookup).
        total_budget: Total available budget.
        baseline_allocation: Optional baseline allocation for comparison.
        target_roas: Optional target ROAS threshold.
        target_ctr: Optional target CTR threshold.

    Returns:
        TuningResult with evaluation metrics including CTR.
    """

    # Merge with original features to get ROAS
    # Handle various column name variations for ROAS
    roas_col = None
    for col in ["roas_7d", "purchase_roas_rolling_7d", "purchase_roas", "adset_roas"]:
        if col in df_features.columns:
            roas_col = col
            break

    if roas_col is None:
        # No ROAS column found, use default
        allocation_results_copy = allocation_results.copy()
        allocation_results_copy["roas_7d"] = 0.0
        merged = allocation_results_copy
    else:
        merged = allocation_results.merge(
            df_features[["adset_id", roas_col]].rename(
                columns={roas_col: "feature_roas"}
            ),
            on="adset_id",
            how="left",
        )
        merged["roas_7d"] = merged["feature_roas"].fillna(0.0)

    # Calculate metrics
    total_adsets = len(allocation_results)
    adsets_with_changes = len(
        allocation_results[
            abs(allocation_results["new_budget"] - allocation_results["current_budget"])
            > 0.01
        ]
    )
    change_rate = adsets_with_changes / total_adsets if total_adsets > 0 else 0.0

    total_budget_allocated = allocation_results["new_budget"].sum()
    budget_utilization = (
        total_budget_allocated / total_budget if total_budget > 0 else 0.0
    )

    # ROAS metrics (only for adsets with budget > 0)
    adsets_with_budget = merged[merged["new_budget"] > 0]
    if len(adsets_with_budget) > 0:
        avg_roas = adsets_with_budget["roas_7d"].mean()
        weighted_avg_roas = (
            (adsets_with_budget["new_budget"] * adsets_with_budget["roas_7d"]).sum()
            / adsets_with_budget["new_budget"].sum()
            if adsets_with_budget["new_budget"].sum() > 0
            else 0.0
        )
        total_revenue = (
            adsets_with_budget["new_budget"] * adsets_with_budget["roas_7d"]
        ).sum()
    else:
        avg_roas = 0.0
        weighted_avg_roas = 0.0
        total_revenue = 0.0

    revenue_efficiency = total_revenue / total_budget if total_budget > 0 else 0.0

    max_single_adset_pct = (
        allocation_results["new_budget"].max() / total_budget
        if total_budget > 0
        else 0.0
    )

    # Additional metrics for improved evaluation
    # ROAS stability
    roas_std = (
        float(adsets_with_budget["roas_7d"].std())
        if len(adsets_with_budget) > 1
        else 0.0
    )

    # Budget distribution quality
    budget_gini = _calculate_gini_coefficient(allocation_results["new_budget"])
    budget_entropy = _calculate_entropy(allocation_results["new_budget"])

    # Budget change analysis
    allocation_results["budget_change_pct"] = (
        (allocation_results["new_budget"] - allocation_results["current_budget"])
        / allocation_results["current_budget"].replace(0, np.nan)
        * 100
    )
    avg_budget_change_pct = (
        float(allocation_results["budget_change_pct"].abs().mean())
        if len(allocation_results) > 0
        else 0.0
    )
    budget_change_volatility = (
        float(allocation_results["budget_change_pct"].std())
        if len(allocation_results) > 1
        else 0.0
    )

    # ROAS-based budget allocation analysis
    if target_roas is not None and len(adsets_with_budget) > 0:
        high_roas_mask = adsets_with_budget["roas_7d"] >= target_roas * 1.2
        low_roas_mask = adsets_with_budget["roas_7d"] < target_roas * 0.8
        high_roas_budget_pct = (
            adsets_with_budget.loc[high_roas_mask, "new_budget"].sum() / total_budget
            if total_budget > 0
            else 0.0
        )
        low_roas_budget_pct = (
            adsets_with_budget.loc[low_roas_mask, "new_budget"].sum() / total_budget
            if total_budget > 0
            else 0.0
        )
    else:
        high_roas_budget_pct = 0.0
        low_roas_budget_pct = 0.0

    # CTR metrics (for engagement optimization)
    ctr_col = None
    for col in ["ctr", "ctr_rolling_7d", "adset_ctr"]:
        if col in df_features.columns:
            ctr_col = col
            break

    if ctr_col is not None:
        # Merge CTR data
        merged = merged.merge(
            df_features[["adset_id", ctr_col]].rename(columns={ctr_col: "feature_ctr"}),
            on="adset_id",
            how="left",
        )
        merged["ctr"] = merged.get("feature_ctr", 0.0).fillna(0.0)

        # Calculate CTR metrics for adsets with budget
        if len(adsets_with_budget) > 0:
            adsets_with_budget = merged[merged["new_budget"] > 0]
            avg_ctr = adsets_with_budget["ctr"].mean()
            weighted_avg_ctr = (
                (adsets_with_budget["new_budget"] * adsets_with_budget["ctr"]).sum()
                / adsets_with_budget["new_budget"].sum()
                if adsets_with_budget["new_budget"].sum() > 0
                else 0.0
            )
            ctr_std = (
                float(adsets_with_budget["ctr"].std())
                if len(adsets_with_budget) > 1
                else 0.0
            )

            # CTR-based budget allocation analysis
            if target_ctr is not None:
                high_ctr_mask = adsets_with_budget["ctr"] >= target_ctr * 1.2
                high_ctr_budget_pct = (
                    adsets_with_budget.loc[high_ctr_mask, "new_budget"].sum()
                    / total_budget
                    if total_budget > 0
                    else 0.0
                )
            else:
                high_ctr_budget_pct = 0.0
        else:
            avg_ctr = 0.0
            weighted_avg_ctr = 0.0
            ctr_std = 0.0
            high_ctr_budget_pct = 0.0
    else:
        avg_ctr = 0.0
        weighted_avg_ctr = 0.0
        ctr_std = 0.0
        high_ctr_budget_pct = 0.0

    # Baseline comparison (if provided)
    baseline_comparison = None
    if baseline_allocation is not None and len(baseline_allocation) > 0:
        # Use same ROAS column detection for baseline
        if roas_col is not None:
            baseline_merged = baseline_allocation.merge(
                df_features[["adset_id", roas_col]].rename(
                    columns={roas_col: "feature_roas"}
                ),
                on="adset_id",
                how="left",
            )
            baseline_merged["roas_7d"] = baseline_merged["feature_roas"].fillna(0.0)
        else:
            baseline_merged = baseline_allocation.copy()
            baseline_merged["roas_7d"] = 0.0
        baseline_with_budget = baseline_merged[baseline_merged["new_budget"] > 0]
        if len(baseline_with_budget) > 0:
            baseline_weighted_roas = (
                (
                    baseline_with_budget["new_budget"] * baseline_with_budget["roas_7d"]
                ).sum()
                / baseline_with_budget["new_budget"].sum()
                if baseline_with_budget["new_budget"].sum() > 0
                else 0.0
            )
            # Improvement percentage
            baseline_comparison = (
                (weighted_avg_roas - baseline_weighted_roas)
                / baseline_weighted_roas
                * 100
                if baseline_weighted_roas > 0
                else 0.0
            )

    return TuningResult(
        param_config={},  # Will be filled by caller
        total_adsets=total_adsets,
        adsets_with_changes=adsets_with_changes,
        change_rate=change_rate,
        total_budget_allocated=total_budget_allocated,
        budget_utilization=budget_utilization,
        avg_roas=avg_roas,
        weighted_avg_roas=weighted_avg_roas,
        total_revenue=total_revenue,
        revenue_efficiency=revenue_efficiency,
        max_single_adset_pct=max_single_adset_pct,
        roas_std=roas_std,
        budget_gini=budget_gini,
        budget_entropy=budget_entropy,
        avg_budget_change_pct=avg_budget_change_pct,
        budget_change_volatility=budget_change_volatility,
        high_roas_budget_pct=high_roas_budget_pct,
        low_roas_budget_pct=low_roas_budget_pct,
        baseline_comparison=baseline_comparison,
        avg_ctr=avg_ctr,
        weighted_avg_ctr=weighted_avg_ctr,
        ctr_std=ctr_std,
        high_ctr_budget_pct=high_ctr_budget_pct,
    )
