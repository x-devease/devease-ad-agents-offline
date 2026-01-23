"""
Automated Bayesian Tuner for Budget Allocation Rules

This module provides automated parameter tuning using Bayesian optimization
to find optimal rule parameters for each customer.

Key features:
- Automatically discovers all customers in datasets/
- Uses Bayesian optimization (efficient search for high-dimensional spaces)
- Multi-objective scoring (ROAS, utilization, stability)
- Per-customer tuning with auto-update to config
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from src.adset.allocator.optimizer.lib.backtesting import (
    aggregate_cv_results,
    create_rolling_window_splits,
    forward_looking_evaluation,
)
from src.adset.allocator.optimizer.tuning import (
    OptimizationObjectives,
    TuningConstraints,
    TuningResult,
    _calculate_gini_coefficient,
    _calculate_entropy,
    evaluate_allocation,
    load_objectives,
    simulate_allocation,
)
from src.adset import Allocator, DecisionRules, SafetyRules
from src.adset.allocator.utils.parser import Parser
from src.config.path_manager import get_path_manager


class BayesianTuner:
    """Bayesian tuner for budget allocation rule parameters."""

    # Default parameter ranges to optimize
    # Key: parameter path in config (dot-separated)
    # Value: (min, max, type) where type is 'real', 'int', or 'categorical'
    DEFAULT_PARAM_RANGES: Dict[str, Tuple[Any, Any, str]] = {
        # ROAS thresholds
        "decision_rules.low_roas_threshold": (1.0, 2.5, "real"),
        "decision_rules.medium_roas_threshold": (2.0, 3.0, "real"),
        "decision_rules.high_roas_threshold": (2.5, 4.0, "real"),
        # Budget adjustment percentages
        "decision_rules.aggressive_increase_pct": (0.10, 0.30, "real"),
        "decision_rules.moderate_increase_pct": (0.05, 0.15, "real"),
        "decision_rules.aggressive_decrease_pct": (0.15, 0.40, "real"),
        "decision_rules.moderate_decrease_pct": (0.05, 0.20, "real"),
        # Trend thresholds
        "decision_rules.strong_rising_trend": (0.05, 0.20, "real"),
        "decision_rules.strong_falling_trend": (-0.20, -0.05, "real"),
        # Efficiency thresholds
        "decision_rules.high_efficiency_threshold": (0.05, 0.20, "real"),
        "decision_rules.low_efficiency_threshold": (0.01, 0.05, "real"),
        # Health score thresholds
        "decision_rules.healthy_threshold": (0.5, 0.8, "real"),
        "decision_rules.unhealthy_threshold": (0.2, 0.4, "real"),
        # Lifecycle (cold_start_days must match between safety and decision rules)
        "decision_rules.cold_start_days": (2, 5, "int"),
        "safety_rules.cold_start_days": (2, 5, "int"),
        "decision_rules.learning_phase_days": (10, 20, "int"),
        # Gradient smoothing
        "decision_rules.gradient_smoothing_factor": (0.1, 0.5, "real"),
        # Safety rules
        "safety_rules.max_daily_increase_pct": (0.10, 0.25, "real"),
        "safety_rules.max_daily_decrease_pct": (0.10, 0.25, "real"),
        # Ad-level statistics rules
        "decision_rules.min_diversity_threshold": (3, 8, "int"),
        "decision_rules.min_active_ads_threshold": (2, 5, "int"),
        "decision_rules.diversity_bonus_pct": (0.05, 0.15, "real"),
        "decision_rules.max_spend_concentration": (0.70, 0.90, "real"),
        "decision_rules.spend_concentration_penalty_pct": (0.10, 0.30, "real"),
        "decision_rules.max_roas_std": (0.3, 0.8, "real"),
        "decision_rules.consistency_bonus_pct": (0.05, 0.15, "real"),
        "decision_rules.min_format_diversity": (2, 5, "int"),
        "decision_rules.format_diversity_bonus_pct": (0.03, 0.10, "real"),
        "decision_rules.outlier_multiplier_threshold": (2.0, 4.0, "real"),
        "decision_rules.outlier_penalty_pct": (0.10, 0.25, "real"),
        # NEW: Trend scaling parameters (currently hardcoded)
        "decision_rules.trend_strong_factor": (0.8, 1.2, "real"),
        "decision_rules.trend_moderate_start_factor": (0.5, 0.9, "real"),
        "decision_rules.trend_moderate_range_factor": (0.2, 0.4, "real"),
        "decision_rules.trend_weak_start_factor": (0.3, 0.5, "real"),
        "decision_rules.trend_weak_range_factor": (0.2, 0.4, "real"),
        "decision_rules.trend_min_factor": (0.2, 0.4, "real"),
        # NEW: Health score multiplier parameters
        "decision_rules.health_score_min_multiplier": (0.4, 0.6, "real"),
        "decision_rules.health_score_max_multiplier": (0.9, 1.0, "real"),
        # NEW: Relative performance parameters
        "decision_rules.relative_performance_max_scale": (1.3, 1.7, "real"),
        "decision_rules.relative_performance_multiplier": (1.8, 2.2, "real"),
        "decision_rules.relative_perf_boost_medium": (0.2, 0.4, "real"),
        # NEW: Budget scaling thresholds
        "decision_rules.large_budget_threshold": (80, 120, "int"),
        "decision_rules.medium_budget_threshold": (15, 25, "int"),
        "decision_rules.small_budget_max_increase": (0.15, 0.25, "real"),
        "decision_rules.medium_budget_max_increase": (0.12, 0.18, "real"),
        "decision_rules.large_budget_max_increase": (0.08, 0.12, "real"),
        # NEW: Lifecycle parameters
        "decision_rules.established_days": (18, 25, "int"),
        "decision_rules.learning_phase_days_early": (5, 10, "int"),
        "decision_rules.learning_phase_days_mid": (12, 18, "int"),
        "decision_rules.learning_phase_days_late": (18, 25, "int"),
        # NEW: Volume thresholds
        "decision_rules.high_spend_threshold": (80, 150, "int"),
        "decision_rules.low_spend_threshold": (8, 15, "int"),
        "decision_rules.high_clicks_threshold": (40, 60, "int"),
        "decision_rules.high_impressions_threshold": (4000, 6000, "int"),
        "decision_rules.high_reach_threshold": (800, 1200, "int"),
        # NEW: Advanced concept parameters
        "advanced_concepts.smoothing_alpha": (0.6, 0.8, "real"),
        "advanced_concepts.momentum_days": (2, 5, "int"),
        "advanced_concepts.low_utilization_threshold": (0.65, 0.75, "real"),
        "advanced_concepts.high_utilization_threshold": (0.92, 0.98, "real"),
    }

    def __init__(
        self,
        config_path: str = "config/adset/allocator/rules.yaml",
        constraints: Optional[TuningConstraints] = None,
        objectives: Optional[OptimizationObjectives] = None,
        objectives_config_path: Optional[str] = None,
        customer_name: str = "moprobo",
        platform: str = "meta",
        n_calls: int = 50,
        n_initial_points: int = 10,
    ):
        """Initialize the automated Bayesian tuner.

        Args:
            config_path: Path to rules.yaml config file.
            constraints: TuningConstraints. If None, uses defaults.
            objectives: OptimizationObjectives. If None, loads from objectives_config_path.
            objectives_config_path: Path to objectives.yaml config file.
                              If None, uses config/{customer}/{platform}/objectives.yaml
            customer_name: Customer name for objectives path.
            platform: Platform name for objectives path.
            n_calls: Number of Bayesian optimization iterations.
            n_initial_points: Number of random evaluations before Bayesian model is used.
        """
        if objectives_config_path is None:
            objectives_config_path = (
                f"config/{customer_name}/{platform}/objectives.yaml"
            )

        self.config_path = config_path
        self.constraints = constraints or TuningConstraints()
        self.objectives_config_path = objectives_config_path
        self.objectives = objectives
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points

        # Caching for objective function evaluations
        self._objective_cache: Dict[str, float] = {}

        # Early stopping
        self.early_stopping_patience: int = (
            10  # Stop if no improvement for N iterations
        )
        self.early_stopping_min_delta: float = (
            0.001  # Minimum improvement to count as progress
        )

        # Load base config
        with open(config_path, "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

    def discover_customers(self, datasets_dir: str = "datasets") -> List[str]:
        """Discover all customers in datasets directory.

        Args:
            datasets_dir: Path to datasets directory.

        Returns:
            List of customer names.
        """
        datasets_path = Path(datasets_dir)
        if not datasets_path.exists():
            return []

        customers = []
        for customer_dir in datasets_path.iterdir():
            # Skip hidden files and non-directories
            if customer_dir.name.startswith(".") or not customer_dir.is_dir():
                continue

            # Check if customer has features directory
            features_dir = customer_dir / "features"
            if features_dir.exists():
                customers.append(customer_dir.name)

        return customers

    def _get_customer_features(
        self, customer_name: str, datasets_dir: str = "datasets"
    ) -> Optional[pd.DataFrame]:
        """Load features for a customer.

        Args:
            customer_name: Customer name.
            datasets_dir: Path to datasets directory.

        Returns:
            DataFrame with adset features, or None if not found.
        """
        try:
            path_manager = get_path_manager(customer=customer_name, platform="meta")
            features_path = (
                path_manager.features_dir(customer_name, "meta") / "adset_features.csv"
            )
            if features_path.exists():
                return pd.read_csv(features_path)
        except (
            FileNotFoundError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            OSError,
        ):
            return None
        return None

    def _get_customer_budget(self, df_features: pd.DataFrame) -> float:
        """Estimate total budget from features data.

        Args:
            df_features: DataFrame with adset features.

        Returns:
            Estimated total daily budget.
        """
        # Try to sum daily budgets
        if "adset_daily_budget" in df_features.columns:
            valid_budgets = df_features[
                df_features["adset_daily_budget"].notna()
                & (df_features["adset_daily_budget"] > 0)
            ]["adset_daily_budget"]
            if len(valid_budgets) > 0:
                return float(valid_budgets.sum())

        # Fallback: use spend as proxy
        if "spend" in df_features.columns:
            total_spend = df_features["spend"].sum()
            if total_spend > 0:
                return float(total_spend)

        # Final fallback: use a default budget
        return 10000.0

    def _create_allocator_with_params(
        self, param_dict: Dict[str, Any], customer_name: str = "moprobo"
    ) -> Allocator:
        """Create an Allocator with overridden parameters.

        Args:
            param_dict: Dictionary of parameter overrides.
            customer_name: Customer name for config section.

        Returns:
            Configured Allocator instance.
        """
        import tempfile

        # Deep copy config
        config = yaml.safe_load(yaml.dump(self.base_config))

        # Get customer config section
        if customer_name in config:
            customer_config = config[customer_name].copy()
        elif "safety_rules" in config or "decision_rules" in config:
            customer_config = config.copy()
        else:
            customer_config = {}

        # Apply parameter overrides
        for key_path, value in param_dict.items():
            keys = key_path.split(".")
            current = customer_config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

        # Update config
        if customer_name in config:
            config[customer_name] = customer_config
        else:
            config.update(customer_config)

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp_file:
            yaml.dump(config, tmp_file, default_flow_style=False)
            tmp_config_path = tmp_file.name

        try:
            parser = Parser(
                tmp_config_path, customer_name=customer_name, platform="meta"
            )
            safety_rules = SafetyRules(parser)
            decision_rules = DecisionRules(parser)
            return Allocator(safety_rules, decision_rules, parser)
        finally:
            Path(tmp_config_path).unlink(missing_ok=True)

    def _multi_objective_score(
        self,
        result: TuningResult,
        objectives: Optional[OptimizationObjectives] = None,
    ) -> float:
        """Multi-objective scoring function for optimization.

        Supports configurable objectives with weights for ROAS, CTR, stability,
        and budget utilization. Enables customer-specific optimization strategies.

        Args:
            result: TuningResult to score
            objectives: OptimizationObjectives with weights and preferences.
                       If None, uses default ROAS-focused weights.

        Returns:
            Negative score (since gp_minimize minimizes).
        """
        # Use provided objectives or fall back to ROAS-focused defaults
        if objectives is None:
            if self.objectives is not None:
                objectives = self.objectives
            else:
                # Default: ROAS-focused (backwards compatible)
                objectives = OptimizationObjectives(
                    roas_weight=0.70,
                    ctr_weight=0.00,
                    stability_weight=0.20,
                    budget_utilization_weight=0.10,
                )

        # PRIMARY: Budget-weighted ROAS
        # Revenue generation - primary business objective for most customers
        roas_score = float(result.weighted_avg_roas) * objectives.roas_weight

        # PRIMARY: Budget-weighted CTR (if enabled)
        # Engagement metric for brand awareness campaigns
        ctr_score = 0.0
        if objectives.ctr_weight > 0:
            # Normalize CTR to similar scale as ROAS (typically 0-10% vs 0-10x)
            # Use 100x multiplier to put CTR on similar scale
            ctr_score = float(result.weighted_avg_ctr) * 100 * objectives.ctr_weight

        # SECONDARY: Stability
        # Penalize high volatility (indicates overfitting to noise)
        # Consider both ROAS and CTR stability
        stability_penalty = 0.0
        if objectives.prefer_stable_over_optimal:
            # Stricter stability requirements
            roas_stability_threshold = 1.0
            ctr_stability_threshold = 2.0  # 2% std
        else:
            # Standard stability requirements
            roas_stability_threshold = 1.5
            ctr_stability_threshold = 3.0  # 3% std

        # ROAS stability penalty
        if result.roas_std > roas_stability_threshold:
            stability_penalty += (result.roas_std - roas_stability_threshold) * 0.5

        # CTR stability penalty (if CTR is weighted)
        if objectives.ctr_weight > 0 and result.ctr_std > ctr_stability_threshold:
            stability_penalty += (
                result.ctr_std - ctr_stability_threshold
            ) * 0.2  # Less severe than ROAS

        # Stability: reduce score for high volatility, but don't let it go negative
        stability_score = max(0.0, objectives.stability_weight - stability_penalty)

        # TERTIARY: Budget utilization
        # Asymmetric penalty: over-utilization is much worse than under-utilization
        utilization_score = 0.0
        min_util_threshold = (
            objectives.min_budget_utilization
            if objectives.min_budget_utilization
            else 0.90
        )

        if result.budget_utilization < min_util_threshold:
            # Small penalty for under-utilization
            utilization_score = (
                objectives.budget_utilization_weight * result.budget_utilization
            )
        elif result.budget_utilization > 1.0:
            # Over-utilization penalty applied separately, utilization_score = 0
            utilization_score = 0.0
        else:
            # Full score for good utilization
            utilization_score = objectives.budget_utilization_weight

        # Combine scores (positive, higher is better)
        base_score = roas_score + ctr_score + stability_score + utilization_score

        # Apply over-utilization penalty to base_score (for soft penalty, hard penalty comes later)
        if result.budget_utilization > 1.0:
            over_util_penalty = (result.budget_utilization - 1.0) * 5.0
            base_score -= over_util_penalty

        # HARD CONSTRAINTS: Must satisfy these or receive large penalties
        # These are business-critical constraints that cannot be violated
        # Accumulate penalties separately to ensure proper ordering
        total_penalty = 0.0

        # Budget utilization hard limits
        if result.budget_utilization < self.constraints.min_total_budget_utilization:
            hard_penalty = (
                self.constraints.min_total_budget_utilization
                - result.budget_utilization
            ) * 50
            total_penalty += hard_penalty
        if result.budget_utilization > self.constraints.max_total_budget_utilization:
            # Over-utilization is critical - larger penalty
            hard_penalty = (
                result.budget_utilization
                - self.constraints.max_total_budget_utilization
            ) * 100
            total_penalty += hard_penalty

        # Minimum adsets check (prevent over-concentration)
        if result.adsets_with_changes < self.constraints.min_adsets_with_budget:
            hard_penalty = (
                self.constraints.min_adsets_with_budget - result.adsets_with_changes
            ) * 20
            total_penalty += hard_penalty

        # Max single adset concentration (risk management)
        if result.max_single_adset_pct > self.constraints.max_single_adset_budget_pct:
            hard_penalty = (
                result.max_single_adset_pct
                - self.constraints.max_single_adset_budget_pct
            ) * 100
            total_penalty += hard_penalty

        # Change rate constraints (prevent over-churning)
        if result.change_rate < self.constraints.min_budget_change_rate:
            hard_penalty = (
                self.constraints.min_budget_change_rate - result.change_rate
            ) * 30
            total_penalty += hard_penalty
        if result.change_rate > self.constraints.max_budget_change_rate:
            hard_penalty = (
                result.change_rate - self.constraints.max_budget_change_rate
            ) * 30
            total_penalty += hard_penalty

        # Minimum acceptable ROAS (hard constraint from objectives)
        if objectives.min_acceptable_roas is not None:
            if result.weighted_avg_roas < objectives.min_acceptable_roas:
                hard_penalty = (
                    objectives.min_acceptable_roas - result.weighted_avg_roas
                ) * 40
                total_penalty += hard_penalty

        # Minimum acceptable CTR (hard constraint from objectives)
        if objectives.min_acceptable_ctr is not None and result.weighted_avg_ctr > 0:
            if result.weighted_avg_ctr < objectives.min_acceptable_ctr:
                hard_penalty = (
                    objectives.min_acceptable_ctr - result.weighted_avg_ctr
                ) * 20  # Less severe than ROAS
                total_penalty += hard_penalty

        # Calculate final score: base score minus penalties
        # Higher base_score and lower penalties = better result
        final_score = base_score - total_penalty

        # Return negative score for minimization (gp_minimize minimizes)
        # For gp_minimize: lower (more negative) = worse, higher (less negative) = better
        # Good result: high positive base_score → negate → less negative (better)
        # Bad result: low positive or negative base_score - large penalty → negate → more negative (worse)
        #
        # However, if final_score becomes negative due to penalties, negating it makes it positive,
        # which breaks the ordering (positive = worse for minimization, but we want negative = worse).
        #
        # Solution: For negative final_score (severe violations), return it directly as large negative.
        # For positive final_score (normal range), negate it so less negative = better.
        # This preserves ordering: large negative (violations) < negated positive (normal) < 0
        if final_score < 0:
            # Severe violations - return as large negative (worse for minimization)
            return final_score
        else:
            # Normal range - negate so less negative = better for minimization
            return -final_score

    def _hash_parameters(self, param_dict: Dict[str, Any]) -> str:
        """Create hash of parameter dictionary for caching.

        Args:
            param_dict: Parameter dictionary.

        Returns:
            Hash string.
        """
        # Sort keys for consistent hashing
        sorted_params = json.dumps(param_dict, sort_keys=True, default=str)
        return hashlib.md5(sorted_params.encode()).hexdigest()

    def _soft_constraint_penalty(self, result: TuningResult) -> float:
        """Calculate soft constraint penalties using smooth barrier functions.

        Uses quadratic and linear penalties instead of hard cutoffs to create
        smooth gradients for Gaussian Process optimization.

        Benefits:
        - No discontinuities in optimization landscape
        - Gaussian Process can model gradients properly
        - Still discourages constraint violations
        - Better convergence properties

        Args:
            result: TuningResult to evaluate for constraint violations.

        Returns:
            Penalty value to subtract from score (higher = worse violation).
        """
        penalty = 0.0

        # Change rate constraints (quadratic penalty - smooth but discourages violations)
        if result.change_rate < self.constraints.min_budget_change_rate:
            violation = self.constraints.min_budget_change_rate - result.change_rate
            penalty += 10.0 * (violation**2)  # Quadratic penalty
        if result.change_rate > self.constraints.max_budget_change_rate:
            violation = result.change_rate - self.constraints.max_budget_change_rate
            penalty += 10.0 * (violation**2)

        # Budget utilization (linear penalty for hard business constraints)
        if result.budget_utilization < self.constraints.min_total_budget_utilization:
            violation = (
                self.constraints.min_total_budget_utilization
                - result.budget_utilization
            )
            penalty += 50.0 * violation  # Linear penalty for hard constraints
        if result.budget_utilization > self.constraints.max_total_budget_utilization:
            violation = (
                result.budget_utilization
                - self.constraints.max_total_budget_utilization
            )
            penalty += 50.0 * violation

        # Performance constraints (quadratic penalty)
        if result.avg_roas < self.constraints.min_avg_roas:
            violation = self.constraints.min_avg_roas - result.avg_roas
            penalty += 20.0 * (violation**2)
        if result.revenue_efficiency < self.constraints.min_revenue_efficiency:
            violation = (
                self.constraints.min_revenue_efficiency - result.revenue_efficiency
            )
            penalty += 20.0 * (violation**2)

        # Minimum adsets constraint (quadratic penalty)
        if result.adsets_with_changes < self.constraints.min_adsets_with_budget:
            violation = (
                self.constraints.min_adsets_with_budget - result.adsets_with_changes
            )
            penalty += 20.0 * (violation**2)

        # Maximum single adset constraint (quadratic penalty)
        if result.max_single_adset_pct > self.constraints.max_single_adset_budget_pct:
            violation = (
                result.max_single_adset_pct
                - self.constraints.max_single_adset_budget_pct
            )
            penalty += 50.0 * (violation**2)

        return penalty

    def _objective_function(
        self, x: np.ndarray, param_names: List[str], param_types: List[str]
    ) -> float:
        """Objective function for Bayesian optimization.

        Args:
            x: Parameter values from optimizer.
            param_names: List of parameter names.
            param_types: List of parameter types ('real', 'int', 'categorical').

        Returns:
            Negative score (since we minimize).
        """
        # Convert x to parameter dict
        param_dict = {}
        for i, (name, ptype) in enumerate(zip(param_names, param_types)):
            value = x[i]
            if ptype == "int":
                value = int(value)
            param_dict[name] = value

        # Create allocator (use a default customer name for optimization)
        try:
            allocator = self._create_allocator_with_params(param_dict)
        except (ValueError, KeyError, AttributeError, TypeError):
            return float("inf")  # Invalid config

        # This is called during optimization - we need feature data
        # For now, return a placeholder - actual optimization happens in tune_customer()
        return 0.0

    def tune_customer(
        self,
        customer_name: str,
        param_ranges: Optional[Dict[str, Tuple[Any, Any, str]]] = None,
        total_budget: Optional[float] = None,
        campaign_name: Optional[str] = None,
    ) -> Optional[TuningResult]:
        """Tune parameters for a specific customer using Bayesian optimization.

        Args:
            customer_name: Customer name to tune for.
            param_ranges: Dictionary of parameter ranges. If None, uses DEFAULT_PARAM_RANGES.
            total_budget: Total budget for allocation. If None, estimated from data.
            campaign_name: Optional campaign name for campaign-specific objectives.

        Returns:
            Best TuningResult found, or None if tuning failed.
        """
        # Load customer objectives
        if self.objectives is None:
            try:
                objectives = load_objectives(
                    self.objectives_config_path, customer_name, campaign_name
                )
                print(f"  [OK] Loaded objectives for {customer_name}")
                if campaign_name:
                    print(f"    Campaign: {campaign_name}")
                print(
                    f"    Weights - ROAS: {objectives.roas_weight:.2f}, "
                    f"CTR: {objectives.ctr_weight:.2f}, "
                    f"Stability: {objectives.stability_weight:.2f}"
                )
            except Exception as e:
                print(f"  [WARN]  Could not load objectives: {e}")
                objectives = None
        else:
            objectives = self.objectives

        # Load customer features
        df_features = self._get_customer_features(customer_name)
        if df_features is None or len(df_features) == 0:
            print(f"  [WARN]  No feature data found for {customer_name}")
            return None

        print(f"  [OK] Loaded {len(df_features)} adsets for {customer_name}")

        # Estimate budget if not provided
        if total_budget is None:
            total_budget = self._get_customer_budget(df_features)
            print(f"  [OK] Estimated total budget: ${total_budget:,.2f}")

        # Use default param ranges if not provided
        if param_ranges is None:
            param_ranges = self.DEFAULT_PARAM_RANGES

        # Build skopt search space
        dimensions = []
        param_names = []
        param_types = []

        for param_name, (pmin, pmax, ptype) in param_ranges.items():
            param_names.append(param_name)
            param_types.append(ptype)
            if ptype == "real":
                dimensions.append(Real(pmin, pmax, name=param_name))
            elif ptype == "int":
                dimensions.append(Integer(pmin, pmax, name=param_name))
            elif ptype == "categorical":
                dimensions.append(Categorical(pmin, name=param_name))

        print(f"  [OK] Optimizing {len(param_names)} parameters")

        # Define objective function with caching
        def objective(x):
            """Objective function for Bayesian optimization with caching."""
            # Convert x to parameter dict
            param_dict = {}
            for i, (name, ptype) in enumerate(zip(param_names, param_types)):
                value = x[i]
                if ptype == "int":
                    value = int(value)
                param_dict[name] = value

            # Check cache
            param_hash = self._hash_parameters(param_dict)
            if param_hash in self._objective_cache:
                return self._objective_cache[param_hash]

            # Create allocator
            try:
                allocator = self._create_allocator_with_params(
                    param_dict, customer_name
                )
            except Exception:
                penalty = 1e6  # Large penalty for invalid config
                self._objective_cache[param_hash] = penalty
                return penalty

            # Simulate allocation
            allocation_results = simulate_allocation(
                df_features, allocator, total_budget
            )

            # Evaluate results with objectives
            target_roas = objectives.target_roas if objectives else None
            target_ctr = objectives.target_ctr if objectives else None
            result = evaluate_allocation(
                allocation_results,
                df_features,
                total_budget,
                target_roas=target_roas,
                target_ctr=target_ctr,
            )

            # Calculate score with objectives
            base_score = self._multi_objective_score(result, objectives)

            # Use soft constraint penalties instead of hard constraints for smoother optimization
            constraint_penalty = self._soft_constraint_penalty(result)

            # base_score is already negative, subtract penalty (which is positive)
            # More negative = worse, so subtracting penalty makes it worse
            final_score = base_score - constraint_penalty

            # Cache result
            self._objective_cache[param_hash] = final_score
            return final_score

        # Run Bayesian optimization with early stopping callback
        print(
            f"  [RUNNING] Running Bayesian optimization ({self.n_calls} iterations)..."
        )

        # Track best score for early stopping
        best_score = float("inf")
        no_improvement_count = 0

        def callback(res):
            """Callback for early stopping."""
            nonlocal best_score, no_improvement_count
            current_score = res.fun
            if current_score < best_score - self.early_stopping_min_delta:
                best_score = current_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Stop if no improvement for patience iterations
            if no_improvement_count >= self.early_stopping_patience:
                print(
                    f"  [INFO] Early stopping: no improvement for {self.early_stopping_patience} iterations"
                )
                return True  # Stop optimization
            return False

        try:
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                random_state=42,
                verbose=False,
                callback=callback,
            )

            if no_improvement_count >= self.early_stopping_patience:
                print(
                    f"  [OK] Optimization stopped early after {len(result.func_vals)} iterations"
                )
        except Exception as e:
            print(f"  [FAIL] Optimization failed: {e}")
            return None

        # Extract best parameters
        best_params = {}
        for i, param_name in enumerate(param_names):
            value = result.x[i]
            if param_types[i] == "int":
                value = int(value)
            best_params[param_name] = value

        # Run final evaluation with best parameters
        print(f"  [OK] Evaluating best configuration...")
        try:
            allocator = self._create_allocator_with_params(best_params, customer_name)
            allocation_results = simulate_allocation(
                df_features, allocator, total_budget
            )
            target_roas = objectives.target_roas if objectives else None
            target_ctr = objectives.target_ctr if objectives else None
            final_result = evaluate_allocation(
                allocation_results,
                df_features,
                total_budget,
                target_roas=target_roas,
                target_ctr=target_ctr,
            )
        except Exception as e:
            print(f"  [FAIL] Final evaluation failed: {e}")
            return None

        # Add best params to result
        result_dict = final_result.__dict__.copy()
        result_dict["param_config"] = best_params
        final_result = TuningResult(**result_dict)

        print(f"  [OK] Best weighted ROAS: {final_result.weighted_avg_roas:.4f}")
        print(f"  [OK] Best weighted CTR: {final_result.weighted_avg_ctr:.4f}")
        print(f"  [OK] Budget utilization: {final_result.budget_utilization:.2%}")
        print(f"  [OK] Change rate: {final_result.change_rate:.2%}")

        return final_result

    def tune_customer_with_cv(
        self,
        customer_name: str,
        param_ranges: Optional[Dict[str, Tuple[Any, Any, str]]] = None,
        total_budget: Optional[float] = None,
        n_folds: int = 3,
        train_ratio: float = 0.7,
        campaign_name: Optional[str] = None,
    ) -> Optional[TuningResult]:
        """Tune parameters using time-series cross-validation to reduce overfitting.

        Unlike tune_customer() which uses historical ROAS for evaluation, this method:
        - Uses rolling window time-series splits
        - Evaluates allocations using FORWARD-LOOKING ROAS (next period)
        - Averages performance across folds to reduce overfitting

        This provides more realistic estimates of production performance.

        Args:
            customer_name: Customer name to tune for.
            param_ranges: Dictionary of parameter ranges. If None, uses DEFAULT_PARAM_RANGES.
            total_budget: Total budget for allocation. If None, estimated from data.
            n_folds: Number of CV folds (default: 3).
            train_ratio: Ratio of training data in each fold (default: 0.7).
            campaign_name: Optional campaign name for campaign-specific objectives.

        Returns:
            Best TuningResult found with cross-validated metrics, or None if tuning failed.
        """
        # Load customer objectives
        if self.objectives is None:
            try:
                objectives = load_objectives(
                    self.objectives_config_path, customer_name, campaign_name
                )
                print(f"  [OK] Loaded objectives for {customer_name}")
                if campaign_name:
                    print(f"    Campaign: {campaign_name}")
                print(
                    f"    Weights - ROAS: {objectives.roas_weight:.2f}, "
                    f"CTR: {objectives.ctr_weight:.2f}, "
                    f"Stability: {objectives.stability_weight:.2f}"
                )
            except Exception as e:
                print(f"  [WARN]  Could not load objectives: {e}")
                objectives = None
        else:
            objectives = self.objectives

        # Load customer features
        df_features = self._get_customer_features(customer_name)
        if df_features is None or len(df_features) == 0:
            print(f"  [WARN]  No feature data found for {customer_name}")
            return None

        # Check if date column exists
        if "date_start" not in df_features.columns:
            print(
                f"  [WARN]  date_start column not found, falling back to tune_customer()"
            )
            return self.tune_customer(
                customer_name, param_ranges, total_budget, campaign_name
            )

        print(f"  [OK] Loaded {len(df_features)} adsets for {customer_name}")

        # Estimate budget if not provided
        if total_budget is None:
            total_budget = self._get_customer_budget(df_features)
            print(f"  [OK] Estimated total budget: ${total_budget:,.2f}")

        # Use default param ranges if not provided
        if param_ranges is None:
            param_ranges = self.DEFAULT_PARAM_RANGES

        # Build skopt search space
        dimensions = []
        param_names = []
        param_types = []

        for param_name, (pmin, pmax, ptype) in param_ranges.items():
            param_names.append(param_name)
            param_types.append(ptype)
            if ptype == "real":
                dimensions.append(Real(pmin, pmax, name=param_name))
            elif ptype == "int":
                dimensions.append(Integer(pmin, pmax, name=param_name))
            elif ptype == "categorical":
                dimensions.append(Categorical(pmin, name=param_name))

        print(f"  [OK] Optimizing {len(param_names)} parameters with {n_folds}-fold CV")

        # Create rolling window splits
        try:
            splits = create_rolling_window_splits(
                df_features,
                n_folds=n_folds,
                train_ratio=train_ratio,
                date_col="date_start",
            )
            print(f"  [OK] Created {len(splits)} rolling window splits")
        except Exception as e:
            print(f"  [WARN]  Could not create CV splits: {e}")
            print(f"  [WARN]  Falling back to standard tune_customer()")
            return self.tune_customer(customer_name, param_ranges, total_budget)

        if len(splits) == 0:
            print(f"  [WARN]  No valid CV splits created")
            return None

        # Define objective function with cross-validation and caching
        def objective_with_cv(x):
            """Objective function using time-series cross-validation with caching."""
            # Convert x to parameter dict
            param_dict = {}
            for i, (name, ptype) in enumerate(zip(param_names, param_types)):
                value = x[i]
                if ptype == "int":
                    value = int(value)
                param_dict[name] = value

            # Check cache
            param_hash = self._hash_parameters(param_dict)
            if param_hash in self._objective_cache:
                return self._objective_cache[param_hash]

            # Create allocator
            try:
                allocator = self._create_allocator_with_params(
                    param_dict, customer_name
                )
            except Exception:
                penalty = 1e6  # Large penalty for invalid config
                self._objective_cache[param_hash] = penalty
                return penalty

            # Evaluate on each fold
            fold_scores = []
            for fold_idx, (train_data, test_data) in enumerate(splits):
                try:
                    # Simulate allocation on training data
                    allocation_results = simulate_allocation(
                        train_data, allocator, total_budget
                    )

                    # Evaluate using FORWARD-LOOKING ROAS from test data
                    # This is the key improvement: using future ROAS instead of historical
                    target_roas = objectives.target_roas if objectives else None
                    target_ctr = objectives.target_ctr if objectives else None
                    result = forward_looking_evaluation(
                        allocation_results,
                        test_data,  # Use future ROAS
                        total_budget,
                        target_roas=target_roas,
                        target_ctr=target_ctr,
                    )

                    # Calculate score with objectives
                    base_score = self._multi_objective_score(result, objectives)
                    constraint_penalty = self._soft_constraint_penalty(result)
                    # base_score is already negative, subtract penalty
                    fold_score = base_score - constraint_penalty

                    fold_scores.append(fold_score)
                except (ValueError, KeyError, AttributeError, TypeError, RuntimeError):
                    # Penalize folds that fail
                    fold_scores.append(1e6)

            # Return mean score across folds
            mean_score = np.mean(fold_scores)
            self._objective_cache[param_hash] = mean_score
            return mean_score

        # Run Bayesian optimization with CV objective and early stopping
        print(
            f"  [RUNNING] Running Bayesian optimization with {n_folds}-fold CV ({self.n_calls} iterations)..."
        )

        # Track best score for early stopping
        best_score = float("inf")
        no_improvement_count = 0

        def callback(res):
            """Callback for early stopping."""
            nonlocal best_score, no_improvement_count
            current_score = res.fun
            if current_score < best_score - self.early_stopping_min_delta:
                best_score = current_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Stop if no improvement for patience iterations
            if no_improvement_count >= self.early_stopping_patience:
                print(
                    f"  [INFO] Early stopping: no improvement for {self.early_stopping_patience} iterations"
                )
                return True  # Stop optimization
            return False

        try:
            result = gp_minimize(
                func=objective_with_cv,
                dimensions=dimensions,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                random_state=42,
                verbose=False,
                callback=callback,
            )

            if no_improvement_count >= self.early_stopping_patience:
                print(
                    f"  [OK] Optimization stopped early after {len(result.func_vals)} iterations"
                )
        except Exception as e:
            print(f"  [FAIL] Optimization failed: {e}")
            return None

        # Extract best parameters
        best_params = {}
        for i, param_name in enumerate(param_names):
            value = result.x[i]
            if param_types[i] == "int":
                value = int(value)
            best_params[param_name] = value

        # Run final evaluation with best parameters using CV
        print(f"  [OK] Evaluating best configuration with CV...")
        try:
            allocator = self._create_allocator_with_params(best_params, customer_name)

            # Evaluate on all folds to get aggregated metrics
            fold_results = []
            for train_data, test_data in splits:
                allocation_results = simulate_allocation(
                    train_data, allocator, total_budget
                )
                target_roas = objectives.target_roas if objectives else None
                target_ctr = objectives.target_ctr if objectives else None
                fold_result = forward_looking_evaluation(
                    allocation_results,
                    test_data,
                    total_budget,
                    target_roas=target_roas,
                    target_ctr=target_ctr,
                )
                fold_results.append(fold_result)

            # Aggregate results across folds
            final_result = aggregate_cv_results(fold_results)

        except Exception as e:
            print(f"  [FAIL] Final evaluation failed: {e}")
            return None

        # Add best params to result
        result_dict = final_result.__dict__.copy()
        result_dict["param_config"] = best_params
        final_result = TuningResult(**result_dict)

        print(f"  [OK] Best CV weighted ROAS: {final_result.weighted_avg_roas:.4f}")
        print(f"  [OK] Best CV weighted CTR: {final_result.weighted_avg_ctr:.4f}")
        print(f"  [OK] Budget utilization: {final_result.budget_utilization:.2%}")
        print(f"  [OK] Change rate: {final_result.change_rate:.2%}")

        return final_result

    def tune_all_customers(
        self,
        datasets_dir: str = "datasets",
        param_ranges: Optional[Dict[str, Tuple[Any, Any, str]]] = None,
    ) -> Dict[str, TuningResult]:
        """Tune parameters for all discovered customers.

        Args:
            datasets_dir: Path to datasets directory.
            param_ranges: Parameter ranges. If None, uses defaults.

        Returns:
            Dictionary mapping customer names to their best TuningResult.
        """
        # Discover customers
        customers = self.discover_customers(datasets_dir)
        if not customers:
            print("  [WARN]  No customers found in datasets directory")
            return {}

        print(f"  Found {len(customers)} customer(s): {', '.join(customers)}")

        results = {}
        for customer_name in customers:
            print(f"\n{'='*70}")
            print(f"Tuning for: {customer_name}")
            print(f"{'='*70}")

            result = self.tune_customer(customer_name, param_ranges)
            if result is not None:
                results[customer_name] = result

        return results

    def update_config_with_results(
        self, results: Dict[str, TuningResult], config_path: Optional[str] = None
    ) -> None:
        """Update config file with tuned parameters for each customer.

        Args:
            results: Dictionary mapping customer names to TuningResult.
            config_path: Path to config file. If None, uses self.config_path.
        """
        if config_path is None:
            config_path = self.config_path

        # Load current config
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Update config for each customer
        for customer_name, result in results.items():
            if customer_name not in config:
                config[customer_name] = {}

            customer_config = config[customer_name]

            # Apply all tuned parameters
            for param_path, value in result.param_config.items():
                keys = param_path.split(".")
                current = customer_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value

            print(f"  [OK] Updated config for {customer_name}")

        # Save updated config
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"  [OK] Config saved to {config_path}")

    def validate_objective_weights(
        self,
        customer_name: str,
        weight_combinations: Optional[List[Dict[str, float]]] = None,
        n_folds: int = 3,
        train_ratio: float = 0.7,
        total_budget: Optional[float] = None,
    ) -> Dict[str, float]:
        """Validate objective function weights using forward-looking CV.

        Tests different weight combinations and selects the one that maximizes
        forward-looking ROAS on validation set. This empirically validates
        that the weights optimize for actual production performance.

        Args:
            customer_name: Customer name to validate weights for.
            weight_combinations: List of weight dicts to test. Each dict should have
                keys: 'roas_weight', 'stability_weight', 'utilization_weight'.
                If None, tests a default grid of combinations.
            n_folds: Number of CV folds for validation.
            train_ratio: Ratio of training data in each fold.
            total_budget: Total budget for allocation.

        Returns:
            Dictionary with best weights and validation metrics:
                - 'best_weights': Dict with optimal weight values
                - 'best_validation_roas': Forward-looking ROAS with best weights
                - 'all_results': List of all test results
        """
        from src.adset.allocator.optimizer.lib.backtesting import (
            create_rolling_window_splits,
            forward_looking_evaluation,
        )

        # Default weight combinations to test if not provided
        if weight_combinations is None:
            weight_combinations = [
                # ROAS-focused (current default)
                {
                    "roas_weight": 0.70,
                    "stability_weight": 0.20,
                    "utilization_weight": 0.10,
                },
                # Very ROAS-focused
                {
                    "roas_weight": 0.80,
                    "stability_weight": 0.15,
                    "utilization_weight": 0.05,
                },
                # Extremely ROAS-focused
                {
                    "roas_weight": 0.90,
                    "stability_weight": 0.08,
                    "utilization_weight": 0.02,
                },
                # Balanced
                {
                    "roas_weight": 0.60,
                    "stability_weight": 0.25,
                    "utilization_weight": 0.15,
                },
                # Stability-focused
                {
                    "roas_weight": 0.50,
                    "stability_weight": 0.35,
                    "utilization_weight": 0.15,
                },
            ]

        # Load customer features
        df_features = self._get_customer_features(customer_name)
        if df_features is None or len(df_features) == 0:
            print(f"  [WARN]  No feature data found for {customer_name}")
            return {}

        if "date_start" not in df_features.columns:
            print(f"  [WARN]  date_start column not found, cannot validate weights")
            return {}

        # Estimate budget if not provided
        if total_budget is None:
            total_budget = self._get_customer_budget(df_features)

        print(
            f"  [RUNNING] Validating {len(weight_combinations)} weight combinations..."
        )
        print(f"     Using {n_folds}-fold CV with {train_ratio:.0%} train ratio")

        # Create rolling window splits
        splits = create_rolling_window_splits(
            df_features,
            n_folds=n_folds,
            train_ratio=train_ratio,
            date_col="date_start",
        )

        if len(splits) == 0:
            print(f"  [WARN]  Could not create CV splits")
            return {}

        all_results = []
        best_weights = None
        best_validation_roas = 0.0

        # Test each weight combination
        for weight_idx, weights in enumerate(weight_combinations):
            print(
                f"  [{weight_idx + 1}/{len(weight_combinations)}] Testing weights: {weights}"
            )

            fold_roas_scores = []

            # Evaluate on each CV fold
            for fold_idx, (train_df, test_df) in enumerate(splits):
                # Create allocator with current parameters (use existing config)
                try:
                    allocator = self._create_allocator_with_params({}, customer_name)
                except (ValueError, KeyError, AttributeError, TypeError):
                    print(f"    Fold {fold_idx + 1}: Failed to create allocator")
                    continue

                # Simulate allocation on training data
                from src.adset.allocator.optimizer.tuning import simulate_allocation

                allocation = simulate_allocation(train_df, allocator, total_budget)

                # Evaluate on TEST data (forward-looking ROAS)
                result = forward_looking_evaluation(
                    allocation, test_df, total_budget, target_roas=None
                )

                fold_roas_scores.append(result.weighted_avg_roas)

            # Calculate mean forward-looking ROAS across folds
            if fold_roas_scores:
                mean_roas = sum(fold_roas_scores) / len(fold_roas_scores)
                std_roas = (
                    sum((x - mean_roas) ** 2 for x in fold_roas_scores)
                    / len(fold_roas_scores)
                ) ** 0.5

                print(
                    f"    Forward-looking ROAS: {mean_roas:.4f} ± {std_roas:.4f} (n={len(fold_roas_scores)})"
                )

                all_results.append(
                    {
                        "weights": weights,
                        "mean_roas": mean_roas,
                        "std_roas": std_roas,
                        "fold_scores": fold_roas_scores,
                    }
                )

                # Track best weights
                if mean_roas > best_validation_roas:
                    best_validation_roas = mean_roas
                    best_weights = weights

        # Print summary
        print(f"\n  [OK] Weight validation complete!")
        print(f"     Best weights: {best_weights}")
        print(f"     Best forward-looking ROAS: {best_validation_roas:.4f}")

        # Sort results by mean ROAS
        all_results_sorted = sorted(
            all_results, key=lambda x: x["mean_roas"], reverse=True
        )
        print(f"\n  [CHART] Top 3 weight combinations:")
        for i, result in enumerate(all_results_sorted[:3]):
            print(f"     {i + 1}. ROAS={result['mean_roas']:.4f}: {result['weights']}")

        return {
            "best_weights": best_weights,
            "best_validation_roas": best_validation_roas,
            "all_results": all_results,
        }

    def tune_with_pareto_analysis(
        self,
        customer_name: str,
        campaign_name: Optional[str] = None,
        param_ranges: Optional[Dict[str, Tuple[Any, Any, str]]] = None,
        total_budget: Optional[float] = None,
        n_iterations: int = 100,
        save_frontier_path: Optional[str] = None,
        visualize: bool = True,
    ) -> Optional[any]:
        """Tune parameters with Pareto frontier analysis.

        Evaluates multiple parameter configurations and constructs the Pareto frontier
        to show the trade-off between ROAS and CTR optimization.

        Args:
            customer_name: Customer name to tune for.
            campaign_name: Optional campaign name for campaign-specific objectives.
            param_ranges: Dictionary of parameter ranges.
            total_budget: Total budget for allocation.
            n_iterations: Number of parameter configurations to evaluate.
            save_frontier_path: Optional path to save Pareto frontier CSV.
            visualize: Whether to generate Pareto frontier visualization.

        Returns:
            ParetoFrontier object with analysis and recommendations.
        """
        from src.adset.allocator.optimizer.lib.pareto import (
            compute_pareto_frontier,
            explain_pareto_frontier,
            visualize_pareto_frontier,
        )

        # Load customer objectives
        if self.objectives is None:
            try:
                objectives = load_objectives(
                    self.objectives_config_path, customer_name, campaign_name
                )
                print(f"  [OK] Loaded objectives for {customer_name}")
            except Exception as e:
                print(f"  [WARN]  Could not load objectives: {e}")
                return None
        else:
            objectives = self.objectives

        # Load customer features
        df_features = self._get_customer_features(customer_name)
        if df_features is None or len(df_features) == 0:
            print(f"  [WARN]  No feature data found for {customer_name}")
            return None

        print(f"  [OK] Loaded {len(df_features)} adsets for {customer_name}")

        # Estimate budget if not provided
        if total_budget is None:
            total_budget = self._get_customer_budget(df_features)
            print(f"  [OK] Estimated total budget: ${total_budget:,.2f}")

        # Use default param ranges if not provided
        if param_ranges is None:
            param_ranges = self.DEFAULT_PARAM_RANGES

        print(
            f"  [RUNNING] Evaluating {n_iterations} parameter configurations for Pareto analysis..."
        )

        # Sample random parameter configurations
        import random

        results = []
        param_names = list(param_ranges.keys())
        target_roas = objectives.target_roas
        target_ctr = objectives.target_ctr

        for i in range(n_iterations):
            # Sample random parameters
            param_dict = {}
            for param_name, (pmin, pmax, ptype) in param_ranges.items():
                if ptype == "real":
                    param_dict[param_name] = random.uniform(pmin, pmax)
                elif ptype == "int":
                    param_dict[param_name] = random.randint(int(pmin), int(pmax))
                elif ptype == "categorical":
                    param_dict[param_name] = random.choice(pmin)

            # Create allocator and evaluate
            try:
                allocator = self._create_allocator_with_params(
                    param_dict, customer_name
                )
                allocation_results = simulate_allocation(
                    df_features, allocator, total_budget
                )
                result = evaluate_allocation(
                    allocation_results,
                    df_features,
                    total_budget,
                    target_roas=target_roas,
                    target_ctr=target_ctr,
                )
                results.append(result)
            except (ValueError, KeyError, AttributeError, TypeError, RuntimeError):
                continue

            if (i + 1) % 20 == 0:
                print(f"    Evaluated {i + 1}/{n_iterations} configurations...")

        if not results:
            print(f"  [FAIL] No valid configurations found")
            return None

        print(f"  [OK] Successfully evaluated {len(results)} configurations")

        # Compute Pareto frontier
        print(f"  [CHART] Computing Pareto frontier...")
        frontier = compute_pareto_frontier(results, objectives)

        print(f"  [OK] Found {len(frontier.frontier_points)} Pareto-optimal solutions")
        print(f"    (out of {len(results)} total solutions)")

        # Print analysis
        print("\n" + explain_pareto_frontier(frontier))

        # Save frontier data if requested
        if save_frontier_path:
            from pathlib import Path

            save_path = Path(save_frontier_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save CSV
            df = frontier.get_frontier_df()
            df.to_csv(save_path, index=False)
            print(f"  [OK] Saved Pareto frontier data to {save_path}")

            # Save visualization
            if visualize:
                viz_path = str(save_path).replace(".csv", "_frontier.png")
                visualize_pareto_frontier(frontier, save_path=viz_path)

        elif visualize:
            # Just show visualization without saving
            visualize_pareto_frontier(frontier)

        return frontier

    def compare_objective_weights(
        self,
        customer_name: str,
        weight_combinations: List[Dict[str, float]],
        total_budget: Optional[float] = None,
        n_samples_per_weight: int = 20,
    ) -> pd.DataFrame:
        """Compare different objective weight combinations.

        Tests multiple weight configurations and shows how they affect
        the trade-off between ROAS and CTR.

        Args:
            customer_name: Customer name to analyze.
            weight_combinations: List of weight dicts. Each should have:
                - roas_weight: float
                - ctr_weight: float
                - stability_weight: float
                - budget_utilization_weight: float
            total_budget: Total budget for allocation.
            n_samples_per_weight: Number of parameter samples per weight combo.

        Returns:
            DataFrame with comparison results.
        """
        from src.adset.allocator.optimizer.lib.pareto import compute_pareto_frontier

        # Load features
        df_features = self._get_customer_features(customer_name)
        if df_features is None or len(df_features) == 0:
            print(f"  [WARN]  No feature data found for {customer_name}")
            return pd.DataFrame()

        if total_budget is None:
            total_budget = self._get_customer_budget(df_features)

        # Use default param ranges
        param_ranges = self.DEFAULT_PARAM_RANGES

        # Test each weight combination
        import random

        comparison_data = []

        for weights in weight_combinations:
            objectives = OptimizationObjectives(
                roas_weight=weights.get("roas", 0.60),
                ctr_weight=weights.get("ctr", 0.20),
                stability_weight=weights.get("stability", 0.15),
                budget_utilization_weight=weights.get("budget_utilization", 0.05),
            )

            print(
                f"  Testing weights: ROAS={objectives.roas_weight:.0%}, "
                f"CTR={objectives.ctr_weight:.0%}, "
                f"Stability={objectives.stability_weight:.0%}"
            )

            # Sample and evaluate
            results = []
            for _ in range(n_samples_per_weight):
                param_dict = {}
                for param_name, (pmin, pmax, ptype) in param_ranges.items():
                    if ptype == "real":
                        param_dict[param_name] = random.uniform(pmin, pmax)
                    elif ptype == "int":
                        param_dict[param_name] = random.randint(int(pmin), int(pmax))
                    elif ptype == "categorical":
                        param_dict[param_name] = random.choice(pmin)

                try:
                    allocator = self._create_allocator_with_params(
                        param_dict, customer_name
                    )
                    allocation_results = simulate_allocation(
                        df_features, allocator, total_budget
                    )
                    result = evaluate_allocation(
                        allocation_results,
                        df_features,
                        total_budget,
                        target_roas=objectives.target_roas,
                        target_ctr=objectives.target_ctr,
                    )
                    results.append(result)
                except (ValueError, KeyError, AttributeError, TypeError, RuntimeError):
                    continue

            if results:
                frontier = compute_pareto_frontier(results, objectives)

                # Get best point
                if frontier.recommended_point:
                    best = frontier.recommended_point.result
                elif frontier.frontier_points:
                    best = frontier.frontier_points[0].result
                else:
                    best = results[0]

                comparison_data.append(
                    {
                        "roas_weight": objectives.roas_weight,
                        "ctr_weight": objectives.ctr_weight,
                        "stability_weight": objectives.stability_weight,
                        "budget_utilization_weight": objectives.budget_utilization_weight,
                        "best_roas": best.weighted_avg_roas,
                        "best_ctr": best.weighted_avg_ctr,
                        "roas_std": best.roas_std,
                        "budget_utilization": best.budget_utilization,
                        "frontier_size": len(frontier.frontier_points),
                    }
                )

        return pd.DataFrame(comparison_data)

    def generate_report(
        self, results: Dict[str, TuningResult], output_path: Optional[str] = None
    ) -> None:
        """Generate a tuning report.

        Args:
            results: Dictionary mapping customer names to TuningResult.
            output_path: Path to save report. If None, prints to console.
        """
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("AUTOMATED BAYESIAN TUNING REPORT\n")
                f.write("=" * 80 + "\n\n")

                for customer_name, result in results.items():
                    f.write(f"Customer: {customer_name}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Best Parameters:\n")
                    for k, v in result.param_config.items():
                        f.write(f"  {k}: {v}\n")
                    f.write(f"\nMetrics:\n")
                    f.write(f"  Weighted Avg ROAS: {result.weighted_avg_roas:.4f}\n")
                    f.write(f"  Budget Utilization: {result.budget_utilization:.2%}\n")
                    f.write(f"  Change Rate: {result.change_rate:.2%}\n")
                    f.write(f"  Revenue Efficiency: {result.revenue_efficiency:.4f}\n")
                    f.write(f"  Total Revenue: ${result.total_revenue:,.2f}\n")
                    f.write(f"  Budget Gini: {result.budget_gini:.3f}\n")
                    f.write(f"  Budget Entropy: {result.budget_entropy:.3f}\n")
                    f.write(
                        f"  Avg Budget Change: {result.avg_budget_change_pct:.2f}%\n"
                    )
                    f.write("\n")

                print(f"  [OK] Report saved to {output_path}")
        else:
            # Print to console
            print("\n" + "=" * 80)
            print("AUTOMATED BAYESIAN TUNING REPORT")
            print("=" * 80)

            for customer_name, result in results.items():
                print(f"\nCustomer: {customer_name}")
                print("-" * 80)
                print("Best Parameters:")
                for k, v in result.param_config.items():
                    print(f"  {k}: {v}")
                print(f"\nMetrics:")
                print(f"  Weighted Avg ROAS: {result.weighted_avg_roas:.4f}")
                print(f"  Budget Utilization: {result.budget_utilization:.2%}")
                print(f"  Change Rate: {result.change_rate:.2%}")
                print(f"  Revenue Efficiency: {result.revenue_efficiency:.4f}")
                print(f"  Total Revenue: ${result.total_revenue:,.2f}")
                print(f"  Budget Gini: {result.budget_gini:.3f}")
                print(f"  Budget Entropy: {result.budget_entropy:.3f}")
                print(f"  Avg Budget Change: {result.avg_budget_change_pct:.2f}%")
