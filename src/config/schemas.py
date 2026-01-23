"""Configuration validation schemas using Pydantic.

Provides type-safe configuration with validation for all budget allocation
parameters.
"""

from pathlib import Path
from typing import Dict, List, Optional

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:
    # Fallback for environments without pydantic
    BaseModel = object

    class Field:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def field_validator(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


class SafetyRulesConfig(BaseModel):
    """Safety rule constraints for budget changes."""

    max_budget_increase: float = Field(
        default=0.5, ge=0, le=1, description="Max budget increase rate"
    )
    max_budget_decrease: float = Field(
        default=0.3, ge=0, le=1, description="Max budget decrease rate"
    )
    freeze_threshold_roas: float = Field(
        default=0.5, ge=0, description="ROAS below which budget is frozen"
    )
    min_budget: float = Field(default=10.0, ge=0, description="Minimum absolute budget")
    max_budget: float = Field(
        default=10000.0, gt=0, description="Maximum absolute budget"
    )
    # NEW: New adset budget initialization
    new_adset_initial_fraction: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Fraction of equal share for new adsets",
    )
    new_adset_max_initial_budget: float = Field(
        default=100.0,
        ge=1.0,
        description="Maximum initial budget for new adsets",
    )


class DecisionRulesConfig(BaseModel):
    """Decision rule thresholds and parameters."""

    excellent_roas_threshold: float = Field(
        default=4.0, ge=0, description="ROAS threshold for excellent performance"
    )
    good_roas_threshold: float = Field(
        default=2.0, ge=0, description="ROAS threshold for good performance"
    )
    target_roas_threshold: float = Field(
        default=1.0, ge=0, description="Target ROAS threshold"
    )
    poor_roas_threshold: float = Field(
        default=0.5, ge=0, description="ROAS threshold for poor performance"
    )

    excellent_increase: float = Field(
        default=0.5, ge=0, le=1, description="Budget increase rate for excellent"
    )
    good_increase: float = Field(
        default=0.2, ge=0, le=1, description="Budget increase rate for good"
    )
    poor_decrease: float = Field(
        default=0.2, ge=0, le=1, description="Budget decrease rate for poor"
    )
    terrible_decrease: float = Field(
        default=0.5, ge=0, le=1, description="Budget decrease rate for terrible"
    )

    learning_phase_days: int = Field(
        default=7, ge=0, description="Days to consider as learning phase"
    )


class AdvancedConceptsConfig(BaseModel):
    """Advanced allocation concepts configuration."""

    enable_adaptive_target: bool = Field(
        default=True, description="Enable adaptive target ROAS"
    )
    enable_marginal_roas: bool = Field(
        default=True, description="Enable marginal ROAS analysis"
    )
    smoothing_factor: float = Field(
        default=0.2, ge=0, le=1, description="Exponential smoothing factor"
    )


class MonthlyBudgetConfig(BaseModel):
    """Monthly budget tracking configuration."""

    monthly_budget_cap: float = Field(
        default=10000.0, ge=0, description="Monthly budget cap"
    )
    conservative_factor: float = Field(
        default=0.95,
        ge=0.8,
        le=1.0,
        description="Conservative spending factor (prevents front-loading)",
    )
    reset_day: int = Field(
        default=1, ge=1, le=31, description="Day of month to reset tracking"
    )
    archive_daily_allocations: bool = Field(
        default=True, description="Archive daily allocation files"
    )
    # NEW: Period management for mid-month starts
    period_length_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days in budget period (for mid-month starts)",
    )
    # NEW: Day 1 conservative handling
    day1_budget_multiplier: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Conservative multiplier for first day of period",
    )


class ObjectivesConfig(BaseModel):
    """Optimization objectives configuration."""

    objective: str = Field(default="roas", description="Primary optimization objective")
    constraints: List[str] = Field(
        default_factory=list, description="Additional optimization constraints"
    )


class RulesConfig(BaseModel):
    """Complete rules configuration."""

    rolling_windows: List[int] = Field(
        default_factory=lambda: [7, 14], description="Rolling window sizes"
    )
    safety_rules: SafetyRulesConfig = Field(
        default_factory=SafetyRulesConfig, description="Safety rules"
    )
    decision_rules: DecisionRulesConfig = Field(
        default_factory=DecisionRulesConfig, description="Decision rules"
    )
    advanced_concepts: AdvancedConceptsConfig = Field(
        default_factory=AdvancedConceptsConfig, description="Advanced concepts"
    )
    objectives: ObjectivesConfig = Field(
        default_factory=ObjectivesConfig, description="Optimization objectives"
    )
    monthly_budget: MonthlyBudgetConfig = Field(
        default_factory=MonthlyBudgetConfig, description="Monthly budget configuration"
    )


class PathsConfig(BaseModel):
    """Path configuration for data directories."""

    base_dir: str = Field(default=".", description="Base directory for all paths")
    data_dir: str = Field(default="datasets", description="Data directory name")
    results_dir: str = Field(default="results", description="Results directory name")
    logs_dir: str = Field(default="logs", description="Logs directory name")
    cache_dir: str = Field(default="cache", description="Cache directory name")
    config_dir: str = Field(default="config", description="Config directory name")


class SystemConfig(BaseModel):
    """Complete system configuration."""

    environment: str = Field(
        default="production",
        description="Environment name (development, staging, production)",
    )
    test_mode: bool = Field(default=False, description="Enable test mode")
    customer: str = Field(default="moprobo", description="Customer name")
    platform: str = Field(default="meta", description="Platform name")
    paths: PathsConfig = Field(
        default_factory=PathsConfig, description="Path configuration"
    )
    rules: RulesConfig = Field(
        default_factory=RulesConfig, description="Rules configuration"
    )
    verbose: bool = Field(default=False, description="Enable verbose logging")
