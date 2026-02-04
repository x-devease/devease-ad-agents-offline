"""
PM Agent - Product Manager for Ad Generator Development Team.

Responsible for:
- Translating Judge Agent findings into experiment specs
- Setting logical boundaries for changes (Prompt vs Core vs Config)
- Maximizing experiment ROI while controlling evolution risk
- Retrieving historical context from Memory Agent

Author: Ad System Dev Team
Date: 2026-02-04
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ChangeScope(Enum):
    """Scope of allowed changes for an experiment."""

    PROMPT_ONLY = "prompt_only"  # Only modify prompt templates and configurations
    LOGIC_ONLY = "logic_only"  # Only modify business logic (not prompts)
    CONFIG_ONLY = "config_only"  # Only modify configuration parameters
    FEATURE_ENGINEERING = "feature_engineering"  # Modify feature extraction/logic
    FULL_STACK = "full_stack"  # Any change is allowed


class ExperimentPriority(Enum):
    """Priority level for experiments."""

    CRITICAL = "critical"  # Production issues, hotfixes
    HIGH = "high"  # Important improvements
    MEDIUM = "medium"  # Normal improvements
    LOW = "low"  # Exploratory changes
    RESEARCH = "research"  # Pure research without immediate impact


class Component(Enum):
    """Components in the ad/generator system."""

    AD_MINER = "ad_miner"  # Creative recommendation engine
    AD_GENERATOR = "ad_generator"  # Creative image generation
    ADSET_ALLOCATOR = "adset_allocator"  # Budget allocation engine
    ADSET_GENERATOR = "adset_generator"  # Audience configuration engine
    NANO_BANANA_PRO = "nano_banana_pro"  # Prompt enhancement agent
    SHARED_UTILS = "shared_utils"  # Shared utilities
    FRAMEWORK = "framework"  # Agent framework


@dataclass
class ExperimentConstraint:
    """Constraints for an experiment."""

    max_files_changed: int = 5
    max_lines_added: int = 200
    max_lines_removed: int = 100
    forbidden_modules: List[str] = field(default_factory=list)
    required_tests: List[str] = field(default_factory=list)
    allowed_file_patterns: List[str] = field(default_factory=list)
    blocked_file_patterns: List[str] = field(default_factory=list)

    def allows_change(self, file_path: str, lines_added: int, lines_removed: int) -> bool:
        """Check if a change is allowed under these constraints."""
        # Check file patterns
        import re

        if self.blocked_file_patterns:
            for pattern in self.blocked_file_patterns:
                if re.search(pattern, file_path):
                    return False

        if self.allowed_file_patterns:
            allowed = any(re.search(pattern, file_path) for pattern in self.allowed_file_patterns)
            if not allowed:
                return False

        # Check line limits
        if lines_added > self.max_lines_added or lines_removed > self.max_lines_removed:
            return False

        return True


@dataclass
class ExperimentSpec:
    """Experiment specification created by PM Agent."""

    spec_id: str
    title: str
    description: str

    # Target component
    component: Component
    affected_modules: List[str]

    # What to fix/improve
    problem_statement: str
    success_criteria: List[str]
    failure_conditions: List[str]

    # Constraints
    scope: ChangeScope
    constraints: ExperimentConstraint
    priority: ExperimentPriority

    # Context from memory
    related_historical_experiments: List[str] = field(default_factory=list)
    similar_past_failures: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "pm_agent"
    status: str = "pending"  # pending, in_progress, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spec_id": self.spec_id,
            "title": self.title,
            "description": self.description,
            "component": self.component.value,
            "affected_modules": self.affected_modules,
            "problem_statement": self.problem_statement,
            "success_criteria": self.success_criteria,
            "failure_conditions": self.failure_conditions,
            "scope": self.scope.value,
            "constraints": {
                "max_files_changed": self.constraints.max_files_changed,
                "max_lines_added": self.constraints.max_lines_added,
                "max_lines_removed": self.constraints.max_lines_removed,
                "forbidden_modules": self.constraints.forbidden_modules,
                "required_tests": self.constraints.required_tests,
            },
            "priority": self.priority.value,
            "related_historical_experiments": self.related_historical_experiments,
            "similar_past_failures": self.similar_past_failures,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "status": self.status,
        }


@dataclass
class JudgeFindings:
    """Findings from Judge Agent about issues to address."""

    issue_type: str  # performance_drop, bug, feature_request, optimization
    component: Component
    severity: str  # critical, high, medium, low
    description: str
    evidence: Dict[str, Any]
    suggested_priority: ExperimentPriority

    # Metrics
    current_metrics: Optional[Dict[str, float]] = None
    target_metrics: Optional[Dict[str, float]] = None
    impact_estimate: Optional[str] = None


class PMAgent:
    """
    Product Manager Agent for ad/generator development.

    Responsibilities:
    1. Receive findings from Judge Agent
    2. Query Memory Agent for historical context
    3. Create experiment specs with appropriate constraints
    4. Prioritize experiments based on ROI and risk
    5. Set logical boundaries for changes
    """

    def __init__(
        self,
        memory_client: Optional[Any] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize PM Agent.

        Args:
            memory_client: Client for querying Memory Agent
            config_path: Path to PM Agent configuration
        """
        self.memory_client = memory_client
        self.config_path = config_path

        # Load configuration
        self._load_config()

        logger.info("PMAgent initialized for ad/generator development")

    def _load_config(self):
        """Load PM Agent configuration."""
        # Default risk tolerances by component
        self.risk_tolerances = {
            Component.AD_MINER: {
                "max_logic_changes": 3,
                "requires_backtest": True,
                "requires_golden_set": True,
            },
            Component.AD_GENERATOR: {
                "max_prompt_changes": 10,
                "requires_visual_review": True,
                "requires_reference_images": True,
            },
            Component.ADSET_ALLOCATOR: {
                "max_budget_changes": 2,
                "requires_simulation": True,
                "requires_safety_checks": True,
            },
            Component.ADSET_GENERATOR: {
                "max_audience_changes": 5,
                "requires_historical_validation": True,
            },
            Component.NANO_BANANA_PRO: {
                "max_technique_changes": 3,
                "requires_quality_verification": True,
            },
            Component.SHARED_UTILS: {
                "max_changes": 2,
                "requires_all_tests": True,
                "affects_all_components": True,
            },
            Component.FRAMEWORK: {
                "max_changes": 1,
                "requires_all_tests": True,
                "requires_migration_guide": True,
                "affects_all_agents": True,
            },
        }

        # Default constraints by scope
        self.scope_constraints = {
            ChangeScope.PROMPT_ONLY: ExperimentConstraint(
                max_files_changed=10,
                max_lines_added=100,
                max_lines_removed=50,
                allowed_file_patterns=[
                    r".*prompts?.*\.py$",
                    r".*prompt.*\.yaml$",
                    r".*templates?\.yaml$",
                    r".*text_templates\.yaml$",
                ],
                forbidden_modules=[
                    "src/meta/ad/generator/core",
                    "src/meta/adset/allocator/lib",
                    "src/utils",
                ],
            ),
            ChangeScope.LOGIC_ONLY: ExperimentConstraint(
                max_files_changed=5,
                max_lines_added=150,
                max_lines_removed=100,
                blocked_file_patterns=[
                    r".*prompts?.*\.py$",
                    r".*\.yaml$",
                ],
            ),
            ChangeScope.CONFIG_ONLY: ExperimentConstraint(
                max_files_changed=3,
                max_lines_added=50,
                max_lines_removed=50,
                allowed_file_patterns=[
                    r".*\.yaml$",
                    r".*\.json$",
                ],
                blocked_file_patterns=[
                    r".*\.py$",
                ],
            ),
            ChangeScope.FEATURE_ENGINEERING: ExperimentConstraint(
                max_files_changed=5,
                max_lines_added=200,
                max_lines_removed=100,
                allowed_file_patterns=[
                    r".*features?/.*\.py$",
                    r".*extractors?/.*\.py$",
                ],
                required_tests=[
                    "test_feature_extraction",
                    "test_backtest",
                ],
            ),
            ChangeScope.FULL_STACK: ExperimentConstraint(
                max_files_changed=10,
                max_lines_added=500,
                max_lines_removed=200,
                required_tests=[
                    "test_unit",
                    "test_integration",
                ],
            ),
        }

    def create_experiment_spec(
        self,
        findings: JudgeFindings,
        historical_context: Optional[Dict[str, Any]] = None,
    ) -> ExperimentSpec:
        """
        Create an experiment spec from Judge Agent findings.

        Args:
            findings: Issues identified by Judge Agent
            historical_context: Optional context from Memory Agent

        Returns:
            Experiment specification with constraints
        """
        # Generate spec ID
        spec_id = self._generate_spec_id(findings)

        # Determine appropriate scope
        scope = self._determine_scope(findings, historical_context)

        # Get constraints for scope
        constraints = self.scope_constraints.get(
            scope, self.scope_constraints[ChangeScope.FULL_STACK]
        )

        # Get risk tolerance for component
        risk_tolerance = self.risk_tolerances.get(findings.component, {})

        # Adjust constraints based on risk tolerance
        constraints = self._adjust_constraints_for_risk(constraints, risk_tolerance)

        # Build title and description
        title = self._build_title(findings)
        description = self._build_description(findings, historical_context)

        # Determine success criteria
        success_criteria = self._build_success_criteria(findings)

        # Determine failure conditions
        failure_conditions = self._build_failure_conditions(findings, historical_context)

        # Create spec
        spec = ExperimentSpec(
            spec_id=spec_id,
            title=title,
            description=description,
            component=findings.component,
            affected_modules=self._identify_affected_modules(findings),
            problem_statement=findings.description,
            success_criteria=success_criteria,
            failure_conditions=failure_conditions,
            scope=scope,
            constraints=constraints,
            priority=findings.suggested_priority,
            related_historical_experiments=(
                historical_context.get("related_experiments", [])
                if historical_context
                else []
            ),
            similar_past_failures=(
                historical_context.get("similar_failures", [])
                if historical_context
                else []
            ),
        )

        logger.info(f"Created experiment spec: {spec_id} - {title}")
        return spec

    def _generate_spec_id(self, findings: JudgeFindings) -> str:
        """Generate unique spec ID from findings."""
        import hashlib

        content = f"{findings.component.value}:{findings.issue_type}:{datetime.now().isoformat()}"
        return f"spec-{hashlib.md5(content.encode()).hexdigest()[:8]}"

    def _determine_scope(
        self,
        findings: JudgeFindings,
        historical_context: Optional[Dict[str, Any]],
    ) -> ChangeScope:
        """
        Determine appropriate change scope based on findings and history.

        More conservative if similar changes failed in the past.
        """
        # Check for similar past failures
        if historical_context and historical_context.get("similar_failures"):
            # Be conservative - restrict scope
            if findings.component == Component.AD_MINER:
                return ChangeScope.FEATURE_ENGINEERING
            elif findings.component == Component.AD_GENERATOR:
                return ChangeScope.PROMPT_ONLY
            else:
                return ChangeScope.CONFIG_ONLY

        # Determine scope based on issue type
        if findings.issue_type == "bug":
            # Bugs might need logic fixes
            return ChangeScope.LOGIC_ONLY
        elif findings.issue_type == "performance_drop":
            # Performance drops might need tuning
            return ChangeScope.CONFIG_ONLY
        elif findings.issue_type == "feature_request":
            # New features can vary in scope
            return ChangeScope.FULL_STACK
        elif findings.issue_type == "optimization":
            # Optimizations are usually logic or config
            return ChangeScope.LOGIC_ONLY
        else:
            return ChangeScope.FULL_STACK

    def _adjust_constraints_for_risk(
        self,
        constraints: ExperimentConstraint,
        risk_tolerance: Dict[str, Any],
    ) -> ExperimentConstraint:
        """Adjust constraints based on component risk tolerance."""
        import copy

        adjusted = copy.deepcopy(constraints)

        # Reduce limits if component is high-risk
        if risk_tolerance.get("affects_all_components"):
            adjusted.max_files_changed = min(adjusted.max_files_changed, 2)
            adjusted.max_lines_added = min(adjusted.max_lines_added, 100)

        # Add required tests
        for test_type in ["requires_backtest", "requires_simulation", "requires_all_tests"]:
            if risk_tolerance.get(test_type):
                if test_type == "requires_backtest" and "test_backtest" not in adjusted.required_tests:
                    adjusted.required_tests.append("test_backtest")
                elif test_type == "requires_simulation" and "test_simulation" not in adjusted.required_tests:
                    adjusted.required_tests.append("test_simulation")
                elif test_type == "requires_all_tests":
                    adjusted.required_tests.extend(["test_unit", "test_integration"])

        return adjusted

    def _build_title(self, findings: JudgeFindings) -> str:
        """Build experiment title from findings."""
        component_name = findings.component.value.replace("_", " ").title()
        issue_type = findings.issue_type.replace("_", " ").title()
        return f"{issue_type} in {component_name}: {findings.description[:50]}..."

    def _build_description(
        self,
        findings: JudgeFindings,
        historical_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build experiment description."""
        desc = f"Address {findings.issue_type} in {findings.component.value}.\n\n"
        desc += f"**Problem:** {findings.description}\n\n"

        if findings.current_metrics:
            desc += f"**Current Metrics:**\n"
            for metric, value in findings.current_metrics.items():
                desc += f"  - {metric}: {value}\n"
            desc += "\n"

        if findings.target_metrics:
            desc += f"**Target Metrics:**\n"
            for metric, value in findings.target_metrics.items():
                desc += f"  - {metric}: {value}\n"
            desc += "\n"

        if historical_context:
            if historical_context.get("related_experiments"):
                desc += f"**Related Experiments:** {len(historical_context['related_experiments'])} found\n"
            if historical_context.get("similar_failures"):
                desc += f"**Warning:** {len(historical_context['similar_failures'])} similar past failures\n"

        return desc

    def _build_success_criteria(self, findings: JudgeFindings) -> List[str]:
        """Build success criteria for the experiment."""
        criteria = []

        if findings.target_metrics:
            for metric, target in findings.target_metrics.items():
                criteria.append(f"{metric} reaches {target}")

        # Add general criteria
        criteria.append("All existing tests pass")
        criteria.append("No regression in other components")
        criteria.append("Code follows existing patterns")

        if findings.component == Component.AD_MINER:
            criteria.append("Feature extraction accuracy maintained or improved")
        elif findings.component == Component.AD_GENERATOR:
            criteria.append("Generated images meet quality standards")
        elif findings.component == Component.ADSET_ALLOCATOR:
            criteria.append("Budget allocation passes safety checks")
        elif findings.component == Component.NANO_BANANA_PRO:
            criteria.append("Prompt quality score >= 0.8")

        return criteria

    def _build_failure_conditions(
        self,
        findings: JudgeFindings,
        historical_context: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Build failure conditions for the experiment."""
        conditions = []

        # General failure conditions
        conditions.extend([
            "Any existing test fails",
            "New bugs introduced in other components",
            "Performance degrades in other scenarios",
        ])

        # Component-specific conditions
        if findings.component == Component.AD_MINER:
            conditions.extend([
                "Feature extraction accuracy drops > 5%",
                "ROAS prediction error increases",
            ])
        elif findings.component == Component.AD_GENERATOR:
            conditions.extend([
                "Generated images fail quality guardrails",
                "Prompt hallucination detected",
            ])
        elif findings.component == Component.ADSET_ALLOCATOR:
            conditions.extend([
                "Budget exceeds monthly caps",
                "Safety rules violated",
            ])
        elif findings.component == Component.SHARED_UTILS:
            conditions.extend([
                "Breaking change to public API",
                "Multiple component tests fail",
            ])

        return conditions

    def _identify_affected_modules(self, findings: JudgeFindings) -> List[str]:
        """Identify likely affected modules based on component."""
        module_map = {
            Component.AD_MINER: [
                "src/meta/ad/miner",
                "src/meta/ad/recommender",
            ],
            Component.AD_GENERATOR: [
                "src/meta/ad/generator",
                "src/meta/ad/reviewer",
            ],
            Component.ADSET_ALLOCATOR: [
                "src/meta/adset/allocator",
            ],
            Component.ADSET_GENERATOR: [
                "src/meta/adset/generator",
            ],
            Component.NANO_BANANA_PRO: [
                "src/agents/nano",
            ],
            Component.SHARED_UTILS: [
                "src/utils",
                "src/config",
            ],
            Component.FRAMEWORK: [
                "src/agents/framework",
            ],
        }

        return module_map.get(findings.component, [])

    def prioritize_experiments(self, specs: List[ExperimentSpec]) -> List[ExperimentSpec]:
        """
        Prioritize experiments based on ROI and risk.

        Args:
            specs: List of experiment specs

        Returns:
            Sorted list of specs (highest priority first)
        """
        def priority_score(spec: ExperimentSpec) -> tuple:
            """Return tuple for sorting (higher is better)."""
            priority_order = {
                ExperimentPriority.CRITICAL: 5,
                ExperimentPriority.HIGH: 4,
                ExperimentPriority.MEDIUM: 3,
                ExperimentPriority.LOW: 2,
                ExperimentPriority.RESEARCH: 1,
            }

            # Factors:
            # 1. Priority level (higher is better)
            # 2. Number of similar failures (fewer is better - less risky)
            # 3. Scope (more restrictive is better for safety)

            scope_order = {
                ChangeScope.CONFIG_ONLY: 5,
                ChangeScope.PROMPT_ONLY: 4,
                ChangeScope.LOGIC_ONLY: 3,
                ChangeScope.FEATURE_ENGINEERING: 2,
                ChangeScope.FULL_STACK: 1,
            }

            return (
                priority_order.get(spec.priority, 0),
                -len(spec.similar_past_failures),
                scope_order.get(spec.scope, 0),
            )

        return sorted(specs, key=priority_score, reverse=True)


# Convenience function for creating PM Agent
def create_pm_agent(
    memory_client: Optional[Any] = None,
    config_path: Optional[Path] = None,
) -> PMAgent:
    """
    Create a PM Agent instance.

    Args:
        memory_client: Client for querying Memory Agent
        config_path: Path to PM Agent configuration

    Returns:
        Configured PM Agent
    """
    return PMAgent(memory_client=memory_client, config_path=config_path)
