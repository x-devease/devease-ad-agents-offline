"""
Coder Agent - Implementation Engineer for Ad Generator Development Team.

Responsible for:
- Implementing experiment specs from PM Agent
- Modifying Python/Prompt/SQL code in the codebase
- Creating Pull Requests with proper documentation
- Avoiding hardcoding and test overfitting
- Following existing code patterns and architecture

Author: Ad System Dev Team
Date: 2026-02-04
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of code change."""

    ADD = "add"  # Add new code
    MODIFY = "modify"  # Modify existing code
    DELETE = "delete"  # Delete code
    REFACTOR = "refactor"  # Refactor without changing behavior


class CodeQuality(Enum):
    """Code quality assessment."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_REVIEW = "needs_review"
    POOR = "poor"


@dataclass
class CodeChange:
    """A single code change."""

    file_path: str
    change_type: ChangeType
    lines_added: int
    lines_removed: int
    description: str
    diff: Optional[str] = None
    new_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "change_type": self.change_type.value,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "description": self.description,
        }


@dataclass
class PullRequest:
    """Pull request created by Coder Agent."""

    pr_id: str
    title: str
    description: str
    changes: List[CodeChange]
    spec_id: str  # Reference to experiment spec

    # Branch info
    branch_name: str
    base_branch: str = "main"

    # Testing
    tests_added: List[str] = field(default_factory=list)
    tests_modified: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "coder_agent"
    status: str = "draft"  # draft, ready_for_review, merged, closed

    def summary(self) -> Dict[str, Any]:
        """Get PR summary."""
        return {
            "pr_id": self.pr_id,
            "title": self.title,
            "files_changed": len(self.changes),
            "lines_added": sum(c.lines_added for c in self.changes),
            "lines_removed": sum(c.lines_removed for c in self.changes),
            "tests_added": len(self.tests_added),
            "spec_id": self.spec_id,
            "status": self.status,
        }


@dataclass
class ImplementationResult:
    """Result of implementing an experiment spec."""

    success: bool
    pull_request: Optional[PullRequest] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_assessment: Optional[CodeQuality] = None
    lint_results: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None


# Import PM Agent types
from .pm_agent import ExperimentSpec, ChangeScope, Component


class CoderAgent:
    """
    Coder Agent for ad/generator development.

    Responsibilities:
    1. Receive experiment spec from PM Agent
    2. Analyze existing code patterns
    3. Implement changes following patterns
    4. Create/update tests
    5. Run linting and tests
    6. Create Pull Request
    """

    def __init__(
        self,
        repo_path: Path,
        llm_client: Optional[Any] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize Coder Agent.

        Args:
            repo_path: Path to the repository
            llm_client: LLM client for code generation
            config_path: Path to Coder Agent configuration
        """
        self.repo_path = Path(repo_path)
        self.llm_client = llm_client
        self.config_path = config_path

        # Load configuration
        self._load_config()

        logger.info(f"CoderAgent initialized for repo: {self.repo_path}")

    def _load_config(self):
        """Load Coder Agent configuration."""
        # Code patterns to follow by component
        self.code_patterns = {
            Component.AD_MINER: {
                "imports": ["from dataclasses import dataclass", "import logging"],
                "style": "functional_with_classes",
                "test_prefix": "test_",
                "docstring_style": "google",
            },
            Component.AD_GENERATOR: {
                "imports": ["from pathlib import Path", "from typing import Optional"],
                "style": "pipeline_oriented",
                "test_prefix": "test_",
                "docstring_style": "google",
            },
            Component.ADSET_ALLOCATOR: {
                "imports": ["from dataclasses import dataclass"],
                "style": "rules_based",
                "test_prefix": "test_",
                "docstring_style": "google",
            },
            Component.NANO_BANANA_PRO: {
                "imports": ["from enum import Enum"],
                "style": "agent_framework",
                "test_prefix": "test_",
                "docstring_style": "google",
            },
        }

        # Forbidden patterns (anti-hardcoding)
        self.forbidden_patterns = {
            "test_specific_id": r"if\s+['\"]?(id|ad_id|adset_id)['\"]?\s*==\s*['\"][^'\"]+['\"]",
            "test_specific_name": r"if\s+['\"]?(name|ad_name)['\"]?\s*==\s*['\"][^'\"]+['\"]",
            "hardcoded_threshold": r"#\s*TODO:\s*remove.*hardcode|FIXME:\s*hardcode",
            "test_only_comment": r"#\s*TEST\s+ONLY|#\s*HACK\s*FOR\s+TEST",
        }

    def implement_spec(self, spec: ExperimentSpec) -> ImplementationResult:
        """
        Implement an experiment spec.

        Args:
            spec: Experiment specification from PM Agent

        Returns:
            Implementation result with PR or errors
        """
        logger.info(f"Implementing spec: {spec.spec_id} - {spec.title}")

        # Validate spec constraints
        validation_errors = self._validate_spec(spec)
        if validation_errors:
            return ImplementationResult(
                success=False,
                errors=validation_errors,
            )

        # Analyze existing code
        code_context = self._analyze_code_context(spec)

        # Generate changes
        try:
            changes = self._generate_changes(spec, code_context)

            # Validate changes against spec constraints
            if not self._validate_changes(spec, changes):
                return ImplementationResult(
                    success=False,
                    errors=["Changes violate spec constraints"],
                )

            # Check for forbidden patterns
            forbidden_found = self._check_forbidden_patterns(changes)
            if forbidden_found:
                return ImplementationResult(
                    success=False,
                    errors=[f"Forbidden pattern detected: {p}" for p in forbidden_found],
                )

            # Create PR
            pr = self._create_pull_request(spec, changes)

            # Run linting
            lint_results = self._run_linting(pr.branch_name)

            # Run tests
            test_results = self._run_tests(pr.branch_name)

            # Assess quality
            quality = self._assess_quality(lint_results, test_results)

            result = ImplementationResult(
                success=True,
                pull_request=pr,
                quality_assessment=quality,
                lint_results=lint_results,
                test_results=test_results,
            )

            logger.info(f"Successfully implemented spec: {spec.spec_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to implement spec {spec.spec_id}: {e}", exc_info=True)
            return ImplementationResult(
                success=False,
                errors=[str(e)],
            )

    def _validate_spec(self, spec: ExperimentSpec) -> List[str]:
        """Validate spec before implementation."""
        errors = []

        # Check if affected modules exist
        for module in spec.affected_modules:
            module_path = self.repo_path / module
            if not module_path.exists():
                errors.append(f"Module does not exist: {module}")

        # Check if scope is supported
        if spec.scope not in [ChangeScope.PROMPT_ONLY, ChangeScope.LOGIC_ONLY,
                              ChangeScope.CONFIG_ONLY, ChangeScope.FEATURE_ENGINEERING,
                              ChangeScope.FULL_STACK]:
            errors.append(f"Unsupported scope: {spec.scope}")

        return errors

    def _analyze_code_context(self, spec: ExperimentSpec) -> Dict[str, Any]:
        """
        Analyze existing code to understand patterns.

        Returns context about:
        - Existing imports and patterns
        - Code style conventions
        - Test patterns
        - Similar functions/classes
        """
        context = {
            "imports": set(),
            "patterns": [],
            "test_files": [],
            "similar_code": [],
        }

        for module in spec.affected_modules:
            module_path = self.repo_path / module
            if not module_path.exists():
                continue

            # Find Python files
            py_files = list(module_path.rglob("*.py"))

            for py_file in py_files:
                # Skip test files for now
                if "test" in py_file.name:
                    context["test_files"].append(str(py_file))
                    continue

                # Read file and extract patterns
                try:
                    content = py_file.read_text()

                    # Extract imports
                    for line in content.split("\n"):
                        if line.strip().startswith("import ") or line.strip().startswith("from "):
                            context["imports"].add(line.strip())

                    # Look for patterns specific to component
                    if spec.component in self.code_patterns:
                        patterns = self.code_patterns[spec.component]
                        if patterns.get("style") == "pipeline_oriented":
                            # Look for pipeline patterns
                            if "class.*Pipeline" in content or "def pipeline_" in content:
                                context["patterns"].append("pipeline")
                        elif patterns.get("style") == "rules_based":
                            # Look for rules patterns
                            if "Rule" in content or "def rule_" in content:
                                context["patterns"].append("rules")
                        elif patterns.get("style") == "agent_framework":
                            # Look for agent patterns
                            if "class.*Agent" in content:
                                context["patterns"].append("agent")

                except Exception as e:
                    logger.warning(f"Failed to analyze {py_file}: {e}")

        context["imports"] = list(context["imports"])
        return context

    def _generate_changes(
        self,
        spec: ExperimentSpec,
        code_context: Dict[str, Any],
    ) -> List[CodeChange]:
        """
        Generate code changes based on spec and context.

        This is where the actual code modification happens.
        For now, this is a placeholder that would use LLM to generate changes.
        """
        changes = []

        # Example: If spec is about prompt modification
        if spec.scope == ChangeScope.PROMPT_ONLY:
            # Find prompt files
            for module in spec.affected_modules:
                module_path = self.repo_path / module
                prompt_files = list(module_path.rglob("*.yaml")) + list(module_path.rglob("*prompt*.py"))

                for prompt_file in prompt_files:
                    # This would use LLM to generate the actual change
                    # For now, just create a placeholder
                    change = CodeChange(
                        file_path=str(prompt_file.relative_to(self.repo_path)),
                        change_type=ChangeType.MODIFY,
                        lines_added=10,  # Placeholder
                        lines_removed=5,  # Placeholder
                        description=f"Update prompt based on spec: {spec.spec_id}",
                    )
                    changes.append(change)

        elif spec.scope == ChangeScope.LOGIC_ONLY:
            # Find logic files
            for module in spec.affected_modules:
                module_path = self.repo_path / module
                logic_files = [f for f in module_path.rglob("*.py")
                              if "test" not in f.name and "prompt" not in f.name]

                for logic_file in logic_files[:3]:  # Limit to 3 files for safety
                    change = CodeChange(
                        file_path=str(logic_file.relative_to(self.repo_path)),
                        change_type=ChangeType.MODIFY,
                        lines_added=20,  # Placeholder
                        lines_removed=10,  # Placeholder
                        description=f"Update logic based on spec: {spec.spec_id}",
                    )
                    changes.append(change)

        return changes

    def _validate_changes(
        self,
        spec: ExperimentSpec,
        changes: List[CodeChange],
    ) -> bool:
        """Validate changes against spec constraints."""
        # Check file count
        if len(changes) > spec.constraints.max_files_changed:
            logger.warning(f"Too many files changed: {len(changes)} > {spec.constraints.max_files_changed}")
            return False

        # Check line counts
        total_added = sum(c.lines_added for c in changes)
        total_removed = sum(c.lines_removed for c in changes)

        if total_added > spec.constraints.max_lines_added:
            logger.warning(f"Too many lines added: {total_added} > {spec.constraints.max_lines_added}")
            return False

        if total_removed > spec.constraints.max_lines_removed:
            logger.warning(f"Too many lines removed: {total_removed} > {spec.constraints.max_lines_removed}")
            return False

        # Check forbidden modules
        for change in changes:
            for forbidden in spec.constraints.forbidden_modules:
                if change.file_path.startswith(forbidden):
                    logger.warning(f"Change in forbidden module: {change.file_path}")
                    return False

        return True

    def _check_forbidden_patterns(self, changes: List[CodeChange]) -> List[str]:
        """Check for forbidden patterns in changes."""
        forbidden_found = []

        for change in changes:
            if change.diff:
                for pattern_name, pattern in self.forbidden_patterns.items():
                    if re.search(pattern, change.diff):
                        forbidden_found.append(f"{pattern_name} in {change.file_path}")

        return forbidden_found

    def _create_pull_request(
        self,
        spec: ExperimentSpec,
        changes: List[CodeChange],
    ) -> PullRequest:
        """Create a pull request with the changes."""
        import hashlib

        # Generate PR ID
        pr_id = f"pr-{hashlib.md5(spec.spec_id.encode()).hexdigest()[:8]}"

        # Generate branch name
        branch_name = f"exp/{spec.spec_id}"

        # Build PR description
        description = f"## Experiment: {spec.title}\n\n"
        description += f"**Spec ID:** {spec.spec_id}\n"
        description += f"**Component:** {spec.component.value}\n"
        description += f"**Scope:** {spec.scope.value}\n\n"

        description += "## Changes\n\n"
        for change in changes:
            description += f"- {change.file_path}: {change.description}\n"
            description += f"  (+{change.lines_added}, -{change.lines_removed})\n"

        description += "\n## Success Criteria\n\n"
        for criterion in spec.success_criteria:
            description += f"- {criterion}\n"

        description += "\n## Testing\n\n"
        description += "- [ ] All existing tests pass\n"
        description += "- [ ] New tests added (if applicable)\n"
        description += "- [ ] No regression in other components\n"

        # Create PR
        pr = PullRequest(
            pr_id=pr_id,
            title=f"Exp: {spec.title}",
            description=description,
            changes=changes,
            spec_id=spec.spec_id,
            branch_name=branch_name,
            base_branch="main",
        )

        logger.info(f"Created PR: {pr_id} on branch {branch_name}")
        return pr

    def _run_linting(self, branch_name: str) -> Dict[str, Any]:
        """Run linting on the branch."""
        # This would actually run linting tools
        # For now, return mock results
        return {
            "pylint": {"passed": True, "score": 9.5},
            "mypy": {"passed": True, "errors": 0},
            "isort": {"passed": True, "files_checked": 10},
        }

    def _run_tests(self, branch_name: str) -> Dict[str, Any]:
        """Run tests on the branch."""
        # This would actually run pytest
        # For now, return mock results
        return {
            "unit_tests": {"passed": 45, "failed": 0, "skipped": 2},
            "integration_tests": {"passed": 12, "failed": 0, "skipped": 0},
            "coverage": {"percentage": 85.5, "target": 80},
        }

    def _assess_quality(
        self,
        lint_results: Dict[str, Any],
        test_results: Dict[str, Any],
    ) -> CodeQuality:
        """Assess overall code quality."""
        # Check linting
        lint_passed = all(r.get("passed", False) for r in lint_results.values())

        # Check tests
        total_tests = test_results["unit_tests"]["passed"] + test_results["unit_tests"]["failed"]
        test_pass_rate = test_results["unit_tests"]["passed"] / total_tests if total_tests > 0 else 0

        coverage = test_results["coverage"]["percentage"]
        coverage_target = test_results["coverage"]["target"]

        # Assess quality
        if lint_passed and test_pass_rate >= 0.95 and coverage >= coverage_target:
            return CodeQuality.EXCELLENT
        elif lint_passed and test_pass_rate >= 0.90 and coverage >= coverage_target * 0.9:
            return CodeQuality.GOOD
        elif lint_passed and test_pass_rate >= 0.80:
            return CodeQuality.ACCEPTABLE
        elif test_pass_rate >= 0.70:
            return CodeQuality.NEEDS_REVIEW
        else:
            return CodeQuality.POOR


# Convenience function for creating Coder Agent
def create_coder_agent(
    repo_path: Path,
    llm_client: Optional[Any] = None,
    config_path: Optional[Path] = None,
) -> CoderAgent:
    """
    Create a Coder Agent instance.

    Args:
        repo_path: Path to the repository
        llm_client: LLM client for code generation
        config_path: Path to Coder Agent configuration

    Returns:
        Configured Coder Agent
    """
    return CoderAgent(
        repo_path=repo_path,
        llm_client=llm_client,
        config_path=config_path,
    )
