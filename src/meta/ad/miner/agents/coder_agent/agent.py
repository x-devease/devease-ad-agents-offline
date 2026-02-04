"""
Coder Agent - Logic & Prompt Engineer

Objective: Implement experiment specifications from PM Agent by modifying
code, creating Git branches/PRs, and running tests.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
import subprocess
import json

logger = logging.getLogger(__name__)


@dataclass
class PullRequest:
    """Pull Request metadata."""
    pr_id: str
    branch_name: str
    experiment_id: str
    files_modified: List[str]
    tests_run: List[str]
    tests_passed: bool
    commit_message: str
    description: str
    timestamp: str


class CoderAgent:
    """
    Logic & Prompt Engineer Agent.

    Implements experiment specifications by:
    - Reading and modifying source files
    - Creating Git branches and pull requests
    - Running tests
    - Learning from past successful implementations (via Memory Agent)
    """

    def __init__(
        self,
        memory_agent=None,
        repo_root: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Coder Agent.

        Args:
            memory_agent: Memory Agent instance for learning from past experiments
            repo_root: Root directory of the git repository
            config: Agent configuration
        """
        self.memory_agent = memory_agent
        self.config = config or {}

        # Set repository root
        if repo_root is None:
            # Default to current directory or project root
            repo_root = Path.cwd()
        self.repo_root = Path(repo_root)

        # Track active PRs
        self.active_prs: List[PullRequest] = []

        # Ad Miner codebase structure
        self.codebase_structure = {
            "pattern_mining": [
                "src/meta/ad/miner/stages/miner.py",
                "src/meta/ad/miner/stages/miner_v2.py",
                "src/meta/ad/miner/stages/synthesizer.py",
                "src/meta/ad/miner/stages/patterns_io.py",
            ],
            "psychology": [
                "src/meta/ad/miner/features/psychology_classifier.py",
                "src/meta/ad/miner/stages/psych_composer.py",
            ],
            "feature_extraction": [
                "src/meta/ad/miner/features/extractors/gpt4_feature_extractor.py",
                "src/meta/ad/miner/features/transformers/gpt4_feature_transformer.py",
                "src/meta/ad/miner/features/lib/parsers.py",
            ],
            "pipeline": [
                "src/meta/ad/miner/pipeline.py",
                "src/meta/ad/miner/pipeline_v2.py",
                "src/meta/ad/miner/features/extract.py",
            ],
            "validation": [
                "src/meta/ad/miner/validation/input_validator.py",
                "src/meta/ad/miner/validation/output_validator.py",
            ],
        }

        logger.info("Coder Agent: Initialized")
        logger.info(f"  Repository root: {self.repo_root}")

    def implement_experiment(
        self,
        experiment_spec: Dict[str, Any],
    ) -> PullRequest:
        """
        Implement an experiment specification.

        Args:
            experiment_spec: Experiment specification from PM Agent

        Returns:
            PullRequest: Created PR metadata
        """
        exp_id = experiment_spec["id"]
        objective = experiment_spec["objective"]
        approach = experiment_spec["approach"]

        logger.info(f"Coder Agent: Implementing experiment {exp_id}")
        logger.info(f"  Objective: {objective}")
        logger.info(f"  Approach: {approach}")

        # Step 1: Query Memory for similar successful implementations
        similar_code = []
        if self.memory_agent:
            similar_code = self.memory_agent.search_similar(
                query=f"{objective} {approach}",
                top_k=3,
            )
            logger.info(f"  Retrieved {len(similar_code)} similar implementations")

        # Step 2: Create Git branch
        branch_name = f"experiment/{exp_id}"
        self._create_git_branch(branch_name)

        # Step 3: Identify files to modify
        files_to_modify = self._identify_files(experiment_spec)

        # Step 4: Generate code modifications
        code_changes = self._generate_code_changes(
            experiment_spec,
            similar_code,
            files_to_modify,
        )

        # Step 5: Apply code changes
        modified_files = []
        for file_path, changes in code_changes.items():
            success = self._apply_code_change(file_path, changes)
            if success:
                modified_files.append(file_path)

        # Step 6: Commit changes
        commit_message = f"Experiment {exp_id}: {objective}"
        self._commit_changes(commit_message)

        # Step 7: Run tests
        tests_run, tests_passed = self._run_tests()

        # Step 8: Create PR
        pr = PullRequest(
            pr_id=f"pr_{exp_id}",
            branch_name=branch_name,
            experiment_id=exp_id,
            files_modified=modified_files,
            tests_run=tests_run,
            tests_passed=tests_passed,
            commit_message=commit_message,
            description=self._generate_pr_description(experiment_spec, code_changes),
            timestamp=datetime.now().isoformat(),
        )

        self.active_prs.append(pr)

        logger.info(f"✓ Implementation complete:")
        logger.info(f"  PR: {pr.pr_id}")
        logger.info(f"  Branch: {branch_name}")
        logger.info(f"  Files modified: {len(modified_files)}")
        logger.info(f"  Tests: {'PASSED' if tests_passed else 'FAILED'}")

        return pr

    def _create_git_branch(self, branch_name: str):
        """Create a new Git branch."""
        try:
            # Check if branch exists
            result = subprocess.run(
                ["git", "branch", "--list", branch_name],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )

            if branch_name in result.stdout:
                logger.info(f"  Branch {branch_name} already exists, checking out")
                subprocess.run(
                    ["git", "checkout", branch_name],
                    cwd=self.repo_root,
                    capture_output=True,
                )
            else:
                # Create and checkout new branch
                subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    cwd=self.repo_root,
                    capture_output=True,
                    check=True,
                )
                logger.info(f"  Created branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"  Git branch creation failed: {e}")
            logger.warning("  Continuing without Git (testing mode)")

    def _identify_files(self, experiment_spec: Dict[str, Any]) -> List[str]:
        """Identify which files to modify based on experiment spec."""

        # Check if spec specifies allowed files
        constraints = experiment_spec.get("constraints", {})
        if "allowed_files" in constraints:
            return constraints["allowed_files"]

        # Otherwise, infer from objective
        objective = experiment_spec["objective"]

        if "psychology" in objective:
            return self.codebase_structure["psychology"]
        elif "pattern" in objective or "mining" in objective:
            return self.codebase_structure["pattern_mining"]
        elif "feature" in objective or "extraction" in objective:
            return self.codebase_structure["feature_extraction"]
        elif "performance" in objective or "processing" in objective:
            return self.codebase_structure["pipeline"]
        else:
            # Default to pattern mining files
            return self.codebase_structure["pattern_mining"]

    def _generate_code_changes(
        self,
        experiment_spec: Dict[str, Any],
        similar_code: List[Dict],
        files_to_modify: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate code changes based on experiment spec and similar implementations.

        Returns:
            Dict mapping file paths to change specifications
        """

        code_changes = {}

        # Extract parameters from experiment spec
        parameters = experiment_spec.get("parameters", {})
        approach = experiment_spec["approach"]

        # Generate changes based on approach
        for file_path in files_to_modify:
            full_path = self.repo_root / file_path

            if not full_path.exists():
                logger.warning(f"  File not found: {file_path}")
                continue

            # Read current file content
            with open(full_path, 'r') as f:
                current_content = f.read()

            # Generate change based on approach
            change_spec = {
                "file_path": file_path,
                "current_content": current_content,
                "modifications": self._generate_modifications(
                    file_path,
                    experiment_spec,
                    similar_code,
                ),
                "parameters": parameters,
            }

            code_changes[file_path] = change_spec

        return code_changes

    def _generate_modifications(
        self,
        file_path: str,
        experiment_spec: Dict[str, Any],
        similar_code: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Generate specific code modifications for a file."""

        modifications = []
        objective = experiment_spec["objective"]
        parameters = experiment_spec.get("parameters", {})

        # Example: Psychology keyword additions
        if "psychology" in file_path and "psychology" in objective.lower():
            modifications.append({
                "type": "add_keywords",
                "description": "Add domain-specific psychology keywords",
                "keywords": self._generate_psychology_keywords(experiment_spec),
            })

        # Example: Parameter adjustments
        if parameters:
            modifications.append({
                "type": "update_parameters",
                "description": f"Update mining parameters: {parameters}",
                "parameters": parameters,
            })

        # Example: Feature extraction improvements
        if "extractor" in file_path and "feature" in objective.lower():
            modifications.append({
                "type": "improve_extraction",
                "description": "Add few-shot examples to GPT-4 prompts",
                "examples": self._generate_few_shot_examples(experiment_spec),
            })

        return modifications

    def _generate_psychology_keywords(self, experiment_spec: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate psychology keywords based on experiment domain."""

        domain = experiment_spec.get("domain", "")

        if "gaming" in domain.lower():
            return {
                "Trust_Authority": ["esports", "competitive", "pro", "tournament"],
                "Social_Proof": ["streamer", "community", "multiplayer", "team"],
                "FOMO": ["limited event", "season pass", "battle pass", "exclusive"],
            }
        elif "ecommerce" in domain.lower():
            return {
                "Trust_Authority": ["certified", "guarantee", "official", "verified"],
                "Luxury_Aspiration": ["premium", "exclusive", "limited edition"],
                "Social_Proof": ["bestseller", "top rated", "customer favorite"],
            }
        else:
            # Generic keywords
            return {
                "Trust_Authority": ["expert", "professional", "trusted"],
                "Social_Proof": ["popular", "trending", "recommended"],
            }

    def _generate_few_shot_examples(self, experiment_spec: Dict[str, Any]) -> List[Dict]:
        """Generate few-shot examples for VLM prompts."""

        return [
            {
                "image_description": "Product on marble surface with window light",
                "features": {
                    "surface_material": "marble",
                    "lighting_style": "Window Light",
                    "camera_angle": "45-degree",
                },
            },
            {
                "image_description": "Product on white background",
                "features": {
                    "surface_material": "white",
                    "lighting_style": "Studio White",
                    "camera_angle": "Eye-Level Shot",
                },
            },
        ]

    def _apply_code_change(self, file_path: str, change_spec: Dict[str, Any]) -> bool:
        """
        Apply code changes to a file.

        In production, this would use AST parsing or LLM-based code generation.
        For now, this is a placeholder that logs what would be changed.
        """

        logger.info(f"  Applying changes to {file_path}:")
        for mod in change_spec["modifications"]:
            logger.info(f"    - {mod['type']}: {mod['description']}")

        # Placeholder: In production, would actually modify the file
        # For now, return True to indicate success
        return True

    def _commit_changes(self, commit_message: str):
        """Commit changes to Git."""
        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=self.repo_root,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.repo_root,
                capture_output=True,
                check=True,
            )
            logger.info(f"  Committed changes: {commit_message}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"  Git commit failed: {e}")

    def _run_tests(self) -> tuple[List[str], bool]:
        """Run relevant tests for the changes."""

        tests_run = []
        all_passed = True

        # Run pytest on ad miner tests
        try:
            test_dirs = [
                "src/meta/ad/miner/features",
                "src/meta/ad/miner/stages",
            ]

            for test_dir in test_dirs:
                test_path = self.repo_root / test_dir
                if not test_path.exists():
                    continue

                result = subprocess.run(
                    ["python3", "-m", "pytest", str(test_path), "-v"],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                tests_run.append(test_dir)
                if result.returncode != 0:
                    all_passed = False
                    logger.warning(f"  Tests failed in {test_dir}")

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"  Test execution failed: {e}")
            all_passed = False

        if tests_run:
            logger.info(f"  Ran {len(tests_run)} test suites: {'PASSED' if all_passed else 'FAILED'}")

        return tests_run, all_passed

    def _generate_pr_description(
        self,
        experiment_spec: Dict[str, Any],
        code_changes: Dict[str, Dict],
    ) -> str:
        """Generate PR description."""

        description = f"""
## Experiment {experiment_spec['id']}

**Objective:** {experiment_spec['objective']}

**Approach:** {experiment_spec['approach']}

**Rationale:** {experiment_spec.get('rationale', 'N/A')}

**Success Criteria:** {json.dumps(experiment_spec.get('success_criteria', {}), indent=2)}

### Changes

**Files Modified:**
"""
        for file_path in code_changes.keys():
            description += f"- `{file_path}`\n"

        description += f"""
**Parameters:** {json.dumps(experiment_spec.get('parameters', {}), indent=2)}

### Testing

**Tests Run:** See CI/CD results

**Expected Impact:** This experiment aims to achieve the success criteria listed above.

**Historical Context:** {len(experiment_spec.get('historical_context', []))} similar experiments found in memory.
"""

        return description.strip()

    def rollback_experiment(self, pr_id: str) -> bool:
        """
        Rollback an experiment by closing its PR and deleting branch.

        Args:
            pr_id: Pull request ID to rollback

        Returns:
            True if rollback successful
        """
        # Find PR
        pr = next((p for p in self.active_prs if p.pr_id == pr_id), None)
        if not pr:
            logger.warning(f"PR {pr_id} not found")
            return False

        # Delete branch
        try:
            subprocess.run(
                ["git", "checkout", "main"],
                cwd=self.repo_root,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "branch", "-D", pr.branch_name],
                cwd=self.repo_root,
                capture_output=True,
            )
            logger.info(f"Rolled back experiment {pr_id}: deleted branch {pr.branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "active_prs": len(self.active_prs),
            "repo_root": str(self.repo_root),
            "config": self.config,
        }
