"""Path resolution utilities.

This module provides helper functions for resolving repository root,
data directories, and feature CSV files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence


def find_repo_root(start: Path | None = None) -> Path:
    """
    Best-effort repo root discovery.

    Strategy: walk up from cwd and look for common project markers.
    """
    if start is None:
        start = Path.cwd()
    start = start.expanduser().resolve()

    markers: Iterable[str] = ("pyproject.toml", "requirements.txt", ".git")
    for current_path in [start, *start.parents]:
        if any((current_path / marker).exists() for marker in markers):
            return current_path
    return start


def get_data_dir(repo_root: Path | None = None) -> Path:
    """
    Data directory resolution.

    - Defaults to <repo>/datasets
    - Can be overridden by CREATIVE_SCORER_DATA_DIR
    """
    if repo_root is None:
        repo_root = find_repo_root()
    override = os.environ.get("CREATIVE_SCORER_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (repo_root / "datasets").resolve()


def resolve_features_csv(
    repo_root: Path | None = None,
    filename: str = "creative_image_features_optimized_clean.csv",
    *,
    extra_candidates: Sequence[Path] = (),
) -> Path:
    """
    Resolve the creative features CSV path.

    Priority:
    1) CREATIVE_SCORER_FEATURES_CSV (absolute or relative path)
    2) common locations under data/
    3) extra_candidates

    Raises FileNotFoundError with actionable instructions if not found.
    """
    if repo_root is None:
        repo_root = find_repo_root()
    repo_root = repo_root.expanduser().resolve()

    override = os.environ.get("CREATIVE_SCORER_FEATURES_CSV")
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            override_path = (repo_root / override_path).resolve()
        override_path = override_path.resolve()
        if override_path.exists():
            return override_path
        raise FileNotFoundError(
            "CREATIVE_SCORER_FEATURES_CSV is set but the file does not exist:\n"
            f"  {override_path}\n"
            "Fix: point CREATIVE_SCORER_FEATURES_CSV to an existing CSV file."
        )

    data_dir = get_data_dir(repo_root)
    candidates: list[Path] = [
        data_dir / filename,
        data_dir / "moprobo" / filename,
        data_dir / "moprobo" / "results" / filename,
        data_dir / "results" / filename,
    ]
    candidates.extend(
        [candidate.expanduser().resolve() for candidate in extra_candidates]
    )

    for candidate_path in candidates:
        if candidate_path.exists():
            return candidate_path

    searched = "\n".join(
        f"  - {candidate_path}" for candidate_path in candidates
    )
    raise FileNotFoundError(
        f"Creative features CSV not found (expected '{filename}').\n"
        "Searched:\n"
        f"{searched}\n\n"
        "How to fix:\n"
        "  - Put the CSV under <repo>/datasets/ (default), OR\n"
        "  - Set CREATIVE_SCORER_DATA_DIR to point to your data folder, OR\n"
        "  - Set CREATIVE_SCORER_FEATURES_CSV to the full path of the CSV.\n"
        "\n"
        "Tip: this file is typically produced by running the offline pipeline, "
        "or exported from previous experiments."
    )


def resolve_and_validate_input_csv(
    input_csv: Path | str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    Resolve and validate input CSV file path.

    This function resolves relative paths to absolute paths and validates
    that the file exists. It logs the result and returns None if the file
    is not found.

    Args:
        input_csv: Input CSV path (can be relative or absolute)
        logger: Logger instance for logging (if None, creates one)

    Returns:
        Resolved Path object if file exists, None otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    input_csv = Path(input_csv)
    if not input_csv.is_absolute():
        repo_root = find_repo_root()
        if str(input_csv).startswith("datasets/"):
            input_csv = repo_root / input_csv
        else:
            data_dir = get_data_dir(repo_root)
            input_csv = data_dir / input_csv

    logger.info("Input file: %s", input_csv)

    if not input_csv.exists():
        logger.error("Input file not found: %s", input_csv)
        return None

    return input_csv


# === AD MINER PATH MANAGEMENT ===


class MinerPaths:
    """
    Centralized path management for ad miner.

    Hierarchical structure based on granularity levels:
    - Level 1: config/ad/miner/mined_patterns/{customer}/{product}/{branch}/{goal}/
    - Level 2: config/ad/miner/mined_patterns/{customer}/{product}/{goal}/
    - Level 3: config/ad/miner/mined_patterns/{customer}/{goal}/
    - Level 4: config/ad/miner/mined_patterns/{customer}/
    """

    def __init__(
        self,
        customer: str,
        product: str = "all",
        branch: str = "all",
        goal: str = "all",
        granularity_level: int = 1,
        base_dir: Optional[Path | str] = None,
    ):
        """
        Initialize path manager.

        Args:
            customer: Customer/account name
            product: Product name (default: "all")
            branch: Branch/region (default: "all")
            goal: Campaign goal (default: "all")
            granularity_level: Granularity level 1-4 (default: 1)
            base_dir: Base directory (default: config/ad/miner/)
        """
        self.customer = customer
        self.product = product
        self.branch = branch
        self.goal = goal
        self.granularity_level = granularity_level

        if base_dir is None:
            # Default to config/ad/miner/
            self.base_dir = Path("config") / "ad" / "miner"
        else:
            self.base_dir = Path(base_dir)

    # === ROOT DIRECTORIES ===

    @property
    def mined_patterns_dir(self) -> Path:
        """Get root mined patterns directory."""
        return self.base_dir / "mined_patterns"

    @property
    def input_dir(self) -> Path:
        """Get input data directory."""
        return self.base_dir / "input"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory."""
        return self.base_dir / "cache"

    @property
    def results_dir(self) -> Path:
        """Get analysis results directory."""
        return self.base_dir / "results"

    @property
    def schemas_dir(self) -> Path:
        """Get schemas directory."""
        return self.base_dir / "schemas"

    # === SEGMENT-SPECIFIC PATHS ===

    def segment_patterns_dir(self) -> Path:
        """
        Get patterns directory for current segment.

        Constructs path based on granularity_level:
        - Level 1: {customer}/{product}/{branch}/{goal}/
        - Level 2: {customer}/{product}/{goal}/
        - Level 3: {customer}/{goal}/
        - Level 4: {customer}/

        Returns:
            Path to segment patterns directory
        """
        base = self.mined_patterns_dir / self.customer

        if self.granularity_level == 1:
            return base / self.product / self.branch / self.goal
        elif self.granularity_level == 2:
            return base / self.product / self.goal
        elif self.granularity_level == 3:
            return base / self.goal
        else:  # Level 4
            return base

    def patterns_json(self) -> Path:
        """
        Get path to patterns JSON file.

        Returns:
            Path to patterns.json
        """
        return self.segment_patterns_dir() / "patterns.json"

    def patterns_md(self) -> Path:
        """
        Get path to patterns Markdown file.

        Returns:
            Path to patterns.md
        """
        return self.segment_patterns_dir() / "patterns.md"

    def segment_index(self) -> Path:
        """
        Get path to segment index JSON.

        Returns:
            Path to index.json
        """
        return self.segment_patterns_dir() / "index.json"

    # === CACHE PATHS ===

    def extracted_features_cache(self) -> Path:
        """
        Get path to extracted features cache.

        Returns:
            Path to features_cache.pkl
        """
        return self.cache_dir / f"{self.customer}_features_cache.pkl"

    def checkpoint_path(self, checkpoint_name: str) -> Path:
        """
        Get path to checkpoint file.

        Args:
            checkpoint_name: Name of checkpoint

        Returns:
            Path to checkpoint file
        """
        return self.cache_dir / f"{self.customer}_{checkpoint_name}.ckpt"

    # === INPUT PATHS ===

    def input_csv(self, filename: str = "creative_features.csv") -> Path:
        """
        Get path to input CSV file.

        Args:
            filename: CSV filename (default: creative_features.csv)

        Returns:
            Path to input CSV
        """
        return self.input_dir / self.customer / filename

    def validation_report(self, filename: str = "validation_report.json") -> Path:
        """
        Get path to validation report.

        Args:
            filename: Report filename

        Returns:
            Path to validation report
        """
        return self.input_dir / self.customer / filename

    # === ANALYSIS RESULTS PATHS ===

    def analysis_results_dir(self) -> Path:
        """
        Get directory for analysis results.

        Returns:
            Path to analysis results directory
        """
        return self.results_dir / self.customer

    def statistical_test_results(self) -> Path:
        """
        Get path to statistical test results.

        Returns:
            Path to statistical_tests.json
        """
        return self.analysis_results_dir() / "statistical_tests.json"

    def feature_importance_plot(self) -> Path:
        """
        Get path to feature importance plot.

        Returns:
            Path to feature_importance.png
        """
        return self.analysis_results_dir() / "feature_importance.png"

    def quartile_comparison_plot(self) -> Path:
        """
        Get path to quartile comparison plot.

        Returns:
            Path to quartile_comparison.png
        """
        return self.analysis_results_dir() / "quartile_comparison.png"

    # === UTILITY METHODS ===

    def ensure_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.segment_patterns_dir(),
            self.input_dir / self.customer,
            self.cache_dir,
            self.analysis_results_dir(),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_segment_key(self) -> str:
        """
        Get unique segment key for caching/indexing.

        Returns:
            String key representing this segment
        """
        if self.granularity_level == 1:
            return f"{self.customer}:{self.product}:{self.branch}:{self.goal}"
        elif self.granularity_level == 2:
            return f"{self.customer}:{self.product}:{self.goal}"
        elif self.granularity_level == 3:
            return f"{self.customer}:{self.goal}"
        else:  # Level 4
            return self.customer

    def __repr__(self) -> str:
        """String representation of paths."""
        return (
            f"MinerPaths(customer={self.customer}, product={self.product}, "
            f"branch={self.branch}, goal={self.goal}, level={self.granularity_level})"
        )


# === AD MINER UTILITY FUNCTIONS ===


def get_default_paths(customer: str) -> MinerPaths:
    """
    Get default paths for a customer.

    Args:
        customer: Customer name

    Returns:
        MinerPaths instance with default settings
    """
    return MinerPaths(
        customer=customer,
        product="all",
        branch="all",
        goal="all",
        granularity_level=1,
    )


def get_segment_paths(
    customer: str,
    product: str,
    branch: str,
    goal: str,
) -> MinerPaths:
    """
    Get paths for a specific segment.

    Args:
        customer: Customer name
        product: Product name
        branch: Branch/region
        goal: Campaign goal

    Returns:
        MinerPaths instance for segment
    """
    return MinerPaths(
        customer=customer,
        product=product,
        branch=branch,
        goal=goal,
        granularity_level=1,
    )
