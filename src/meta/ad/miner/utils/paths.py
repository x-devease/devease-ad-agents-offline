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
