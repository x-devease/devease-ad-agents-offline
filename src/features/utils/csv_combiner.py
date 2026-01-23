"""
CSV Combiner for Meta Ads data.

This module provides functionality to combine multiple time-period split CSV files
into single combined files, matching the moprobo data structure.

Example:
    >>> combiner = CSVCombiner(
    ...     source_dir=Path("notebooks/ecoflow/meta/raw/daily-ad-1y"),
    ...     output_dir=Path("datasets/ecoflow/meta/raw")
    ... )
    >>> combiner.process_entity_type(
    ...     entity_type="ad",
    ...     granularity="daily",
    ...     skip_validation=False,
    ...     dry_run=False
    ... )
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Constants
ENTITY_TYPE_AD = "ad"
ENTITY_TYPE_ADSET = "adset"
GRANULARITY_DAILY = "daily"
GRANULARITY_HOURLY = "hourly"

VALID_ENTITY_TYPES = {ENTITY_TYPE_AD, ENTITY_TYPE_ADSET}
VALID_GRANULARITIES = {GRANULARITY_DAILY, GRANULARITY_HOURLY}


class CSVCombinerError(Exception):
    """Base exception for CSV combiner errors."""

    pass


class CSVParseError(CSVCombinerError):
    """Exception raised when CSV parsing fails."""

    pass


class ColumnValidationError(CSVCombinerError):
    """Exception raised when column validation fails."""

    pass


class CSVCombiner:
    """Combines multiple Meta Ads CSV files into single combined files."""

    # Pattern to extract dates from filenames like:
    # ad_daily_insights_2025-01-15_2025-02-14.csv
    DATE_PATTERN = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$")

    def __init__(self, source_dir: Path, output_dir: Path):
        """Initialize with source and output directories.

        Args:
            source_dir: Directory containing split CSV files
            output_dir: Directory to write combined CSV files
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)

    def discover_files(self, entity_type: str, granularity: str) -> List[Path]:
        """Discover CSV files for a specific entity type and granularity.

        Args:
            entity_type: 'ad' or 'adset'
            granularity: 'daily' or 'hourly'

        Returns:
            Sorted list of CSV file paths
        """
        pattern = f"{entity_type}_{granularity}_insights_*.csv"
        files = sorted(self.source_dir.glob(pattern))

        if not files:
            logger.warning(
                "No files found matching pattern: %s in %s", pattern, self.source_dir
            )

        return files

    def extract_date_range(self, filepath: Path) -> Optional[Tuple[str, str]]:
        """Extract start and end dates from filename.

        Args:
            filepath: Path to CSV file

        Returns:
            Tuple of (start_date, end_date) or None if pattern doesn't match
        """
        match = self.DATE_PATTERN.search(filepath.name)
        if match:
            return match.group(1), match.group(2)
        return None

    def validate_columns(self, dataframes: List[pd.DataFrame]) -> bool:
        """Validate that all DataFrames have consistent columns.

        Args:
            dataframes: List of DataFrames to validate

        Returns:
            True if columns are consistent, False otherwise

        Raises:
            ColumnValidationError: If validation fails and exceptions are enabled
        """
        if not dataframes:
            return True

        reference_columns = set(dataframes[0].columns)

        for i, df in enumerate(dataframes[1:], start=1):
            current_columns = set(df.columns)

            if current_columns != reference_columns:
                missing = sorted(reference_columns - current_columns)
                extra = sorted(current_columns - reference_columns)

                logger.warning(
                    "Column mismatch detected in file #%d:\n"
                    "  Missing columns: %s\n"
                    "  Extra columns: %s",
                    i,
                    missing if missing else "None",
                    extra if extra else "None",
                )

                return False

        return True

    def combine_csv_files(
        self, entity_type: str, granularity: str, skip_validation: bool = False
    ) -> Optional[pd.DataFrame]:
        """Combine CSV files for a specific entity type and granularity.

        Args:
            entity_type: 'ad' or 'adset'
            granularity: 'daily' or 'hourly'
            skip_validation: If True, skip column validation

        Returns:
            Combined DataFrame sorted by date and entity IDs, or None if no files
        """
        logger.info(
            "Combining %s %s files from %s", entity_type, granularity, self.source_dir
        )

        # Discover files
        files = self.discover_files(entity_type, granularity)

        if not files:
            logger.warning("No files found for %s %s", entity_type, granularity)
            return None

        logger.info("Found %d file(s) to combine", len(files))

        # Load all files
        dataframes = []
        all_date_ranges = []

        for filepath in files:
            try:
                logger.info("Loading: %s", filepath.name)
                # Use on_bad_lines='warn' to skip malformed lines but continue loading
                df = pd.read_csv(filepath, on_bad_lines="warn", engine="python")

                # Extract date range from filename
                date_range = self.extract_date_range(filepath)
                if date_range:
                    all_date_ranges.append(date_range)

                dataframes.append(df)
                logger.info("  Loaded %d rows, %d columns", len(df), len(df.columns))

            except Exception as e:
                logger.error("Error loading %s: %s", filepath.name, e)
                continue

        if not dataframes:
            logger.error("No dataframes successfully loaded")
            return None

        # Validate columns
        if not skip_validation and not self.validate_columns(dataframes):
            logger.warning(
                "Column validation failed. Use --skip-validation to combine anyway."
            )
            return None

        # Combine dataframes
        logger.info("Combining %d dataframe(s)...", len(dataframes))
        combined = pd.concat(dataframes, ignore_index=True)

        # Remove duplicates
        entity_id_col = f"{entity_type}_id"
        if entity_id_col in combined.columns and "date_start" in combined.columns:
            subset = [entity_id_col, "date_start"]
            if granularity == "hourly" and "hour" in combined.columns:
                subset.append("hour")

            before_dedup = len(combined)
            combined = combined.drop_duplicates(subset=subset, keep="first")
            duplicates_removed = before_dedup - len(combined)

            if duplicates_removed > 0:
                logger.info(
                    "Removed %d duplicate row(s) based on %s",
                    duplicates_removed,
                    ", ".join(subset),
                )

        # Sort data
        sort_columns = ["date_start"]
        if entity_id_col in combined.columns:
            sort_columns.append(entity_id_col)

        combined = combined.sort_values(sort_columns).reset_index(drop=True)

        logger.info(
            "Combined result: %d rows, %d columns", len(combined), len(combined.columns)
        )

        return combined

    def generate_output_filename(
        self, entity_type: str, granularity: str, start_date: str, end_date: str
    ) -> str:
        """Generate output filename with combined date range.

        Args:
            entity_type: 'ad' or 'adset'
            granularity: 'daily' or 'hourly'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Output filename
        """
        return f"{entity_type}_{granularity}_insights_{start_date}_{end_date}.csv"

    def save_combined_file(
        self,
        df: pd.DataFrame,
        entity_type: str,
        granularity: str,
        start_date: str,
        end_date: str,
    ) -> Path:
        """Save combined DataFrame to CSV.

        Args:
            df: Combined DataFrame
            entity_type: 'ad' or 'adset'
            granularity: 'daily' or 'hourly'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = self.generate_output_filename(
            entity_type, granularity, start_date, end_date
        )
        output_path = self.output_dir / filename

        # Save to CSV
        logger.info("Saving combined file to: %s", output_path)
        df.to_csv(output_path, index=False)

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info("Saved %d rows to %s (%.2f MB)", len(df), filename, file_size)

        return output_path

    def process_entity_type(
        self,
        entity_type: str,
        granularity: str,
        skip_validation: bool = False,
        dry_run: bool = False,
    ) -> bool:
        """Process a single entity type and granularity.

        Args:
            entity_type: 'ad' or 'adset'
            granularity: 'daily' or 'hourly'
            skip_validation: If True, skip column validation
            dry_run: If True, don't actually write files

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info("Processing: %s %s", entity_type, granularity)
        logger.info("=" * 70)

        # Combine files
        combined_df = self.combine_csv_files(entity_type, granularity, skip_validation)

        if combined_df is None:
            logger.warning("No data combined for %s %s", entity_type, granularity)
            return False

        # Calculate combined date range
        all_date_ranges = []

        for filepath in self.discover_files(entity_type, granularity):
            date_range = self.extract_date_range(filepath)
            if date_range:
                all_date_ranges.append(date_range)

        if not all_date_ranges:
            logger.warning("Could not extract date ranges from filenames")
            return False

        # Find overall start and end dates
        start_dates = [dr[0] for dr in all_date_ranges]
        end_dates = [dr[1] for dr in all_date_ranges]

        overall_start = min(start_dates)
        overall_end = max(end_dates)

        logger.info(
            "Combined date range: %s to %s (%d file periods)",
            overall_start,
            overall_end,
            len(all_date_ranges),
        )

        # Save or dry run
        if dry_run:
            logger.info(
                "[DRY RUN] Would save to: %s",
                self.output_dir
                / self.generate_output_filename(
                    entity_type, granularity, overall_start, overall_end
                ),
            )
            return True
        else:
            self.save_combined_file(
                combined_df, entity_type, granularity, overall_start, overall_end
            )
            return True
