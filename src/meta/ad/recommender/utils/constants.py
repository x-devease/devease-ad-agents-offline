"""Centralized constants for the entire project.

This module provides a centralized location for all constants used across
the project, organized by module/component. This eliminates the need for
multiple constants files scattered throughout the codebase.

Usage:
    from src.utils import (
        FeaturesConstants,
        DataConstants,
        ModelConstants,
    )  # noqa: E501
    # or
    from src.utils.constants import (
        FeaturesConstants,
        DataConstants,
        ModelConstants,
    )

    batch_size = FeaturesConstants.DEFAULT_BATCH_SIZE
    roas_column = DataConstants.DEFAULT_ROAS_COLUMNS[0]
"""

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


class FeaturesConstants:
    """Constants for the feature extractor module."""

    # API Configuration
    DEFAULT_API_TIMEOUT: float = 180.0  # 3 minutes
    DEFAULT_BATCH_SIZE: int = 5
    DEFAULT_RATE_LIMIT_DELAY: float = 3.0
    MAX_RETRIES: int = 10
    BASE_RETRY_DELAY: float = 2.0
    MAX_RETRY_DELAY: float = 60.0

    # File Extensions
    SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
    }

    # Default Paths
    DEFAULT_IMAGES_FOLDER: str = "images"
    DEFAULT_OUTPUT_DIR: str = "config/ad/recommender/features"
    # Note: Progress/checkpoint/results files should go to results/ or cache/, not config/
    # These constants are unused and kept for reference only
    DEFAULT_PROGRESS_FILE: str = "results/ad/recommender/analysis_progress.json"
    DEFAULT_CHECKPOINT_FILE: str = "cache/ad/recommender/image_features_checkpoint.json"
    DEFAULT_RESULTS_JSON: str = "results/ad/recommender/gpt4_analysis_results.json"
    DEFAULT_RESULTS_CSV: str = "results/ad/recommender/gpt4_analysis_results.csv"

    # GPT Model Configuration
    DEFAULT_GPT_MODEL: str = "gpt-4.1"
    FALLBACK_GPT_MODELS: List[str] = ["gpt-4o", "gpt-4-vision-preview"]

    # Performance Labels
    PERFORMANCE_LABEL_HIGH: str = "high"
    PERFORMANCE_LABEL_LOW: str = "low"
    PERFORMANCE_LABEL_UNKNOWN: str = "unknown"
    PERFORMANCE_LABEL_PREDICTION: str = "prediction"

    # Batch Processing
    DEFAULT_TOP_150_FILE: str = "results/ad/recommender/top_150_images.csv"
    DEFAULT_BOTTOM_150_FILE: str = "results/ad/recommender/bottom_150_images.csv"
    BATCH_SIZE_FOR_CLASSIFIED: int = 5  # For top/bottom 150 processing

    # Feature Count
    TOTAL_FEATURE_COUNT: int = 29  # Number of features in the model

    def __repr__(self) -> str:
        """Return string representation of FeaturesConstants."""
        return "FeaturesConstants"

    def get_class_name(self) -> str:
        """Return the class name."""
        return "FeaturesConstants"


class DataConstants:
    """Constants for data loading and processing."""

    # ROAS Column Names (priority order)
    DEFAULT_ROAS_COLUMNS: List[str] = [
        "mean_roas",
        "total_roas",
        "weighted_roas",
    ]

    # Identifier Column Names
    DEFAULT_ID_COLUMNS: List[str] = ["filename", "creative_id"]

    # ROAS Exclusion Patterns (exclude ALL ROAS-related columns except target)
    ROAS_EXCLUDE_PATTERNS: List[str] = [
        "roas",
        "roi",
        "revenue",
        "purchase_value",
        "conversion",
    ]

    # ID/Path Exclusion Patterns (exclude all identifier columns)
    ID_EXCLUDE_PATTERNS: List[str] = [
        "filename",
        "image_filename",
        "image_path",
        "creative_id",
        "creative_name",
    ]

    # Metric Exclusion Patterns (exclude all business performance metrics)
    METRIC_EXCLUDE_PATTERNS: List[str] = [
        "spend",
        "impressions",
        "clicks",
        "ad_count",
        "performance_label",
        "ctr",
        "cpm",
        "cpc",
        "cpa",
        "cvr",
        "conversion",
        "calculated_ctr",
        "calculated_cpm",
        "calculated_cpc",
        "total_spend",
        "total_clicks",
        "total_impressions",
        "total_purchase_value",
    ]

    # Metadata Exclusion Patterns (exclude data quality and metadata columns)
    METADATA_EXCLUDE_PATTERNS: List[str] = [
        "data_completeness",
        "data_quality",
        "quality_grade",
        "completeness_score",
        "quality_score",
        "creative_name",
        "ad_name",
        "campaign_name",
        "created_date",
        "updated_date",
        "timestamp",
    ]

    # Annotation Column Names (exclude annotation columns)
    ANNOTATION_COLUMNS: List[str] = [
        "sample_type",
        "is_negative_sample",
        "has_any_business",
        "has_engagement",
        "roas_missing",
        "is_business_missing_candidate",
        "has_no_data",
    ]

    # Sample Type Values
    SAMPLE_TYPE_POSITIVE: str = "positive"
    SAMPLE_TYPE_NEGATIVE: str = "negative"
    SAMPLE_TYPE_UNKNOWN: str = "unknown"

    # Business Metric Columns (for annotation)
    BUSINESS_COLUMNS: List[str] = [
        "total_spend",
        "total_impressions",
        "total_clicks",
    ]

    # Data Quality Defaults
    DEFAULT_MIN_SAMPLES: int = 50
    DEFAULT_MAX_MISSING_RATE: float = 0.3
    DEFAULT_ROAS_RANGE: Tuple[float, float] = (0.0, 100.0)

    # Default Paths
    DEFAULT_INPUT_PATH: str = "data/creative_image_features.csv"
    DEFAULT_OUTPUT_PATH: str = "data/results/"

    def __repr__(self) -> str:
        """Return string representation of DataConstants."""
        return "DataConstants"

    def get_class_name(self) -> str:
        """Return the class name."""
        return "DataConstants"


class ModelConstants:
    """Constants for model training and configuration."""

    # Random State
    DEFAULT_RANDOM_STATE: int = 42

    # Train/Test Split
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_CV_FOLDS: int = 5

    # Classification Threshold
    DEFAULT_CLASS_THRESHOLD_PCT: float = 75.0

    # CatBoost Regression Defaults
    DEFAULT_REGRESSION_ITERATIONS: int = 500
    DEFAULT_REGRESSION_DEPTH: int = 6
    DEFAULT_REGRESSION_LR: float = 0.05
    DEFAULT_REGRESSION_L2_LEAF_REG: float = 3.0
    DEFAULT_REGRESSION_MIN_DATA_IN_LEAF: int = 1  # pylint: disable=invalid-name
    DEFAULT_REGR_EARLY_STOP_ROUNDS: int = 50

    # CatBoost Classification Defaults
    DEFAULT_CLASSIFICATION_ITERATIONS: int = 300  # pylint: disable=invalid-name
    DEFAULT_CLASSIFICATION_DEPTH: int = 4
    DEFAULT_CLASS_LEARNING_RATE: float = 0.05
    DEFAULT_CLASS_L2_LEAF_REG: float = 10.0
    DEFAULT_CLASS_MIN_DATA_IN_LEAF: int = 1
    DEFAULT_CLASS_EARLY_STOP_ROUNDS: int = 50

    # Random Forest Defaults
    DEFAULT_RF_N_ESTIMATORS: int = 100
    DEFAULT_RF_MAX_DEPTH: int = 5
    DEFAULT_RF_MIN_SAMPLES_SPLIT: int = 5

    def __repr__(self) -> str:
        """Return string representation of ModelConstants."""
        return "ModelConstants"

    def get_class_name(self) -> str:
        """Return the class name."""
        return "ModelConstants"


class AnalysisConstants:
    """Constants for analysis configuration."""

    # Feature Importance
    DEFAULT_TOP_N_FEATURES: int = 10
    # Options: regression, classification, average
    DEFAULT_IMPORTANCE_METHOD: str = "regression"

    # Feature Value Analysis
    DEFAULT_SIGNIFICANCE_LEVEL: float = 0.05
    DEFAULT_MIN_EFFECT_SIZE: float = 0.3
    DEFAULT_TOP_PERCENTILE: int = 75
    DEFAULT_BOTTOM_PERCENTILE: int = 25
    DEFAULT_USE_MEDIAN: bool = True
    DEFAULT_MIN_GROUP_SIZE_1B: int = 5  # pylint: disable=invalid-name

    # Feature Interaction
    DEFAULT_TOP_N_INTERACTIONS: int = 10
    DEFAULT_MIN_INTERACTION_STR: float = 0.1  # pylint: disable=invalid-name

    # Feature Combination
    DEFAULT_MAX_COMBINATIONS: int = 20  # pylint: disable=invalid-name
    DEFAULT_MIN_IMPROVEMENT: float = 0.05  # pylint: disable=invalid-name
    DEFAULT_SAMPLES_PER_COMBINATION: int = 10

    # Correlation Threshold
    DEFAULT_CORRELATION_THRESHOLD: float = 0.3
    DEFAULT_MIN_SAMPLES_PER_CELL: int = 5

    def __repr__(self) -> str:
        """Return string representation of AnalysisConstants."""
        return "AnalysisConstants"

    def get_class_name(self) -> str:
        """Return the class name."""
        return "AnalysisConstants"


class EnvironmentConstants:
    """Constants for environment and system requirements."""

    # Python Version Requirements
    MIN_PYTHON_MAJOR: int = 3
    MIN_PYTHON_MINOR: int = 12

    # Required Packages (package name -> import name mapping)
    REQUIRED_PACKAGES: Dict[str, str] = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "catboost": "catboost",
        "pyyaml": "yaml",
    }

    # Environment Variable Names
    ENV_PIPELINE_ENV: str = "PIPELINE_ENV"
    ENV_PIPELINE_DEBUG: str = "PIPELINE_DEBUG"
    ENV_PIPELINE_SKIP_VALIDATION: str = "PIPELINE_SKIP_VALIDATION"

    # Default Environment Values
    DEFAULT_PIPELINE_ENV: str = "dev"
    VALID_ENVIRONMENTS: List[str] = ["dev", "staging", "prod"]

    # Test File Names
    TEST_WRITE_FILE: str = ".pipeline_test_write"

    def __repr__(self) -> str:
        """Return string representation of EnvironmentConstants."""
        return "EnvironmentConstants"

    def get_class_name(self) -> str:
        """Return the class name."""
        return "EnvironmentConstants"


class APIConstants:
    """Constants for API configuration."""

    # API Key Names
    OPENAI_API_KEY_NAME: str = "OPENAI_API_KEY"
    FAL_KEY_NAME: str = "FAL_KEY"

    # API Key File Path
    DEFAULT_KEYS_FILE_PATH: str = "~/.devease/keys"
    KEYS_FILE_RELATIVE_PATH: str = ".devease/keys"

    def __repr__(self) -> str:
        """Return string representation of APIConstants."""
        return "APIConstants"

    def get_class_name(self) -> str:
        """Return the class name."""
        return "APIConstants"


class OutputConstants:
    """Constants for output configuration."""

    # Output Formats
    DEFAULT_OUTPUT_FORMATS: List[str] = ["json", "html", "markdown"]

    def __repr__(self) -> str:
        """Return string representation of OutputConstants."""
        return "OutputConstants"

    def get_class_name(self) -> str:
        """Return the class name."""
        return "OutputConstants"


class FeatureLinkConstants:
    """Constants for feature linking and ROAS processing."""

    # Synthetic ROAS Generation
    RANDOM_ROAS_MAX: float = 10.0
    MIN_ROAS_VALUE: float = 0.1

    def __repr__(self) -> str:
        """Return string representation of FeatureLinkConstants."""
        return "FeatureLinkConstants"

    def get_class_name(self) -> str:
        """Return the class name."""
        return "FeatureLinkConstants"

    SYNTHETIC_METHOD_RANDOM: str = "random"
    SYNTHETIC_METHOD_FEATURE_BASED: str = "feature_based"
    SYNTHETIC_METHODS: List[str] = ["random", "feature_based"]

    # Feature-based synthetic ROAS mappings
    BRIGHTNESS_MAP: Dict[str, float] = {
        "bright": 2.0,
        "medium": 1.0,
        "dark": 0.0,
    }
    COLOR_VIBRANCY_MAP: Dict[str, float] = {
        "vibrant": 2.0,
        "moderate": 1.0,
        "muted": 0.0,
    }
    HUMAN_PRESENCE_MAP: Dict[str, float] = {
        True: 1.5,
        False: 0.5,
        "True": 1.5,
        "False": 0.5,
        "true": 1.5,
        "false": 0.5,
    }

    # Feature-based synthetic ROAS random ranges
    BRIGHTNESS_ROAS_RANGE: Tuple[float, float] = (0.5, 1.5)
    COLOR_VIBRANCY_ROAS_RANGE: Tuple[float, float] = (0.3, 1.2)
    HUMAN_PRESENCE_ROAS_RANGE: Tuple[float, float] = (0.5, 2.0)
    NOISE_MEAN: float = 0.0
    NOISE_STD: float = 1.0


def get_excluded_feature_cols(
    df: pd.DataFrame,
    target_col: str,
    exclude_patterns: Optional[List[str]] = None,
    additional_exclude_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Get excluded and feature columns from a dataframe.

    This function identifies columns to exclude based on patterns and specific
    column names, and returns the excluded columns and feature columns.

    Args:
        df: Input dataframe
        target_col: Target column name (always excluded from features)
        exclude_patterns: List of patterns to exclude (if None, uses defaults)
        additional_exclude_cols: Additional specific columns to exclude

    Returns:
        Tuple of (excluded_cols, feature_cols)
    """
    if exclude_patterns is None:
        exclude_patterns = (
            DataConstants.ROAS_EXCLUDE_PATTERNS
            + DataConstants.ID_EXCLUDE_PATTERNS
            + DataConstants.METRIC_EXCLUDE_PATTERNS
            + ["ad_count"]  # Additional common patterns
        )

    excluded_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if any(pattern in col.lower() for pattern in exclude_patterns):
            excluded_cols.append(col)

    if additional_exclude_cols:
        excluded_cols.extend(additional_exclude_cols)

    feature_cols = [
        col
        for col in df.columns
        if col not in excluded_cols and col != target_col
    ]

    return excluded_cols, feature_cols
