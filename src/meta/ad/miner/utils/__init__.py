"""Utilities Module

This module contains utility functions and classes for statistics,
visualization, file operations, API key management, and configuration
management.
"""

from .api_keys import get_fal_api_key, get_openai_api_key
from .config_manager import ConfigManager
from .constants import (
    AnalysisConstants,
    APIConstants,
    DataConstants,
    EnvironmentConstants,
    FeaturesConstants,
    ModelConstants,
    OutputConstants,
)
from .statistics import (
    calculate_effect_size,
    chi_square_and_cramers_v,
    chi_square_test,
    cramers_v,
)
from .paths import (
    MinerPaths,
    find_repo_root,
    get_data_dir,
    resolve_features_csv,
    resolve_and_validate_input_csv,
    get_default_paths,
    get_segment_paths,
)

# Note: Remaining modules are planned for future implementation
# from .visualization import plot_feature_importance, plot_feature_values
# from .file_utils import save_json, save_csv

__all__ = [
    "chi_square_test",
    "cramers_v",
    "calculate_effect_size",
    "chi_square_and_cramers_v",
    "get_openai_api_key",
    "get_fal_api_key",
    "ConfigManager",
    "FeaturesConstants",
    "DataConstants",
    "ModelConstants",
    "AnalysisConstants",
    "EnvironmentConstants",
    "APIConstants",
    "OutputConstants",
    # Path utilities
    "MinerPaths",
    "find_repo_root",
    "get_data_dir",
    "resolve_features_csv",
    "resolve_and_validate_input_csv",
    "get_default_paths",
    "get_segment_paths",
    # "plot_feature_importance",
    # "plot_feature_values",
    # "save_json",
    # "save_csv",
]
