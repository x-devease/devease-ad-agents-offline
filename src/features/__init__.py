"""
Feature extractor module for joining multi-level data.
Enriches ad-level data with campaign, adset, and account level features.
"""

# Core components
from .core.extractor import Extractor

# Library modules
from .lib.aggregator import Aggregator
from .lib.joiner import Joiner
from .lib.loader import Loader

# Utilities
from .utils.file_discovery import FileDiscovery
from .utils.json_parser import JSONParser
from .utils.meta_data_processor import MetaDataProcessor
from .lib.preprocessor import Preprocessor

__all__ = [
    "Extractor",
    "Aggregator",
    "FileDiscovery",
    "Joiner",
    "JSONParser",
    "Loader",
    "MetaDataProcessor",
    "Preprocessor",
]
