"""API key management utilities.

This module provides helper functions for loading API keys from environment
variables and configuration files. It centralizes API key management for the
entire project.

Usage:
    from src.utils.api_keys import get_openai_api_key, get_fal_api_key

    api_key = get_openai_api_key()
    if not api_key:
        raise ValueError("OpenAI API key not found")
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_keys(key_name: str, keys_file_path: Optional[str] = None) -> str:
    """Load API key from keys file.

    Reads the specified API key value from a configuration file. The file format
    should be key=value pairs, one per line.

    Args:
        key_name: Name of the API key to load (e.g., "OPENAI_API_KEY").
        keys_file_path: Path to keys file (default: ~/.devease/keys).

    Returns:
        API key string if found, empty string otherwise.
    """
    if keys_file_path is None:
        # Default keys file location
        keys_file_path = str(Path.home() / ".devease" / "keys")

    keys_path = Path(keys_file_path)
    if not keys_path.exists():
        return ""

    try:
        with open(keys_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if line.startswith(f"{key_name}="):
                    # Extract value after '='
                    api_key = line.split("=", 1)[1].strip()
                    # Remove quotes if present
                    if api_key.startswith('"') and api_key.endswith('"'):
                        api_key = api_key[1:-1]
                    elif api_key.startswith("'") and api_key.endswith("'"):
                        api_key = api_key[1:-1]
                    return api_key
    except (IOError, OSError, ValueError) as error:
        logger.warning("Error reading keys file: %s", error)

    return ""


def _get_api_key(env_var_name: str, key_name: str) -> Optional[str]:
    """Get API key from environment variable or configuration file.

    Args:
        env_var_name: Environment variable name to check first
        key_name: Key name to use when loading from keys file

    Returns:
        API key string if found, None otherwise.
    """
    # Try environment variable first
    api_key = os.getenv(env_var_name)
    if api_key:
        return api_key

    # Try loading from keys file
    api_key = load_keys(key_name)
    if api_key:
        return api_key

    return None


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment variable or configuration file.

    Checks environment variables first, then falls back to ~/.devease/keys file.

    Returns:
        API key string if found, None otherwise.
    """
    return _get_api_key("OPENAI_API_KEY", "OPENAI_API_KEY")


def get_fal_api_key() -> Optional[str]:
    """Get FAL API key from environment variable or configuration file.

    Checks environment variables first, then falls back to ~/.devease/keys file.

    Returns:
        API key string if found, None otherwise.
    """
    return _get_api_key("FAL_KEY", "FAL_KEY")
