"""Unified path management for the budget allocation system.

Provides centralized path resolution with support for different environments,
customers, and platforms.
"""

import os
from pathlib import Path
from typing import Optional

from .schemas import PathsConfig


class PathManager:
    """Centralized path resolution manager.

    Handles all path operations for the budget allocation system, providing
    a single source of truth for data locations.
    """

    def __init__(
        self,
        customer: str,
        platform: str,
        paths_config: Optional[PathsConfig] = None,
        base_dir_override: Optional[Path] = None,
    ):
        """Initialize the path manager.

        Args:
            customer: Customer name
            platform: Platform name (e.g., 'meta', 'google')
            paths_config: Path configuration (uses defaults if None)
            base_dir_override: Override base directory (useful for testing)
        """
        self.customer = customer
        self.platform = platform
        self.config = paths_config or PathsConfig()

        # Helper to safely extract config values (handles Pydantic Field objects in Python 3.12)
        def _get_config_value(field_name: str, default: str = ".") -> str:
            """Safely extract config value, handling Pydantic Field objects."""
            value = getattr(self.config, field_name, default)
            # If it's a Field object (Python 3.12 compatibility), get its default
            if hasattr(value, "default") and not isinstance(value, (str, Path)):
                return value.default if callable(value.default) else value.default
            return value if isinstance(value, (str, Path)) else default

        # Allow base directory override for testing
        if base_dir_override:
            self.base_dir = Path(base_dir_override)
        elif "BASE_DIR" in os.environ:
            self.base_dir = Path(os.environ["BASE_DIR"])
        else:
            self.base_dir = Path(_get_config_value("base_dir", "."))

        # Store config values as attributes to avoid repeated Field access
        self._data_dir = _get_config_value("data_dir", "datasets")
        self._results_dir = _get_config_value("results_dir", "results")
        self._logs_dir = _get_config_value("logs_dir", "logs")
        self._cache_dir = _get_config_value("cache_dir", "cache")
        self._config_dir = _get_config_value("config_dir", "config")

    # === Input Data Paths ===

    @property
    def data_base(self) -> Path:
        """Base data directory."""
        return self.base_dir / self._data_dir

    def raw_data_dir(
        self, customer: Optional[str] = None, platform: Optional[str] = None
    ) -> Path:
        """Raw input data directory for a customer/platform."""
        customer = customer or self.customer
        platform = platform or self.platform
        return self.data_base / customer / platform / "raw"

    def features_dir(
        self, customer: Optional[str] = None, platform: Optional[str] = None
    ) -> Path:
        """Features directory for a customer/platform."""
        customer = customer or self.customer
        platform = platform or self.platform
        return self.data_base / customer / platform / "features"

    def ad_features_path(
        self, customer: Optional[str] = None, platform: Optional[str] = None
    ) -> Path:
        """Path to ad-level features file."""
        return self.features_dir(customer, platform) / "ad_features.csv"

    def adset_features_path(
        self, customer: Optional[str] = None, platform: Optional[str] = None
    ) -> Path:
        """Path to adset-level features file."""
        return self.features_dir(customer, platform) / "adset_features.csv"

    # === Output Paths ===

    @property
    def results_base(self) -> Path:
        """Base results directory."""
        return self.base_dir / self._results_dir

    def results_dir(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        method: str = "rules",
    ) -> Path:
        """Results directory for a customer/platform/method."""
        customer = customer or self.customer
        platform = platform or self.platform
        return self.results_base / customer / platform / method

    def allocations_path(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        method: str = "rules",
        filename: str = "budget_allocations.csv",
    ) -> Path:
        """Path to budget allocations file."""
        return self.results_dir(customer, platform, method) / filename

    # === Log Paths ===

    @property
    def logs_base(self) -> Path:
        """Base logs directory."""
        return self.base_dir / self._logs_dir

    def log_file(self, name: str) -> Path:
        """Path to a specific log file."""
        return self.logs_base / f"{name}.log"

    # === Cache Paths ===

    @property
    def cache_base(self) -> Path:
        """Base cache directory."""
        return self.base_dir / self._cache_dir

    def cache_path(self, cache_key: str) -> Path:
        """Path to a specific cache item."""
        return self.cache_base / f"{cache_key}.pkl"

    # === Config Paths ===

    @property
    def config_base(self) -> Path:
        """Base config directory."""
        return self.base_dir / self._config_dir

    def rules_config_path(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        filename: str = "rules.yaml",
    ) -> Path:
        """Path to rules configuration file."""
        customer = customer or self.customer
        platform = platform or self.platform
        return self.config_base / customer / platform / filename

    def system_config_path(
        self,
        environment: Optional[str] = None,
        filename: str = "system.yaml",
    ) -> Path:
        """Path to system configuration file."""
        environment = environment or "default"
        return self.config_base / f"{environment}.yaml"

    # === State Paths ===

    def state_dir(
        self, customer: Optional[str] = None, platform: Optional[str] = None
    ) -> Path:
        """State directory for monthly tracking."""
        customer = customer or self.customer
        platform = platform or self.platform
        return self.results_base / customer / platform

    def monthly_state_path(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        month: Optional[str] = None,
    ) -> Path:
        """Path to monthly state JSON file."""
        customer = customer or self.customer
        platform = platform or self.platform
        if month is None:
            from datetime import datetime

            month = datetime.now().strftime("%Y-%m")
        return self.state_dir(customer, platform) / f"monthly_state_{month}.json"

    def archive_dir(
        self,
        customer: Optional[str] = None,
        platform: Optional[str] = None,
        month: Optional[str] = None,
    ) -> Path:
        """Archive directory for historical allocations."""
        customer = customer or self.customer
        platform = platform or self.platform
        if month is None:
            from datetime import datetime

            month = datetime.now().strftime("%Y-%m")
        return self.state_dir(customer, platform) / "archive" / month

    # === Utility Methods ===

    def ensure_directories(self, create: bool = True) -> None:
        """Ensure all required directories exist.

        Args:
            create: If True, create directories that don't exist
        """
        directories = [
            self.raw_data_dir(),
            self.features_dir(),
            self.results_dir(),
            self.logs_base,
            self.cache_base,
        ]

        for directory in directories:
            if create:
                directory.mkdir(parents=True, exist_ok=True)
            elif not directory.exists():
                raise FileNotFoundError(f"Directory does not exist: {directory}")

    def validate_paths(self, require_exists: bool = False) -> bool:
        """Validate that paths are correctly configured.

        Args:
            require_exists: If True, require paths to exist

        Returns:
            True if validation passes
        """
        if require_exists:
            self.ensure_directories(create=False)
        return True

    def override_base_dir(self, new_base: Path) -> None:
        """Override the base directory (useful for testing).

        Args:
            new_base: New base directory path
        """
        self.base_dir = Path(new_base)


# Global path manager instance (lazy initialized)
_path_manager: Optional[PathManager] = None


def get_path_manager(
    customer: str = "moprobo",
    platform: str = "meta",
    paths_config: Optional[PathsConfig] = None,
    base_dir_override: Optional[Path] = None,
    force_refresh: bool = False,
) -> PathManager:
    """Get or create the global path manager instance.

    Args:
        customer: Customer name
        platform: Platform name
        paths_config: Path configuration
        base_dir_override: Override base directory
        force_refresh: Force recreation of path manager

    Returns:
        PathManager instance
    """
    global _path_manager

    if _path_manager is None or force_refresh:
        _path_manager = PathManager(
            customer=customer,
            platform=platform,
            paths_config=paths_config,
            base_dir_override=base_dir_override,
        )

    return _path_manager


def reset_path_manager() -> None:
    """Reset the global path manager instance (useful for testing)."""
    global _path_manager
    _path_manager = None
