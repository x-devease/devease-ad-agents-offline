"""
Optional import utilities.

Provides utilities for handling optional dependencies gracefully.
"""

import importlib
import logging
from typing import Any, Callable, Optional, TypeVar


T = TypeVar("T")

logger = logging.getLogger(__name__)


def optional_import(
    module_path: str,
    item: Optional[str] = None,
    default: Optional[T] = None,
) -> Any:
    """
    Import a module or item with graceful fallback.

    Args:
        module_path: Dot-notation path to module (e.g., "package.module")
        item: Item to import from module (if None, imports module itself)
        default: Value to return if import fails (default: None)

    Returns:
        Imported module/item, or default if import fails

    Examples:
        >>> # Import module
        >>> numpy = optional_import("numpy")
        >>> if numpy is None:
        ...     print("NumPy not available")

        >>> # Import specific item
        >>> Figure = optional_import("matplotlib.pyplot", "Figure")
        >>> if Figure is None:
        ...     print("matplotlib not available")
    """
    try:
        if item:
            mod = importlib.import_module(module_path)
            return getattr(mod, item)
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug(
            "Optional import failed: %s%s",
            module_path,
            f".{item}" if item else "",
            exc_info=False,
        )
        return default


def lazy_import(
    module_path: str,
    item: Optional[str] = None,
) -> Callable[[], Any]:
    """
    Create a lazy import that is evaluated on first call.

    Useful for delaying imports until they're actually needed.

    Args:
        module_path: Dot-notation path to module
        item: Item to import from module (if None, imports module itself)

    Returns:
        Callable that returns the imported module/item when called

    Examples:
        >>> get_numpy = lazy_import("numpy")
        >>> # ... later
        >>> np = get_numpy()
        >>> if np is not None:
        ...     np.array([1, 2, 3])
    """
    cached: Any = None
    import_attempted = False

    def _import() -> Any:
        nonlocal cached, import_attempted
        if not import_attempted:
            cached = optional_import(module_path, item)
            import_attempted = True
        return cached

    return _import
