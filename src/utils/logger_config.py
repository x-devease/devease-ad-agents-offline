"""
Shared logging configuration for the project.

Provides enhanced logging with:
- Structured output with context
- File rotation
- Configurable log levels per module
- Request/correlation ID tracking support
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file settings
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    format_string: Optional[str] = None,
    enable_json: bool = False,
) -> logging.Logger:
    """
    Configure enhanced logging for the application.

    Args:
        level: Logging level (default: INFO).
        log_file: Optional log file name. If provided, logs will be written to file
            with rotation. Defaults to None (console only).
        log_dir: Directory for log files (default: "logs/"). Ignored if log_file is None.
        max_bytes: Maximum size of log file before rotation (default: 10 MB).
        backup_count: Number of backup files to keep (default: 5).
        format_string: Custom format string. If None, uses DEFAULT_FORMAT.
        enable_json: If True, use JSON formatting for structured logging (default: False).

    Returns:
        Configured logger instance.

    Example:
        >>> # Console logging only
        >>> logger = setup_logging()
        >>>
        >>> # Console and file logging with rotation
        >>> logger = setup_logging(log_file="app.log")
        >>>
        >>> # JSON structured logging
        >>> logger = setup_logging(log_file="app.log", enable_json=True)
    """
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Define formatter
    if format_string is None:
        if enable_json:
            format_string = '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        else:
            format_string = DEFAULT_FORMAT

    formatter = logging.Formatter(
        fmt=format_string,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation (if log_file specified)
    if log_file:
        if log_dir is None:
            log_dir = DEFAULT_LOG_DIR

        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Full path to log file
        log_file_path = log_path / log_file

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logging.getLogger(__name__)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance with optional level override.

    Args:
        name: Logger name (typically __name__ of the module).
        level: Optional log level override for this logger.

    Returns:
        Logger instance.

    Example:
        >>> from src.utils.logger_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello world")
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


def set_module_log_level(module_name: str, level: int) -> None:
    """
    Set log level for a specific module.

    Args:
        module_name: Name of the module (e.g., "src.optimizer").
        level: Logging level (logging.DEBUG, logging.INFO, etc.).

    Example:
        >>> import logging
        >>> set_module_log_level("src.optimizer", logging.DEBUG)
    """
    logging.getLogger(module_name).setLevel(level)


class ContextFilter(logging.Filter):
    """
    Logging filter that adds contextual information to log records.

    Can be used to add request IDs, user IDs, or other contextual info.
    """

    def __init__(self, context_key: str, context_value: str):
        """
        Initialize context filter.

        Args:
            context_key: Key name for the context (e.g., "request_id").
            context_value: Value for the context.
        """
        super().__init__()
        self.context_key = context_key
        self.context_value = context_value

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context to the log record.

        Args:
            record: Log record to filter.

        Returns:
            True (always allows the record).
        """
        setattr(record, self.context_key, self.context_value)
        return True


def add_context_to_logger(
    logger: logging.Logger, context_key: str, context_value: str
) -> None:
    """
    Add contextual filter to a logger.

    Args:
        logger: Logger instance.
        context_key: Key name for the context.
        context_value: Value for the context.

    Example:
        >>> logger = get_logger(__name__)
        >>> add_context_to_logger(logger, "request_id", "abc-123")
        >>> logger.info("Processing request")  # Will include request_id in output
    """
    context_filter = ContextFilter(context_key, context_value)
    logger.addFilter(context_filter)


# Convenience function for backward compatibility
def get_default_logger() -> logging.Logger:
    """
    Get default logger with basic configuration.

    For new code, prefer using setup_logging() or get_logger().
    """
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(__name__)
