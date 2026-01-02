"""Logging utilities for Traigent SDK."""

# Traceability: CONC-Layer-Infra CONC-Quality-Observability CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import logging


def setup_logging(
    level: str = "INFO", format_string: str | None = None, use_rich: bool = False
) -> None:
    """Setup logging configuration for Traigent.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        use_rich: Whether to use rich formatting for console output
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Use standard handler (rich disabled for basic testing)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))

    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Set specific levels for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Create default logger
logger = get_logger(__name__)
