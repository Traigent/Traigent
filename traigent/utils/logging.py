"""Logging utilities for Traigent SDK."""

# Traceability: CONC-Layer-Infra CONC-Quality-Observability CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import logging
import os
import sys
from types import ModuleType
from typing import Any, cast


def _resolve_logging_level(level: str | None = None) -> str:
    """Resolve SDK logging level, honoring the documented env override."""

    configured = os.environ.get("TRAIGENT_LOG_LEVEL") or level or "INFO"
    return configured.strip().upper()


def configure_litellm_logging(
    level: str | None = None,
    *,
    litellm_module: ModuleType | None = None,
) -> None:
    """Apply Traigent's LiteLLM logging policy.

    The logger policy can be applied before LiteLLM is imported. When a LiteLLM
    module is available, the same function also toggles its debug banner flag.
    """

    log_level = _resolve_logging_level(level)
    debug_enabled = log_level == "DEBUG"

    litellm_logger = logging.getLogger("LiteLLM")
    if debug_enabled:
        litellm_logger.setLevel(logging.DEBUG)
        litellm_logger.propagate = True
    else:
        litellm_logger.setLevel(logging.WARNING)
        litellm_logger.propagate = False

    module = litellm_module
    if module is None:
        candidate = sys.modules.get("litellm")
        if isinstance(candidate, ModuleType):
            module = candidate

    if module is not None:
        cast(Any, module).suppress_debug_info = not debug_enabled


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
    resolved_level = _resolve_logging_level(level)
    numeric_level = getattr(logging, resolved_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {resolved_level}")

    # Use standard handler (rich disabled for basic testing)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))

    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Set specific levels for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    configure_litellm_logging(resolved_level)


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

# Configure third-party loggers early without importing optional dependencies.
configure_litellm_logging()
