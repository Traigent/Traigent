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
    level: str = "INFO",
    format_string: str | None = None,
    use_rich: bool = False,
    logger_name: str | None = None,
) -> None:
    """Setup logging configuration for Traigent.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        use_rich: Whether to use rich formatting for console output
        logger_name: Optional opt-in for host applications that embed Traigent.
            Defaults to ``None``, which preserves Traigent's historical behavior
            of clearing and reconfiguring the **ROOT** logger
            (``logging.getLogger()``) — existing callers (``configure()``,
            ``initialize()``, the CLI, diagnostics) are unaffected. Pass a
            logger name (e.g. ``"traigent"``) to scope this call to that
            logger tree instead: only that logger's handlers are cleared and
            reconfigured, and the ROOT logger is left completely untouched.
            The scoped logger also gets ``propagate = False`` so records it
            handles do not bubble up and get emitted a second time by the
            host's root handlers. This lets a host application that has
            already configured `logging` for itself opt into
            Traigent-scoped logging without losing its own root
            configuration.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Resolve the target logger: ROOT by default (unchanged behavior), or an
    # explicit opt-in named logger when the host asks for one.
    target_logger = (
        logging.getLogger(logger_name) if logger_name else logging.getLogger()
    )

    # Remove existing handlers on the target logger only.
    for handler in target_logger.handlers[:]:
        target_logger.removeHandler(handler)

    # Configure logging level
    resolved_level = _resolve_logging_level(level)
    numeric_level = getattr(logging, resolved_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {resolved_level}")

    # Use standard handler (rich disabled for basic testing)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))

    target_logger.addHandler(handler)
    target_logger.setLevel(numeric_level)
    if logger_name:
        # Scoped opt-in: our handler now emits these records; stop them from
        # bubbling up to the host's root handlers (double emission).
        target_logger.propagate = False

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
