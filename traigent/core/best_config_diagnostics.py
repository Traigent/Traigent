"""The one place best-config diagnostics are worded.

Customer-controlled text must never reach SDK logs, exception text or tracebacks on the
best-config load / fetch / publish / cache paths. That covers config ids, function refs,
environment values, file paths, cache directories, URLs, manifest and spec keys, and the
message of any exception raised by I/O, JSON parsing, transport or validation.

Rules this module enforces by construction:

* a log line is a **fixed message** plus a **category** (:func:`log`); the optional
  exception contributes only its *type name*, never its text;
* a raised SDK error (:func:`refusal`) carries a fixed message; the category, the inner
  exception *type name* and any rejected values live on ``exc.details`` only (never in
  ``str`` / ``repr`` / ``args``);
* the raised error has no printable cause: :func:`refusal` returns it with
  ``__cause__ = None`` and ``__suppress_context__ = True``, and call sites still write
  ``raise ... from None`` so a traceback cannot print the inner exception either.

Callers pass only string *literals* as ``message``; nothing interpolated.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from traigent.utils.exceptions import ConfigurationError


class Category(StrEnum):
    """Closed set of best-config diagnostic categories."""

    LOAD_FAILED = "load_failed"
    SOURCE_MODE = "source_mode_unknown"
    CACHE_WRITE_FAILED = "cache_write_failed"
    CACHE_STALE = "cache_stale"
    CACHE_INVALID = "cache_invalid"
    REPO_INVALID = "repo_invalid"
    FETCH_FAILED = "fetch_failed"
    FETCH_INTEGRITY_FAILED = "fetch_integrity_failed"
    PUBLISH_FAILED = "publish_failed"
    PUBLISH_INTEGRITY_FAILED = "publish_integrity_failed"
    TRANSPORT_FAILED = "transport_failed"
    SOURCE_IGNORED = "source_ignored"
    FILESYSTEM_FAILED = "filesystem_failed"
    RESULT_INVALID = "result_invalid"


_SAFE_MARK = "_best_config_diag_safe_message"


class SafeConfigurationError(ConfigurationError):
    """A ``ConfigurationError`` whose message is a fixed literal written in this SDK.

    Raised only with string literals (or a literal plus an exception *type* name);
    :func:`fixed_reason` may therefore print its message. A plain ``ConfigurationError``
    (e.g. from a dependency) is not trusted.
    """

    _best_config_diag_safe_message = True


def mark_safe(error: ConfigurationError) -> ConfigurationError:
    """Mark ``error`` as carrying a fixed, literal message (see :func:`fixed_reason`)."""

    setattr(error, _SAFE_MARK, True)
    return error


def is_safe(exc: BaseException) -> bool:
    """True for a ``ConfigurationError`` built by :func:`refusal` / :func:`mark_safe`."""

    return isinstance(exc, ConfigurationError) and bool(getattr(exc, _SAFE_MARK, False))


def fixed_reason(exc: BaseException) -> str:
    """Log-safe reason: a SDK refusal's fixed message, else only the exception type.

    Only an error built by :func:`refusal` (marked with ``_SAFE_MARK``) contributes its
    message; any other exception, including a plain ``ConfigurationError`` raised by a
    dependency (its text may interpolate customer content), contributes its type name.
    """

    if isinstance(exc, ConfigurationError) and getattr(exc, _SAFE_MARK, False):
        return exc.message
    return type(exc).__name__


def log(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    category: Category,
    exc: BaseException | None = None,
) -> None:
    """Emit ``message`` (a fixed literal) tagged with ``category``.

    ``exc`` contributes its fixed reason (see :func:`fixed_reason`) only; no traceback
    is attached.
    """

    if exc is None:
        logger.log(level, "%s [%s]", message, category.value)
    else:
        logger.log(level, "%s [%s]: %s", message, category.value, fixed_reason(exc))


def refusal(
    message: str,
    *,
    category: Category,
    cause: BaseException | None = None,
    error_cls: type[ConfigurationError] = ConfigurationError,
    **details: Any,
) -> ConfigurationError:
    """Build a sanitized error: fixed ``message``; everything else on ``details``.

    Use as ``raise refusal("...", category=..., cause=exc) from None``.
    """

    payload: dict[str, Any] = {"category": category.value, **details}
    if cause is not None:
        inner = getattr(cause, "details", None)
        if getattr(cause, _SAFE_MARK, False) and isinstance(inner, dict):
            # a refusal wrapping a refusal keeps the original exception type name
            payload["error_type"] = inner.get("error_type", type(cause).__name__)
        else:
            payload["error_type"] = type(cause).__name__
    error = error_cls(message, details=payload)
    error.__cause__ = None
    error.__suppress_context__ = True
    return mark_safe(error)


@contextmanager
def fs_guard(message: str) -> Iterator[None]:
    """Turn any ``OSError`` in the block into a fixed refusal (type name in details only).

    ``OSError`` text quotes the path (which carries customer-chosen components), so it is
    never printed or chained. ``message`` is a literal.
    """

    try:
        yield
    except OSError as exc:
        raise refusal(message, category=Category.FILESYSTEM_FAILED, cause=exc) from None


def transport_failed_message(operation: str, exc: BaseException) -> str:
    """Fixed transport-failure text: the operation (a literal) and the exception type."""

    return f"{operation} transport failed ({type(exc).__name__})"


__all__ = [
    "Category",
    "fixed_reason",
    "fs_guard",
    "is_safe",
    "SafeConfigurationError",
    "log",
    "mark_safe",
    "refusal",
    "transport_failed_message",
]
