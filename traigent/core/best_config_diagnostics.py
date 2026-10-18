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

Threat model
------------
The guarantee: accidental customer data values never enter SDK-authored diagnostic
messages on the hardened paths. Those paths cover ordinary customer objects, exceptions
raised by dependencies, and exception chaining.

What is excluded:

* deliberate use of the internal trust APIs (``_mark_safe``, ``_safe_error`` and the
  registry behind them). Their leading underscore is a convention, not a security
  boundary; a test confines their use to ``traigent/core`` but code that wants to forge
  trust can still do so;
* monkeypatching;
* code identifiers such as class names, which are not customer content.

This is not a sandbox and not a universal no-egress guarantee. In particular, exception
text can still reach separately configured reporting paths: the observability client's
record mode captures and posts exception text, and backend status reporting sends an
``error_message``. Those paths are outside this module and need their own closure.

Stopping rule: a new finding against this module counts as a blocker only if it shows an
accidental data-value path that uses neither the internal trust APIs nor code
identifiers.
"""

from __future__ import annotations

import functools
import logging
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, TypeVar

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


# Trust is an identity fact, not a property the object carries. A marker attribute (or a
# class) can be forged by any value whose ``__str__`` raises ``SafeConfigurationError(...)``,
# so only instances registered here, by this module's own constructors, are trusted.
# The registry maps ``id(exc)`` to a weak reference and is checked by identity, so a copy
# (``copy.copy`` / ``deepcopy`` build a new object), a subclass instance or an instance
# built anywhere else is untrusted, and a subclass cannot spoof it through ``__eq__`` /
# ``__hash__``.
_TRUSTED: dict[int, weakref.ref[BaseException]] = {}


def _trust(error: ConfigurationError) -> ConfigurationError:
    key = id(error)
    _TRUSTED[key] = weakref.ref(error, lambda _ref, key=key: _TRUSTED.pop(key, None))
    return error


class SafeConfigurationError(ConfigurationError):
    """A ``ConfigurationError`` whose message is a fixed literal written in this SDK.

    Build it only through :func:`_safe_error`: an instance constructed any other way
    (customer code, a subclass, a copy) is not trusted and :func:`is_safe` is False.
    """


def _safe_error(message: str, **kwargs: Any) -> SafeConfigurationError:
    """Build a trusted :class:`SafeConfigurationError`; ``message`` is a string literal."""

    return _trust(SafeConfigurationError(message, **kwargs))  # type: ignore[return-value]


def _mark_safe(error: ConfigurationError) -> ConfigurationError:
    """Trust ``error`` as carrying a fixed, literal message (see :func:`fixed_reason`)."""

    return _trust(error)


def is_safe(exc: BaseException) -> bool:
    """True only for an instance registered by :func:`_safe_error` / :func:`refusal`."""

    ref = _TRUSTED.get(id(exc))
    return ref is not None and ref() is exc and isinstance(exc, ConfigurationError)


def fixed_reason(exc: BaseException) -> str:
    """Log-safe reason: a SDK refusal's fixed message, else only the exception type.

    Only an error built by :func:`refusal` (registered as trusted) contributes its
    message; any other exception, including a plain ``ConfigurationError`` raised by a
    dependency (its text may interpolate customer content), contributes its type name.
    """

    if is_safe(exc):
        return exc.message  # type: ignore[attr-defined]
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
        if is_safe(cause) and isinstance(inner, dict):
            # a refusal wrapping a refusal keeps the original exception type name
            payload["error_type"] = inner.get("error_type", type(cause).__name__)
        else:
            payload["error_type"] = type(cause).__name__
    error = error_cls(message, details=payload)
    error.__cause__ = None
    error.__suppress_context__ = True
    return _mark_safe(error)


@contextmanager
def fs_guard(message: str) -> Iterator[None]:
    """Turn any ``OSError`` or decode failure in the block into a fixed refusal.

    ``OSError`` text quotes the path (which carries customer-chosen components) and a
    ``UnicodeDecodeError`` holds the undecodable bytes of the customer file in ``args``,
    so neither is printed or chained. ``message`` is a literal.
    """

    error: ConfigurationError | None = None
    try:
        yield
    except (OSError, UnicodeDecodeError) as exc:
        error = refusal(message, category=Category.FILESYSTEM_FAILED, cause=exc)
    if error is not None:
        # raised outside the ``except`` block: inside it the original exception would
        # be chained as ``__context__``, carrying the path / undecodable bytes
        # (contextlib throws the original into this generator, so the interpreter links
        # it again on ``raise``; clear it after that link and re-raise unchanged)
        try:
            raise error from None
        except ConfigurationError as raised:
            raised.__cause__ = None
            raised.__context__ = None
            raised.__suppress_context__ = True
            raise


_F = TypeVar("_F", bound=Callable[..., Any])


def boundary(
    category: Category,
    message: str,
    *,
    convert: Callable[[BaseException], ConfigurationError] | None = None,
) -> Callable[[_F], _F]:
    """Decorate a PUBLIC best-config entry point with the one diagnostic boundary.

    Whatever leaves the wrapped function is either a trusted refusal (:func:`is_safe`,
    passed through unchanged apart from dropping any chained cause or context) or a fixed
    refusal built here: ``message`` (a literal), ``category`` and the inner exception
    *type name* on ``details`` only. This closes the class the per-shape fixes kept
    reopening: an exception nobody anticipated (a dependency's, a ``__repr__`` or
    ``__str__`` of a customer value, a decoder's) cannot carry customer text out through
    ``str`` / ``repr`` / ``args`` / ``__cause__`` / ``__context__``.

    ``convert`` builds the refusal when callers rely on a different class on that path
    (it must return a trusted error and must not interpolate the exception text).
    Only ``Exception`` is converted; ``KeyboardInterrupt``, ``SystemExit`` and
    ``GeneratorExit`` pass untouched. Fail-open paths inside the function (those that
    return ``None`` after a fixed log line) are unaffected: only what escapes is wrapped.
    """

    def decorate(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            error: BaseException
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if is_safe(exc):
                    error = exc
                else:
                    try:
                        error = (
                            convert(exc)
                            if convert is not None
                            else refusal(message, category=category, cause=exc)
                        )
                    except Exception:  # noqa: BLE001 - the converter must not leak
                        error = refusal(message, category=category)
            # raise outside the ``except`` block: inside it the original exception would
            # be chained as ``__context__``
            error.__cause__ = None
            error.__context__ = None
            error.__suppress_context__ = True
            raise error from None

        return wrapper  # type: ignore[return-value]

    return decorate


def transport_failed_message(operation: str, exc: BaseException) -> str:
    """Fixed transport-failure text: the operation (a literal) and the exception type."""

    return f"{operation} transport failed ({type(exc).__name__})"


__all__ = [
    "Category",
    "boundary",
    "fixed_reason",
    "fs_guard",
    "is_safe",
    "SafeConfigurationError",
    "log",
    "refusal",
    "transport_failed_message",
]
