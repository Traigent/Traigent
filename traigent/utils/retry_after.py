"""Shared ``Retry-After`` parsing for the SDK's retry clients (#3011).

A server's ``Retry-After`` is either delta-seconds or an HTTP-date (RFC 9110).
``parse_retry_after`` returns the requested wait in seconds, finite and
non-negative, *without* clamping it: the caller compares it against its own
ceiling and remaining deadline and stops retrying (rather than retrying early)
when the server asks for longer than it can wait.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

#: Longest server-requested wait the SDK will sleep through before giving up.
MAX_RETRY_AFTER_SECONDS = 60.0

_DECIMAL_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def parse_retry_after(
    value: Any, *, now: Callable[[], datetime] | None = None
) -> float | None:
    """Seconds named by a ``Retry-After`` value, or ``None`` if absent/invalid."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if _DECIMAL_RE.fullmatch(text):
        seconds = float(text)
        return seconds if math.isfinite(seconds) else None
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    current = now() if now is not None else datetime.now(UTC)
    return max(0.0, (when - current).total_seconds())


def retry_after_from_headers(
    headers: Any, *, now: Callable[[], datetime] | None = None
) -> float | None:
    """``parse_retry_after`` applied to a header mapping (case-insensitive)."""
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Retry-After")
    if raw is None:
        raw = getter("retry-after")
    return parse_retry_after(raw, now=now)


def retry_after_from_error(exc: BaseException) -> float | None:
    """The wait an exception carries: ``retry_after`` attribute, else response header."""
    for attr in ("retry_after", "retry_after_seconds"):
        parsed = parse_retry_after(getattr(exc, attr, None))
        if parsed is not None:
            return parsed
    response = getattr(exc, "response", None)
    return retry_after_from_headers(getattr(response, "headers", None))
