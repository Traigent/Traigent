"""Per-call vendor retry for transient provider outages (SDK-41).

Mirrors traigent-js ``src/integrations/vendor-retry.ts``: the retry budget is
scoped to ONE provider call (never a whole trial), only rate-limit (429) and
service-unavailable (503) errors are retried, and the delay is the vendor's
``retry_after`` when it supplies one, else ``base * 2**n``.

Every attempt is recorded in the active trial's capture scope -- including
attempts that failed and were retried -- because a rejected attempt may still
have been billed. Failed attempts usually carry no usage; they are recorded
with ``usage=None`` / ``usage_known=False`` (unknown), never a faked zero.

Environment (strict; read at run start by the orchestrator):

* ``TRAIGENT_VENDOR_MAX_RETRIES``: integer >= 0 (unset/empty -> 0 = off).
* ``TRAIGENT_VENDOR_RETRY_BACKOFF``: finite float >= 0 (unset/empty -> 1.0).
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from email.utils import parsedate_to_datetime
from datetime import UTC, datetime
from dataclasses import dataclass
from typing import Any, TypeVar

from traigent.core.exception_handler import VendorErrorCategory, classify_vendor_error
from traigent.utils.exceptions import ConfigurationError
from traigent.utils.langchain_interceptor import (
    begin_vendor_call,
    record_vendor_attempt,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

MAX_RETRIES_ENV = "TRAIGENT_VENDOR_MAX_RETRIES"
BACKOFF_ENV = "TRAIGENT_VENDOR_RETRY_BACKOFF"
DEFAULT_BACKOFF_SECONDS = 1.0

#: Upper bound on any single wait, so a hostile/huge ``retry_after`` cannot
#: park a run for hours.
MAX_RETRY_DELAY_SECONDS = 120.0

RETRYABLE_CATEGORIES = frozenset(
    {VendorErrorCategory.RATE_LIMIT, VendorErrorCategory.SERVICE_UNAVAILABLE}
)

T = TypeVar("T")

_INT_RE = re.compile(r"^[+]?\d+$")


@dataclass(frozen=True)
class VendorRetrySettings:
    """Validated vendor-retry settings."""

    max_retries: int = 0
    base_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS


def resolve_vendor_retry_settings(
    env: Mapping[str, str] | None = None,
) -> VendorRetrySettings:
    """Parse the two vendor-retry variables strictly.

    Raises:
        ConfigurationError: a variable is set to anything but the accepted
            form. The message names the variable and the accepted form and
            echoes the offending (non-secret, numeric) value.
    """
    source = os.environ if env is None else env

    retries = 0
    raw = source.get(MAX_RETRIES_ENV)
    if raw is not None and raw.strip():
        text = raw.strip()
        if not _INT_RE.match(text):
            raise ConfigurationError(
                f"{MAX_RETRIES_ENV}={raw!r} is invalid: expected an integer "
                ">= 0 (for example 0, 1, 3). Unset it or use 0 to disable retries."
            )
        retries = int(text)

    backoff = DEFAULT_BACKOFF_SECONDS
    raw = source.get(BACKOFF_ENV)
    if raw is not None and raw.strip():
        try:
            backoff = float(raw.strip())
        except ValueError:
            backoff = math.nan
        if not math.isfinite(backoff) or backoff < 0:
            raise ConfigurationError(
                f"{BACKOFF_ENV}={raw!r} is invalid: expected a finite number "
                f">= 0 in seconds (for example 0, 0.5, 2). Unset it for the "
                f"default of {DEFAULT_BACKOFF_SECONDS}."
            )
    return VendorRetrySettings(retries, backoff)


_TERMINAL_MARKERS = (
    "insufficient_quota",
    "exceeded your current quota",
    "quota_exceeded",
    "exceeded your current billing",
    "billing hard limit",
    "insufficient_funds",
    "insufficient funds",
    "insufficient credits",
    "payment required",
)

#: Closed set for the ``error_type`` field of an attempt record.
ERROR_TYPES = frozenset(c.value for c in VendorErrorCategory) | {"other"}


def _http_status(exc: BaseException) -> int | None:
    """Structured HTTP status of an error, when it carries one."""
    candidates = [
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def classify_retryable(exc: BaseException) -> VendorErrorCategory | None:
    """Category when ``exc`` is a retryable outage (429/503), else ``None``.

    A structured HTTP status is authoritative: only 429 and 503 retry, whatever
    the message says. Message/class-name classification is used only when no
    status is available. Terminal quota/billing wording wins even over a 429:
    waiting does not clear an exhausted account.
    """
    if not isinstance(exc, Exception):
        return None
    try:
        message = str(exc).lower()
        if any(marker in message for marker in _TERMINAL_MARKERS):
            return None
        status = _http_status(exc)
        if status is not None:
            if status == 429:
                return VendorErrorCategory.RATE_LIMIT
            if status == 503:
                return VendorErrorCategory.SERVICE_UNAVAILABLE
            return None
        category = classify_vendor_error(exc)
        if category is None:
            name = type(exc).__name__.lower()
            if "ratelimit" in name:
                category = VendorErrorCategory.RATE_LIMIT
            elif "serviceunavailable" in name:
                category = VendorErrorCategory.SERVICE_UNAVAILABLE
    except Exception:  # noqa: BLE001 - classification must never mask the error
        return None
    return category if category in RETRYABLE_CATEGORIES else None


def _error_type(exc: BaseException) -> str:
    """Closed-vocabulary error label for an attempt record."""
    try:
        category = classify_retryable(exc)
        if category is None and isinstance(exc, Exception):
            category = classify_vendor_error(exc)
    except Exception:  # noqa: BLE001
        category = None
    return category.value if category is not None else "other"


def _header(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    try:
        value = headers.get(name)
        if value is not None:
            return value
        for key in list(headers.keys()):
            if str(key).lower() == name:
                return headers[key]
    except Exception:  # noqa: BLE001
        return None
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out >= 0 else None


def vendor_wait_seconds(exc: BaseException) -> float | None:
    """The wait the vendor asked for, in seconds, or ``None`` if it gave none.

    Reads ``retry-after-ms``, then a ``retry_after`` attribute, then
    ``retry-after`` (seconds or an HTTP-date) from ``exc.headers``,
    ``exc.response.headers`` and ``exc.litellm_response_headers``.
    """
    sources = [
        getattr(exc, "headers", None),
        getattr(getattr(exc, "response", None), "headers", None),
        getattr(exc, "litellm_response_headers", None),
        getattr(exc, "responseHeaders", None),
    ]
    for headers in sources:
        ms = _number(_header(headers, "retry-after-ms"))
        if ms is not None:
            return ms / 1000.0
    attr = _number(getattr(exc, "retry_after", None))
    if attr is not None:
        return attr
    for headers in sources:
        raw = _header(headers, "retry-after")
        seconds = _number(raw)
        if seconds is not None:
            return seconds
        if isinstance(raw, str) and raw.strip():
            try:
                when = parsedate_to_datetime(raw.strip())
            except (TypeError, ValueError):
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            return max(0.0, (when - datetime.now(UTC)).total_seconds())
    return None


def compute_delay(
    exc: BaseException, base_backoff: float, retry_index: int
) -> float | None:
    """Seconds to wait before the next attempt, or ``None`` to stop retrying.

    The vendor's requested wait wins; if it exceeds ``MAX_RETRY_DELAY_SECONDS``
    we stop (propagate) rather than retry earlier than the vendor allows. Our
    own exponential backoff is clamped to the cap.
    """
    requested = vendor_wait_seconds(exc)
    if requested is not None:
        return None if requested > MAX_RETRY_DELAY_SECONDS else requested
    return min(base_backoff * (2**retry_index), MAX_RETRY_DELAY_SECONDS)


def _usage_of(response: Any) -> dict[str, int] | None:
    usage = (
        response.get("usage")
        if isinstance(response, Mapping)
        else getattr(response, "usage", None)
    )
    if usage is None:
        return None
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = (
            usage.get(key) if isinstance(usage, Mapping) else getattr(usage, key, None)
        )
        if isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
    return out or None


def _record(
    *,
    call_index: int,
    attempt: int,
    outcome: str,
    error: BaseException | None = None,
    backoff_seconds: float | None = None,
    usage: dict[str, int] | None = None,
) -> None:
    record_vendor_attempt(
        {
            "call_index": call_index,
            "attempt": attempt,
            "outcome": outcome,
            "error_type": _error_type(error) if error is not None else None,
            "retried": backoff_seconds is not None,
            "backoff_seconds": backoff_seconds,
            "usage": usage,
            # Unknown, not zero: a failed attempt reports no usage.
            "usage_known": usage is not None,
        }
    )


def _fault_hook(attempt_index: int) -> None:
    from traigent.testing.vendor_faults import maybe_raise_injected_failure

    maybe_raise_injected_failure(attempt_index)


class _AttemptRecordingStream:
    """Proxy a provider stream and record its FINAL outcome on consumption.

    Opening a stream succeeds before any chunk exists; the attempt is only
    known to have worked once the stream is exhausted. A failure mid-stream is
    recorded as a failed attempt and is never retried (chunks were emitted).
    """

    def __init__(self, stream: Any, call_index: int, attempt: int) -> None:
        self._stream = stream
        self._call_index = call_index
        self._attempt = attempt
        self._usage: dict[str, int] | None = None
        self._done = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _observe(self, chunk: Any) -> None:
        usage = _usage_of(chunk)
        if usage:
            self._usage = usage

    def _finish(self, error: BaseException | None) -> None:
        if self._done:
            return
        self._done = True
        if error is None:
            _record(
                call_index=self._call_index,
                attempt=self._attempt,
                outcome="success",
                usage=self._usage,
            )
        else:
            _record(
                call_index=self._call_index,
                attempt=self._attempt,
                outcome="error",
                error=error,
            )

    def __iter__(self) -> _AttemptRecordingStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finish(None)
            raise
        except Exception as exc:
            self._finish(exc)
            raise
        self._observe(chunk)
        return chunk

    def __aiter__(self) -> _AttemptRecordingStream:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish(None)
            raise
        except Exception as exc:
            self._finish(exc)
            raise
        self._observe(chunk)
        return chunk


def _wrap_stream(
    response: Any, call_index: int, attempt: int, *, is_async: bool
) -> Any:
    iterable = (
        hasattr(response, "__anext__") if is_async else hasattr(response, "__next__")
    )
    if not iterable:
        # Not a stream we can observe: record the open as the outcome.
        _record(
            call_index=call_index,
            attempt=attempt,
            outcome="success",
            usage=_usage_of(response),
        )
        return response
    return _AttemptRecordingStream(response, call_index, attempt)


def _on_failure(
    exc: Exception, settings: VendorRetrySettings, attempt: int, call_index: int
) -> float | None:
    """Return the wait before retrying, or ``None`` to propagate ``exc``."""
    category = classify_retryable(exc)
    delay = (
        compute_delay(exc, settings.base_backoff_seconds, attempt - 1)
        if category is not None and attempt <= settings.max_retries
        else None
    )
    if delay is None:
        _record(call_index=call_index, attempt=attempt, outcome="error", error=exc)
        return None
    _record(
        call_index=call_index,
        attempt=attempt,
        outcome="error",
        error=exc,
        backoff_seconds=delay,
    )
    assert category is not None
    logger.warning(
        "traigent.vendor_call_retry category=%s retry=%s/%s backoff=%.2fs",
        category.value,
        attempt,
        settings.max_retries,
        delay,
    )
    return delay


def run_with_vendor_retry(call: Callable[[], T], *, stream: bool = False) -> T:
    """Run a single provider call with the per-call retry budget (sync).

    With ``stream=True`` only failures raised before the stream object is
    returned are retried; the returned stream records its own final outcome.
    """
    settings = resolve_vendor_retry_settings()
    call_index = begin_vendor_call()
    attempt = 0
    while True:
        attempt += 1
        try:
            _fault_hook(attempt - 1)
            response = call()
        except Exception as exc:
            delay = _on_failure(exc, settings, attempt, call_index)
            if delay is None:
                raise
            if delay > 0:
                time.sleep(delay)
            continue
        if stream:
            return _wrap_stream(response, call_index, attempt, is_async=False)
        _record(
            call_index=call_index,
            attempt=attempt,
            outcome="success",
            usage=_usage_of(response),
        )
        return response


async def run_with_vendor_retry_async(
    call: Callable[[], Awaitable[T]], *, stream: bool = False
) -> T:
    """Run a single provider call with the per-call retry budget (async)."""
    settings = resolve_vendor_retry_settings()
    call_index = begin_vendor_call()
    attempt = 0
    while True:
        attempt += 1
        try:
            _fault_hook(attempt - 1)
            response = await call()
        except Exception as exc:
            delay = _on_failure(exc, settings, attempt, call_index)
            if delay is None:
                raise
            if delay > 0:
                await asyncio.sleep(delay)
            continue
        if stream:
            return _wrap_stream(response, call_index, attempt, is_async=True)
        _record(
            call_index=call_index,
            attempt=attempt,
            outcome="success",
            usage=_usage_of(response),
        )
        return response


def summarize_bucket(bucket: Any) -> dict[str, Any]:
    """Trial/run metadata from a capture bucket's attempt ledger (empty if none).

    Totals come from independent counters, so they stay exact even when the
    record list was capped (``vendor_call_attempt_records_truncated``).
    """
    with bucket.lock:
        total = bucket.vendor_attempt_total
        if not total:
            return {}
        return {
            "vendor_call_attempts": total,
            "vendor_call_retries": bucket.vendor_retry_total,
            "vendor_call_unknown_usage_attempts": bucket.vendor_unknown_usage_total,
            "vendor_call_attempt_records": [dict(a) for a in bucket.vendor_attempts],
            "vendor_call_attempt_records_truncated": bool(
                bucket.vendor_records_truncated
            ),
        }
