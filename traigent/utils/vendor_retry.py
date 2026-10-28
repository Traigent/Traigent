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
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast

from traigent.core.exception_handler import VendorErrorCategory, classify_vendor_error
from traigent.utils.exceptions import ConfigurationError
from traigent.utils.langchain_interceptor import (
    VendorAttemptHandle,
    begin_vendor_call,
    finish_vendor_attempt,
    open_vendor_attempt,
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


def _retry_after_header_seconds(headers: Any) -> float | None:
    """Seconds named by one ``retry-after`` header (number or HTTP-date)."""
    raw = _header(headers, "retry-after")
    seconds = _number(raw)
    if seconds is not None:
        return seconds
    if not (isinstance(raw, str) and raw.strip()):
        return None
    try:
        when = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


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
        wait = _retry_after_header_seconds(headers)
        if wait is not None:
            return wait
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
    return float(min(base_backoff * (2**retry_index), MAX_RETRY_DELAY_SECONDS))


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


def _open(call_index: int, attempt: int) -> VendorAttemptHandle:
    """Count the attempt now; it is updated in place when it finishes."""
    return open_vendor_attempt(
        {
            "call_index": call_index,
            "attempt": attempt,
            "error_type": None,
            "retried": False,
            "backoff_seconds": None,
            "usage": None,
            "usage_known": False,
        }
    )


def _finish(
    handle: VendorAttemptHandle,
    outcome: str,
    *,
    error: BaseException | None = None,
    backoff_seconds: float | None = None,
    usage: dict[str, int] | None = None,
) -> None:
    finish_vendor_attempt(
        handle,
        outcome=outcome,
        error_type=_error_type(error) if error is not None else None,
        retried=backoff_seconds is not None,
        backoff_seconds=backoff_seconds,
        usage=usage,
        # Unknown, not zero: a failed attempt reports no usage.
        usage_known=usage is not None,
    )


def _fault_hook(attempt_index: int) -> None:
    from traigent.testing.vendor_faults import maybe_raise_injected_failure

    maybe_raise_injected_failure(attempt_index)


class _AttemptRecordingStream:
    """Proxy a provider stream and record its FINAL outcome on its attempt.

    The attempt was counted when the stream was opened (``in_progress``). Its
    final outcome is ``success`` (exhausted), ``error`` (raised mid-stream,
    never retried), ``closed_early`` (``close``/``aclose``) or ``cancelled``.
    A stream that is simply abandoned (for example the consumer is cancelled
    between pulls) stays ``in_progress`` and is reported ``incomplete`` with
    unknown usage when the trial's metadata is taken.
    """

    def __init__(self, stream: Any, handle: VendorAttemptHandle) -> None:
        self._stream = stream
        self._handle = handle
        self._usage: dict[str, int] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _observe(self, chunk: Any) -> None:
        usage = _usage_of(chunk)
        if usage:
            self._usage = usage

    def _finish(self, outcome: str, error: BaseException | None = None) -> None:
        _finish(self._handle, outcome, error=error, usage=self._usage)

    def close(self) -> Any:
        try:
            closer = getattr(self._stream, "close", None)
            return closer() if callable(closer) else None
        finally:
            self._finish("closed_early")

    async def aclose(self) -> Any:
        try:
            closer = getattr(self._stream, "aclose", None)
            if callable(closer):
                return await closer()
            return None
        finally:
            self._finish("closed_early")

    def __iter__(self) -> _AttemptRecordingStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finish("success")
            raise
        except Exception as exc:
            self._finish("error", exc)
            raise
        except BaseException:
            self._finish("cancelled")
            raise
        self._observe(chunk)
        return chunk

    def __aiter__(self) -> _AttemptRecordingStream:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._finish("success")
            raise
        except Exception as exc:
            self._finish("error", exc)
            raise
        except BaseException:  # asyncio.CancelledError and friends
            self._finish("cancelled")
            raise
        self._observe(chunk)
        return chunk


def _wrap_stream(response: Any, handle: VendorAttemptHandle, *, is_async: bool) -> Any:
    iterable = (
        hasattr(response, "__anext__") if is_async else hasattr(response, "__next__")
    )
    if not iterable:
        # Not a stream we can observe: the open is the outcome.
        _finish(handle, "success", usage=_usage_of(response))
        return response
    return _AttemptRecordingStream(response, handle)


def _on_failure(
    exc: Exception,
    settings: VendorRetrySettings,
    attempt: int,
    handle: VendorAttemptHandle,
) -> float | None:
    """Return the wait before retrying, or ``None`` to propagate ``exc``."""
    category = classify_retryable(exc)
    delay = (
        compute_delay(exc, settings.base_backoff_seconds, attempt - 1)
        if category is not None and attempt <= settings.max_retries
        else None
    )
    if delay is None or category is None:
        _finish(handle, "error", error=exc)
        return None
    _finish(handle, "error", error=exc, backoff_seconds=delay)
    logger.warning(
        "traigent.vendor_call_retry category=%s retry=%s/%s backoff=%.2fs",
        category.value,
        attempt,
        settings.max_retries,
        delay,
    )
    return delay


def run_with_vendor_retry[T](call: Callable[[], T], *, stream: bool = False) -> T:
    """Run a single provider call with the per-call retry budget (sync).

    With ``stream=True`` only failures raised before the stream object is
    returned are retried; the returned stream records its own final outcome.
    """
    settings = resolve_vendor_retry_settings()
    call_index = begin_vendor_call()
    attempt = 0
    while True:
        attempt += 1
        handle = _open(call_index, attempt)
        try:
            _fault_hook(attempt - 1)
            response = call()
        except Exception as exc:
            delay = _on_failure(exc, settings, attempt, handle)
            if delay is None:
                raise
            if delay > 0:
                time.sleep(delay)
            continue
        except BaseException:
            _finish(handle, "cancelled")
            raise
        if stream:
            return cast(T, _wrap_stream(response, handle, is_async=False))
        _finish(handle, "success", usage=_usage_of(response))
        return response


async def run_with_vendor_retry_async[T](
    call: Callable[[], Awaitable[T]], *, stream: bool = False
) -> T:
    """Run a single provider call with the per-call retry budget (async)."""
    settings = resolve_vendor_retry_settings()
    call_index = begin_vendor_call()
    attempt = 0
    while True:
        attempt += 1
        handle = _open(call_index, attempt)
        try:
            _fault_hook(attempt - 1)
            response = await call()
        except Exception as exc:
            delay = _on_failure(exc, settings, attempt, handle)
            if delay is None:
                raise
            if delay > 0:
                await asyncio.sleep(delay)
            continue
        except BaseException:
            _finish(handle, "cancelled")
            raise
        if stream:
            return cast(T, _wrap_stream(response, handle, is_async=True))
        _finish(handle, "success", usage=_usage_of(response))
        return response


def summarize_bucket(bucket: Any) -> dict[str, Any]:
    """Trial/run metadata from the attempt ledger (empty if none).

    Totals come from independent counters, so they stay exact even when the
    record list was capped (``vendor_call_attempt_records_truncated``). An
    attempt still ``in_progress`` is reported as ``incomplete`` with unknown
    usage and counted in ``vendor_call_incomplete_attempts``.
    """
    root = bucket.root or bucket
    with root.lock:
        total = root.vendor_attempt_total
        if not total:
            return {}
        records = []
        for rec in root.vendor_attempts:
            out = dict(rec)
            if out.get("outcome") == "in_progress":
                out["outcome"] = "incomplete"
            records.append(out)
        unknown = total - root.vendor_known_usage_total
        # Attempts inside an example whose agent explicitly reported its total
        # cost are covered by that report (same rule as the evaluator).
        covered = sum(
            root.vendor_slot_attempts.get(slot, 0) - root.vendor_slot_known.get(slot, 0)
            for slot in root.vendor_reported_slots
        )
        return {
            "vendor_call_attempts": total,
            "vendor_call_retries": root.vendor_retry_total,
            "vendor_call_unknown_usage_attempts": unknown,
            "vendor_call_cost_reported_attempts": covered,
            "vendor_call_unmeasured_attempts": unknown - covered,
            "vendor_call_incomplete_attempts": total - root.vendor_done_total,
            "vendor_call_attempt_records": records,
            "vendor_call_attempt_records_truncated": bool(
                root.vendor_records_truncated
            ),
        }


def mark_if_agent_reported_cost(slot: int | None, output: Any) -> None:
    """Mark ``slot`` covered when ``output`` carries a valid reported total cost.

    Mirrors the evaluator, which treats a valid ``__traigent_meta__`` total as
    the example's measured spend and clears its ``unmeasured`` flag.
    """
    from traigent.core.meta_types import is_traigent_metadata
    from traigent.utils.langchain_interceptor import mark_example_cost_reported

    if slot is None or not isinstance(output, dict):
        return
    meta = output.get("__traigent_meta__")
    if isinstance(meta, dict) and "calls" in meta:
        meta = {k: v for k, v in meta.items() if k != "calls"}
    if meta is not None and is_traigent_metadata(meta):
        mark_example_cost_reported(slot)
