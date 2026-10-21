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


def classify_retryable(exc: BaseException) -> VendorErrorCategory | None:
    """Category when ``exc`` is a retryable outage (429/503), else ``None``.

    Terminal quota/billing wording wins even over a 429 status: waiting does
    not clear an exhausted account.
    """
    if not isinstance(exc, Exception):
        return None
    try:
        message = str(exc).lower()
        if any(marker in message for marker in _TERMINAL_MARKERS):
            return None
        category = classify_vendor_error(exc)
        if category is None:
            status = getattr(exc, "status_code", None)
            name = type(exc).__name__.lower()
            if status == 429 or "ratelimit" in name:
                category = VendorErrorCategory.RATE_LIMIT
            elif status == 503 or "serviceunavailable" in name:
                category = VendorErrorCategory.SERVICE_UNAVAILABLE
    except Exception:  # noqa: BLE001 - classification must never mask the error
        return None
    return category if category in RETRYABLE_CATEGORIES else None


def _retry_after_seconds(exc: BaseException) -> float | None:
    candidate = getattr(exc, "retry_after", None)
    if candidate is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                candidate = headers.get("retry-after")
            except Exception:  # noqa: BLE001
                candidate = None
    try:
        value = float(candidate) if candidate is not None else None
    except (TypeError, ValueError):
        return None
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return value


def compute_delay(exc: BaseException, base_backoff: float, retry_index: int) -> float:
    """``retry_after`` when supplied, else ``base * 2**retry_index`` (bounded)."""
    delay = _retry_after_seconds(exc)
    if delay is None:
        delay = base_backoff * (2**retry_index)
    return min(delay, MAX_RETRY_DELAY_SECONDS)


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
    category: VendorErrorCategory | None = None,
    backoff_seconds: float | None = None,
    response: Any = None,
) -> None:
    usage = _usage_of(response) if outcome == "success" else None
    record_vendor_attempt(
        {
            "call_index": call_index,
            "attempt": attempt,
            "outcome": outcome,
            "error_type": type(error).__name__ if error is not None else None,
            "error_category": category.value if category is not None else None,
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


def run_with_vendor_retry(call: Callable[[], T]) -> T:
    """Run a single provider call with the per-call retry budget (sync)."""
    settings = resolve_vendor_retry_settings()
    call_index = begin_vendor_call()
    attempt = 0
    while True:
        attempt += 1
        try:
            _fault_hook(attempt - 1)
            response = call()
        except Exception as exc:
            category = classify_retryable(exc)
            if category is None or attempt > settings.max_retries:
                _record(
                    call_index=call_index,
                    attempt=attempt,
                    outcome="error",
                    error=exc,
                    category=category,
                )
                raise
            delay = compute_delay(exc, settings.base_backoff_seconds, attempt - 1)
            _record(
                call_index=call_index,
                attempt=attempt,
                outcome="error",
                error=exc,
                category=category,
                backoff_seconds=delay,
            )
            logger.warning(
                "traigent.vendor_call_retry category=%s retry=%s/%s backoff=%.2fs",
                category.value,
                attempt,
                settings.max_retries,
                delay,
            )
            if delay > 0:
                time.sleep(delay)
            continue
        _record(
            call_index=call_index, attempt=attempt, outcome="success", response=response
        )
        return response


async def run_with_vendor_retry_async(call: Callable[[], Awaitable[T]]) -> T:
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
            category = classify_retryable(exc)
            if category is None or attempt > settings.max_retries:
                _record(
                    call_index=call_index,
                    attempt=attempt,
                    outcome="error",
                    error=exc,
                    category=category,
                )
                raise
            delay = compute_delay(exc, settings.base_backoff_seconds, attempt - 1)
            _record(
                call_index=call_index,
                attempt=attempt,
                outcome="error",
                error=exc,
                category=category,
                backoff_seconds=delay,
            )
            logger.warning(
                "traigent.vendor_call_retry category=%s retry=%s/%s backoff=%.2fs",
                category.value,
                attempt,
                settings.max_retries,
                delay,
            )
            if delay > 0:
                await asyncio.sleep(delay)
            continue
        _record(
            call_index=call_index, attempt=attempt, outcome="success", response=response
        )
        return response


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Trial/run metadata derived from attempt records (empty when none)."""
    if not attempts:
        return {}
    retries = sum(1 for a in attempts if a.get("retried"))
    return {
        "vendor_call_attempts": len(attempts),
        "vendor_call_retries": retries,
        "vendor_call_unknown_usage_attempts": sum(
            1 for a in attempts if not a.get("usage_known")
        ),
        "vendor_call_attempt_records": [dict(a) for a in attempts],
    }
