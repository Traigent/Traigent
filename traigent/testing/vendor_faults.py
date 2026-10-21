"""Test-only fault injection for provider calls (SDK-41).

``inject_vendor_failures`` makes provider calls made through Traigent's LiteLLM
interceptor fail on purpose with a retryable 429/503 error, so a developer can
see how many attempts Traigent made and what they cost. It works in mock mode
and in real mode, refuses to activate in production, and changes nothing when
it is not active.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["inject_vendor_failures"]

_KINDS = ("rate_limit", "service_unavailable")


@dataclass
class _Plan:
    kind: str
    count: int
    per_call: bool
    remaining: int


_lock = threading.Lock()
_active: _Plan | None = None


def _refuse_in_production() -> None:
    from traigent.utils.env_config import is_production_strict_env

    if is_production_strict_env():
        raise RuntimeError(
            "inject_vendor_failures() is a test-only fault hook and is refused "
            "when ENVIRONMENT=production."
        )


@contextmanager
def inject_vendor_failures(
    count: int, kind: str = "rate_limit", per_call: bool = True
) -> Iterator[None]:
    """Make provider calls raise a retryable error ``count`` times, then succeed.

    Args:
        count: Number of injected failures (integer >= 1).
        kind: ``"rate_limit"`` (429) or ``"service_unavailable"`` (503).
        per_call: ``True`` -- every provider call fails ``count`` times before
            succeeding (so with per-call retries ``>= count`` each call
            recovers). ``False`` -- ``count`` is a single pool shared across
            all attempts while the context is active (e.g. exactly one failed
            attempt in a whole run).

    Injected errors are real ``RateLimitError`` / ``ServiceUnavailableError``
    instances, so they go through the same retry and trial-level handling as a
    genuine 429/503, and each is counted as an attempt.

    Raises:
        RuntimeError: when ``ENVIRONMENT=production``, or if already active.
        ValueError: for a bad ``count`` or ``kind``.
    """
    global _active
    _refuse_in_production()
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be an integer >= 1")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    with _lock:
        if _active is not None:
            raise RuntimeError("inject_vendor_failures() is already active")
        _active = _Plan(kind, count, per_call, count)
    try:
        yield
    finally:
        with _lock:
            _active = None


def maybe_raise_injected_failure(attempt_index: int) -> None:
    """Raise the injected error for this attempt, if one is due (internal).

    ``attempt_index`` is the 0-based attempt number within one provider call.
    A no-op unless :func:`inject_vendor_failures` is active.
    """
    plan = _active
    if plan is None:
        return
    with _lock:
        plan = _active
        if plan is None:
            return
        if plan.per_call:
            due = attempt_index < plan.count
        elif plan.remaining > 0:
            plan.remaining -= 1
            due = True
        else:
            due = False
        kind = plan.kind
    if not due:
        return
    _refuse_in_production()
    from traigent.utils.exceptions import RateLimitError, ServiceUnavailableError

    if kind == "rate_limit":
        raise RateLimitError(
            "Injected rate limit (status 429) from inject_vendor_failures"
        )
    raise ServiceUnavailableError(
        "Injected service unavailable (status 503) from inject_vendor_failures",
        status_code=503,
    )
