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
from contextvars import ContextVar
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
#: Context-local: tasks and threads that did not inherit the ``with`` block's
#: context are unaffected. Contexts copied FROM inside the block (asyncio tasks
#: created there, ``copy_context().run`` workers) share the one locked plan.
_active: ContextVar[_Plan | None] = ContextVar(
    "traigent_vendor_fault_plan", default=None
)


def _refuse_in_production() -> None:
    # Fail-closed policy check: unset or unknown environments count as
    # production, so the hook only works when the environment is explicitly a
    # non-production one (development, test, local, stage, ...).
    from traigent.utils.env_config import treat_as_production_policy

    if treat_as_production_policy():
        raise RuntimeError(
            "inject_vendor_failures() is a test-only fault hook and is refused "
            "in production or when the environment is unset/unknown. Set "
            "ENVIRONMENT=development (or test) to use it."
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

    Refusal uses ``treat_as_production_policy``: it also refuses when no
    environment label is set, or the label is unknown. The hook is scoped to
    the current context (task/thread tree), not the whole process.
    """
    _refuse_in_production()
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be an integer >= 1")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    if _active.get() is not None:
        raise RuntimeError("inject_vendor_failures() is already active")
    token = _active.set(_Plan(kind, count, per_call, count))
    try:
        yield
    finally:
        _active.reset(token)


def maybe_raise_injected_failure(attempt_index: int) -> None:
    """Raise the injected error for this attempt, if one is due (internal).

    ``attempt_index`` is the 0-based attempt number within one provider call.
    A no-op unless :func:`inject_vendor_failures` is active.
    """
    plan = _active.get()
    if plan is None:
        return
    with _lock:
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
