"""Live policy for SDK-owned telemetry delivery."""

import os
from contextvars import Context, ContextVar

from traigent.utils.env_config import is_backend_offline, is_truthy

# Shared session and trial endpoints also serve optimization. Mark only this task's
# analytics calls, without changing the request payload or shared client state.
_usage_analytics_submission: ContextVar[bool] = ContextVar(
    "usage_analytics_submission", default=False
)
_telemetry_key_validation: ContextVar[bool] = ContextVar(
    "telemetry_key_validation", default=False
)


class _TelemetryAuthSuppressed(Exception):
    """Telemetry policy withheld validation, so no auth headers were issued."""


def _is_telemetry_key_validation() -> bool:
    return _telemetry_key_validation.get() or _usage_analytics_submission.get()


def _telemetry_key_validation_context() -> Context:
    """Carry just the telemetry marker to a validation worker."""
    context = Context()
    context.run(_telemetry_key_validation.set, True)
    return context


def may_send_telemetry() -> bool:
    """Return whether the current backend and telemetry settings allow delivery."""
    return not is_backend_offline() and not is_truthy(
        os.getenv("TRAIGENT_DISABLE_TELEMETRY")
    )
