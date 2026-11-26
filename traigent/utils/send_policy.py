"""Live policy for SDK-owned telemetry delivery."""

import os
from contextvars import ContextVar

from traigent.utils.env_config import is_backend_offline, is_truthy

# Shared session and trial endpoints also serve optimization. Mark only this task's
# analytics calls, without changing the request payload or shared client state.
_usage_analytics_submission: ContextVar[bool] = ContextVar(
    "usage_analytics_submission", default=False
)


def may_send_telemetry() -> bool:
    """Return whether the current backend and telemetry settings allow delivery."""
    return not is_backend_offline() and not is_truthy(
        os.getenv("TRAIGENT_DISABLE_TELEMETRY")
    )
