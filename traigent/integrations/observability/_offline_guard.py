"""Fail-closed offline gate for the third-party tracker integrations.

MLflow and W&B send run names, tags and tuned configuration values to a
user-configured tracking server. While offline mode is on (``TRAIGENT_OFFLINE_MODE``
or a run with ``offline=True``) no such request is made: the call raises
:class:`OfflineModeError`, the same behaviour as the Langfuse client. The check
runs per call so a toggle takes effect immediately.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def offline_guarded(operation: str) -> Callable[[F], F]:
    """Decorate a method that talks to a tracking server."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from traigent.utils.env_config import raise_if_backend_offline

            raise_if_backend_offline(operation)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
