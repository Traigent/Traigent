"""Shared environment handling for Traigent trace enablement."""

from __future__ import annotations

import os
from collections.abc import Mapping

TRACE_ENABLED_ENV = "TRAIGENT_TRACE_ENABLED"

_TRUTHY_TRACE_VALUES = frozenset(("true", "1", "yes"))
_FALSY_TRACE_VALUES = frozenset(("false", "0", "no"))


def _parse_trace_enabled(value: str, *, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUTHY_TRACE_VALUES:
        return True
    if normalized in _FALSY_TRACE_VALUES:
        return False
    return default


def is_trace_enabled(
    environ: Mapping[str, str] | None = None,
    *,
    default: bool = False,
) -> bool:
    """Return whether Traigent tracing is enabled by environment.

    ``TRAIGENT_TRACE_ENABLED`` is the canonical variable.
    """
    env = os.environ if environ is None else environ
    canonical_value = env.get(TRACE_ENABLED_ENV)

    if canonical_value is not None:
        return _parse_trace_enabled(canonical_value, default=default)
    return default


__all__ = [
    "TRACE_ENABLED_ENV",
    "is_trace_enabled",
]
