"""Shared environment handling for Traigent trace enablement."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping

TRACE_ENABLED_ENV = "TRAIGENT_TRACE_ENABLED"
LEGACY_TRACES_ENABLED_ENV = "TRAIGENT_TRACES_ENABLED"

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

    ``TRAIGENT_TRACE_ENABLED`` is canonical and always takes precedence.
    ``TRAIGENT_TRACES_ENABLED`` is a deprecated alias that is honored only
    when the canonical variable is unset.
    """
    env = os.environ if environ is None else environ
    canonical_value = env.get(TRACE_ENABLED_ENV)
    legacy_value = env.get(LEGACY_TRACES_ENABLED_ENV)

    if legacy_value is not None:
        if canonical_value is None:
            message = (
                f"{LEGACY_TRACES_ENABLED_ENV} is deprecated; use "
                f"{TRACE_ENABLED_ENV} instead."
            )
        else:
            message = (
                f"{LEGACY_TRACES_ENABLED_ENV} is deprecated and ignored because "
                f"{TRACE_ENABLED_ENV} is set."
            )
        warnings.warn(message, DeprecationWarning, stacklevel=2)

    if canonical_value is not None:
        return _parse_trace_enabled(canonical_value, default=default)
    if legacy_value is not None:
        return _parse_trace_enabled(legacy_value, default=default)
    return default


__all__ = [
    "LEGACY_TRACES_ENABLED_ENV",
    "TRACE_ENABLED_ENV",
    "is_trace_enabled",
]
