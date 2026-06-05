"""Regression tests for trace enablement environment handling."""

from __future__ import annotations

import pytest

from traigent.core.trace_env import (
    LEGACY_TRACES_ENABLED_ENV,
    TRACE_ENABLED_ENV,
    is_trace_enabled,
)


def test_trace_enabled_canonical_only_is_honored() -> None:
    assert is_trace_enabled({TRACE_ENABLED_ENV: "true"}) is True


def test_trace_enabled_alias_only_warns_and_is_honored() -> None:
    with pytest.warns(DeprecationWarning, match=LEGACY_TRACES_ENABLED_ENV):
        enabled = is_trace_enabled({LEGACY_TRACES_ENABLED_ENV: "true"})

    assert enabled is True


def test_trace_enabled_both_set_canonical_takes_precedence() -> None:
    with pytest.warns(DeprecationWarning, match="ignored"):
        enabled = is_trace_enabled(
            {
                TRACE_ENABLED_ENV: "false",
                LEGACY_TRACES_ENABLED_ENV: "true",
            }
        )

    assert enabled is False


def test_trace_enabled_neither_set_defaults_off() -> None:
    assert is_trace_enabled({}) is False
