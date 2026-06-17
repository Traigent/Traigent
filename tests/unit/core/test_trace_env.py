"""Regression tests for trace enablement environment handling."""

from __future__ import annotations

from traigent.core.trace_env import TRACE_ENABLED_ENV, is_trace_enabled


def test_trace_enabled_canonical_only_is_honored() -> None:
    assert is_trace_enabled({TRACE_ENABLED_ENV: "true"}) is True


def test_trace_enabled_not_set_defaults_off() -> None:
    assert is_trace_enabled({}) is False


def test_trace_enabled_false_value_is_honored() -> None:
    assert is_trace_enabled({TRACE_ENABLED_ENV: "false"}) is False
