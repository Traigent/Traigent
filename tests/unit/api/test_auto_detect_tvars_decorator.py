"""Tests for auto-detected tuned-variable configuration in @optimize."""

from __future__ import annotations

import pytest

from traigent.api.decorators import optimize
from traigent.core.optimized_function import OptimizedFunction


def test_auto_detect_tvars_apply_populates_configuration_space() -> None:
    @optimize(auto_detect_tvars_mode="apply")
    def sample_agent(query: str) -> str:
        temperature = 0.7  # noqa: F841
        model = "gpt-4o-mini"  # noqa: F841
        return f"answer: {query}"

    assert isinstance(sample_agent, OptimizedFunction)
    assert sample_agent.configuration_space is not None
    assert "temperature" in sample_agent.configuration_space
    assert "model" in sample_agent.configuration_space


def test_auto_detect_tvars_apply_respects_explicit_configuration_space() -> None:
    @optimize(
        configuration_space={"manual_param": [1, 2, 3]},
        auto_detect_tvars_mode="apply",
    )
    def sample_agent(query: str) -> str:
        temperature = 0.7  # noqa: F841
        return f"answer: {query}"

    assert isinstance(sample_agent, OptimizedFunction)
    assert sample_agent.configuration_space == {"manual_param": [1, 2, 3]}


def test_auto_detect_tvars_apply_include_filter() -> None:
    @optimize(
        auto_detect_tvars_mode="apply",
        auto_detect_tvars_include={"temperature"},
    )
    def sample_agent(query: str) -> str:
        temperature = 0.7  # noqa: F841
        model = "gpt-4o-mini"  # noqa: F841
        return f"answer: {query}"

    assert isinstance(sample_agent, OptimizedFunction)
    assert set(sample_agent.configuration_space) == {"temperature"}


def test_auto_detect_tvars_apply_exclude_filter() -> None:
    @optimize(
        auto_detect_tvars_mode="apply",
        auto_detect_tvars_exclude={"temperature"},
    )
    def sample_agent(query: str) -> str:
        temperature = 0.7  # noqa: F841
        model = "gpt-4o-mini"  # noqa: F841
        return f"answer: {query}"

    assert isinstance(sample_agent, OptimizedFunction)
    assert "temperature" not in sample_agent.configuration_space
    assert "model" in sample_agent.configuration_space


def test_auto_detect_tvars_bool_remains_suggest_only() -> None:
    with pytest.raises(ValueError, match="Configuration space cannot be empty"):

        @optimize(auto_detect_tvars=True)
        def sample_agent(query: str) -> str:
            temperature = 0.7  # noqa: F841
            return f"answer: {query}"


def test_auto_detect_tvars_mode_invalid_raises() -> None:
    with pytest.raises(TypeError, match="auto_detect_tvars_mode"):
        optimize(auto_detect_tvars_mode="invalid")
