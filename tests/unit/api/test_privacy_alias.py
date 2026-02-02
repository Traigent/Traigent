"""Tests for execution mode handling in decorators.

The privacy and standard modes have been removed. This test verifies that
using them raises ConfigurationError with a clear message.
"""

from __future__ import annotations

import pytest

from traigent.api.decorators import optimize
from traigent.utils.exceptions import ConfigurationError


def test_decorator_privacy_mode_raises_configuration_error():
    """Test that privacy mode raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="No such mode"):

        @optimize(
            configuration_space={"x": [1]},
            objectives=["accuracy"],
            execution_mode="privacy",
        )
        def f(v: int) -> int:
            return v


def test_decorator_standard_mode_raises_configuration_error():
    """Test that standard mode raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="No such mode"):

        @optimize(
            configuration_space={"x": [1]},
            objectives=["accuracy"],
            execution_mode="standard",
        )
        def f(v: int) -> int:
            return v


def test_decorator_cloud_mode_raises_configuration_error():
    """Test that cloud mode raises ConfigurationError (not yet supported)."""
    with pytest.raises(ConfigurationError, match="not yet supported"):

        @optimize(
            configuration_space={"x": [1]},
            objectives=["accuracy"],
            execution_mode="cloud",
        )
        def f(v: int) -> int:
            return v


def test_decorator_hybrid_mode_raises_configuration_error():
    """Test that hybrid mode raises ConfigurationError (not yet supported)."""
    with pytest.raises(ConfigurationError, match="not yet supported"):

        @optimize(
            configuration_space={"x": [1]},
            objectives=["accuracy"],
            execution_mode="hybrid",
        )
        def f(v: int) -> int:
            return v


def test_decorator_edge_analytics_mode_works():
    """Test that edge_analytics mode (default) works correctly."""

    @optimize(
        configuration_space={"x": [1]},
        objectives=["accuracy"],
        execution_mode="edge_analytics",
    )
    def f(v: int) -> int:
        return v

    # Basic property checks
    assert hasattr(f, "optimize")
    assert f.execution_mode == "edge_analytics"
