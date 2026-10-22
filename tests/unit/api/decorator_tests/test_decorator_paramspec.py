"""Tests for issue #1445 — @optimize preserves call signature via ParamSpec/Generic.

These tests verify:
1. The decorated function is still callable with the original arguments (runtime).
2. OptimizedFunction is the wrapper type (not Any).
3. The .optimize() / .optimize_sync() method surface is still accessible.
4. ParamSpec/Generic annotations are present so type-checkers can thread the types.
"""

from __future__ import annotations

import inspect

import pytest  # noqa: F401  (used indirectly for monkeypatch fixture)

from traigent.api.decorators import optimize
from traigent.core.optimized_function import OptimizedFunction

# ---------------------------------------------------------------------------
# Runtime tests (no mypy required — verify behavior, not types at import time)
# ---------------------------------------------------------------------------


def test_decorated_function_is_optimized_function_instance():
    """@optimize returns an OptimizedFunction, not Any or the bare callable."""

    @optimize(configuration_space={"model": ["gpt-4", "gpt-3.5"]})
    def my_fn(question: str, temperature: float = 0.7) -> str:
        return f"answer to {question}"

    assert isinstance(my_fn, OptimizedFunction)


def test_decorated_function_callable_with_original_signature():
    """The wrapped __call__ must forward the original arguments correctly."""

    @optimize(configuration_space={"model": ["a", "b"]})
    def greet(name: str, formal: bool = False) -> str:
        return f"Good day, {name}." if formal else f"Hi, {name}!"

    # Called with positional + keyword args matching the original signature
    result = greet("Alice")
    assert result == "Hi, Alice!"

    result_formal = greet("Bob", formal=True)
    assert result_formal == "Good day, Bob."


def test_decorated_function_exposes_optimize_method():
    """The OptimizedFunction must expose the .optimize() coroutine method."""

    @optimize(configuration_space={"temp": [0.5, 1.0]})
    def sample(prompt: str) -> str:
        return prompt

    assert hasattr(sample, "optimize"), "OptimizedFunction must have .optimize"
    assert hasattr(sample, "optimize_sync"), (
        "OptimizedFunction must have .optimize_sync"
    )
    assert callable(sample.optimize)
    assert callable(sample.optimize_sync)


def test_optimized_function_class_is_generic():
    """OptimizedFunction must be Generic (carries its P and R type parameters)."""
    # Generic is verified by checking __class_getitem__ exists, which is how
    # Generic subclasses are subscripted (e.g. OptimizedFunction[P, R]).
    assert hasattr(OptimizedFunction, "__class_getitem__"), (
        "OptimizedFunction must be generic over [P, R] to support type subscripting"
    )


def test_class_type_params_are_paramspec_and_typevar():
    """OptimizedFunction's PEP 695 type parameters are (ParamSpec, TypeVar)."""
    from typing import ParamSpec, TypeVar

    params = OptimizedFunction.__type_params__
    assert len(params) == 2
    assert isinstance(params[0], ParamSpec), "first type parameter must be a ParamSpec"
    assert isinstance(params[1], TypeVar), "second type parameter must be a TypeVar"


def test_optimized_function_call_signature_uses_paramspec():
    """OptimizedFunction.__call__ must have *args/P.args and **kwargs/P.kwargs."""
    sig = inspect.signature(OptimizedFunction.__call__)
    params = list(sig.parameters.values())
    # self, *args, **kwargs
    assert len(params) == 3
    names = [p.name for p in params]
    assert "args" in names
    assert "kwargs" in names


def test_passthrough_when_disabled_still_callable(monkeypatch):
    """When Traigent is disabled, @optimize must still return a callable."""
    monkeypatch.setenv("TRAIGENT_DISABLED", "1")

    @optimize(configuration_space={"k": [1, 2]})
    def fn(x: int) -> int:
        return x * 2

    # Even in passthrough mode the decorated name must be callable
    assert callable(fn)
    assert fn(5) == 10


def test_decorated_function_preserves_name():
    """Decorated function must retain __name__ from the original."""

    @optimize(configuration_space={"a": [1, 2]})
    def my_unique_name(x: str) -> str:
        return x

    assert my_unique_name.__name__ == "my_unique_name"
