"""Removed budget_* aliases refuse with a message that names the replacement.

Traigent#1370 Item 9: budget_limit / budget_metric / budget_include_pruned were
removed in 0.13.0. Both the decorator and the runtime ``.optimize()`` path must
refuse them by name, not with the generic "unknown keyword" message.
"""

from __future__ import annotations

import asyncio

import pytest

from traigent.api.decorators import optimize
from traigent.core.optimized_function import OptimizedFunction

REMOVED_ALIASES = {
    "budget_limit": (
        "budget_limit parameter was removed in 0.13.0",
        "cost_limit=<USD>",
    ),
    "budget_metric": (
        "budget_metric parameter was removed in 0.13.0",
        "metric_name=<name>",
    ),
    "budget_include_pruned": (
        "budget_include_pruned parameter was removed in 0.13.0",
        "metric_include_pruned=<bool>",
    ),
    "strategy": ("strategy parameter was removed", "algorithm=<optimizer>"),
}


def _target():
    def fn(x: int) -> int:
        return x

    return optimize(configuration_space={"a": [1, 2]})(fn)


@pytest.mark.parametrize("name", sorted(REMOVED_ALIASES))
def test_decorator_refuses_removed_alias_naming_replacement(name: str) -> None:
    def fn(x: int) -> int:
        return x

    with pytest.raises(TypeError) as exc_info:
        optimize(configuration_space={"a": [1, 2]}, **{name: 1})(fn)

    fragment, replacement = REMOVED_ALIASES[name]
    message = str(exc_info.value)
    assert fragment in message
    assert replacement in message
    assert "Unknown keyword" not in message


@pytest.mark.parametrize("name", sorted(REMOVED_ALIASES))
def test_runtime_optimize_refuses_removed_alias_naming_replacement(name: str) -> None:
    target = _target()

    with pytest.raises(TypeError) as exc_info:
        asyncio.run(target.optimize(**{name: 1}))

    fragment, replacement = REMOVED_ALIASES[name]
    message = str(exc_info.value)
    assert fragment in message
    assert replacement in message
    assert "Unknown keyword" not in message


NONE_VALUED_ALIASES = sorted(REMOVED_ALIASES)


class _StopAfterValidation(Exception):
    """Raised by the spy once the real ``_prepare_algorithm_kwargs`` has returned."""


def _spy_real_prepare(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[object]]:
    """Wrap the real preparation so its refusals run, then stop the run deterministically.

    ``calls`` records each call's input kwargs and merged output. A regression
    raises ``TypeError`` from the real checks and propagates; any other earlier
    failure also propagates, so nothing is swallowed.
    """
    calls: dict[str, list[object]] = {"input": [], "merged": []}
    real_prepare = OptimizedFunction._prepare_algorithm_kwargs

    def spy_prepare(self: OptimizedFunction, kwargs: dict[str, object]) -> object:
        calls["input"].append(dict(kwargs))
        merged = real_prepare(self, kwargs)
        calls["merged"].append(dict(merged))  # type: ignore[call-overload]
        raise _StopAfterValidation

    monkeypatch.setattr(OptimizedFunction, "_prepare_algorithm_kwargs", spy_prepare)
    return calls


@pytest.mark.parametrize("name", NONE_VALUED_ALIASES)
def test_decorator_treats_none_valued_removed_alias_as_absent(name: str) -> None:
    """``@optimize(<alias>=None)`` builds the decorator like an absent alias."""

    def fn(x: int) -> int:
        return x

    wrapped = optimize(configuration_space={"a": [1, 2]}, **{name: None})(fn)

    assert hasattr(wrapped, "optimize")
    assert hasattr(wrapped, "optimize_sync")


@pytest.mark.parametrize("name", NONE_VALUED_ALIASES)
def test_runtime_optimize_treats_none_valued_removed_alias_as_absent(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``.optimize(<alias>=None)`` reaches the real preparation without the alias."""
    calls = _spy_real_prepare(monkeypatch)
    target = _target()

    with pytest.raises(_StopAfterValidation):
        asyncio.run(target.optimize(**{name: None}))

    assert len(calls["input"]) == 1
    assert name not in calls["input"][0]


@pytest.mark.parametrize("name", NONE_VALUED_ALIASES)
def test_optimize_sync_treats_none_valued_removed_alias_as_absent(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``.optimize_sync(<alias>=None)`` reaches the real preparation without the alias."""
    calls = _spy_real_prepare(monkeypatch)
    target = _target()

    with pytest.raises(_StopAfterValidation):
        target.optimize_sync(**{name: None})

    assert len(calls["input"]) == 1
    assert name not in calls["input"][0]


def test_constructor_none_valued_aliases_are_not_stored_or_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct ``OptimizedFunction(...)`` with None aliases stores none of them.

    The merged kwargs that reach validation at ``.optimize()`` must not carry
    them either, or the name-only refusal in the optimization pipeline fires.
    """

    def fn(x: int) -> int:
        return x

    opt = OptimizedFunction(
        func=fn,
        configuration_space={"a": [1, 2]},
        objectives=["accuracy"],
        budget_limit=None,
        strategy=None,
    )
    stored = opt._decorator_runtime_overrides
    assert "budget_limit" not in stored
    assert "strategy" not in stored

    calls = _spy_real_prepare(monkeypatch)
    with pytest.raises(_StopAfterValidation):
        asyncio.run(opt.optimize())

    assert len(calls["merged"]) == 1
    merged = calls["merged"][0]
    assert isinstance(merged, dict)
    assert "budget_limit" not in merged
    assert "strategy" not in merged
