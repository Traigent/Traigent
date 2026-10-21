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
    """``.optimize(<alias>=None)`` must not reach the removed-alias or unknown-key checks.

    The spy wraps the real ``_prepare_algorithm_kwargs`` so the real checks run;
    a regression raises ``TypeError`` from them, which fails the test.
    """
    seen: dict[str, object] = {}
    real_prepare = OptimizedFunction._prepare_algorithm_kwargs

    def spy_prepare(self: OptimizedFunction, kwargs: dict[str, object]) -> object:
        seen.update(kwargs)
        return real_prepare(self, kwargs)

    monkeypatch.setattr(OptimizedFunction, "_prepare_algorithm_kwargs", spy_prepare)
    target = _target()

    try:
        asyncio.run(target.optimize(**{name: None}))
    except TypeError as exc:  # the regression: a removed-alias or unknown-key refusal
        pytest.fail(f"None-valued {name} was refused: {exc}")
    except Exception:  # noqa: BLE001 - later, unrelated validation may stop the run
        pass

    assert name not in seen


@pytest.mark.parametrize("name", NONE_VALUED_ALIASES)
def test_optimize_sync_treats_none_valued_removed_alias_as_absent(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    real_prepare = OptimizedFunction._prepare_algorithm_kwargs

    def spy_prepare(self: OptimizedFunction, kwargs: dict[str, object]) -> object:
        seen.update(kwargs)
        return real_prepare(self, kwargs)

    monkeypatch.setattr(OptimizedFunction, "_prepare_algorithm_kwargs", spy_prepare)
    target = _target()

    try:
        target.optimize_sync(**{name: None})
    except TypeError as exc:  # the regression: a removed-alias or unknown-key refusal
        pytest.fail(f"None-valued {name} was refused: {exc}")
    except Exception:  # noqa: BLE001 - later, unrelated validation may stop the run
        pass

    assert name not in seen
