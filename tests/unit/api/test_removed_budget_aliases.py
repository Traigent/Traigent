"""Removed budget_* aliases refuse with a message that names the replacement.

Traigent#1370 Item 9: budget_limit / budget_metric / budget_include_pruned were
removed in 0.13.0. Both the decorator and the runtime ``.optimize()`` path must
refuse them by name, not with the generic "unknown keyword" message.
"""

from __future__ import annotations

import asyncio

import pytest

from traigent.api.decorators import optimize

REMOVED_BUDGET_ALIASES = {
    "budget_limit": "cost_limit=<USD>",
    "budget_metric": "metric_name=<name>",
    "budget_include_pruned": "metric_include_pruned=<bool>",
}


def _target():
    def fn(x: int) -> int:
        return x

    return optimize(configuration_space={"a": [1, 2]})(fn)


@pytest.mark.parametrize("name", sorted(REMOVED_BUDGET_ALIASES))
def test_decorator_refuses_removed_budget_alias_naming_replacement(name: str) -> None:
    def fn(x: int) -> int:
        return x

    with pytest.raises(TypeError) as exc_info:
        optimize(configuration_space={"a": [1, 2]}, **{name: 1})(fn)

    message = str(exc_info.value)
    assert f"{name} parameter was removed in 0.13.0" in message
    assert REMOVED_BUDGET_ALIASES[name] in message
    assert "Unknown keyword" not in message


@pytest.mark.parametrize("name", sorted(REMOVED_BUDGET_ALIASES))
def test_runtime_optimize_refuses_removed_budget_alias_naming_replacement(
    name: str,
) -> None:
    target = _target()

    with pytest.raises(TypeError) as exc_info:
        asyncio.run(target.optimize(**{name: 1}))

    message = str(exc_info.value)
    assert f"{name} parameter was removed in 0.13.0" in message
    assert REMOVED_BUDGET_ALIASES[name] in message
    assert "Unknown keyword" not in message
