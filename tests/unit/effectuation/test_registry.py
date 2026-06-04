"""Tests for effectuation strategy registration."""

from __future__ import annotations

from traigent.effectuation import STRATEGY_REGISTRY, get_strategy


def test_builtin_strategies_register_and_resolve_by_id() -> None:
    assert {"framework_param", "self_consistency"} <= set(STRATEGY_REGISTRY)
    assert get_strategy("framework_param") is STRATEGY_REGISTRY["framework_param"]
    assert get_strategy("self_consistency") is STRATEGY_REGISTRY["self_consistency"]


def test_builtin_strategy_supported_kinds_are_correct() -> None:
    assert get_strategy("framework_param").supported_kinds == {"value"}
    assert get_strategy("self_consistency").supported_kinds == {"cardinality"}
