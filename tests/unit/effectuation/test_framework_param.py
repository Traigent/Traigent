"""Tests for the framework_param value strategy."""

from __future__ import annotations

from traigent.config_generator.catalog import load_catalog
from traigent.effectuation import get_strategy


def test_framework_param_projects_plain_value_set_for_optimizer() -> None:
    strategy = get_strategy("framework_param")
    effect = strategy.compile(
        {
            "name": "temperature",
            "kind": "value",
            "value_set": [0.0, 0.5, 1.0],
        }
    )

    assert effect.project_for_optimizer() == {"temperature": [0.0, 0.5, 1.0]}


def test_framework_param_wrap_callable_returns_same_callable() -> None:
    strategy = get_strategy("framework_param")
    effect = strategy.compile(
        {
            "name": "model",
            "kind": "value",
            "value_set": ["gpt-4o-mini", "gpt-4o"],
        }
    )

    def target() -> str:
        return "ok"

    assert effect.wrap_callable(target, plan={}) is target
    assert effect.emit_events() == []


def test_catalog_value_entries_are_only_marked_executable_when_strategy_exists() -> (
    None
):
    value_entries = [entry for entry in load_catalog() if entry["kind"] == "value"]

    for entry in value_entries:
        if entry.get("effectuation_strategy") == "framework_param":
            assert entry["effectuation_status"] == "executable"
        else:
            assert entry["effectuation_status"] != "executable"
