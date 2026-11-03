"""Characterization for validation._validate_default_config_value dispatch."""

from __future__ import annotations

import logging
from typing import Any

import pytest

import traigent.utils.validation as v
from traigent.api.parameter_ranges import Choices, IntRange, Range


class _LoggingDict(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log: list[str] = []

    def get(self, key, default=None):
        self.log.append(key)
        return super().get(key, default)


@pytest.fixture
def calls(monkeypatch):
    rec: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(
        v, "_validate_default_choice", lambda *a, **k: rec.append(("choice", a, k))
    )
    monkeypatch.setattr(
        v, "_numeric_bounds_domain", lambda *a, **k: rec.append(("bounds", a, k))
    )
    return rec


def _run(value, domain, name="p"):
    v._validate_default_config_value(name, value, domain)


def test_list_and_tuple_dispatch(calls):
    _run(1, [1, 2])
    _run(1, (0, 3))
    _run(1, (0, 1, 2))
    _run(1, 5)
    _run(1, "abc")
    assert calls == [
        ("choice", ("p", 1, [1, 2]), {}),
        ("bounds", ("p", 1, 0, 3), {}),
    ]


def test_parameter_range_objects_are_converted(calls):
    _run(1, Choices(["a", "b"]))
    _run(0.5, Range(0.0, 1.0))
    _run(2, IntRange(0, 3))
    assert [c[0] for c in calls] == ["choice", "bounds", "bounds"]
    assert calls[0][1][2] == ["a", "b"]


@pytest.mark.parametrize("kind", ["categorical", "choice"])
def test_categorical_dict_prefers_choices_then_values(calls, kind):
    _run("x", {"type": kind, "choices": ["a"], "values": ["b"]})
    _run("x", {"type": kind, "values": ("b",)})
    _run("x", {"type": kind, "choices": [], "values": ["b"]})
    _run("x", {"type": kind})
    _run("x", {"type": kind, "choices": "abc"})
    assert calls == [
        ("choice", ("p", "x", ["a"]), {}),
        ("choice", ("p", "x", ("b",)), {}),
        ("choice", ("p", "x", ["b"]), {}),
    ]


def test_numeric_dict_bounds_aliases_step_and_integer(calls):
    _run(1, {"type": "float", "low": 0, "high": 1, "step": 0.5})
    _run(1, {"type": "int", "min": 0, "max": 3})
    _run(1, {"type": "integer", "low": 0, "min": 9, "high": 2, "max": 9, "step": 2})
    _run(1, {"type": "float", "low": 0, "high": 1, "log": True, "step": 0.5})
    _run(1, {"type": "float", "low": 0, "high": 1, "log": 0, "step": 0.5})
    assert calls == [
        ("bounds", ("p", 1, 0, 1), {"integer": False, "step": 0.5}),
        ("bounds", ("p", 1, 0, 3), {"integer": True, "step": None}),
        ("bounds", ("p", 1, 0, 2), {"integer": True, "step": 2}),
        ("bounds", ("p", 1, 0, 1), {"integer": False, "step": None}),
        ("bounds", ("p", 1, 0, 1), {"integer": False, "step": 0.5}),
    ]


def test_missing_bounds_pass_none(calls):
    _run(1, {"type": "float"})
    assert calls == [("bounds", ("p", 1, None, None), {"integer": False, "step": None})]


def test_unknown_or_missing_type_does_nothing(calls):
    _run(1, {"type": "weird"})
    _run(1, {})
    assert calls == []


def test_type_is_inferred_when_missing(calls):
    _run(1, {"low": 0, "high": 1})
    assert calls == [("bounds", ("p", 1, 0, 1), {"integer": False, "step": None})]


def test_dict_subclass_get_order_numeric(calls):
    d = _LoggingDict(type="float", low=0, high=1, step=1)
    _run(0.5, d)
    # inner fallbacks are evaluated first: get("low", get("min"))
    assert d.log == ["type", "min", "low", "max", "high", "log", "step"]


def test_dict_subclass_get_order_categorical(calls):
    d = _LoggingDict(type="categorical", choices=None, values=["x"])
    _run("x", d)
    assert d.log == ["type", "choices", "values"]
    d = _LoggingDict(type="categorical", choices=["x"], values=["y"])
    _run("x", d)
    assert d.log == ["type", "choices"]


def test_end_to_end_outside_space_warns_without_raising(caplog):
    with caplog.at_level(logging.WARNING):
        _run(9, [1, 2])
        _run(5, {"type": "float", "low": 0, "high": 1})
        _run(0.5, {"type": "float", "low": 0, "high": 1})
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "outside the choices" in messages[0]
    assert "outside the numeric range" in messages[1]
