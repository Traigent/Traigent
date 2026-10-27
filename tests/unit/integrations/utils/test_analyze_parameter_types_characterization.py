"""Golden characterization for ParameterDiscovery.analyze_parameter_types."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from traigent.integrations.utils.discovery import ParameterDiscovery

P = inspect.Parameter


def _patch(monkeypatch, params, types):
    monkeypatch.setattr(
        ParameterDiscovery, "discover_init_parameters", staticmethod(lambda cls: params)
    )
    monkeypatch.setattr(
        ParameterDiscovery,
        "infer_parameter_type",
        staticmethod(lambda p: types[p]),
    )


def test_real_class_groups_by_type_and_category():
    class C:
        def __init__(self, a: str = "x", b: int = 1, c: bool = True, d: list = None):
            pass

    out = ParameterDiscovery.analyze_parameter_types(C)
    assert set(out) == {"by_type", "by_category"}
    # insertion order of names follows signature order within each bucket
    flat = [n for names in out["by_category"].values() for n in names]
    assert set(flat) <= {"a", "b", "c", "d"}
    assert out["by_type"] == {
        k: v for k, v in out["by_type"].items() if all(isinstance(n, str) for n in v)
    }


def test_categories_and_unnamed_types(monkeypatch):
    p = {k: object() for k in "abcdefghi"}
    types = {
        p["a"]: str,
        p["b"]: int,
        p["c"]: float,
        p["d"]: bool,
        p["e"]: list,
        p["f"]: dict,
        p["g"]: "number",  # no __name__ -> str()
        p["h"]: None,  # falsy -> skipped
        p["i"]: SimpleNamespace,  # other
    }
    _patch(monkeypatch, p, types)
    out = ParameterDiscovery.analyze_parameter_types(object)
    assert out == {
        "by_type": {
            "str": ["a"],
            "int": ["b"],
            "float": ["c"],
            "bool": ["d"],
            "list": ["e"],
            "dict": ["f"],
            "number": ["g"],
            "SimpleNamespace": ["i"],
        },
        "by_category": {
            "string": ["a"],
            "number": ["b", "c", "g"],
            "boolean": ["d"],
            "collection": ["e", "f"],
            "other": ["i"],
        },
    }
    assert list(out["by_category"]) == [
        "string",
        "number",
        "boolean",
        "collection",
        "other",
    ]


def test_same_type_collects_names(monkeypatch):
    p = {"x": object(), "y": object()}
    _patch(monkeypatch, p, {p["x"]: int, p["y"]: int})
    out = ParameterDiscovery.analyze_parameter_types(object)
    assert out["by_type"] == {"int": ["x", "y"]}
    assert out["by_category"] == {"number": ["x", "y"]}


@pytest.mark.parametrize("where", ["discover", "infer"])
def test_any_exception_returns_empty(monkeypatch, where):
    def boom(*_a, **_k):
        raise RuntimeError("x")

    if where == "discover":
        monkeypatch.setattr(
            ParameterDiscovery, "discover_init_parameters", staticmethod(boom)
        )
    else:
        monkeypatch.setattr(
            ParameterDiscovery,
            "discover_init_parameters",
            staticmethod(lambda cls: {"a": object()}),
        )
        monkeypatch.setattr(
            ParameterDiscovery, "infer_parameter_type", staticmethod(boom)
        )
    assert ParameterDiscovery.analyze_parameter_types(object) == {
        "by_type": {},
        "by_category": {},
    }


def test_partial_results_discarded_on_late_failure(monkeypatch):
    p = {"a": object(), "b": object()}

    def infer(param):
        if param is p["b"]:
            raise ValueError
        return int

    monkeypatch.setattr(
        ParameterDiscovery, "discover_init_parameters", staticmethod(lambda cls: p)
    )
    monkeypatch.setattr(ParameterDiscovery, "infer_parameter_type", staticmethod(infer))
    assert ParameterDiscovery.analyze_parameter_types(object) == {
        "by_type": {},
        "by_category": {},
    }
