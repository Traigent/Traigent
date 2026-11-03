"""Characterization tests for ``PlotGenerator._determine_objective_orientations``.

The orientation resolver is replaced by a recorder so the exact calls (names,
explicit values, order) are pinned together with the returned mapping.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.visualization import plots as plots_module
from traigent.visualization.plots import PlotGenerator


@pytest.fixture
def calls(monkeypatch):
    recorded: list[tuple[str, str | None]] = []

    def fake_resolve(name, explicit=None):
        recorded.append((name, explicit))
        value = explicit if explicit is not None else name
        return "minimize" if str(value).startswith("min") else "maximize"

    monkeypatch.setattr(plots_module, "resolve_objective_orientation", fake_resolve)
    return recorded


@pytest.fixture
def plotter() -> PlotGenerator:
    return PlotGenerator(use_matplotlib=False)


def _result(metadata):
    return SimpleNamespace(metadata=metadata)


def test_defaults_only_and_output_follows_requested_order(plotter, calls):
    out = plotter._determine_objective_orientations(
        _result({}), ["min_cost", "accuracy", "latency"]
    )
    assert list(out) == ["min_cost", "accuracy", "latency"]
    assert out == {"min_cost": False, "accuracy": True, "latency": True}
    assert calls == [("min_cost", None), ("accuracy", None), ("latency", None)]


def test_missing_metadata_is_treated_as_empty(plotter, calls):
    out = plotter._determine_objective_orientations(_result(None), ["accuracy"])
    assert out == {"accuracy": True}


def test_schema_object_with_objectives_attribute(plotter, calls):
    schema = SimpleNamespace(
        objectives=[
            SimpleNamespace(name="a", orientation="MINIMIZE"),
            SimpleNamespace(name="b", orientation="maximize"),
        ]
    )
    out = plotter._determine_objective_orientations(
        _result({"objective_schema": schema}), ["a", "b"]
    )
    assert out == {"a": False, "b": True}
    assert calls == [("a", "minimize"), ("b", "maximize")]


def test_schema_dict_with_dict_and_object_definitions(plotter, calls):
    schema = {
        "objectives": [
            {"name": "a", "orientation": "minimize"},
            SimpleNamespace(name="b", orientation="maximize"),
            {"name": "c"},
            {"orientation": "maximize"},
            {"name": "", "orientation": "maximize"},
            {"name": "d", "orientation": ""},
        ]
    }
    out = plotter._determine_objective_orientations(
        _result({"objective_schema": schema}), ["a", "b", "c", "d"]
    )
    assert out == {"a": False, "b": True, "c": True, "d": True}
    # c and d were skipped by the schema loop and fell back to defaults.
    assert calls == [
        ("a", "minimize"),
        ("b", "maximize"),
        ("c", None),
        ("d", None),
    ]


def test_band_orientation_is_skipped_then_defaulted(plotter, calls):
    schema = {"objectives": [{"name": "x", "orientation": "Band"}]}
    out = plotter._determine_objective_orientations(
        _result({"objective_schema": schema}), ["x"]
    )
    assert out == {"x": True}
    assert calls == [("x", None)]


@pytest.mark.parametrize("schema", ["text", 5, [], {"nope": 1}, {"objectives": []}])
def test_unusable_schema_is_ignored(plotter, calls, schema):
    out = plotter._determine_objective_orientations(
        _result({"objective_schema": schema}), ["a"]
    )
    assert out == {"a": True}
    assert calls == [("a", None)]


def test_object_without_name_attribute_uses_dict_lookup_fallback(plotter, calls):
    class Weird(dict):
        orientation = None

    definition = Weird(name="w", orientation="minimize")
    out = plotter._determine_objective_orientations(
        _result({"objective_schema": {"objectives": [definition]}}), ["w"]
    )
    assert out == {"w": False}


def test_metadata_orientations_strings_lowercased_and_non_strings_coerced(
    plotter, calls
):
    meta = {
        "objective_orientations": {
            "a": "MINIMIZE",
            "b": "maximize",
            "c": 0,
            "d": 1,
            "e": None,
        }
    }
    out = plotter._determine_objective_orientations(
        _result(meta), ["a", "b", "c", "d", "e"]
    )
    assert out == {"a": False, "b": True, "c": False, "d": True, "e": False}
    assert calls == [("a", "minimize"), ("b", "maximize")]


def test_schema_entries_are_not_overridden_by_metadata_orientations(plotter, calls):
    schema = {"objectives": [{"name": "a", "orientation": "minimize"}]}
    meta = {
        "objective_schema": schema,
        "objective_orientations": {"a": "maximize", "b": "minimize"},
    }
    out = plotter._determine_objective_orientations(_result(meta), ["a", "b"])
    assert out == {"a": False, "b": False}
    assert calls == [("a", "minimize"), ("b", "minimize")]


def test_non_dict_metadata_orientations_ignored(plotter, calls):
    out = plotter._determine_objective_orientations(
        _result({"objective_orientations": ["a"]}), ["a"]
    )
    assert out == {"a": True}
    assert calls == [("a", None)]


def test_extra_declared_objectives_are_dropped_from_result(plotter, calls):
    meta = {"objective_orientations": {"a": "maximize", "extra": "minimize"}}
    out = plotter._determine_objective_orientations(_result(meta), ["a"])
    assert out == {"a": True}
    assert calls == [("a", "maximize"), ("extra", "minimize")]


def test_resolver_errors_propagate(plotter, monkeypatch):
    def boom(name, explicit=None):
        raise ValueError(f"has no declared orientation: {name}")

    monkeypatch.setattr(plots_module, "resolve_objective_orientation", boom)
    with pytest.raises(ValueError, match="has no declared orientation: zzz"):
        plotter._determine_objective_orientations(_result({}), ["zzz"])
