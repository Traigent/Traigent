"""Characterize objective and Pareto chart behavior before helper extraction."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from traigent.utils.optimization_analyzer import OptimizationAnalyzer


class TraceDict(dict):
    def __init__(self, *args, trace: list[tuple], label: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace
        self.label = label

    def get(self, key, default=None):
        self.trace.append((self.label, "get", key))
        return super().get(key, default)

    def __contains__(self, key):
        self.trace.append((self.label, "contains", key))
        return super().__contains__(key)

    def __getitem__(self, key):
        self.trace.append((self.label, "getitem", key))
        return super().__getitem__(key)


def _record(events: list[tuple], name: str, result=None):
    return MagicMock(
        side_effect=lambda *args, **kwargs: (
            events.append((name, args, kwargs)),
            result,
        )[1]
    )


@pytest.fixture
def analyzer(tmp_path):
    return OptimizationAnalyzer(base_path=tmp_path, file_version="2")


def test_default_objective_mapping_reads_and_pareto_chart_order(analyzer, monkeypatch):
    pytest.importorskip("matplotlib.pyplot")
    import matplotlib.pyplot as plt

    events: list[tuple] = []
    rows = [
        {"accuracy": 0.5, "latency": 10},
        {"accuracy": 0.8, "latency": 5},
        {"latency": 7},  # Missing the first objective short-circuits its second read.
        {"accuracy": 0.5, "latency": 10},  # Equal duplicates both remain on the front.
        {"accuracy": 0.4, "latency": 7},  # Dominated under maximize-both behavior.
        {"accuracy": float("nan"), "latency": 6},  # NaN comparisons retain this point.
    ]
    trials = [
        TraceDict(
            {"metrics": TraceDict(row, trace=events, label=f"m{index}")},
            trace=events,
            label=f"t{index}",
        )
        for index, row in enumerate(rows)
    ]
    objective0 = TraceDict(
        {"name": "accuracy", "direction": "maximize"},
        trace=events,
        label="obj0",
    )
    objectives_data = TraceDict(
        {"objectives": [objective0, "latency"]},
        trace=events,
        label="objectives_data",
    )
    data = TraceDict(
        {"trials": trials, "objectives": objectives_data},
        trace=events,
        label="data",
    )
    analyzer.load_run = MagicMock(
        side_effect=lambda *args, **kwargs: (
            events.append(("load_run", args, kwargs)),
            data,
        )[1]
    )

    fig, ax = MagicMock(name="figure"), MagicMock(name="axes")
    ax.scatter = _record(events, "scatter")
    ax.plot = _record(events, "plot")
    ax.set_xlabel = _record(events, "set_xlabel")
    ax.set_ylabel = _record(events, "set_ylabel")
    ax.set_title = _record(events, "set_title")
    ax.legend = _record(events, "legend")
    ax.grid = _record(events, "grid")
    subplots = _record(events, "subplots", result=(fig, ax))
    show = _record(events, "show")
    monkeypatch.setattr(plt, "subplots", subplots)
    monkeypatch.setattr(plt, "show", show)

    result = analyzer.plot_pareto_front("exp", "run")

    assert result is fig
    assert events[:5] == [
        ("load_run", ("exp", "run"), {}),
        ("data", "contains", "trials"),
        ("data", "getitem", "trials"),
        ("data", "get", "objectives"),
        ("objectives_data", "get", "objectives"),
    ]
    assert ("obj0", "get", "name") in events
    assert ("m2", "contains", "latency") not in events
    plot_events = events[-10:]
    assert [event[0] for event in plot_events] == [
        "subplots",
        "scatter",
        "scatter",
        "plot",
        "set_xlabel",
        "set_ylabel",
        "set_title",
        "legend",
        "grid",
        "show",
    ]
    assert plot_events[0][2] == {"figsize": (10, 8)}
    all_x, all_y = ax.scatter.call_args_list[0].args
    assert all_x[:4] == [0.5, 0.8, 0.5, 0.4] and math.isnan(all_x[4])
    assert all_y == [10, 5, 10, 7, 6]
    pareto_x, pareto_y = ax.scatter.call_args_list[1].args
    assert list(pareto_x[:3]) == [0.5, 0.5, 0.8] and math.isnan(pareto_x[3])
    assert list(pareto_y) == [10, 10, 5, 6]
    plot_x, plot_y = ax.plot.call_args_list[0].args[:2]
    assert list(plot_x[:3]) == [0.5, 0.5, 0.8] and math.isnan(plot_x[3])
    assert list(plot_y) == [10, 10, 5, 6]
    assert plot_events[1][2] == {"alpha": 0.5, "label": "All trials"}
    assert plot_events[2][2] == {
        "color": "red",
        "s": 100,
        "marker": "*",
        "label": "Pareto front",
        "zorder": 5,
    }
    assert plot_events[3][2] == {"alpha": 0.5, "linewidth": 1}
    assert plot_events[4:] == [
        ("set_xlabel", ("Accuracy",), {}),
        ("set_ylabel", ("Latency",), {}),
        ("set_title", ("Pareto Front: exp/run",), {}),
        ("legend", (), {}),
        ("grid", (True,), {"alpha": 0.3}),
        ("show", (), {}),
    ]


def test_explicit_objectives_skip_saved_objective_reads(analyzer, monkeypatch):
    pytest.importorskip("matplotlib.pyplot")
    import matplotlib.pyplot as plt

    trace: list[tuple] = []
    metrics = TraceDict({"quality": 1, "cost": 2}, trace=trace, label="metrics")
    trial = TraceDict({"metrics": metrics}, trace=trace, label="trial")
    data = TraceDict({"trials": [trial]}, trace=trace, label="data")
    monkeypatch.setattr(analyzer, "load_run", MagicMock(return_value=data))
    fig, ax = MagicMock(), MagicMock()
    monkeypatch.setattr(plt, "subplots", MagicMock(return_value=(fig, ax)))
    monkeypatch.setattr(plt, "show", MagicMock())

    assert (
        analyzer.plot_pareto_front(
            "exp", "run", objectives=["quality", "cost"], show=False
        )
        is fig
    )
    assert trace == [
        ("data", "contains", "trials"),
        ("data", "getitem", "trials"),
        ("trial", "get", "metrics"),
        ("metrics", "contains", "quality"),
        ("metrics", "contains", "cost"),
        ("metrics", "getitem", "quality"),
        ("metrics", "getitem", "cost"),
    ]


def test_insufficient_objectives_warns_before_trial_metric_reads(
    analyzer, monkeypatch, caplog
):
    data = {"trials": [{"metrics": {"quality": 1}}], "objectives": ["quality"]}
    monkeypatch.setattr(analyzer, "load_run", MagicMock(return_value=data))
    pyplot = pytest.importorskip("matplotlib.pyplot")
    subplots = MagicMock()
    monkeypatch.setattr(pyplot, "subplots", subplots)

    assert analyzer.plot_pareto_front("exp", "run", show=False) is None
    assert "Need at least 2 objectives for Pareto front" in caplog.text
    subplots.assert_not_called()


def test_comparison_exception_propagates_before_plot_creation(analyzer, monkeypatch):
    pytest.importorskip("matplotlib.pyplot")
    import matplotlib.pyplot as plt

    class ComparisonBomb:
        def __ge__(self, other):
            raise RuntimeError("first greater-or-equal comparison")

        def __gt__(self, other):
            raise AssertionError("greater-than must not run first")

    data = {
        "trials": [
            {"metrics": {"quality": ComparisonBomb(), "cost": 1}},
            {"metrics": {"quality": ComparisonBomb(), "cost": 2}},
        ]
    }
    monkeypatch.setattr(analyzer, "load_run", MagicMock(return_value=data))
    subplots = MagicMock()
    monkeypatch.setattr(plt, "subplots", subplots)

    with pytest.raises(RuntimeError, match="first greater-or-equal comparison"):
        analyzer.plot_pareto_front(
            "exp", "run", objectives=["quality", "cost"], show=False
        )
    subplots.assert_not_called()
