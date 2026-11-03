"""Characterize display-only optimization analyzer output and collaborator access."""

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

    def items(self):
        self.trace.append((self.label, "items"))
        return super().items()


@pytest.fixture
def analyzer(tmp_path):
    return OptimizationAnalyzer(base_path=tmp_path, file_version="2")


def test_plot_convergence_objective_selection_and_metric_access_trace(
    analyzer, monkeypatch
):
    pytest.importorskip("matplotlib.pyplot")
    import matplotlib.pyplot as plt

    trace: list[tuple] = []
    metrics = [
        TraceDict({"accuracy": float("nan"), "other": 0.2}, trace=trace, label="m0"),
        TraceDict({"accuracy": 0.4}, trace=trace, label="m1"),
        TraceDict({"other": 0.7}, trace=trace, label="m2"),
        TraceDict({"accuracy": 0.4}, trace=trace, label="m3"),
    ]
    trials = [
        TraceDict({"metrics": item}, trace=trace, label=f"t{index}")
        for index, item in enumerate(metrics)
    ]
    objective_list = TraceDict(
        {"name": "accuracy", "direction": "maximize"},
        trace=trace,
        label="objective0",
    )
    objectives = TraceDict(
        {"objectives": [objective_list]}, trace=trace, label="objectives"
    )
    data = TraceDict(
        {"trials": trials, "objectives": objectives}, trace=trace, label="data"
    )
    monkeypatch.setattr(analyzer, "load_run", MagicMock(return_value=data))
    fig = MagicMock(name="figure")
    ax = MagicMock(name="axes")
    monkeypatch.setattr(plt, "subplots", MagicMock(return_value=(fig, ax)))
    show = MagicMock()
    monkeypatch.setattr(plt, "show", show)

    result = analyzer.plot_convergence("exp", "run", show=False)

    assert result is fig
    assert trace == [
        ("data", "contains", "trials"),
        ("data", "getitem", "trials"),
        ("data", "get", "objectives"),
        ("objectives", "get", "objectives"),
        ("objective0", "get", "name"),
        ("t0", "get", "metrics"),
        ("m0", "contains", "accuracy"),
        ("m0", "getitem", "accuracy"),
        ("t1", "get", "metrics"),
        ("m1", "contains", "accuracy"),
        ("m1", "getitem", "accuracy"),
        ("t2", "get", "metrics"),
        ("m2", "contains", "accuracy"),
        ("t3", "get", "metrics"),
        ("m3", "contains", "accuracy"),
        ("m3", "getitem", "accuracy"),
    ]
    values = list(ax.plot.call_args_list[0].args[1])
    best_values = list(ax.plot.call_args_list[1].args[1])
    assert math.isnan(values[0]) and values[1:] == [0.4, 0.4]
    assert math.isnan(best_values[0]) and best_values[1:] == [0.4, 0.4]
    ax.set_ylabel.assert_called_once_with("Accuracy")
    show.assert_not_called()


def test_plot_convergence_explicit_objective_skips_selection_reads(
    analyzer, monkeypatch
):
    pytest.importorskip("matplotlib.pyplot")
    import matplotlib.pyplot as plt

    trace: list[tuple] = []
    metrics = TraceDict({"latency": 0}, trace=trace, label="metrics")
    trial = TraceDict({"metrics": metrics}, trace=trace, label="trial")
    data = TraceDict({"trials": [trial]}, trace=trace, label="data")
    monkeypatch.setattr(analyzer, "load_run", MagicMock(return_value=data))
    fig, ax = MagicMock(), MagicMock()
    monkeypatch.setattr(plt, "subplots", MagicMock(return_value=(fig, ax)))
    monkeypatch.setattr(plt, "show", MagicMock())

    assert (
        analyzer.plot_convergence("exp", "run", objective="latency", show=True) is fig
    )
    assert trace == [
        ("data", "contains", "trials"),
        ("data", "getitem", "trials"),
        ("trial", "get", "metrics"),
        ("metrics", "contains", "latency"),
        ("metrics", "getitem", "latency"),
    ]


def test_run_summary_exact_text_and_data_access_order(analyzer, monkeypatch):
    trace: list[tuple] = []
    session = TraceDict(
        {"session_id": "s-1", "status": "completed", "duration": 1.25},
        trace=trace,
        label="session",
    )
    best_metrics = {"accuracy": 0.87555, "count": 2}
    metrics = TraceDict(
        {
            "total_trials": 3,
            "successful_trials": 2,
            "failed_trials": 1,
            "success_rate": 2 / 3,
            "algorithm": "bayesian",
            "best_metrics": best_metrics,
        },
        trace=trace,
        label="metrics",
    )
    config = TraceDict(
        {"model": "m-1", "temperature": 0.3}, trace=trace, label="config"
    )
    best_config = TraceDict({"config": config}, trace=trace, label="best_config")
    weighted_config = TraceDict(
        {"model": "m-2", "temperature": 0.5}, trace=trace, label="weighted_config"
    )
    weighted = TraceDict(
        {
            "best_weighted_score": 0.81234,
            "best_weighted_config": weighted_config,
            "objective_weights": {"accuracy": 0.7, "latency": 0.3},
        },
        trace=trace,
        label="weighted",
    )
    data = TraceDict(
        {
            "session": session,
            "metrics_summary": metrics,
            "best_config": best_config,
            "weighted_results": weighted,
        },
        trace=trace,
        label="data",
    )
    monkeypatch.setattr(analyzer, "load_run", MagicMock(return_value=data))

    summary = analyzer.get_run_summary("exp", "run")

    assert summary == (
        "============================================================\n"
        "Optimization Run Summary\n"
        "============================================================\n"
        "Experiment: exp\n"
        "Run ID: run\n"
        "Session ID: s-1\n"
        "Status: completed\n"
        "Duration: 1.25 seconds\n"
        "\nTrials:\n"
        "  Total: 3\n"
        "  Successful: 2\n"
        "  Failed: 1\n"
        "  Success Rate: 66.7%\n"
        "\nAlgorithm: bayesian\n"
        "\nBest Metrics:\n"
        "  accuracy: 0.8756\n"
        "  count: 2\n"
        "\nBest Configuration:\n"
        "  model: m-1\n"
        "  temperature: 0.3\n"
        "\nWeighted Scoring Results:\n"
        "  Best Weighted Score: 0.8123\n"
        "  Best Weighted Config:\n"
        "    model: m-2\n"
        "    temperature: 0.5\n"
        "  Objective Weights:\n"
        "    accuracy: 0.7\n"
        "    latency: 0.3\n"
        "============================================================"
    )
    assert trace == [
        ("data", "contains", "session"),
        ("data", "getitem", "session"),
        ("session", "get", "session_id"),
        ("session", "get", "status"),
        ("session", "get", "duration"),
        ("data", "contains", "metrics_summary"),
        ("data", "getitem", "metrics_summary"),
        ("metrics", "get", "total_trials"),
        ("metrics", "get", "successful_trials"),
        ("metrics", "get", "failed_trials"),
        ("metrics", "get", "success_rate"),
        ("metrics", "get", "algorithm"),
        ("metrics", "contains", "best_metrics"),
        ("metrics", "getitem", "best_metrics"),
        ("data", "contains", "best_config"),
        ("data", "getitem", "best_config"),
        ("best_config", "get", "config"),
        ("config", "items"),
        ("data", "contains", "weighted_results"),
        ("data", "getitem", "weighted_results"),
        ("weighted", "get", "best_weighted_score"),
        ("weighted", "contains", "best_weighted_config"),
        ("weighted", "getitem", "best_weighted_config"),
        ("weighted_config", "items"),
        ("weighted", "contains", "objective_weights"),
        ("weighted", "getitem", "objective_weights"),
    ]


def test_run_summary_missing_data_and_formatting_error_are_unchanged(
    analyzer, monkeypatch
):
    monkeypatch.setattr(analyzer, "load_run", MagicMock(return_value=None))
    assert analyzer.get_run_summary("exp", "run") == "No data found for exp/run"

    monkeypatch.setattr(
        analyzer,
        "load_run",
        MagicMock(return_value={"weighted_results": {"best_weighted_score": None}}),
    )
    with pytest.raises(TypeError):
        analyzer.get_run_summary("exp", "run")
