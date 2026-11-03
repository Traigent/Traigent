"""Characterization of analytics.render._render_pareto using a fake pyplot."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.analytics.render import ChartRenderError, _render_pareto


@pytest.fixture
def plt():
    p = MagicMock(name="plt")
    p.fig, p.ax = MagicMock(name="fig"), MagicMock(name="ax")
    p.subplots.return_value = (p.fig, p.ax)
    return p


def _pt(cid, cost, quality, knee=False):
    return {
        "config_id": cid,
        "metrics": {"cost": cost, "quality": quality},
        "is_knee": knee,
    }


def test_full_plot_calls(plt):
    payload = {
        "run_id": "r1",
        "shape": "s",
        "frontier": [_pt("b", 2, 0.9, knee=True), _pt("a", 1, 0.5)],
        "dominated_points": [_pt("d", 3, 0.1)],
    }
    assert _render_pareto(plt, payload) is plt.fig
    ax = plt.ax
    plt.subplots.assert_called_once_with(figsize=(7, 5))
    assert [c.args[0] for c in ax.annotate.call_args_list] == ["b", "a"]
    scatter = ax.scatter.call_args_list
    assert [c.kwargs["label"] for c in scatter] == ["dominated", "frontier", "knee"]
    assert scatter[0].args == ([3.0], [0.1])
    assert scatter[1].args == ([2.0, 1.0], [0.9, 0.5])
    assert scatter[2].args == ([2.0], [0.9])
    assert ax.plot.call_args.args == ([1.0, 2.0], [0.5, 0.9])
    ax.set_title.assert_called_once_with("Pareto frontier — r1 (s)")
    ax.set_xlabel.assert_called_once()
    ax.set_ylabel.assert_called_once()
    plt.fig.tight_layout.assert_called_once_with()


def test_title_defaults_and_dominated_fallback_key(plt):
    payload = {"frontier": [], "dominated": [_pt("d", "3", "0.25")]}
    _render_pareto(plt, payload)
    plt.ax.set_title.assert_called_once_with("Pareto frontier — run")
    assert plt.ax.scatter.call_args.args == ([3.0], [0.25])
    plt.ax.plot.assert_not_called()
    plt.ax.annotate.assert_not_called()


def test_dominated_points_key_wins_over_dominated(plt):
    payload = {
        "frontier": [],
        "dominated_points": [_pt("x", 1, 1)],
        "dominated": [_pt("y", 9, 9)],
    }
    _render_pareto(plt, payload)
    assert plt.ax.scatter.call_args.args == ([1.0], [1.0])


def test_measures_remap_metric_keys(plt):
    payload = {
        "measures": {"quality": "acc", "cost": "usd"},
        "frontier": [{"metrics": {"acc": 0.5, "usd": 2}, "config_id": ""}],
    }
    _render_pareto(plt, payload)
    plt.ax.scatter.assert_called_once()
    plt.ax.annotate.assert_not_called()


@pytest.mark.parametrize("measures", [["x"], "y", 5])
def test_non_dict_truthy_measures_use_axis_names(plt, measures):
    _render_pareto(plt, {"measures": measures, "frontier": [_pt("a", 1, 1)]})
    plt.ax.scatter.assert_called_once()


def test_unplottable_points_skipped_then_error(plt):
    bad = [
        "str",
        {},
        {"metrics": {"cost": 1}},
        {"metrics": {"cost": True, "quality": 1}},
        {"metrics": {"cost": "nan", "quality": 1}},
        {"metrics": None},
    ]
    with pytest.raises(ChartRenderError, match=r"metrics\.cost and metrics\.quality"):
        _render_pareto(plt, {"frontier": bad, "dominated": bad})
    plt.ax.set_title.assert_not_called()


def test_error_names_remapped_keys(plt):
    with pytest.raises(ChartRenderError, match=r"metrics\.usd and metrics\.acc"):
        _render_pareto(plt, {"measures": {"quality": "acc", "cost": "usd"}})


def test_last_knee_wins(plt):
    payload = {"frontier": [_pt("a", 1, 1, True), _pt("b", 2, 2, True)]}
    _render_pareto(plt, payload)
    assert plt.ax.scatter.call_args_list[-1].args == ([2.0], [2.0])


def test_frontier_non_dict_metrics_raises_attribute_error(plt):
    with pytest.raises(AttributeError):
        _render_pareto(plt, {"frontier": [{"metrics": [1]}]})


def test_frontier_not_iterable_raises_type_error(plt):
    with pytest.raises(TypeError):
        _render_pareto(plt, {"frontier": 5})
