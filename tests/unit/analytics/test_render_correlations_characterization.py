"""Characterization of analytics.render._render_correlations with a fake pyplot."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.analytics.render import ChartRenderError, _render_correlations


@pytest.fixture
def plt():
    p = MagicMock(name="plt")
    p.fig, p.ax = MagicMock(name="fig"), MagicMock(name="ax")
    p.subplots.return_value = (p.fig, p.ax)
    return p


def test_full_plot_calls(plt):
    payload = {
        "run_id": "r1",
        "method": "pearson",
        "sample_size": 12,
        "measure_correlations": [
            {"x": "a", "y": "b", "r": 0.5},
            {"strength": "weak", "r": "-0.25"},
            {"x": "a", "r": -1},
            "junk",
            {"x": "q", "y": "z", "r": None},
            {"x": "q", "y": "z", "r": float("nan")},
            {"x": "c", "y": "d", "r": 0},
        ],
    }
    assert _render_correlations(plt, payload) is plt.fig
    ax = plt.ax
    plt.subplots.assert_called_once_with(figsize=(7, 5))
    assert ax.bar.call_args.args == ([0, 1, 2, 3], [0.5, -0.25, -1.0, 0.0])
    assert ax.bar.call_args.kwargs["color"] == [
        "#2ca02c",
        "#d62728",
        "#d62728",
        "#2ca02c",
    ]
    assert ax.set_xticklabels.call_args.args[0] == ["a↔b", "weak", "?", "c↔d"]
    ax.set_title.assert_called_once_with("Measure correlations — r1 (pearson, n=12)")
    ax.set_ylim.assert_called_once_with(-1.05, 1.05)
    plt.fig.tight_layout.assert_called_once_with()


@pytest.mark.parametrize(
    ("extra", "title"),
    [
        ({}, "Measure correlations — run"),
        ({"method": "m"}, "Measure correlations — run (m)"),
        ({"sample_size": 0}, "Measure correlations — run (n=0)"),
        ({"run_id": "", "method": ""}, "Measure correlations — run"),
    ],
)
def test_title_variants(plt, extra, title):
    payload = {"measure_correlations": [{"x": "a", "y": "b", "r": 1}], **extra}
    _render_correlations(plt, payload)
    plt.ax.set_title.assert_called_once_with(title)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"measure_correlations": None},
        {"measure_correlations": []},
        {"measure_correlations": ["x", {"r": "abc"}, {"r": None}]},
    ],
)
def test_no_plottable_entries_raises_before_subplots(plt, payload):
    with pytest.raises(ChartRenderError, match="no plottable measure_correlations"):
        _render_correlations(plt, payload)
    plt.subplots.assert_not_called()
