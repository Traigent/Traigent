"""Characterization tests for ``MetricsTracker.format_as_summary_stats``.

Each scenario's complete output (with the clock-derived fields normalised) is
compared with a recorded snapshot, pinning built-in/optional/custom metric
selection, null cost handling and the unmeasured-cost flag/floor rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
)


def ex(
    inp=100,
    out=50,
    rt=200.0,
    cost=0.02,
    tps=None,
    success=True,
    measured=True,
    unmeasured=False,
    unpriced=False,
    known=0.0,
    custom=None,
    sdk_eval=False,
    acc=None,
    scored=True,
):
    cm = CostMetrics(input_cost=cost, output_cost=0.0)
    cm.unmeasured = unmeasured
    cm.unpriced = unpriced
    cm.known_cost = known
    m = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=inp, output_tokens=out),
        response=ResponseMetrics(response_time_ms=rt, tokens_per_second=tps),
        cost=cm,
        success=success,
        measured=measured,
        custom_metrics=dict(custom or {}),
        sdk_evaluation_cost=sdk_eval,
    )
    # Accuracy is only ever a row's own value (never derived from success), so
    # a scored fixture row carries the explicit 1/0 it used to be credited
    # with. ``scored=False`` builds a row that has no accuracy value at all.
    if scored:
        m.custom_metrics.setdefault(
            "accuracy", acc if acc is not None else (1.0 if success else 0.0)
        )
    return m


def tracker(rows, *, unattributed=False, unattributed_cost=0.0):
    t = MetricsTracker()
    t.example_metrics = list(rows)
    t.unattributed_spend = unattributed
    t.unattributed_known_cost = unattributed_cost
    return t


SCENARIOS = {
    "empty": (lambda: tracker([]), False),
    "single_with_tps_and_custom": (
        lambda: tracker([ex(tps=40.0, custom={"f1": 0.5, "accuracy": 1.0})]),
        False,
    ),
    "measured_plus_unmeasured_row": (
        lambda: tracker([ex(), ex(inp=0, out=0, rt=0.0, cost=0.0, measured=False)]),
        False,
    ),
    "cost_unmeasured_with_known_floor": (
        lambda: tracker([ex(), ex(cost=0.0, unmeasured=True, known=0.05)]),
        False,
    ),
    "unpriced_adds_floor": (
        lambda: tracker([ex(), ex(cost=0.03, unpriced=True)]),
        False,
    ),
    "no_cost_measured_nullifies": (
        lambda: tracker(
            [ex(cost=0.0, unmeasured=True), ex(cost=0.0, unmeasured=True, known=0.01)]
        ),
        False,
    ),
    "all_rows_unmeasured": (lambda: tracker([ex(measured=False, cost=0.0)]), False),
    "unattributed_spend": (
        lambda: tracker([ex()], unattributed=True, unattributed_cost=0.07),
        False,
    ),
    "unattributed_zero_floor": (lambda: tracker([ex()], unattributed=True), False),
    "partial_custom_metric_defaults_zero": (
        lambda: tracker([ex(custom={"f1": 0.4}), ex(), ex(custom={"f1": 0.6, "g": 2})]),
        False,
    ),
    "user_evaluation_cost_missing_rows_skipped": (
        lambda: tracker([ex(custom={"evaluation_cost": 0.5}), ex()]),
        False,
    ),
    "sdk_evaluation_cost_default_zero": (
        lambda: tracker([ex(custom={"evaluation_cost": 0.5}, sdk_eval=True), ex()]),
        False,
    ),
    "mixed_sdk_and_user_evaluation_cost": (
        lambda: tracker(
            [
                ex(custom={"evaluation_cost": 0.5}, sdk_eval=True),
                ex(custom={"evaluation_cost": 0.25}),
            ]
        ),
        False,
    ),
    "user_supplied_accuracy": (
        lambda: tracker([ex(custom={"accuracy": 0.25}), ex(success=False, acc=0.0)]),
        True,
    ),
    "errored_but_measured": (
        lambda: tracker([ex(success=False, cost=0.04), ex()]),
        False,
    ),
}


def run(name):
    build, usa = SCENARIOS[name]
    out = build().format_as_summary_stats(user_supplied_accuracy=usa)
    out["execution_time"] = "T"
    out["metadata"] = {**out["metadata"], "timestamp": "TS", "sdk_version": "V"}
    return out


SNAPSHOT = Path(__file__).parent / "fixtures" / "summary_stats_snapshots.json"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_format_as_summary_stats_snapshot(name):
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))[name]
    assert run(name) == expected


def test_rows_without_an_accuracy_value_are_left_out_of_every_view():
    """A row with no accuracy value is unmeasured: not a hit, not a miss."""
    t = tracker([ex(acc=0.5), ex(scored=False), ex(success=False, scored=False)])
    for usa in (False, True):
        stats = t.format_as_summary_stats(user_supplied_accuracy=usa)["metrics"][
            "accuracy"
        ]
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(0.5)
        assert t.format_for_backend(user_supplied_accuracy=usa)[
            "accuracy"
        ] == pytest.approx(0.5)
    none_scored = tracker([ex(scored=False), ex(scored=False)])
    assert none_scored.format_for_backend()["accuracy"] is None
    assert none_scored.format_as_summary_stats()["metrics"]["accuracy"]["count"] == 0
