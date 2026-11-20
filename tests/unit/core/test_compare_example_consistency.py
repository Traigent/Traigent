"""``compare(..., repeats=N)`` per-example consistency labels.

Runs the real local compare pipeline with a stub agent (no provider calls). The
agent's answer for one example alternates between calls, so that example is
flaky for the config while the others are stable.
"""

from __future__ import annotations

import itertools

import pytest

import traigent

_ROWS = [{"input": {"text": f"item {i}"}, "expected_output": "yes"} for i in range(3)]


def _make():
    flip = itertools.count()

    def agent(text: str) -> str:
        cfg = traigent.get_config()
        if text == "item 0":
            return "no"  # always wrong
        if text == "item 1":
            return "yes"  # always right
        # item 2: right on every other evaluation
        if cfg.get("mode") == "steady":
            return "yes"
        return "yes" if next(flip) % 2 == 0 else "no"

    return traigent.optimize(
        eval_dataset=_ROWS,
        objectives=["accuracy"],
        configuration_space={"mode": ["steady", "wobbly"]},
        execution_mode="local",
        max_trials=2,
    )(agent)


def test_repeats_label_every_example_for_each_config() -> None:
    comparison = _make().compare([{"mode": "wobbly"}, {"mode": "steady"}], repeats=4)

    wobbly = comparison.example_consistency(0)
    labels = {eid: row["label"] for eid, row in wobbly["examples"].items()}
    assert sorted(labels.values()) == ["consistent_fail", "consistent_pass", "flaky"]
    assert wobbly["reps"] == 4
    assert wobbly["counts"] == {
        "consistent_pass": 1,
        "consistent_fail": 1,
        "flaky": 1,
    }
    flaky = next(r for r in wobbly["examples"].values() if r["label"] == "flaky")
    assert flaky["passes"] + flaky["fails"] == 4 and flaky["passes"] == 2

    steady = comparison.example_consistency(config_index=1)
    assert sorted(r["label"] for r in steady["examples"].values()) == [
        "consistent_fail",
        "consistent_pass",
        "consistent_pass",
    ]


def test_a_single_repeat_labels_nothing() -> None:
    comparison = _make().compare([{"mode": "steady"}], repeats=1)

    report = comparison.example_consistency()
    assert report["examples"] == {}
    assert report["counts"] == {
        "consistent_pass": 0,
        "consistent_fail": 0,
        "flaky": 0,
    }


def test_config_index_out_of_range_raises_index_error() -> None:
    comparison = _make().compare([{"mode": "steady"}], repeats=2)

    with pytest.raises(IndexError):
        comparison.example_consistency(config_index=5)


def test_integer_and_string_ids_that_print_alike_are_not_merged() -> None:
    rows = [
        {
            "input": {"text": "bad"},
            "expected_output": "yes",
            "metadata": {"example_id": 1},
        },
        {
            "input": {"text": "good"},
            "expected_output": "yes",
            "metadata": {"example_id": "1"},
        },
    ]

    def agent(text: str) -> str:
        return "no" if text == "bad" else "yes"

    wrapped = traigent.optimize(
        eval_dataset=rows,
        objectives=["accuracy"],
        configuration_space={"mode": ["a"]},
        execution_mode="local",
        max_trials=1,
    )(agent)
    comparison = wrapped.compare([{"mode": "a"}], repeats=3)

    with pytest.raises(ValueError, match=r"1 \(int\).*'1' \(str\)"):
        comparison.example_consistency()
