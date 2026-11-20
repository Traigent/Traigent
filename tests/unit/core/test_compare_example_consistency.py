"""``compare(..., repeats=N)`` per-example consistency labels.

Runs the real local compare pipeline with a stub agent (no provider calls). The
agent's answer for one example alternates between calls, so that example is
mixed for the config while the others are stable.
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
    assert sorted(labels.values()) == ["all_fail", "all_pass", "mixed"]
    assert wobbly["reps"] == 4
    assert wobbly["metric"] in {"score", "accuracy", "score_or_accuracy"}
    assert wobbly["threshold"] == 0.5
    assert wobbly["counts"] == {
        "all_pass": 1,
        "all_fail": 1,
        "mixed": 1,
        "errored": 0,
    }
    mixed_row = next(r for r in wobbly["examples"].values() if r["label"] == "mixed")
    assert mixed_row["passes"] + mixed_row["fails"] == 4 and mixed_row["passes"] == 2

    steady = comparison.example_consistency(config_index=1)
    assert sorted(r["label"] for r in steady["examples"].values()) == [
        "all_fail",
        "all_pass",
        "all_pass",
    ]


def test_a_single_repeat_labels_nothing() -> None:
    comparison = _make().compare([{"mode": "steady"}], repeats=1)

    report = comparison.example_consistency()
    assert report["examples"] == {}
    assert report["counts"] == {
        "all_pass": 0,
        "all_fail": 0,
        "mixed": 0,
        "errored": 0,
    }


@pytest.mark.parametrize("index", [5, 1, -1, -2])
def test_config_index_out_of_range_or_negative_raises_index_error(index) -> None:
    comparison = _make().compare([{"mode": "steady"}], repeats=2)

    with pytest.raises(IndexError, match="outside 0..0"):
        comparison.example_consistency(config_index=index)


@pytest.mark.parametrize("index", [True, False, 1.0, "0", None])
def test_config_index_that_is_not_an_int_raises_type_error(index) -> None:
    comparison = _make().compare([{"mode": "steady"}], repeats=2)

    with pytest.raises(TypeError, match="config_index must be an int"):
        comparison.example_consistency(config_index=index)


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


def _run_result(scores: list[float] | None):
    """A single-trial run result, or one whose repeat crashed (``None``)."""
    from types import SimpleNamespace

    from traigent.api.types import ExampleResult

    if scores is None:
        trials: list = []
    else:
        rows = [
            ExampleResult(f"e{i}", {}, "y", "y", {"score": v}, 0.1, True).to_dict()
            for i, v in enumerate(scores)
        ]
        trials = [
            SimpleNamespace(
                metadata={"example_results": rows},
                metrics={"accuracy": 1.0},
                score=1.0,
                is_successful=True,
            )
        ]
    stats = SimpleNamespace(
        total_cost=0.0, cost_unmeasured_trials=0, cost_lower_bound=None
    )
    return SimpleNamespace(trials=trials, experiment_stats=stats)


def test_a_crashed_repeat_is_an_errored_attempt_not_a_dropped_one() -> None:
    from traigent.core.comparison import aggregate_config

    runs = [
        _run_result([1.0, 0.0]),
        _run_result(None),  # crashed: no trial at all
        _run_result([1.0, 0.0]),
        _run_result([1.0, 0.0]),
    ]
    cmp = aggregate_config({"mode": "a"}, runs, "accuracy", ("e0", "e1"))

    report = cmp.example_consistency()
    assert report["reps"] == 4
    assert {r["attempted"] for r in report["examples"].values()} == {4}
    assert {r["errored"] for r in report["examples"].values()} == {1}
    assert report["examples"]["e0"]["label"] == "all_pass"
    assert report["examples"]["e1"]["label"] == "all_fail"


def test_a_repeat_without_per_example_results_is_an_errored_attempt() -> None:
    from traigent.core.comparison import aggregate_config

    bare = _run_result([1.0, 0.0])
    bare.trials[0].metadata = {}  # ran, but returned no per-example results
    runs = [_run_result([1.0, 0.0]), bare, _run_result([1.0, 0.0])]
    cmp = aggregate_config({"mode": "a"}, runs, "accuracy", ("e0", "e1"))

    report = cmp.example_consistency()
    assert report["reps"] == 3
    assert {r["errored"] for r in report["examples"].values()} == {1}


def test_mostly_crashed_repeats_give_no_label() -> None:
    from traigent.core.comparison import aggregate_config

    runs = [_run_result([1.0]), _run_result(None), _run_result(None)]
    cmp = aggregate_config({"mode": "a"}, runs, "accuracy", ("e0",))

    report = cmp.example_consistency()
    assert report["examples"]["e0"]["label"] is None
    assert report["counts"]["errored"] == 1
