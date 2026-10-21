"""``OptimizedFunction.compare`` -- fixed configs as one experiment (#1952).

All tests run the real local pipeline with a stub agent (no provider calls).
"""

from __future__ import annotations

import pytest

import traigent
from traigent.core.comparison import validate_comparison_configs

_ROWS = [
    {"input": {"text": f"item {i}"}, "expected_output": "yes" if i % 2 else "no"}
    for i in range(6)
]
_HOLDOUT = [
    {"input": {"text": f"unseen {i}"}, "expected_output": "yes" if i % 2 else "no"}
    for i in range(4)
]


def _make(rows=None):
    def agent(text: str) -> str:
        cfg = traigent.get_config()
        if cfg.get("mode") == "good":
            return "yes" if int(text.split()[-1]) % 2 else "no"
        return "yes"

    return traigent.optimize(
        eval_dataset=rows if rows is not None else _ROWS,
        objectives=["accuracy"],
        configuration_space={"mode": ["good", "bad"], "temperature": (0.0, 1.0)},
        execution_mode="local",
        max_trials=2,
    )(agent)


_GOOD = {"mode": "good", "temperature": 0.0}
_BAD = {"mode": "bad", "temperature": 0.5}
_BAD2 = {"mode": "bad", "temperature": 0.1}


def test_three_configs_two_repeats_is_one_result_with_per_config_stats():
    fn = _make()
    result = fn.compare([_GOOD, _BAD, _BAD2], repeats=2)

    assert len(result.configs) == 3
    assert result.repeats == 2
    assert [c.repeats for c in result.configs] == [2, 2, 2]
    good, bad, _ = result.configs
    assert good.mean_score == pytest.approx(1.0)
    assert good.score_stdev == pytest.approx(0.0)
    assert len(good.scores) == 2
    assert bad.mean_score == pytest.approx(0.5)
    assert result.best_config == good.config
    assert fn.comparison_history == [result]
    assert result.to_dict()["configs"][0]["config"] == good.config


def test_compare_does_not_change_the_wrapper_state():
    fn = _make()

    def state():
        return (
            fn.current_config,
            fn.get_best_config(),
            len(fn.get_optimization_history()),
            fn.best_config_snapshot,
        )

    before = state()
    fn.compare([_GOOD, _BAD])
    assert state() == before


def test_every_config_sees_the_same_examples():
    result = _make().compare([_GOOD, _BAD], repeats=2)
    assert result.same_examples_across_configs is True
    assert result.n_examples == len(_ROWS)
    assert all(c.example_ids == result.example_ids for c in result.configs)
    assert len(set(result.example_ids)) == len(_ROWS)


@pytest.mark.parametrize(
    "bad_config",
    [
        {"mode": "good", "temperature": 0.0, "extra": 1},  # unknown parameter
        {"mode": "nope", "temperature": 0.0},  # choice outside the space
        {"mode": "good", "temperature": 7.0},  # range violation
        {"mode": "good"},  # not fully specified
    ],
)
def test_config_outside_space_raises_value_error_without_echoing_values(bad_config):
    with pytest.raises(ValueError) as exc:
        _make().compare([_GOOD, bad_config])
    text = str(exc.value)
    assert "#1" in text
    for leaked in ("nope", "7.0", "extra"):
        assert leaked not in text


def test_validator_rejects_empty_duplicates_and_non_dict():
    space = {"mode": ["a", "b"]}
    with pytest.raises(ValueError):
        validate_comparison_configs([], space)
    with pytest.raises(ValueError):
        validate_comparison_configs([{"mode": "a"}, {"mode": "a"}], space)
    with pytest.raises(ValueError):
        validate_comparison_configs(["a"], space)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        validate_comparison_configs([{"mode": True}], {"mode": [1, 0]})


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_repeats_must_be_a_positive_int(repeats):
    with pytest.raises(ValueError):
        _make().compare([_GOOD], repeats=repeats)


def test_unknown_cost_stays_none_with_a_floor_never_zero():
    result = _make().compare([_GOOD, _BAD], repeats=2)
    for row in result.configs:
        # The stub agent makes no LLM call, so no cost telemetry exists: the
        # spend is unknown, not $0.
        assert row.total_cost is None
        assert row.cost_lower_bound is not None and row.cost_lower_bound >= 0.0
        assert row.cost_unmeasured_repeats == 2
    assert result.total_cost is None
    assert result.cost_lower_bound is not None


def test_default_dataset_is_flagged_as_not_a_holdout():
    result = _make().compare([_GOOD])
    assert result.holdout.status == "tuning_dataset"
    assert result.holdout.holdout_valid is False


def test_disjoint_eval_dataset_is_a_valid_holdout_and_is_honored():
    fn = _make()
    result = fn.compare([_GOOD, _BAD], eval_dataset=_HOLDOUT)
    assert result.holdout.status == "holdout"
    assert result.holdout.holdout_valid is True
    assert result.holdout.n_overlapping_examples == 0
    assert result.n_examples == len(_HOLDOUT)
    # The decorated dataset is untouched afterwards.
    assert fn.compare([_GOOD]).n_examples == len(_ROWS)


def test_overlapping_eval_dataset_is_flagged_not_trusted():
    result = _make().compare([_GOOD], eval_dataset=_ROWS[:3] + _HOLDOUT)
    assert result.holdout.status == "overlaps_tuning_dataset"
    assert result.holdout.holdout_valid is False
    assert result.holdout.n_overlapping_examples == 3
