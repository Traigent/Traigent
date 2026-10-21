"""``OptimizedFunction.compare`` -- fixed configs as one experiment (#1952).

All tests run the real local pipeline with a stub agent (no provider calls).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import traigent
from traigent.core import comparison as cmp
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


# --- rework: direct evaluator path (no optimize()/orchestrator/backend) -------


def test_requested_config_is_evaluated_not_the_decorated_default():
    seen: list[dict] = []

    def agent(text: str) -> str:
        seen.append(dict(traigent.get_config()))
        return "yes"

    fn = traigent.optimize(
        eval_dataset=_ROWS,
        objectives=["accuracy"],
        configuration_space={"mode": ["good", "bad"], "temperature": (0.0, 1.0)},
        default_config={"mode": "bad", "temperature": 0.9},
        execution_mode="local",
    )(agent)
    result = fn.compare([_GOOD])
    assert {c["mode"] for c in seen} == {"good"}
    assert result.configs[0].config == _GOOD
    assert result.configs[0].trials[0].config == _GOOD


def test_compare_never_touches_orchestrator_backend_or_persistence():
    fn = _make()

    def boom(*args, **kwargs):
        raise AssertionError("compare() must not use this path")

    with (
        patch("traigent.core.optimized_function.OptimizationOrchestrator", boom),
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager.__init__",
            boom,
        ),
        patch.object(type(fn), "optimize", boom),
        patch.object(type(fn), "save_optimization_results", boom),
        patch.object(type(fn), "_try_cloud_execution", boom),
    ):
        result = fn.compare([_GOOD, _BAD], repeats=2)
    assert len(result.configs) == 2
    assert fn.get_optimization_history() == []


def test_example_ids_are_the_evaluators_own_and_coverage_is_reported():
    result = _make().compare([_GOOD, _BAD], repeats=2)
    assert result.same_examples_across_configs is True
    assert result.coverage_complete is True
    assert result.max_examples_applied is None
    assert result.n_source_examples == result.n_examples == len(_ROWS)
    assert len(result.example_ids) == len(_ROWS)


def test_max_examples_cap_is_respected_and_reported():
    def agent(text: str) -> str:
        return "yes"

    fn = traigent.optimize(
        eval_dataset=_ROWS,
        objectives=["accuracy"],
        configuration_space={"mode": ["good", "bad"], "temperature": (0.0, 1.0)},
        execution_mode="local",
    )(agent)
    fn.max_examples = 2
    result = fn.compare([_GOOD])
    assert result.n_source_examples == len(_ROWS)
    assert result.n_examples == 2
    assert result.max_examples_applied == 2
    assert len(result.example_ids) == 2
    assert result.coverage_complete is True


def test_evaluation_exception_is_a_failed_repeat_with_unknown_cost():
    fn = _make()
    calls = {"n": 0}
    real = None

    async def flaky(self, func, config, dataset, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("secret provider text")
        return await real(self, func, config, dataset, **kw)

    from traigent.evaluators.local import LocalEvaluator

    real = LocalEvaluator.evaluate
    with patch.object(LocalEvaluator, "evaluate", flaky):
        result = fn.compare([_GOOD], repeats=2)
    row = result.configs[0]
    assert row.failed_repeats == 1
    assert row.scores[1] is None
    assert row.total_cost is None
    assert result.same_examples_across_configs is False
    assert result.coverage_complete is False
    assert "secret" not in repr(row.trials[-1].error_message)


def test_cancellation_leaves_history_and_slot_untouched():
    fn = _make()
    first = fn.compare([_GOOD])
    assert fn.comparison_history == [first]
    history_before = fn.get_optimization_history()

    from traigent.evaluators.local import LocalEvaluator

    async def cancelled(self, *a, **k):
        raise asyncio.CancelledError

    with patch.object(LocalEvaluator, "evaluate", cancelled):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(fn.compare_async([_GOOD, _BAD]))

    assert fn.comparison_history == [first]
    assert fn.get_optimization_history() == history_before
    assert fn.__dict__.get("_exclusive_slot") is None
    assert len(fn.compare([_BAD]).configs) == 1  # slot reusable


def test_compare_holds_the_exclusive_slot_for_its_whole_duration():
    from traigent.evaluators.local import LocalEvaluator
    from traigent.utils.exceptions import OverlappingOptimizationError

    fn = _make()
    real = LocalEvaluator.evaluate
    observed = []

    async def spy(self, func, config, dataset, **kw):
        observed.append(fn.__dict__.get("_exclusive_slot"))
        with pytest.raises(OverlappingOptimizationError):
            fn.apply_best_config(None)
        return await real(self, func, config, dataset, **kw)

    with patch.object(LocalEvaluator, "evaluate", spy):
        fn.compare([_GOOD, _BAD], repeats=2)
    assert observed == ["run"] * 4
    assert fn.__dict__.get("_exclusive_slot") is None


def test_custom_judge_receives_the_call_time_holdout_inputs_and_labels():
    judged: list[tuple[object, object]] = []

    def judge(output, expected_output=None, **_):
        judged.append((output, expected_output))
        return 1.0 if output == expected_output else 0.0

    def agent(text: str) -> str:
        return "yes"

    fn = traigent.optimize(
        eval_dataset=_ROWS,
        objectives=["accuracy"],
        configuration_space={"mode": ["good", "bad"], "temperature": (0.0, 1.0)},
        scoring_function=judge,
        execution_mode="local",
    )(agent)
    result = fn.compare([_GOOD], eval_dataset=_HOLDOUT)
    labels = sorted(str(e) for _, e in judged)
    assert labels == sorted(str(r["expected_output"]) for r in _HOLDOUT)
    assert result.configs[0].mean_score == pytest.approx(0.5)


def test_minimize_objective_picks_the_lowest_mean():
    def err(output, expected_output=None, **_):
        return 0.0 if output == expected_output else 1.0

    def agent(text: str) -> str:
        cfg = traigent.get_config()
        return (
            "yes"
            if int(text.split()[-1]) % 2
            else ("no" if cfg.get("mode") == "good" else "yes")
        )

    fn = traigent.optimize(
        eval_dataset=_ROWS,
        objectives=["error_rate"],
        metric_functions={"error_rate": err},
        configuration_space={"mode": ["good", "bad"], "temperature": (0.0, 1.0)},
        execution_mode="local",
    )(agent)
    result = fn.compare([_BAD, _GOOD])
    assert result.orientation == "minimize"
    assert result.configs[0].mean_score > result.configs[1].mean_score
    assert result.best_config == _GOOD


def _row(mean, cost=None):
    return cmp.ConfigComparison(
        config={"x": mean},
        repeats=1,
        scores=[mean],
        mean_score=mean,
        score_stdev=None,
        score_min=mean,
        score_max=mean,
        metrics_mean={},
        total_cost=cost,
        cost_lower_bound=None,
        cost_unmeasured_repeats=0,
        failed_repeats=0,
        example_ids=(),
    )


def _result(rows, orientation, band=None):
    holdout = cmp.build_holdout_check(["a"], ["b"], eval_dataset_supplied=True)
    return cmp.ComparisonResult(
        comparison_id="c",
        primary_metric="m",
        orientation=orientation,
        band=band,
        objectives=["m"],
        repeats=1,
        n_examples=1,
        n_source_examples=1,
        max_examples_applied=None,
        example_ids=("a",),
        same_examples_across_configs=True,
        coverage_complete=True,
        holdout=holdout,
        configs=rows,
    )


def test_winner_honours_minimize_maximize_and_band():
    rows = [_row(0.2), _row(0.5), _row(0.9)]
    assert _result(rows, "maximize").best.mean_score == 0.9
    assert _result(rows, "minimize").best.mean_score == 0.2
    assert _result(rows, "band", band=(0.4, 0.6)).best.mean_score == 0.5
    assert _result(rows, "band", band=(0.0, 0.1)).best.mean_score == 0.2


def _fake_run(total_cost, unmeasured, floor, score=1.0):
    trial = SimpleNamespace(
        score=score,
        metrics={"accuracy": score},
        is_successful=True,
    )
    stats = SimpleNamespace(
        total_cost=total_cost,
        cost_unmeasured_trials=1 if unmeasured else 0,
        cost_lower_bound=floor,
    )
    return SimpleNamespace(trials=[trial], experiment_stats=stats)


def test_mixed_measured_and_unknown_repeats_give_none_and_a_floor():
    row = cmp.aggregate_config(
        {"x": 1},
        [_fake_run(0.25, False, None), _fake_run(None, True, 0.1)],
        "accuracy",
        (),
    )
    assert row.total_cost is None
    assert row.cost_lower_bound == pytest.approx(0.35)
    assert row.cost_unmeasured_repeats == 1
    all_known = cmp.aggregate_config(
        {"x": 1}, [_fake_run(0.25, False, None)] * 2, "accuracy", ()
    )
    assert all_known.total_cost == pytest.approx(0.5)
    assert all_known.cost_lower_bound is None
    unknown_zero_floor = cmp.aggregate_config(
        {"x": 1}, [_fake_run(None, True, None)], "accuracy", ()
    )
    assert unknown_zero_floor.total_cost is None


@pytest.mark.parametrize(
    "spec,good,bad",
    [
        ({"type": "int", "low": 1, "high": 10}, 3, [3.5, 0, 11, True]),
        ({"type": "int", "low": 0, "high": 10, "step": 5}, 5, [3, 15]),
        ({"type": "float", "low": 0.0, "high": 1.0, "step": 0.25}, 0.75, [0.3, 1.25]),
        ({"type": "categorical", "choices": ["a", "b"]}, "a", ["c", 1]),
        ({"type": "fixed", "value": 7}, 7, [8]),
        ((1, 5), 2, [2.5, 6]),
        ((0.0, 1.0), 0.5, [1.5, True]),
    ],
)
def test_domain_validation_uses_sdk_domain_semantics(spec, good, bad):
    space = {"p": spec}
    assert validate_comparison_configs([{"p": good}], space) == [{"p": good}]
    for value in bad:
        with pytest.raises(ValueError):
            validate_comparison_configs([{"p": value}], space)
