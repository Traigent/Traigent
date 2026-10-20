"""Strict cost-objective runs stop after the first no-usage trial (#2421).

Before: a run that declared a cost objective but never captured LLM usage
raised ``UnknownModelError`` only after every trial had run (and been billed),
and the raise discarded the result. These tests use plain Python functions
(no network, no provider client) with mock LLM mode explicitly off.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from traigent.api.types import ExampleResult
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils import cost_calculator as cc
from traigent.utils.cost_calculator import UnknownModelError

N_CONFIGS = 4


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"question": "q0"}, expected_output="ok")
        ],
        name="no-usage-stop",
    )


def _run(
    monkeypatch,
    tmp_path,
    *,
    strict: str | None = None,
    mock: bool = False,
    objectives=("accuracy", "cost"),
    capture_usage: bool = False,
    fail_first: bool = False,
    winner_reps: int = 0,
):
    from traigent.config.context import get_config

    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    # ``None`` is the default path: strict accounting comes from the run-scope
    # default that a declared cost objective switches on, not from the env.
    if strict is None:
        monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    else:
        monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", strict)
    calls: list[int] = []

    def agent(question: str) -> str:
        config = get_config()
        value = (
            config.get("x") if isinstance(config, dict) else getattr(config, "x", None)
        )
        calls.append(value)
        if capture_usage:
            cc.record_captured_usage()
        return "ok"

    def custom_evaluator(func, config, example):
        output = func(**example.input_data)
        failing = fail_first and calls and calls[-1] == 0
        return ExampleResult(
            example_id="e0",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=None if failing else output,
            # An evaluator that reports a literal 0.0 cost: the trial-level
            # strict check sees a "measured" cost, so only the run-level
            # usage guard can tell it is an unmeasured $0.
            metrics={"cost": 0.0} if failing else {"accuracy": 1.0, "cost": 0.0},
            execution_time=0.0,
            success=not failing,
            error_message="boom" if failing else None,
            metadata={},
        )

    opt = OptimizedFunction(
        func=agent,
        configuration_space={"x": list(range(N_CONFIGS))},
        objectives=list(objectives),
        eval_dataset=_dataset(),
        custom_evaluator=custom_evaluator,
        max_trials=N_CONFIGS,
        **({"winner_stability_reps": winner_reps} if winner_reps else {}),
    )
    with patch("traigent.core.optimized_function.is_mock_llm", return_value=mock):
        try:
            result = opt.optimize_sync(
                algorithm="grid",
                max_trials=N_CONFIGS,
                progress_bar=False,
            )
            return result, calls, None
        except UnknownModelError as err:
            return None, calls, err


def test_strict_cost_run_stops_after_first_no_usage_trial(monkeypatch, tmp_path):
    result, calls, err = _run(monkeypatch, tmp_path)
    assert err is not None, "strict cost run with no usage must raise"
    assert len(calls) == 1, f"expected 1 evaluated trial, got {len(calls)}"
    attached = err.result
    assert len(attached.trials) == 1
    assert attached.stop_reason == "no_usage_captured"
    message = str(err)
    assert "stopped after the first successful trial" in message
    assert "remaining trials were not billed" not in message
    assert "already-dispatched trials" in message
    assert str(tmp_path) not in message


def test_failed_first_trial_does_not_trigger_stop(monkeypatch, tmp_path):
    result, calls, err = _run(monkeypatch, tmp_path, fail_first=True)
    assert err is not None
    assert len(err.result.trials) == 2
    assert not err.result.trials[0].is_successful
    assert err.result.trials[1].is_successful
    assert len(calls) == 2


def test_captured_usage_runs_every_trial(monkeypatch, tmp_path):
    result, calls, err = _run(monkeypatch, tmp_path, capture_usage=True)
    assert err is None
    assert len(calls) == N_CONFIGS
    assert result.stop_reason != "no_usage_captured"


@pytest.mark.parametrize("strict,mock", [("false", False), (None, True)])
def test_non_strict_or_mock_keeps_warning_and_runs_all(
    monkeypatch, tmp_path, strict, mock
):
    result, calls, err = _run(monkeypatch, tmp_path, strict=strict, mock=mock)
    assert err is None
    assert len(calls) == N_CONFIGS
    assert "COST_OBJECTIVE_NO_USAGE_CAPTURED" in result.warning_codes


def test_no_cost_objective_never_stops(monkeypatch, tmp_path):
    result, calls, err = _run(monkeypatch, tmp_path, objectives=("accuracy",))
    assert err is None
    assert len(calls) == N_CONFIGS
    assert result.stop_reason != "no_usage_captured"


def test_winner_stability_does_not_rerun_after_no_usage_stop(monkeypatch, tmp_path):
    """#2421: no_usage_captured is terminal; reruns must not spend or mask it."""
    result, calls, err = _run(monkeypatch, tmp_path, winner_reps=3)
    assert err is not None, "strict cost run with no usage must raise"
    assert len(calls) == 1, f"winner reruns dispatched extra calls: {len(calls)}"
    assert err.result.stop_reason == "no_usage_captured"
    assert len(err.result.trials) == 1
