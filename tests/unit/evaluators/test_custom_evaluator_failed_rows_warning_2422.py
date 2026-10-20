"""Failed custom_evaluator rows are surfaced as a result warning code (#2422).

Scoring is unchanged: a raising row scores 0.0 on quality metrics. Only counts
are recorded; no row content or exception text.
"""

from __future__ import annotations

import pytest

from traigent.api.types import ExampleResult, TrialStatus
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample

CODE = "CUSTOM_EVALUATOR_ROWS_FAILED"
SECRET = "row-secret-text-do-not-leak"


def _run(tmp_path, monkeypatch, raising_rows: set[int]):
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))

    def target(value: int, **_config) -> int:
        return value

    def custom_evaluator(func, config, example):
        index = example.input_data["value"]
        if index in raising_rows:
            raise RuntimeError(SECRET)
        return ExampleResult(
            example_id=f"row-{index}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=func(**example.input_data),
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(3)
        ],
        name="failed-rows-2422",
    )
    opt_func = OptimizedFunction(
        func=target,
        configuration_space={"model": ["a"]},
        objectives=["accuracy"],
        eval_dataset=dataset,
        custom_evaluator=custom_evaluator,
        max_trials=1,
    )
    return opt_func.optimize_sync(algorithm="grid", max_trials=1, progress_bar=False)


def test_one_of_three_rows_raising_is_counted_and_coded(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, {1})

    trial = result.trials[0]
    assert trial.status == TrialStatus.COMPLETED
    assert trial.metrics["accuracy"] == pytest.approx(2 / 3)  # unchanged scoring
    assert trial.metadata["custom_evaluator_failed_rows"] == 1
    assert trial.metadata["custom_evaluator_total_rows"] == 3
    assert CODE in result.warning_codes
    assert SECRET not in repr(result.warnings)
    assert SECRET not in repr(result.metadata.get("session_summary"))
    summary = result.metadata["session_summary"]
    assert summary["custom_evaluator_failed_rows_by_trial"] == {trial.trial_id: 1}


def test_no_raising_rows_has_no_code(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, set())

    trial = result.trials[0]
    assert trial.status == TrialStatus.COMPLETED
    assert "custom_evaluator_failed_rows" not in trial.metadata
    assert CODE not in result.warning_codes


def test_all_rows_raising_fails_trial_without_code(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, {0, 1, 2})

    assert all(t.status == TrialStatus.FAILED for t in result.trials)
    assert CODE not in result.warning_codes
