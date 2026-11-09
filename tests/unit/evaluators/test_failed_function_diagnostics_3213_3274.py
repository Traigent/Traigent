"""A raising user function names its exception in the trial error and the log.

#3213: ``TrialResult.error_message`` must name the exception class and message
instead of the generic "No examples succeeded".
#3274: the log carries the class and a single traceback per trial, not one
message-only line (or traceback) per example.
"""

from __future__ import annotations

import logging

import pytest

from traigent.core.trial_result_factory import build_success_result
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator

N_EXAMPLES = 4


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"row": i}, expected_output=i)
            for i in range(N_EXAMPLES)
        ],
        name="failing",
    )


async def _evaluate(func):
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    return await evaluator.evaluate(func, {"m": "a"}, _dataset())


def _raise_value_error(row):
    raise ValueError("boom")


@pytest.mark.asyncio
async def test_all_examples_raising_names_exception_in_trial_error_message():
    eval_result = await _evaluate(_raise_value_error)
    trial = build_success_result(
        trial_id="t1",
        evaluation_config={"m": "a"},
        eval_result=eval_result,
        duration=0.1,
        examples_attempted=N_EXAMPLES,
        total_cost=None,
        optuna_trial_id=None,
    )

    assert trial.status.value == "failed"
    assert "ValueError" in trial.error_message
    assert "boom" in trial.error_message
    assert "No examples succeeded" not in trial.error_message


@pytest.mark.asyncio
async def test_mixed_exceptions_report_most_common_with_count():
    def func(row):
        if row == 0:
            raise KeyError("rare")
        raise ValueError("common")

    eval_result = await _evaluate(func)
    trial = build_success_result(
        trial_id="t1",
        evaluation_config={"m": "a"},
        eval_result=eval_result,
        duration=0.1,
        examples_attempted=N_EXAMPLES,
        total_cost=None,
        optuna_trial_id=None,
    )

    assert "ValueError: common" in trial.error_message
    assert f"{N_EXAMPLES - 1} of {N_EXAMPLES} examples" in trial.error_message


@pytest.mark.asyncio
async def test_error_message_is_redacted():
    def func(row):
        raise ValueError("bad input for alice@example.com")

    eval_result = await _evaluate(func)
    trial = build_success_result(
        trial_id="t1",
        evaluation_config={"m": "a"},
        eval_result=eval_result,
        duration=0.1,
        examples_attempted=N_EXAMPLES,
        total_cost=None,
        optuna_trial_id=None,
    )

    assert "alice@example.com" not in trial.error_message
    assert "ValueError" in trial.error_message


@pytest.mark.asyncio
async def test_log_has_class_and_one_traceback_for_n_examples(caplog):
    with caplog.at_level(logging.WARNING, logger="traigent"):
        await _evaluate(_raise_value_error)

    failures = [r for r in caplog.records if "Function call failed" in r.getMessage()]
    assert len(failures) == 1
    text = failures[0].getMessage()
    assert "ValueError" in text
    assert "boom" in text
    assert text.count("Traceback (most recent call last)") == 1
    assert failures[0].levelno == logging.WARNING
