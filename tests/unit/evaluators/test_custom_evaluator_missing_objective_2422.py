"""#2422: a custom_evaluator that never reports an objective fails closed.

``CustomEvaluatorWrapper._aggregate_custom_metrics`` read an absent objective
key as 0.0, so an evaluator emitting ``score`` instead of ``accuracy`` scored
every trial 0.0 with no warning and still produced a ``best_config``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traigent
from traigent.api.decorators import EvaluationOptions
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.evaluators.base import Dataset, EvaluationExample, ExampleResult
from traigent.utils.exceptions import EvaluationError


def _row(example: EvaluationExample, out: str, metrics: dict) -> ExampleResult:
    return ExampleResult(
        example_id=example.input_data["q"],
        input_data=example.input_data,
        expected_output=example.expected_output,
        actual_output=out,
        metrics=metrics,
        execution_time=0.0,
        success=True,
    )


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": q}, expected_output=q.upper())
            for q in ("a", "b", "c")
        ]
    )


def _identity(q: str) -> str:
    return q.upper()


async def test_objective_missing_from_every_row_raises() -> None:
    def evaluator(func, config, example):
        return _row(example, func(**example.input_data), {"score": 1.0})

    wrapper = CustomEvaluatorWrapper(
        evaluator, metrics=["accuracy"], capture_llm_metrics=False
    )
    with pytest.raises(EvaluationError, match=r"\['accuracy'\].*'score'"):
        await wrapper.evaluate(_identity, {}, _dataset())


async def test_objective_missing_from_some_rows_keeps_existing_mean() -> None:
    def evaluator(func, config, example):
        metrics = {} if example.input_data["q"] == "b" else {"accuracy": 1.0}
        return _row(example, func(**example.input_data), metrics)

    wrapper = CustomEvaluatorWrapper(
        evaluator, metrics=["accuracy"], capture_llm_metrics=False
    )
    result = await wrapper.evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["accuracy"] == pytest.approx(2 / 3)


async def test_sdk_measured_objectives_are_not_required_from_the_evaluator() -> None:
    def evaluator(func, config, example):
        return _row(example, func(**example.input_data), {"accuracy": 1.0})

    wrapper = CustomEvaluatorWrapper(
        evaluator,
        metrics=["accuracy", "cost", "latency"],
        capture_llm_metrics=False,
    )
    result = await wrapper.evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["accuracy"] == 1.0


def test_wrong_key_run_fails_trials_instead_of_scoring_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    data = tmp_path / "qa.jsonl"
    data.write_text(
        "".join(
            json.dumps({"input": {"q": q}, "output": q.upper()}) + "\n"
            for q in ("a", "b", "c")
        )
    )

    def evaluator(func, config, example):
        return _row(example, func(**example.input_data), {"score": 1.0})

    @traigent.optimize(
        evaluation=EvaluationOptions(
            eval_dataset=str(data), custom_evaluator=evaluator
        ),
        objectives=["accuracy"],
        configuration_space={"temperature": [0.0, 0.5]},
        offline=True,
        algorithm="grid",
    )
    def answer(q: str) -> str:
        traigent.get_config()
        return q.upper()

    result = answer.optimize_sync(max_trials=2)

    assert result.best_config is None
    assert result.trials
    assert all(not t.is_successful for t in result.trials)
    assert all(t.metrics.get("accuracy") != 0.0 for t in result.trials)
