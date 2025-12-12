import asyncio
import math
from collections.abc import Callable
from typing import Any

import pytest

from tests.shared.mocks.optimizers import MockOptimizer
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator

PER_TRIAL_CAP = 8
TOTAL_BUDGET = 60


def _build_dataset(*, as_mapping: bool = False) -> Dataset:
    examples: list[EvaluationExample] = []
    for idx in range(PER_TRIAL_CAP):
        if as_mapping:
            input_data: Any = {"value": idx}
        else:
            input_data = idx
        examples.append(
            EvaluationExample(
                input_data=input_data,
                expected_output=idx,
                metadata={"index": idx},
            )
        )

    return Dataset(
        examples=examples,
        name="budget_dataset",
        description="Dataset capped per trial",
    )


async def _run_budget_scenario(
    *,
    func_factory: Callable[[bool], Callable[..., Any]],
    is_async: bool,
    parallel_trials: int,
    evaluator_workers: int,
    as_mapping: bool = False,
) -> tuple[OptimizationOrchestrator, Any, list[int]]:
    dataset = _build_dataset(as_mapping=as_mapping)

    optimizer = MockOptimizer(
        config_space={"alpha": [0, 1, 2, 3]},
        objectives=["accuracy"],
    )
    optimizer.set_max_suggestions(50)

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=evaluator_workers)

    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=32,
        max_total_examples=TOTAL_BUDGET,
        parallel_trials=parallel_trials,
        config=TraigentConfig.edge_analytics_mode(),
    )

    eval_func = func_factory(is_async)
    result = await orchestrator.optimize(eval_func, dataset)

    consumed = [
        int(trial.metadata.get("sample_budget_consumed", 0)) for trial in result.trials
    ]
    return orchestrator, result, consumed


def _default_func_factory(is_async: bool) -> Callable[..., Any]:
    if is_async:

        async def async_eval(example_value: int) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"value": example_value}

        return async_eval

    def sync_eval(example_value: int) -> dict[str, Any]:
        return {"value": example_value}

    return sync_eval


@pytest.mark.parametrize(
    ("is_async", "parallel_trials", "evaluator_workers"),
    [
        (False, 1, 1),
        (False, 4, 1),
        (True, 1, 1),
        (True, 4, 2),
    ],
)
@pytest.mark.asyncio
async def test_sample_budget_enforcement_variants(
    is_async: bool, parallel_trials: int, evaluator_workers: int
) -> None:
    orchestrator, result, consumed = await _run_budget_scenario(
        func_factory=_default_func_factory,
        is_async=is_async,
        parallel_trials=parallel_trials,
        evaluator_workers=evaluator_workers,
    )

    assert orchestrator._stop_reason == "max_samples_reached"  # noqa: SLF001
    assert result.status.name == "COMPLETED"

    assert sum(consumed) == TOTAL_BUDGET
    assert all(0 < value <= PER_TRIAL_CAP for value in consumed)
    assert any(value < PER_TRIAL_CAP for value in consumed)

    expected_min_trials = math.ceil(TOTAL_BUDGET / PER_TRIAL_CAP)
    assert len(consumed) >= expected_min_trials
    assert len(consumed) <= expected_min_trials + (parallel_trials - 1)


def _parameter_func_factory(is_async: bool) -> Callable[..., Any]:
    if is_async:

        async def async_parameter_eval(**kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"value": kwargs.get("value")}

        async_parameter_eval._traigent_injection_mode = "parameter"  # type: ignore[attr-defined]
        return async_parameter_eval

    def parameter_eval(**kwargs: Any) -> dict[str, Any]:
        return {"value": kwargs.get("value")}

    parameter_eval._traigent_injection_mode = "parameter"  # type: ignore[attr-defined]
    return parameter_eval


@pytest.mark.asyncio
async def test_sample_budget_enforcement_parameter_injection() -> None:
    orchestrator, result, consumed = await _run_budget_scenario(
        func_factory=_parameter_func_factory,
        is_async=False,
        parallel_trials=4,
        evaluator_workers=2,
        as_mapping=True,
    )

    assert orchestrator._stop_reason == "max_samples_reached"  # noqa: SLF001
    assert result.status.name == "COMPLETED"

    assert sum(consumed) == TOTAL_BUDGET
    assert all(0 < value <= PER_TRIAL_CAP for value in consumed)
    assert any(value < PER_TRIAL_CAP for value in consumed)
