"""Cost must not be averaged over rows that never measured it (issue #2404).

Before the fix, a failed row got ``cost: 0.0`` and a row that omitted ``cost``
was read as ``0.0``, so a trial whose true per-example cost was 0.005 reported
0.0025 when half its rows failed — and could win a cost objective.
"""

import pytest

from traigent.api.types import ExampleResult
from traigent.core.cost_enforcement import CostTrackingRequiredError
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator


def _dataset(n: int = 4) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(n)
        ],
        name="cost-coverage",
    )


async def _identity(value: int) -> int:
    return value


def _evaluator(behaviour):
    """Row i: 'priced' -> cost 0.005, 'raise' -> evaluator raises, 'nocost' -> no key."""

    async def custom_evaluator(func, config, example):
        index = example.input_data["value"]
        kind = behaviour[index]
        if kind == "raise":
            raise RuntimeError("provider call failed after spending")
        metrics = {"accuracy": 1.0}
        if kind == "priced":
            metrics["cost"] = 0.005
        return ExampleResult(
            example_id=f"row-{index}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=await func(**example.input_data),
            metrics=metrics,
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    return CustomEvaluatorWrapper(
        custom_evaluator, metrics=["accuracy", "cost"], capture_llm_metrics=False
    )


@pytest.fixture
def lenient(monkeypatch):
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "false")


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["raise", "nocost"])
async def test_partial_cost_is_averaged_only_over_measured_rows(lenient, missing):
    evaluator = _evaluator(["priced", "priced", missing, missing])
    result = await evaluator.evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["cost"] == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_failed_rows_still_count_as_zero_quality(lenient):
    # Quality keeps its conservative behaviour: a crashed row scores 0.
    evaluator = _evaluator(["priced", "priced", "raise", "raise"])
    result = await evaluator.evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["accuracy"] == pytest.approx(0.5)
    assert all("cost" not in r.metrics for r in result.example_results if not r.success)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["raise", "nocost"])
async def test_strict_accounting_fails_a_trial_with_partial_cost(strict, missing):
    evaluator = _evaluator(["priced", "priced", missing, missing])
    with pytest.raises(CostTrackingRequiredError, match=r"cost.*2 of 4"):
        await evaluator.evaluate(_identity, {}, _dataset())


@pytest.mark.asyncio
async def test_full_cost_coverage_is_unchanged_under_strict(strict):
    evaluator = _evaluator(["priced"] * 4)
    result = await evaluator.evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["cost"] == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_no_row_reporting_cost_keeps_the_previous_zero(strict):
    # An evaluator that never emits cost is left to the run-level
    # "no usage captured" handling; this change must not start failing it.
    evaluator = _evaluator(["nocost"] * 4)
    result = await evaluator.evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["cost"] == 0.0


def test_simple_scoring_aggregation_ignores_unmeasured_cost(lenient):
    evaluator = SimpleScoringEvaluator(
        scoring_function=lambda output, expected: 1.0, metrics=["accuracy", "cost"]
    )
    rows = [
        {"accuracy": 1.0, "cost": 0.005},
        {"accuracy": 1.0, "cost": 0.005},
        evaluator._create_failed_example_result(
            EvaluationExample(input_data={}, expected_output=None),
            2,
            RuntimeError("boom"),
        ).metrics,
    ]
    aggregated = evaluator._aggregate_custom_metrics(rows)
    assert aggregated["cost"] == pytest.approx(0.005)
    assert aggregated["accuracy"] == pytest.approx(2 / 3)
