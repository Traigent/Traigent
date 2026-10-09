"""A hybrid wire response that omits cost is unmeasured, never a measured $0.

End to end through ``HybridAPIEvaluator.evaluate`` with a mocked transport
(Traigent#2517 review blocker): reported-vs-omitted cost must survive the
protocol dataclass, the per-example rows, the aggregate, the progress
callback, the budgeted submission and Pareto selection. A reported ``0.0``
stays a measured ``$0`` and a reported ``0.25`` is unchanged.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission,
)
from traigent.core.orchestrator_helpers import extract_cost_from_results
from traigent.core.trial_result_factory import build_success_result
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.hybrid_api import HybridAPIEvaluator
from traigent.hybrid.protocol import HybridEvaluateResponse, HybridExecuteResponse
from traigent.utils.multi_objective import ParetoFrontCalculator


def _wire(
    operational_metrics: dict[str, Any],
    item_extra: dict[str, Any] | None = None,
    *,
    n: int = 2,
    quality_metrics: dict[str, float] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    item_extra = item_extra or {}
    wire: dict[str, Any] = {
        "request_id": "r1",
        "execution_id": "e1",
        "status": status,
        "outputs": [
            {"example_id": f"ex_{i}", "output": "ok", "metrics": {"accuracy": 1.0}}
            | item_extra
            for i in range(n)
        ],
        "operational_metrics": operational_metrics,
    }
    if quality_metrics is not None:
        wire["quality_metrics"] = quality_metrics
    return wire


def _evaluator(
    responses: list[dict[str, Any]],
    *,
    batch_size: int,
    supports_evaluate: bool = False,
    evaluate_wire: dict[str, Any] | None = None,
) -> HybridAPIEvaluator:
    evaluator = HybridAPIEvaluator(
        api_endpoint="http://unused.invalid", batch_size=batch_size
    )
    transport = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[HybridExecuteResponse.from_dict(w) for w in responses]
        ),
        evaluate=AsyncMock(
            return_value=(
                HybridEvaluateResponse.from_dict(evaluate_wire or {})
                if evaluate_wire is not None
                else None
            )
        ),
    )
    evaluator._get_transport = AsyncMock(return_value=transport)
    evaluator._get_capabilities = AsyncMock(
        return_value=SimpleNamespace(supports_evaluate=supports_evaluate)
    )
    evaluator._ensure_lifecycle_manager = AsyncMock()
    return evaluator


def _dataset(n: int = 2) -> Dataset:
    return Dataset(
        name="d",
        examples=[
            EvaluationExample(input_data={"v": i}, expected_output=i) for i in range(n)
        ],
    )


async def _run(wire: dict[str, Any], *, progress_callback: Any = None):
    evaluator = _evaluator([wire], batch_size=2)
    return await evaluator.evaluate(
        lambda v: v, {}, _dataset(), progress_callback=progress_callback
    )


async def _run_batches(wires: list[dict[str, Any]]):
    """One example per batch: each wire response is its own batch."""
    evaluator = _evaluator(wires, batch_size=1)
    return await evaluator.evaluate(lambda v: v, {}, _dataset(len(wires)))


def _trial(tid: str, metrics: dict[str, Any]) -> TrialResult:
    return TrialResult(
        trial_id=tid,
        config={},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def _assert_unmeasured(agg: dict[str, Any]) -> None:
    assert agg["cost_unmeasured"] == 1.0
    assert "cost" not in agg
    assert "total_cost" not in agg


# --- protocol: reported vs not reported -------------------------------------


def test_protocol_distinguishes_omitted_from_reported_zero() -> None:
    omitted = HybridExecuteResponse.from_dict(_wire({"latency_ms": 1.0}))
    zero = HybridExecuteResponse.from_dict(_wire({"total_cost_usd": 0.0}))
    legacy_key = HybridExecuteResponse.from_dict(_wire({"cost_usd": 0.5}))
    junk = HybridExecuteResponse.from_dict(
        _wire({"total_cost_usd": float("nan"), "cost_usd": True})
    )

    assert omitted.get_reported_total_cost() is None
    assert zero.get_reported_total_cost() == 0.0
    assert legacy_key.get_reported_total_cost() == 0.5
    assert junk.get_reported_total_cost() is None
    # The legacy accessor keeps its 0.0 default for callers that want it.
    assert omitted.get_total_cost() == 0.0


# --- the omitted-cost response is unmeasured ---------------------------------


@pytest.mark.asyncio
async def test_omitted_cost_is_flagged_and_claims_no_total() -> None:
    result = await _run(_wire({"latency_ms": 10.0}))
    agg = result.aggregated_metrics

    _assert_unmeasured(agg)
    assert all(r.cost_unmeasured for r in result.example_results)
    stats = result.summary_stats or {}
    assert "cost" not in (stats.get("metrics") or {})
    assert "total_cost" not in (stats.get("metrics") or {})


@pytest.mark.asyncio
async def test_omitted_cost_in_combined_mode_is_unmeasured() -> None:
    result = await _run(_wire({"latency_ms": 10.0}, quality_metrics={"accuracy": 1.0}))

    _assert_unmeasured(result.aggregated_metrics)


@pytest.mark.asyncio
async def test_omitted_cost_in_two_phase_mode_is_unmeasured() -> None:
    evaluator = _evaluator(
        [_wire({"latency_ms": 10.0})],
        batch_size=2,
        supports_evaluate=True,
        evaluate_wire={
            "request_id": "r1",
            "status": "completed",
            "results": [
                {"example_id": f"ex_{i}", "metrics": {"accuracy": 1.0}}
                for i in range(2)
            ],
            "aggregate_metrics": {},
        },
    )
    result = await evaluator.evaluate(lambda v: v, {}, _dataset())

    _assert_unmeasured(result.aggregated_metrics)


@pytest.mark.asyncio
async def test_non_finite_per_item_cost_is_unmeasured_not_nan() -> None:
    result = await _run(_wire({}, {"cost_usd": float("nan")}))
    agg = result.aggregated_metrics

    _assert_unmeasured(agg)
    assert not any(math.isnan(r.cost_usd) for r in result.example_results)


@pytest.mark.asyncio
async def test_progress_callback_claims_no_running_total_when_unmeasured() -> None:
    seen: list[dict[str, Any]] = []

    await _run(
        _wire({"latency_ms": 10.0}),
        progress_callback=lambda step, info: seen.append(info),
    )

    assert seen
    assert all("total_cost" not in info for info in seen)
    assert all(info.get("cost_unmeasured") == 1.0 for info in seen)


@pytest.mark.asyncio
async def test_progress_callback_keeps_reported_running_total() -> None:
    seen: list[dict[str, Any]] = []

    await _run(
        _wire({"total_cost_usd": 0.25}),
        progress_callback=lambda step, info: seen.append(info),
    )

    assert [info["total_cost"] for info in seen] == [pytest.approx(0.25)]
    assert all("cost_unmeasured" not in info for info in seen)


# --- downstream consumers keep it unmeasured ---------------------------------


@pytest.mark.asyncio
async def test_omitted_cost_survives_trial_assembly_without_a_total() -> None:
    result = await _run(_wire({"latency_ms": 10.0}))

    attempted, total_cost = extract_cost_from_results(result, None, "t1")
    trial = build_success_result(
        trial_id="t1",
        evaluation_config={},
        eval_result=result,
        duration=1.0,
        examples_attempted=attempted,
        total_cost=total_cost,
        optuna_trial_id=None,
    )

    assert total_cost is None
    assert trial.metrics["cost_unmeasured"] == 1.0
    assert "cost" not in trial.metrics
    assert "total_cost" not in trial.metrics


@pytest.mark.asyncio
async def test_omitted_cost_is_not_backfilled_at_budget_submission() -> None:
    result = await _run(_wire({"latency_ms": 10.0}))
    metrics = dict(result.aggregated_metrics)
    client = SimpleNamespace(_cost_budget_armed_sessions={"s1"})

    mutated = ensure_cost_metric_for_budgeted_completed_submission(
        client=client, session_id="s1", metrics=metrics, status="COMPLETED"
    )

    assert mutated is False
    assert "cost" not in metrics


@pytest.mark.asyncio
async def test_omitted_cost_loses_to_measured_cost_in_pareto() -> None:
    omitted = (await _run(_wire({"latency_ms": 10.0}))).aggregated_metrics
    measured = {"accuracy": 1.0, "cost": 0.50}
    calc = ParetoFrontCalculator(maximize={"accuracy": True, "cost": False})

    front = calc.calculate_pareto_front(
        [
            _trial("omitted", {"accuracy": 1.0, **omitted}),
            _trial("measured", measured),
        ],
        ["accuracy", "cost"],
    )

    assert [p.trial.trial_id for p in front] == ["measured"]


# --- mixed trials: a floor is not a total ------------------------------------


@pytest.mark.asyncio
async def test_one_priced_batch_and_one_omitted_batch_is_unmeasured() -> None:
    result = await _run_batches(
        [_wire({"total_cost_usd": 0.25}, n=1), _wire({"latency_ms": 1.0}, n=1)]
    )

    _assert_unmeasured(result.aggregated_metrics)


@pytest.mark.asyncio
async def test_failed_batch_without_cost_does_not_poison_a_priced_trial() -> None:
    failed = _wire({}, n=1, status="failed")
    failed["error"] = {"code": "boom", "message": "agent crashed"}

    result = await _run_batches([_wire({"total_cost_usd": 0.25}, n=1), failed])
    agg = result.aggregated_metrics

    assert not agg.get("cost_unmeasured")
    assert agg["cost"] == pytest.approx(0.25)
    assert agg["total_cost"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_every_example_errored_is_unmeasured_not_free() -> None:
    failed = _wire({}, status="failed")
    failed["error"] = {"code": "boom", "message": "agent crashed"}

    result = await _run(failed)

    assert result.successful_examples == 0
    _assert_unmeasured(result.aggregated_metrics)


# --- reported values are untouched (controls) --------------------------------


@pytest.mark.asyncio
async def test_reported_zero_is_a_measured_zero() -> None:
    result = await _run(_wire({"total_cost_usd": 0.0}))
    agg = result.aggregated_metrics

    assert agg["cost"] == 0.0
    assert agg["total_cost"] == 0.0
    assert not agg.get("cost_unmeasured")


@pytest.mark.asyncio
async def test_reported_cost_is_unchanged() -> None:
    result = await _run(_wire({"total_cost_usd": 0.25}))
    agg = result.aggregated_metrics

    assert agg["cost"] == pytest.approx(0.25)
    assert agg["total_cost"] == pytest.approx(0.25)
    assert not agg.get("cost_unmeasured")


@pytest.mark.asyncio
async def test_per_item_cost_without_response_total_is_measured() -> None:
    result = await _run(_wire({}, {"cost_usd": 0.1}))
    agg = result.aggregated_metrics

    assert agg["cost"] == pytest.approx(0.2)
    assert not agg.get("cost_unmeasured")


@pytest.mark.asyncio
async def test_per_item_reported_zero_is_a_measured_zero() -> None:
    result = await _run(_wire({}, {"cost_usd": 0.0}))
    agg = result.aggregated_metrics

    assert agg["cost"] == 0.0
    assert not agg.get("cost_unmeasured")
