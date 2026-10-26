"""Caller-owned reservations through real SDK evaluators; no provider calls."""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import asdict
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from tests.integration.test_execution_budget_multiphase import (
    MockSequentialOptimizer,
    _build_orchestrator,
    _dataset,
)
from traigent import get_config, with_usage
from traigent.api.decorators import optimize
from traigent.api.types import ExampleResult, TrialStatus
from traigent.cloud.session_types import SessionCreationResult
from traigent.config.types import TraigentConfig
from traigent.core.execution_budget import BudgetDebitExceeded, ExecutionBudget
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.exceptions import TrialPrunedError


@pytest.fixture(autouse=True)
def isolated_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for key in (
        "TRAIGENT_RUN_COST_LIMIT",
        "TRAIGENT_STRICT_COST_ACCOUNTING",
        "TRAIGENT_REQUIRE_COST_TRACKING",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_AUTO_SYNC", "false")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")

    def refuse_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Budget tests must not contact a provider or backend")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [True, False])
@pytest.mark.parametrize("concurrency", [1, 2])
async def test_signal_survives_evaluator_lanes(sync: bool, concurrency: int) -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.01)
    with pytest.raises(BudgetDebitExceeded) as original:
        budget.reserve_debit(0.05)
    signal = original.value

    def thread_agent(value: int) -> int:
        raise signal

    async def async_agent(value: int) -> int:
        raise signal

    dataset = Dataset([EvaluationExample({"value": 1}, 1)])
    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=concurrency)
    assert evaluator.max_workers == concurrency
    with pytest.raises(BudgetDebitExceeded) as propagated:
        await evaluator.evaluate(thread_agent if sync else async_agent, {}, dataset)
    assert propagated.value is signal


@pytest.mark.asyncio
@pytest.mark.timeout(3)
@pytest.mark.parametrize("pruning", ["budget", "ordinary"])
async def test_concurrent_pruning_attaches_each_partial_row_once(pruning: str) -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.1 if pruning == "budget" else 1)
    second_started = asyncio.Event()
    second_cancelled = asyncio.Event()
    never = asyncio.Event()
    calls: list[int] = []
    raised: list[TrialPrunedError] = []
    supplied_row = ExampleResult(
        example_id="nested-partial",
        input_data={"value": 99},
        expected_output=99,
        actual_output=99,
        metrics={"accuracy": 1.0},
        execution_time=0.0,
        success=True,
    )
    ordinary_signal = TrialPrunedError("ordinary prune", example_results=[supplied_row])

    async def agent(value: int) -> int:
        try:
            if value == 3 and pruning == "ordinary":
                raise ordinary_signal
            with budget.reserve_debit(0.05):
                calls.append(value)
                if value == 1:
                    await second_started.wait()
                elif value == 2:
                    second_started.set()
                    try:
                        await never.wait()
                    except asyncio.CancelledError:
                        second_cancelled.set()
                        raise
                return value
        except TrialPrunedError as exc:
            raised.append(exc)
            raise

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=2, detailed=True)
    assert evaluator.max_workers == 2
    dataset = Dataset([EvaluationExample({"value": i}, i) for i in (1, 2, 3)])
    expected_error = BudgetDebitExceeded if pruning == "budget" else TrialPrunedError
    with pytest.raises(expected_error) as propagated:
        await asyncio.wait_for(evaluator.evaluate(agent, {}, dataset, budget=budget), 3)
    assert propagated.value is raised[0]
    assert calls == [1, 2]
    assert second_cancelled.is_set()
    rows = propagated.value.example_results
    if pruning == "budget":
        assert len(rows) == 1
        assert [row.input_data for row in rows] == [{"value": 1}]
    else:
        assert propagated.value is ordinary_signal
        assert len(rows) == 2
        assert [row.input_data for row in rows] == [{"value": 1}, {"value": 99}]
        assert rows[1] is supplied_row
    assert rows[0].actual_output == 1
    assert rows[0].success
    assert budget.debit_snapshot().consumed_budget_debit_usd == pytest.approx(0.1)
    assert budget.debit_snapshot().reserved_budget_debit_usd == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluator_type", [LocalEvaluator, SimpleScoringEvaluator])
async def test_depletion_retains_partial_rows_and_completed_trials(
    evaluator_type: type,
) -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.16)
    calls = 0

    async def agent(query: str, **_kwargs: Any) -> str:
        nonlocal calls
        with budget.reserve_debit(0.05):
            calls += 1
            return query.replace("q", "a")

    orchestrator = _build_orchestrator(budget, max_suggestions=5, max_trials=5)
    orchestrator.evaluator = (
        LocalEvaluator(metrics=["accuracy"], max_workers=1, detailed=True)
        if evaluator_type is LocalEvaluator
        else SimpleScoringEvaluator(lambda output, expected: float(output == expected))
    )
    result = await orchestrator.optimize(func=agent, dataset=_dataset())
    assert result.stop_reason == "execution_budget"
    assert calls == 3
    assert len(result.trials) == 2
    assert result.trials[0].status == TrialStatus.COMPLETED
    assert result.trials[1].status == TrialStatus.PRUNED
    assert len(result.trials[1].metadata["example_results"]) == 1
    snap = budget.debit_snapshot()
    assert snap.consumed_budget_debit_usd == pytest.approx(0.15)
    assert snap.reserved_budget_debit_usd == 0

    # Denial is run-local, so a real later phase can fund its smaller call.
    async def holdout_agent(query: str, **_kwargs: Any) -> str:
        with budget.reserve_debit(0.01):
            return query.replace("q", "a")

    holdout = _build_orchestrator(budget, max_suggestions=1, max_trials=1)
    holdout.evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    holdout_result = await holdout.optimize(func=holdout_agent, dataset=_dataset(1))
    assert len(holdout_result.trials) == 1
    assert holdout_result.trials[0].status == TrialStatus.COMPLETED
    assert budget.debit_exhausted


@pytest.mark.asyncio
async def test_parallel_trials_settle_admitted_work_and_stop() -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.16)
    calls = 0

    async def agent(query: str, **_kwargs: Any) -> str:
        nonlocal calls
        with budget.reserve_debit(0.05):
            calls += 1
            await asyncio.sleep(0)
            return query.replace("q", "a")

    orchestrator = _build_orchestrator(
        budget, max_suggestions=8, max_trials=8, parallel_trials=2
    )
    orchestrator.evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    result = await orchestrator.optimize(func=agent, dataset=_dataset())
    assert result.stop_reason == "execution_budget"
    assert calls == 3
    assert budget.debit_snapshot().consumed_budget_debit_usd == pytest.approx(0.15)
    assert budget.debit_snapshot().reserved_budget_debit_usd == 0


@pytest.mark.asyncio
async def test_ordinary_pruning_does_not_become_budget_stop() -> None:
    budget = ExecutionBudget(max_budget_debit_usd=1)

    async def agent(query: str, **_kwargs: Any) -> str:
        raise TrialPrunedError("ordinary evaluator prune")

    orchestrator = _build_orchestrator(budget, max_suggestions=2, max_trials=2)
    orchestrator.evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    result = await orchestrator.optimize(func=agent, dataset=_dataset())
    assert result.stop_reason != "execution_budget"
    assert len(result.trials) == 2
    assert all(trial.status == TrialStatus.PRUNED for trial in result.trials)
    assert budget.debit_snapshot().consumed_budget_debit_usd == 0


@pytest.mark.asyncio
async def test_async_contention_and_cancellation_retain_allocations() -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.1)
    admitted = asyncio.Event()

    async def provider_door() -> None:
        with budget.reserve_debit(0.1):
            admitted.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(provider_door())
    await admitted.wait()
    with pytest.raises(BudgetDebitExceeded):
        budget.reserve_debit(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert budget.debit_snapshot().consumed_budget_debit_usd == pytest.approx(0.1)
    assert budget.debit_snapshot().reserved_budget_debit_usd == 0


@pytest.mark.asyncio
async def test_native_sdk_four_unpriced_calls_and_local_only_state() -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.21)
    calls = 0

    @optimize(
        eval_dataset=[
            {"input": {"question": "q1"}, "expected_output": "A"},
            {"input": {"question": "q2"}, "expected_output": "A"},
        ],
        configuration_space={"temperature": [0.1, 0.2]},
        objectives=["accuracy"],
        offline=True,
        injection_mode="seamless",
    )
    def agent(question: str, temperature: float = 0.1, **_kwargs: Any) -> str:
        nonlocal calls
        with budget.reserve_debit(0.05):
            calls += 1
            return "A"  # no fabricated usage or price

    result = await agent.optimize(budget=budget, progress_bar=False)
    assert calls == 4
    assert result.best_config is not None
    assert result.total_cost is None
    assert result.get_summary()["total_cost"] is None
    assert all(trial.metrics.get("total_cost") is None for trial in result.trials)
    with pytest.raises(BudgetDebitExceeded):
        budget.reserve_debit(0.05)
    nominal = asdict(budget.debit_snapshot())
    assert nominal["consumed_budget_debit_usd"] == pytest.approx(0.2)
    assert nominal["remaining_budget_debit_usd"] == pytest.approx(0.01)
    serialized = json.dumps(asdict(result), default=str)
    for name in nominal:
        assert f'"{name}":' not in serialized
    assert result.metadata["execution_budget"]["cost_tracking"] == "untracked"


@pytest.mark.asyncio
async def test_shared_phases_do_not_auto_charge_nominal_debits() -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.21)
    calls = 0

    async def agent(query: str, **_kwargs: Any) -> str:
        nonlocal calls
        with budget.reserve_debit(0.05):
            calls += 1
            return query.replace("q", "a")

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    await evaluator.evaluate(agent, {}, _dataset(1), budget=budget)
    orchestrator = _build_orchestrator(budget, max_suggestions=1, max_trials=1)
    orchestrator.evaluator = evaluator
    await orchestrator.optimize(func=agent, dataset=_dataset(2))
    await evaluator.evaluate(agent, {}, _dataset(1), budget=budget)
    assert calls == 4
    assert budget.debit_snapshot().consumed_budget_debit_usd == pytest.approx(0.2)
    assert budget.consumed_cost == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reported_cost", [None, 0.0, 0.125])
async def test_debits_do_not_supply_or_change_observed_cost(
    reported_cost: float | None,
    request: pytest.FixtureRequest,
) -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.2)
    use_debits = False

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        configuration_space={"temperature": [0.1, 0.2]},
        objectives=["accuracy"],
        offline=True,
    )
    def agent(question: str, **_kwargs: Any) -> Any:
        get_config()["temperature"]
        lease = budget.reserve_debit(0.05) if use_debits else None
        try:
            return (
                "A"
                if reported_cost is None
                else with_usage("A", total_cost=reported_cost)
            )
        finally:
            if lease is not None:
                lease.settle_debit(reported_cost)

    baseline = await agent.optimize(max_trials=1, progress_bar=False)
    use_debits = True
    result = await agent.optimize(budget=budget, max_trials=1, progress_bar=False)
    assert result.total_cost == baseline.total_cost
    assert result.get_summary()["total_cost"] == baseline.get_summary()["total_cost"]
    assert result.trials[0].metrics.get("total_cost") == reported_cost
    if reported_cost is not None:
        assert not result.trials[0].metrics.get("cost_unmeasured")
        assert not result.trials[0].metrics.get("cost_unpriced")
    request.node.user_properties.append(
        (
            "actual_cost_oracle",
            json.dumps(
                {
                    "provider_reported": reported_cost,
                    "no_budget_aggregate": baseline.total_cost,
                    "with_debits_aggregate": result.total_cost,
                    "with_debits_trial": result.trials[0].metrics.get("total_cost"),
                    "cost_unmeasured": result.trials[0].metrics.get("cost_unmeasured"),
                }
            ),
        )
    )
    assert budget.consumed_cost == (reported_cost or 0)
    assert budget.debit_snapshot().consumed_budget_debit_usd == pytest.approx(
        0.05 if reported_cost is None else reported_cost
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["explicit_cap", "strict", "cost_objective"])
async def test_unknown_actual_cost_policies_are_not_relaxed(
    policy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = ExecutionBudget(max_budget_debit_usd=0.2)
    if policy == "strict":
        monkeypatch.setenv("TRAIGENT_REQUIRE_COST_TRACKING", "true")

    @optimize(
        eval_dataset=[{"input": {"question": "q"}, "expected_output": "A"}],
        configuration_space={"temperature": [0.1, 0.2]},
        objectives=["accuracy", "cost"] if policy == "cost_objective" else ["accuracy"],
        offline=True,
    )
    def agent(question: str, **_kwargs: Any) -> str:
        get_config()["temperature"]
        with budget.reserve_debit(0.05):
            return "A"

    kwargs = {"cost_limit": 1.0} if policy == "explicit_cap" else {}
    with pytest.raises(Exception) as rejected:
        await agent.optimize(budget=budget, max_trials=1, progress_bar=False, **kwargs)
    chain: list[str] = []
    error: BaseException | None = rejected.value
    while error is not None:
        chain.append(type(error).__name__ + ": " + str(error))
        error = error.__cause__
    assert any("CostTrackingRequiredError" in message for message in chain)
    assert not isinstance(rejected.value, BudgetDebitExceeded)
    assert budget.debit_snapshot().consumed_budget_debit_usd == pytest.approx(0.05)
    assert budget.consumed_cost == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("allowance, consumed", [(0.21, 0.2), (0.16, 0.15)])
async def test_mock_connected_outgoing_payloads_exclude_local_ledger(
    monkeypatch: pytest.MonkeyPatch,
    allowance: float,
    consumed: float,
) -> None:
    """Capture real create/trial/finalize builders at a fake transport door."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    client = Mock()
    client.create_session.return_value = SessionCreationResult.connected(
        session_id="budget-session"
    )
    client.get_session_mapping.return_value = Mock(experiment_run_id="budget-run")
    client.auth_manager.has_api_key.return_value = True
    client.request_trial_slot = AsyncMock(
        side_effect=["budget-trial-1", "budget-trial-2"]
    )
    client.register_trial_start = AsyncMock()
    client._submit_trial_result_via_session = AsyncMock(return_value=True)
    client.update_trial_weighted_scores = AsyncMock(return_value=True)
    client.finalize_session_sync.return_value = {"status": "completed"}
    monkeypatch.setattr(
        "traigent.cloud.backend_client.BackendIntegratedClient",
        lambda **_kwargs: client,
    )
    budget = ExecutionBudget(max_budget_debit_usd=allowance)

    async def agent(query: str, **_kwargs: Any) -> str:
        with budget.reserve_debit(0.05):
            return query.replace("q", "a")

    orchestrator = OptimizationOrchestrator(
        optimizer=MockSequentialOptimizer(
            {"param1": (0, 100)}, ["accuracy"], max_suggestions=2
        ),
        evaluator=LocalEvaluator(metrics=["accuracy"], max_workers=1, detailed=True),
        config=TraigentConfig(offline=False, algorithm="grid", privacy_enabled=False),
        max_trials=2,
    )
    orchestrator.execution_budget = budget
    result = await orchestrator.optimize(func=agent, dataset=_dataset())
    assert client.create_session.call_count == 1
    assert client._submit_trial_result_via_session.call_count == 2
    assert client.finalize_session_sync.call_count == 1
    assert budget.debit_snapshot().consumed_budget_debit_usd == pytest.approx(consumed)
    if allowance == 0.16:
        assert result.stop_reason == "execution_budget"
        assert result.trials[1].status == TrialStatus.PRUNED
    calls = list(client.mock_calls)
    assert calls  # assertions above prevent a vacuous offline payload check
    serialized = json.dumps(
        [(name, args, kwargs) for name, args, kwargs in calls], default=str
    )
    for name in asdict(budget.debit_snapshot()):
        assert f'"{name}":' not in serialized
    assert '"reserved_debit_usd":' not in serialized
    assert "BudgetDebitSnapshot" not in serialized
    assert result.total_cost is None
    for call in client._submit_trial_result_via_session.call_args_list:
        assert call.kwargs["metrics"].get("cost") is None
        assert call.kwargs["metrics"].get("total_cost") is None
