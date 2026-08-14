"""Winner-stability rerun tests (opt-in ``winner_stability_reps``).

Measured-only contract: with ``winner_stability_reps=3`` a completed run
carries ``result.metadata["winner_stability"]`` with 3 scores measured by
re-executing the already-selected winner through the normal trial execution
path; with the default ``0`` the key is absent and no extra evaluation runs.
The rerun never enters ``result.trials`` and never changes which config wins.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest

from traigent.api.types import OptimizationStatus, TrialResult
from traigent.config.types import TraigentConfig
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import (
    OptimizationError,
    RateLimitError,
)


class StabilityOptimizer(BaseOptimizer):
    """Deterministic optimizer: suggests param1=0,1,2 then stops."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self._suggest_count = 0
        self._max_suggestions = 3

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        config = {"param1": self._suggest_count}
        self._suggest_count += 1
        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        return self._suggest_count >= self._max_suggestions

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:
        return None

    def is_finished(self) -> bool:
        return self._suggest_count >= self._max_suggestions


class StabilityEvaluator(BaseEvaluator):
    """Deterministic evaluator: accuracy = 0.5 + param1 * 0.1; counts calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_count = 0
        self.evaluated_configs: list[dict[str, Any]] = []
        self.should_fail = False

    async def evaluate(
        self,
        func: Callable,
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease=None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
        **_kwargs,
    ) -> EvaluationResult:
        self.evaluation_count += 1
        self.evaluated_configs.append(dict(config))
        if self.should_fail:
            raise OptimizationError(f"Evaluation failed for config: {config}")

        processed = 0
        for index, _example in enumerate(dataset.examples):
            if sample_lease and not sample_lease.try_take(1):
                break
            processed += 1
            if progress_callback:
                progress_callback(index, {"success": True})

        accuracy = 0.5 + config.get("param1", 0) * 0.1
        metrics = {"accuracy": accuracy, "examples_attempted": processed}
        result = EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=processed,
            successful_examples=processed,
            duration=0.01,
            metrics=metrics,
            outputs=[f"output_{i}" for i in range(processed)],
            errors=[None for _ in range(processed)],
        )
        result.sample_budget_exhausted = False
        result.examples_consumed = processed
        return result


class AccountingStabilityEvaluator(StabilityEvaluator):
    """Stability evaluator that exposes a deterministic per-trial cost."""

    def __init__(self, *, cost: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.cost = cost
        self.fail_on_calls: set[int] = set()

    async def evaluate(self, *args: Any, **kwargs: Any) -> EvaluationResult:
        if self.evaluation_count + 1 in self.fail_on_calls:
            self.evaluation_count += 1
            raise OptimizationError("deterministic stability rerun failure")
        result = await super().evaluate(*args, **kwargs)
        result.metrics["total_cost"] = self.cost
        result.aggregated_metrics["total_cost"] = self.cost
        return result


@pytest.fixture(autouse=True)
def isolated_optimization_logs(monkeypatch, tmp_path):
    """Keep orchestrator logging isolated from developer-local log history."""
    monkeypatch.setenv(
        "TRAIGENT_OPTIMIZATION_LOG_DIR",
        str(tmp_path / "optimization_logs"),
    )


@pytest.fixture
def sample_dataset() -> Dataset:
    examples = [
        EvaluationExample({"query": "Hello"}, "Hi there!"),
        EvaluationExample({"query": "Goodbye"}, "See you later!"),
    ]
    return Dataset(examples, name="test_dataset", description="Test dataset")


@pytest.fixture
def mock_function():
    async def test_function(input_data: dict[str, Any], **config) -> Any:
        return input_data.get("query", "default response")

    return test_function


def _build_orchestrator(
    evaluator: StabilityEvaluator, **extra_kwargs: Any
) -> OptimizationOrchestrator:
    optimizer = StabilityOptimizer({"param1": [0, 1, 2]}, ["accuracy"])
    return OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=3,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        **extra_kwargs,
    )


class TestWinnerStabilityRerun:
    @pytest.mark.asyncio
    async def test_block_appears_with_three_scores(self, sample_dataset, mock_function):
        """reps=3: the block records exactly 3 measured rerun scores."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator, winner_stability_reps=3)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        block = result.metadata["winner_stability"]
        assert block["reps"] == 3
        assert block["scores"] == [0.7, 0.7, 0.7]
        assert block["mean"] == pytest.approx(0.7)
        assert block["std"] == pytest.approx(0.0)
        assert isinstance(block["config_hash"], str) and block["config_hash"]
        # evaluated_at must be a parseable ISO-8601 timestamp.
        datetime.fromisoformat(block["evaluated_at"])
        # 3 search evaluations + 3 winner reruns, no more, no fewer.
        assert evaluator.evaluation_count == 6
        # Every rerun executed the winning config through the normal path.
        assert [c.get("param1") for c in evaluator.evaluated_configs[3:]] == [2, 2, 2]

    @pytest.mark.asyncio
    async def test_absent_by_default(self, sample_dataset, mock_function):
        """Default (0): no block, no extra evaluations."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert "winner_stability" not in result.metadata
        assert evaluator.evaluation_count == 3

    @pytest.mark.asyncio
    async def test_rerun_stays_out_of_trials_and_selection(
        self, sample_dataset, mock_function
    ):
        """The rerun never enters result.trials and never moves the winner."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator, winner_stability_reps=2)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert len(result.trials) == 3
        assert result.best_config is not None
        assert result.best_config.get("param1") == 2
        assert result.best_score == pytest.approx(0.7)
        assert result.metadata["winner_stability"]["reps"] == 2

    @pytest.mark.asyncio
    async def test_reruns_use_permits_and_are_included_in_total_cost(
        self, sample_dataset, mock_function
    ):
        """Rerun spend is tracked without adding reruns to selection history."""
        evaluator = AccountingStabilityEvaluator()
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            cost_limit=10,
            cost_approved=True,
        )

        result = await orchestrator.optimize(mock_function, sample_dataset)

        status = orchestrator.cost_enforcer.get_status()
        assert status.trial_count == 5
        assert status.accumulated_cost_usd == pytest.approx(1.25)
        assert status.in_flight_count == 0
        assert result.total_cost == pytest.approx(1.25)
        assert result.metrics["total_cost"] == pytest.approx(1.25)
        assert len(result.trials) == 3

    @pytest.mark.asyncio
    async def test_reruns_debit_execution_budget_cost_trials_and_samples(
        self, sample_dataset, mock_function
    ):
        """Shared budget accounting includes stability reruns, not their history."""
        evaluator = AccountingStabilityEvaluator()
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            max_total_examples=20,
            cost_limit=10,
            cost_approved=True,
        )
        budget = ExecutionBudget(max_cost_usd=10, max_examples=20)
        orchestrator.execution_budget = budget

        result = await orchestrator.optimize(mock_function, sample_dataset)

        snapshot = budget.snapshot()
        assert snapshot.trials == 5
        assert snapshot.consumed_cost == pytest.approx(1.25)
        assert snapshot.consumed_examples == 10
        assert snapshot.cost_tracking == "complete"
        assert len(result.trials) == 3

    @pytest.mark.asyncio
    async def test_reruns_respect_sample_budget(self, sample_dataset, mock_function):
        """Stability reruns consume the shared sample pool without overspending it."""
        evaluator = AccountingStabilityEvaluator()
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            max_total_examples=8,
        )

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert orchestrator._sample_budget_manager is not None
        assert orchestrator._sample_budget_manager.consumed() == 8
        assert orchestrator._sample_budget_manager.remaining() == 0
        assert len(result.trials) == 3

    @pytest.mark.asyncio
    async def test_partial_reruns_still_account_failed_attempts(
        self, sample_dataset, mock_function
    ):
        """A failed measured rerun counts spend but does not create stability evidence."""
        evaluator = AccountingStabilityEvaluator()
        evaluator.fail_on_calls = {4}
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            cost_limit=10,
            cost_approved=True,
        )
        budget = ExecutionBudget(max_cost_usd=10, max_examples=20)
        orchestrator.execution_budget = budget

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.metadata["winner_stability"]["reps"] == 1
        assert evaluator.evaluation_count == 5
        status = orchestrator.cost_enforcer.get_status()
        assert status.trial_count == 5
        assert status.accumulated_cost_usd == pytest.approx(1.0)
        snapshot = budget.snapshot()
        assert snapshot.trials == 5
        assert snapshot.cost_tracking == "partial"
        assert result.total_cost == pytest.approx(1.0)
        assert len(result.trials) == 3

    @pytest.mark.asyncio
    async def test_shared_execution_budget_stops_before_second_rerun(
        self, sample_dataset, mock_function
    ):
        """A stability rerun cannot start after the shared cost cap is spent."""
        evaluator = AccountingStabilityEvaluator()
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            cost_limit=10,
            cost_approved=True,
        )
        budget = ExecutionBudget(max_cost_usd=1.0, max_examples=20)
        orchestrator.execution_budget = budget

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert evaluator.evaluation_count == 4
        assert [c.get("param1") for c in evaluator.evaluated_configs[3:]] == [2]
        assert result.metadata["winner_stability"]["reps"] == 1
        assert len(result.trials) == 3
        assert budget.snapshot().trials == 4

    @pytest.mark.asyncio
    async def test_rethrown_vendor_failure_is_accounted_as_untracked_attempt(
        self, sample_dataset, mock_function
    ):
        """A lifecycle vendor pause consumes a trial slot without fake cost."""
        evaluator = AccountingStabilityEvaluator()
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            cost_limit=10,
            cost_approved=True,
        )
        budget = ExecutionBudget(max_cost_usd=10, max_examples=20)
        orchestrator.execution_budget = budget
        original_evaluate = evaluator.evaluate

        async def pause_after_dispatch(*args: Any, **kwargs: Any):
            if evaluator.evaluation_count >= 3:
                evaluator.evaluation_count += 1
                raise RateLimitError("provider rate limit")
            return await original_evaluate(*args, **kwargs)

        evaluator.evaluate = pause_after_dispatch  # type: ignore[method-assign]

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert "winner_stability" not in result.metadata
        assert len(result.trials) == 3
        assert evaluator.evaluation_count == 4
        status = orchestrator.cost_enforcer.get_status()
        assert status.trial_count == 4
        assert status.accumulated_cost_usd == pytest.approx(0.75)
        assert status.unknown_cost_mode is True
        snapshot = budget.snapshot()
        assert snapshot.trials == 4
        assert snapshot.untracked_trials == 1
        assert snapshot.consumed_cost == pytest.approx(0.75)
        assert snapshot.cost_tracking == "partial"

    @pytest.mark.asyncio
    async def test_cancellation_releases_stability_permit(
        self, sample_dataset, mock_function
    ):
        """Pre-dispatch cancellation releases its permit without an attempt debit."""
        evaluator = AccountingStabilityEvaluator()
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            cost_limit=10,
            cost_approved=True,
        )
        budget = ExecutionBudget(max_cost_usd=10, max_examples=20)
        orchestrator.execution_budget = budget
        original_run_trial = orchestrator._trial_lifecycle.run_trial

        async def cancel_after_search(*args: Any, **kwargs: Any):
            if evaluator.evaluation_count >= 3:
                raise asyncio.CancelledError
            return await original_run_trial(*args, **kwargs)

        orchestrator._trial_lifecycle.run_trial = cancel_after_search

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.CANCELLED
        status = orchestrator.cost_enforcer.get_status()
        # The wrapper cancels before the real lifecycle/evaluator dispatch, so
        # no provider attempt exists to account for.
        assert evaluator.evaluation_count == 3
        assert status.trial_count == 3
        assert status.unknown_cost_mode is False
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == pytest.approx(0.0)
        snapshot = budget.snapshot()
        assert snapshot.trials == 3
        assert snapshot.untracked_trials == 0
        assert snapshot.cost_tracking == "complete"
        assert len(result.trials) == 3

    @pytest.mark.asyncio
    async def test_post_dispatch_cancellation_accounts_untracked_stability_attempt(
        self, sample_dataset, mock_function
    ):
        """Cancellation from the evaluator accounts the dispatched rerun once."""
        evaluator = AccountingStabilityEvaluator()
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            cost_limit=10,
            cost_approved=True,
        )
        budget = ExecutionBudget(max_cost_usd=10, max_examples=20)
        orchestrator.execution_budget = budget
        original_evaluate = evaluator.evaluate

        async def cancel_from_dispatched_evaluator(*args: Any, **kwargs: Any):
            if evaluator.evaluation_count >= 3:
                evaluator.evaluation_count += 1
                raise asyncio.CancelledError
            return await original_evaluate(*args, **kwargs)

        evaluator.evaluate = cancel_from_dispatched_evaluator  # type: ignore[method-assign]

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.CANCELLED
        assert evaluator.evaluation_count == 4
        status = orchestrator.cost_enforcer.get_status()
        assert status.trial_count == 4
        assert status.unknown_cost_mode is True
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == pytest.approx(0.0)
        snapshot = budget.snapshot()
        assert snapshot.trials == 4
        assert snapshot.untracked_trials == 1
        assert snapshot.cost_tracking == "partial"
        assert len(result.trials) == 3

    @pytest.mark.asyncio
    async def test_cancellation_waiting_for_rerun_accounting_debits_once(
        self, sample_dataset, mock_function
    ):
        """Repeated cancellation still drains and debits a completed rerun once."""
        evaluator = AccountingStabilityEvaluator()
        orchestrator = _build_orchestrator(
            evaluator,
            winner_stability_reps=2,
            cost_limit=10,
            cost_approved=True,
        )
        budget = ExecutionBudget(max_cost_usd=10, max_examples=20)
        orchestrator.execution_budget = budget
        original_run_trial = orchestrator._trial_lifecycle.run_trial
        release_accounting = asyncio.Event()

        class TrackingLock:
            """Expose the second lock entrant without changing lock semantics."""

            def __init__(self) -> None:
                self._lock = asyncio.Lock()
                self.entry_attempted = asyncio.Event()

            async def __aenter__(self) -> TrackingLock:
                self.entry_attempted.set()
                await self._lock.acquire()
                return self

            async def __aexit__(self, *args: Any) -> None:
                self._lock.release()

        async def return_rerun_while_accounting_lock_is_held(
            *args: Any, **kwargs: Any
        ) -> TrialResult:
            rerun_trial = await original_run_trial(*args, **kwargs)
            if evaluator.evaluation_count == 4:
                tracking_lock = TrackingLock()
                orchestrator._state_lock = tracking_lock

                async def hold_accounting_lock() -> None:
                    async with tracking_lock:
                        await release_accounting.wait()

                holder = asyncio.create_task(hold_accounting_lock())
                await tracking_lock.entry_attempted.wait()
                tracking_lock.entry_attempted.clear()
                orchestrator._winner_stability_test_holder = holder
            return rerun_trial

        orchestrator._trial_lifecycle.run_trial = (  # type: ignore[method-assign]
            return_rerun_while_accounting_lock_is_held
        )
        optimization_task = asyncio.create_task(
            orchestrator.optimize(mock_function, sample_dataset)
        )

        try:
            while not hasattr(orchestrator, "_winner_stability_test_holder"):
                await asyncio.sleep(0)
            tracking_lock = orchestrator._state_lock
            await tracking_lock.entry_attempted.wait()
            optimization_task.cancel()
            await asyncio.sleep(0)
            assert not optimization_task.done()
            optimization_task.cancel()
            await asyncio.sleep(0)
            assert not optimization_task.done()
            release_accounting.set()
            result = await optimization_task
        finally:
            release_accounting.set()
            holder = getattr(orchestrator, "_winner_stability_test_holder", None)
            if holder is not None:
                await holder

        assert result.status == OptimizationStatus.CANCELLED
        status = orchestrator.cost_enforcer.get_status()
        assert status.trial_count == 4
        assert status.accumulated_cost_usd == pytest.approx(1.0)
        assert status.unknown_cost_mode is False
        assert status.in_flight_count == 0
        assert status.reserved_cost_usd == pytest.approx(0.0)
        snapshot = budget.snapshot()
        assert snapshot.trials == 4
        assert snapshot.consumed_cost == pytest.approx(1.0)
        assert snapshot.untracked_trials == 0
        assert snapshot.cost_tracking == "complete"
        assert result.total_cost == pytest.approx(status.accumulated_cost_usd)
        assert result.metrics["total_cost"] == pytest.approx(
            status.accumulated_cost_usd
        )
        assert len(result.trials) == 3

    @pytest.mark.asyncio
    async def test_tracing_helper_accepts_legacy_direct_call_without_dispatch_callback(
        self, sample_dataset, mock_function
    ):
        """The internal tracing helper keeps its optional callback default."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator)

        trial = await orchestrator._trial_lifecycle._execute_trial_with_tracing(
            func=mock_function,
            dataset=sample_dataset,
            trial_id="legacy-direct-tracing-call",
            backend_trial_id=None,
            evaluation_config={"param1": 0},
            start_time=time.time(),
            optuna_trial_id=None,
            progress_callback=None,
            progress_state=None,
            lease=None,
            span=None,
        )

        assert trial.is_successful
        assert evaluator.evaluation_count == 1

    @pytest.mark.asyncio
    async def test_no_winner_skips_the_rerun(self, sample_dataset, mock_function):
        """A run with no selected winner records no block and spends nothing."""
        evaluator = StabilityEvaluator()
        evaluator.should_fail = True
        orchestrator = _build_orchestrator(evaluator, winner_stability_reps=3)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.best_config is None
        assert "winner_stability" not in result.metadata
        # Only the 3 (failed) search evaluations ran; zero rerun spend.
        assert evaluator.evaluation_count == 3

    @pytest.mark.asyncio
    async def test_rerun_failure_never_fails_the_run(
        self, sample_dataset, mock_function
    ):
        """A rerun that starts failing keeps the run COMPLETED, block-free."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator, winner_stability_reps=2)

        original_evaluate = evaluator.evaluate

        async def failing_after_search(*args: Any, **kwargs: Any):
            if evaluator.evaluation_count >= 3:
                evaluator.evaluation_count += 1
                raise OptimizationError("provider went away mid-rerun")
            return await original_evaluate(*args, **kwargs)

        evaluator.evaluate = failing_after_search  # type: ignore[method-assign]

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert result.best_config is not None
        # Nothing was measured, so no block is recorded (measured-only).
        assert "winner_stability" not in result.metadata
