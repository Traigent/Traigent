"""Stop-condition to public stop_reason contract tests."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, get_args

import pytest

from tests.shared.mocks.optimizers import MockOptimizer
from traigent.api.types import StopReason, TrialResult, TrialStatus
from traigent.core.exception_handler import VendorErrorCategory
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.parallel_execution_manager import PermittedTrialResult
from traigent.core.stop_conditions import (
    ExecutionBudgetStopCondition,
    StopCondition,
)
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.utils.exceptions import VendorPauseError


class NoopEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        func: Any,
        config: dict[str, Any],
        dataset: Dataset,
        **kwargs: Any,
    ) -> EvaluationResult:
        metrics = {"accuracy": 1.0}
        return EvaluationResult(
            config=config,
            example_results=[],
            aggregated_metrics=metrics,
            total_examples=0,
            successful_examples=0,
            duration=0.0,
            metrics=metrics,
        )


class AlwaysStopCondition(StopCondition):
    reason = "custom_user_condition"

    def reset(self) -> None:
        return

    def should_stop(self, trials: Any) -> bool:
        return True


class VendorPauseLifecycle:
    async def run_sequential_trial(self, **kwargs: Any) -> tuple[int, str]:
        raise VendorPauseError(
            "quota exhausted",
            category=VendorErrorCategory.QUOTA_EXHAUSTED,
        )


class StubPromptAdapter:
    def __init__(self, vendor_decision: str) -> None:
        self._vendor_decision = vendor_decision
        self.vendor_calls = 0

    def prompt_vendor_pause(self, error: Any, category: Any) -> str:
        self.vendor_calls += 1
        return self._vendor_decision

    def prompt_budget_pause(self, accumulated: float, limit: float) -> str:
        return "stop"


def _trial(
    trial_id: str,
    metrics: dict[str, Any] | None = None,
    status: TrialStatus = TrialStatus.COMPLETED,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={},
        metrics=metrics or {},
        status=status,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )


def _orchestrator(**kwargs: Any) -> OptimizationOrchestrator:
    return OptimizationOrchestrator(
        optimizer=MockOptimizer({"temperature": [0.1]}, ["accuracy"]),
        evaluator=NoopEvaluator(),
        **kwargs,
    )


def test_max_trials_condition_maps_to_public_stop_reason():
    orchestrator = _orchestrator(max_trials=1)
    orchestrator._trials = [_trial("t1")]

    assert orchestrator._should_stop(trial_count=1)
    assert orchestrator._stop_reason == "max_trials_reached"


def test_max_samples_condition_maps_to_public_stop_reason():
    orchestrator = _orchestrator(max_total_examples=3)
    orchestrator._trials = [_trial("t1", {"examples_attempted": 3})]

    assert orchestrator._should_stop(trial_count=1)
    assert orchestrator._stop_reason == "max_samples_reached"


def test_timeout_maps_to_public_stop_reason():
    orchestrator = _orchestrator(timeout=1.0)
    orchestrator._start_time = time.time() - 2.0

    assert orchestrator._should_stop(trial_count=0)
    assert orchestrator._stop_reason == "timeout"


def test_optimizer_stop_maps_to_public_stop_reason():
    orchestrator = _orchestrator(max_trials=10)
    orchestrator.optimizer.force_stop()

    assert orchestrator._should_stop(trial_count=0)
    assert orchestrator._stop_reason == "optimizer"


def test_metric_limit_maps_to_public_metric_limit_stop_reason():
    orchestrator = _orchestrator(metric_limit=0.2, metric_name="total_cost")
    orchestrator._trials = [_trial("t1", {"total_cost": 0.2})]

    assert orchestrator._should_stop(trial_count=1)
    assert orchestrator._stop_reason == "metric_limit"


def test_metric_limit_requires_metric_name():
    with pytest.raises(ValueError, match="metric_name is required"):
        _orchestrator(metric_limit=0.2)


def test_cost_enforcer_limit_maps_to_cost_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")
    orchestrator = _orchestrator(cost_limit=0.5, cost_approved=True)
    permit = orchestrator.cost_enforcer.acquire_permit()
    assert permit.is_granted
    orchestrator.cost_enforcer.track_cost(0.5, permit=permit)

    assert orchestrator._should_stop(trial_count=1)
    assert orchestrator._stop_reason == "cost_limit"


def test_plateau_condition_maps_to_public_stop_reason():
    schema = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition("accuracy", "maximize", 1.0)]
    )
    orchestrator = _orchestrator(
        objective_schema=schema,
        plateau_window=2,
        plateau_epsilon=0.0,
    )
    orchestrator._trials = [
        _trial("t1", {"accuracy": 0.8}),
        _trial("t2", {"accuracy": 0.8}),
    ]

    assert orchestrator._should_stop(trial_count=2)
    assert orchestrator._stop_reason == "plateau"


def test_hypervolume_convergence_maps_to_public_convergence_stop_reason():
    schema = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition("accuracy", "maximize", 1.0)]
    )
    orchestrator = _orchestrator(
        objective_schema=schema,
        convergence_metric="hypervolume_improvement",
        convergence_window=2,
        convergence_threshold=0.01,
    )
    orchestrator._trials = [
        _trial("t1", {"accuracy": 0.5}),
        _trial("t2", {"accuracy": 0.5}),
        _trial("t3", {"accuracy": 0.5}),
    ]

    assert orchestrator._should_stop(trial_count=3)
    assert orchestrator._stop_reason == "convergence"


def test_semantic_saturation_maps_to_public_stop_reason():
    schema = ObjectiveSchema.from_objectives(
        [ObjectiveDefinition("accuracy", "maximize", 1.0)]
    )
    orchestrator = _orchestrator(
        objective_schema=schema,
        semantic_saturation={
            "window": 3,
            "min_trials": 3,
            "continuous_objectives": [],
        },
    )
    orchestrator._trials = [
        _trial(
            "t1",
            {"accuracy": 1.0},
        ),
        _trial(
            "t2",
            {"accuracy": 1.0},
        ),
        _trial(
            "t3",
            {"accuracy": 1.0},
        ),
    ]
    for index, trial in enumerate(orchestrator._trials, start=1):
        trial.config = {"temperature": index}
        trial.metadata = {
            "example_results": [
                {"example_id": "ex1", "metrics": {"accuracy": 1.0}},
                {"example_id": "ex2", "metrics": {"accuracy": 1.0}},
            ]
        }

    assert orchestrator._should_stop(trial_count=3)
    assert orchestrator._stop_reason == "semantic_saturation"


def test_custom_stop_condition_maps_to_generic_condition_stop_reason():
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._stop_condition_manager.add_condition(AlwaysStopCondition())

    assert orchestrator._should_stop(trial_count=0)
    assert orchestrator._stop_reason == "condition"


# ---------------------------------------------------------------------------
# ExecutionBudget stop-reason contract (issue #1980)
# ---------------------------------------------------------------------------


def test_execution_budget_is_in_stop_reason_literal():
    """The public StopReason contract includes the cumulative budget reason."""
    assert "execution_budget" in get_args(StopReason)


def test_execution_budget_stop_condition_reason_is_execution_budget():
    """The condition advertises the exact literal used in reason_mapping."""
    assert ExecutionBudgetStopCondition.reason == "execution_budget"


def test_execution_budget_condition_maps_to_public_stop_reason():
    """An exhausted ExecutionBudget stop condition maps to 'execution_budget'."""
    orchestrator = _orchestrator(max_trials=10)

    budget = ExecutionBudget(max_cost_usd=0.10)
    budget.debit_trial(cost=0.10)  # fully spend -> exhausted_dimension == "cost"
    assert budget.exhausted_dimension == "cost"

    orchestrator.execution_budget = budget
    orchestrator._stop_condition_manager.register_execution_budget_condition(budget)

    assert orchestrator._should_stop(trial_count=1)
    assert orchestrator._stop_reason == "execution_budget"


def test_execution_budget_masks_cost_limit_when_both_fire(
    monkeypatch: pytest.MonkeyPatch,
):
    """Front-insertion: when the clamped cost enforcer AND the cumulative budget
    both fire, the reported reason is the cumulative 'execution_budget', never the
    per-run 'cost_limit' it masks.
    """
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")
    orchestrator = _orchestrator(cost_limit=0.5, cost_approved=True)

    # Drive the (front-registered) CostLimitStopCondition to fire.
    permit = orchestrator.cost_enforcer.acquire_permit()
    assert permit.is_granted
    orchestrator.cost_enforcer.track_cost(0.5, permit=permit)
    assert orchestrator.cost_enforcer.is_limit_reached

    # Attach a cumulative budget that is ALSO exhausted and front-register it.
    budget = ExecutionBudget(max_cost_usd=0.5)
    budget.debit_trial(cost=0.5)
    assert budget.exhausted_dimension == "cost"
    orchestrator.execution_budget = budget
    orchestrator._stop_condition_manager.register_execution_budget_condition(budget)

    assert orchestrator._should_stop(trial_count=1)
    # The cumulative reason wins over the masked per-run cost_limit.
    assert orchestrator._stop_reason == "execution_budget"


@pytest.mark.asyncio
async def test_vendor_pause_break_maps_to_vendor_error_stop_reason():
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._trial_lifecycle = VendorPauseLifecycle()
    orchestrator._prompt_adapter = None

    trial_count, action = await orchestrator._dispatch_trial(
        func=lambda _: "ok",
        dataset=Dataset([], name="empty"),
        session_id=None,
        function_identifier=None,
        trial_count=0,
        remaining=1.0,
        remaining_samples=None,
    )

    assert trial_count == 0
    assert action == "break"
    assert orchestrator._stop_reason == "vendor_error"


@pytest.mark.asyncio
async def test_vendor_pause_resume_does_not_set_stop_reason():
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._trial_lifecycle = VendorPauseLifecycle()
    orchestrator._prompt_adapter = StubPromptAdapter(vendor_decision="resume")

    trial_count, action = await orchestrator._dispatch_trial(
        func=lambda _: "ok",
        dataset=Dataset([], name="empty"),
        session_id=None,
        function_identifier=None,
        trial_count=0,
        remaining=1.0,
        remaining_samples=None,
    )

    assert trial_count == 0
    assert action == "continue"
    assert orchestrator._stop_reason is None
    assert orchestrator._prompt_adapter.vendor_calls == 1


def _vendor_failed_trial(trial_id: str, message: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={},
        metrics={},
        status=TrialStatus.FAILED,
        duration=0.0,
        timestamp=datetime.now(UTC),
        error_message=message,
    )


def _permitted(result: TrialResult) -> PermittedTrialResult:
    return PermittedTrialResult(result=result, permit=None)


@pytest.mark.asyncio
async def test_batch_vendor_errors_all_failed_stops_with_vendor_error():
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._prompt_adapter = StubPromptAdapter(vendor_decision="stop")

    results = [
        _permitted(_vendor_failed_trial("t1", "429 Too Many Requests")),
        _permitted(_vendor_failed_trial("t2", "rate limit exceeded")),
    ]

    assert await orchestrator._check_batch_vendor_errors(results) is True
    assert orchestrator._prompt_adapter.vendor_calls == 1


@pytest.mark.asyncio
async def test_batch_vendor_errors_resume_decision_does_not_stop():
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._prompt_adapter = StubPromptAdapter(vendor_decision="resume")

    results = [
        _permitted(_vendor_failed_trial("t1", "429 Too Many Requests")),
        _permitted(_vendor_failed_trial("t2", "quota exhausted for model")),
    ]

    assert await orchestrator._check_batch_vendor_errors(results) is False
    assert orchestrator._prompt_adapter.vendor_calls == 1


@pytest.mark.asyncio
async def test_batch_vendor_errors_partial_failures_does_not_stop():
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._prompt_adapter = StubPromptAdapter(vendor_decision="stop")

    results = [
        _permitted(_vendor_failed_trial("t1", "429 Too Many Requests")),
        _permitted(_vendor_failed_trial("t2", "division by zero")),
    ]

    assert await orchestrator._check_batch_vendor_errors(results) is False
    assert orchestrator._prompt_adapter.vendor_calls == 0


@pytest.mark.asyncio
async def test_batch_vendor_errors_empty_results_does_not_stop():
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._prompt_adapter = StubPromptAdapter(vendor_decision="stop")

    assert await orchestrator._check_batch_vendor_errors([]) is False
    assert orchestrator._prompt_adapter.vendor_calls == 0


@pytest.mark.asyncio
async def test_batch_vendor_errors_no_adapter_stops_like_sequential():
    """Non-interactive runs (no adapter) should stop on a fully vendor-failed
    batch, matching sequential mode where _handle_vendor_pause returns "break"
    when there's no adapter."""
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._prompt_adapter = None

    results = [
        _permitted(_vendor_failed_trial("t1", "429 Too Many Requests")),
        _permitted(_vendor_failed_trial("t2", "quota exhausted")),
    ]

    assert await orchestrator._check_batch_vendor_errors(results) is True


@pytest.mark.asyncio
async def test_batch_vendor_errors_detects_raised_vendor_pause_error():
    """ParallelExecutionManager carries raised exceptions in
    PermittedTrialResult.result. A fully-raised batch must still trigger the
    vendor-error stop path."""
    orchestrator = _orchestrator(max_trials=10)
    orchestrator._prompt_adapter = StubPromptAdapter(vendor_decision="stop")

    results = [
        PermittedTrialResult(
            result=VendorPauseError(
                "rate limit",
                category=VendorErrorCategory.RATE_LIMIT,
            ),
            permit=None,
        ),
        PermittedTrialResult(
            result=VendorPauseError(
                "rate limit",
                category=VendorErrorCategory.RATE_LIMIT,
            ),
            permit=None,
        ),
    ]

    assert await orchestrator._check_batch_vendor_errors(results) is True
    assert orchestrator._prompt_adapter.vendor_calls == 1


@pytest.mark.asyncio
async def test_batch_vendor_errors_detects_unwrapped_vendor_exception():
    """Any vendor-classifiable BaseException in .result should count as a
    vendor failure, even if it was never wrapped into VendorPauseError."""
    from traigent.utils.exceptions import RateLimitError

    orchestrator = _orchestrator(max_trials=10)
    orchestrator._prompt_adapter = StubPromptAdapter(vendor_decision="resume")

    results = [
        PermittedTrialResult(result=RateLimitError("429"), permit=None),
        PermittedTrialResult(result=RateLimitError("429"), permit=None),
    ]

    assert await orchestrator._check_batch_vendor_errors(results) is False
    assert orchestrator._prompt_adapter.vendor_calls == 1


@pytest.mark.asyncio
async def test_batch_vendor_errors_falls_back_when_vendor_pause_category_missing():
    """VendorPauseError.category is optional. If production code ever drops it,
    we should still classify via original_error or the message instead of
    silently treating the trial as non-vendor."""
    from traigent.utils.exceptions import RateLimitError

    orchestrator = _orchestrator(max_trials=10)
    orchestrator._prompt_adapter = StubPromptAdapter(vendor_decision="stop")

    # category=None, but original_error carries the signal
    exc_with_original = VendorPauseError(
        "unspecified",
        original_error=RateLimitError("429 rate limited"),
    )
    # category=None AND no original_error — must classify via str(payload)
    exc_message_only = VendorPauseError("rate limit exceeded")
    # category=None, original_error present but opaque — must cascade to the
    # message instead of short-circuiting to None.
    exc_opaque_original = VendorPauseError(
        "rate limit exceeded",
        original_error=RuntimeError("opaque wrapper"),
    )

    results = [
        PermittedTrialResult(result=exc_with_original, permit=None),
        PermittedTrialResult(result=exc_message_only, permit=None),
        PermittedTrialResult(result=exc_opaque_original, permit=None),
    ]

    assert await orchestrator._check_batch_vendor_errors(results) is True
    assert orchestrator._prompt_adapter.vendor_calls == 1


class RecordingPromptAdapter(StubPromptAdapter):
    def __init__(self, vendor_decision: str) -> None:
        super().__init__(vendor_decision)
        self.last_category: VendorErrorCategory | None = None

    def prompt_vendor_pause(self, error: Any, category: Any) -> str:
        self.last_category = category
        return super().prompt_vendor_pause(error, category)


@pytest.mark.asyncio
async def test_batch_vendor_errors_mixed_categories_prefers_non_recoverable():
    """Mixed batches should surface the most severe category so the prompt
    adapter can make the right call (e.g. INSUFFICIENT_FUNDS cannot be
    resolved by resuming)."""
    orchestrator = _orchestrator(max_trials=10)
    adapter = RecordingPromptAdapter(vendor_decision="stop")
    orchestrator._prompt_adapter = adapter

    results = [
        PermittedTrialResult(
            result=VendorPauseError(
                "rate limit",
                category=VendorErrorCategory.RATE_LIMIT,
            ),
            permit=None,
        ),
        PermittedTrialResult(
            result=VendorPauseError(
                "insufficient funds",
                category=VendorErrorCategory.INSUFFICIENT_FUNDS,
            ),
            permit=None,
        ),
    ]

    assert await orchestrator._check_batch_vendor_errors(results) is True
    assert adapter.last_category == VendorErrorCategory.INSUFFICIENT_FUNDS
