"""Regression tests for #1681: smart algorithms must hard-fail, never return a
silent 0-trial COMPLETED result with ``best_config=None``.

Root cause (pre-fix), two cooperating gaps:

1. ``OptimizedFunction._policy_for_runtime_algorithm`` re-resolved the policy
   only for *local* runtime overrides (``grid``/``random``). A smart runtime
   override (``optimize(algorithm="bayesian")`` on an ``auto`` wrapper) kept
   the laxer construction-time ``CLOUD_BRAIN`` policy, so an unavailable
   managed path silently degraded instead of failing closed with
   ``CLOUD_REQUIRED`` semantics. (Covered in
   ``optimized_function_tests/test_grid_runtime_routing_1421.py``.)
2. ``OptimizationOrchestrator._run_optimization_with_tracing`` unconditionally
   set ``COMPLETED`` for any non-timeout stop — including a cloud-required
   managed run that never executed a single trial — producing the silent empty
   "success".

The fix adds ``_fail_closed_on_empty_smart_managed_run``: an empty (0-trial)
run under a ``CLOUD_REQUIRED`` policy raises an actionable
``OptimizationError`` instead of finalizing as COMPLETED. This file covers the
guard directly, the full ``orchestrator.optimize()`` flow, the decorator-level
``fn.optimize(algorithm=<smart>)`` surface, and the conservative non-goals
(local/cloud-brain runs, explicit ``max_trials=0``, already-owned stop
reasons such as cost_limit/#1684).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import traigent
from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
from traigent.config.types import (
    ExecutionIntent,
    ResolvedExecutionPolicy,
    TraigentConfig,
    resolve_execution_policy,
)
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.utils.exceptions import ConfigurationError, OptimizationError

from tests.shared.mocks.optimizers import MockOptimizer

SMART_ALGORITHMS = [
    "bayesian",
    "tpe",
    "cmaes",
    "nsga2",
    "optuna",
    "optuna_tpe",
    "optuna_cmaes",
    "optuna_nsga2",
]

_SPACE = {"temperature": [0.0, 1.0]}


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
            total_examples=1,
            successful_examples=1,
            duration=0.0,
            metrics=metrics,
        )


def _dataset() -> Dataset:
    return Dataset(
        [EvaluationExample({"text": "case-0"}, "ok")],
        name="smart_algo_hard_fail_1681",
    )


def _trial(trial_id: str = "t1") -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"temperature": 0.0},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )


def _cloud_required_policy(algorithm: str = "bayesian") -> ResolvedExecutionPolicy:
    policy = resolve_execution_policy(algorithm=algorithm, source_hint="test_1681")
    assert policy.intent is ExecutionIntent.CLOUD_REQUIRED
    return policy


def _orchestrator(
    *,
    policy: ResolvedExecutionPolicy | None,
    max_suggestions: int = 0,
    **kwargs: Any,
) -> OptimizationOrchestrator:
    optimizer = MockOptimizer(_SPACE, ["accuracy"])
    optimizer.set_max_suggestions(max_suggestions)
    config = TraigentConfig()
    config.execution_policy = policy
    return OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=NoopEvaluator(),
        config=config,
        **kwargs,
    )


class TestFailClosedGuardUnit:
    """Direct coverage of ``_fail_closed_on_empty_smart_managed_run``."""

    def test_empty_cloud_required_run_raises_actionable_error(self) -> None:
        orchestrator = _orchestrator(policy=_cloud_required_policy("bayesian"))
        orchestrator._stop_reason = "optimizer"

        with pytest.raises(OptimizationError) as excinfo:
            orchestrator._fail_closed_on_empty_smart_managed_run()

        message = str(excinfo.value)
        # Actionable, not tautological: names the requested algorithm, states
        # what the local SDK supports today, and how smart algorithms are
        # obtained.
        assert "'bayesian'" in message
        assert "managed" in message
        assert "without executing a single trial" in message
        assert "'grid'" in message
        assert "'random'" in message
        assert "backend" in message

    @pytest.mark.parametrize("algorithm", SMART_ALGORITHMS)
    def test_all_smart_algorithms_fail_closed(self, algorithm: str) -> None:
        orchestrator = _orchestrator(policy=_cloud_required_policy(algorithm))

        with pytest.raises(OptimizationError, match=f"'{algorithm}'"):
            orchestrator._fail_closed_on_empty_smart_managed_run()

    def test_empty_managed_completion_stop_reason_still_raises(self) -> None:
        """The live bug shape: the managed path 'completes' (e.g. the backend
        reports max_trials_reached) without a single executed trial."""
        orchestrator = _orchestrator(policy=_cloud_required_policy("tpe"))
        orchestrator._stop_reason = "max_trials_reached"

        with pytest.raises(OptimizationError, match="'tpe'"):
            orchestrator._fail_closed_on_empty_smart_managed_run()

    def test_no_raise_when_trials_executed(self) -> None:
        orchestrator = _orchestrator(policy=_cloud_required_policy("bayesian"))
        orchestrator._trials = [_trial()]
        orchestrator._fail_closed_on_empty_smart_managed_run()

    def test_no_raise_without_policy(self) -> None:
        orchestrator = _orchestrator(policy=None)
        orchestrator._fail_closed_on_empty_smart_managed_run()

    @pytest.mark.parametrize("algorithm", ["auto", "grid", "random"])
    def test_no_raise_for_non_cloud_required_policies(self, algorithm: str) -> None:
        """CLOUD_BRAIN (auto) and LOCAL_ONLY (grid/random) empty runs keep
        their existing semantics — the guard is scoped to smart algorithms."""
        policy = resolve_execution_policy(algorithm=algorithm, source_hint="test_1681")
        assert policy.intent is not ExecutionIntent.CLOUD_REQUIRED
        orchestrator = _orchestrator(policy=policy)
        orchestrator._fail_closed_on_empty_smart_managed_run()

    def test_no_raise_for_explicit_zero_max_trials(self) -> None:
        """An explicit ``max_trials=0`` no-op run is a legitimate empty stop.

        The constructor rejects 0, but the ``max_trials`` property setter
        accepts runtime updates (e.g. remote/discovery overrides), so the
        guard stays conservative for that shape.
        """
        orchestrator = _orchestrator(policy=_cloud_required_policy("bayesian"))
        orchestrator.max_trials = 0
        orchestrator._fail_closed_on_empty_smart_managed_run()

    @pytest.mark.parametrize(
        "stop_reason",
        ["timeout", "user_cancelled", "cost_limit", "vendor_error", "network_error", "error"],
    )
    def test_no_raise_for_owned_stop_reasons(self, stop_reason: str) -> None:
        """Empty stops already owned by another cause (timeout, user cancel,
        cost gate #1684, provider/network errors) are not relabeled."""
        orchestrator = _orchestrator(policy=_cloud_required_policy("bayesian"))
        orchestrator._stop_reason = stop_reason  # type: ignore[assignment]
        orchestrator._fail_closed_on_empty_smart_managed_run()


class TestFailClosedThroughOptimizeFlow:
    """The guard fires (or not) through the real ``orchestrator.optimize()``."""

    @pytest.mark.asyncio
    async def test_empty_cloud_required_optimize_raises(self) -> None:
        orchestrator = _orchestrator(
            policy=_cloud_required_policy("bayesian"), max_trials=5
        )

        with pytest.raises(OptimizationError, match="'bayesian'"):
            await orchestrator.optimize(lambda **_: "ok", _dataset())

        # The run is FAILED — never a silent COMPLETED with best_config=None.
        assert orchestrator._status is OptimizationStatus.FAILED

    @pytest.mark.asyncio
    async def test_cloud_required_with_trials_completes(self) -> None:
        """A managed run that actually executes trials still completes."""
        orchestrator = _orchestrator(
            policy=_cloud_required_policy("bayesian"),
            max_suggestions=2,
            max_trials=2,
        )

        result = await orchestrator.optimize(lambda **_: "ok", _dataset())

        assert result.status is OptimizationStatus.COMPLETED
        assert len(result.trials) == 2
        assert result.best_config is not None

    @pytest.mark.asyncio
    async def test_empty_cloud_brain_run_keeps_completed_semantics(self) -> None:
        """Negative control / conservative scope: an empty ``auto``
        (CLOUD_BRAIN) run is untouched by the guard and still completes —
        proving the fix cannot hijack non-smart empty stops."""
        policy = resolve_execution_policy(algorithm="auto", source_hint="test_1681")
        orchestrator = _orchestrator(policy=policy, max_trials=5)

        result = await orchestrator.optimize(lambda **_: "ok", _dataset())

        assert result.status is OptimizationStatus.COMPLETED
        assert len(result.trials) == 0


class TestDecoratorSmartAlgorithmSurface:
    """``fn.optimize(algorithm=<smart>)`` without a functioning managed path
    must raise an actionable error — never return an empty result."""

    @staticmethod
    def _wrapper():
        @traigent.optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space=_SPACE,
            injection_mode="parameter",
        )
        def answer(text: str, config) -> str:
            return "ok"

        return answer

    @pytest.mark.asyncio
    @pytest.mark.parametrize("algorithm", SMART_ALGORITHMS)
    async def test_runtime_smart_override_raises_without_managed_path(
        self, algorithm: str
    ) -> None:
        """Under the suite's offline default (no managed path reachable) a
        smart runtime override fails closed at optimize() time with an
        actionable ConfigurationError — the earliest layer that knows."""
        answer = self._wrapper()
        # Precondition: the stored construction-time policy is the laxer auto
        # default, i.e. the exact stale-policy setup #1681 reported.
        assert answer.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN

        with pytest.raises(ConfigurationError) as excinfo:
            await answer.optimize(algorithm=algorithm, max_trials=3)

        message = str(excinfo.value)
        assert f"'{algorithm}'" in message
        assert "managed" in message
        assert "'grid'" in message
        assert "'random'" in message

    @pytest.mark.asyncio
    async def test_runtime_grid_still_runs_locally(self) -> None:
        """Local algorithms are untouched: runtime grid still runs and
        completes exhaustively (regression fence for #1421 behavior)."""
        answer = self._wrapper()
        result = await answer.optimize(algorithm="grid")
        assert result.status is OptimizationStatus.COMPLETED
        assert len(result.trials) == len(_SPACE["temperature"])

    @pytest.mark.asyncio
    async def test_runtime_random_still_runs_locally(self) -> None:
        answer = self._wrapper()
        result = await answer.optimize(algorithm="random", max_trials=2)
        assert result.status is OptimizationStatus.COMPLETED
        assert len(result.trials) == 2
