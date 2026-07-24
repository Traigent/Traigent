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
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

import traigent
from traigent.api.types import OptimizationStatus, TrialResult, TrialStatus
from traigent.config.types import (
    ExecutionIntent,
    ResolvedExecutionPolicy,
    TraigentConfig,
    resolve_execution_policy,
)
from traigent.core.execution_budget import ExecutionBudget
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
        [
            "timeout",
            "user_cancelled",
            "cost_limit",
            "execution_budget",
            "vendor_error",
            "network_error",
            "error",
        ],
    )
    def test_no_raise_for_owned_stop_reasons(self, stop_reason: str) -> None:
        """Empty stops already owned by another cause (timeout, user cancel,
        cost gate #1684, a cumulative ExecutionBudget stop #1980, provider/network
        errors) are not relabeled."""
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
    async def test_empty_cloud_required_with_exhausted_budget_returns_gracefully(
        self,
    ) -> None:
        """Finding #2 (#1980): a CLOUD_REQUIRED smart phase started with an already
        -exhausted shared ExecutionBudget pre-batch-blocks at 0 trials and must
        RETURN gracefully with ``stop_reason == "execution_budget"`` — never be
        relabeled into the silent-empty OptimizationError by the fail-closed guard.
        """
        # A shared budget whose examples are already fully spent by an earlier phase.
        budget = ExecutionBudget(max_examples=2)
        # Fully spend the examples dimension (as an earlier phase would have).
        budget.debit_trial(cost=0.0, examples=2)
        assert budget.exhausted_dimension == "examples"

        # max_suggestions=5 proves it is the BUDGET (not the optimizer) that blocks.
        orchestrator = _orchestrator(
            policy=_cloud_required_policy("bayesian"),
            max_suggestions=5,
            max_trials=5,
        )
        orchestrator.execution_budget = budget

        # Must NOT raise: the exhausted-budget stop is owned by #1980.
        result = await orchestrator.optimize(lambda **_: "ok", _dataset())

        assert orchestrator._stop_reason == "execution_budget"
        assert len(result.trials) == 0
        # Graceful stop, not the FAILED silent-empty error path.
        assert result.status is OptimizationStatus.COMPLETED

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


class TestDecorationTimeSmartAlgorithmRepro:
    """The issue's LITERAL end-to-end repro: the smart algorithm is requested
    at DECORATION time (``@traigent.optimize(algorithm="bayesian",
    offline=False, ...)``), then a plain ``fn.optimize()`` — no runtime
    override — reaches the managed cloud path, which comes back without a
    single executed trial. Pre-fix this finalized as a silent COMPLETED with
    0 trials and ``best_config=None``; it must raise the actionable error."""

    @staticmethod
    def _smart_wrapper(algorithm: str):
        @traigent.optimize(
            algorithm=algorithm,
            offline=False,
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space=_SPACE,
            injection_mode="parameter",
        )
        def answer(text: str, config) -> str:
            return "ok"

        return answer

    @staticmethod
    def _fake_backend_client() -> Any:
        """Minimal run-tracking backend stub (pattern from
        tests/unit/core/test_execution_policy_execution.py): the tracking
        session succeeds so the run reaches the managed optimization path."""
        from traigent.core.session_types import SessionCreationResult

        client = SimpleNamespace(
            create_session=Mock(
                return_value=SessionCreationResult.connected("sess-1681")
            ),
            get_session_mapping=Mock(
                return_value=SimpleNamespace(
                    experiment_id="exp-1681",
                    experiment_run_id="run-1681",
                )
            ),
            upload_example_features=Mock(return_value=True),
            submit_result=Mock(),
            request_trial_slot=AsyncMock(return_value="slot-unused"),
            _submit_trial_result_via_session=AsyncMock(return_value=True),
            update_trial_weighted_scores=AsyncMock(return_value=True),
            finalize_session_sync=Mock(return_value={"status": "completed"}),
            close=AsyncMock(),
        )
        client.auth_manager = SimpleNamespace(has_api_key=lambda: True)
        client.auth = client.auth_manager
        return client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("algorithm", ["bayesian"])
    async def test_plain_optimize_raises_when_managed_run_is_empty(
        self, algorithm: str, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Online + portal key: the issue's production condition (the suite's
        # offline default would short-circuit before the managed path).
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")

        answer = self._smart_wrapper(algorithm)
        # Precondition: decoration-time smart algorithm resolves to the
        # fallback-forbidden managed policy.
        assert answer.execution_policy.intent is ExecutionIntent.CLOUD_REQUIRED
        assert answer.execution_policy.algorithm == algorithm

        def _empty_managed_optimizer(
            config_space: dict[str, Any], objectives: list[str], **_kwargs: Any
        ) -> MockOptimizer:
            # The managed/cloud optimizer never suggests a trial — the empty
            # managed-path behavior this guard must continue surfacing loudly.
            optimizer = MockOptimizer(config_space, objectives)
            optimizer.set_max_suggestions(0)
            return optimizer

        with (
            patch(
                "traigent.optimizers.interactive_optimizer.InteractiveOptimizer",
                side_effect=_empty_managed_optimizer,
            ) as MockInteractive,
            patch("traigent.cloud.client.TraigentCloudClient"),
            patch(
                "traigent.core.backend_session_manager.BackendSessionManager"
                ".create_backend_client",
                return_value=self._fake_backend_client(),
            ),
        ):
            with pytest.raises(OptimizationError) as excinfo:
                await answer.optimize()

        # The managed path WAS entered (this is the cloud-required route) …
        MockInteractive.assert_called_once()
        # … and its emptiness surfaced as the actionable hard fail, never as
        # a 0-trial COMPLETED result.
        message = str(excinfo.value)
        assert f"'{algorithm}'" in message
        assert "managed" in message
        assert "without executing a single trial" in message
        assert "'grid'" in message
        assert "'random'" in message
        assert "backend" in message
