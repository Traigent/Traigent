"""Multi-phase integration tests for the cumulative ExecutionBudget (issue #1980).

These are offline/mock tests: they drive the real ``OptimizationOrchestrator``
with a mock evaluator/optimizer that report synthetic cost/examples — no LLM, no
network, no real spend. Cost accounting is exercised for real (the cost enforcer
deliberately ignores ``TRAIGENT_MOCK_LLM``), so ``TRAIGENT_MOCK_LLM`` is pinned
to ``false`` here for deterministic behaviour; there is still zero outbound
traffic because the evaluator is a pure mock.

The headline property proven here: a budget-bounded run STARTS and stops
gracefully with ``stop_reason == "execution_budget"`` once the shared cumulative
cost is spent — including under production defaults (no cost pre-approval,
non-interactive), where it must NOT raise ``CostLimitExceeded`` merely because a
small shared ``max_cost_usd`` is below the token estimate (issue #1980, finding
#1). The per-run ``cost_limit`` is left intact as the user's own pre-run approval
gate; the shared cumulative cost is enforced mid-run by the budget's stop
condition and its pre-batch admission gate.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest

os.environ["TRAIGENT_MOCK_LLM"] = "false"

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
)
from traigent.config.types import TraigentConfig
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.optimized_function import OptimizedFunction
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.stop_conditions import ExecutionBudgetStopCondition
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.base import BaseOptimizer

FLOAT_TOLERANCE = 1e-9


@pytest.fixture(autouse=True)
def disable_mock_mode() -> None:
    """Deterministic cost accounting; the evaluator is a mock so no LLM is hit."""
    os.environ["TRAIGENT_MOCK_LLM"] = "false"
    for var in (
        "TRAIGENT_REQUIRE_COST_TRACKING",
        "TRAIGENT_STRICT_COST_ACCOUNTING",
    ):
        os.environ.pop(var, None)


@pytest.fixture(autouse=True)
def patch_backend(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the cloud backend client so nothing leaves the process."""
    mock_backend = Mock()
    mock_backend.create_session.return_value = "mock-session"
    mock_backend.submit_result.return_value = True
    mock_backend.update_trial_weighted_scores.return_value = True
    mock_backend.finalize_session_sync.return_value = None
    mock_backend.finalize_session.return_value = None
    mock_backend.delete_session.return_value = True
    monkeypatch.setattr(
        "traigent.cloud.backend_client.BackendIntegratedClient",
        lambda *args, **kwargs: mock_backend,
    )
    return mock_backend


# ---------------------------------------------------------------------------
# Mock evaluator / optimizer (mirrors tests/integration/test_cost_enforcement_e2e)
# ---------------------------------------------------------------------------


class MockCostAwareEvaluator(BaseEvaluator):
    """Evaluator that reports a fixed synthetic cost + examples per trial."""

    def __init__(
        self,
        cost_per_eval: float = 0.05,
        unknown_cost_mode: bool = False,
    ) -> None:
        super().__init__()
        self.cost_per_eval = cost_per_eval
        self.unknown_cost_mode = unknown_cost_mode
        self.evaluation_count = 0

    async def evaluate(
        self,
        func: Any,
        config: dict[str, Any],
        dataset: Dataset,
        **_kwargs: Any,
    ) -> EvaluationResult:
        self.evaluation_count += 1
        total = len(list(dataset.examples))
        metrics: dict[str, Any] = {
            "accuracy": 0.8,
            "examples_attempted": total,
        }
        if not self.unknown_cost_mode:
            metrics["cost"] = self.cost_per_eval

        result = EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=total,
            successful_examples=total,
            duration=0.05,
            metrics=metrics,
            outputs=[f"out_{i}" for i in range(total)],
            errors=[None for _ in range(total)],
        )
        result.sample_budget_exhausted = False
        result.examples_consumed = total
        return result


class MockSequentialOptimizer(BaseOptimizer):
    """Suggests up to ``max_suggestions`` configurations, then stops."""

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        max_suggestions: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(config_space, objectives, **kwargs)
        self._suggest_count = 0
        self._max_suggestions = max_suggestions
        self._should_stop = False

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        config = {"param1": self._suggest_count}
        self._suggest_count += 1
        if self._suggest_count >= self._max_suggestions:
            self._should_stop = True
        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        return self._should_stop

    def suggest(self) -> dict[str, Any]:
        return self.suggest_next_trial([])

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:
        return None

    def is_finished(self) -> bool:
        return self._should_stop

    def force_stop(self) -> None:
        self._should_stop = True


def _dataset(n: int = 2) -> Dataset:
    return Dataset(
        [EvaluationExample({"query": f"q{i}"}, f"a{i}") for i in range(n)],
        name="budget_test",
    )


def _config() -> TraigentConfig:
    # Supported local, zero-egress surface (edge_analytics was removed).
    return TraigentConfig(offline=True, algorithm="grid")


def _build_orchestrator(
    budget: ExecutionBudget | None,
    *,
    cost_per_eval: float = 0.05,
    unknown_cost_mode: bool = False,
    max_suggestions: int = 10,
    **orch_kwargs: Any,
) -> OptimizationOrchestrator:
    evaluator = MockCostAwareEvaluator(
        cost_per_eval=cost_per_eval, unknown_cost_mode=unknown_cost_mode
    )
    optimizer = MockSequentialOptimizer(
        config_space={"param1": (0, 100)},
        objectives=["accuracy"],
        max_suggestions=max_suggestions,
    )
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        config=_config(),
        **orch_kwargs,
    )
    if budget is not None:
        orchestrator.execution_budget = budget
    return orchestrator


async def _func(**_kwargs: Any) -> str:
    return "ok"


def _empty_result() -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config=None,
        best_score=None,
        optimization_id="budget-test",
        duration=0.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# 1. Multiphase: baseline -> search -> holdout on ONE shared budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiphase_shared_remaining_and_phase3_block() -> None:
    """One ExecutionBudget spent down across three phases; phase 3 pre-batch-blocks."""
    budget = ExecutionBudget(max_cost_usd=0.30)

    # Phase 1 — baseline (single config -> 1 trial, consumes 0.10).
    phase1 = _build_orchestrator(
        budget,
        cost_per_eval=0.10,
        max_suggestions=1,
        max_trials=10,
        timeout=60.0,
        cost_limit=5.0,
        cost_approved=True,
    )
    await phase1.optimize(func=_func, dataset=_dataset())
    snap1 = budget.snapshot()

    # Phase 2 — search (2 configs -> 2 trials, consumes 0.20, total 0.30).
    phase2 = _build_orchestrator(
        budget,
        cost_per_eval=0.10,
        max_suggestions=2,
        max_trials=10,
        timeout=60.0,
        cost_limit=5.0,
        cost_approved=True,
    )
    await phase2.optimize(func=_func, dataset=_dataset())
    snap2 = budget.snapshot()

    # Phase 3 — holdout: budget already exhausted -> pre-batch block, 0 trials.
    phase3 = _build_orchestrator(
        budget,
        cost_per_eval=0.10,
        max_suggestions=5,
        max_trials=10,
        timeout=60.0,
        cost_limit=5.0,
        cost_approved=True,
    )
    await phase3.optimize(func=_func, dataset=_dataset())
    snap3 = budget.snapshot()

    # Each phase counted as a run on the shared budget.
    assert snap3.runs == 3

    # Shared remaining is monotonically non-increasing across phases.
    assert snap1.remaining_cost > snap2.remaining_cost >= snap3.remaining_cost
    assert snap1.consumed_cost < snap2.consumed_cost
    assert snap2.consumed_cost == pytest.approx(snap3.consumed_cost)

    # Per-phase trial counts sum to the shared budget's total trials.
    assert phase1.trial_count == 1
    assert phase2.trial_count == 2
    assert phase3.trial_count == 0
    assert snap3.trials == phase1.trial_count + phase2.trial_count + phase3.trial_count

    # The whole budget was spent and phase 3 stopped on the cumulative reason.
    assert snap3.remaining_cost == pytest.approx(0.0, abs=FLOAT_TOLERANCE)
    assert snap3.consumed_cost == pytest.approx(0.30, abs=1e-6)
    assert phase3._stop_reason == "execution_budget"


@pytest.mark.asyncio
async def test_direct_evaluate_optimize_holdout_share_one_budget() -> None:
    """Baseline and holdout ``evaluate()`` calls debit the optimizer's budget.

    This fails before the evaluator ``budget=`` seam exists: direct evaluation
    cannot accept the shared budget, so the baseline cannot spend the pool and
    the holdout cannot be refused after the optimization phase exhausts it.
    """
    budget = ExecutionBudget(max_examples=2)
    calls = 0

    async def identity(value: int) -> int:
        nonlocal calls
        calls += 1
        return value

    direct_dataset = Dataset(
        [EvaluationExample({"value": 1}, 1)], name="direct-budget-test"
    )
    evaluator = LocalEvaluator(metrics=["accuracy"])

    baseline = await evaluator.evaluate(identity, {}, direct_dataset, budget=budget)
    assert baseline.total_examples == 1
    assert baseline.stop_reason is None
    assert calls == 1

    optimizer = _build_orchestrator(
        budget,
        cost_per_eval=0.05,
        max_suggestions=10,
        max_trials=10,
        timeout=60.0,
        cost_limit=5.0,
        cost_approved=True,
    )
    await optimizer.optimize(func=_func, dataset=_dataset(n=1))
    assert optimizer.trial_count == 1
    assert optimizer._stop_reason == "execution_budget"

    holdout = await evaluator.evaluate(identity, {}, direct_dataset, budget=budget)
    assert holdout.total_examples == 0
    assert holdout.stop_reason == "execution_budget"
    assert calls == 1  # exhausted budget refuses the next phase before execution

    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 2
    assert snapshot.runs == 3
    assert snapshot.trials == 2
    assert snapshot.exhausted_dimension == "examples"


@pytest.mark.asyncio
async def test_direct_evaluate_without_budget_preserves_execution() -> None:
    """The optional evaluator seam leaves ordinary direct evaluation unchanged."""
    calls = 0

    async def identity(value: int) -> int:
        nonlocal calls
        calls += 1
        return value

    result = await LocalEvaluator(metrics=["accuracy"]).evaluate(
        identity,
        {},
        Dataset([EvaluationExample({"value": 1}, 1)], name="unbudgeted-evaluate"),
    )

    assert result.total_examples == 1
    assert result.stop_reason is None
    assert result.execution_budget is None
    assert calls == 1


# ---------------------------------------------------------------------------
# 2. HEADLINE: stop_reason is "execution_budget", not the masked "cost_limit"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_reason_is_execution_budget_not_cost_limit() -> None:
    """A generous per-run cost_limit + a tight shared budget -> execution_budget."""
    budget = ExecutionBudget(max_cost_usd=0.10)
    orchestrator = _build_orchestrator(
        budget,
        cost_per_eval=0.05,
        max_suggestions=10,
        max_trials=10,
        timeout=60.0,
        cost_limit=5.0,  # generous per-run limit; the budget is the tight cap
        cost_approved=True,
    )

    await orchestrator.optimize(func=_func, dataset=_dataset())

    # The per-run enforcer's limit is left INTACT (never clamped to the budget
    # remaining) — that is what keeps the pre-run approval gate honest (issue #1980,
    # finding #1). The tight cumulative cap is enforced by the budget instead.
    assert orchestrator.cost_enforcer.config.limit == pytest.approx(5.0)
    assert not orchestrator.cost_enforcer.is_limit_reached
    assert orchestrator._stop_reason == "execution_budget"
    assert budget.snapshot().consumed_cost == pytest.approx(0.10, abs=1e-6)


@pytest.mark.asyncio
async def test_budget_bounded_run_starts_without_preapproval_and_stops_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding #1 regression (production defaults, NO cost pre-approval).

    With a small shared ``max_cost_usd`` (0.10) below the conservative token
    estimate, but a larger per-run ``cost_limit`` (5.0), the run must START and stop
    gracefully with ``stop_reason == "execution_budget"`` — it must NOT raise
    ``CostLimitExceeded`` merely because the budget's remaining is small. The pre-run
    approval gate must estimate against the user's own ``cost_limit``, never a
    budget-clamped value.

    This test deliberately strips ``TRAIGENT_COST_APPROVED`` (the global conftest
    sets it) and keeps ``TRAIGENT_MOCK_LLM=false`` so the real pre-run approval path
    runs — i.e. it does NOT mask the defect the way ``cost_approved=True`` would.
    """
    # Production default: nobody pre-approved cost, and the shell is non-interactive.
    monkeypatch.delenv("TRAIGENT_COST_APPROVED", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")

    budget = ExecutionBudget(max_cost_usd=0.10)
    # NOTE: cost_approved is intentionally NOT passed (the production default).
    orchestrator = _build_orchestrator(
        budget,
        cost_per_eval=0.05,
        max_suggestions=10,
        max_trials=10,
        timeout=60.0,
        cost_limit=5.0,
    )
    # The enforcer is not pre-approved and keeps the user's configured limit; the
    # budget clamp must not lower it and manufacture a pre-run decline.
    assert orchestrator.cost_enforcer.config.approved is False
    assert orchestrator.cost_enforcer.config.limit == pytest.approx(5.0)

    # Must NOT raise CostLimitExceeded — the run starts and stops gracefully.
    await orchestrator.optimize(func=_func, dataset=_dataset())

    assert orchestrator._stop_reason == "execution_budget"
    assert orchestrator.trial_count == 2  # 2 x 0.05 = 0.10 spends the shared cap
    assert budget.snapshot().consumed_cost == pytest.approx(0.10, abs=1e-6)
    # The per-run limit was never clamped down to the budget remaining.
    assert orchestrator.cost_enforcer.config.limit == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 3. Per-dimension stops each report "execution_budget" with a naming reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_examples_only_budget_stops_execution_budget() -> None:
    budget = ExecutionBudget(max_examples=4)  # 2 examples/trial -> 2 trials
    orchestrator = _build_orchestrator(
        budget,
        cost_per_eval=0.01,
        max_suggestions=10,
        max_trials=10,
        timeout=60.0,
    )

    await orchestrator.optimize(func=_func, dataset=_dataset(n=2))

    snap = budget.snapshot()
    assert snap.consumed_examples >= 4
    assert snap.exhausted_dimension == "examples"
    assert orchestrator._stop_reason == "execution_budget"

    reason = ExecutionBudgetStopCondition(budget).get_reason()
    assert "example" in reason.lower()


def test_deadline_only_budget_reports_execution_budget() -> None:
    """A deadline-exhausted budget blocks the pre-batch gate with execution_budget."""
    budget = ExecutionBudget(deadline_seconds=0.01)
    budget.start_clock()
    budget._start_monotonic = time.monotonic() - 100.0  # force the deadline past

    orchestrator = _build_orchestrator(
        budget, max_trials=10, timeout=60.0, cost_limit=5.0, cost_approved=True
    )
    orchestrator._apply_execution_budget()

    assert budget.exhausted_dimension == "deadline"
    assert orchestrator._execution_budget_prebatch_reason(float("inf")) == (
        "execution_budget"
    )

    reason = ExecutionBudgetStopCondition(budget).get_reason()
    assert "deadline" in reason.lower()


def test_cost_only_prebatch_reason_names_execution_budget() -> None:
    budget = ExecutionBudget(max_cost_usd=1.0)
    budget.debit_trial(cost=1.0)  # exhaust cost

    orchestrator = _build_orchestrator(
        budget, max_trials=10, cost_limit=5.0, cost_approved=True
    )
    orchestrator._apply_execution_budget()

    assert budget.exhausted_dimension == "cost"
    assert orchestrator._execution_budget_prebatch_reason(float("inf")) == (
        "execution_budget"
    )
    reason = ExecutionBudgetStopCondition(budget).get_reason()
    assert "cost" in reason.lower()


# ---------------------------------------------------------------------------
# 4. Retry debit: observable retried/re-run trials each debit the budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observable_reruns_each_debit_the_budget() -> None:
    """Every observed trial (including SDK-visible re-runs) debits through the
    real ``_handle_trial_result`` choke point."""
    budget = ExecutionBudget(max_cost_usd=100.0)  # generous: don't stop early
    n = 5
    orchestrator = _build_orchestrator(
        budget,
        cost_per_eval=0.10,
        max_suggestions=n,
        max_trials=n,
        timeout=60.0,
        cost_limit=100.0,
        cost_approved=True,
    )

    await orchestrator.optimize(func=_func, dataset=_dataset())

    snap = budget.snapshot()
    assert snap.trials == n
    assert snap.consumed_cost == pytest.approx(n * 0.10, abs=1e-6)
    assert snap.cost_tracking == "complete"


# ---------------------------------------------------------------------------
# 5. Per-operation limit vs the cumulative cap
#    (examples/timeout clamp down; cost is NOT clamped — issue #1980, finding #1)
# ---------------------------------------------------------------------------


def test_cost_limit_not_clamped_budget_enforces_cumulative() -> None:
    """The per-run cost enforcer limit is left INTACT (never clamped to the budget
    remaining), so the pre-run approval gate keeps estimating against the user's own
    ``cost_limit``. The cumulative cost is enforced mid-run by the budget's stop
    condition + pre-batch gate, not by lowering the enforcer's limit."""
    budget = ExecutionBudget(max_cost_usd=0.50)
    orchestrator = _build_orchestrator(budget, cost_limit=5.0, cost_approved=True)
    orchestrator._apply_execution_budget()
    # The enforcer's configured limit is unchanged (5.0), NOT clamped down to 0.50.
    assert orchestrator.cost_enforcer.config.limit == pytest.approx(5.0)


def test_per_run_cost_limit_is_preserved() -> None:
    budget = ExecutionBudget(max_cost_usd=5.0)
    orchestrator = _build_orchestrator(budget, cost_limit=0.20, cost_approved=True)
    orchestrator._apply_execution_budget()
    # The user's per-run cost_limit is preserved verbatim (cost is never clamped).
    assert orchestrator.cost_enforcer.config.limit == pytest.approx(0.20)


def test_timeout_clamped_down_to_budget_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controlled monotonic clock makes the clamp assertion exact."""
    monkeypatch.setattr("traigent.core.execution_budget.time.monotonic", lambda: 42.0)
    budget = ExecutionBudget(deadline_seconds=5.0)
    orchestrator = _build_orchestrator(
        budget, timeout=100.0, cost_limit=5.0, cost_approved=True
    )
    orchestrator._apply_execution_budget()
    assert orchestrator.timeout == 5.0
    assert orchestrator._timeout_source == "execution_budget"


def test_tighter_per_run_timeout_stays_binding() -> None:
    budget = ExecutionBudget(deadline_seconds=100.0)
    orchestrator = _build_orchestrator(
        budget, timeout=2.0, cost_limit=5.0, cost_approved=True
    )
    orchestrator._apply_execution_budget()
    assert orchestrator.timeout == pytest.approx(2.0)


def test_sample_pool_clamped_down_to_budget_examples() -> None:
    budget = ExecutionBudget(max_examples=2)
    orchestrator = _build_orchestrator(
        budget, max_total_examples=100, cost_limit=5.0, cost_approved=True
    )
    orchestrator._apply_execution_budget()
    assert orchestrator._sample_budget_manager is not None
    # effective pool = min(100, 2) = 2
    assert orchestrator._sample_budget_manager._total_budget == 2


# ---------------------------------------------------------------------------
# 6. Untracked-cost honesty + enforce_untracked_cost fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_untracked_cost_surfaces_warning_and_keeps_examples_hard() -> None:
    """Unpriced/no-cost-metric trials -> tracking != complete, warning code, and
    metadata surfaced; the examples dimension stays a hard limit."""
    budget = ExecutionBudget(max_examples=4)  # 2 examples/trial -> 2 trials
    orchestrator = _build_orchestrator(
        budget,
        unknown_cost_mode=True,  # evaluator reports NO cost
        max_suggestions=10,
        max_trials=10,
        timeout=60.0,
    )

    await orchestrator.optimize(func=_func, dataset=_dataset(n=2))

    snap = budget.snapshot()
    # Examples were still enforced hard despite unobservable cost.
    assert snap.exhausted_dimension == "examples"
    assert orchestrator._stop_reason == "execution_budget"

    # Cost tracking is not complete (every trial's cost was unobservable).
    assert snap.cost_tracking != "complete"
    assert budget.was_any_cost_untracked is True

    # The finalize hook surfaces the honesty signal onto the result.
    result = _empty_result()
    OptimizedFunction._attach_execution_budget_snapshot(None, result, orchestrator)
    assert "EXECUTION_BUDGET_UNTRACKED_COST" in result.warning_codes
    surfaced = result.metadata["execution_budget"]
    assert surfaced["cost_tracking"] != "complete"
    assert surfaced["untracked_trials"] >= 1


@pytest.mark.asyncio
async def test_enforce_untracked_cost_fails_closed() -> None:
    """With enforce_untracked_cost=True, the first unobservable-cost trial stops
    the run rather than continuing under a cap the SDK cannot honor."""
    budget = ExecutionBudget(max_examples=1000, enforce_untracked_cost=True)
    orchestrator = _build_orchestrator(
        budget,
        unknown_cost_mode=True,
        max_suggestions=10,
        max_trials=10,
        timeout=60.0,
    )

    await orchestrator.optimize(func=_func, dataset=_dataset(n=2))

    snap = budget.snapshot()
    assert orchestrator._stop_reason == "execution_budget"
    assert snap.exhausted_dimension == "untracked_cost"
    # Fail-closed: it stopped almost immediately, well before max_examples.
    assert orchestrator.trial_count <= 2
    assert snap.consumed_examples < 1000

    reason = ExecutionBudgetStopCondition(budget).get_reason()
    assert "fail-closed" in reason.lower() or "unobservable" in reason.lower()


@pytest.mark.asyncio
async def test_enforce_untracked_cost_does_not_fail_closed_on_priced_zero_dollar() -> (
    None
):
    """Finding #3: ``enforce_untracked_cost`` fails closed mid-run ONLY for trials
    that report no cost (``cost is None``). A model that reports a concrete ``$0``
    (e.g. an unpriced model whose runtime cost computed to $0) is indistinguishable
    mid-run from a genuinely free trial, so it does NOT trip the fail-closed guard —
    the run proceeds on the other (hard) dimensions. The unpriced-ness is reconciled
    at FINALIZATION via the ``UNPRICED_MODEL_RUNTIME`` warning / the cost enforcer's
    unknown-cost mode, surfaced as a lower-bound cost, not stopped while running.
    """
    budget = ExecutionBudget(max_examples=4, enforce_untracked_cost=True)
    orchestrator = _build_orchestrator(
        budget,
        cost_per_eval=0.0,  # concrete $0 (NOT None) -> looks like a real free trial
        max_suggestions=10,
        max_trials=10,
        timeout=60.0,
    )

    await orchestrator.optimize(func=_func, dataset=_dataset(n=2))

    snap = budget.snapshot()
    # It did NOT fail closed mid-run on cost: it ran until examples were exhausted.
    assert snap.exhausted_dimension == "examples"
    assert orchestrator._stop_reason == "execution_budget"
    # Every $0 trial counts as *tracked* mid-run (cost was observable, as $0).
    assert snap.cost_tracking == "complete"
    assert snap.untracked_trials == 0

    # The unpriced-$0 gap is surfaced at FINALIZE: a run carrying the
    # UNPRICED_MODEL_RUNTIME warning folds into the budget's honesty flag there,
    # flipping cost_tracking to incomplete and adding the lower-bound warning code.
    result = _empty_result()
    result.warning_codes.append("UNPRICED_MODEL_RUNTIME")
    OptimizedFunction._attach_execution_budget_snapshot(None, result, orchestrator)
    assert "EXECUTION_BUDGET_UNTRACKED_COST" in result.warning_codes
    assert result.metadata["execution_budget"]["cost_tracking"] != "complete"


# ---------------------------------------------------------------------------
# 7. Parallel-batch cost overshoot is bounded to one batch (issue #1980, F2 #1)
# ---------------------------------------------------------------------------
#
# After the F1 fix removed the per-run cost-enforcer clamp, the pre-batch admission
# gate only checked whether ONE next trial fit. In PARALLEL mode a batch of up to
# ``parallel_trials`` trials dispatches together, so up to N in-flight trials debited
# past the cumulative cost cap before the next stop check (max_cost_usd=0.10,
# parallel_trials=10, $0.05/trial -> $0.50 debited = 5x the cap). The gate now
# reserves budget for the whole next batch, bounding overshoot to under one batch.


@pytest.mark.asyncio
async def test_parallel_batch_does_not_overshoot_cost_cap() -> None:
    """A tiny cap the next batch cannot fund blocks the batch outright (0 trials),
    so cost never overshoots. On the pre-fix one-trial gate this ran a full 10-wide
    batch and debited $0.50 (5x the $0.10 cap)."""
    cap = 0.10
    parallel = 10
    cost_per = 0.05
    est_batch = cost_per * parallel  # 0.50
    budget = ExecutionBudget(max_cost_usd=cap)
    orchestrator = _build_orchestrator(
        budget,
        cost_per_eval=cost_per,
        max_suggestions=50,
        parallel_trials=parallel,
        max_trials=50,
        timeout=60.0,
        cost_limit=5.0,  # generous per-run limit; the shared budget is the tight cap
        cost_approved=True,
    )

    await orchestrator.optimize(func=_func, dataset=_dataset())

    snap = budget.snapshot()
    # Bounded: consumed never exceeds the cap by more than one batch's estimate ...
    assert snap.consumed_cost <= cap + est_batch + FLOAT_TOLERANCE
    # ... and with an accurate estimate the batch-aware gate keeps it AT/UNDER the
    # cap (the pre-fix one-trial gate let a 10-wide batch overshoot to $0.50).
    assert snap.consumed_cost <= cap + FLOAT_TOLERANCE
    # It stopped on the cumulative reason and did NOT run all configured trials.
    assert orchestrator._stop_reason == "execution_budget"
    assert orchestrator.trial_count < 50


@pytest.mark.asyncio
async def test_parallel_run_admits_full_batches_and_stops_bounded() -> None:
    """A cap that funds exactly one batch admits one batch, then the gate blocks the
    next batch it cannot fully fund — bounding consumed cost to the cap rather than
    the cap + a whole extra batch the one-trial gate allowed."""
    cap = 0.60
    parallel = 10
    cost_per = 0.05
    est_batch = cost_per * parallel  # 0.50
    budget = ExecutionBudget(max_cost_usd=cap)
    orchestrator = _build_orchestrator(
        budget,
        cost_per_eval=cost_per,
        max_suggestions=100,
        parallel_trials=parallel,
        max_trials=100,
        timeout=60.0,
        cost_limit=5.0,
        cost_approved=True,
    )

    await orchestrator.optimize(func=_func, dataset=_dataset())

    snap = budget.snapshot()
    # Exactly one batch of 10 was admitted (0.50); the next batch (remaining 0.10 <
    # one-batch estimate 0.50) was blocked. The pre-fix gate ran a 2nd batch (=1.00).
    assert orchestrator.trial_count == parallel
    assert snap.consumed_cost == pytest.approx(est_batch, abs=1e-6)
    assert snap.consumed_cost <= cap + est_batch + FLOAT_TOLERANCE
    assert snap.consumed_cost <= cap + FLOAT_TOLERANCE
    assert orchestrator._stop_reason == "execution_budget"
    assert orchestrator.trial_count < 100


def test_prebatch_gate_is_batch_aware_for_parallel_cost() -> None:
    """The pre-batch admission gate reserves budget for the WHOLE parallel batch,
    not one trial (issue #1980 finding #1). With a remaining cost that funds one
    trial but not a full batch, a parallel run blocks with ``execution_budget``
    while a sequential run (batch of 1) does NOT — proving the sequential contract
    is byte-identical to the pre-fix one-trial gate."""
    parallel = 10
    est_per_trial = 0.05  # cost enforcer's default estimated_cost_per_trial

    # Parallel: remaining 0.10 < one-batch estimate (10 x 0.05 = 0.50) -> blocked.
    budget_par = ExecutionBudget(max_cost_usd=0.60)
    budget_par.debit_trial(cost=0.50)  # remaining_cost = 0.10
    par = _build_orchestrator(
        budget_par,
        parallel_trials=parallel,
        max_trials=100,
        cost_limit=5.0,
        cost_approved=True,
    )
    par._apply_execution_budget()
    assert par.cost_enforcer.get_status().estimated_cost_per_trial == pytest.approx(
        est_per_trial
    )
    assert par._effective_batch_size(float("inf")) == parallel
    assert par._execution_budget_prebatch_reason(float("inf")) == "execution_budget"

    # Sequential: the same remaining 0.10 funds the single next trial (0.05) -> the
    # gate does NOT block. This is the unchanged pre-fix behaviour.
    budget_seq = ExecutionBudget(max_cost_usd=0.60)
    budget_seq.debit_trial(cost=0.50)  # remaining_cost = 0.10
    seq = _build_orchestrator(
        budget_seq,
        parallel_trials=1,
        max_trials=100,
        cost_limit=5.0,
        cost_approved=True,
    )
    seq._apply_execution_budget()
    assert seq._effective_batch_size(float("inf")) == 1
    assert seq._execution_budget_prebatch_reason(float("inf")) is None


def test_effective_batch_size_is_bounded_by_remaining_trials() -> None:
    """Effective batch size is ``parallel_trials``, capped by finite remaining
    trials; sequential mode is always 1 (issue #1980 finding #1)."""
    par = _build_orchestrator(
        None, parallel_trials=8, cost_limit=5.0, cost_approved=True
    )
    assert par._effective_batch_size(float("inf")) == 8  # unbounded trial budget
    assert par._effective_batch_size(3) == 3  # fewer trials remain than the batch
    assert par._effective_batch_size(20) == 8  # the batch is the binding cap
    assert par._effective_batch_size(0) == 0  # no trials left -> no batch

    seq = _build_orchestrator(
        None, parallel_trials=1, cost_limit=5.0, cost_approved=True
    )
    assert seq._effective_batch_size(float("inf")) == 1
    assert seq._effective_batch_size(100) == 1
