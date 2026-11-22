"""Parallel runs that hit cost_limit report it honestly (#2943).

A trial whose cost permit is denied never starts: it spent nothing, must not
consume ``max_trials``, and must not turn the run total into "unknown". The
run stops with ``stop_reason == "cost_limit"`` and a real ``total_cost``,
exactly like the sequential path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import traigent
from traigent.api.types import TrialResult, TrialStatus
from traigent.core.mandatory_metrics import NEVER_STARTED_METADATA_KEY
from traigent.core.metric_registry import MetricRegistry
from traigent.core.metrics_aggregator import aggregate_metrics

_ROWS = [{"input": {"text": f"q{i}"}, "output": "yes"} for i in range(5)]
PER_TRIAL_COST = 0.05  # 5 examples x 0.01


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.delenv("TRAIGENT_RUN_COST_LIMIT", raising=False)


def _run(*, concurrency: int):
    def agent(text: str) -> str:
        return traigent.with_usage("yes", total_cost=0.01)

    kwargs = {}
    if concurrency > 1:
        kwargs["parallel_config"] = {"trial_concurrency": concurrency}
    fn = traigent.optimize(
        eval_dataset=_ROWS,
        objectives=["accuracy"],
        configuration_space={"k": [1, 2, 3, 4, 5, 6]},
        execution_mode="local",
        cost_limit=0.12,
        cost_approved=True,
        **kwargs,
    )(agent)
    return fn.optimize_sync(algorithm="grid", max_trials=6, progress_bar=False)


def _assert_cost_limited(result) -> None:
    assert result.stop_reason == "cost_limit"
    assert result.total_cost is not None
    assert result.total_cost == pytest.approx(2 * PER_TRIAL_COST)
    assert result.metrics.get("cost_unmeasured_trials", 0.0) == 0.0
    completed = [t for t in result.trials if t.status == TrialStatus.COMPLETED]
    assert len(completed) == 2


def test_sequential_cost_limit_reports_cost_limit_and_total() -> None:
    _assert_cost_limited(_run(concurrency=1))


def test_parallel_cost_limit_reports_cost_limit_and_total() -> None:
    result = _run(concurrency=3)
    _assert_cost_limited(result)
    # Permit-denied placeholders never ran: they are marked abandoned so they
    # do not consume max_trials.
    for trial in result.trials:
        if trial.status == TrialStatus.CANCELLED:
            assert trial.metadata.get("abandoned") is True
            assert trial.metadata.get(NEVER_STARTED_METADATA_KEY) is True


def _trial(status: TrialStatus, metrics: dict, metadata: dict) -> TrialResult:
    return TrialResult(
        trial_id="t",
        config={},
        metrics=metrics,
        status=status,
        duration=0.0,
        timestamp=datetime.now(UTC),
        metadata=metadata,
    )


def test_abandoned_never_started_trial_is_not_unmeasured_spend() -> None:
    agg = aggregate_metrics(
        [
            _trial(TrialStatus.COMPLETED, {"total_cost": 0.1}, {}),
            _trial(TrialStatus.CANCELLED, {}, {NEVER_STARTED_METADATA_KEY: True}),
        ],
        MetricRegistry(),
    )
    assert agg.total_cost == pytest.approx(0.1)
    assert agg.cost_unmeasured_trials == 0


def test_started_cancelled_trial_without_cost_stays_unmeasured() -> None:
    agg = aggregate_metrics(
        [
            _trial(TrialStatus.COMPLETED, {"total_cost": 0.1}, {}),
            _trial(TrialStatus.CANCELLED, {}, {}),
        ],
        MetricRegistry(),
    )
    assert agg.total_cost is None
    assert agg.cost_unmeasured_trials == 1


@pytest.mark.parametrize("marker", ["false", 1, "abandoned"])
def test_foreign_abandoned_metadata_does_not_hide_unknown_spend(marker) -> None:
    agg = aggregate_metrics(
        [
            _trial(TrialStatus.COMPLETED, {"total_cost": 0.1}, {}),
            _trial(
                TrialStatus.CANCELLED,
                {},
                {"abandoned": marker, NEVER_STARTED_METADATA_KEY: marker},
            ),
        ],
        MetricRegistry(),
    )
    assert agg.total_cost is None
    assert agg.cost_unmeasured_trials == 1


@pytest.mark.asyncio
async def test_denial_probe_distinguishes_transient_from_exhausted() -> None:
    from types import SimpleNamespace

    from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig
    from traigent.core.orchestrator import OptimizationOrchestrator

    probe = OptimizationOrchestrator._cost_budget_exhausted_after_denial

    roomy = CostEnforcer(config=CostEnforcerConfig(limit=10.0))
    assert await probe(SimpleNamespace(cost_enforcer=roomy)) is False
    assert roomy.get_status().reserved_cost_usd == 0.0  # probe released

    full = CostEnforcer(config=CostEnforcerConfig(limit=0.10))
    permit = await full.acquire_permit_async()
    await full.track_cost_async(0.10, permit=permit)
    assert await probe(SimpleNamespace(cost_enforcer=full)) is True
