"""End to end: an unmeasured hybrid cost is never a number and never wins (#2517).

Drives the REAL path -- a ``TraigentService`` execute handler, a JSON wire
round trip, ``HybridAPIEvaluator``, trial assembly, Pareto / best-config
selection, the optimization logger's flush payload and the budgeted session
submission -- and asserts at every stage that a cost the handler never
reported is not present as a number and never beats a measured trial.

Review history this pins (five rounds each found the same class in a new place):

* the server turned "handler reported no cost" into an explicit ``0.0`` per item
  and as the batch total, and claimed an incomplete sum as a total;
* a per-item ``metrics.cost_unmeasured = 0`` from the service overwrote the
  evaluator's own flag, so budget submission invented ``cost = 0.0``;
* a failed row without a cost dropped its flag, so a mixed trial claimed the
  priced rows' sum and could outrank a fully measured trial;
* the logger's flush payload wrote an unflagged ``cost_usd = 0.0`` for a row
  whose cost was omitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

from traigent.api.types import TrialResult
from traigent.cloud.session_budgets import (
    ensure_cost_metric_for_budgeted_completed_submission,
)
from traigent.core.orchestrator_helpers import extract_cost_from_results
from traigent.core.result_selection import select_best_configuration
from traigent.core.trial_result_factory import build_success_result
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.hybrid_api import HybridAPIEvaluator
from traigent.hybrid.protocol import HybridEvaluateResponse, HybridExecuteResponse
from traigent.utils.multi_objective import ParetoFrontCalculator
from traigent.utils.optimization_logger import OptimizationLogger
from traigent.wrapper.service import TraigentService

Mode = Literal["combined", "two_phase", "execute_only"]

_COST_KEYS = ("cost", "total_cost", "input_cost", "output_cost")


# --------------------------------------------------------------------------- #
# The real server, behind an in-process transport                              #
# --------------------------------------------------------------------------- #
def _roundtrip(payload: Any) -> Any:
    """What actually crosses the wire: strict JSON, no NaN."""
    return json.loads(json.dumps(payload, allow_nan=False))


class _InProcessTransport:
    """HybridTransport double that executes against a real ``TraigentService``."""

    def __init__(self, service: TraigentService, *, strip_quality: bool) -> None:
        self._service = service
        self._strip_quality = strip_quality
        self.execute_wires: list[dict[str, Any]] = []

    async def execute(self, request: Any) -> HybridExecuteResponse:
        wire = _roundtrip(
            await self._service.handle_execute(_roundtrip(request.to_dict()))
        )
        if self._strip_quality:
            wire.pop("quality_metrics", None)
        self.execute_wires.append(wire)
        return HybridExecuteResponse.from_dict(wire)

    async def evaluate(self, request: Any) -> HybridEvaluateResponse:
        wire = _roundtrip(
            await self._service.handle_evaluate(_roundtrip(request.to_dict()))
        )
        return HybridEvaluateResponse.from_dict(wire)


class _RawWireTransport:
    """Serves hand-written execute responses: no server in the loop.

    Isolates the client-side guarantees from the server-side ones, so each
    review blocker is pinned by a test that fails for exactly its own cause.
    """

    def __init__(self, wires: list[dict[str, Any]], *, supports_evaluate: bool) -> None:
        self._wires = [_roundtrip(w) for w in wires]
        self.execute_wires: list[dict[str, Any]] = []
        self._supports_evaluate = supports_evaluate

    async def execute(self, request: Any) -> HybridExecuteResponse:
        wire = self._wires[len(self.execute_wires)]
        self.execute_wires.append(wire)
        return HybridExecuteResponse.from_dict(wire)

    async def evaluate(self, request: Any) -> HybridEvaluateResponse:
        results = [
            {
                "example_id": e["example_id"],
                "metrics": {"accuracy": 1.0, "cost_unmeasured": 0},
            }
            for e in request.evaluations or []
        ]
        return HybridEvaluateResponse.from_dict(
            {"request_id": "r", "status": "completed", "results": results}
        )


def _dataset(n: int) -> Dataset:
    return Dataset(
        name="d",
        examples=[
            EvaluationExample(input_data={"v": i}, expected_output=i) for i in range(n)
        ],
    )


@dataclass
class Run:
    """Everything one pass through the real pipeline produced."""

    wires: list[dict[str, Any]]
    evaluation: Any
    progress: list[dict[str, Any]]
    trial: TrialResult
    attempted_total_cost: float | None
    logged_rows: list[dict[str, Any]]
    budget_metrics: dict[str, Any]
    budget_mutated: bool
    rows: list[Any] = field(default_factory=list)

    @property
    def aggregated(self) -> dict[str, Any]:
        return self.evaluation.aggregated_metrics

    @property
    def operational_metrics(self) -> dict[str, Any]:
        return self.wires[0]["operational_metrics"]

    @property
    def items(self) -> list[dict[str, Any]]:
        return [item for wire in self.wires for item in wire["outputs"]]


async def _run(
    handler: Any,
    tmp_path: Path,
    *,
    n: int = 2,
    mode: Mode = "combined",
    batch_size: int | None = None,
    trial_id: str = "t-under-test",
    config: dict[str, Any] | None = None,
    wires: list[dict[str, Any]] | None = None,
) -> Run:
    """Run one trial through the real pipeline.

    ``handler`` drives the real ``TraigentService``; ``wires`` (with
    ``handler=None``) serves hand-written responses instead.
    """
    config = config or {"p": 1}
    transport: Any
    if wires is not None:
        transport = _RawWireTransport(wires, supports_evaluate=mode == "two_phase")
    else:
        service = TraigentService()
        service.execute(handler)
        if mode == "two_phase":

            @service.evaluate
            def _score(output: Any, target: Any, kwargs: Any) -> dict[str, float]:
                # An evaluate handler is also the service: it may try to speak
                # for Traigent's own cost verdict.
                return {"accuracy": 1.0, "cost_unmeasured": 0}

        transport = _InProcessTransport(service, strip_quality=mode != "combined")
    evaluator = HybridAPIEvaluator(
        transport=transport, tunable_id="default", batch_size=batch_size or n
    )
    evaluator._get_transport = AsyncMock(return_value=transport)
    evaluator._get_capabilities = AsyncMock(
        return_value=SimpleNamespace(supports_evaluate=mode == "two_phase")
    )
    evaluator._ensure_lifecycle_manager = AsyncMock()

    progress: list[dict[str, Any]] = []
    evaluation = await evaluator.evaluate(
        lambda v: v,
        dict(config),
        _dataset(n),
        progress_callback=lambda step, info: progress.append(dict(info)),
    )

    attempted, total_cost = extract_cost_from_results(evaluation, None, trial_id)
    trial = build_success_result(
        trial_id=trial_id,
        evaluation_config=dict(config),
        eval_result=evaluation,
        duration=1.0,
        examples_attempted=attempted,
        total_cost=total_cost,
        optuna_trial_id=None,
    )

    log = OptimizationLogger(
        experiment_name="e2e",
        session_id="e2e-session",
        execution_mode="local",
        base_path=tmp_path,
        buffer_size=1,
    )
    log.log_trial_result(trial)
    log._flush_trial_buffer()
    logged_rows = _logged_example_rows(log.run_path, trial_id)

    budget_metrics = dict(trial.metrics)
    budget_mutated = ensure_cost_metric_for_budgeted_completed_submission(
        client=SimpleNamespace(_cost_budget_armed_sessions={"s1"}),
        session_id="s1",
        metrics=budget_metrics,
        status="COMPLETED",
    )
    return Run(
        wires=transport.execute_wires,
        evaluation=evaluation,
        progress=progress,
        trial=trial,
        attempted_total_cost=total_cost,
        logged_rows=logged_rows,
        budget_metrics=budget_metrics,
        budget_mutated=budget_mutated,
        rows=list(evaluation.example_results),
    )


def _logged_example_rows(run_path: Path, trial_id: str) -> list[dict[str, Any]]:
    for path in sorted(run_path.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("trial_id") == trial_id and record.get("example_results"):
                return list(record["example_results"])
    raise AssertionError(f"no logged example rows for {trial_id} under {run_path}")


# --------------------------------------------------------------------------- #
# Handlers: what a customer's agent might (not) report                         #
# --------------------------------------------------------------------------- #
def _out(example_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "output": f"answer-{example_id}",
        "metrics": {"accuracy": 1.0},
        **extra,
    }


def _omits_cost(example_id: str, data: Any, config: Any) -> dict[str, Any]:
    return _out(example_id)


def _reports_zero(example_id: str, data: Any, config: Any) -> dict[str, Any]:
    return _out(example_id, cost_usd=0.0)


def _reports_quarter(example_id: str, data: Any, config: Any) -> dict[str, Any]:
    return _out(example_id, cost_usd=0.25)


def _mixed_priced_and_omitted(
    example_id: str, data: Any, config: Any
) -> dict[str, Any]:
    if data["v"] == 0:
        return _out(example_id, cost_usd=0.25)
    return _out(example_id)


def _priced_then_errors(example_id: str, data: Any, config: Any) -> dict[str, Any]:
    if data["v"] == 0:
        return _out(example_id, cost_usd=0.25)
    raise RuntimeError("agent crashed after spending an unknown amount")


def _priced_omitted_and_errors(
    example_id: str, data: Any, config: Any
) -> dict[str, Any]:
    if data["v"] == 0:
        return _out(example_id, cost_usd=0.25)
    if data["v"] == 1:
        return _out(example_id)
    raise RuntimeError("agent crashed after spending an unknown amount")


def _omits_cost_and_claims_it_is_measured(
    example_id: str, data: Any, config: Any
) -> dict[str, Any]:
    out = _out(example_id)
    out["metrics"]["cost_unmeasured"] = 0
    return out


# --------------------------------------------------------------------------- #
# Stage assertions                                                             #
# --------------------------------------------------------------------------- #
async def _measured_trial(tmp_path: Path) -> TrialResult:
    """A fully measured $0.50 trial, produced by the same real pipeline."""
    run = await _run(
        _reports_quarter, tmp_path / "measured", trial_id="measured", config={"p": 2}
    )
    assert run.trial.metrics["total_cost"] == pytest.approx(0.50)
    return run.trial


def _assert_never_a_number(mapping: dict[str, Any], where: str) -> None:
    for key in _COST_KEYS:
        assert mapping.get(key) is None, f"{where}: {key}={mapping.get(key)!r}"


async def _assert_unmeasured_through_the_pipeline(run: Run, tmp_path: Path) -> None:
    """The whole chain says 'unknown' and never offers a number."""
    # evaluator
    assert run.aggregated["cost_unmeasured"] == 1.0
    _assert_never_a_number(run.aggregated, "aggregated_metrics")
    stats = (run.evaluation.summary_stats or {}).get("metrics", {})
    assert "cost" not in stats and "total_cost" not in stats
    for info in run.progress:
        assert "total_cost" not in info, info
    # trial assembly
    assert run.attempted_total_cost is None
    assert run.trial.metrics["cost_unmeasured"] == 1.0
    _assert_never_a_number(run.trial.metrics, "trial.metrics")
    # session budget submission
    assert run.budget_mutated is False
    assert "cost" not in run.budget_metrics
    # ranking: neither the Pareto front nor best-config selection picks it
    measured = await _measured_trial(tmp_path)
    front = ParetoFrontCalculator(
        maximize={"accuracy": True, "cost": False}
    ).calculate_pareto_front([run.trial, measured], ["accuracy", "cost"])
    assert [p.trial.trial_id for p in front] == ["measured"]
    selection = select_best_configuration(
        trials=[run.trial, measured],
        primary_objective="accuracy",
        config_space_keys={"p"},
        aggregate_configs=False,
        objective_order=["accuracy", "cost"],
        objective_orientations={"accuracy": "maximize", "cost": "minimize"},
    )
    assert selection.best_config != {"p": 1}, "unmeasured trial won selection"


def _unmeasured_rows(run: Run) -> list[Any]:
    return [r for r in run.rows if r.cost_unmeasured]


# --------------------------------------------------------------------------- #
# Cases                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["combined", "two_phase", "execute_only"])
async def test_handler_that_omits_cost_is_unmeasured_at_every_stage(
    tmp_path: Path, mode: Mode
) -> None:
    run = await _run(_omits_cost, tmp_path, mode=mode)

    # server: no per-item 0.0; the batch total is flagged, not a measurement
    assert all("cost_usd" not in item for item in run.items)
    assert run.operational_metrics["cost_unmeasured"] == 1.0
    # rows
    assert len(_unmeasured_rows(run)) == 2
    # logger flush payload
    assert [row["cost_usd"] for row in run.logged_rows] == [None, None]
    assert all(row.get("cost_unmeasured") is True for row in run.logged_rows)
    await _assert_unmeasured_through_the_pipeline(run, tmp_path)


@pytest.mark.asyncio
async def test_handler_that_reports_zero_is_a_measured_zero(tmp_path: Path) -> None:
    run = await _run(_reports_zero, tmp_path)

    assert [item["cost_usd"] for item in run.items] == [0.0, 0.0]
    assert "cost_unmeasured" not in run.operational_metrics
    assert run.operational_metrics["total_cost_usd"] == 0.0
    assert run.aggregated["cost"] == 0.0
    assert run.aggregated["total_cost"] == 0.0
    assert not run.aggregated.get("cost_unmeasured")
    assert run.trial.metrics["total_cost"] == 0.0
    assert [row["cost_usd"] for row in run.logged_rows] == [0.0, 0.0]
    assert run.budget_metrics.get("cost") == 0.0
    # A measured $0 legitimately beats a measured $0.50.
    front = ParetoFrontCalculator(
        maximize={"accuracy": True, "cost": False}
    ).calculate_pareto_front(
        [run.trial, await _measured_trial(tmp_path)], ["accuracy", "cost"]
    )
    assert [p.trial.trial_id for p in front] == ["t-under-test"]


@pytest.mark.asyncio
async def test_handler_that_reports_a_cost_is_unchanged(tmp_path: Path) -> None:
    run = await _run(_reports_quarter, tmp_path)

    assert [item["cost_usd"] for item in run.items] == [0.25, 0.25]
    assert run.operational_metrics["total_cost_usd"] == pytest.approx(0.5)
    assert "cost_unmeasured" not in run.operational_metrics
    assert run.aggregated["total_cost"] == pytest.approx(0.5)
    assert not run.aggregated.get("cost_unmeasured")
    assert run.trial.metrics["total_cost"] == pytest.approx(0.5)
    assert [row["cost_usd"] for row in run.logged_rows] == [0.25, 0.25]
    assert run.budget_mutated is False
    assert run.budget_metrics["cost"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_mixed_priced_and_omitted_batch_claims_no_total(tmp_path: Path) -> None:
    run = await _run(_mixed_priced_and_omitted, tmp_path)

    # server: the priced item keeps its cost, the omitted one has none, and the
    # incomplete sum is flagged instead of being claimed as the batch total
    assert run.items[0]["cost_usd"] == 0.25
    assert "cost_usd" not in run.items[1]
    assert run.operational_metrics["cost_unmeasured"] == 1.0
    # rows keep their own truth
    assert [r.cost_unmeasured for r in run.rows] == [False, True]
    assert [row["cost_usd"] for row in run.logged_rows] == [0.25, None]
    # the trial is unmeasured: $0.25 is not its cost and must not beat $0.50
    await _assert_unmeasured_through_the_pipeline(run, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "n"),
    [(_priced_then_errors, 2), (_priced_omitted_and_errors, 3)],
)
async def test_errored_row_without_cost_does_not_make_a_mixed_trial_cheap(
    tmp_path: Path, handler: Any, n: int
) -> None:
    run = await _run(handler, tmp_path, n=n)

    failed = [item for item in run.items if "error" in item]
    assert failed, "scenario must include an errored example"
    assert all("cost_usd" not in item for item in failed)
    assert run.operational_metrics["cost_unmeasured"] == 1.0
    # a failure alone does not establish zero spend: the flag survives
    assert any(r.cost_unmeasured and not r.success for r in run.rows)
    await _assert_unmeasured_through_the_pipeline(run, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["combined", "two_phase", "execute_only"])
async def test_service_cannot_override_the_unmeasured_flag(
    tmp_path: Path, mode: Mode
) -> None:
    run = await _run(_omits_cost_and_claims_it_is_measured, tmp_path, mode=mode)

    # the per-item ``cost_unmeasured: 0`` was ignored, not averaged in
    assert run.aggregated["cost_unmeasured"] == 1.0
    assert all("cost_unmeasured" not in r.metrics for r in run.rows)
    assert "cost_unmeasured" not in (run.evaluation.summary_stats or {}).get(
        "metrics", {}
    )
    await _assert_unmeasured_through_the_pipeline(run, tmp_path)


# --------------------------------------------------------------------------- #
# Isolated blockers: hand-written wires, so each fails for its own cause only  #
# --------------------------------------------------------------------------- #
def _wire(
    n: int,
    *,
    operational_metrics: dict[str, Any],
    item_overrides: dict[int, dict[str, Any]] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    overrides = item_overrides or {}
    outputs = []
    for i in range(n):
        item: dict[str, Any] = {
            "example_id": f"ex_{i}",
            "output": "ok",
            "metrics": {"accuracy": 1.0, "cost_unmeasured": 0},
        }
        item.update(overrides.get(i, {}))
        outputs.append(item)
    wire: dict[str, Any] = {
        "request_id": "r",
        "execution_id": "e",
        "status": status,
        "outputs": outputs,
        "operational_metrics": operational_metrics,
        "quality_metrics": {"accuracy": 1.0},
    }
    return wire


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["combined", "two_phase", "execute_only"])
async def test_isolated_external_per_item_flag_cannot_overwrite_the_computed_flag(
    tmp_path: Path, mode: Mode
) -> None:
    """Blocker 2: ``metrics.cost_unmeasured = 0`` from the service is ignored."""
    wire = _wire(2, operational_metrics={"latency_ms": 1.0})
    if mode != "combined":
        wire.pop("quality_metrics")
    run = await _run(None, tmp_path, mode=mode, wires=[wire])

    assert run.aggregated["cost_unmeasured"] == 1.0
    _assert_never_a_number(run.aggregated, "aggregated_metrics")
    assert run.trial.metrics["cost_unmeasured"] == 1.0
    # ...so the budgeted submission does not invent a zero
    assert run.budget_mutated is False
    assert "cost" not in run.budget_metrics
    assert all(row["cost_usd"] is None for row in run.logged_rows)


@pytest.mark.asyncio
async def test_isolated_failed_row_without_cost_keeps_its_flag(tmp_path: Path) -> None:
    """Blocker 3: a failed, unreported row must not leave the trial 'measured'."""
    wire = _wire(
        2,
        operational_metrics={"latency_ms": 1.0},
        item_overrides={
            0: {"cost_usd": 0.25},
            1: {"error": "agent crashed", "output": None},
        },
    )
    run = await _run(None, tmp_path, wires=[wire])

    assert run.aggregated["cost_unmeasured"] == 1.0
    _assert_never_a_number(run.aggregated, "aggregated_metrics")
    assert run.trial.metrics["cost_unmeasured"] == 1.0
    front = ParetoFrontCalculator(
        maximize={"accuracy": True, "cost": False}
    ).calculate_pareto_front(
        [run.trial, await _measured_trial(tmp_path)], ["accuracy", "cost"]
    )
    assert [p.trial.trial_id for p in front] == ["measured"]


@pytest.mark.asyncio
async def test_isolated_logger_payload_for_an_omitted_cost_row(tmp_path: Path) -> None:
    """Blocker 4: the flush payload carries the row flag, not an unflagged 0.0."""
    wire = _wire(
        2,
        operational_metrics={"latency_ms": 1.0},
        item_overrides={0: {"cost_usd": 0.25}},
    )
    run = await _run(None, tmp_path, wires=[wire])

    assert run.logged_rows[0]["cost_usd"] == 0.25
    assert "cost_unmeasured" not in run.logged_rows[0]
    assert run.logged_rows[1]["cost_usd"] is None
    assert run.logged_rows[1]["cost_unmeasured"] is True


@pytest.mark.asyncio
async def test_isolated_server_never_fabricates_a_measured_zero() -> None:
    """Blocker 1: handler output without cost is not sent as 0.0 -- per item or
    as the batch total -- and an incomplete sum is not claimed as a total."""
    service = TraigentService()
    service.execute(_mixed_priced_and_omitted)
    wire = _roundtrip(
        await service.handle_execute(
            {
                "examples": [
                    {"example_id": "a", "data": {"v": 0}},
                    {"example_id": "b", "data": {"v": 1}},
                ]
            }
        )
    )

    assert wire["outputs"][0]["cost_usd"] == 0.25
    assert "cost_usd" not in wire["outputs"][1]
    assert wire["operational_metrics"]["cost_unmeasured"] == 1.0

    only_omitted = TraigentService()
    only_omitted.execute(_omits_cost)
    wire = _roundtrip(
        await only_omitted.handle_execute(
            {"examples": [{"example_id": "a", "data": {"v": 0}}]}
        )
    )
    assert "cost_usd" not in wire["outputs"][0]
    assert wire["operational_metrics"]["cost_unmeasured"] == 1.0

    priced = TraigentService()
    priced.execute(_reports_zero)
    wire = _roundtrip(
        await priced.handle_execute(
            {"examples": [{"example_id": "a", "data": {"v": 0}}]}
        )
    )
    assert wire["outputs"][0]["cost_usd"] == 0.0
    assert "cost_unmeasured" not in wire["operational_metrics"]


@pytest.mark.asyncio
async def test_negative_per_item_cost_is_not_a_measurement(tmp_path: Path) -> None:
    """A reported negative cost is not spend: it is unmeasured, not netted."""
    wire = _wire(
        2,
        operational_metrics={"latency_ms": 1.0},
        item_overrides={0: {"cost_usd": 0.5}, 1: {"cost_usd": -0.5}},
    )
    run = await _run(None, tmp_path, wires=[wire])

    assert [r.cost_unmeasured for r in run.rows] == [False, True]
    assert run.aggregated["cost_unmeasured"] == 1.0
    _assert_never_a_number(run.aggregated, "aggregated_metrics")
    assert run.logged_rows[1]["cost_usd"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operational_metrics",
    [
        {"total_cost_usd": -1.0},
        {"cost_usd": -1.0},
        {"total_cost_usd": 0.5, "cost_unmeasured": True},
        {"total_cost_usd": 0.5, "cost_unmeasured": 1.0},
    ],
)
async def test_negative_or_flagged_response_total_is_not_a_measurement(
    tmp_path: Path, operational_metrics: dict[str, Any]
) -> None:
    """A negative response total, or one the service flags, claims no spend."""
    wire = _wire(2, operational_metrics=operational_metrics)
    run = await _run(None, tmp_path, wires=[wire])

    assert run.aggregated["cost_unmeasured"] == 1.0
    _assert_never_a_number(run.aggregated, "aggregated_metrics")


@pytest.mark.asyncio
async def test_response_total_flagged_unmeasured_keeps_priced_rows_measured(
    tmp_path: Path,
) -> None:
    """The flag withholds the *total*; a row that carries its own cost keeps it."""
    wire = _wire(
        2,
        operational_metrics={"total_cost_usd": 0.25, "cost_unmeasured": 1.0},
        item_overrides={0: {"cost_usd": 0.25}},
    )
    run = await _run(None, tmp_path, wires=[wire])

    assert [r.cost_unmeasured for r in run.rows] == [False, True]
    assert run.rows[0].cost_usd == 0.25
    assert run.aggregated["cost_unmeasured"] == 1.0
