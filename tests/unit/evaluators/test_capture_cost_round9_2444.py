"""Preview provenance and strict late-spend cleanup regressions (#2444)."""

import asyncio
import threading
from typing import Any

import pytest

from tests.unit.evaluators.test_capture_cost_round6_2444 import _timeout_after_event
from tests.unit.evaluators.test_local_capture_ownership_2444 import (
    MODEL,
    _Resp,
    _cost,
    _dataset,
)
from traigent.api.types import ExampleResult
from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
)
from traigent.utils.cost_calculator import UnknownModelError
from traigent.utils.langchain_interceptor import capture_langchain_response


@pytest.mark.parametrize("with_progress", [False, True])
def test_custom_evaluator_final_known_floor_survives(with_progress: bool) -> None:
    def custom(func: Any, config: Any, example: Any) -> ExampleResult:
        return ExampleResult(
            example_id="example_0",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output="ok",
            metrics={"cost_unmeasured": 1.0, "cost_lower_bound": 0.25},
            execution_time=0.0,
            success=True,
        )

    def agent(text: str, **config: Any) -> str:
        raise AssertionError("the custom evaluator owns execution")

    result = asyncio.run(
        LocalEvaluator(
            metrics=["accuracy"], detailed=True, custom_eval_func=custom
        ).evaluate(
            agent,
            {},
            _dataset(1),
            progress_callback=(lambda index, payload: None) if with_progress else None,
        )
    )
    row = result.example_results[0].metrics
    assert row["cost_unmeasured"] == 1.0
    assert row["cost_lower_bound"] == 0.25
    assert all(
        key not in row for key in ("cost", "input_cost", "output_cost", "total_cost")
    )


@pytest.mark.parametrize("authoritative_when", ["before-preview", "after-preview"])
def test_preview_cleanup_preserves_authoritative_floor(
    authoritative_when: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = ExampleResult(
        example_id="example_0",
        input_data={},
        expected_output=None,
        actual_output="ok",
        metrics={},
        execution_time=0.0,
        success=True,
    )
    authority = {"cost_unmeasured": 1.0, "cost_lower_bound": 0.25}
    if authoritative_when == "before-preview":
        row.metrics.update(authority)
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    preview = ExampleMetrics(cost=CostMetrics(unmeasured=True, known_cost=0.5))
    monkeypatch.setattr(evaluator, "_extract_progress_call_metrics", lambda *a: preview)
    evaluator._extract_response_metrics("example_0", None, row, {})
    assert row.metrics["cost_lower_bound"] == 0.5
    if authoritative_when == "after-preview":
        row.metrics.update(authority)
    final = ExampleMetrics(cost=CostMetrics(unmeasured=True, known_cost=0.1))
    evaluator._update_example_metric_from_result(final, row, 0)
    evaluator._enforce_row_cost_invariant(final, row)
    for metrics in (row.metrics, final.custom_metrics):
        assert metrics["cost_unmeasured"] == 1.0
        assert metrics["cost_lower_bound"] == 0.25
        assert all(
            key not in metrics
            for key in ("cost", "input_cost", "output_cost", "total_cost")
        )


def test_strict_late_pricing_error_still_debits_once_and_finalizes_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    release = threading.Event()
    captured = threading.Event()
    sibling_captured = threading.Event()
    _timeout_after_event(monkeypatch, sibling_captured)
    measured = _Resp(2000, 700)
    late = _Resp(9000, 3000, model=None)
    known_floor = _cost(measured)
    budget = ExecutionBudget(max_cost_usd=10.0, max_examples=10)
    original_summary = MetricsTracker.format_as_summary_stats
    original_create_lease = budget._create_example_lease
    original_debit = budget.debit_trial
    leases: list[Any] = []
    closures: list[Any] = []
    debits: list[Any] = []

    def create_lease() -> Any:
        lease = original_create_lease()
        leases.append(lease)
        original_finalize = lease.finalize

        def finalize() -> Any:
            closure = original_finalize()
            closures.append(closure)
            return closure

        monkeypatch.setattr(lease, "finalize", finalize)
        return lease

    def debit(**kwargs: Any) -> Any:
        debits.append(kwargs)
        return original_debit(**kwargs)

    def agent(text: str, **config: Any) -> str:
        if text == "0":
            assert release.wait(5)
            capture_langchain_response(late)
            captured.set()
        else:
            capture_langchain_response(measured)
            sibling_captured.set()
        return "ok"

    def summary(tracker: MetricsTracker, **kwargs: Any) -> Any:
        release.set()
        assert captured.wait(5)
        return original_summary(tracker, **kwargs)

    monkeypatch.setattr(budget, "_create_example_lease", create_lease)
    monkeypatch.setattr(budget, "debit_trial", debit)
    monkeypatch.setattr(MetricsTracker, "format_as_summary_stats", summary)
    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        max_workers=2,
        timeout=10.0,
        execution_mode="local",
    )
    try:
        with pytest.raises(UnknownModelError, match="no model name"):
            asyncio.run(
                evaluator.evaluate(agent, {"model": MODEL}, _dataset(2), budget=budget)
            )
        assert len(debits) == 1
        assert debits[0]["cost"] == pytest.approx(known_floor)
        assert debits[0]["untracked"]
        snapshot = budget.snapshot()
        assert snapshot.consumed_cost == pytest.approx(known_floor)
        assert snapshot.trials == 1
        assert snapshot.untracked_trials == 1
        assert snapshot.consumed_examples == 2
        assert len(leases) == len(closures) == 1
        assert closures[0].consumed == 2
        assert leases[0]._closed
    finally:
        release.set()
        assert captured.wait(5)
