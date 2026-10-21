"""Authoritative uncertainty and partial late-batch debit regressions (#2444)."""

import asyncio
import copy
import threading
from typing import Any

import pytest

from tests.unit.evaluators.test_capture_cost_round6_2444 import _timeout_after_event
from tests.unit.evaluators.test_local_capture_ownership_2444 import (
    _Resp,
    _cost,
    _dataset,
    MODEL,
    SDK_COST_FIELDS,
)
from traigent.api.types import ExampleResult
from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import MetricsTracker
from traigent.utils.cost_calculator import UnknownModelError
from traigent.utils.langchain_interceptor import capture_langchain_response


@pytest.mark.parametrize("with_progress", [False, True])
@pytest.mark.parametrize("floor_case", ["higher", "lower", "equal"])
def test_custom_uncertainty_overrides_captured_numeric_cost(
    with_progress: bool, floor_case: str
) -> None:
    response = _Resp(1000, 500)
    captured_cost = _cost(response)
    custom_floor = {"higher": 0.25, "lower": captured_cost / 2, "equal": captured_cost}[
        floor_case
    ]
    payloads: list[Any] = []

    def custom(func: Any, config: Any, example: Any) -> ExampleResult:
        capture_langchain_response(response)
        return ExampleResult(
            example_id="example_0",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output="ok",
            metrics={"cost_unmeasured": 1.0, "cost_lower_bound": custom_floor},
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
            {"model": "gpt-4o"},  # The response's own model must win.
            _dataset(1),
            progress_callback=(lambda i, p: payloads.append(copy.deepcopy(p)))
            if with_progress
            else None,
        )
    )
    if with_progress:
        assert len(payloads) == 2
        # In the equality case, preview and final authority carry exactly the
        # same floor and flag. Exercise real preview writes via evaluate().
        assert payloads[0]["metrics"]["cost_lower_bound"] == custom_floor
        assert payloads[0]["metrics"]["cost_unmeasured"] == 1.0
    for metrics in [result.example_results[0].metrics] + [
        p["metrics"] for p in payloads[1:]
    ]:
        assert metrics["cost_unmeasured"] == 1.0
        assert metrics["cost_lower_bound"] == max(custom_floor, captured_cost)
        assert all(key not in metrics for key in SDK_COST_FIELDS)


def test_priced_then_strict_failing_late_batch_debits_every_known_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    release = threading.Event()
    captured = threading.Event()
    sibling_captured = threading.Event()
    _timeout_after_event(monkeypatch, sibling_captured)
    measured = _Resp(2000, 700)
    late_priced = _Resp(3000, 900)
    late = _Resp(9000, 3000, model=None)
    known_floor = _cost(measured) + _cost(late_priced)
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
            capture_langchain_response(late_priced)
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
