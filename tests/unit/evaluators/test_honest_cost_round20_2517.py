"""Round 20 (#2517): core findings from the full-branch review.

* a captured response without usage must keep the example unmeasured even when
  another response in the same example was measured;
* a failed row with no cost telemetry is an unknown spend, unless it reports a
  cost explicitly or reports zero provider attempts;
* ``known_spend_lower_bound`` takes the largest of ALL cost aliases;
* a strict-mode rejection still debits the shared execution budget.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from traigent.api.types import ExampleResult, TrialResult, TrialStatus
from traigent.core.cost_enforcement import (
    CostEnforcer,
    CostEnforcerConfig,
    CostTrackingRequiredError,
)
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    TokenMetrics,
)
from traigent.utils.trial_costs import (
    known_spend_lower_bound,
    trial_cost_unknown,
    unmeasured_example_ids,
)


# --------------------------------------------------------------------------- #
# 1. Mixed priced + usage-less responses in one example                        #
# --------------------------------------------------------------------------- #
def _priced() -> ExampleMetrics:
    return ExampleMetrics(
        tokens=TokenMetrics(input_tokens=10, output_tokens=5),
        cost=CostMetrics(input_cost=0.3, output_cost=0.3),
    )


def _no_usage() -> ExampleMetrics:
    return ExampleMetrics(measured=False)


def _wrapper_with_calls(*calls: ExampleMetrics) -> CustomEvaluatorWrapper:
    wrapper = object.__new__(CustomEvaluatorWrapper)
    wrapper.capture_llm_metrics = True
    wrapper._metrics_available = True
    responses = [object() for _ in calls]
    by_id = {id(r): c for r, c in zip(responses, calls, strict=True)}
    wrapper._get_all_captured_responses = lambda: responses
    wrapper._extract_llm_metrics = lambda response, **_kw: by_id[id(response)]
    wrapper._reconstruct_original_prompt = lambda example: None
    wrapper._extract_response_text = lambda output: None
    wrapper._extract_model_from_response = lambda response, config: "m"
    return wrapper


def _row() -> ExampleResult:
    return ExampleResult("e", {}, None, None, {}, 0.0, True)


@pytest.mark.parametrize("order", ["priced_first", "unmeasured_first"])
def test_priced_plus_usageless_call_keeps_the_example_unmeasured(order: str) -> None:
    calls = (_priced(), _no_usage())
    wrapper = _wrapper_with_calls(*(calls if order == "priced_first" else calls[::-1]))
    row = _row()

    captured = wrapper._capture_llm_metrics_for_example({}, None, row, 0)
    wrapper._enhance_result_with_llm_metrics(row, captured, 0.01)

    assert row.metrics["cost_unmeasured"] == 1.0
    assert "total_cost" not in row.metrics
    # the measured call is kept as a floor, never as the total
    assert row.metrics["cost_lower_bound"] == pytest.approx(0.6)
    assert known_spend_lower_bound(row.metrics) == pytest.approx(0.6)


def test_fully_measured_calls_stay_measured() -> None:
    wrapper = _wrapper_with_calls(_priced(), _priced())
    row = _row()
    captured = wrapper._capture_llm_metrics_for_example({}, None, row, 0)
    wrapper._enhance_result_with_llm_metrics(row, captured, 0.01)

    assert "cost_unmeasured" not in row.metrics
    assert row.metrics["total_cost"] == pytest.approx(1.2)


# --------------------------------------------------------------------------- #
# 2. Failed rows with no cost telemetry                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("rows", "successes", "unknown"),
    [
        # failed row, no telemetry: the provider call may have succeeded first
        ([{"total_cost": 0.25}, {}], [True, False], True),
        ([{"total_cost": 0.25}, {"accuracy": 0.0}], [True, False], True),
        # explicit cost (zero included) is a report
        ([{"total_cost": 0.25}, {"total_cost": 0.0}], [True, False], False),
        # explicit zero provider attempts is a report
        ([{"total_cost": 0.25}, {"provider_attempts": 0}], [True, False], False),
        ([{"provider_attempts": 0}], [False], False),
        ([{"provider_attempts": 2}], [False], True),
        ([{"provider_attempts": True}], [False], True),
        ([{}, {}], [False, False], True),
    ],
)
def test_failed_row_classification(rows, successes, unknown) -> None:
    assert trial_cost_unknown(rows, successes) is unknown


def test_failed_row_without_telemetry_is_named_as_unmeasured() -> None:
    results = [
        SimpleNamespace(
            example_id="ok", metrics={"total_cost": 0.2}, success=True
        ),
        SimpleNamespace(example_id="boom", metrics={}, success=False),
        SimpleNamespace(
            example_id="none", metrics={"provider_attempts": 0}, success=False
        ),
    ]
    assert unmeasured_example_ids(results) == ["boom"]


def test_wrapper_aggregation_flags_failed_row_without_cost() -> None:
    wrapper = object.__new__(CustomEvaluatorWrapper)
    results = [
        ExampleResult("a", {}, None, None, {"total_cost": 0.25}, 0.0, True),
        ExampleResult("b", {}, None, None, {}, 0.0, False),
    ]
    aggregated = wrapper._aggregate_llm_metrics([r.metrics for r in results], results)

    assert aggregated["cost_unmeasured"] == 1.0
    assert "total_cost" not in aggregated
    assert aggregated["cost_lower_bound"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# 3. Every cost alias counts toward the floor                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "source",
    [
        {"cost": 0, "total_cost": 0.4},
        {"cost_unmeasured": 1, "cost": 0, "total_cost": 0.4},
        {"cost": 0.0, "input_cost": 0.1, "output_cost": 0.3},
        {"cost": 0.0, "total_example_cost": 0.4},
        {"metrics": {"cost": 0.0, "total_cost": 0.4}},
    ],
)
def test_zero_alias_does_not_mask_a_reported_cost(source) -> None:
    assert known_spend_lower_bound(source) == pytest.approx(0.4)


def test_floor_is_zero_when_no_alias_reports_spend() -> None:
    assert known_spend_lower_bound({"cost": 0, "total_cost": 0}) == 0.0


# --------------------------------------------------------------------------- #
# 4. Strict rejection still debits the shared budget                           #
# --------------------------------------------------------------------------- #
def test_strict_rejection_debits_shared_budget_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("TRAIGENT_REQUIRE_COST_TRACKING", "TRAIGENT_RUN_COST_LIMIT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TRAIGENT_STRICT_COST_ACCOUNTING", "true")
    enforcer = CostEnforcer(CostEnforcerConfig(limit=5.0, limit_explicit=True))
    budget = ExecutionBudget(max_cost_usd=100.0)
    orch = SimpleNamespace(
        cost_enforcer=enforcer,
        execution_budget=budget,
        _extract_trial_cost=lambda t: 0.5,
        _examples_of=lambda t: 1,
    )
    trial = TrialResult(
        trial_id="t",
        config={},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metrics={"cost_unmeasured": 1.0, "cost_lower_bound": 0.5},
    )
    permit = enforcer.acquire_permit()

    with pytest.raises(CostTrackingRequiredError):
        asyncio.run(
            OptimizationOrchestrator._account_trial_spend(orch, trial, permit=permit)
        )

    snap = budget.snapshot()
    assert snap.consumed_cost == pytest.approx(0.5)
    assert snap.untracked_trials == 1
