"""Writer-inventory sweep, round 6 (#2517): absent cost is never written as $0.

Each test pins one writer found by the full inventory of places that turn a
missing cost into a number. Companion to the end-to-end hybrid module
``test_honest_cost_end_to_end_2517.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.core.cost_estimator import CostEstimator
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.trial_lifecycle import _span_cost
from traigent.evaluators.hybrid_api import HybridExampleResult
from traigent.integrations.observability import workflow_traces as wt
from traigent.utils.insights import _analyze_top_configurations
from traigent.utils.optimization_logger import _serialize_example_result
from traigent.utils.outcome_matrix import _extract_cell
from traigent.utils.trial_costs import is_cost_unmeasured


# --------------------------------------------------------------------------- #
# The shared flag reader                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (1.0, True),
        (1, True),
        (True, True),  # a row dataclass serializes the flag as a bool
        (0.0, False),
        (0, False),
        (False, False),
        (None, False),
        (float("nan"), False),
        ("1", False),
    ],
)
def test_flag_reader_accepts_the_bool_a_row_dataclass_serializes(
    flag: Any, expected: bool
) -> None:
    assert is_cost_unmeasured({"cost_unmeasured": flag}) is expected


# --------------------------------------------------------------------------- #
# Logger payload and outcome matrix                                            #
# --------------------------------------------------------------------------- #
def test_logger_payload_for_a_flagged_dataclass_row_has_no_cost_number() -> None:
    row = HybridExampleResult(example_id="e", cost_usd=0.0, cost_unmeasured=True)

    payload = _serialize_example_result(row)

    assert payload["cost_usd"] is None
    assert payload["cost_unmeasured"] is True


def test_logger_payload_for_a_flagged_dict_row_has_no_cost_number() -> None:
    payload = _serialize_example_result(
        {"example_id": "e", "cost_usd": 0.0, "cost_unmeasured": True, "metrics": {}}
    )

    assert payload["cost_usd"] is None
    assert payload["cost_unmeasured"] is True


def test_logger_payload_keeps_a_measured_zero_and_a_reported_cost() -> None:
    zero = _serialize_example_result(
        HybridExampleResult(example_id="e", cost_usd=0.0, cost_unmeasured=False)
    )
    priced = _serialize_example_result(
        {"example_id": "e", "cost_usd": 0.25, "metrics": {}}
    )

    assert zero["cost_usd"] == 0.0
    assert "cost_unmeasured" not in zero
    assert priced["cost_usd"] == 0.25


def test_logger_payload_for_a_row_with_no_cost_at_all_is_null_not_zero() -> None:
    """A local-lane row has no ``cost_usd``; it must not be written as 0.0."""
    payload = _serialize_example_result({"example_id": "e", "metrics": {}})

    assert payload["cost_usd"] is None


def test_logger_payload_reads_the_local_lane_total_cost_metric() -> None:
    payload = _serialize_example_result(
        {"example_id": "e", "metrics": {"total_cost": 0.125}}
    )

    assert payload["cost_usd"] == 0.125


def test_logger_payload_honours_the_flag_in_row_metrics() -> None:
    payload = _serialize_example_result(
        {"example_id": "e", "metrics": {"cost_unmeasured": 1.0, "total_cost": 0.0}}
    )

    assert payload["cost_usd"] is None


def test_outcome_matrix_cell_for_a_flagged_row_has_null_cost() -> None:
    cell = _extract_cell(
        {"example_id": "e", "cost_usd": 0.0, "cost_unmeasured": True, "metrics": {}}
    )

    assert cell["cost_usd"] is None


def test_outcome_matrix_cell_keeps_a_measured_zero() -> None:
    cell = _extract_cell(
        {"example_id": "e", "cost_usd": 0.0, "cost_unmeasured": False, "metrics": {}}
    )

    assert cell["cost_usd"] == 0.0


# --------------------------------------------------------------------------- #
# Workflow span cost                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("privacy", [False, True])
def test_span_cost_is_omitted_for_a_trial_that_recorded_none(privacy: bool) -> None:
    """The deferred #2446 item 4: a failed/pruned trial has no ``total_cost``."""
    assert _span_cost({}, privacy) is None
    assert _span_cost(None, privacy) is None
    assert _span_cost({"total_cost": None}, privacy) is None
    assert _span_cost({"total_cost": float("nan")}, privacy) is None


@pytest.mark.parametrize("privacy", [False, True])
def test_span_cost_keeps_a_recorded_cost_including_a_measured_zero(
    privacy: bool,
) -> None:
    assert _span_cost({"total_cost": 0.0}, privacy) == 0.0
    assert _span_cost({"total_cost": 0.5}, privacy) == 0.5


def test_otel_span_without_a_cost_attribute_reports_no_cost() -> None:
    assert wt._optional_span_cost(None) is None
    assert wt._optional_span_cost("n/a") is None
    assert wt._optional_span_cost(True) is None
    assert wt._optional_span_cost(-1.0) is None
    assert wt._optional_span_cost(float("inf")) is None
    assert wt._optional_span_cost(0.0) == 0.0
    assert wt._optional_span_cost("0.5") == 0.5


def test_manual_span_without_a_cost_is_flagged_unmeasured_in_the_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wt, "requests", MagicMock())
    monkeypatch.setattr(wt, "REQUESTS_AVAILABLE", True)
    tracker = wt.WorkflowTracesTracker(
        backend_url="http://localhost:5000", auth_token="t", auto_send=False
    )

    with tracker.trace_trial("config_1"):
        tracker.add_span(
            span_id="s1",
            span_name="n",
            span_type="node",
            start_time=datetime.now(UTC),
        )
        tracker.add_span(
            span_id="s2",
            span_name="n",
            span_type="node",
            start_time=datetime.now(UTC),
            cost_usd=0.0,
        )
        payloads = [span.to_dict() for span in tracker._spans]

    unreported, reported_zero = payloads
    assert unreported["cost_usd"] == 0.0
    assert unreported["metadata"]["traigent.cost_unmeasured"] is True
    assert reported_zero["cost_usd"] == 0.0


# --------------------------------------------------------------------------- #
# Run-level summaries                                                          #
# --------------------------------------------------------------------------- #
def _trial(trial_id: str, metrics: dict[str, Any], **config: Any) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config or {"p": trial_id},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def _result(trials: list[TrialResult]) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config={"p": "measured"},
        best_score=1.0,
        optimization_id="opt",
        duration=2.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
    )


def test_summary_and_experiment_stats_withhold_a_total_for_unmeasured_trials() -> None:
    result = _result(
        [
            _trial("measured", {"accuracy": 1.0, "total_cost": 0.5}),
            _trial("unknown", {"accuracy": 1.0, "cost_unmeasured": 1.0}),
        ]
    )

    summary = result.get_summary()
    stats = result.experiment_stats

    assert summary["total_cost"] is None
    assert summary["cost_unmeasured_trials"] == 1
    assert summary["cost_lower_bound"] == pytest.approx(0.5)
    assert stats.total_cost is None
    assert stats.cost_per_configuration is None
    assert stats.cost_unmeasured_trials == 1
    assert stats.cost_lower_bound == pytest.approx(0.5)


def test_all_unmeasured_run_has_no_zero_total_or_per_config_cost() -> None:
    result = _result(
        [
            _trial("a", {"accuracy": 1.0, "cost_unmeasured": 1.0}),
            _trial("b", {"accuracy": 1.0, "cost_unmeasured": 1.0}),
        ]
    )

    summary = result.get_summary()
    stats = result.experiment_stats

    assert summary["total_cost"] is None
    assert summary["cost_lower_bound"] is None
    assert stats.total_cost is None
    assert stats.cost_per_configuration is None


def test_fully_measured_run_keeps_its_totals() -> None:
    result = _result(
        [
            _trial("a", {"accuracy": 1.0, "total_cost": 0.25}),
            _trial("b", {"accuracy": 1.0, "total_cost": 0.5}),
        ]
    )

    summary = result.get_summary()
    stats = result.experiment_stats

    assert summary["total_cost"] == pytest.approx(0.75)
    assert "cost_unmeasured_trials" not in summary
    assert stats.total_cost == pytest.approx(0.75)
    assert stats.cost_per_configuration == pytest.approx(0.375)
    assert stats.cost_unmeasured_trials == 0


# --------------------------------------------------------------------------- #
# Insights                                                                     #
# --------------------------------------------------------------------------- #
def test_insights_report_no_cost_analysis_for_an_unmeasured_trial() -> None:
    flagged = _trial("a", {"accuracy": 1.0, "cost_unmeasured": 1.0, "total_cost": 0.0})
    measured = _trial("b", {"accuracy": 0.9, "cost": 0.0, "total_cost": 0.0})

    rows = _analyze_top_configurations([flagged, measured], ["accuracy"], 5)

    by_id = {row["trial_id"]: row for row in rows}
    assert "cost_analysis" not in by_id["a"]
    assert by_id["b"]["cost_analysis"]["cost_per_query"] == 0.0


# --------------------------------------------------------------------------- #
# Optimizer notification                                                       #
# --------------------------------------------------------------------------- #
def _orchestrator(objectives: list[str]) -> tuple[Any, MagicMock]:
    orch = object.__new__(OptimizationOrchestrator)
    optimizer = SimpleNamespace(
        objectives=objectives,
        report_trial_result=MagicMock(),
        report_trial_failure=MagicMock(),
    )
    orch.optimizer = optimizer  # type: ignore[attr-defined]
    return orch, optimizer  # type: ignore[return-value]


def test_optimizer_is_not_told_zero_for_an_unmeasured_cost_objective() -> None:
    orch, optimizer = _orchestrator(["accuracy", "cost"])
    trial = _trial("a", {"accuracy": 1.0, "cost_unmeasured": 1.0})

    handled = orch._notify_completed_trial(trial, 7)

    assert handled is True
    optimizer.report_trial_result.assert_not_called()
    optimizer.report_trial_failure.assert_called_once()
    assert optimizer.report_trial_failure.call_args.args[0] == 7


def test_optimizer_is_not_told_zero_when_the_cost_objective_is_absent() -> None:
    orch, optimizer = _orchestrator(["accuracy", "cost"])
    trial = _trial("a", {"accuracy": 1.0})

    orch._notify_completed_trial(trial, 3)

    optimizer.report_trial_result.assert_not_called()
    optimizer.report_trial_failure.assert_called_once()


def test_optimizer_still_gets_a_measured_cost_and_a_measured_zero() -> None:
    orch, optimizer = _orchestrator(["accuracy", "cost"])

    orch._notify_completed_trial(_trial("a", {"accuracy": 1.0, "cost": 0.25}), 1)
    orch._notify_completed_trial(
        _trial("b", {"accuracy": 1.0, "cost": 0.0, "total_cost": 0.0}), 2
    )

    assert optimizer.report_trial_result.call_args_list[0].args == (1, [1.0, 0.25])
    assert optimizer.report_trial_result.call_args_list[1].args == (2, [1.0, 0.0])
    optimizer.report_trial_failure.assert_not_called()


def test_optimizer_without_a_cost_objective_is_unaffected_by_the_flag() -> None:
    orch, optimizer = _orchestrator(["accuracy"])

    orch._notify_completed_trial(
        _trial("a", {"accuracy": 0.8, "cost_unmeasured": 1.0}), 1
    )

    optimizer.report_trial_result.assert_called_once_with(1, 0.8)


# --------------------------------------------------------------------------- #
# Cost enforcement input                                                       #
# --------------------------------------------------------------------------- #
def test_enforcement_sees_unknown_not_a_metadata_number_for_a_flagged_trial() -> None:
    trial = _trial("a", {"accuracy": 1.0, "cost_unmeasured": 1.0})
    trial.metadata = {"total_example_cost": 0.0, "cost": 0.0}

    # Nothing reported: still unknown, a metadata zero is not a measurement.
    assert CostEstimator.extract_trial_cost(trial) is None


def test_enforcement_charges_the_known_floor_of_a_flagged_trial() -> None:
    """Budgets charge the reported spend floor; the flag stays on the trial."""
    trial = _trial(
        "a", {"accuracy": 1.0, "cost_unmeasured": 1.0, "cost_lower_bound": 1.9}
    )
    trial.metadata = {"total_example_cost": 0.0, "cost": 0.0}

    assert CostEstimator.extract_trial_cost(trial) == 1.9
    assert trial.metrics["cost_unmeasured"] == 1.0


def test_enforcement_still_reads_a_measured_cost_and_a_measured_zero() -> None:
    assert CostEstimator.extract_trial_cost(_trial("a", {"total_cost": 0.25})) == 0.25
    assert CostEstimator.extract_trial_cost(_trial("b", {"total_cost": 0.0})) == 0.0


# --------------------------------------------------------------------------- #
# Integration helpers: LangChain, PydanticAI, Langfuse                         #
# --------------------------------------------------------------------------- #
def _llm_response(tokens: dict[str, int] | None) -> MagicMock:
    response = MagicMock()
    response.llm_output = {"token_usage": tokens} if tokens is not None else {}
    return response


@pytest.fixture
def langchain_handler(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real ``TraigentHandler``, usable without langchain-core installed."""
    from traigent.integrations.langchain import handler as lc

    monkeypatch.setattr(lc, "LANGCHAIN_AVAILABLE", True)
    return lc.TraigentHandler(trace_id="t")


def test_langchain_call_without_usage_is_unmeasured_not_free(
    langchain_handler: Any,
) -> None:
    handler = langchain_handler
    handler.on_llm_start(serialized={}, prompts=["hi"], run_id="r1")
    handler.on_llm_end(response=_llm_response(None), run_id="r1")

    metrics = handler.get_metrics()
    measures = metrics.to_measures_dict(prefix="lc_")

    assert metrics.cost_unmeasured is True
    assert "lc_total_cost" not in measures
    assert measures["lc_cost_unmeasured"] == 1.0


def test_langchain_errored_call_is_unmeasured_not_free(
    langchain_handler: Any,
) -> None:
    handler = langchain_handler
    handler.on_llm_start(serialized={}, prompts=["hi"], run_id="r1")
    handler.on_llm_error(RuntimeError("boom"), run_id="r1")

    assert handler.get_metrics().cost_unmeasured is True


def test_langchain_failed_estimate_is_unmeasured_not_free(
    langchain_handler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = langchain_handler
    monkeypatch.setattr(handler, "_estimate_cost", lambda *a, **k: None)
    handler.on_llm_start(
        serialized={"kwargs": {"model": "gpt-4o"}}, prompts=["hi"], run_id="r1"
    )
    handler.on_llm_end(
        response=_llm_response(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        ),
        run_id="r1",
    )

    metrics = handler.get_metrics()
    assert metrics.cost_unmeasured is True
    assert "total_cost" not in metrics.to_measures_dict()


def test_langchain_priced_call_keeps_its_total(
    langchain_handler: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = langchain_handler
    monkeypatch.setattr(handler, "_estimate_cost", lambda *a, **k: 0.002)
    handler.on_llm_start(
        serialized={"kwargs": {"model": "gpt-4o"}}, prompts=["hi"], run_id="r1"
    )
    handler.on_llm_end(
        response=_llm_response(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        ),
        run_id="r1",
    )

    metrics = handler.get_metrics()
    assert metrics.cost_unmeasured is False
    assert metrics.to_measures_dict()["total_cost"] == pytest.approx(0.002)


def test_pydantic_ai_failed_run_and_failed_estimate_are_unmeasured() -> None:
    from traigent.integrations.pydantic_ai._types import (
        AgentRunMetrics,
        PydanticAIHandlerMetrics,
    )

    priced = AgentRunMetrics(cost=0.25)
    unknown = AgentRunMetrics(cost_unmeasured=True)

    assert PydanticAIHandlerMetrics(runs=[priced]).to_measures_dict()[
        "total_cost"
    ] == pytest.approx(0.25)
    measures = PydanticAIHandlerMetrics(runs=[priced, unknown]).to_measures_dict("p_")
    assert "p_total_cost" not in measures
    assert measures["p_cost_unmeasured"] == 1.0


def test_pydantic_ai_handler_flags_a_failed_run_and_a_failed_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.integrations.pydantic_ai import handler as pa
    from traigent.integrations.pydantic_ai.handler import PydanticAIHandler

    monkeypatch.setattr(pa, "PYDANTICAI_AVAILABLE", True)
    agent = MagicMock()
    agent.model = "openai:gpt-4o"
    handler = PydanticAIHandler(agent)

    handler._record_error(0.0)
    assert handler.get_metrics().cost_unmeasured is True

    handler2 = PydanticAIHandler(agent)
    monkeypatch.setattr(handler2, "_estimate_cost", lambda *a, **k: None)
    result = MagicMock()
    result.usage.return_value = SimpleNamespace(
        requests=1, input_tokens=10, output_tokens=5, total_tokens=15
    )
    handler2._record_usage(result, 0.0)
    assert handler2.get_metrics().cost_unmeasured is True


def test_langfuse_generation_without_a_cost_is_unmeasured() -> None:
    from traigent.integrations.langfuse.client import (
        LangfuseClient,
        LangfuseObservation,
        LangfuseTraceMetrics,
    )

    client = LangfuseClient.__new__(LangfuseClient)
    unpriced = client._dict_to_observation(
        {"id": "o1", "name": "gen", "type": "GENERATION", "model": "x"}
    )
    priced = client._dict_to_observation(
        {"id": "o2", "name": "gen", "type": "GENERATION", "calculatedTotalCost": 0.0}
    )

    assert isinstance(unpriced, LangfuseObservation)
    assert unpriced.cost_reported is False
    assert priced.cost_reported is True  # a reported 0.0 is a measured zero

    measures = LangfuseTraceMetrics(
        trace_id="t", total_cost=0.0, cost_unmeasured=True
    ).to_measures_dict(prefix="lf_")
    assert "lf_total_cost" not in measures
    assert measures["lf_cost_unmeasured"] == 1.0
    assert (
        LangfuseTraceMetrics(trace_id="t", total_cost=0.0).to_measures_dict()[
            "total_cost"
        ]
        == 0.0
    )
