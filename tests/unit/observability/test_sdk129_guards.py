"""Regression tests for the SDK-129 overflow and type guards.

Each test fails on develop (the guard is missing) and passes on this branch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from traigent.config.context import TrialContext, WorkflowTraceContext
from traigent.core.workflow_trace_manager import WorkflowTraceManager
from traigent.observability import add_agent_span
from traigent.observability.dtos import SessionDTO


class _BackendClient:
    def get_session_mapping(self, session_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            experiment_id="exp_sdk129",
            experiment_run_id=f"run_{session_id}",
        )


def _manager() -> WorkflowTraceManager:
    return WorkflowTraceManager(
        workflow_traces_tracker=object(),
        backend_client=_BackendClient(),
        function_descriptor=SimpleNamespace(identifier="mock_optimized_function"),
        optimizer_config_space={"temperature": object()},
        max_trials=2,
        optimizer_class_name="MockOptimizer",
        optimization_id="trace-sdk129",
    )


def _run_in_trial(manager: WorkflowTraceManager, call: Any) -> Any:
    with TrialContext(trial_id="trial-1", metadata={"config": {}}):
        with WorkflowTraceContext(
            {
                "configuration_run_id": "config-run-1",
                "workflow_trace_id": "trace-sdk129",
                "workflow_trace_manager": manager,
            }
        ):
            return call()


def test_huge_int_metadata_value_is_dropped_not_raised() -> None:
    """An int too large for float (10**400) must be dropped with a receipt entry.

    On develop, `_is_safe_number` calls float() on it and raises OverflowError
    out of the public `add_agent_span` call.
    """
    manager = _manager()

    result = _run_in_trial(
        manager,
        lambda: add_agent_span(
            "planner",
            metadata={"retrieval_count": 10**400, "safe_score": 0.5},
        ),
    )

    assert result.accepted is True
    assert "retrieval_count" in result.dropped_metadata_keys
    span = manager._group_spans_by_config_run()["config-run-1"][0]
    assert "retrieval_count" not in span.to_dict()["metadata"]
    assert span.to_dict()["metadata"]["safe_score"] == 0.5


def test_unrepresentable_latency_keeps_span_with_zero_length_interval() -> None:
    """A finite but unrepresentable latency (1e30 ms) must not raise.

    On develop, `timedelta(milliseconds=1e30)` raises OverflowError out of
    `add_agent_span`. On this branch the span is kept and its timestamps
    collapse to start == end.
    """
    manager = _manager()

    result = _run_in_trial(
        manager,
        lambda: add_agent_span("planner", input_tokens=3, latency_ms=1e30),
    )

    assert result.accepted is True
    spans = manager._group_spans_by_config_run()["config-run-1"]
    assert len(spans) == 1
    payload = spans[0].to_dict()
    assert payload["input_tokens"] == 3
    assert payload["start_time"] == payload["end_time"]


def test_non_string_optional_identifier_is_rejected() -> None:
    """A short list passes the length check; it must be rejected as a non-string.

    On develop, `_validate_optional_string` only checks length, so the
    DTO is constructed without error and the test fails.
    """
    try:
        SessionDTO(id="session_sdk129", user_id=["a", "b"])  # type: ignore[arg-type]
    except ValueError as exc:
        assert str(exc) == "user_id must be a string"
    else:
        raise AssertionError("non-string user_id was accepted")
