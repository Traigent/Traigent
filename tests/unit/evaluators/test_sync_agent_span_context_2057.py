"""Traigent#2057: ``add_agent_span()`` must work from a SYNC agent function.

The workflow trace context is set on the orchestrator's task; a sync user
function runs in a worker thread, and ``contextvars`` do not cross that
boundary on their own. ``BaseEvaluator._execute_sync_in_thread`` now runs the
worker inside ``copy_context()`` of the caller, so the span is accepted. This
pins that behaviour through the real evaluator dispatch path, which the
helper-level tests in ``tests/unit/observability`` never exercise.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from traigent.config.context import WorkflowTraceContext
from traigent.evaluators.local import LocalEvaluator
from traigent.observability.agent_spans import add_agent_span


class _RecordingTraceManager:
    is_enabled = True

    def __init__(self) -> None:
        self.nodes: list[str] = []
        self.spans: list[Any] = []

    def register_node(self, node_id: str, node_type: str = "agent") -> None:
        self.nodes.append(node_id)

    def collect_span(self, span: Any) -> None:
        self.spans.append(span)


@pytest.mark.asyncio
async def test_sync_agent_function_span_is_accepted_through_evaluator() -> None:
    manager = _RecordingTraceManager()
    caller_thread = threading.get_ident()
    seen: dict[str, Any] = {}

    def sync_agent(question: str) -> str:
        seen["thread"] = threading.get_ident()
        seen["receipt"] = add_agent_span("planner", input_tokens=3, output_tokens=5)
        return question.upper()

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    with WorkflowTraceContext(
        {
            "workflow_trace_manager": manager,
            "configuration_run_id": "cr-2057",
            "workflow_trace_id": "trace-2057",
        }
    ):
        output, error = await evaluator._execute_function(
            sync_agent, {}, {"question": "hi"}
        )

    assert error is None
    assert output == "HI"
    # The sync function really ran off the caller's thread (the hazard).
    assert seen["thread"] != caller_thread
    receipt = seen["receipt"]
    assert receipt.reason is None, f"span dropped: {receipt.reason}"
    assert receipt.accepted is True
    assert [s.trace_id for s in manager.spans] == ["trace-2057"]
    assert manager.nodes == ["planner"]
