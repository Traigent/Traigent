"""Tests for public agent workflow span emission."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any

from traigent.api.types import AgentDefinition
from traigent.config.context import (
    TrialContext,
    WorkflowTraceContext,
    copy_context_to_thread,
)
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.core.workflow_trace_manager import WorkflowTraceManager
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.observability import add_agent_span
from traigent.optimizers.base import BaseOptimizer


class _BackendClient:
    def get_session_mapping(self, session_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            experiment_id="exp_agent_spans",
            experiment_run_id=f"run_{session_id}",
        )


class _Optimizer(BaseOptimizer):
    def __init__(self, config_space: dict[str, Any]) -> None:
        super().__init__(config_space=config_space, objectives=[])

    def suggest_next_trial(self, history: list[Any]) -> dict[str, Any]:
        return {}

    def should_stop(self, history: list[Any]) -> bool:
        return True


class _Evaluator(BaseEvaluator):
    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        **kwargs: Any,
    ) -> Any:
        raise AssertionError("not used by agent span graph registration tests")


def _manager() -> WorkflowTraceManager:
    return WorkflowTraceManager(
        workflow_traces_tracker=object(),
        backend_client=_BackendClient(),
        function_descriptor=SimpleNamespace(identifier="mock_optimized_function"),
        optimizer_config_space={"temperature": object()},
        max_trials=2,
        optimizer_class_name="MockOptimizer",
        optimization_id="trace-agent-spans",
    )


def _active_trial(manager: WorkflowTraceManager) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(TrialContext(trial_id="trial-1", metadata={"config": {}}))
    stack.enter_context(
        WorkflowTraceContext(
            {
                "configuration_run_id": "config-run-1",
                "workflow_trace_id": "trace-agent-spans",
                "workflow_trace_manager": manager,
            }
        )
    )
    return stack


def _graph_node_ids(manager: WorkflowTraceManager) -> set[str]:
    graph = manager._create_optimization_workflow_graph("session-1")
    assert graph is not None
    return {node.id for node in graph.nodes}


def test_add_agent_span_inside_trial_collects_span_and_registers_node() -> None:
    manager = _manager()

    with _active_trial(manager):
        add_agent_span(
            "planner",
            input_tokens=12,
            output_tokens=5,
            cost_usd=0.03,
            latency_ms=42,
            metadata={"retrieval_count": 3},
        )

    spans_by_run = manager._group_spans_by_config_run()
    span = spans_by_run["config-run-1"][0]
    payload = span.to_dict()

    assert span.node_id == "planner"
    assert span.trace_id == "trace-agent-spans"
    assert span.configuration_run_id == "config-run-1"
    assert span.input_tokens == 12
    assert span.output_tokens == 5
    assert span.cost_usd == 0.03
    assert payload["metadata"]["retrieval_count"] == 3
    assert payload["metadata"]["latency_ms"] == 42.0
    assert "input_data" not in payload
    assert "output_data" not in payload
    assert "planner" in _graph_node_ids(manager)


def test_add_agent_span_registers_two_distinct_nodes() -> None:
    manager = _manager()

    with _active_trial(manager):
        add_agent_span("planner", input_tokens=1)
        add_agent_span("critic", output_tokens=2)

    node_ids = _graph_node_ids(manager)
    assert {"planner", "critic"}.issubset(node_ids)


def test_add_agent_span_outside_trial_is_debug_noop(
    caplog: Any,
) -> None:
    with caplog.at_level(logging.DEBUG):
        add_agent_span("planner", input_tokens=1)

    assert "no active optimization trial context" in caplog.text


def test_add_agent_span_collection_failure_logs_warning(caplog: Any) -> None:
    manager = _manager()

    def fail_collect(_: Any) -> None:
        raise RuntimeError("collector down")

    manager.collect_span = fail_collect  # type: ignore[method-assign]

    with _active_trial(manager), caplog.at_level(logging.WARNING):
        add_agent_span("planner", input_tokens=1)

    assert "Failed to collect agent workflow span" in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_add_agent_span_drops_text_metadata() -> None:
    manager = _manager()

    with _active_trial(manager):
        add_agent_span(
            "privacy_node",
            metadata={
                "prompt": "do not ship",
                "expected": "do not ship",
                "actual_output": "do not ship",
                "completion": "do not ship",
                "safe_score": 0.91,
                "freeform_label": "do not ship",
            },
        )

    span = manager._group_spans_by_config_run()["config-run-1"][0]
    payload_text = str(span.to_dict())

    assert span.metadata == {"safe_score": 0.91}
    assert "do not ship" not in payload_text


def test_add_agent_span_drops_credential_like_metadata_keys() -> None:
    """Regression for issue #1649: agent_spans' own metadata-key filter used
    to only recognize free-form content markers (prompt/response/etc.), so a
    numeric-valued credential-shaped key such as `auth_token_count` would
    have passed through untouched. It now shares the canonical union set
    with the other two sanitizers and drops it too."""
    manager = _manager()

    with _active_trial(manager):
        add_agent_span(
            "privacy_node",
            metadata={
                "auth_token_count": 5,
                "api_key_rotations": 2,
                "safe_score": 0.91,
            },
        )

    span = manager._group_spans_by_config_run()["config-run-1"][0]

    assert span.metadata == {"safe_score": 0.91}


def test_declared_agents_and_agent_prefixes_register_graph_nodes(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setattr(
        BackendSessionManager,
        "create_backend_client",
        staticmethod(lambda config: _BackendClient()),
    )

    explicit_orchestrator = OptimizationOrchestrator(
        optimizer=_Optimizer({"planner_model": ["gpt-4"], "critic_model": ["gpt-4"]}),
        evaluator=_Evaluator(),
        max_trials=1,
        workflow_traces_tracker=object(),
        agents={
            "planner": AgentDefinition(
                display_name="Planner",
                parameter_keys=["planner_model"],
            ),
            "critic": AgentDefinition(
                display_name="Critic",
                parameter_keys=["critic_model"],
            ),
        },
    )
    explicit_nodes = _graph_node_ids(explicit_orchestrator._workflow_trace_manager)
    assert {"planner", "critic"}.issubset(explicit_nodes)

    prefix_orchestrator = OptimizationOrchestrator(
        optimizer=_Optimizer({"planner_model": ["gpt-4"]}),
        evaluator=_Evaluator(),
        max_trials=1,
        workflow_traces_tracker=object(),
        agent_prefixes=["planner"],
    )
    prefix_nodes = _graph_node_ids(prefix_orchestrator._workflow_trace_manager)
    assert "planner" in prefix_nodes


def test_add_agent_span_works_with_restored_thread_context() -> None:
    manager = _manager()

    with _active_trial(manager):
        snapshot = copy_context_to_thread()

    def worker() -> None:
        with snapshot.restore():
            add_agent_span("threaded_agent", input_tokens=7)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)

    spans = manager._group_spans_by_config_run()["config-run-1"]
    assert spans[0].node_id == "threaded_agent"
    assert "threaded_agent" in _graph_node_ids(manager)
