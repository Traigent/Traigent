"""Tests for WorkflowTraceManager (traigent.core.workflow_trace_manager)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.core.workflow_trace_manager import WorkflowTraceManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeSpan:
    """Minimal SpanPayload for testing."""

    trace_id: str = "trace-abc123"
    configuration_run_id: str = "run-1"


@dataclass
class _FakeIngestResponse:
    success: bool = True
    graph_id: str | None = None
    error: str | None = None


def _make_manager(
    *,
    tracker: Any = None,
    backend_client: Any = None,
    function_descriptor: Any = None,
    config_space: dict | None = None,
    max_trials: int | None = 5,
) -> WorkflowTraceManager:
    return WorkflowTraceManager(
        workflow_traces_tracker=tracker,
        backend_client=backend_client,
        function_descriptor=function_descriptor,
        optimizer_config_space=config_space or {"temp": [0.0, 1.0]},
        max_trials=max_trials,
        optimizer_class_name="random",
        optimization_id="opt-123",
    )


# ---------------------------------------------------------------------------
# collect_span
# ---------------------------------------------------------------------------


class TestCollectSpan:
    def test_collects_when_tracker_set(self) -> None:
        mgr = _make_manager(tracker=MagicMock())
        span = _FakeSpan()
        mgr.collect_span(span)
        assert len(mgr._collected_spans) == 1

    def test_ignores_when_no_tracker(self) -> None:
        mgr = _make_manager(tracker=None)
        mgr.collect_span(_FakeSpan())
        assert len(mgr._collected_spans) == 0


# ---------------------------------------------------------------------------
# submit_traces
# ---------------------------------------------------------------------------


class TestSubmitTraces:
    @pytest.mark.asyncio
    async def test_returns_early_when_no_tracker(self) -> None:
        mgr = _make_manager(tracker=None)
        await mgr.submit_traces()  # should not raise

    @pytest.mark.asyncio
    async def test_returns_early_when_no_spans(self) -> None:
        mgr = _make_manager(tracker=MagicMock())
        await mgr.submit_traces()  # no spans collected

    @pytest.mark.asyncio
    async def test_skips_in_offline_mode(self) -> None:
        tracker = MagicMock()
        mgr = _make_manager(tracker=tracker)
        mgr._collected_spans = [_FakeSpan()]
        with patch(
            "traigent.core.workflow_trace_manager.is_backend_offline", return_value=True
        ):
            await mgr.submit_traces()
        assert mgr._collected_spans == []
        tracker.ingest_traces_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_mock_session(self) -> None:
        tracker = MagicMock()
        mgr = _make_manager(tracker=tracker)
        mgr._collected_spans = [_FakeSpan()]
        with patch(
            "traigent.core.workflow_trace_manager.is_backend_offline",
            return_value=False,
        ):
            await mgr.submit_traces(session_id="mock_session_abc")
        assert mgr._collected_spans == []

    @pytest.mark.asyncio
    async def test_skips_mock_session_dash(self) -> None:
        tracker = MagicMock()
        mgr = _make_manager(tracker=tracker)
        mgr._collected_spans = [_FakeSpan()]
        with patch(
            "traigent.core.workflow_trace_manager.is_backend_offline",
            return_value=False,
        ):
            await mgr.submit_traces(session_id="mock-session-xyz")
        assert mgr._collected_spans == []

    @pytest.mark.asyncio
    async def test_submits_spans_successfully(self) -> None:
        tracker = MagicMock()
        tracker.ingest_traces_async = AsyncMock(
            return_value=_FakeIngestResponse(success=True, graph_id="graph-1")
        )
        mgr = _make_manager(tracker=tracker)
        mgr._collected_spans = [_FakeSpan(trace_id="t1", configuration_run_id="run-1")]

        with patch(
            "traigent.core.workflow_trace_manager.is_backend_offline",
            return_value=False,
        ):
            await mgr.submit_traces(session_id="real-session-id")

        tracker.ingest_traces_async.assert_called_once()
        assert mgr._collected_spans == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_message",
        [
            "Auth failed (401)",
            "Auth failed (403)",
            "Unauthorized",
            "Forbidden",
        ],
    )
    async def test_auth_rejection_logs_at_debug_not_warning(
        self, caplog, error_message: str
    ) -> None:
        import logging

        tracker = MagicMock()
        tracker.ingest_traces_async = AsyncMock(
            return_value=_FakeIngestResponse(success=False, error=error_message)
        )
        mgr = _make_manager(tracker=tracker)
        mgr._collected_spans = [_FakeSpan()]

        with caplog.at_level(logging.DEBUG), patch(
            "traigent.core.workflow_trace_manager.is_backend_offline",
            return_value=False,
        ):
            await mgr.submit_traces(session_id="real-session")

        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        assert any("auth rejected" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_handles_failed_ingestion_logs_warning(self, caplog) -> None:
        import logging

        tracker = MagicMock()
        tracker.ingest_traces_async = AsyncMock(
            return_value=_FakeIngestResponse(success=False, error="network error")
        )
        mgr = _make_manager(tracker=tracker)
        mgr._collected_spans = [_FakeSpan()]

        with patch(
            "traigent.core.workflow_trace_manager.is_backend_offline",
            return_value=False,
        ):
            with caplog.at_level(logging.WARNING):
                await mgr.submit_traces(session_id="real-session")

        assert mgr._collected_spans == []
        assert any("Failed to submit spans" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_handles_exception_during_submission(self) -> None:
        tracker = MagicMock()
        tracker.ingest_traces_async = AsyncMock(side_effect=RuntimeError("boom"))
        mgr = _make_manager(tracker=tracker)
        mgr._collected_spans = [_FakeSpan()]

        with patch(
            "traigent.core.workflow_trace_manager.is_backend_offline",
            return_value=False,
        ):
            await mgr.submit_traces(session_id="real-session")

        # Spans should be cleared even on error
        assert mgr._collected_spans == []

    @pytest.mark.asyncio
    async def test_groups_spans_by_config_run(self) -> None:
        tracker = MagicMock()
        tracker.ingest_traces_async = AsyncMock(
            return_value=_FakeIngestResponse(success=True)
        )
        mgr = _make_manager(tracker=tracker)
        mgr._collected_spans = [
            _FakeSpan(configuration_run_id="run-1"),
            _FakeSpan(configuration_run_id="run-2"),
            _FakeSpan(configuration_run_id="run-1"),
        ]

        with patch(
            "traigent.core.workflow_trace_manager.is_backend_offline",
            return_value=False,
        ):
            await mgr.submit_traces(session_id="real-session")

        # Should be called twice (once for run-1, once for run-2)
        assert tracker.ingest_traces_async.call_count == 2


# ---------------------------------------------------------------------------
# _group_spans_by_config_run
# ---------------------------------------------------------------------------


class TestGroupSpansByConfigRun:
    def test_groups_correctly(self) -> None:
        mgr = _make_manager(tracker=MagicMock())
        mgr._collected_spans = [
            _FakeSpan(configuration_run_id="a"),
            _FakeSpan(configuration_run_id="b"),
            _FakeSpan(configuration_run_id="a"),
        ]
        groups = mgr._group_spans_by_config_run()
        assert len(groups) == 2
        assert len(groups["a"]) == 2
        assert len(groups["b"]) == 1


# ---------------------------------------------------------------------------
# _create_optimization_workflow_graph
# ---------------------------------------------------------------------------


class TestCreateOptimizationWorkflowGraph:
    def test_returns_none_when_no_experiment_id(self) -> None:
        mgr = _make_manager(tracker=MagicMock())
        result = mgr._create_optimization_workflow_graph(session_id=None)
        assert result is None

    def test_returns_none_for_mock_experiment_id(self) -> None:
        backend = MagicMock()
        mapping = MagicMock()
        mapping.experiment_id = "mock_exp_123"
        backend.get_session_mapping.return_value = mapping
        mgr = _make_manager(tracker=MagicMock(), backend_client=backend)
        result = mgr._create_optimization_workflow_graph(session_id="sess-1")
        assert result is None

    def test_returns_none_for_mock_exp_dash(self) -> None:
        backend = MagicMock()
        mapping = MagicMock()
        mapping.experiment_id = "mock-exp-456"
        backend.get_session_mapping.return_value = mapping
        mgr = _make_manager(tracker=MagicMock(), backend_client=backend)
        result = mgr._create_optimization_workflow_graph(session_id="sess-1")
        assert result is None

    def test_creates_graph_with_real_experiment(self) -> None:
        backend = MagicMock()
        mapping = MagicMock()
        mapping.experiment_id = "real-exp-789"
        mapping.experiment_run_id = "run-xyz"
        backend.get_session_mapping.return_value = mapping

        descriptor = MagicMock()
        descriptor.identifier = "my_func"

        mgr = _make_manager(
            tracker=MagicMock(),
            backend_client=backend,
            function_descriptor=descriptor,
            config_space={"temp": [0.0, 1.0], "model": ["a", "b"]},
        )
        graph = mgr._create_optimization_workflow_graph(session_id="sess-1")
        assert graph is not None
        assert graph.experiment_id == "real-exp-789"
        assert graph.experiment_run_id == "run-xyz"
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.metadata["function_name"] == "my_func"

    def test_creates_graph_without_descriptor(self) -> None:
        backend = MagicMock()
        mapping = MagicMock()
        mapping.experiment_id = "real-exp-1"
        mapping.experiment_run_id = None
        backend.get_session_mapping.return_value = mapping

        mgr = _make_manager(
            tracker=MagicMock(),
            backend_client=backend,
            function_descriptor=None,
        )
        graph = mgr._create_optimization_workflow_graph(session_id="sess-1")
        assert graph is not None
        assert graph.metadata["function_name"] == "optimization"


# ---------------------------------------------------------------------------
# _get_experiment_id_from_session / _get_experiment_run_id_from_session
# ---------------------------------------------------------------------------


class TestGetExperimentIds:
    def test_returns_none_without_session(self) -> None:
        mgr = _make_manager(tracker=MagicMock(), backend_client=MagicMock())
        assert mgr._get_experiment_id_from_session(None) is None

    def test_returns_none_without_client(self) -> None:
        mgr = _make_manager(tracker=MagicMock(), backend_client=None)
        assert mgr._get_experiment_id_from_session("sess") is None

    def test_returns_experiment_id(self) -> None:
        backend = MagicMock()
        mapping = MagicMock()
        mapping.experiment_id = "exp-123"
        backend.get_session_mapping.return_value = mapping
        mgr = _make_manager(tracker=MagicMock(), backend_client=backend)
        assert mgr._get_experiment_id_from_session("sess") == "exp-123"

    def test_returns_none_on_exception(self) -> None:
        backend = MagicMock()
        backend.get_session_mapping.side_effect = RuntimeError("fail")
        mgr = _make_manager(tracker=MagicMock(), backend_client=backend)
        assert mgr._get_experiment_id_from_session("sess") is None

    def test_returns_experiment_run_id(self) -> None:
        backend = MagicMock()
        mapping = MagicMock()
        mapping.experiment_run_id = "run-456"
        backend.get_session_mapping.return_value = mapping
        mgr = _make_manager(tracker=MagicMock(), backend_client=backend)
        assert mgr._get_experiment_run_id_from_session("sess") == "run-456"

    def test_run_id_returns_none_without_session(self) -> None:
        mgr = _make_manager(tracker=MagicMock(), backend_client=MagicMock())
        assert mgr._get_experiment_run_id_from_session(None) is None

    def test_run_id_returns_none_on_exception(self) -> None:
        backend = MagicMock()
        backend.get_session_mapping.side_effect = RuntimeError("fail")
        mgr = _make_manager(tracker=MagicMock(), backend_client=backend)
        assert mgr._get_experiment_run_id_from_session("sess") is None
