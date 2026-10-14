"""Tests for WorkflowTraceManager (traigent.core.workflow_trace_manager)."""

from __future__ import annotations

from dataclasses import dataclass
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

    def test_rebind_configuration_run_id_updates_matching_collected_spans(self) -> None:
        mgr = _make_manager(tracker=MagicMock())
        mgr._collected_spans = [
            _FakeSpan(configuration_run_id="client-trial-1"),
            _FakeSpan(configuration_run_id="other-trial"),
            _FakeSpan(configuration_run_id="client-trial-1"),
        ]

        updated = mgr.rebind_configuration_run_id(
            "client-trial-1",
            "backend-trial-1",
        )

        assert updated == 2
        assert [span.configuration_run_id for span in mgr._collected_spans] == [
            "backend-trial-1",
            "other-trial",
            "backend-trial-1",
        ]


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
    async def test_warns_on_zero_spans_when_trials_executed(self, caplog) -> None:
        """#2060: default (unknown trial count) preserves the wiring-fault
        warning, and an explicit trials_executed=True keeps it too."""
        import logging

        mgr = _make_manager(tracker=MagicMock())

        with caplog.at_level(logging.WARNING):
            await mgr.submit_traces(trials_executed=True)

        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert "no workflow spans" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_no_warning_on_zero_spans_when_no_trials_executed(
        self, caplog
    ) -> None:
        """#2060: a legitimate zero-trial completion (e.g. an exhausted
        shared ExecutionBudget, issue #1980) must not fire the zero-span
        wiring-fault WARNING; it stays at DEBUG."""
        import logging

        mgr = _make_manager(tracker=MagicMock())

        with caplog.at_level(
            logging.DEBUG, logger="traigent.core.workflow_trace_manager"
        ):
            await mgr.submit_traces(trials_executed=False)

        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        assert any("zero trials" in r.message.lower() for r in caplog.records)

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

        with (
            caplog.at_level(
                logging.DEBUG, logger="traigent.core.workflow_trace_manager"
            ),
            patch(
                "traigent.core.workflow_trace_manager.is_backend_offline",
                return_value=False,
            ),
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


# ---------------------------------------------------------------------------
# submit_traces: ingest-count honesty (#2578)
# ---------------------------------------------------------------------------


def _ingest_resp(data: dict[str, Any]) -> Any:
    from traigent.integrations.observability.workflow_traces import (
        TraceIngestionResponse,
    )

    return TraceIngestionResponse.from_dict({"success": True, "data": data})


async def _submit(responses: list[Any], spans: list[_FakeSpan], caplog: Any) -> None:
    import logging

    tracker = MagicMock()
    tracker.ingest_traces_async = AsyncMock(side_effect=responses)
    mgr = _make_manager(tracker=tracker)
    mgr._collected_spans = list(spans)
    with (
        caplog.at_level(logging.DEBUG, logger="traigent.core.workflow_trace_manager"),
        patch(
            "traigent.core.workflow_trace_manager.is_backend_offline",
            return_value=False,
        ),
    ):
        await mgr.submit_traces(session_id="real-session")
    assert mgr._collected_spans == []


def _span_msgs(caplog: Any) -> list[tuple[int, str]]:
    return [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if "workflow spans" in r.getMessage()
    ]


class TestTraceIngestionResponseCounts:
    def test_reported_count_and_duplicates_are_parsed(self) -> None:
        resp = _ingest_resp(
            {"spans_ingested": 1, "skipped_duplicates": 2, "skipped_span_ids": ["a"]}
        )
        assert resp.spans_ingested == 1
        assert resp.spans_ingested_reported is True
        assert resp.skipped_duplicates == 2

    def test_absent_count_is_marked_unreported(self) -> None:
        resp = _ingest_resp({"graph_id": "g"})
        assert resp.spans_ingested == 0
        assert resp.spans_ingested_reported is False
        assert resp.skipped_duplicates == 0

    @pytest.mark.parametrize("bad", ["3", None, 2.5, True, -1])
    def test_non_integer_count_is_marked_unreported(self, bad: Any) -> None:
        resp = _ingest_resp({"spans_ingested": bad, "skipped_duplicates": bad})
        assert resp.spans_ingested == 0
        assert resp.spans_ingested_reported is False
        assert resp.skipped_duplicates == 0

    def test_direct_construction_keeps_stated_count(self) -> None:
        from traigent.integrations.observability.workflow_traces import (
            TraceIngestionResponse,
        )

        resp = TraceIngestionResponse(success=True, spans_ingested=4)
        assert resp.spans_ingested_reported is True
        assert resp.skipped_duplicates == 0


class TestSubmitTracesIngestCounts:
    @pytest.mark.asyncio
    async def test_full_ingest_logs_submitted_at_info(self, caplog) -> None:
        import logging

        await _submit(
            [_ingest_resp({"spans_ingested": 3})],
            [_FakeSpan(trace_id="t1") for _ in range(3)],
            caplog,
        )
        msgs = _span_msgs(caplog)
        assert any(
            lvl == logging.INFO and msg.startswith("Submitted 3 workflow spans")
            for lvl, msg in msgs
        ), msgs
        assert not any(lvl >= logging.WARNING for lvl, _ in msgs), msgs

    @pytest.mark.asyncio
    async def test_shortfall_warns_with_counts_not_submitted_n(self, caplog) -> None:
        import logging

        await _submit(
            [_ingest_resp({"spans_ingested": 1, "skipped_duplicates": 1})],
            [_FakeSpan(trace_id="t1") for _ in range(3)],
            caplog,
        )
        msgs = _span_msgs(caplog)
        assert not any("Submitted 3" in msg for _, msg in msgs), msgs
        warnings = [msg for lvl, msg in msgs if lvl == logging.WARNING]
        assert len(warnings) == 1, msgs
        assert "ingested 1 of 3 workflow spans" in warnings[0]
        assert "1 skipped as duplicate span ids" in warnings[0]
        assert "1 not ingested" in warnings[0]
        assert "missing those spans" in warnings[0]

    @pytest.mark.asyncio
    async def test_absent_count_is_reported_as_unconfirmed(self, caplog) -> None:
        import logging

        await _submit(
            [_ingest_resp({"graph_id": "g"})],
            [_FakeSpan(trace_id="t1") for _ in range(2)],
            caplog,
        )
        msgs = _span_msgs(caplog)
        assert not any(lvl >= logging.WARNING for lvl, _ in msgs), msgs
        assert any("count unconfirmed" in msg and "2" in msg for _, msg in msgs), msgs
        assert not any(msg.startswith("Backend ingested") for _, msg in msgs), msgs

    @pytest.mark.asyncio
    async def test_counts_aggregate_across_config_runs(self, caplog) -> None:
        import logging

        await _submit(
            [
                _ingest_resp({"spans_ingested": 2}),
                _ingest_resp({"spans_ingested": 0}),
            ],
            [
                _FakeSpan(trace_id="t1", configuration_run_id="run-1"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-1"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-2"),
            ],
            caplog,
        )
        warnings = [msg for lvl, msg in _span_msgs(caplog) if lvl == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert "ingested 2 of 3 workflow spans" in warnings[0]

    @pytest.mark.asyncio
    async def test_duplicates_only_shortfall_does_not_claim_missing(
        self, caplog
    ) -> None:
        import logging

        await _submit(
            [_ingest_resp({"spans_ingested": 1, "skipped_duplicates": 2})],
            [_FakeSpan(trace_id="t1") for _ in range(3)],
            caplog,
        )
        warnings = [msg for lvl, msg in _span_msgs(caplog) if lvl == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert "ingested 1 of 3 workflow spans" in warnings[0]
        assert "2 skipped as duplicate span ids" in warnings[0]
        assert "missing" not in warnings[0]
        assert "unreported reason" not in warnings[0]

    @pytest.mark.asyncio
    async def test_duplicate_count_is_clamped_to_the_shortfall(self, caplog) -> None:
        import logging

        await _submit(
            [_ingest_resp({"spans_ingested": 2, "skipped_duplicates": 5})],
            [_FakeSpan(trace_id="t1") for _ in range(3)],
            caplog,
        )
        warnings = [msg for lvl, msg in _span_msgs(caplog) if lvl == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert "1 skipped as duplicate span ids" in warnings[0]

    @pytest.mark.asyncio
    async def test_over_report_cannot_hide_another_groups_shortfall(
        self, caplog
    ) -> None:
        import logging

        await _submit(
            [
                _ingest_resp({"spans_ingested": 5}),
                _ingest_resp({"spans_ingested": 0}),
            ],
            [
                _FakeSpan(trace_id="t1", configuration_run_id="run-1"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-1"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-1"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-2"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-2"),
            ],
            caplog,
        )
        warnings = [msg for lvl, msg in _span_msgs(caplog) if lvl == logging.WARNING]
        assert len(warnings) == 2, warnings
        assert "reported 5 workflow spans ingested" in warnings[0]
        assert "only 3 were sent" in warnings[0]
        assert "ingested 3 of 5 workflow spans" in warnings[1]
        assert "2 not ingested" in warnings[1]

    @pytest.mark.asyncio
    async def test_mixed_shortfall_and_unconfirmed_groups(self, caplog) -> None:
        import logging

        await _submit(
            [
                _ingest_resp({"spans_ingested": 0}),
                _ingest_resp({"graph_id": "g"}),
            ],
            [
                _FakeSpan(trace_id="t1", configuration_run_id="run-1"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-2"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-2"),
            ],
            caplog,
        )
        warnings = [msg for lvl, msg in _span_msgs(caplog) if lvl == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert "ingested 0 of 1 workflow spans" in warnings[0]
        assert "2 more submitted with the ingested count unconfirmed" in warnings[0]

    @pytest.mark.asyncio
    async def test_response_without_count_fields_is_unconfirmed(self, caplog) -> None:
        import logging

        await _submit(
            [_FakeIngestResponse(success=True), _FakeIngestResponse(success=True)],
            [
                _FakeSpan(trace_id="t1", configuration_run_id="run-1"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-2"),
            ],
            caplog,
        )
        msgs = _span_msgs(caplog)
        assert not any(lvl >= logging.WARNING for lvl, _ in msgs), msgs
        assert any("count unconfirmed" in msg for _, msg in msgs), msgs

    @pytest.mark.asyncio
    async def test_over_report_alone_is_logged_not_absorbed(self, caplog) -> None:
        import logging

        await _submit(
            [_ingest_resp({"spans_ingested": 9})],
            [_FakeSpan(trace_id="t1") for _ in range(3)],
            caplog,
        )
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1, warnings
        assert "reported 9 workflow spans ingested" in warnings[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fields",
        [
            {"spans_ingested_reported": True, "spans_ingested": -1},
            {"spans_ingested_reported": True, "spans_ingested": "3"},
            {"spans_ingested_reported": "yes", "spans_ingested": 1},
            {"spans_ingested_reported": True, "spans_ingested": True},
            {"spans_ingested": 0},
        ],
    )
    async def test_invalid_count_values_are_unconfirmed(
        self, caplog, fields: dict[str, Any]
    ) -> None:
        import logging

        await _submit(
            [_CountResponse(**fields)],
            [_FakeSpan(trace_id="t1") for _ in range(3)],
            caplog,
        )
        msgs = _span_msgs(caplog)
        assert not any(r.levelno >= logging.WARNING for r in caplog.records), msgs
        assert any("count unconfirmed" in msg for _, msg in msgs), msgs

    @pytest.mark.asyncio
    async def test_magicmock_response_does_not_abort_submission(self, caplog) -> None:
        import logging

        await _submit(
            [MagicMock(success=True), MagicMock(success=True)],
            [
                _FakeSpan(trace_id="t1", configuration_run_id="run-1"),
                _FakeSpan(trace_id="t1", configuration_run_id="run-2"),
            ],
            caplog,
        )
        assert not any(r.levelno >= logging.WARNING for r in caplog.records), [
            r.getMessage() for r in caplog.records
        ]
        assert any("count unconfirmed" in msg for _, msg in _span_msgs(caplog))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dups", [None, -2, "1"])
    async def test_missing_or_invalid_duplicates_count_as_zero_duplicates(
        self, caplog, dups: Any
    ) -> None:
        import logging

        fields: dict[str, Any] = {"spans_ingested_reported": True, "spans_ingested": 1}
        if dups is not None:
            fields["skipped_duplicates"] = dups
        await _submit(
            [_CountResponse(**fields)],
            [_FakeSpan(trace_id="t1") for _ in range(3)],
            caplog,
        )
        warnings = [msg for lvl, msg in _span_msgs(caplog) if lvl == logging.WARNING]
        assert len(warnings) == 1, warnings
        assert "0 skipped as duplicate span ids, 2 not ingested" in warnings[0]


class _CountResponse:
    """Duck-typed ingest response carrying only the attributes given."""

    def __init__(self, **fields: Any) -> None:
        self.success = True
        self.graph_id = None
        self.error = None
        for key, value in fields.items():
            setattr(self, key, value)


def _spans(**counts: int) -> list[_FakeSpan]:
    return [
        _FakeSpan(trace_id="t1", configuration_run_id=run)
        for run, n in counts.items()
        for _ in range(n)
    ]


class TestSubmitTracesExactMessages:
    """Pin the exact counts and level of every summary line (#2578)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("responses", "spans", "expected"),
        [
            pytest.param(
                [{"spans_ingested": 3}, {}],
                {"r1": 3, "r2": 2},
                [
                    (
                        "INFO",
                        "Submitted 5 workflow spans for trace t1... (backend did "
                        "not report how many were ingested for 2 of them; count "
                        "unconfirmed)",
                    )
                ],
                id="partial-unconfirmed",
            ),
            pytest.param(
                [{"spans_ingested": 1, "skipped_duplicates": 2}, {"graph_id": "g"}],
                {"r1": 3, "r2": 2},
                [
                    (
                        "WARNING",
                        "Backend ingested 1 of 3 workflow spans for trace t1..., "
                        "graph_id=g; 2 skipped as duplicate span ids (already "
                        "stored, or repeated within the batch); 2 more submitted "
                        "with the ingested count unconfirmed.",
                    )
                ],
                id="duplicates-plus-unconfirmed",
            ),
            pytest.param(
                [{"spans_ingested": 0, "skipped_duplicates": 1, "graph_id": "g"}],
                {"r1": 3},
                [
                    (
                        "WARNING",
                        "Backend ingested 0 of 3 workflow spans for trace t1..., "
                        "graph_id=g (1 skipped as duplicate span ids, 2 not "
                        "ingested for an unreported reason); the workflow traces "
                        "view will be missing those spans.",
                    )
                ],
                id="dropped",
            ),
            pytest.param(
                [{"spans_ingested": 9}],
                {"r1": 3},
                [
                    (
                        "WARNING",
                        "Backend reported 9 workflow spans ingested for config_run "
                        "r1 but only 3 were sent; counting 3.",
                    ),
                    ("INFO", "Submitted 3 workflow spans for trace t1..."),
                ],
                id="over-report",
            ),
        ],
    )
    async def test_summary_lines(
        self,
        caplog,
        responses: list[dict[str, Any]],
        spans: dict[str, int],
        expected: list[tuple[str, str]],
    ) -> None:
        await _submit([_ingest_resp(r) for r in responses], _spans(**spans), caplog)
        lines = [
            (r.levelname, r.getMessage())
            for r in caplog.records
            if "workflow spans" in r.getMessage()
        ]
        assert lines == expected

    @pytest.mark.asyncio
    async def test_all_groups_failed_logs_no_summary(self, caplog) -> None:
        await _submit(
            [
                _FakeIngestResponse(success=False, error="network error"),
                _FakeIngestResponse(success=False, error="network error"),
            ],
            _spans(r1=1, r2=1),
            caplog,
        )
        assert not any(
            "Submitted" in r.getMessage() or "Backend ingested" in r.getMessage()
            for r in caplog.records
        ), [r.getMessage() for r in caplog.records]
