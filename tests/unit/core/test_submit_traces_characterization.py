"""Characterization tests pinning ``WorkflowTraceManager.submit_traces``."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from traigent.core.workflow_trace_manager import WorkflowTraceManager

MOD = "traigent.core.workflow_trace_manager"
GRAPH = object()


def _span(run_id, trace="abcdefghijkl"):
    return SimpleNamespace(trace_id=trace, configuration_run_id=run_id)


def _resp(
    success=True,
    ingested=None,
    reported=True,
    dup=0,
    graph_id=None,
    error=None,
):
    return SimpleNamespace(
        success=success,
        spans_ingested=ingested,
        spans_ingested_reported=reported,
        skipped_duplicates=dup,
        graph_id=graph_id,
        error=error,
    )


class _Tracker:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ingest_traces_async(self, **kwargs):
        self.calls.append(
            (
                "graph" if kwargs["graph"] is GRAPH else kwargs["graph"],
                len(kwargs["spans"]),
                kwargs["trace_id"],
                kwargs["configuration_run_id"],
            )
        )
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


# name -> (spans, tracker responses, session_id, trials_executed, offline, tracker?)
CASES = {
    "no_tracker": ([_span("a")], [], "s", True, False, False),
    "no_spans": ([], [], "s", True, False, True),
    "no_spans_no_trials": ([], [], "s", False, False, True),
    "offline": ([_span("a")], [], "s", True, True, True),
    "mock_session": ([_span("a")], [], "mock_session_1", True, False, True),
    "mock_dash_session": ([_span("a")], [], "mock-session-1", True, False, True),
    "no_session_id": ([_span("a")], [_resp(ingested=1)], None, True, False, True),
    "ok_confirmed": (
        [_span("a"), _span("a")],
        [_resp(ingested=2, graph_id="g1")],
        "s",
        True,
        False,
        True,
    ),
    "ok_unconfirmed": (
        [_span("a"), _span("a")],
        [_resp(reported=False)],
        "s",
        True,
        False,
        True,
    ),
    "unconfirmed_bad_count": (
        [_span("a")],
        [_resp(ingested=True)],
        "s",
        True,
        False,
        True,
    ),
    "unconfirmed_negative": (
        [_span("a")],
        [_resp(ingested=-1)],
        "s",
        True,
        False,
        True,
    ),
    "reported_flag_not_true": (
        [_span("a")],
        [_resp(ingested=1, reported=1)],
        "s",
        True,
        False,
        True,
    ),
    "over_report": (
        [_span("a"), _span("a")],
        [_resp(ingested=5)],
        "s",
        True,
        False,
        True,
    ),
    "duplicates_only": (
        [_span("a")] * 4,
        [_resp(ingested=2, dup=2)],
        "s",
        True,
        False,
        True,
    ),
    "dropped": (
        [_span("a")] * 4,
        [_resp(ingested=1, dup=1)],
        "s",
        True,
        False,
        True,
    ),
    "dup_overreport_clamped": (
        [_span("a")] * 2,
        [_resp(ingested=0, dup=9)],
        "s",
        True,
        False,
        True,
    ),
    "dup_not_count": (
        [_span("a")] * 2,
        [_resp(ingested=1, dup="x")],
        "s",
        True,
        False,
        True,
    ),
    "two_groups_mixed": (
        [_span("a"), _span("b"), _span("b"), _span("c")],
        [
            _resp(ingested=1, graph_id="g"),
            _resp(ingested=1, dup=1),
            _resp(reported=False, graph_id="g2"),
        ],
        "s",
        True,
        False,
        True,
    ),
    "group_failure_auth": (
        [_span("a"), _span("b")],
        [_resp(success=False, error="Unauthorized"), _resp(ingested=1)],
        "s",
        True,
        False,
        True,
    ),
    "group_failure_other": (
        [_span("a")],
        [_resp(success=False, error="boom")],
        "s",
        True,
        False,
        True,
    ),
    "group_failure_no_error": (
        [_span("a")],
        [_resp(success=False, error=None)],
        "s",
        True,
        False,
        True,
    ),
    "ingest_raises": (
        [_span("a"), _span("b")],
        [RuntimeError("net down")],
        "s",
        True,
        False,
        True,
    ),
    "all_failed_nothing_logged": (
        [_span("a")],
        [_resp(success=False, error="forbidden")],
        "s",
        True,
        False,
        True,
    ),
}


def _snapshot(name, caplog):
    spans, responses, session_id, trials, offline, has_tracker = CASES[name]
    tracker = _Tracker(responses) if has_tracker else None
    mgr = WorkflowTraceManager(tracker, None, None, {}, None, "Opt", "oid")
    mgr._collected_spans = list(spans)
    caplog.clear()
    with (
        patch(f"{MOD}.is_backend_offline", return_value=offline),
        patch.object(mgr, "_create_optimization_workflow_graph", return_value=GRAPH),
        caplog.at_level(logging.DEBUG, logger=MOD),
    ):
        asyncio.run(mgr.submit_traces(session_id, trials_executed=trials))
    logs = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == MOD]
    return {
        "logs": repr(logs),
        "calls": repr(tracker.calls if tracker else None),
        "left": len(mgr._collected_spans),
    }


GOLDEN: dict = {
    "no_tracker": {"logs": "[]", "calls": "None", "left": 1},
    "no_spans": {
        "logs": "[('WARNING', 'Workflow tracing is enabled but no workflow spans were "
        "collected for this run; nothing will appear in the workflow traces view. "
        "Check that add_agent_span() is reached and that its SpanResult receipt "
        "reports accepted=True.')]",
        "calls": "[]",
        "left": 0,
    },
    "no_spans_no_trials": {
        "logs": "[('DEBUG', 'Workflow tracing is enabled but the run executed zero "
        "trials; no workflow spans were collected. Nothing will appear in "
        "the workflow traces view.')]",
        "calls": "[]",
        "left": 0,
    },
    "offline": {
        "logs": "[('DEBUG', 'Skipping workflow trace submission: backend is offline')]",
        "calls": "[]",
        "left": 0,
    },
    "mock_session": {
        "logs": "[('DEBUG', \"Skipping workflow trace submission: mock session "
        "'mock_session_1'\")]",
        "calls": "[]",
        "left": 0,
    },
    "mock_dash_session": {
        "logs": "[('DEBUG', \"Skipping workflow trace submission: mock session "
        "'mock-session-1'\")]",
        "calls": "[]",
        "left": 0,
    },
    "no_session_id": {
        "logs": "[('INFO', 'Submitted 1 workflow spans for trace abcdefgh...')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "ok_confirmed": {
        "logs": "[('INFO', 'Submitted 2 workflow spans for trace abcdefgh..., "
        "graph_id=g1')]",
        "calls": "[('graph', 2, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "ok_unconfirmed": {
        "logs": "[('INFO', 'Submitted 2 workflow spans for trace abcdefgh... (backend "
        "did not report how many were ingested for 2 of them; count "
        "unconfirmed)')]",
        "calls": "[('graph', 2, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "unconfirmed_bad_count": {
        "logs": "[('INFO', 'Submitted 1 workflow spans for trace abcdefgh... "
        "(backend did not report how many were ingested for 1 of them; "
        "count unconfirmed)')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "unconfirmed_negative": {
        "logs": "[('INFO', 'Submitted 1 workflow spans for trace abcdefgh... "
        "(backend did not report how many were ingested for 1 of them; "
        "count unconfirmed)')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "reported_flag_not_true": {
        "logs": "[('INFO', 'Submitted 1 workflow spans for trace abcdefgh... "
        "(backend did not report how many were ingested for 1 of them; "
        "count unconfirmed)')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "over_report": {
        "logs": "[('WARNING', 'Backend reported 5 workflow spans ingested for config_run "
        "a but only 2 were sent; counting 2.'), ('INFO', 'Submitted 2 workflow "
        "spans for trace abcdefgh...')]",
        "calls": "[('graph', 2, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "duplicates_only": {
        "logs": "[('WARNING', 'Backend ingested 2 of 4 workflow spans for trace "
        "abcdefgh...; 2 skipped as duplicate span ids (already stored, or "
        "repeated within the batch).')]",
        "calls": "[('graph', 4, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "dropped": {
        "logs": "[('WARNING', 'Backend ingested 1 of 4 workflow spans for trace abcdefgh... "
        "(1 skipped as duplicate span ids, 2 not ingested for an unreported reason); "
        "the workflow traces view will be missing those spans.')]",
        "calls": "[('graph', 4, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "dup_overreport_clamped": {
        "logs": "[('WARNING', 'Backend ingested 0 of 2 workflow spans for "
        "trace abcdefgh...; 2 skipped as duplicate span ids (already "
        "stored, or repeated within the batch).')]",
        "calls": "[('graph', 2, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "dup_not_count": {
        "logs": "[('WARNING', 'Backend ingested 1 of 2 workflow spans for trace "
        "abcdefgh... (0 skipped as duplicate span ids, 1 not ingested for an "
        "unreported reason); the workflow traces view will be missing those "
        "spans.')]",
        "calls": "[('graph', 2, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "two_groups_mixed": {
        "logs": "[('WARNING', 'Backend ingested 2 of 3 workflow spans for trace "
        "abcdefgh..., graph_id=g2; 1 skipped as duplicate span ids (already "
        "stored, or repeated within the batch); 1 more submitted with the "
        "ingested count unconfirmed.')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a'), (None, 2, 'abcdefghijkl', "
        "'b'), (None, 1, 'abcdefghijkl', 'c')]",
        "left": 0,
    },
    "group_failure_auth": {
        "logs": "[('DEBUG', 'Trace ingestion auth rejected for config_run a'), "
        "('INFO', 'Submitted 1 workflow spans for trace abcdefgh...')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a'), (None, 1, 'abcdefghijkl', 'b')]",
        "left": 0,
    },
    "group_failure_other": {
        "logs": "[('WARNING', 'Failed to submit spans for config_run a: boom')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "group_failure_no_error": {
        "logs": "[('WARNING', 'Failed to submit spans for config_run a: None')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "ingest_raises": {
        "logs": "[('WARNING', 'Workflow trace submission failed: net down')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
    "all_failed_nothing_logged": {
        "logs": "[('DEBUG', 'Trace ingestion auth rejected for config_run a')]",
        "calls": "[('graph', 1, 'abcdefghijkl', 'a')]",
        "left": 0,
    },
}


@pytest.mark.parametrize("name", list(CASES))
def test_submit_traces(name, caplog):
    assert _snapshot(name, caplog) == GOLDEN[name]
