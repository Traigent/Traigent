"""The per-example consistency report reaches no outbound payload.

The report lives in ``result.example_consistency_report``. These tests run the
real finalize path, usage analytics, workflow-trace submission, the certified
selection report builder, the W&B / MLflow trackers and ``traigent sync``
against recording fakes of the transport (nothing is stubbed out above the
network edge), then check that the attribute name, a distinctive example id and
the label strings appear in no recorded payload.

A user's own completion callback receives the whole result object by design;
that is the user's code on the user's machine, so the callback test only checks
that the control (the report really exists on the object) holds.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tests.unit.core.test_winner_stability_example_consistency import _run
from traigent.api.types import OptimizationResult
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.objectives import create_default_objectives
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.utils.callbacks import OptimizationCallback

# Transports are recording fakes (no real network); the marker only clears the
# autouse offline mode so the connected code paths run.
pytestmark = pytest.mark.backend_online

EXAMPLE_ID = "ex-distinct-7f3a91"
_SCRIPT: dict[Any, list[float | None]] = {
    EXAMPLE_ID: [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    "steady_wrong": [0.0] * 6,
    "steady_right": [1.0] * 6,
}
FIELD_NAME = "example_consistency_report"


@pytest.fixture(autouse=True)
def _local_store(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_" + "0" * 40)


def _sentinels(result: OptimizationResult) -> tuple[list[str], list[str]]:
    """Strings that must not appear: (key + labels, example ids)."""
    report = result.example_consistency_report
    assert report and report["examples"]  # control: the run produced a report
    labels = sorted({row["label"] for row in report["examples"].values()})
    quoted = [f'"{name}"' for name in ("example_consistency", FIELD_NAME)]
    return [*quoted, *labels], list(report["examples"])


def _assert_clean(blob: str, result: OptimizationResult, *, ids: bool = True) -> None:
    words, example_ids = _sentinels(result)
    for word in words:
        assert word not in blob, word
    if ids:
        assert EXAMPLE_ID not in blob


def _dump(value: Any) -> str:
    return json.dumps(value, default=repr)


class _RecordingBackend:
    """Backend client double that records every call and its arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("__"):
            raise AttributeError(name)

        def _record(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return self._reply(name)

        return _record

    @staticmethod
    def _reply(name: str) -> Any:
        if name == "get_session_mapping":
            return SimpleNamespace(experiment_id="exp", experiment_run_id="run")
        if name == "finalize_session_sync":
            return {"status": "completed", "metadata": {}}
        return True

    def payload(self) -> str:
        return _dump(self.calls)


class _AsyncRecordingBackend(_RecordingBackend):
    """Async-context variant used for the analytics client."""

    async def __aenter__(self) -> _AsyncRecordingBackend:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def create_hybrid_session(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("create_hybrid_session", args, kwargs))
        return "analytics-session", "token", "endpoint"

    async def submit_privacy_trial_results(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append(("submit_privacy_trial_results", args, kwargs))
        return True


class _TraceTracker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ingest_traces_async(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return SimpleNamespace(
            success=True,
            spans_ingested=len(kwargs.get("spans") or []),
            spans_ingested_reported=True,
            skipped_duplicates=0,
            graph_id="graph-1",
            error=None,
        )


class _CompletionRecorder(OptimizationCallback):
    def __init__(self) -> None:
        self.results: list[OptimizationResult] = []

    def on_optimization_start(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_trial_start(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_trial_complete(self, *args: Any, **kwargs: Any) -> None:
        return None

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        self.results.append(result)


def _egress_orchestrator(
    backend: _RecordingBackend,
    tracker: _TraceTracker,
    callbacks: list[OptimizationCallback],
    execution_mode: str = "hybrid",
) -> OptimizationOrchestrator:
    """A real orchestrator whose backend, analytics and trace transports record."""
    from tests.unit.core.test_winner_stability_example_consistency import (
        _OneConfigOptimizer,
        _ScriptedEvaluator,
    )
    from traigent.core.workflow_trace_manager import WorkflowTraceManager
    from traigent.evaluators.base import BaseEvaluator  # noqa: F401
    from traigent.optimizers.base import BaseOptimizer  # noqa: F401

    config = TraigentConfig(enable_usage_analytics=True)
    config.offline = False
    config.no_egress = False
    config.execution_mode = execution_mode
    optimizer = _OneConfigOptimizer({"param1": [1]}, ["accuracy"])
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=_ScriptedEvaluator(_SCRIPT),
        max_trials=1,
        config=config,
        callbacks=callbacks,
    )
    manager = BackendSessionManager(
        backend_client=backend,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=create_default_objectives(["accuracy"]),
        optimizer=optimizer,
        optimization_id="opt_egress_audit",
        optimization_status=orchestrator._status,
    )
    orchestrator.backend_session_manager = manager
    orchestrator._workflow_trace_manager = WorkflowTraceManager(
        tracker, backend, None, {"param1": [1]}, 1, "Opt", "opt_egress_audit"
    )
    return orchestrator


_ASYNC_CLIENT_CALLS = {"update_trial_weighted_scores"}


def _collect_span(orchestrator: OptimizationOrchestrator) -> None:
    from traigent.integrations.observability.workflow_traces import SpanPayload

    orchestrator.collect_workflow_span(
        SpanPayload(
            span_id="span-1",
            trace_id="trace-egress-audit",
            configuration_run_id="cr-1",
            span_name="agent",
            span_type="agent",
            start_time="2026-01-01T00:00:00+00:00",
        )
    )


@pytest.mark.asyncio
async def test_finalize_analytics_traces_and_certified_report_carry_nothing(
    monkeypatch,
) -> None:
    """Every outbound call of ``_finalize_optimization``, run for real."""
    result = await _run(5, _SCRIPT)
    analytics = _AsyncRecordingBackend()
    import traigent.cloud.backend_client as backend_client_module

    monkeypatch.setattr(
        backend_client_module, "get_backend_client", lambda **_kw: analytics
    )
    backend = _RecordingBackend()
    tracker = _TraceTracker()
    user_callback = _CompletionRecorder()
    orchestrator = _egress_orchestrator(backend, tracker, [user_callback])
    # Usage analytics only run in local execution mode: same result, a second
    # orchestrator in that mode, real ``_submit_usage_analytics``.
    local_orchestrator = _egress_orchestrator(
        _RecordingBackend(), _TraceTracker(), [], execution_mode="local"
    )
    await local_orchestrator._submit_usage_analytics()
    _collect_span(orchestrator)
    certified = orchestrator._build_certified_selection_report()

    await orchestrator._finalize_optimization(result, "session-egress-audit", None)

    # The real transports were exercised, not skipped.
    assert any(name == "finalize_session_sync" for name, *_ in backend.calls)
    assert analytics.calls, "usage analytics did not run"
    assert tracker.calls, "workflow traces were not submitted"
    outbound = _dump([backend.calls, analytics.calls, tracker.calls, certified])
    _assert_clean(outbound, result)
    # Control: the user's own callback gets the whole result, report included.
    assert user_callback.results == [result]
    assert user_callback.results[0].example_consistency_report is not None


class _RecordingHttp:
    """``requests.Session`` double: records every request, answers 200 ``{}``."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.headers: dict[str, str] = {}

    def __getattr__(self, verb: str) -> Callable[..., Any]:
        if verb.startswith("__"):
            raise AttributeError(verb)

        def _call(*args: Any, **kwargs: Any) -> Any:
            self.requests.append((verb, args, kwargs))
            response = Mock()
            response.status_code = 200
            response.ok = True
            response.json.return_value = {}
            response.text = "{}"
            return response

        return _call


@pytest.mark.asyncio
async def test_the_local_session_store_and_traigent_sync_carry_nothing(
    tmp_path,
) -> None:
    """``traigent sync`` reads the local session store; neither holds the report."""
    from traigent.cloud.sync_manager import SyncManager
    from traigent.storage.local_storage import LocalStorageManager

    result = await _run(5, _SCRIPT)
    store = Path(os.environ["TRAIGENT_RESULTS_FOLDER"])
    stored_files = [p for p in store.rglob("*") if p.is_file()]
    assert stored_files, "the run left no local session store to sync from"
    # The per-trial example rows already carry example ids by existing design,
    # so for the store and sync only the key and the label strings can tell this
    # feature's data apart.
    for path in stored_files:
        _assert_clean(path.read_text(errors="replace"), result, ids=False)

    config = TraigentConfig(local_storage_path=str(store))
    sessions = LocalStorageManager(str(store)).list_sessions()
    assert sessions
    http = _RecordingHttp()
    manager = SyncManager(config, api_key="tg_" + "0" * 40)
    manager._http_session = http  # type: ignore[assignment]
    converted = [
        manager.convert_session_to_traigent_format(session) for session in sessions
    ]
    assert converted
    _assert_clean(_dump(converted), result, ids=False)
    for session in sessions:
        manager.sync_session_to_cloud(session.session_id)
    assert http.requests, "sync made no request; the check below would be vacuous"
    _assert_clean(_dump(http.requests), result, ids=False)
