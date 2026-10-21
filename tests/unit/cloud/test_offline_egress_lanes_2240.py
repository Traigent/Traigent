"""offline=True fails closed on trace export, langfuse and standalone clients (#2240)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent.core import tracing
from traigent.integrations.langfuse.client import LangfuseClient
from traigent.integrations.observability.workflow_traces import WorkflowTracesClient
from traigent.utils.env_config import is_backend_offline, offline_run_scope
from traigent.utils.error_handler import OfflineModeError

CANARY = "CANARY_OFFLINE_2240_x9z"


@pytest.fixture(autouse=True)
def _clear_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TRAIGENT_OFFLINE_MODE", "TRAIGENT_OFFLINE"):
        monkeypatch.delenv(name, raising=False)


def _wire_tripwire(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch every HTTP entry the lanes use; record anything that is called."""
    seen: list[str] = []

    def boom(*args: Any, **kwargs: Any) -> Any:
        seen.append(repr((args, kwargs)))
        raise AssertionError("network call attempted while offline")

    import requests

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(requests, "post", boom)
    import httpx

    monkeypatch.setattr(httpx.AsyncClient, "request", boom, raising=False)
    monkeypatch.setattr(httpx.AsyncClient, "post", boom, raising=False)
    monkeypatch.setattr(httpx.AsyncClient, "get", boom, raising=False)
    return seen


def test_scope_flips_and_restores_is_backend_offline() -> None:
    assert is_backend_offline() is False
    with offline_run_scope(True):
        assert is_backend_offline() is True
    assert is_backend_offline() is False
    with offline_run_scope(False):
        assert is_backend_offline() is False


def test_tracer_not_returned_while_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(tracing, "_initialize_tracer", lambda: sentinel)
    assert tracing.get_tracer() is sentinel
    with offline_run_scope(True):
        assert tracing.get_tracer() is None
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    assert tracing.get_tracer() is None


def test_langfuse_refuses_offline_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _wire_tripwire(monkeypatch)
    client = LangfuseClient(public_key=CANARY, secret_key=CANARY)
    with offline_run_scope(True):
        with pytest.raises(OfflineModeError):
            client.get_trace("t1")
        with pytest.raises(OfflineModeError):
            client.get_trace_metrics("t1")
        with pytest.raises(OfflineModeError):
            client.get_observations_for_trace("t1")
    assert seen == []
    assert not any(CANARY in s for s in seen)


@pytest.mark.asyncio
async def test_langfuse_async_refuses_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _wire_tripwire(monkeypatch)
    client = LangfuseClient(public_key=CANARY, secret_key=CANARY)
    with offline_run_scope(True):
        with pytest.raises(OfflineModeError):
            await client.get_trace_async("t1")
        with pytest.raises(OfflineModeError):
            await client.get_observations_for_trace_async("t1")
    assert seen == []


def test_workflow_trace_ingest_refused_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _wire_tripwire(monkeypatch)
    client = WorkflowTracesClient(backend_url="https://example.invalid")
    with offline_run_scope(True):
        resp = client.ingest_traces(graph=None, spans=None)
    assert resp.success is False
    assert seen == []


@pytest.mark.asyncio
async def test_standalone_analytics_client_refuses_offline() -> None:
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    client = BackendAnalyticsClient(backend_url="https://example.invalid")
    with offline_run_scope(True):
        with pytest.raises(OfflineModeError):
            client._get_client()


def test_decorator_offline_true_alone_sets_scope_during_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production path: offline=True with no env var turns every lane off."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
    (tmp_path / "d.jsonl").write_text(
        '{"input":{"q":"1"},"output":"aa"}\n{"input":{"q":"2"},"output":"aa"}\n'
    )
    observed: list[tuple[bool, Any]] = []

    @traigent.optimize(
        configuration_space={"v": ["aa", "bb"]},
        objectives=["accuracy"],
        offline=True,
        max_trials=1,
        algorithm="grid",
        eval_dataset="d.jsonl",
    )
    def agent(q: str) -> str:
        observed.append((is_backend_offline(), tracing.get_tracer()))
        return traigent.get_config()["v"]

    agent.optimize_sync()
    assert observed and all(o is True and t is None for o, t in observed)
    assert is_backend_offline() is False


def test_plugin_path_is_gated_at_sdk_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a traigent_tracing plugin installed, offline still gates every entry."""
    import importlib
    import sys
    import types
    from contextlib import contextmanager

    calls: list[str] = []
    fake = types.ModuleType("traigent_tracing")
    fake.__version__ = "0.1.1"  # accepted by the #1894 version guard

    @contextmanager
    def _span(*args: Any, **kwargs: Any):
        calls.append("span")
        yield "real-span"

    def _record(*args: Any, **kwargs: Any) -> str:
        calls.append("record")
        return "recorded"

    def _get_tracer() -> str:
        calls.append("tracer")
        return "real-tracer"

    for name, val in {
        "OTEL_AVAILABLE": True,
        "TRACING_ENABLED": True,
        "SecureIdGenerator": object,
        "clear_test_context": lambda: None,
        "get_test_context": dict,
        "set_test_context": lambda *a, **k: None,
        "example_evaluation_span": _span,
        "optimization_session_span": _span,
        "trial_span": _span,
        "record_example_result": _record,
        "record_optimization_complete": _record,
        "record_trial_result": _record,
        "get_tracer": _get_tracer,
    }.items():
        setattr(fake, name, val)
    monkeypatch.setitem(sys.modules, "traigent_tracing", fake)
    try:
        mod = importlib.reload(tracing)
        assert mod._PLUGIN_AVAILABLE is True
        # online: delegated to the plugin
        assert mod.get_tracer() == "real-tracer"
        with mod.trial_span("t", 1, {}) as sp:
            assert sp == "real-span"
        assert mod.record_trial_result(None, "ok") == "recorded"
        calls.clear()
        with offline_run_scope(True):
            assert mod.get_tracer() is None
            for cm in (
                mod.optimization_session_span("f"),
                mod.trial_span("t", 1, {}),
                mod.example_evaluation_span("e", 0, {}),
            ):
                with cm as sp:
                    assert sp is None
            assert mod.record_trial_result(None, "ok") is None
            assert mod.record_example_result(None, None) is None
            assert mod.record_optimization_complete(None, {}) is None
        assert calls == []
    finally:
        monkeypatch.delitem(sys.modules, "traigent_tracing", raising=False)
        importlib.reload(tracing)
