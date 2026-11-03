"""optimization_only: every SDK-owned trace delivery path must validate the key (once,
bounded, fail closed) BEFORE sending trace bytes to Traigent, even when the latch is unset
(fresh process). Each test starts from a cleared latch and preflight cache."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import traigent.observability.otel as otel
from traigent.utils import optimization_only_egress as oo

from .harness import capture_egress
from .scenario import BACKEND_URL, FAKE_API_KEY
from .test_optimization_only_egress import _env

pytestmark = pytest.mark.backend_online

OTLP_BASE = f"{BACKEND_URL}/api/v1beta/observability/otlp"
RESTRICTED = {"valid": True, "optimization_only": True}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _env(monkeypatch, tmp_path)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    monkeypatch.delenv("TRAIGENT_DISABLE_TELEMETRY", raising=False)
    oo.reset_latch()
    yield
    otel.shutdown()
    oo.reset_latch()


def _is_trace(r: Any) -> bool:
    return any(s in r.path for s in ("/otlp", "/v1/traces", "/ingest"))


def _assert_no_trace_egress(cap: Any, *, validated: bool = True) -> None:
    assert [r for r in cap.requests if _is_trace(r)] == []
    assert cap.guard_violations == []
    if validated:
        assert any(r.path.endswith("/keys/validate") for r in cap.requests)


def _emit(endpoint: str | None = None) -> None:
    handle = otel.init(
        api_key=FAKE_API_KEY,
        endpoint=endpoint,
        schedule_delay_s=0.05,
        exit_flush=False,
    )
    with handle.tracer.start_as_current_span("x"):
        pass
    otel.flush(2)
    otel.shutdown()


def test_restricted_key_never_posts_spans_in_fresh_process() -> None:
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        _emit()
    _assert_no_trace_egress(cap)


def test_valid_false_fails_closed() -> None:
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": False}
        _emit()
    _assert_no_trace_egress(cap)


def test_unreachable_validation_fails_closed() -> None:
    with capture_egress() as cap:
        cap.backend.inject("POST", r"/keys/validate$", ["connection"] * 5)
        _emit()
    _assert_no_trace_egress(cap)


@pytest.mark.parametrize(
    "endpoint",
    [
        OTLP_BASE,
        "https://backend.example.test:443/api/v1beta/observability/otlp",
        "https://backend.example.test./api/v1beta/observability/otlp",
        "https://BACKEND.example.test/api/v1beta/observability/otlp/",
    ],
)
def test_endpoint_spellings_cannot_skip_preflight(endpoint: str) -> None:
    assert oo.is_traigent_trace_endpoint(endpoint)
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        _emit(endpoint)
    _assert_no_trace_egress(cap)


def test_staging_traigent_host_is_validated_at_its_own_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = "https://otlp.staging.traigent.ai"
    seen = _patch_validation(monkeypatch, restricted_for={host})
    assert oo.is_traigent_trace_endpoint(f"{host}:8443/v1")
    assert not oo.ensure_trace_egress_allowed(FAKE_API_KEY, f"{host}/v1")
    assert seen == [host]


def test_third_party_collector_is_not_preflighted() -> None:
    assert not oo.is_traigent_trace_endpoint("https://collector.example.org:4318")
    with capture_egress() as cap:
        assert oo.ensure_trace_egress_allowed(
            FAKE_API_KEY, "https://collector.example.org"
        )
    assert not cap.requests


def test_latch_failure_keeps_auth_manager_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: Any, **_k: Any) -> bool:
        raise RuntimeError("latch broken")

    monkeypatch.setattr(oo, "note_key_validation", boom)
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        _emit()
    _assert_no_trace_egress(cap)
    scope = oo.credential_scope(BACKEND_URL, {"X-API-Key": FAKE_API_KEY})
    assert not oo.is_latched(
        scope
    )  # the latch really failed; the manager still refused


def test_stalled_validation_returns_within_deadline_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud.auth import AuthManager

    release = threading.Event()

    async def stall(self: Any, api_key: str) -> None:
        release.wait(30)  # blocks only the daemon worker; released at teardown

    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", stall)
    monkeypatch.setattr(oo, "TRACE_PREFLIGHT_DEADLINE_S", 0.5)
    try:
        with capture_egress() as cap:
            started = time.monotonic()
            _emit()
            elapsed = time.monotonic() - started
        assert elapsed < 5
        _assert_no_trace_egress(cap, validated=False)
    finally:
        release.set()


def test_normal_key_still_exports_and_validates_once() -> None:
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        _emit()
        _emit()  # re-init reuses the cached decision
    assert [r for r in cap.requests if "/otlp" in r.path]
    assert len([r for r in cap.requests if r.path.endswith("/keys/validate")]) == 1
    assert cap.guard_violations == []


# ---- the other SDK-owned trace paths, each starting unlatched -------------------------


def test_legacy_observability_client_sends_nothing() -> None:
    from traigent.observability.client import ObservabilityClient
    from traigent.observability.config import ObservabilityConfig

    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        client = ObservabilityClient(
            ObservabilityConfig(backend_origin=BACKEND_URL, api_key=FAKE_API_KEY)
        )
        assert client._post_batch_sync([{"trace_id": "t"}]) is None
    _assert_no_trace_egress(cap)


def test_workflow_traces_client_sends_nothing() -> None:
    from traigent.integrations.observability.workflow_traces import (
        WorkflowGraphPayload,
        WorkflowTracesClient,
    )

    graph = WorkflowGraphPayload(experiment_id="e", nodes=[], edges=[])
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        client = WorkflowTracesClient(BACKEND_URL, FAKE_API_KEY)
        assert client.ingest_traces(graph=graph).error == "optimization_only"
        oo.reset_latch()  # the async path must also validate from scratch
        assert (
            asyncio.run(client.ingest_traces_async(graph=graph)).error
            == "optimization_only"
        )
    _assert_no_trace_egress(cap)


def test_workflow_exporter_factory_refuses() -> None:
    from traigent.integrations.observability import workflow_traces as wt

    if not wt.OTEL_AVAILABLE:
        pytest.skip("opentelemetry not installed")
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        assert wt.setup_workflow_tracing(BACKEND_URL, FAKE_API_KEY) is None
    _assert_no_trace_egress(cap)


def _live_core_tracer(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Embedded core tracing with stub OTel pieces (the HTTP exporter is optional)."""
    from unittest.mock import MagicMock

    from traigent.core import tracing

    if getattr(tracing, "_PLUGIN_AVAILABLE", False):
        pytest.skip("traigent-tracing plugin installed")

    class _Provider:
        shut = False

        def __init__(self, **_k: Any) -> None:
            pass

        def add_span_processor(self, _p: Any) -> None:
            pass

        def shutdown(self) -> None:
            _Provider.shut = True

    for name, value in {
        "OTEL_AVAILABLE": True,
        "TRACING_ENABLED": True,
        "TracerProvider": _Provider,
        "Resource": MagicMock(),
        "OTLPHttpSpanExporter": MagicMock(),
        "BatchSpanProcessor": MagicMock(),
        "SecureIdGenerator": MagicMock(),
        "trace": MagicMock(),
        "_initialized": False,
        "_tracer": None,
        "_provider": None,
    }.items():
        monkeypatch.setattr(tracing, name, value, raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"{BACKEND_URL}/otlp")
    return tracing


def test_core_tracing_env_exporter_not_built(monkeypatch: pytest.MonkeyPatch) -> None:
    tracing = _live_core_tracer(monkeypatch)
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        assert tracing.get_tracer() is None
    _assert_no_trace_egress(cap)


# ---- round-2 review fixes -------------------------------------------------------------


def _patch_validation(
    monkeypatch: pytest.MonkeyPatch, answer: Any = None, restricted_for: Any = ()
) -> list[str]:
    """Replace the network call; returns the list of backend bases it was asked about."""
    from traigent.cloud.auth import AuthManager

    seen: list[str] = []

    async def fake(self: Any, api_key: str) -> Any:
        base = self.config.backend_base_url
        seen.append(base)
        if base in restricted_for:
            self._server_optimization_only = True
        return answer

    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", fake)
    return seen


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://portal。traigent。ai/otlp",
        "https://portal．traigent．ai/otlp",
        "https://portal.traigent.ai。/otlp",
        "https://PORTAL｡TRAIGENT.ai./otlp",
    ],
)
def test_unicode_dot_hostnames_are_still_traigent(endpoint: str) -> None:
    assert oo.is_traigent_trace_endpoint(endpoint)


def test_unicode_dot_spelling_of_backend_host_is_preflighted() -> None:
    endpoint = "https://backend。example。test/api/v1beta/observability/otlp"
    assert oo.is_traigent_trace_endpoint(endpoint)
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        _emit(endpoint)
    _assert_no_trace_egress(cap)


def test_config_lookup_failure_keeps_explicit_collector_third_party(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.config.backend_config import BackendConfig

    def boom(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("no backend configured")

    monkeypatch.setattr(BackendConfig, "get_backend_url", classmethod(boom))
    monkeypatch.setattr(BackendConfig, "get_cloud_backend_url", classmethod(boom))
    assert not oo.is_traigent_trace_endpoint("https://collector.example.org:4318")
    assert oo.is_traigent_trace_endpoint("https://api.traigent.ai/x")
    assert oo.ensure_trace_egress_allowed(
        FAKE_API_KEY, "https://collector.example.org:4318"
    )


def test_backend_host_compared_with_port(monkeypatch: pytest.MonkeyPatch) -> None:
    from traigent.config.backend_config import BackendConfig

    monkeypatch.setattr(
        BackendConfig, "get_backend_url", classmethod(lambda cls: BACKEND_URL)
    )
    monkeypatch.setattr(
        BackendConfig, "get_cloud_backend_url", classmethod(lambda cls: BACKEND_URL)
    )
    assert oo.is_traigent_trace_endpoint("https://backend.example.test:443/x")
    # same host, different port: a customer collector, not our backend
    assert not oo.is_traigent_trace_endpoint("https://backend.example.test:8443/x")
    # ports are irrelevant for traigent.ai hosts
    assert oo.is_traigent_trace_endpoint("https://otlp.traigent.ai:8443/x")


def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(oo, "_trace_clock", lambda: now[0])
    return now


def test_failed_validation_retries_after_ttl_deny_deny_revalidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch_validation(monkeypatch, answer="backend validation timed out")
    now = _fake_clock(monkeypatch)
    endpoint = "https://api.traigent.ai/x"
    assert oo.ensure_trace_egress_allowed(FAKE_API_KEY, endpoint) is False
    now[0] += 59
    assert oo.ensure_trace_egress_allowed(FAKE_API_KEY, endpoint) is False
    assert len(seen) == 1  # served from the failure cache
    now[0] += 2  # t = 61 s
    assert oo.ensure_trace_egress_allowed(FAKE_API_KEY, endpoint) is False
    assert len(seen) == 2  # revalidated after the 60 s window


def test_recovery_after_failure_window_allows_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(["backend validation timed out", None])
    from traigent.cloud.auth import AuthManager

    async def fake(self: Any, api_key: str) -> Any:
        return next(answers)

    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", fake)
    now = _fake_clock(monkeypatch)
    endpoint = "https://api.traigent.ai/x"
    assert not oo.ensure_trace_egress_allowed(FAKE_API_KEY, endpoint)
    now[0] += 61
    assert oo.ensure_trace_egress_allowed(FAKE_API_KEY, endpoint)


def test_validation_runs_against_the_trace_destination_not_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = "https://a.traigent.ai", "https://b.traigent.ai"
    seen = _patch_validation(monkeypatch, restricted_for={b})
    assert oo.ensure_trace_egress_allowed(FAKE_API_KEY, f"{a}/otlp")
    assert not oo.ensure_trace_egress_allowed(FAKE_API_KEY, f"{b}/otlp")
    assert seen == [a, b]
    assert oo.ensure_trace_egress_allowed(FAKE_API_KEY, f"{a}/otlp")
    assert seen == [a, b]  # cached per (key, validation origin)


def test_switching_backend_config_revalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.config.backend_config import BackendConfig

    a, b = "https://a.traigent.ai", "https://b.traigent.ai"
    seen = _patch_validation(monkeypatch, restricted_for={b})
    current = [a]
    monkeypatch.setattr(
        BackendConfig, "get_cloud_backend_url", classmethod(lambda cls: current[0])
    )
    monkeypatch.setattr(
        BackendConfig, "get_backend_url", classmethod(lambda cls: current[0])
    )
    assert oo.ensure_trace_egress_allowed(FAKE_API_KEY, None)
    current[0] = b
    assert not oo.ensure_trace_egress_allowed(FAKE_API_KEY, None)
    assert seen == [a, b]


def test_stalled_validation_never_accumulates_validators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.cloud.auth import AuthManager

    release = threading.Event()
    calls: list[int] = []

    async def stall(self: Any, api_key: str) -> None:
        calls.append(1)
        await asyncio.to_thread(release.wait, 30)

    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", stall)
    monkeypatch.setattr(oo, "TRACE_PREFLIGHT_DEADLINE_S", 0.2)
    now = _fake_clock(monkeypatch)
    endpoint = "https://api.traigent.ai/x"

    def live() -> int:
        return len(
            [
                t
                for t in threading.enumerate()
                if t.name == "traigent-trace-preflight" and t.is_alive()
            ]
        )

    try:
        for _ in range(4):
            assert not oo.ensure_trace_egress_allowed(FAKE_API_KEY, endpoint)
            now[0] += 100  # past the failure window and the deadline
            assert live() <= 1
        assert len(calls) == 1  # later callers were denied, not given a new worker
    finally:
        release.set()


def test_stalled_resolver_does_not_keep_the_process_alive(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = tmp_path / "stall.py"
    script.write_text(
        "import asyncio, time\n"
        "from traigent.cloud.auth import AuthManager\n"
        "from traigent.utils import optimization_only_egress as oo\n"
        "async def stall(self, api_key):\n"
        "    await asyncio.to_thread(time.sleep, 600)\n"
        "AuthManager._validate_api_key_with_backend = stall\n"
        "oo.TRACE_PREFLIGHT_DEADLINE_S = 0.3\n"
        "print(oo.ensure_trace_egress_allowed('k', 'https://api.traigent.ai/x'))\n"
    )
    started = time.monotonic()
    out = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(Path(oo.__file__).resolve().parents[2]),
        env={
            **os.environ,
            "PYTHONPATH": str(Path(oo.__file__).resolve().parents[2])
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.stdout.strip().endswith("False"), out.stderr[-500:]
    assert time.monotonic() - started < 40


def _legacy_client(**cfg: Any) -> Any:
    from traigent.observability.client import ObservabilityClient
    from traigent.observability.config import ObservabilityConfig

    return ObservabilityClient(ObservabilityConfig(backend_origin=BACKEND_URL, **cfg))


def test_api_key_in_extra_headers_is_validated_and_exports() -> None:
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        client = _legacy_client(extra_headers={"X-API-Key": FAKE_API_KEY})
        client._post_batch_sync([{"trace_id": "t"}])
    assert any(r.path.endswith("/keys/validate") for r in cap.requests)
    assert any(r.path.endswith("/ingest") for r in cap.requests)


def test_extra_headers_restricted_key_sends_nothing() -> None:
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        client = _legacy_client(extra_headers={"x-api-key": FAKE_API_KEY})
        client._post_batch_sync([{"trace_id": "t"}])
    _assert_no_trace_egress(cap)


def test_bearer_only_config_fails_closed_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with capture_egress() as cap, caplog.at_level("WARNING"):
        client = _legacy_client(api_key=None, jwt_token="header.payload.sig")
        assert client._post_batch_sync([{"trace_id": "t"}]) is None
    assert [r for r in cap.requests if _is_trace(r)] == []
    assert "Bearer" in caplog.text and "cannot be validated" in caplog.text


def test_env_flag_after_construction_blocks_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_validation(monkeypatch)
    endpoint = "https://api.traigent.ai/x"
    assert oo.ensure_trace_egress_allowed(FAKE_API_KEY, endpoint)
    monkeypatch.setenv(oo.ENV_VAR, "true")
    assert not oo.ensure_trace_egress_allowed(FAKE_API_KEY, endpoint)


# ---- core tracing: the gate also applies at send time --------------------------------


def test_core_tracer_torn_down_when_env_flag_set_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _live_core_tracer(monkeypatch)
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        assert tracing.get_tracer() is not None
        monkeypatch.setenv(oo.ENV_VAR, "true")
        assert tracing.get_tracer() is None
        assert tracing._tracer is None
        assert tracing._provider is None
    assert cap.guard_violations == []


def test_core_tracer_torn_down_when_credential_latches_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.config.backend_config import BackendConfig

    tracing = _live_core_tracer(monkeypatch)
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        assert tracing.get_tracer() is not None
        key = BackendConfig.get_api_key()
        oo.latch_on(oo.credential_scope(BACKEND_URL, {"X-API-Key": key}))
        assert tracing.get_tracer() is None
    assert cap.guard_violations == []


def test_installed_plugin_wrappers_respect_optimization_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import sys
    import types

    from traigent.core import tracing

    names = [
        "OTEL_AVAILABLE",
        "TRACING_ENABLED",
        "SecureIdGenerator",
        "clear_test_context",
        "example_evaluation_span",
        "get_test_context",
        "get_tracer",
        "optimization_session_span",
        "record_example_result",
        "record_optimization_complete",
        "record_trial_result",
        "set_test_context",
        "trial_span",
    ]
    fake = types.ModuleType("traigent_tracing")
    fake.__version__ = "9.9.9"  # type: ignore[attr-defined]
    for n in names:
        setattr(fake, n, lambda *a, **k: "PLUGIN")
    monkeypatch.setitem(sys.modules, "traigent_tracing", fake)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    try:
        importlib.reload(tracing)
        assert tracing._PLUGIN_AVAILABLE
        assert tracing.get_tracer() == "PLUGIN"
        monkeypatch.setenv(oo.ENV_VAR, "true")
        assert tracing.get_tracer() is None
        assert tracing.record_trial_result() is None
    finally:
        monkeypatch.delitem(sys.modules, "traigent_tracing", raising=False)
        importlib.reload(tracing)
