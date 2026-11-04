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

        def get_tracer(self, *_a: Any, **_k: Any) -> Any:
            return MagicMock()

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


def _mark_unrestricted(manager: Any, result: Any) -> None:
    """Model the backend's explicit ``optimization_only: false`` for a fake validator."""
    if result is None and not manager._server_optimization_only:
        manager._key_validation_unrestricted = True


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
        result = answer
        _mark_unrestricted(self, result)
        return result

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
        result = next(answers)
        _mark_unrestricted(self, result)
        return result

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
    assert (
        "Bearer" in caplog.text
        and "cannot be checked for optimization_only" in caplog.text
    )


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


# ---- round-3 review fixes (real OpenTelemetry SDK, no stubbed provider) -----------------


def _real_core_tracer(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Embedded core tracing with the REAL SDK provider/BatchSpanProcessor/OTLP exporter."""
    from traigent.core import tracing

    if getattr(tracing, "_PLUGIN_AVAILABLE", False) or not tracing.OTEL_AVAILABLE:
        pytest.skip("embedded tracing with real OpenTelemetry SDK not available")
    for name, value in {
        "TRACING_ENABLED": True,
        "_initialized": False,
        "_tracer": None,
        "_provider": None,
    }.items():
        monkeypatch.setattr(tracing, name, value, raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"{BACKEND_URL}/otlp")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", raising=False)
    monkeypatch.setattr(
        "traigent.config.backend_config.BackendConfig.get_api_key",
        classmethod(lambda cls: FAKE_API_KEY),
    )
    return tracing


def _teardown_provider(tracing: Any) -> None:
    provider = getattr(tracing, "_provider", None)
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:  # noqa: BLE001
            pass


def _trace_posts(cap: Any) -> list[Any]:
    return [r for r in cap.requests if r.path.endswith("/v1/traces")]


def test_already_returned_tracer_does_not_export_after_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _real_core_tracer(monkeypatch)
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        tracer = tracing.get_tracer()
        assert tracer is not None
        oo.reset_trace_gate()
        cap.backend.validate_response = dict(RESTRICTED)
        with tracer.start_as_current_span("late"):
            pass
        tracing._provider.force_flush(2000)
        _teardown_provider(tracing)
    assert _trace_posts(cap) == []
    assert cap.guard_violations == []


def test_unrestricted_core_tracer_still_exports_positive_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _real_core_tracer(monkeypatch)
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        tracer = tracing.get_tracer()
        with tracer.start_as_current_span("ok"):
            pass
        tracing._provider.force_flush(2000)
        _teardown_provider(tracing)
    assert len(_trace_posts(cap)) >= 1


def test_restriction_discards_queued_spans_instead_of_draining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _real_core_tracer(monkeypatch)
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        tracer = tracing.get_tracer()
        assert tracer is not None
        with tracer.start_as_current_span("queued"):
            pass  # sits in the BatchSpanProcessor queue
        oo.reset_trace_gate()
        cap.backend.validate_response = dict(RESTRICTED)
        assert tracing.get_tracer() is None  # tears the provider down (shutdown drains)
    assert _trace_posts(cap) == []
    assert cap.guard_violations == []


def test_otel_headers_key_is_the_validated_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _real_core_tracer(monkeypatch)
    other = "tk_other_restricted_probe_key"
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", f"x-api-key={other},foo=bar")
    validated: list[str] = []
    from traigent.cloud.auth import AuthManager

    async def fake(self: Any, api_key: str) -> Any:
        validated.append(api_key)
        if api_key == other:
            self._server_optimization_only = True
        result = None
        _mark_unrestricted(self, result)
        return result

    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", fake)
    with capture_egress() as cap:
        assert tracing.get_tracer() is None
    assert other in validated and FAKE_API_KEY not in validated
    assert _trace_posts(cap) == []


def test_traces_headers_take_precedence_over_general_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _real_core_tracer(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=general")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "X-API-Key=specific")
    assert tracing._effective_trace_api_key() == "specific"
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
    assert tracing._effective_trace_api_key() == "general"
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS")
    assert tracing._effective_trace_api_key() == FAKE_API_KEY


def test_transient_denial_does_not_permanently_disable_core_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _real_core_tracer(monkeypatch)
    from traigent.cloud.auth import AuthManager

    answers = iter(["backend validation timed out", None])

    async def fake(self: Any, api_key: str) -> Any:
        result = next(answers)
        _mark_unrestricted(self, result)
        return result

    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", fake)
    now = _fake_clock(monkeypatch)
    try:
        assert tracing.get_tracer() is None
        now[0] += 61
        assert tracing.get_tracer() is not None  # recovered after the failure TTL
    finally:
        _teardown_provider(tracing)


def test_mixed_bearer_and_api_key_is_refused(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The backend authenticates Bearer first and the SDK cannot validate its tenant
    restriction, so any Bearer (even with an X-API-Key) is refused without validating."""
    seen = _patch_validation(monkeypatch)
    monkeypatch.setattr(oo, "_warned_bearer_only", False)
    with caplog.at_level("WARNING"):
        assert not oo.ensure_trace_egress_allowed(
            None,
            "https://api.traigent.ai/x",
            headers={"X-API-Key": FAKE_API_KEY, "Authorization": "Bearer a.b.c"},
        )
    assert seen == []
    assert "with a Bearer credential cannot be checked" in caplog.text


def test_api_key_only_headers_still_export(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch_validation(monkeypatch)
    assert oo.ensure_trace_egress_allowed(
        None, "https://api.traigent.ai/x", headers={"X-API-Key": FAKE_API_KEY}
    )
    assert len(seen) == 1


# ---- round-4 review fixes: frozen credential identity + request-level gate ---------------


def _per_key_validation(
    monkeypatch: pytest.MonkeyPatch, restricted: set[str]
) -> list[str]:
    """Fake backend validation: keys in ``restricted`` latch optimization_only."""
    from traigent.cloud.auth import AuthManager

    validated: list[str] = []

    async def fake(self: Any, api_key: str) -> Any:
        validated.append(api_key)
        if api_key in restricted:
            self._server_optimization_only = True
        result = None
        _mark_unrestricted(self, result)
        return result

    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", fake)
    return validated


def test_duplicate_api_key_headers_last_wins_and_restricted_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _real_core_tracer(monkeypatch)
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_HEADERS",
        "x-api-key=tk_probe_allowed,x-api-key=tk_probe_restricted",
    )
    validated = _per_key_validation(monkeypatch, {"tk_probe_restricted"})
    with capture_egress() as cap:
        tracer = tracing.get_tracer()
        if tracer is not None:
            with tracer.start_as_current_span("dup"):
                pass
            tracing._provider.force_flush(2000)
        _teardown_provider(tracing)
    assert "tk_probe_restricted" in validated and "tk_probe_allowed" not in validated
    assert tracer is None
    assert _trace_posts(cap) == []
    assert cap.guard_violations == []


def test_env_change_after_init_cannot_swap_the_gated_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracing = _real_core_tracer(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=tk_frozen_a")
    restricted: set[str] = set()
    validated = _per_key_validation(monkeypatch, restricted)
    with capture_egress() as cap:
        tracer = tracing.get_tracer()
        assert tracer is not None
        # credential A latches; the environment then points at an allowed key B
        restricted.add("tk_frozen_a")
        oo.reset_trace_gate()
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=tk_other_b")
        with tracer.start_as_current_span("late"):
            pass
        tracing._provider.force_flush(2000)
        _teardown_provider(tracing)
    assert "tk_other_b" not in validated
    assert _trace_posts(cap) == []
    assert cap.guard_violations == []


def test_retry_inside_export_is_not_sent_after_credential_latches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests
    from requests.adapters import HTTPAdapter

    tracing = _real_core_tracer(monkeypatch)
    restricted: set[str] = set()
    _per_key_validation(monkeypatch, restricted)
    sent: list[str] = []

    def fake_send(self: Any, request: Any, **kwargs: Any) -> Any:
        sent.append(request.url)
        # first attempt: transient 503, and the credential latches meanwhile
        restricted.add(FAKE_API_KEY)
        oo.reset_trace_gate()
        resp = requests.Response()
        resp.status_code = 503
        resp.url = request.url
        resp._content = b""
        return resp

    with capture_egress():
        tracer = tracing.get_tracer()
        assert tracer is not None
        monkeypatch.setattr(HTTPAdapter, "send", fake_send)
        with tracer.start_as_current_span("retry"):
            pass
        tracing._provider.force_flush(10000)
        _teardown_provider(tracing)
    assert len(sent) == 1  # the internal retry was refused at request level


def test_env_headers_set_between_capture_and_exporter_construction_send_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OTel re-reads OTEL_EXPORTER_OTLP_HEADERS when handed empty headers; the request
    then carries a key the frozen/validated one is not. The prepared request is gated."""
    tracing = _real_core_tracer(monkeypatch)
    restricted: set[str] = {"tk_racy_restricted"}
    _per_key_validation(monkeypatch, restricted)
    real_resolve = tracing._resolve_trace_headers

    def resolve_then_race() -> dict[str, str]:
        out = real_resolve()  # empty: the key will come from BackendConfig
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=tk_racy_restricted")
        return out

    monkeypatch.setattr(tracing, "_resolve_trace_headers", resolve_then_race)
    with capture_egress() as cap:
        tracer = tracing.get_tracer()
        if tracer is not None:
            with tracer.start_as_current_span("race"):
                pass
            tracing._provider.force_flush(2000)
        _teardown_provider(tracing)
    assert _trace_posts(cap) == []
    assert cap.guard_violations == []


def test_redirect_from_third_party_collector_to_traigent_is_not_followed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older OTel exporters post with the default allow_redirects=True."""
    import io

    import requests
    from requests.adapters import HTTPAdapter

    tracing = _real_core_tracer(monkeypatch)
    collector = "https://collector.example.org"
    seen: list[str] = []

    def fake_send(self: Any, request: Any, **kwargs: Any) -> Any:
        seen.append(request.url)
        resp = requests.Response()
        resp.status_code = 307
        resp.url = request.url
        resp.headers["Location"] = (
            f"{BACKEND_URL}/api/v1beta/observability/otlp/v1/traces"
        )
        resp.request = request
        resp.raw = io.BytesIO(b"")
        return resp

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    session = tracing._GatedSession(collector, "tk_third_party")
    resp = session.post(
        f"{collector}/v1/traces", data=b"x", headers={"X-API-Key": "tk_third_party"}
    )
    assert resp.status_code == 307
    assert seen == [f"{collector}/v1/traces"], seen


# ---- round-6: validate-then-reread race in the sibling delivery paths -------------------

KEY_A = "tk_race_allowed_a"
KEY_R = "tk_race_restricted_r"


def _race_on_validation(monkeypatch: pytest.MonkeyPatch, switch: Any) -> list[str]:
    """Validating allowed key A swaps the credential source to restricted key R."""
    from traigent.cloud.auth import AuthManager

    from . import harness

    # keep X-API-Key on captured requests so the test can see which key was sent
    monkeypatch.setattr(
        harness, "SECRET_HEADERS", harness.SECRET_HEADERS - {"x-api-key"}
    )
    validated: list[str] = []

    async def fake(self: Any, api_key: str) -> Any:
        validated.append(api_key)
        if api_key == KEY_A:
            switch()
        elif api_key == KEY_R:
            self._server_optimization_only = True
        result = None
        _mark_unrestricted(self, result)
        return result

    monkeypatch.setattr(AuthManager, "_validate_api_key_with_backend", fake)
    return validated


def _carries_key(cap: Any, key: str) -> list[Any]:
    return [
        r for r in cap.requests if _is_trace(r) and any(v == key for _k, v in r.headers)
    ]


def test_otel_api_credential_change_during_validation_never_sends_new_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.config.backend_config import BackendConfig

    current = {"key": KEY_A}
    monkeypatch.setattr(
        BackendConfig, "get_api_key", classmethod(lambda cls: current["key"])
    )
    _race_on_validation(monkeypatch, lambda: current.update(key=KEY_R))
    with capture_egress() as cap:
        handle = otel.init(endpoint=OTLP_BASE, schedule_delay_s=0.05, exit_flush=False)
        with handle.tracer.start_as_current_span("x"):
            pass
        otel.flush(2)
        otel.shutdown()
    assert _carries_key(cap, KEY_R) == []
    assert cap.guard_violations == []


def test_legacy_client_credential_change_during_validation_never_sends_new_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with capture_egress() as cap:
        client = _legacy_client(api_key=KEY_A)

        def switch() -> None:
            client.config.api_key = KEY_R

        _race_on_validation(monkeypatch, switch)
        client._post_batch_sync([{"trace_id": "t"}])
    assert _carries_key(cap, KEY_R) == []
    assert cap.guard_violations == []


def test_workflow_client_credential_change_during_validation_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.integrations.observability.workflow_traces import (
        WorkflowGraphPayload,
        WorkflowTracesClient,
    )

    graph = WorkflowGraphPayload(experiment_id="e", nodes=[], edges=[])
    with capture_egress() as cap:
        client = WorkflowTracesClient(BACKEND_URL, KEY_A)

        def switch() -> None:
            client.auth_token = KEY_R

        _race_on_validation(monkeypatch, switch)
        client.ingest_traces(graph=graph)
        client.auth_token = KEY_A
        oo.reset_latch()
        oo.reset_trace_gate()
        asyncio.run(client.ingest_traces_async(graph=graph))
    assert _carries_key(cap, KEY_R) == []
    assert cap.guard_violations == []


# ---- round-7 review fixes: empty snapshot key, bundled plugin exporter -------------------

R_KEY = "tk_restricted_latched_key"


def _carries_key(cap: Any, key: str) -> list[Any]:
    return [
        r
        for r in _trace_posts(cap)
        if any(k.lower() == "x-api-key" and v == key for k, v in r.headers)
    ]


def test_empty_snapshot_key_then_restricted_key_is_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import traigent.observability.otel.api as otel_api
    from traigent.config.backend_config import BackendConfig

    current = {"key": ""}
    monkeypatch.setattr(
        BackendConfig, "get_api_key", classmethod(lambda cls: current["key"])
    )
    real_preflight = otel_api._preflight_allows_export

    def preflight_then_key_appears(*a: Any, **k: Any) -> bool:
        allowed = real_preflight(*a, **k)  # empty key: nothing to validate -> allowed
        current["key"] = R_KEY  # the key (restricted) shows up after the gate ran
        return allowed

    monkeypatch.setattr(
        otel_api, "_preflight_allows_export", preflight_then_key_appears
    )
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        handle = otel.init(schedule_delay_s=0.05, exit_flush=False)
        with handle.tracer.start_as_current_span("x"):
            pass
        otel.flush(2)
        otel.shutdown()
    assert _carries_key(cap, R_KEY) == []
    assert [r for r in cap.requests if _is_trace(r)] == []


def _plugin_tracing(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    """Plugin tracing module (loaded by path) whose provider is captured, not global."""
    import importlib.util
    import types

    root = Path(__file__).resolve().parents[2]
    path = root / "plugins" / "traigent-tracing" / "traigent_tracing" / "tracing.py"
    spec = importlib.util.spec_from_file_location("_plugin_tracing_r7", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not mod.OTEL_AVAILABLE:
        pytest.skip("OpenTelemetry SDK not available")
    providers: list[Any] = []
    real_trace = mod.trace
    shim = types.SimpleNamespace(
        set_tracer_provider=providers.append,
        get_tracer=lambda *a, **k: providers[-1].get_tracer(*a, **k),
    )
    monkeypatch.setattr(mod, "trace", shim)
    del real_trace
    monkeypatch.setattr(mod, "TRACING_ENABLED", True)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"{BACKEND_URL}/otlp")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", raising=False)
    return mod, providers


def test_plugin_never_sends_a_latched_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The add-on built its own exporter with no gate: a restricted key in the OTLP
    headers (validated key A switched to latched R before the build) was sent."""
    plugin, providers = _plugin_tracing(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", f"x-api-key={R_KEY}")
    oo.latch_on(oo.credential_scope(BACKEND_URL, {"X-API-Key": R_KEY}))
    with capture_egress() as cap:
        tracer = plugin.get_tracer()
        if tracer is not None:
            with tracer.start_as_current_span("late"):
                pass
            for p in providers:
                p.force_flush(2000)
                p.shutdown()
    assert _carries_key(cap, R_KEY) == []
    assert _trace_posts(cap) == []
    assert cap.guard_violations == []


def test_previously_returned_plugin_tracer_does_not_export_after_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, providers = _plugin_tracing(monkeypatch)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.setattr(
        "traigent.config.backend_config.BackendConfig.get_api_key",
        classmethod(lambda cls: FAKE_API_KEY),
    )
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        tracer = plugin.get_tracer()
        assert tracer is not None
        oo.latch_on(oo.credential_scope(BACKEND_URL, {"X-API-Key": FAKE_API_KEY}))
        with tracer.start_as_current_span("late"):
            pass
        for p in providers:
            p.force_flush(2000)
            p.shutdown()
    assert _trace_posts(cap) == []
    assert cap.guard_violations == []


@pytest.mark.parametrize(
    "response",
    [
        {"valid": True},
        {"valid": True, "optimization_only": None},
        {"valid": True, "optimization_only": "false"},
        {"valid": True, "optimization_only": 0},
    ],
    ids=["absent", "null", "string", "int"],
)
def test_validation_without_explicit_false_flag_fails_closed(
    response: dict[str, Any],
) -> None:
    """Only an explicit JSON boolean false counts as unrestricted (P1 fail-open)."""
    with capture_egress() as cap:
        cap.backend.validate_response = response
        _emit()
    _assert_no_trace_egress(cap)


def test_offline_flip_between_reads_never_builds_unvalidated_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offline state is read once: offline at the preflight stays offline."""
    import traigent.observability.otel.api as otel_api

    calls = {"n": 0}

    def flipping() -> bool:
        calls["n"] += 1
        return calls["n"] == 1  # offline on the first read, online afterwards

    monkeypatch.setattr(otel_api, "is_backend_offline", flipping)
    with capture_egress() as cap:
        cap.backend.validate_response = dict(RESTRICTED)
        _emit()
    assert [r for r in cap.requests if _is_trace(r)] == []
    assert cap.guard_violations == []


def test_send_time_preflight_revalidates_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with capture_egress() as cap:
        cap.backend.validate_response = {"valid": True, "optimization_only": False}
        handle = otel.init(
            api_key=FAKE_API_KEY, schedule_delay_s=0.05, exit_flush=False
        )
        monkeypatch.setattr(oo, "ensure_trace_egress_allowed", lambda *a, **k: False)
        with handle.tracer.start_as_current_span("x"):
            pass
        otel.flush(2)
        otel.shutdown()
    assert [
        r for r in cap.requests if "/otlp" in r.path or "/v1/traces" in r.path
    ] == []


def test_async_workflow_ingest_never_follows_a_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import traigent.integrations.observability.workflow_traces as wt

    sent: list[str] = []

    class _Resp:
        status = 307
        headers = {"Location": "https://elsewhere.invalid/ingest"}

        def raise_for_status(self) -> None:
            return None

        async def json(self) -> dict[str, Any]:
            return {"success": True}

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    class _Session:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        def post(self, url: str, **kwargs: Any) -> _Resp:
            sent.append(url)
            if kwargs.get("allow_redirects", True):  # aiohttp's default follows
                sent.append(_Resp.headers["Location"])
            return _Resp()

    class _FakeAiohttp:
        ClientSession = _Session
        ClientError = Exception

        @staticmethod
        def ClientTimeout(**k: Any) -> Any:  # noqa: N802
            return None

    monkeypatch.setattr(wt, "aiohttp", _FakeAiohttp)
    monkeypatch.setattr(wt, "AIOHTTP_AVAILABLE", True)
    monkeypatch.setattr(oo, "ensure_trace_egress_allowed", lambda *a, **k: True)
    monkeypatch.setattr(oo, "is_active", lambda *a, **k: False)
    graph = wt.WorkflowGraphPayload(experiment_id="e", nodes=[], edges=[])
    client = wt.WorkflowTracesClient(BACKEND_URL, FAKE_API_KEY)
    result = asyncio.run(client.ingest_traces_async(graph=graph))
    assert len(sent) == 1
    assert result.success is False
    assert "Redirect refused" in (result.error or "")
