"""Regression tests: bounded process-global state and locked lazy singletons."""

from __future__ import annotations

import sys
import threading
import time

from traigent.cloud import billing, integration_manager, session_operations
from traigent.labels import transport_gate
from traigent.utils import env_config, langchain_interceptor
from traigent.wrapper import service as service_mod
from traigent.wrapper.service import TraigentService


def _service() -> TraigentService:
    return TraigentService(supports_keep_alive=True)


def test_sessions_capped_evicts_least_recently_active(monkeypatch):
    monkeypatch.setattr(service_mod, "MAX_SESSIONS", 5)
    svc = _service()
    for i in range(20):
        svc.handle_keep_alive(f"s{i}")
        svc._sessions[f"s{i}"].last_activity = i  # strictly increasing
    assert len(svc._sessions) == 5
    assert "s19" in svc._sessions and "s0" not in svc._sessions


def test_sessions_expired_are_evicted_first(monkeypatch):
    monkeypatch.setattr(service_mod, "MAX_SESSIONS", 3)
    svc = _service()
    for sid in ("a", "b", "c"):
        svc.create_session(sid)
    svc._sessions["c"].last_activity = 0.0  # long expired
    svc._sessions["a"].last_activity = time.time() - 10
    svc.create_session("d")
    assert "c" not in svc._sessions and {"a", "b", "d"} == set(svc._sessions)


def test_keep_alive_rejects_oversized_or_non_string_ids():
    svc = _service()
    assert svc.handle_keep_alive("x" * 257) is False
    assert svc.handle_keep_alive(123) is False  # type: ignore[arg-type]
    assert svc._sessions == {}
    assert svc.handle_keep_alive("x" * 256) is True


def test_refusal_log_is_bounded(monkeypatch):
    assert transport_gate._refusals.maxlen is not None
    transport_gate.clear_refusals()
    for i in range(transport_gate._MAX_REFUSALS + 50):
        transport_gate._refusals.append((f"t{i}", {}))
    assert len(transport_gate.refusal_log()) == transport_gate._MAX_REFUSALS
    transport_gate.clear_refusals()


def test_global_capture_outside_scope_is_bounded(monkeypatch):
    monkeypatch.setattr(langchain_interceptor, "_MAX_GLOBAL_RESPONSES", 10)
    cap = langchain_interceptor.LangChainMetadataCapture()
    for _ in range(100):
        cap.set_last_response(object())
    assert len(cap.get_all_responses()) == 10
    token = langchain_interceptor._current_key.set(None)
    langchain_interceptor._current_key.reset(token)
    for i in range(100):
        t = langchain_interceptor._current_key.set(f"k{i}")
        try:
            cap.set_last_response(object())
        finally:
            langchain_interceptor._current_key.reset(t)
    assert len(cap._by_key) == 10
    assert cap.get_by_key("k99") is not None and cap.get_by_key("k0") is None


def _race(getter, reset, rounds=40, n=12):
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(rounds):
            reset()
            barrier = threading.Barrier(n)
            out = []

            def work(barrier=barrier, out=out):
                barrier.wait()
                out.append(getter())

            ts = [threading.Thread(target=work) for _ in range(n)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            assert len({id(o) if not isinstance(o, str) else o for o in out}) == 1
    finally:
        sys.setswitchinterval(old)


def test_cost_tracker_singleton_race(monkeypatch):
    monkeypatch.setattr(billing, "_cost_tracker", None)
    _race(billing.get_cost_tracker, lambda: setattr(billing, "_cost_tracker", None))


def test_integration_manager_singleton_race(monkeypatch):
    monkeypatch.setattr(integration_manager, "_integration_manager", None)
    _race(
        integration_manager.get_integration_manager,
        lambda: setattr(integration_manager, "_integration_manager", None),
    )


def test_session_executor_singleton_race(monkeypatch):
    monkeypatch.setattr(session_operations, "_SESSION_EXECUTOR", None)
    _race(
        session_operations._get_session_executor,
        lambda: setattr(session_operations, "_SESSION_EXECUTOR", None),
    )


def test_dev_jwt_secret_race(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_DEV_JWT_SECRET", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setattr(env_config, "_GENERATED_DEV_JWT_SECRET", None)
    import warnings

    def getter():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return env_config.get_jwt_secret()

    _race(getter, lambda: setattr(env_config, "_GENERATED_DEV_JWT_SECRET", None))
