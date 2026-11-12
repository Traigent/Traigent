"""Regression tests: bounded process-global state and locked lazy singletons."""

from __future__ import annotations

import itertools
import threading
import time
import types
import warnings

import pytest

from traigent.cloud import (
    backend_client,
    billing,
    integration_manager,
    production_mcp_client,
    session_operations,
)
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


class _BlockingCtor:
    """Constructor stand-in that blocks until released and counts calls."""

    def __init__(self, n_threads: int, make=object):
        self.make = make
        self.n = n_threads
        self.calls = 0
        self._lock = threading.Lock()
        self.release = threading.Event()

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
        self.release.wait(10)
        return self.make()

    def run(self, getter):
        results: list = []
        threads = [
            threading.Thread(target=lambda: results.append(getter()))
            for _ in range(self.n)
        ]
        for t in threads:
            t.start()
        # Without a lock every thread reaches the constructor and blocks there;
        # with one, only the first does. Wait for the unlocked case, then let go.
        deadline = time.time() + 2.0
        while self.calls < self.n and time.time() < deadline:
            time.sleep(0.005)
        self.release.set()
        for t in threads:
            t.join(10)
        assert len(results) == self.n
        return results


def _assert_single_construction(monkeypatch, patch_target, getter, reset, make=object):
    ctor = _BlockingCtor(4, make)
    patch_target(monkeypatch, ctor)
    reset()
    results = ctor.run(getter)
    assert ctor.calls == 1
    assert len({id(r) for r in results}) == 1


def test_cost_tracker_constructed_once(monkeypatch):
    _assert_single_construction(
        monkeypatch,
        lambda mp, c: mp.setattr(billing, "CostTracker", c),
        billing.get_cost_tracker,
        lambda: monkeypatch.setattr(billing, "_cost_tracker", None),
    )


def test_integration_manager_constructed_once(monkeypatch):
    _assert_single_construction(
        monkeypatch,
        lambda mp, c: mp.setattr(integration_manager, "IntegrationManager", c),
        integration_manager.get_integration_manager,
        lambda: monkeypatch.setattr(integration_manager, "_integration_manager", None),
    )


def test_session_executor_constructed_once(monkeypatch):
    def patch(mp, c):
        fake = types.SimpleNamespace(
            futures=types.SimpleNamespace(ThreadPoolExecutor=c)
        )
        mp.setattr(session_operations, "concurrent", fake)

    _assert_single_construction(
        monkeypatch,
        patch,
        session_operations._get_session_executor,
        lambda: monkeypatch.setattr(session_operations, "_SESSION_EXECUTOR", None),
    )


def test_backend_client_constructed_once(monkeypatch):
    _assert_single_construction(
        monkeypatch,
        lambda mp, c: mp.setattr(backend_client, "BackendIntegratedClient", c),
        backend_client.get_backend_client,
        lambda: monkeypatch.setattr(backend_client, "_backend_client", None),
    )


def test_production_mcp_client_constructed_once(monkeypatch):
    _assert_single_construction(
        monkeypatch,
        lambda mp, c: mp.setattr(production_mcp_client, "ProductionMCPClient", c),
        production_mcp_client.get_production_mcp_client,
        lambda: monkeypatch.setattr(production_mcp_client, "_production_client", None),
        make=lambda: types.SimpleNamespace(server_config=types.SimpleNamespace()),
    )


def test_dev_jwt_secret_generated_once(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_DEV_JWT_SECRET", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    ctor = _BlockingCtor(4)
    counter = itertools.count()

    def fake_token(*_a, **_k):
        ctor()
        return f"secret-{next(counter)}"

    monkeypatch.setattr(
        env_config, "secrets", types.SimpleNamespace(token_urlsafe=fake_token)
    )
    monkeypatch.setattr(env_config, "_GENERATED_DEV_JWT_SECRET", None)

    def getter():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return env_config.get_jwt_secret()

    results = ctor.run(getter)
    assert ctor.calls == 1
    assert len(set(results)) == 1


# --- session id validation (#3128 review) ---------------------------------
BAD_IDS = [["a"], {"k": "v"}, 7, 1.5, True, "x" * 257]


@pytest.mark.parametrize("bad", BAD_IDS)
def test_keep_alive_bad_ids_rejected_without_error(bad):
    svc = _service()
    assert svc.handle_keep_alive(bad) is False
    assert svc._sessions == {}


@pytest.mark.parametrize("bad", BAD_IDS)
def test_touch_or_create_validates_before_lookup(bad):
    svc = _service()
    assert svc._touch_or_create_session(bad) is False


@pytest.mark.parametrize("bad", BAD_IDS)
@pytest.mark.asyncio
async def test_execute_and_evaluate_ignore_bad_session_ids(bad):
    svc = _service()

    @svc.execute
    def run(example_id, data, config):
        return {"output": "ok"}

    @svc.evaluate
    def score(output, target, config):
        return {"accuracy": 1.0}

    await svc.handle_execute(
        {"examples": [{"example_id": "i1", "data": {}}], "session_id": bad}
    )
    await svc.handle_evaluate(
        {
            "evaluations": [{"example_id": "i1", "output": "a", "target": "a"}],
            "session_id": bad,
        }
    )
    assert svc._sessions == {}


@pytest.mark.parametrize("bad", [["a"], {"k": "v"}, 7, "x" * 257])
def test_create_session_validates_id(bad):
    svc = _service()
    with pytest.raises(ValueError):
        svc.create_session(bad)
    assert svc._sessions == {}


def test_create_session_accepts_valid_and_generates():
    svc = _service()
    assert svc.create_session("ok") == "ok"
    assert len(svc.create_session()) <= 256
