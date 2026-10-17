"""Ingest request timeout scaling and the at-exit undelivered report (SDK-39)."""

from __future__ import annotations

import pytest

from traigent.observability.client import ObservabilityClient
from traigent.observability.config import ObservabilityConfig
from traigent.observability.dtos import FlushResult


def _config(**overrides) -> ObservabilityConfig:
    return ObservabilityConfig(
        backend_origin="http://localhost:5000",
        api_key="test-key",  # pragma: allowlist secret
        enable_atexit_flush=False,
        offline_mode=False,
        **overrides,
    )


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b""


def _timeout_used(config: ObservabilityConfig, n: int) -> float:
    client = ObservabilityClient(config, sender=lambda traces: None)
    seen: list[float] = []

    class _Opener:
        def open(self, http_request, timeout):
            seen.append(timeout)
            return _Resp()

    client._http_opener = _Opener()
    client._post_batch_sync([{"id": f"t{i}"} for i in range(n)])
    return seen[0]


def test_hundred_trace_batch_gets_scaled_timeout():
    assert _timeout_used(_config(), 100) == pytest.approx(20.0)


def test_small_batch_keeps_configured_floor():
    assert _timeout_used(_config(), 5) == pytest.approx(10.0)


def test_explicit_larger_request_timeout_is_kept():
    assert _timeout_used(_config(request_timeout=60.0), 100) == pytest.approx(60.0)


def test_per_trace_timeout_is_overridable_and_zero_disables_scaling():
    assert _timeout_used(_config(per_trace_timeout=0.5), 100) == pytest.approx(50.0)
    assert _timeout_used(_config(per_trace_timeout=0), 100) == pytest.approx(10.0)


def test_negative_per_trace_timeout_rejected():
    with pytest.raises(ValueError):
        _config(per_trace_timeout=-1)


def _exit_with(monkeypatch, capsys, pending: int, **overrides) -> str:
    client = ObservabilityClient(_config(**overrides))
    monkeypatch.setattr(
        client,
        "close",
        lambda *, timeout=None: FlushResult(pending == 0, 0, pending, 0, 0, 0),
    )
    client._atexit_close()
    return capsys.readouterr().err


def test_exit_with_pending_prints_exact_count(monkeypatch, capsys):
    err = _exit_with(monkeypatch, capsys, 100)
    assert err.count("\n") == 1
    assert "100 observability trace(s)" in err
    assert "flush()" in err and "flush_timeout" in err


def test_exit_with_nothing_pending_is_silent(monkeypatch, capsys):
    assert _exit_with(monkeypatch, capsys, 0) == ""


def test_exit_warning_can_be_disabled(monkeypatch, capsys):
    assert _exit_with(monkeypatch, capsys, 7, warn_on_undelivered_at_exit=False) == ""


def test_positional_construction_keeps_existing_field_meaning():
    cfg = ObservabilityConfig(
        "http://localhost:5000",
        "test-key",  # pragma: allowlist secret
        None,
        None,
        None,
        "/api/v1beta/observability",
        100,
        5,
        10000,
        4194304,
        30,
        10,
        False,
        True,
    )
    assert cfg.request_timeout == 10
    assert cfg.enable_atexit_flush is False
    assert cfg.offline_mode is True
    assert cfg.per_trace_timeout == 0.2


def test_effective_timeout_capped_at_max():
    from traigent.observability.config import MAX_TIMEOUT_SECONDS

    cfg = _config(per_trace_timeout=100.0)
    assert cfg.effective_request_timeout(10_000) == MAX_TIMEOUT_SECONDS


def test_exit_with_broken_stderr_does_not_raise(monkeypatch):
    class _Broken:
        def write(self, _):
            raise ValueError("I/O operation on closed file")

    client = ObservabilityClient(_config())
    monkeypatch.setattr(
        client, "close", lambda *, timeout=None: FlushResult(False, 0, 3, 0, 0, 0)
    )
    monkeypatch.setattr("sys.stderr", _Broken())
    client._atexit_close()


def test_exit_called_twice_prints_once(monkeypatch, capsys):
    client = ObservabilityClient(_config())
    monkeypatch.setattr(
        client, "close", lambda *, timeout=None: FlushResult(False, 0, 3, 0, 0, 0)
    )
    client._atexit_close()
    client._atexit_close()
    assert capsys.readouterr().err.count("\n") == 1
