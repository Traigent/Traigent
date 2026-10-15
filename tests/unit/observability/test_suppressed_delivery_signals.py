"""Suppressed observability delivery must say so, and name its real cause.

#2602: ``flush()``/``close()`` keep ``success=True`` while delivery is
suppressed (telemetry must never break the host app), but the result has to
carry a warning so it cannot be mistaken for a real delivery.

#2603: control-plane calls blocked by offline mode must name the variable
that actually caused it, not always ``TRAIGENT_OFFLINE_MODE``.
"""

from __future__ import annotations

import pytest

from traigent.observability.client import ObservabilityClient
from traigent.observability.config import ObservabilityConfig
from traigent.utils.exceptions import ClientError

_OFFLINE_ENV_VARS = (
    "TRAIGENT_OFFLINE_MODE",
    "TRAIGENT_OFFLINE",
    "TRAIGENT_DISABLE_TELEMETRY",
)


@pytest.fixture(autouse=True)
def _clear_offline_env(monkeypatch, jwt_development_mode):
    del jwt_development_mode
    for name in _OFFLINE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_ENV", "development")


def _config(**overrides) -> ObservabilityConfig:
    values = {
        "backend_origin": "http://localhost:5000",
        "api_key": "test-key",  # pragma: allowlist secret
        "batch_size": 10,
        "max_buffer_age": 999.0,
        "max_queue_size": 10,
        "enable_atexit_flush": False,
    }
    values.update(overrides)
    return ObservabilityConfig(**values)


# Explicit None, not just unset env vars: BackendConfig otherwise falls back to
# stored CLI credentials, which would make this test depend on the machine.
_NO_CREDENTIALS = {"api_key": None, "jwt_token": None}


def _no_network(*args, **kwargs):
    raise AssertionError("network egress attempted while delivery is suppressed")


@pytest.mark.parametrize("env_var", _OFFLINE_ENV_VARS)
def test_offline_request_error_names_the_variable_that_was_set(monkeypatch, env_var):
    monkeypatch.setenv(env_var, "true")
    client = ObservabilityClient(_config())

    with pytest.raises(ClientError) as exc_info:
        client.list_sessions()
    client.close()

    message = str(exc_info.value)
    assert f"{env_var}=true" in message
    for other in _OFFLINE_ENV_VARS:
        if other != env_var:
            assert f"{other}=" not in message


def test_offline_request_error_names_every_variable_that_was_set(monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "1")
    monkeypatch.setenv("TRAIGENT_DISABLE_TELEMETRY", "yes")
    client = ObservabilityClient(_config())

    with pytest.raises(ClientError) as exc_info:
        client.list_traces()
    client.close()

    message = str(exc_info.value)
    assert "TRAIGENT_OFFLINE_MODE=1" in message
    assert "TRAIGENT_DISABLE_TELEMETRY=yes" in message


def test_disabled_telemetry_flush_and_close_warn_that_nothing_was_sent(monkeypatch):
    monkeypatch.setenv("TRAIGENT_DISABLE_TELEMETRY", "true")
    client = ObservabilityClient(_config())
    monkeypatch.setattr(client._http_opener, "open", _no_network)

    trace_id = client.start_trace("suppressed", trace_id="trace_suppressed")
    client.end_trace(trace_id)

    flush_result = client.flush()
    close_result = client.close()

    for result in (flush_result, close_result):
        # Contract kept: suppression never reports failure to the host app...
        assert result.success is True
        assert result.items_sent == 0
        # ...but the result is no longer indistinguishable from a delivery.
        assert len(result.warnings) == 1
        assert "suppressed" in result.warnings[0]
        assert "TRAIGENT_DISABLE_TELEMETRY=true" in result.warnings[0]
        assert "no traces were sent" in result.warnings[0]


def test_missing_credential_flush_and_close_warn_that_nothing_was_sent(monkeypatch):
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)
    client = ObservabilityClient(_config(**_NO_CREDENTIALS))
    monkeypatch.setattr(client._http_opener, "open", _no_network)

    trace_id = client.start_trace("no-credential", trace_id="trace_no_credential")
    client.end_trace(trace_id)

    flush_result = client.flush()
    close_result = client.close()

    for result in (flush_result, close_result):
        assert result.success is True
        assert result.items_sent == 0
        assert len(result.warnings) == 1
        assert "no credential resolved" in result.warnings[0]
        assert "no traces were sent" in result.warnings[0]


def test_delivering_flush_carries_no_suppression_warning():
    sent: list[list[dict]] = []
    client = ObservabilityClient(_config(), sender=sent.append)

    trace_id = client.start_trace("delivered", trace_id="trace_delivered")
    client.end_trace(trace_id)
    result = client.close()

    assert result.success is True
    assert result.items_sent >= 1
    assert sent
    assert not any("suppressed" in warning for warning in result.warnings)


def test_repeated_close_of_a_suppressed_client_still_warns(monkeypatch):
    monkeypatch.setenv("TRAIGENT_DISABLE_TELEMETRY", "true")
    client = ObservabilityClient(_config())

    client.close()
    second = client.close()

    assert second.items_sent == 0
    assert any("suppressed" in warning for warning in second.warnings)
