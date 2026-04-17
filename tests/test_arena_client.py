"""Unit tests for the Arena Python SDK client (traigent.arena.client)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests as _requests

from traigent.arena.client import ArenaClient
from traigent.arena.config import ArenaConfig
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> ArenaConfig:
    """Build an ArenaConfig with sensible test defaults."""
    defaults = {
        "backend_origin": "https://test.traigent.dev",
        "api_key": "test-key-123",  # pragma: allowlist secret
        "tenant_id": None,
        "project_id": None,
    }
    defaults.update(overrides)
    return ArenaConfig(**defaults)


def make_sender(responses: list[dict[str, Any]]):
    """Return a request_sender that returns canned responses in order."""
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def sender(method: str, path: str, payload: dict[str, Any] | None):
        calls.append((method, path, payload))
        return responses.pop(0)

    return sender, calls


# Minimal valid DTO payloads
_PROVIDER_SOURCE_DATA = {
    "id": "ps-1",
    "label": "My Source",
    "provider": "openai",
    "kind": "api_key",
    "status": "active",
}

_RUN_DATA = {
    "id": "run-1",
    "function_name": "my_func",
    "status": "running",
}

_INVOKE_DATA = {
    "usage_id": "u-1",
    "run_id": "run-1",
    "provider_source_id": "ps-1",
    "provider": "openai",
    "model": "gpt-4o",
    "content": "Hello!",
}


# ---------------------------------------------------------------------------
# 1. Transport uses requests.Session (not urllib)
# ---------------------------------------------------------------------------

class TestTransportUsesRequests:
    """Verify that the client creates a requests.Session and calls session.request()."""

    def test_get_session_creates_requests_session(self):
        config = _make_config()
        client = ArenaClient(config=config)
        session = client._get_session()
        assert isinstance(session, _requests.Session)

    @patch("traigent.arena.client._requests.Session")
    def test_request_json_sync_calls_session_request(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"success": true, "data": {}}'
        mock_session.request.return_value = mock_response
        mock_session_cls.return_value = mock_session

        config = _make_config()
        client = ArenaClient(config=config)

        client._request_json_sync("GET", "/test-path")

        mock_session.request.assert_called_once()
        call_kwargs = mock_session.request.call_args
        assert call_kwargs.kwargs["method"] == "GET"
        assert call_kwargs.kwargs["url"].endswith("/api/v1beta/arena/test-path")


# ---------------------------------------------------------------------------
# 2. Request construction (invoke, create_provider_source, list, create_run)
# ---------------------------------------------------------------------------

class TestRequestConstruction:
    """Test that public methods build correct payloads and hit the right endpoints."""

    def test_invoke_builds_correct_payload(self):
        sender, calls = make_sender([
            {"success": True, "data": _INVOKE_DATA},
        ])
        client = ArenaClient(config=_make_config(), request_sender=sender)

        result = client.invoke(
            run_id="run-1",
            provider_source_id="ps-1",
            model="gpt-4o",
            prompt="Hello",
            temperature=0.7,
            max_tokens=100,
        )

        assert len(calls) == 1
        method, path, payload = calls[0]
        assert method == "POST"
        assert path == "/invoke"
        assert payload["run_id"] == "run-1"
        assert payload["provider_source_id"] == "ps-1"
        assert payload["model"] == "gpt-4o"
        assert payload["prompt"] == "Hello"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100
        assert payload["include_raw"] is False
        assert result.usage_id == "u-1"

    def test_create_provider_source_builds_correct_payload(self):
        sender, calls = make_sender([
            {"success": True, "data": _PROVIDER_SOURCE_DATA},
        ])
        client = ArenaClient(config=_make_config(), request_sender=sender)

        result = client.create_provider_source(
            provider="openai",
            kind="api_key",
            label="My Source",
            allowed_models=["gpt-4o"],
        )

        assert len(calls) == 1
        method, path, payload = calls[0]
        assert method == "POST"
        assert path == "/provider-sources"
        assert payload["provider"] == "openai"
        assert payload["kind"] == "api_key"
        assert payload["label"] == "My Source"
        assert payload["allowed_models"] == ["gpt-4o"]
        assert payload["persist"] is True
        assert result.id == "ps-1"

    def test_list_provider_sources_uses_get_with_pagination(self):
        sender, calls = make_sender([
            {
                "success": True,
                "data": {"items": [_PROVIDER_SOURCE_DATA]},
            },
        ])
        client = ArenaClient(config=_make_config(), request_sender=sender)

        results = client.list_provider_sources(page=2, per_page=10)

        method, path, payload = calls[0]
        assert method == "GET"
        assert "page=2" in path
        assert "per_page=10" in path
        assert payload is None
        assert len(results) == 1
        assert results[0].id == "ps-1"

    def test_create_run_builds_correct_payload(self):
        sender, calls = make_sender([
            {"success": True, "data": _RUN_DATA},
        ])
        client = ArenaClient(config=_make_config(), request_sender=sender)

        result = client.create_run(
            function_name="my_func",
            provider_source_ids=["ps-1", "ps-2"],
            configuration_space={"model": ["gpt-4o"]},
            objectives=[{"name": "accuracy", "direction": "maximize"}],
            max_trials=5,
            name="test-run",
        )

        method, path, payload = calls[0]
        assert method == "POST"
        assert path == "/runs"
        assert payload["function_name"] == "my_func"
        assert payload["max_trials"] == 5
        assert payload["name"] == "test-run"
        assert payload["configuration_space"] == {"model": ["gpt-4o"]}
        assert payload["objectives"] == [{"name": "accuracy", "direction": "maximize"}]
        # provider_source_ids should be expanded into providers list
        assert payload["providers"] == [
            {"provider_source_id": "ps-1"},
            {"provider_source_id": "ps-2"},
        ]
        assert result.id == "run-1"

    def test_invoke_omits_none_optional_fields(self):
        sender, calls = make_sender([
            {"success": True, "data": _INVOKE_DATA},
        ])
        client = ArenaClient(config=_make_config(), request_sender=sender)

        client.invoke(
            run_id="run-1",
            provider_source_id="ps-1",
            model="gpt-4o",
        )

        _, _, payload = calls[0]
        # These optional fields should NOT be present when not provided
        assert "prompt" not in payload
        assert "messages" not in payload
        assert "temperature" not in payload
        assert "max_tokens" not in payload
        assert "metadata" not in payload


# ---------------------------------------------------------------------------
# 3. Error handling (ConnectionError, Timeout, 401/403, 4xx/5xx)
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test that transport exceptions are mapped to the correct SDK exceptions."""

    @patch("traigent.arena.client._requests.Session")
    def test_connection_error_raises_traigent_connection_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.request.side_effect = _requests.ConnectionError("refused")
        mock_session_cls.return_value = mock_session

        client = ArenaClient(config=_make_config())

        with pytest.raises(TraigentConnectionError, match="Could not reach"):
            client._request_json_sync("GET", "/ping")

    @patch("traigent.arena.client._requests.Session")
    def test_timeout_raises_traigent_connection_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.request.side_effect = _requests.Timeout("timed out")
        mock_session_cls.return_value = mock_session

        client = ArenaClient(config=_make_config())

        with pytest.raises(TraigentConnectionError, match="timed out"):
            client._request_json_sync("GET", "/slow")

    @patch("traigent.arena.client._requests.Session")
    def test_http_401_raises_authentication_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = '{"error": "Invalid API key"}'
        mock_session.request.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = ArenaClient(config=_make_config())

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client._request_json_sync("GET", "/secure")

    @patch("traigent.arena.client._requests.Session")
    def test_http_403_raises_authentication_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = '{"error": "Forbidden"}'
        mock_session.request.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = ArenaClient(config=_make_config())

        with pytest.raises(AuthenticationError, match="Forbidden"):
            client._request_json_sync("POST", "/admin")

    @patch("traigent.arena.client._requests.Session")
    def test_http_500_raises_client_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = '{"error": "Internal server error"}'
        mock_session.request.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = ArenaClient(config=_make_config())

        with pytest.raises(ClientError, match="Internal server error"):
            client._request_json_sync("GET", "/broken")

    @patch("traigent.arena.client._requests.Session")
    def test_http_422_raises_client_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = '{"message": "Validation failed"}'
        mock_session.request.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = ArenaClient(config=_make_config())

        with pytest.raises(ClientError, match="Validation failed"):
            client._request_json_sync("POST", "/validate")


# ---------------------------------------------------------------------------
# 4. Response unwrapping (_unwrap_data, _unwrap_items)
# ---------------------------------------------------------------------------

class TestResponseUnwrapping:
    """Test _unwrap_data and _unwrap_items for success and error payloads."""

    def test_unwrap_data_success(self):
        payload = {"success": True, "data": {"id": "x"}}
        result = ArenaClient._unwrap_data(payload, "test")
        assert result == {"id": "x"}

    def test_unwrap_data_missing_data_key(self):
        payload = {"success": True}
        with pytest.raises(ClientError, match="did not include data"):
            ArenaClient._unwrap_data(payload, "test")

    def test_unwrap_data_explicit_failure(self):
        payload = {"success": False, "error": "Something went wrong"}
        with pytest.raises(ClientError, match="Something went wrong"):
            ArenaClient._unwrap_data(payload, "test")

    def test_unwrap_data_explicit_failure_message_field(self):
        payload = {"success": False, "message": "Bad request"}
        with pytest.raises(ClientError, match="Bad request"):
            ArenaClient._unwrap_data(payload, "test")

    def test_unwrap_data_explicit_failure_fallback(self):
        payload = {"success": False}
        with pytest.raises(ClientError, match="request failed"):
            ArenaClient._unwrap_data(payload, "my thing")

    def test_unwrap_items_success(self):
        payload = {"success": True, "data": {"items": [{"id": "a"}, {"id": "b"}]}}
        result = ArenaClient._unwrap_items(payload, "list")
        assert result == [{"id": "a"}, {"id": "b"}]

    def test_unwrap_items_missing_items_key(self):
        payload = {"success": True, "data": {"no_items": True}}
        with pytest.raises(ClientError, match="did not include items"):
            ArenaClient._unwrap_items(payload, "list")

    def test_unwrap_items_data_not_dict(self):
        payload = {"success": True, "data": "oops"}
        with pytest.raises(ClientError, match="invalid paginated data"):
            ArenaClient._unwrap_items(payload, "list")


# ---------------------------------------------------------------------------
# 5. Session reuse (_get_session returns the same session)
# ---------------------------------------------------------------------------

class TestSessionReuse:
    """Test that _get_session returns the same session on repeated calls."""

    def test_get_session_returns_same_instance(self):
        client = ArenaClient(config=_make_config())
        session_a = client._get_session()
        session_b = client._get_session()
        assert session_a is session_b

    def test_get_session_creates_new_after_close(self):
        client = ArenaClient(config=_make_config())
        session_a = client._get_session()
        client.close()
        session_b = client._get_session()
        assert session_a is not session_b


# ---------------------------------------------------------------------------
# 6. Headers (X-API-Key, Content-Type, tenant/project)
# ---------------------------------------------------------------------------

class TestHeaders:
    """Test that ArenaConfig.build_headers() includes expected headers."""

    def test_headers_include_api_key(self):
        config = _make_config(api_key="secret-key")
        headers = config.build_headers()
        assert headers["X-API-Key"] == "secret-key"

    def test_headers_include_content_type(self):
        config = _make_config()
        headers = config.build_headers()
        assert headers["Content-Type"] == "application/json"

    def test_headers_include_tenant_when_set(self):
        config = _make_config(tenant_id="tenant-abc")
        headers = config.build_headers()
        assert headers["X-Tenant-Id"] == "tenant-abc"

    def test_headers_omit_tenant_when_none(self):
        config = _make_config(tenant_id=None)
        headers = config.build_headers()
        assert "X-Tenant-Id" not in headers

    def test_headers_include_project_when_set(self):
        config = _make_config(project_id="proj-xyz")
        headers = config.build_headers()
        assert headers["X-Project-Id"] == "proj-xyz"

    def test_headers_omit_project_when_none(self):
        config = _make_config(project_id=None)
        headers = config.build_headers()
        assert "X-Project-Id" not in headers

    def test_headers_omit_api_key_when_none(self):
        config = _make_config(api_key=None)
        headers = config.build_headers()
        assert "X-API-Key" not in headers

    def test_headers_include_extra_headers(self):
        config = _make_config(extra_headers={"X-Custom": "val"})
        headers = config.build_headers()
        assert headers["X-Custom"] == "val"


# ---------------------------------------------------------------------------
# 7. Close
# ---------------------------------------------------------------------------

class TestClose:
    """Test that close() closes the underlying session."""

    def test_close_closes_session(self):
        client = ArenaClient(config=_make_config())
        session = client._get_session()

        with patch.object(session, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()

        assert client._session is None

    def test_close_noop_when_no_session(self):
        client = ArenaClient(config=_make_config())
        # Should not raise even though no session has been created
        client.close()
        assert client._session is None


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

class TestRequestSenderOverride:
    """Verify request_sender override bypasses real HTTP transport."""

    def test_request_sender_override_is_used(self):
        sender, calls = make_sender([
            {"success": True, "data": _RUN_DATA},
        ])
        client = ArenaClient(config=_make_config(), request_sender=sender)

        client.get_run("run-1")

        assert len(calls) == 1
        method, path, payload = calls[0]
        assert method == "GET"
        assert path == "/runs/run-1"
        assert payload is None

    def test_request_sender_override_does_not_create_session(self):
        sender, _ = make_sender([
            {"success": True, "data": _RUN_DATA},
        ])
        client = ArenaClient(config=_make_config(), request_sender=sender)

        client.get_run("run-1")

        # No session should have been created since we used the override
        assert client._session is None


class TestDecodeJsonPayload:
    """Test _decode_json_payload edge cases."""

    def test_empty_body_returns_empty_dict(self):
        result = ArenaClient._decode_json_payload("")
        assert result == {}

    def test_invalid_json_raises_client_error(self):
        with pytest.raises(ClientError, match="invalid JSON"):
            ArenaClient._decode_json_payload("{not json}")
