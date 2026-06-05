"""Tests for backend cloud best-config client methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from traigent.cloud.backend_client import BackendIntegratedClient

FAKE_TRAIGENT_API_KEY = "tg_" + "x" * 61  # pragma: allowlist secret


def _spec() -> dict:
    return {
        "schema_version": "traigent.best_config.v1",
        "config_id": "answerer",
        "function_ref": "pkg.answer:run",
        "environment": "staging",
        "config": {"temperature": 0.2},
    }


@patch("requests.post")
def test_publish_best_config_sync_posts_to_backend(mock_post, monkeypatch):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "project_cloud_123")
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    response = MagicMock(status_code=201, text="ok")
    response.json.return_value = {
        "success": True,
        "data": {"config_id": "answerer", "version": 1},
    }
    mock_post.return_value = response

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        result = client.publish_best_config_sync(
            _spec(),
            environment="staging",
            if_match='W/"1-old"',
        )

    assert result == {"config_id": "answerer", "version": 1}
    call_args = mock_post.call_args
    assert call_args.args[0].endswith("/api/v1/best-configs")
    assert call_args.kwargs["json"]["spec"]["config_id"] == "answerer"
    assert call_args.kwargs["json"]["environment"] == "staging"
    assert call_args.kwargs["headers"]["If-Match"] == 'W/"1-old"'
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert call_args.kwargs["headers"]["X-Project-Id"] == "project_cloud_123"


@patch("requests.get")
def test_fetch_best_config_sync_gets_current_config(mock_get):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    response = MagicMock(status_code=200, text="ok")
    response.json.return_value = {
        "success": True,
        "data": {
            "config_id": "answerer",
            "etag": 'W/"1-abcdef123456"',
            "spec": _spec(),
        },
    }
    mock_get.return_value = response

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        result = client.fetch_best_config_sync(
            "answerer",
            environment="staging",
            function_ref="pkg.answer:run",
            etag='W/"1-old"',
        )

    assert result is not None
    assert result["spec"]["config"]["temperature"] == 0.2
    call_args = mock_get.call_args
    assert call_args.args[0].endswith("/api/v1/best-configs/answerer")
    assert call_args.kwargs["params"] == {
        "environment": "staging",
        "function_ref": "pkg.answer:run",
    }
    assert call_args.kwargs["headers"]["If-None-Match"] == 'W/"1-old"'


@patch("requests.get")
def test_fetch_best_config_sync_returns_none_for_not_found(mock_get):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    mock_get.return_value = MagicMock(status_code=404, text="missing")

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        assert client.fetch_best_config_sync("missing") is None
