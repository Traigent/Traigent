"""Tests for backend cloud best-config client methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError

FAKE_TRAIGENT_API_KEY = "tg_" + "x" * 61  # pragma: allowlist secret


def _spec() -> dict:
    return {
        "schema_version": "traigent.best_config.v1",
        "config_id": "answerer",
        "function_ref": "pkg.answer:run",
        "environment": "staging",
        "config": {"temperature": 0.2},
    }


def _response(
    status_code: int,
    payload: dict | None = None,
    *,
    text: str = "ok",
) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.headers = {}
    response.json.return_value = payload or {"success": True, "data": {"ok": True}}
    return response


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


@patch("traigent.utils.retry.time.sleep", return_value=None)
@patch("requests.post")
def test_publish_best_config_sync_retries_transient_503_then_succeeds(
    mock_post,
    mock_sleep,
):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    mock_post.side_effect = [
        _response(503, text="temporarily unavailable"),
        _response(201, {"success": True, "data": {"config_id": "answerer"}}),
    ]

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        result = client.publish_best_config_sync(_spec(), environment="staging")

    assert result == {"config_id": "answerer"}
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("traigent.utils.retry.time.sleep", return_value=None)
@patch("requests.post")
def test_publish_best_config_sync_does_not_retry_401(mock_post, mock_sleep):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    mock_post.return_value = _response(401, text="unauthorized")

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        with pytest.raises(CloudServiceError):
            client.publish_best_config_sync(_spec(), environment="staging")

    mock_post.assert_called_once()
    mock_sleep.assert_not_called()


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


@patch("traigent.utils.retry.time.sleep", return_value=None)
@patch("requests.get")
def test_fetch_best_config_sync_retries_transient_503_then_succeeds(
    mock_get,
    mock_sleep,
):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    mock_get.side_effect = [
        _response(503, text="temporarily unavailable"),
        _response(
            200,
            {
                "success": True,
                "data": {"config_id": "answerer", "spec": _spec()},
            },
        ),
    ]

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        result = client.fetch_best_config_sync("answerer", environment="staging")

    assert result is not None
    assert result["config_id"] == "answerer"
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once()


@patch("traigent.utils.retry.time.sleep", return_value=None)
@patch("requests.get")
def test_fetch_best_config_sync_does_not_retry_422(mock_get, mock_sleep):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    mock_get.return_value = _response(422, text="invalid query")

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        with pytest.raises(CloudServiceError):
            client.fetch_best_config_sync("answerer", environment="staging")

    mock_get.assert_called_once()
    mock_sleep.assert_not_called()


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


@patch("traigent.utils.retry.time.sleep", return_value=None)
@patch("requests.post")
def test_upload_example_features_retries_transient_503_then_succeeds(
    mock_post,
    mock_sleep,
):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    mock_post.side_effect = [
        _response(503, text="temporarily unavailable"),
        _response(200),
    ]

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        result = client.upload_example_features(
            "run_123",
            "simhash_v1",
            [{"example_id": "ex_1", "feature": "0f0f"}],
        )

    assert result is True
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("traigent.utils.retry.time.sleep", return_value=None)
@patch("requests.post")
def test_upload_example_features_does_not_retry_422(mock_post, mock_sleep):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test"
    )
    mock_post.return_value = _response(422, text="invalid features")

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        result = client.upload_example_features(
            "run_123",
            "simhash_v1",
            [{"example_id": "ex_1", "feature": "0f0f"}],
        )

    assert result is False
    mock_post.assert_called_once()
    mock_sleep.assert_not_called()
