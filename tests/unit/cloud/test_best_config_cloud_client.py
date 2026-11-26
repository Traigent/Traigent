"""Tests for backend cloud best-config client methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online

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


@pytest.mark.parametrize(("max_retries", "expected_calls"), [(0, 1), (1, 2), (4, 5)])
@patch("traigent.utils.retry.time.sleep", return_value=None)
@patch("requests.get")
def test_fetch_best_config_sync_honours_client_max_retries(
    mock_get, mock_sleep, max_retries, expected_calls
):
    client = BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY,
        base_url="https://api.test",
        max_retries=max_retries,
    )
    mock_get.return_value = _response(429, text="slow down")

    with patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer test-token"}),
    ):
        with pytest.raises(CloudServiceError):
            client.fetch_best_config_sync("answerer", environment="staging")

    assert mock_get.call_count == expected_calls


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


# --- explicit project on flat best-config calls ------------------------------------------

_PCR_BODY = {
    "success": False,
    "error": "PROJECT_CONTEXT_REQUIRED",
    "error_code": "PROJECT_CONTEXT_REQUIRED",
    "message": "project context required",
}
_PCR_TEXT = (
    "this call needs a project: set TRAIGENT_PROJECT_ID or use a project-bound key"
)


def _flat_client(**kwargs):
    return BackendIntegratedClient(
        api_key=FAKE_TRAIGENT_API_KEY, base_url="https://api.test", **kwargs
    )


def _auth(client):
    return patch.object(
        client.auth_manager.auth,
        "get_headers",
        AsyncMock(return_value={"Authorization": "Bearer t"}),
    )


def _call_publish(client):
    return client.publish_best_config_sync(_spec(), environment="staging")


def _call_fetch(client):
    return client.fetch_best_config_sync("answerer", environment="staging")


_FLAT_CALLS = [
    pytest.param("requests.post", _call_publish, id="publish"),
    pytest.param("requests.get", _call_fetch, id="fetch"),
]


@pytest.mark.parametrize(("target", "call"), _FLAT_CALLS)
def test_flat_call_sends_the_configured_project_unchanged(target, call, monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client(project_id="project_explicit_1")
    with _auth(client), patch(target) as mock_http:
        mock_http.return_value = _response(200)
        call(client)
    assert mock_http.call_args.kwargs["headers"]["X-Project-Id"] == "project_explicit_1"


@pytest.mark.parametrize(("target", "call"), _FLAT_CALLS)
def test_flat_call_env_fallback_sends_the_env_project(target, call, monkeypatch):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "project_env_1")
    client = _flat_client()
    with _auth(client), patch(target) as mock_http:
        mock_http.return_value = _response(200)
        call(client)
    assert mock_http.call_args.kwargs["headers"]["X-Project-Id"] == "project_env_1"


@pytest.mark.parametrize(("target", "call"), _FLAT_CALLS)
def test_flat_call_without_project_sends_no_header_and_succeeds(
    target, call, monkeypatch
):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client()
    with _auth(client), patch(target) as mock_http:
        mock_http.return_value = _response(200)
        call(client)  # a project-bound key is accepted by the backend
    assert mock_http.call_count == 1
    assert "X-Project-Id" not in mock_http.call_args.kwargs["headers"]


@pytest.mark.parametrize(("target", "call"), _FLAT_CALLS)
def test_flat_call_maps_project_context_required_after_one_request(
    target, call, monkeypatch
):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client()
    with _auth(client), patch(target) as mock_http:
        mock_http.return_value = _response(422, _PCR_BODY, text="x")
        with pytest.raises(CloudServiceError) as err:
            call(client)
    assert str(err.value) == _PCR_TEXT
    assert mock_http.call_count == 1


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize(("target", "call"), _FLAT_CALLS)
def test_flat_call_explicit_blank_raises_before_transport(
    target, call, blank, monkeypatch
):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "project_env_1")  # never a fallback
    client = _flat_client(project_id=blank)
    with _auth(client), patch(target) as mock_http:
        with pytest.raises(CloudServiceError):
            call(client)
    mock_http.assert_not_called()


@pytest.mark.parametrize(("target", "call"), _FLAT_CALLS)
def test_flat_call_sends_legacy_project_default_unchanged(target, call, monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client(project_id="project_default")
    with _auth(client), patch(target) as mock_http:
        mock_http.return_value = _response(200)
        call(client)
    assert mock_http.call_args.kwargs["headers"]["X-Project-Id"] == "project_default"


@pytest.mark.parametrize("bad", ["proj\nX-Evil: 1", "proj\x00", "a\rb", "a\x7fb"])
@pytest.mark.parametrize(("target", "call"), _FLAT_CALLS)
def test_flat_call_control_character_project_raises_before_transport(
    target, call, bad, monkeypatch
):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client(project_id=bad)
    with _auth(client), patch(target) as mock_http:
        with pytest.raises(CloudServiceError) as err:
            call(client)
    mock_http.assert_not_called()
    assert "Evil" not in str(err.value)


@pytest.mark.parametrize(
    "code", ["PROJECT_CONTEXT_REQUIRED", "PROJECT_CONTEXT_REQUIRED_FOR_WRITE"]
)
@pytest.mark.parametrize(("target", "call"), _FLAT_CALLS)
def test_flat_call_maps_both_project_context_codes(target, call, code, monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client()
    with _auth(client), patch(target) as mock_http:
        mock_http.return_value = _response(422, {"error_code": code}, text="x")
        with pytest.raises(CloudServiceError) as err:
            call(client)
    assert str(err.value) == _PCR_TEXT


def test_message_text_alone_is_not_mapped(monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client()
    with _auth(client), patch("requests.get") as mock_http:
        mock_http.return_value = _response(
            422, {"error_code": "OTHER", "message": "PROJECT_CONTEXT_REQUIRED"}
        )
        with pytest.raises(CloudServiceError) as err:
            _call_fetch(client)
    assert str(err.value) != _PCR_TEXT


@pytest.mark.parametrize("bad", ["p\r\n", "\tp", "p\n"])
def test_flat_call_control_character_at_the_edges_raises_before_transport(
    bad, monkeypatch
):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client(project_id=bad)
    with _auth(client), patch("requests.get") as mock_http:
        with pytest.raises(CloudServiceError):
            _call_fetch(client)
    mock_http.assert_not_called()


@pytest.mark.parametrize(
    "code", ["PROJECT_CONTEXT_REQUIRED", "PROJECT_CONTEXT_REQUIRED_FOR_WRITE"]
)
def test_protected_publish_maps_both_project_context_codes(code, monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client()
    with (
        _auth(client),
        patch.object(
            client,
            "_protected_request",
            return_value=_response(422, {"error_code": code}),
        ) as sent,
    ):
        with pytest.raises(CloudServiceError) as err:
            client._publish_best_config_with_ref(
                _spec(), "staging", None, None, 1, "agentref_" + "a" * 32
            )
    assert str(err.value) == _PCR_TEXT
    assert sent.call_count == 1
    assert "X-Project-Id" not in sent.call_args.kwargs["headers"]


def test_a_list_error_code_on_a_422_is_not_mapped(monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    client = _flat_client()
    with _auth(client), patch("requests.get") as mock_http:
        mock_http.return_value = _response(
            422, {"error_code": ["PROJECT_CONTEXT_REQUIRED"]}
        )
        with pytest.raises(CloudServiceError) as err:
            _call_fetch(client)
    assert str(err.value) != _PCR_TEXT
