from __future__ import annotations

import io
import json
from email.message import Message
from urllib import error

import pytest

from traigent.prompts import (
    ChatPromptMessage,
    PromptApiError,
    PromptConflictError,
    PromptInvalidRequestError,
    PromptManagementClient,
    PromptNotFoundError,
    PromptNotPermittedError,
    PromptPaymentRequiredError,
    PromptType,
    PromptUpstreamError,
)
from traigent.prompts.config import PromptManagementConfig
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    ConfigurationError,
    RateLimitError,
    ServiceUnavailableError,
    TraigentConnectionError,
)


@pytest.fixture(autouse=True)
def _online_backend(jwt_development_mode, monkeypatch):
    """These tests exercise the online client request path, so they opt out of
    the suite-wide TRAIGENT_OFFLINE_MODE=true default (#1068). The transport is
    mocked, so no real egress occurs; depends on jwt_development_mode so this
    override runs after it."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")


def test_prompt_management_client_lists_and_gets_prompts():
    calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path, payload))
        if path.startswith("?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "prompt_1",
                            "name": "support/welcome",
                            "prompt_type": "text",
                            "description": "Support welcome prompt",
                            "latest_version": 2,
                            "version_count": 2,
                            "labels": {"latest": 2, "production": 1},
                            "tags": ["support"],
                        }
                    ],
                    "pagination": {
                        "limit": 20,
                        "next_cursor": None,
                        "has_more": False,
                    },
                }
            }
        return {
            "data": {
                "id": "prompt_1",
                "name": "support/welcome",
                "prompt_type": "text",
                "description": "Support welcome prompt",
                "latest_version": 2,
                "version_count": 2,
                "labels": {"latest": 2, "production": 1},
                "tags": ["support"],
                "versions": [
                    {
                        "id": "version_2",
                        "version": 2,
                        "prompt_type": "text",
                        "prompt_text": "Hello {{ customer_name }}",
                        "chat_messages": None,
                        "config": {"model": "gpt-4.1-mini"},
                        "commit_message": "Update greeting",
                        "variable_names": ["customer_name"],
                        "labels": ["latest"],
                    }
                ],
            }
        }

    client = PromptManagementClient(request_sender=request_sender)

    prompt_list = client.list_prompts(
        search="support", prompt_type=PromptType.TEXT, label="production"
    )
    detail = client.get_prompt("support/welcome")

    assert calls[0] == (
        "GET",
        "?limit=20&search=support&prompt_type=text&label=production",
        None,
    )
    assert calls[1] == ("GET", "/support%2Fwelcome", None)
    assert prompt_list.items[0].name == "support/welcome"
    assert prompt_list.pagination.has_more is False
    assert prompt_list.pagination.next_cursor is None
    assert detail.versions[0].prompt_text == "Hello {{ customer_name }}"


def test_prompt_management_client_creates_versions_and_labels():
    calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path, payload))
        return {
            "data": {
                "id": "prompt_chat",
                "name": "chat/support-agent",
                "prompt_type": "chat",
                "description": None,
                "latest_version": 2,
                "version_count": 2,
                "labels": {"latest": 2, "staging": 2},
                "tags": [],
                "versions": [
                    {
                        "id": "version_2",
                        "version": 2,
                        "prompt_type": "chat",
                        "prompt_text": None,
                        "chat_messages": [
                            {
                                "role": "system",
                                "content": "You are a premium support assistant.",
                            }
                        ],
                        "config": {"model": "gpt-4.1"},
                        "commit_message": "Premium support variant",
                        "variable_names": [],
                        "labels": ["latest", "staging"],
                    }
                ],
            }
        }

    client = PromptManagementClient(request_sender=request_sender)

    created = client.create_chat_prompt(
        "chat/support-agent",
        chat_messages=[
            ChatPromptMessage(role="system", content="You are a support assistant."),
            {"role": "user", "content": "Help {{ customer_name }}."},
        ],
        labels=["production"],
    )
    updated = client.create_prompt_version(
        "chat/support-agent",
        chat_messages=[
            {"role": "system", "content": "You are a premium support assistant."},
        ],
        config={"model": "gpt-4.1"},
        commit_message="Premium support variant",
        labels=["staging"],
    )
    relabeled = client.update_prompt_labels(
        "chat/support-agent",
        {"production": 1, "staging": 2},
    )

    assert calls[0][0] == "POST"
    assert calls[0][1] == ""
    assert calls[0][2]["prompt_type"] == "chat"
    assert calls[1] == (
        "POST",
        "/chat%2Fsupport-agent/versions",
        {
            "config": {"model": "gpt-4.1"},
            "commit_message": "Premium support variant",
            "labels": ["staging"],
            "chat_messages": [
                {
                    "role": "system",
                    "content": "You are a premium support assistant.",
                }
            ],
        },
    )
    assert calls[2] == (
        "PATCH",
        "/chat%2Fsupport-agent/labels",
        {"labels": {"production": 1, "staging": 2}},
    )
    assert created.prompt_type == PromptType.CHAT
    assert updated.latest_version == 2
    assert relabeled.labels["staging"] == 2


def test_prompt_management_client_resolves_prompt_versions():
    def request_sender(method: str, path: str, payload: dict | None):
        assert method == "GET"
        assert payload is None
        assert path == "/ops%2Frunbook/resolve?version=2"
        return {
            "data": {
                "name": "ops/runbook",
                "description": "Runbook prompt",
                "version": 2,
                "prompt_type": "text",
                "prompt_text": "Investigate {{ incident_id }} quickly.",
                "chat_messages": None,
                "config": {"model": "gpt-4.1-mini"},
                "commit_message": "Tighter wording",
                "variable_names": ["incident_id"],
                "labels": ["latest"],
                "resolved_label": None,
            }
        }

    client = PromptManagementClient(request_sender=request_sender)
    resolved = client.resolve_prompt("ops/runbook", version=2, label=None)

    assert resolved.version == 2
    assert resolved.prompt_text == "Investigate {{ incident_id }} quickly."
    assert resolved.variable_names == ["incident_id"]
    assert resolved.to_prompt_reference(variables={"incident_id": "INC-42"}) == {
        "name": "ops/runbook",
        "version": 2,
        "variables": {"incident_id": "INC-42"},
    }


def test_prompt_management_client_fetches_prompt_analytics():
    def request_sender(method: str, path: str, payload: dict | None):
        assert method == "GET"
        assert payload is None
        assert path == "/support%2Fwelcome/analytics?recent_limit=10&recent_page=2"
        return {
            "data": {
                "prompt_id": "prompt_1",
                "prompt_name": "support/welcome",
                "prompt_type": "text",
                "totals": {
                    "link_count": 2,
                    "trace_count": 1,
                    "observation_count": 1,
                    "total_input_tokens": 200,
                    "total_output_tokens": 40,
                    "total_tokens": 240,
                    "total_cost_usd": 0.04,
                    "total_latency_ms": 10000,
                    "last_used_at": "2026-03-10T12:00:05+00:00",
                },
                "versions": [
                    {
                        "version": 2,
                        "prompt_type": "text",
                        "labels": ["latest"],
                        "link_count": 1,
                        "trace_count": 1,
                        "observation_count": 0,
                        "total_input_tokens": 100,
                        "total_output_tokens": 20,
                        "total_tokens": 120,
                        "total_cost_usd": 0.02,
                        "total_latency_ms": 5000,
                        "last_used_at": "2026-03-10T12:00:05+00:00",
                    }
                ],
                "recent_links": [
                    {
                        "id": "link_1",
                        "trace_id": "trace_prompt_001",
                        "prompt_id": "prompt_1",
                        "prompt_version_id": "version_2",
                        "prompt_name": "support/welcome",
                        "prompt_type": "text",
                        "prompt_version": 2,
                        "prompt_label": "latest",
                        "variables": {"customer_name": "Ada"},
                        "trace_name": "prompt-linked-trace",
                        "trace_status": "completed",
                        "session_id": "session_prompt_001",
                        "environment": "production",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                        "cost_usd": 0.02,
                        "latency_ms": 5000,
                        "linked_at": "2026-03-10T12:00:05+00:00",
                    }
                ],
                "recent_links_pagination": {
                    "page": 2,
                    "per_page": 10,
                    "total": 11,
                    "total_pages": 2,
                    "has_next": False,
                    "has_prev": True,
                },
            }
        }

    client = PromptManagementClient(request_sender=request_sender)
    analytics = client.get_prompt_analytics(
        "support/welcome", recent_limit=10, recent_page=2
    )

    assert analytics.totals.link_count == 2
    assert analytics.versions[0].version == 2
    assert analytics.recent_links[0].prompt_name == "support/welcome"
    assert analytics.recent_links_pagination.page == 2


def test_prompt_management_client_runs_playground():
    calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path, payload))
        return {
            "data": {
                "prompt_name": "support/welcome",
                "prompt_type": "text",
                "resolved_version": 2,
                "resolved_label": "latest",
                "variables": {"customer_name": "Ada"},
                "config": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "temperature": 0.2,
                    "max_tokens": 256,
                },
                "rendered_prompt_text": "Hello Ada",
                "rendered_chat_messages": None,
                "executed": True,
                "output": "Hi Ada",
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                "cost_usd": 0.004,
                "latency_ms": 321,
                "trace_id": "trace_123",
                "trace_status": "completed",
            }
        }

    client = PromptManagementClient(request_sender=request_sender)
    result = client.run_playground(
        "support/welcome",
        version=2,
        variables={"customer_name": "Ada"},
        prompt_text="Hello {{ customer_name }}",
        model="gpt-4.1-mini",
        provider="openai",
        temperature=0.2,
        max_tokens=256,
    )

    assert calls == [
        (
            "POST",
            "/support%2Fwelcome/playground/run",
            {
                "variables": {"customer_name": "Ada"},
                "dry_run": False,
                "metadata": {},
                "version": 2,
                "prompt_text": "Hello {{ customer_name }}",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "temperature": 0.2,
                "max_tokens": 256,
            },
        )
    ]
    assert result.executed is True
    assert result.trace_id == "trace_123"
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 15


def test_prompt_management_client_encodes_route_like_prompt_names():
    calls: list[str] = []

    def request_sender(method: str, path: str, payload: dict | None):
        assert method == "GET"
        assert payload is None
        calls.append(path)
        return {
            "data": {
                "id": "prompt_analytics",
                "name": "ops/analytics",
                "prompt_type": "text",
                "description": "Route-like prompt name",
                "latest_version": 1,
                "version_count": 1,
                "labels": {"latest": 1},
                "tags": [],
                "versions": [
                    {
                        "id": "version_1",
                        "version": 1,
                        "prompt_type": "text",
                        "prompt_text": "Investigate {{ incident_id }}",
                        "chat_messages": None,
                        "config": {"model": "gpt-4.1-mini"},
                        "commit_message": "Initial",
                        "variable_names": ["incident_id"],
                        "labels": ["latest"],
                    }
                ],
            }
        }

    client = PromptManagementClient(request_sender=request_sender)

    detail = client.get_prompt("ops/analytics")

    assert calls == ["/ops%2Fanalytics"]
    assert detail.name == "ops/analytics"


def test_prompt_management_client_raises_authentication_error_on_401(monkeypatch):
    client = PromptManagementClient(
        config=PromptManagementConfig(
            backend_origin="https://example.test",
            api_key="test-key",  # pragma: allowlist secret
            project_id="proj_1",
        )
    )

    def raise_http_error(*args, **kwargs):
        raise error.HTTPError(
            url="https://example.test/api/v1/projects/proj_1/prompts/support%2Fwelcome",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("traigent.prompts.client.open_no_redirect", raise_http_error)

    with pytest.raises(AuthenticationError, match="status 401"):
        client.get_prompt("support/welcome")


def test_prompt_management_client_raises_connection_error_on_url_error(monkeypatch):
    client = PromptManagementClient(
        config=PromptManagementConfig(
            backend_origin="https://example.test",
            api_key="test-key",  # pragma: allowlist secret
            project_id="proj_1",
        )
    )

    def raise_url_error(*args, **kwargs):
        raise error.URLError("connection refused")

    monkeypatch.setattr("traigent.prompts.client.open_no_redirect", raise_url_error)

    with pytest.raises(
        TraigentConnectionError, match="Failed to connect to prompt backend"
    ):
        client.list_prompts()


def _summary(name: str) -> dict:
    return {
        "id": f"id_{name}",
        "name": name,
        "prompt_type": "text",
        "description": None,
        "latest_version": 1,
        "version_count": 1,
        "labels": {"latest": 1},
        "tags": [],
    }


def test_iter_prompts_follows_cursor_with_unchanged_filters():
    calls: list[str] = []
    pages = {
        "": (["a", "b"], "cursor_1", True),
        "cursor_1": (["c", "d"], "cursor_2", True),
        "cursor_2": (["e"], None, False),
    }

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append(path)
        cursor = ""
        if "cursor=" in path:
            cursor = path.split("cursor=")[1].split("&")[0]
        names, next_cursor, has_more = pages[cursor]
        return {
            "data": {
                "items": [_summary(n) for n in names],
                "pagination": {
                    "limit": 2,
                    "next_cursor": next_cursor,
                    "has_more": has_more,
                },
            }
        }

    client = PromptManagementClient(request_sender=request_sender)
    names = [p.name for p in client.iter_prompts(limit=2, label="production")]

    assert names == ["a", "b", "c", "d", "e"]
    assert calls == [
        "?limit=2&label=production",
        "?limit=2&cursor=cursor_1&label=production",
        "?limit=2&cursor=cursor_2&label=production",
    ]


def test_iter_prompts_stops_when_has_more_without_cursor():
    def request_sender(method: str, path: str, payload: dict | None):
        return {
            "data": {
                "items": [_summary("a")],
                "pagination": {"limit": 1, "next_cursor": None, "has_more": True},
            }
        }

    client = PromptManagementClient(request_sender=request_sender)

    assert [p.name for p in client.iter_prompts(limit=1)] == ["a"]


def test_delete_prompt_and_get_prompt_version_paths():
    calls: list[tuple[str, str]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path))
        if method == "DELETE":
            return {}
        return {
            "data": {
                "name": "ops/runbook",
                "description": None,
                "version": 3,
                "prompt_type": "text",
                "prompt_text": "Hi",
                "chat_messages": None,
                "config": {},
                "commit_message": None,
                "variable_names": [],
                "labels": [],
                "resolved_label": None,
            }
        }

    client = PromptManagementClient(request_sender=request_sender)

    assert client.delete_prompt("ops/runbook") is None
    assert client.get_prompt_version("ops/runbook", 3).version == 3
    assert calls == [
        ("DELETE", "/ops%2Frunbook"),
        ("GET", "/ops%2Frunbook/versions/3"),
    ]


def test_config_puts_project_id_in_path_not_headers():
    config = PromptManagementConfig(
        backend_origin="https://example.test",
        api_key="test-key",  # pragma: allowlist secret
        project_id="proj/1",
    )

    assert config.api_path == "/api/v1/projects/proj%2F1/prompts"
    assert not any("project" in key.lower() for key in config.build_headers())


def test_config_without_project_raises_when_path_is_needed():
    config = PromptManagementConfig(
        backend_origin="https://example.test",
        api_key="test-key",  # pragma: allowlist secret
        project_id=None,
    )

    with pytest.raises(ConfigurationError, match="project id"):
        _ = config.api_path


class _FakeResponse:
    def __init__(self, body: str = "", status: int = 200) -> None:
        self._body = body.encode()
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __bool__(self) -> bool:
        return True


def _http_error(
    status: int, body: dict | str, content_type: str, extra: dict | None = None
):
    headers = Message()
    headers["Content-Type"] = content_type
    for key, value in (extra or {}).items():
        headers[key] = value
    text = body if isinstance(body, str) else json.dumps(body)
    return error.HTTPError(
        url="https://example.test/x",
        code=status,
        msg="err",
        hdrs=headers,
        fp=io.BytesIO(text.encode()),
    )


def _client(monkeypatch, opener) -> PromptManagementClient:
    monkeypatch.setattr("traigent.prompts.client.open_no_redirect", opener)
    return PromptManagementClient(
        PromptManagementConfig(
            backend_origin="https://example.test",
            api_key="test-key",  # pragma: allowlist secret
            project_id="proj_1",
        )
    )


def test_request_url_is_project_scoped_and_sends_no_project_header(monkeypatch):
    seen = {}

    def opener(http_request, timeout):
        seen["url"] = http_request.full_url
        seen["headers"] = {k.lower() for k in http_request.headers}
        return _FakeResponse(
            json.dumps(
                {
                    "data": {
                        "items": [],
                        "pagination": {
                            "limit": 5,
                            "next_cursor": None,
                            "has_more": False,
                        },
                    }
                }
            )
        )

    response = _client(monkeypatch, opener).list_prompts(limit=5)

    assert seen["url"] == (
        "https://example.test/api/v1/projects/proj_1/prompts?limit=5"
    )
    assert "x-project-id" not in seen["headers"]
    assert response.items == []


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (404, "not_found", PromptNotFoundError),
        (400, "invalid_page_size", PromptInvalidRequestError),
        (400, "invalid_cursor", PromptInvalidRequestError),
        (422, "validation_failed", PromptInvalidRequestError),
        (409, "prompt_already_exists", PromptConflictError),
        (409, "prompt_version_conflict", PromptConflictError),
        (403, "feature_not_entitled", PromptNotPermittedError),
        (403, "provider_not_allowed", PromptNotPermittedError),
        (402, "insufficient_balance", PromptPaymentRequiredError),
        (402, "spend_blocked", PromptPaymentRequiredError),
        (502, "upstream_failed", PromptUpstreamError),
        (409, "some_future_conflict", PromptConflictError),
        (500, "some_future_failure", PromptApiError),
    ],
)
def test_problem_json_maps_to_typed_errors(monkeypatch, status, code, expected):
    problem = {
        "type": "about:blank",
        "title": "Problem title",
        "status": status,
        "code": code,
        "detail": "Fixed detail text.",
        "problem_id": "prob_123",
    }

    def opener(http_request, timeout):
        raise _http_error(status, problem, "application/problem+json; charset=utf-8")

    with pytest.raises(expected) as excinfo:
        _client(monkeypatch, opener).get_prompt("support/welcome")

    err = excinfo.value
    assert isinstance(err, ClientError)
    assert type(err) is expected
    assert err.code == code
    assert err.status_code == status
    assert err.problem_id == "prob_123"
    assert "Fixed detail text." in str(err)


def test_problem_json_403_wins_over_legacy_authentication_mapping(monkeypatch):
    def opener(http_request, timeout):
        raise _http_error(
            403,
            {"title": "t", "status": 403, "code": "feature_not_entitled"},
            "application/problem+json",
        )

    with pytest.raises(PromptNotPermittedError):
        _client(monkeypatch, opener).run_playground("p", dry_run=False)


def test_legacy_envelope_errors_from_shared_layers(monkeypatch):
    def make(status, extra=None):
        def opener(http_request, timeout):
            raise _http_error(
                status, {"success": False, "error": "x"}, "application/json", extra
            )

        return _client(monkeypatch, opener)

    with pytest.raises(AuthenticationError):
        make(401).get_prompt("p")
    with pytest.raises(AuthenticationError):
        make(403).get_prompt("p")
    with pytest.raises(RateLimitError) as rate:
        make(429, {"Retry-After": "7"}).get_prompt("p")
    assert rate.value.retry_after == 7.0
    with pytest.raises(ServiceUnavailableError):
        make(503).get_prompt("p")
    with pytest.raises(ClientError) as other:
        make(500).get_prompt("p")
    assert other.value.status_code == 500
    assert not isinstance(other.value, PromptApiError)


def test_unparseable_problem_body_falls_back_to_status_mapping(monkeypatch):
    def opener(http_request, timeout):
        raise _http_error(404, "not json", "application/problem+json")

    with pytest.raises(ClientError) as excinfo:
        _client(monkeypatch, opener).get_prompt("p")

    assert excinfo.value.status_code == 404
