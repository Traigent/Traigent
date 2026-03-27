from __future__ import annotations

from urllib import error

import pytest

from traigent.prompts import ChatPromptMessage, PromptManagementClient, PromptType
from traigent.prompts.config import PromptManagementConfig
from traigent.utils.exceptions import AuthenticationError, TraigentConnectionError


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
                        "page": 1,
                        "per_page": 20,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
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

    prompt_list = client.list_prompts(search="support", prompt_type=PromptType.TEXT, label="production")
    detail = client.get_prompt("support/welcome")

    assert calls[0] == ("GET", "?page=1&per_page=20&search=support&prompt_type=text&label=production", None)
    assert calls[1] == ("GET", "/support%2Fwelcome", None)
    assert prompt_list.items[0].name == "support/welcome"
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
    analytics = client.get_prompt_analytics("support/welcome", recent_limit=10, recent_page=2)

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
        )
    )

    def raise_http_error(*args, **kwargs):
        raise error.HTTPError(
            url="https://example.test/api/v1beta/prompts",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("traigent.prompts.client.request.urlopen", raise_http_error)

    with pytest.raises(AuthenticationError, match="status 401"):
        client.get_prompt("support/welcome")


def test_prompt_management_client_raises_connection_error_on_url_error(monkeypatch):
    client = PromptManagementClient(
        config=PromptManagementConfig(
            backend_origin="https://example.test",
            api_key="test-key",  # pragma: allowlist secret
        )
    )

    def raise_url_error(*args, **kwargs):
        raise error.URLError("connection refused")

    monkeypatch.setattr("traigent.prompts.client.request.urlopen", raise_url_error)

    with pytest.raises(TraigentConnectionError, match="Failed to connect to prompt backend"):
        client.list_prompts()
