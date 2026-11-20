"""End to end: a real SDK client (ProjectManagementClient) behind the relay."""

from __future__ import annotations

import json

from tests.unit.egress.egress_relay_helpers import wait_lines


def test_project_client_through_relay_is_logged(relay, upstream, log_path, monkeypatch):
    # The SDK refuses plain http to loopback outside dev/test (validate_cloud_base_url),
    # so THIS TEST ONLY sets the environment to "test". The gate itself is unchanged.
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", "test")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")  # suite default is offline
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", relay.origin)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_canary_key_value")

    from traigent.projects.client import ProjectManagementClient
    from traigent.projects.config import ProjectManagementConfig

    upstream.response_body = json.dumps(
        {"data": {"items": [], "total": 0, "page": 1, "per_page": 20}}
    ).encode()
    upstream.extra_headers = [("Content-Type", "application/json")]

    client = ProjectManagementClient(ProjectManagementConfig())
    assert client.config.backend_origin == relay.origin
    client.list_projects(search="alice")

    assert upstream.seen[0].path.startswith("/api/v1beta/projects?")
    sent = {k.lower(): v for k, v in upstream.seen[0].headers.items()}
    assert sent["x-api-key"] == "tg_canary_key_value"
    lines = wait_lines(log_path, 2)  # the response line lands just after the reply
    text = log_path.read_text()
    assert [ln["kind"] for ln in lines] == ["request", "response"]
    assert lines[0]["route"] == "/api/v1beta/projects"
    assert lines[1]["status"] == 200
    assert "alice" not in text and "tg_canary_key_value" not in text
