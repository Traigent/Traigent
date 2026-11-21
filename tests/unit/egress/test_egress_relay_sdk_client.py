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


def _production_posture(monkeypatch, relay, *, switch):
    for name in ("TRAIGENT_ENVIRONMENT", "ENVIRONMENT", "TRAIGENT_ENV"):
        monkeypatch.delenv(name, raising=False)  # no dev/test marker at all
    if switch:
        monkeypatch.setenv("TRAIGENT_EGRESS_RELAY", "1")
    else:
        monkeypatch.delenv("TRAIGENT_EGRESS_RELAY", raising=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", relay.origin)
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_canary_key_value")


def test_project_client_through_relay_in_production_posture(
    relay, upstream, log_path, monkeypatch
):
    _production_posture(monkeypatch, relay, switch=True)

    from traigent.projects.client import ProjectManagementClient
    from traigent.projects.config import ProjectManagementConfig

    upstream.response_body = json.dumps(
        {"data": {"items": [], "total": 0, "page": 1, "per_page": 20}}
    ).encode()
    upstream.extra_headers = [("Content-Type", "application/json")]

    client = ProjectManagementClient(ProjectManagementConfig())
    client.list_projects(search="alice")

    assert upstream.seen[0].path.startswith("/api/v1beta/projects?")
    lines = wait_lines(log_path, 2)
    assert [ln["kind"] for ln in lines] == ["request", "response"]
    assert lines[1]["status"] == 200
    assert "alice" not in log_path.read_text()


def test_project_client_refuses_before_any_byte_with_the_switch_off(
    relay, upstream, log_path, monkeypatch
):
    import pytest

    _production_posture(monkeypatch, relay, switch=False)

    from traigent.projects.client import ProjectManagementClient
    from traigent.projects.config import ProjectManagementConfig

    with pytest.raises(Exception) as excinfo:
        ProjectManagementClient(ProjectManagementConfig()).list_projects(search="alice")
    assert "https" in str(excinfo.value)
    assert upstream.seen == []
    assert not log_path.exists() or log_path.read_text() == ""
