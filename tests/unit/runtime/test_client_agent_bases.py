"""RuntimeClient uses two bases: agent satellites on /api/v1/projects, requirements on v1beta."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.runtime.client import AGENT_API_PREFIX, API_PREFIX, RuntimeClient

ORIGIN = "https://api.example.test"


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake_send(
        self: RuntimeClient, method: str, url: str, body: Any = None, **_kw: Any
    ) -> tuple[int, Any, dict[str, str]]:
        calls.append((method, url))
        if url.endswith("/production-breaches"):
            return 200, {"items": [], "has_more": False}, {}
        if url.endswith("/next-step"):
            return 200, {"step": {"code": "X"}}, {}
        return 200, {"items": []}, {}

    monkeypatch.setattr(RuntimeClient, "send", fake_send)
    return calls


def test_prefixes_are_distinct() -> None:
    assert API_PREFIX == "/api/v1beta/projects"
    assert AGENT_API_PREFIX == "/api/v1/projects"


def test_breaches_and_next_step_use_project_level_agent_paths(seen: list) -> None:
    client = RuntimeClient("proj-1", base_url=ORIGIN, api_key="k")
    client.list_breaches("agent-1")
    client.get_next_step("agent-1")
    assert seen == [
        ("GET", f"{ORIGIN}/api/v1/projects/proj-1/agents/agent-1/production-breaches"),
        ("GET", f"{ORIGIN}/api/v1/projects/proj-1/agents/agent-1/next-step"),
    ]


def test_requirements_stay_on_v1beta(seen: list) -> None:
    client = RuntimeClient("proj-1", base_url=ORIGIN, api_key="k")
    client.list_requirements()
    assert seen == [
        ("GET", f"{ORIGIN}/api/v1beta/projects/proj-1/production-requirements")
    ]


def test_agent_ref_and_project_are_url_encoded(seen: list) -> None:
    client = RuntimeClient("proj 1", base_url=ORIGIN, api_key="k")
    client.get_next_step("agent?x")
    assert seen == [
        ("GET", f"{ORIGIN}/api/v1/projects/proj%201/agents/agent%3Fx/next-step"),
    ]
