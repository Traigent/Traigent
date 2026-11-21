"""The REAL request URL of ``traigent plan`` and the MCP plan tool.

``OptimizationPlanClient`` appends its own ``/api/v1/optimization/plan``, so every
backend value (origin or API base, from env, stored credentials or ``--backend-url``)
must reach it as an origin. Asserts the URL httpx actually sends, not a constructor
argument.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from traigent.cli.plan_command import plan as plan_command
from traigent.mcp import tools

STORED_URL = (
    "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url"
)
PLAN_PATH = "/api/v1/optimization/plan"
_PAYLOAD: dict[str, Any] = {
    "schema_version": "1.0.0",
    "phase": "P1_STATIC",
    "plan": {},
    "steps": [],
    "evidence_level": "medium",
    "caveat": "advisory",
    "advisory": True,
}


@pytest.fixture
def sent_urls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    for name in (
        "TRAIGENT_BACKEND_URL",
        "TRAIGENT_API_URL",
        "TRAIGENT_API_KEY",
        "TRAIGENT_ENVIRONMENT",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setattr(STORED_URL, classmethod(lambda cls: None))
    monkeypatch.setattr(
        "traigent.config.plan_endpoint.BackendConfig.get_api_key",
        classmethod(lambda cls: "k"),
    )
    # validate_cloud_base_url resolves the host; answer with a public address.
    monkeypatch.setattr(
        "traigent.cloud.url_security.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    urls: list[str] = []

    async def fake_send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any):
        urls.append(str(request.url))
        return httpx.Response(200, json=_PAYLOAD, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return urls


_O = "https://api.example.test"
_WANT = f"{_O}{PLAN_PATH}"
# (env, stored, cli_flag, available_to_mcp)
_CASES = [
    pytest.param({"TRAIGENT_BACKEND_URL": _O}, None, None, id="backend-url-origin"),
    pytest.param({"TRAIGENT_BACKEND_URL": f"{_O}/"}, None, None, id="backend-slash"),
    pytest.param({"TRAIGENT_API_URL": _O}, None, None, id="api-url-origin"),
    pytest.param({"TRAIGENT_API_URL": f"{_O}/api/v1"}, None, None, id="api-url-v1"),
    pytest.param(
        {"TRAIGENT_API_URL": f"{_O}/api/v1/"}, None, None, id="api-url-v1-slash"
    ),
    pytest.param({"TRAIGENT_BACKEND_URL": f"{_O}/api/v1"}, None, None, id="backend-v1"),
    pytest.param({}, _O, None, id="stored-origin"),
    pytest.param({}, f"{_O}/api/v1", None, id="stored-v1"),
    pytest.param({}, None, _O, id="flag-origin"),
    pytest.param({}, None, f"{_O}/api/v1", id="flag-v1"),
    pytest.param({}, None, f"{_O}/api/v1/", id="flag-v1-slash"),
]
_PLAN_ARGS = [
    "--task-description", "t", "--dataset-size", "10", "--objective", "accuracy",
    "--max-trials", "5", "--cost-limit", "1", "--json",
]  # fmt: skip


@pytest.mark.parametrize(("env", "stored", "flag"), _CASES)
def test_cli_requests_the_real_plan_url(monkeypatch, sent_urls, env, stored, flag):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(STORED_URL, classmethod(lambda cls: stored))
    args = [*_PLAN_ARGS, *(["--backend-url", flag] if flag else [])]
    result = CliRunner().invoke(plan_command, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == _PAYLOAD
    assert sent_urls == [_WANT]


@pytest.mark.parametrize(("env", "stored", "flag"), _CASES)
def test_mcp_tool_requests_the_real_plan_url(monkeypatch, sent_urls, env, stored, flag):
    if flag:  # the tool has no flag: the same value arrives through the env
        env = {"TRAIGENT_BACKEND_URL": flag}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(STORED_URL, classmethod(lambda cls: stored))
    asyncio.run(
        tools.get_optimization_plan_tool(
            task_description="t",
            dataset_size=10,
            dataset_has_holdout=True,
            objectives=["accuracy"],
            max_trials=5,
            cost_limit_usd=1.0,
        )
    )
    assert sent_urls == [_WANT]


# --- the RAW explicit value is checked before it is reduced to an origin -----

_NO_ECHO = "SECRET-MARKER"


@pytest.mark.parametrize(
    "raw",
    [
        "https://trusted.example\n.evil.test/api/v1",
        "https://trusted.example\t.evil.test/api/v1",
        "https://trusted.example/api/v1\r",
        "https://trusted.example/api/v1 ",
        " https://trusted.example/api/v1",
        "https://h.example/api/v1?x=1",
        "https://h.example/api/v1#f",
        "https://h.example/a/../api/v1",
        "https://h.example/a/%2e%2e/api/v1",
        "https://h.example/a/%2E%2E/api/v1",
        "https://h.example/a/%252e%252e/api/v1",
        "https://user@h.example/api/v1",
        "https://user:pw@h.example/api/v1",  # pragma: allowlist secret
        "https://h.example\\@evil.test/api/v1",
        "https://h.example/api\\v1",
    ],
    ids=[
        "newline-in-host",
        "tab-in-host",
        "trailing-cr",
        "trailing-space",
        "leading-space",
        "query",
        "fragment",
        "dotdot",
        "encoded-dotdot",
        "encoded-dotdot-upper",
        "double-encoded-dotdot",
        "userinfo",
        "userinfo-password",
        "backslash-host",
        "backslash-path",
    ],
)
def test_resolve_refuses_raw_value_before_normalizing(raw: str) -> None:
    from traigent.config.plan_endpoint import resolve_backend_url

    with pytest.raises(ValueError) as excinfo:
        resolve_backend_url(raw)
    message = str(excinfo.value)
    assert "evil" not in message and "h.example" not in message
    assert "trusted" not in message


@pytest.mark.parametrize(
    ("raw", "origin"),
    [
        ("https://host.example/api/v1", "https://host.example"),
        ("https://host.example/api/v1/", "https://host.example"),
        ("https://host.example", "https://host.example"),
        ("http://127.0.0.1:8123", "http://127.0.0.1:8123"),
        ("http://localhost:5000", "http://localhost:5000"),
    ],
)
def test_resolve_keeps_legit_values_working(raw: str, origin: str) -> None:
    from traigent.config.plan_endpoint import resolve_backend_url

    assert resolve_backend_url(raw) == (origin, True)


def _invoke_plan(url: str):
    return CliRunner().invoke(
        plan_command,
        [
            "--task-description", "t",
            "--dataset-size", "10",
            "--objective", "accuracy",
            "--max-trials", "5",
            "--cost-limit", "1",
            "--backend-url", url,
            "--json",
        ],
    )  # fmt: skip


def test_cli_plan_gives_a_click_error_not_a_traceback() -> None:
    result = _invoke_plan(f"https://h.example/api/v1?{_NO_ECHO}=1")
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert "Error:" in result.output
    assert "query" in result.output
    assert _NO_ECHO not in result.output


def test_mcp_plan_tool_returns_closed_error_without_echoing_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", f"https://h.example/api/v1#{_NO_ECHO}")
    result = asyncio.run(
        tools.get_optimization_plan_tool(
            task_description="t",
            dataset_size=10,
            dataset_has_holdout=True,
            objectives=["accuracy"],
            max_trials=5,
            cost_limit_usd=1.0,
        )
    )
    assert result["ok"] is False
    assert result["code"] == "invalid_input"
    assert _NO_ECHO not in json.dumps(result)
