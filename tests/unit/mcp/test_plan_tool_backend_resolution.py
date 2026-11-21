"""get_optimization_plan_tool must resolve its backend like ``traigent plan``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from traigent.config.backend_config import DEFAULT_LOCAL_URL
from traigent.config.plan_endpoint import (
    backend_url_from_env,
    resolve_api_key,
    resolve_backend_url,
)
from traigent.mcp import tools

STORED_URL = (
    "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url"
)
URL_ENVS = ("TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL")


class _FakeClient:
    instances: list[_FakeClient] = []

    def __init__(self, backend_url: str = DEFAULT_LOCAL_URL, api_key: Any = None):
        self.backend_url = backend_url
        self.api_key = api_key
        _FakeClient.instances.append(self)

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get_optimization_plan(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*URL_ENVS, "TRAIGENT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(STORED_URL, classmethod(lambda cls: None))
    monkeypatch.setattr(
        "traigent.config.plan_endpoint.BackendConfig.get_api_key",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(tools, "OptimizationPlanClient", _FakeClient)
    _FakeClient.instances = []


async def _call() -> _FakeClient:
    await tools.get_optimization_plan_tool(
        task_description="t",
        dataset_size=10,
        dataset_has_holdout=True,
        objectives=["accuracy"],
        max_trials=5,
        cost_limit_usd=1.0,
    )
    assert len(_FakeClient.instances) == 1
    return _FakeClient.instances[0]


@pytest.mark.asyncio
async def test_backend_url_env_is_used_not_localhost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "https://backend.example.test")
    client = await _call()
    assert client.backend_url == "https://backend.example.test"
    assert client.backend_url != DEFAULT_LOCAL_URL


@pytest.mark.asyncio
async def test_api_url_env_used_when_backend_url_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_API_URL", "https://api.example.test")
    assert (await _call()).backend_url == "https://api.example.test"


@pytest.mark.asyncio
async def test_backend_url_env_wins_over_api_url_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "https://backend.example.test")
    monkeypatch.setenv("TRAIGENT_API_URL", "https://api.example.test")
    assert (await _call()).backend_url == "https://backend.example.test"


@pytest.mark.asyncio
async def test_stored_credentials_backend_url_used_when_no_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        STORED_URL, classmethod(lambda cls: "https://stored.example.test")
    )
    assert (await _call()).backend_url == "https://stored.example.test"


@pytest.mark.asyncio
async def test_api_key_resolution_matches_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "traigent.config.plan_endpoint.BackendConfig.get_api_key",
        classmethod(lambda cls: "stored-key"),  # pragma: allowlist secret
    )
    client = await _call()
    assert (
        client.api_key == resolve_api_key(None) == "stored-key"
    )  # pragma: allowlist secret


@pytest.mark.asyncio
async def test_nothing_configured_equals_cli_fallback() -> None:
    client = await _call()
    assert (client.backend_url, False) == resolve_backend_url(None)
    assert client.backend_url == DEFAULT_LOCAL_URL
    assert client.api_key is None


@pytest.mark.asyncio
async def test_api_url_with_path_is_passed_through_like_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI hands the raw env value to the client; the tool does the same."""
    monkeypatch.setenv("TRAIGENT_API_URL", "https://api.example.test/api/v1")
    assert (await _call()).backend_url == "https://api.example.test/api/v1"


@pytest.mark.parametrize(
    ("backend_env", "api_env", "stored"),
    [
        (None, None, None),
        ("https://b.example.test", None, None),
        (None, "https://a.example.test", None),
        ("https://b.example.test", "https://a.example.test", None),
        (None, None, "https://s.example.test"),
        ("https://b.example.test", None, "https://s.example.test"),
        ("", "https://a.example.test", "https://s.example.test"),
    ],
)
def test_cli_and_mcp_tool_resolve_identically(
    monkeypatch: pytest.MonkeyPatch,
    backend_env: str | None,
    api_env: str | None,
    stored: str | None,
) -> None:
    from click.testing import CliRunner

    from traigent.cli.plan_command import plan as plan_command

    for name, value in zip(URL_ENVS, (backend_env, api_env), strict=True):
        if value is not None:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(STORED_URL, classmethod(lambda cls: stored))

    cli_clients: list[_FakeClient] = []

    class _CliClient(_FakeClient):
        def __init__(self, backend_url: str, api_key: Any = None) -> None:
            super().__init__(backend_url, api_key)
            cli_clients.append(self)

        async def get_optimization_plan(self, **kwargs: Any) -> dict[str, Any]:
            return {}

    monkeypatch.setattr("traigent.cli.plan_command.OptimizationPlanClient", _CliClient)
    result = CliRunner().invoke(
        plan_command,
        [
            "--task-description",
            "t",
            "--dataset-size",
            "10",
            "--objective",
            "accuracy",
            "--max-trials",
            "5",
            "--cost-limit",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    _FakeClient.instances = []
    mcp_client = asyncio.run(_call())
    assert (mcp_client.backend_url, mcp_client.api_key) == (
        cli_clients[0].backend_url,
        cli_clients[0].api_key,
    )
    assert backend_url_from_env() == (backend_env or api_env or None)
