"""CLI smoke tests for traigent plan."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.config.backend_config import DEFAULT_LOCAL_URL


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def plan_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "phase": "P1_STATIC",
        "plan": {
            "objectives": [
                {"name": "accuracy", "weight": 1.0, "orientation": "maximize"}
            ],
            "models": ["gpt-4o-mini"],
            "knobs": [{"name": "temperature", "values": ["0.0", "0.3"]}],
            "algorithm": "auto",
            "max_trials": 4,
            "cost_limit_usd": 5.0,
            "offline": False,
        },
        "steps": [
            {
                "id": "review_plan",
                "label": "Review plan",
                "command_template": "traigent optimize agent.py --max-trials 4",
            }
        ],
        "evidence_level": "medium",
        "caveat": "Plans are advisory until a run starts.",
        "advisory": True,
    }


class _FakeOptimizationPlanClient:
    def __init__(self, backend_url: str, api_key: str | None = None) -> None:
        self.backend_url = backend_url
        self.api_key = api_key

    async def __aenter__(self) -> _FakeOptimizationPlanClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def get_optimization_plan(self, **kwargs) -> dict[str, object]:
        assert kwargs == {
            "task_description": "Tune a support chatbot.",
            "dataset_size": 20,
            "dataset_has_holdout": True,
            "objectives": ("accuracy", "latency"),
            "max_trials": 4,
            "cost_limit_usd": 5.0,
            "task_type": "chatbot",
        }
        return _FakeOptimizationPlanClient.payload


def _plan_args(*extra: str) -> list[str]:
    return [
        "plan",
        "--task-description",
        "Tune a support chatbot.",
        "--dataset-size",
        "20",
        "--has-holdout",
        "--objective",
        "accuracy",
        "--objective",
        "latency",
        "--max-trials",
        "4",
        "--cost-limit",
        "5.0",
        "--task-type",
        "chatbot",
        *extra,
    ]


def test_plan_table_mode(
    runner: CliRunner,
    plan_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOptimizationPlanClient.payload = plan_payload
    monkeypatch.setattr(
        "traigent.cli.plan_command.OptimizationPlanClient",
        _FakeOptimizationPlanClient,
    )

    result = runner.invoke(cli, _plan_args())

    assert result.exit_code == 0
    assert "Optimization Plan" in result.output
    assert "P1_STATIC" in result.output
    assert "gpt-4o-mini" in result.output
    assert "temperature" in result.output
    assert "traigent optimize agent.py --max-trials 4" in result.output
    assert "Plans are advisory" in result.output


def test_plan_json_mode(
    runner: CliRunner,
    plan_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeOptimizationPlanClient.payload = plan_payload
    monkeypatch.setattr(
        "traigent.cli.plan_command.OptimizationPlanClient",
        _FakeOptimizationPlanClient,
    )

    result = runner.invoke(cli, _plan_args("--json"))

    assert result.exit_code == 0
    assert json.loads(result.output) == plan_payload


def test_plan_uses_explicit_backend_url_and_api_key_over_stored_credentials(
    runner: CliRunner,
    plan_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit --backend-url/--api-key must win over stored credentials."""
    _FakeOptimizationPlanClient.payload = plan_payload
    captured: dict[str, object] = {}

    class _CapturingClient(_FakeOptimizationPlanClient):
        def __init__(self, backend_url: str, api_key: str | None = None) -> None:
            captured["backend_url"] = backend_url
            captured["api_key"] = api_key
            super().__init__(backend_url, api_key)

    monkeypatch.setattr(
        "traigent.cli.plan_command.OptimizationPlanClient", _CapturingClient
    )
    monkeypatch.setattr(
        "traigent.cli.plan_command.BackendConfig.get_configured_backend_url",
        lambda: (_ for _ in ()).throw(
            AssertionError("should not fall back when --backend-url is explicit")
        ),
    )
    monkeypatch.setattr(
        "traigent.cli.plan_command.BackendConfig.get_api_key",
        lambda: (_ for _ in ()).throw(
            AssertionError("should not fall back when --api-key is explicit")
        ),
    )

    result = runner.invoke(
        cli,
        _plan_args(
            "--backend-url",
            "https://explicit.example.test",
            "--api-key",
            "explicit-key",  # pragma: allowlist secret
        ),
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_url"] == "https://explicit.example.test"
    assert captured["api_key"] == "explicit-key"  # pragma: allowlist secret


def test_plan_falls_back_to_stored_cli_credentials_when_unset(
    runner: CliRunner,
    plan_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traigent#1721: plan must not silently ignore stored CLI credentials."""
    monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
    monkeypatch.delenv("TRAIGENT_API_URL", raising=False)
    _FakeOptimizationPlanClient.payload = plan_payload
    captured: dict[str, object] = {}

    class _CapturingClient(_FakeOptimizationPlanClient):
        def __init__(self, backend_url: str, api_key: str | None = None) -> None:
            captured["backend_url"] = backend_url
            captured["api_key"] = api_key
            super().__init__(backend_url, api_key)

    monkeypatch.setattr(
        "traigent.cli.plan_command.OptimizationPlanClient", _CapturingClient
    )
    monkeypatch.setattr(
        "traigent.cli.plan_command.BackendConfig.get_configured_backend_url",
        lambda: "https://stored.example.test",
    )
    monkeypatch.setattr(
        "traigent.cli.plan_command.BackendConfig.get_api_key",
        lambda: "stored-key",  # pragma: allowlist secret
    )

    result = runner.invoke(cli, _plan_args())

    assert result.exit_code == 0, result.output
    assert captured["backend_url"] == "https://stored.example.test"
    assert captured["api_key"] == "stored-key"  # pragma: allowlist secret
    assert "Using backend URL from stored CLI credentials" in result.output


def test_plan_falls_back_to_localhost_when_nothing_configured(
    runner: CliRunner,
    plan_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traigent#1721: with no flag/env/stored cred, plan must fall back to the
    local default, NOT the prod cloud."""
    monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
    monkeypatch.delenv("TRAIGENT_API_URL", raising=False)
    _FakeOptimizationPlanClient.payload = plan_payload
    captured: dict[str, object] = {}

    class _CapturingClient(_FakeOptimizationPlanClient):
        def __init__(self, backend_url: str, api_key: str | None = None) -> None:
            captured["backend_url"] = backend_url
            captured["api_key"] = api_key
            super().__init__(backend_url, api_key)

    monkeypatch.setattr(
        "traigent.cli.plan_command.OptimizationPlanClient", _CapturingClient
    )
    # Genuinely nothing configured: no env (deleted above), no stored cred.
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(
        "traigent.cli.plan_command.BackendConfig.get_api_key",
        lambda: None,
    )

    result = runner.invoke(cli, _plan_args())

    assert result.exit_code == 0, result.output
    assert captured["backend_url"] == DEFAULT_LOCAL_URL
    assert "portal.traigent.ai" not in str(captured["backend_url"])
