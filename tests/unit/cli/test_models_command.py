"""Tests for CLI command: traigent models."""

from __future__ import annotations

import json
import sys
import types

from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.integrations.model_discovery.anthropic_discovery import AnthropicDiscovery
from traigent.integrations.model_discovery.cache import ModelCache


class FakeDiscovery:
    def __init__(self, valid: bool = True) -> None:
        self.list_calls: list[bool] = []
        self.valid_calls: list[str] = []
        self.valid = valid

    def list_models(self, force_refresh: bool = False) -> list[str]:
        self.list_calls.append(force_refresh)
        return ["claude-known-model"]

    def is_valid_model(self, model_id: str) -> bool:
        self.valid_calls.append(model_id)
        return self.valid


def test_models_command_lists_anthropic_models_via_discovery_api(
    monkeypatch,
) -> None:
    discovery = FakeDiscovery()

    def fake_get_model_discovery(provider: str) -> FakeDiscovery:
        assert provider == "anthropic"
        return discovery

    monkeypatch.setattr(
        "traigent.integrations.model_discovery.get_model_discovery",
        fake_get_model_discovery,
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["models", "--provider", "anthropic", "--json"])

    assert result.exit_code == 0
    assert discovery.list_calls == [False]
    assert discovery.valid_calls == []
    payload = json.loads(result.output)
    assert payload["provider"] == "anthropic"
    assert payload["valid"] is None
    assert payload["models"] == ["claude-known-model"]


def test_models_command_check_validates_via_discovery_api(
    monkeypatch,
) -> None:
    discovery = FakeDiscovery(valid=True)

    def fake_get_model_discovery(provider: str) -> FakeDiscovery:
        assert provider == "anthropic"
        return discovery

    monkeypatch.setattr(
        "traigent.integrations.model_discovery.get_model_discovery",
        fake_get_model_discovery,
    )
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "models",
            "--provider",
            "anthropic",
            "--check",
            "claude-known-model",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert discovery.list_calls == [False]
    assert discovery.valid_calls == ["claude-known-model"]
    payload = json.loads(result.output)
    assert payload["model"] == "claude-known-model"
    assert payload["valid"] is True


def test_models_command_rejects_unknown_direct_provider_model(
    monkeypatch,
) -> None:
    discovery = FakeDiscovery(valid=False)

    def fake_get_model_discovery(provider: str) -> FakeDiscovery:
        assert provider == "anthropic"
        return discovery

    monkeypatch.setattr(
        "traigent.integrations.model_discovery.get_model_discovery",
        fake_get_model_discovery,
    )
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["models", "--provider", "anthropic", "--model", "dead-model", "--json"],
    )

    assert result.exit_code == 1
    assert discovery.list_calls == [False]
    assert discovery.valid_calls == ["dead-model"]
    payload = json.loads(result.output)
    assert payload["model"] == "dead-model"
    assert payload["valid"] is False


def test_models_command_check_accepts_current_anthropic_family_first_id(
    monkeypatch,
) -> None:
    discovery = AnthropicDiscovery(cache=ModelCache(enable_file_cache=False))

    def fake_get_model_discovery(provider: str) -> AnthropicDiscovery:
        assert provider == "anthropic"
        return discovery

    monkeypatch.setattr(
        "traigent.integrations.model_discovery.get_model_discovery",
        fake_get_model_discovery,
    )
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "models",
            "--provider",
            "anthropic",
            "--check",
            "claude-sonnet-4-6",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["valid"] is True


def test_models_command_validates_bedrock_model_without_invoke(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    class FakeBedrockClient:
        def list_foundation_models(self) -> dict[str, list[dict[str, str]]]:
            return {
                "modelSummaries": [
                    {"modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"},
                    {"modelId": "amazon.nova-lite-v1:0"},
                ]
            }

    def fake_client(service_name: str, region_name: str | None = None) -> object:
        calls.append((service_name, region_name))
        return FakeBedrockClient()

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=fake_client))
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "models",
            "--provider",
            "bedrock",
            "--region",
            "us-east-1",
            "--model",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert calls == [("bedrock", "us-east-1")]
    payload = json.loads(result.output)
    assert payload["provider"] == "bedrock"
    assert payload["region"] == "us-east-1"
    assert payload["valid"] is True
    assert payload["models"] == [
        "amazon.nova-lite-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
    ]
