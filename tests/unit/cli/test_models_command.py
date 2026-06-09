"""Tests for CLI command: traigent models."""

from __future__ import annotations

import json
import sys
import types

from click.testing import CliRunner

from traigent.cli.main import cli


def test_models_command_lists_anthropic_models_as_json() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["models", "--provider", "anthropic", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["provider"] == "anthropic"
    assert payload["valid"] is None
    assert "claude-3-5-sonnet-20241022" in payload["models"]


def test_models_command_rejects_unknown_direct_provider_model() -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["models", "--provider", "anthropic", "--model", "dead-model", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["model"] == "dead-model"
    assert payload["valid"] is False


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
