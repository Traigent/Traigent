"""Tests for CLI command: traigent recommend."""

from __future__ import annotations

import json

from click.testing import CliRunner

from traigent.cli.main import cli


def test_recommend_command_smoke_table() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["recommend", "rag"])

    assert result.exit_code == 0
    assert "TVar Recommendations for rag" in result.output
    assert "retrieval_k" in result.output
    assert "manual_guidance" in result.output
    assert "task-dependent" in result.output


def test_recommend_command_json_filters() -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "recommend",
            "code_gen",
            "--min-impact",
            "high",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["agent_type"] == "code_gen"
    assert [row["name"] for row in data["recommendations"]] == ["schema_context"]


def test_recommend_command_list_types() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["recommend", "--list-types", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"valid_agent_types": ["code_gen", "rag"]}
