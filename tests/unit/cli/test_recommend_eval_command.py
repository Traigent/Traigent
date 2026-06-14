"""Tests for CLI command: traigent recommend-eval."""

from __future__ import annotations

import json

from click.testing import CliRunner

from traigent.cli.main import cli


def test_recommend_eval_command_list_types() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["recommend-eval", "--list-types", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "valid_task_types": ["code_gen", "general", "rag"]
    }


def test_recommend_eval_command_smoke_table() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["recommend-eval", "rag"])

    assert result.exit_code == 0
    assert "Metric Recommendations for rag" in result.output
    assert "faithfulness" in result.output
    assert "task-dependent" in result.output


def test_recommend_eval_command_json_parses() -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "recommend-eval",
            "code_gen",
            "--measure-type",
            "accuracy",
            "--json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["task_type"] == "code_gen"
    assert data["filters"]["measure_types"] == ["accuracy"]
    assert {row["metric"]["name"] for row in data["recommendations"]} == {
        "execution_accuracy",
        "pass_at_k",
    }


def test_recommend_eval_command_evaluator_json_parses() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["recommend-eval", "general", "--evaluator", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["task_type"] == "general"
    assert data["recommendations"]
    assert "evaluator_binding" in data["recommendations"][0]
