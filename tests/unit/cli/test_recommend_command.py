"""Tests for CLI command: traigent recommend."""

from __future__ import annotations

import json

from click.testing import CliRunner

from traigent.api.functions import recommend_configuration_space
from traigent.cli.main import cli

_EXPECTED_JSON_TOP_LEVEL_KEYS = {
    "schema_version",
    "catalog_version",
    "agent_type",
    "valid_agent_types",
    "filters",
    "caveat",
    "configuration_space",
    "recommendations",
}
_EXPECTED_JSON_ROW_KEYS = {
    "name",
    "range_type",
    "range_kwargs",
    "range_code",
    "suggested_values",
    "category",
    "kind",
    "impact",
    "confidence",
    "evidence_note",
    "effectuation_status",
    "effectuation_strategy",
    "apply_guidance",
    "catalog_entry_id",
}


def test_recommend_command_smoke_table() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["recommend", "rag"])

    assert result.exit_code == 0
    assert "TVar Recommendations for rag" in result.output
    assert "retrieval_k" in result.output
    assert "Manual wiring" in result.output
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
    assert set(data) == _EXPECTED_JSON_TOP_LEVEL_KEYS
    assert data["agent_type"] == "code_gen"
    assert data["schema_version"] == "1"
    assert data["catalog_version"] == "1.0.0"
    assert set(data["configuration_space"]) == {"schema_context"}
    assert [row["name"] for row in data["recommendations"]] == ["schema_context"]
    assert set(data["recommendations"][0]) == _EXPECTED_JSON_ROW_KEYS


def test_recommend_configuration_space_json_round_trips() -> None:
    data = recommend_configuration_space("rag")

    assert json.loads(json.dumps(data)) == data


def test_recommend_command_unknown_agent_type_fails_clearly() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["recommend", "planner"])

    assert result.exit_code != 0
    assert "Unknown agent_type 'planner'" in result.output
    assert "Valid recommendation agent types" in result.output


def test_recommend_command_list_types() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["recommend", "--list-types", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {"valid_agent_types": ["code_gen", "rag"]}
