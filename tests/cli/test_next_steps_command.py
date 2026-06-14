"""CLI smoke tests for traigent next-steps."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def next_steps_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "experiment_run_id": "run_123",
        "caveat": "Recommendations are category-level and should be reviewed.",
        "summary": {"confidence_label": "medium"},
        "next_steps": [
            {
                "id": "step_1",
                "category": "compare_with_baseline",
                "priority": 1,
                "rationale": "Compare the candidate winner to a baseline.",
                "action": {
                    "kind": "cli",
                    "command_template": "traigent results compare run_123 baseline",
                },
                "evidence_level": "medium",
            }
        ],
    }


class _FakeNextStepsClient:
    def __init__(self, backend_url: str) -> None:
        self.backend_url = backend_url

    async def __aenter__(self) -> _FakeNextStepsClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def get_next_steps(self, experiment_run_id: str) -> dict[str, object]:
        assert experiment_run_id == "run_123"
        return _FakeNextStepsClient.payload


def test_next_steps_table_mode(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeNextStepsClient.payload = next_steps_payload
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient",
        _FakeNextStepsClient,
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0
    assert "Next Steps for run_123" in result.output
    assert "compare_with_baseline" in result.output
    assert "traigent results compare run_123 baseline" in result.output
    assert "Caveat:" in result.output
    assert "Recommendations are category-level" in result.output


def test_next_steps_json_mode(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeNextStepsClient.payload = next_steps_payload
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient",
        _FakeNextStepsClient,
    )

    result = runner.invoke(cli, ["next-steps", "run_123", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == next_steps_payload
