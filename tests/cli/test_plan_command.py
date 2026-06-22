"""CLI smoke tests for traigent plan."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli


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
    def __init__(self, backend_url: str) -> None:
        self.backend_url = backend_url

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
