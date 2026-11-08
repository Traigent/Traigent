"""traigent plan rejects non-positive / non-finite numeric options locally."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli

_BASE = [
    "plan",
    "--task-description",
    "x",
    "--objective",
    "accuracy",
    "--json",
    "--backend-url",
    "http://127.0.0.1:1",
]


def _args(**over: str) -> list[str]:
    vals = {"--dataset-size": "10", "--max-trials": "5", "--cost-limit": "1.0"}
    vals.update(over)
    out = list(_BASE)
    for k, v in vals.items():
        out += [k, v]
    return out


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--dataset-size", "0"),
        ("--dataset-size", "-3"),
        ("--max-trials", "0"),
        ("--max-trials", "-5"),
        ("--cost-limit", "-1"),
        ("--cost-limit", "0"),
        ("--cost-limit", "nan"),
        ("--cost-limit", "inf"),
    ],
)
def test_plan_rejects_bad_numeric_option_naming_flag(flag: str, value: str) -> None:
    result = CliRunner().invoke(cli, _args(**{flag: value}))
    assert result.exit_code == 2, result.output
    assert flag in result.output
